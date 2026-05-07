from __future__ import annotations

from typing import Any

import pandas as pd

from mdp.database.connection import get_connection


FETCH_MARKET_CANDLES_AS_OF_SQL = """
SELECT
    id,
    symbol,
    timeframe,
    timestamp_utc,
    open_price,
    high_price,
    low_price,
    close_price,
    tick_volume,
    spread,
    real_volume,
    source,
    created_at
FROM (
    SELECT
        id,
        symbol,
        timeframe,
        timestamp_utc,
        open_price,
        high_price,
        low_price,
        close_price,
        tick_volume,
        spread,
        real_volume,
        source,
        created_at
    FROM market_candles
    WHERE symbol = %s
      AND timeframe = %s
      AND timestamp_utc <= %s
    ORDER BY timestamp_utc DESC
    LIMIT %s
) AS cutoff_candles
ORDER BY timestamp_utc ASC;
"""


TIMEFRAME_SECONDS = {
    "M1": 60,
    "M2": 120,
    "M3": 180,
    "M5": 300,
    "M10": 600,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H2": 7200,
    "H4": 14400,
    "D1": 86400,
}


def get_timeframe_seconds(timeframe: str) -> int | None:
    """
    Returns the candle duration in seconds for common market timeframes.
    """
    # Normalize the label so command-line inputs like m1 and H1 work the same way.
    normalized_timeframe = str(timeframe).strip().upper()
    return TIMEFRAME_SECONDS.get(normalized_timeframe)


def resolve_closed_candle_cutoff(
    timeframe: str,
    end_time: Any,
    closed_candles_only: bool = True,
) -> pd.Timestamp:
    """
    Converts an analysis cutoff into the latest candle-open timestamp allowed.
    """
    cutoff_timestamp = pd.to_datetime(end_time)

    if not closed_candles_only:
        return cutoff_timestamp

    timeframe_seconds = get_timeframe_seconds(timeframe)
    if timeframe_seconds is None:
        raise ValueError(f"Unsupported timeframe for closed-candle cutoff: {timeframe}")

    # MT5 candle timestamps represent candle open time, so subtract one full candle.
    # Example: at 17:00, the H1 candle opened at 17:00 is not closed yet.
    return cutoff_timestamp - pd.Timedelta(seconds=timeframe_seconds)


def load_market_candles_as_of(
    symbol: str,
    timeframe: str,
    end_time: Any,
    limit: int = 300,
    closed_candles_only: bool = True,
) -> pd.DataFrame:
    """
    Loads candles that were fully available at or before a historical cutoff time.
    """
    normalized_timeframe = timeframe.upper()

    # Use only closed candles by default to avoid higher-timeframe look-ahead bias.
    effective_cutoff = resolve_closed_candle_cutoff(
        timeframe=normalized_timeframe,
        end_time=end_time,
        closed_candles_only=closed_candles_only,
    )

    # Convert the cutoff into a Python datetime so MySQL receives a safe value.
    cutoff_datetime = effective_cutoff.to_pydatetime()

    connection = get_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute(
        FETCH_MARKET_CANDLES_AS_OF_SQL,
        (symbol, normalized_timeframe, cutoff_datetime, int(limit)),
    )
    rows = cursor.fetchall()
    cursor.close()
    connection.close()

    # Return a DataFrame in the same shape as the normal candle loader.
    return pd.DataFrame(rows)
