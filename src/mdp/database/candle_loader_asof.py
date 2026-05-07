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


def load_market_candles_as_of(
    symbol: str,
    timeframe: str,
    end_time: Any,
    limit: int = 300,
) -> pd.DataFrame:
    """
    Loads the latest candles available at or before a historical cutoff time.
    """
    # Convert the cutoff into a Python datetime so MySQL receives a safe value.
    cutoff_timestamp = pd.to_datetime(end_time).to_pydatetime()

    # Query only candles that existed at the cutoff to avoid look-ahead bias.
    connection = get_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute(
        FETCH_MARKET_CANDLES_AS_OF_SQL,
        (symbol, timeframe.upper(), cutoff_timestamp, int(limit)),
    )
    rows = cursor.fetchall()
    cursor.close()
    connection.close()

    # Return a DataFrame in the same shape as the normal candle loader.
    return pd.DataFrame(rows)
