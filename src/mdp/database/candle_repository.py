from typing import Any

import pandas as pd

from mdp.database.connection import get_connection


INSERT_MARKET_CANDLE_SQL = """
INSERT INTO market_candles (
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
    source
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
    open_price = VALUES(open_price),
    high_price = VALUES(high_price),
    low_price = VALUES(low_price),
    close_price = VALUES(close_price),
    tick_volume = VALUES(tick_volume),
    spread = VALUES(spread),
    real_volume = VALUES(real_volume),
    source = VALUES(source);
"""


FETCH_MARKET_CANDLES_SQL = """
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
    ORDER BY timestamp_utc DESC
    LIMIT %s
) AS recent_candles
ORDER BY timestamp_utc ASC;
"""

FETCH_MARKET_CANDLES_BETWEEN_SQL = """
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
  AND timestamp_utc >= %s
  AND timestamp_utc < %s
ORDER BY timestamp_utc ASC;
"""

def insert_market_candles(candles: list[dict[str, Any]]) -> int:
    """
    Inserts or updates market candles in the database.

    If a candle with the same symbol, timeframe, and timestamp already exists,
    the existing row is updated instead of duplicated.
    """
    if not candles:
        return 0

    values = [
        (
            candle["symbol"],
            candle["timeframe"],
            candle["timestamp_utc"],
            candle["open_price"],
            candle["high_price"],
            candle["low_price"],
            candle["close_price"],
            candle.get("tick_volume"),
            candle.get("spread"),
            candle.get("real_volume"),
            candle.get("source", "MANUAL_TEST"),
        )
        for candle in candles
    ]

    connection = get_connection()
    cursor = connection.cursor()

    cursor.executemany(INSERT_MARKET_CANDLE_SQL, values)
    connection.commit()

    affected_rows = cursor.rowcount

    cursor.close()
    connection.close()

    return affected_rows


def fetch_market_candles(symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
    """
    Fetches market candles from the database and returns them as a Pandas DataFrame.
    """
    connection = get_connection()
    cursor = connection.cursor(dictionary=True)

    cursor.execute(FETCH_MARKET_CANDLES_SQL, (symbol, timeframe, limit))
    rows = cursor.fetchall()

    cursor.close()
    connection.close()

    return pd.DataFrame(rows)

def insert_market_candles_from_dataframe(candles_df: pd.DataFrame) -> int:
    """
    Converts a market candles DataFrame into database-ready records
    and inserts or updates them in the market_candles table.
    """
    if candles_df.empty:
        return 0

    required_columns = [
        "symbol",
        "timeframe",
        "timestamp_utc",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "tick_volume",
        "spread",
        "real_volume",
        "source",
    ]

    missing_columns = [
        column for column in required_columns if column not in candles_df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    normalized_df = candles_df[required_columns].copy()

    normalized_df["timestamp_utc"] = pd.to_datetime(
        normalized_df["timestamp_utc"]
    ).dt.to_pydatetime()

    normalized_df["tick_volume"] = normalized_df["tick_volume"].astype(int)
    normalized_df["spread"] = normalized_df["spread"].astype(int)
    normalized_df["real_volume"] = normalized_df["real_volume"].astype(int)

    candles = normalized_df.to_dict(orient="records")

    return insert_market_candles(candles)


def fetch_market_candles_between(
    symbol: str,
    timeframe: str,
    start_time,
    end_time,
) -> pd.DataFrame:
    """
    Fetches market candles between a start time and end time.

    The start time is inclusive.
    The end time is exclusive.
    """
    start_timestamp = pd.to_datetime(start_time).to_pydatetime()
    end_timestamp = pd.to_datetime(end_time).to_pydatetime()

    connection = get_connection()
    cursor = connection.cursor(dictionary=True)

    cursor.execute(
        FETCH_MARKET_CANDLES_BETWEEN_SQL,
        (
            symbol,
            timeframe.upper(),
            start_timestamp,
            end_timestamp,
        ),
    )

    rows = cursor.fetchall()

    cursor.close()
    connection.close()

    return pd.DataFrame(rows)