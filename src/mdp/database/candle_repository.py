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
FROM market_candles
WHERE symbol = %s
  AND timeframe = %s
ORDER BY timestamp_utc ASC
LIMIT %s;
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