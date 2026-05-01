from datetime import datetime

from mdp.database.candle_repository import (
    fetch_market_candles,
    insert_market_candles,
)


def main():
    sample_candles = [
        {
            "symbol": "TEST_SYMBOL",
            "timeframe": "M5",
            "timestamp_utc": datetime(2026, 1, 1, 10, 0, 0),
            "open_price": 100.00,
            "high_price": 101.50,
            "low_price": 99.80,
            "close_price": 101.00,
            "tick_volume": 1500,
            "spread": 2,
            "real_volume": 0,
            "source": "MANUAL_TEST",
        },
        {
            "symbol": "TEST_SYMBOL",
            "timeframe": "M5",
            "timestamp_utc": datetime(2026, 1, 1, 10, 5, 0),
            "open_price": 101.00,
            "high_price": 102.20,
            "low_price": 100.70,
            "close_price": 101.80,
            "tick_volume": 1700,
            "spread": 2,
            "real_volume": 0,
            "source": "MANUAL_TEST",
        },
        {
            "symbol": "TEST_SYMBOL",
            "timeframe": "M5",
            "timestamp_utc": datetime(2026, 1, 1, 10, 10, 0),
            "open_price": 101.80,
            "high_price": 103.00,
            "low_price": 101.40,
            "close_price": 102.60,
            "tick_volume": 1800,
            "spread": 2,
            "real_volume": 0,
            "source": "MANUAL_TEST",
        },
    ]

    affected_rows = insert_market_candles(sample_candles)
    print(f"Rows inserted or updated: {affected_rows}")

    candles_df = fetch_market_candles("TEST_SYMBOL", "M5", limit=10)

    print("\nFetched candles from MySQL:")
    print(candles_df)


if __name__ == "__main__":
    main()