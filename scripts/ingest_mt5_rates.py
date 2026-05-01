import argparse

from mdp.database.candle_repository import insert_market_candles_from_dataframe
from mdp.ingestion.mt5_connector import (
    fetch_rates,
    initialize_mt5,
    shutdown_mt5,
)


def parse_arguments():
    """
    Parses command-line arguments for MT5 market data ingestion.
    """
    parser = argparse.ArgumentParser(
        description="Fetch market candles from MT5 and store them in MySQL."
    )

    parser.add_argument(
        "--symbol",
        required=True,
        help='MT5 symbol name, for example: "Step Index" or "Volatility 75 Index".',
    )

    parser.add_argument(
        "--timeframe",
        required=True,
        help="Timeframe, for example: M1, M5, M15, H1, H4, D1.",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of candles to fetch from MT5.",
    )

    parser.add_argument(
        "--start-pos",
        type=int,
        default=1,
        help="Starting candle position. Use 1 to skip the currently forming candle.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    initialize_mt5()

    try:
        rates_df = fetch_rates(
            symbol=args.symbol,
            timeframe=args.timeframe,
            count=args.count,
            start_pos=args.start_pos,
        )

        affected_rows = insert_market_candles_from_dataframe(rates_df)

        print("MT5 ingestion completed successfully.")
        print(f"Symbol: {args.symbol}")
        print(f"Timeframe: {args.timeframe.upper()}")
        print(f"Candles fetched: {len(rates_df)}")
        print(f"Rows inserted or updated: {affected_rows}")

        if not rates_df.empty:
            print("\nFirst timestamp:")
            print(rates_df["timestamp_utc"].min())

            print("\nLast timestamp:")
            print(rates_df["timestamp_utc"].max())

    finally:
        shutdown_mt5()
        print("\nMetaTrader 5 connection closed.")


if __name__ == "__main__":
    main()