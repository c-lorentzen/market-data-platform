import argparse

import pandas as pd

from mdp.database.candle_loader import load_market_candles


def parse_arguments():
    """
    Parses command-line arguments for flexible candle loading.
    """
    parser = argparse.ArgumentParser(
        description="Test flexible candle loading by limit or date range."
    )

    parser.add_argument(
        "--symbol",
        required=True,
        help='Market symbol, for example: "Step Index".',
    )

    parser.add_argument(
        "--timeframe",
        required=True,
        help="Timeframe, for example: M1, M5, M15, H1.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=300,
        help="Number of recent candles to load when no date range is provided.",
    )

    parser.add_argument(
        "--start-time",
        default=None,
        help='Start datetime, for example: "2026-05-04 20:00:00".',
    )

    parser.add_argument(
        "--end-time",
        default=None,
        help='End datetime, for example: "2026-05-04 22:00:00".',
    )

    parser.add_argument(
        "--lookback-days",
        type=int,
        default=None,
        help="Calendar lookback in days.",
    )

    parser.add_argument(
        "--lookback-weeks",
        type=int,
        default=None,
        help="Calendar lookback in weeks.",
    )

    parser.add_argument(
        "--lookback-months",
        type=int,
        default=None,
        help="Calendar-aware lookback in months.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    candles_df = load_market_candles(
        symbol=args.symbol,
        timeframe=args.timeframe.upper(),
        limit=args.limit,
        start_time=args.start_time,
        end_time=args.end_time,
        lookback_days=args.lookback_days,
        lookback_weeks=args.lookback_weeks,
        lookback_months=args.lookback_months,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)

    print("Flexible Candle Loader Test")
    print("=" * 50)
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe.upper()}")
    print(f"Rows loaded: {len(candles_df)}")

    if candles_df.empty:
        print("No candles found.")
        return

    print(f"First timestamp: {candles_df['timestamp_utc'].min()}")
    print(f"Last timestamp: {candles_df['timestamp_utc'].max()}")

    print("\nFirst 5 rows:")
    print(candles_df.head())

    print("\nLast 5 rows:")
    print(candles_df.tail())


if __name__ == "__main__":
    main()