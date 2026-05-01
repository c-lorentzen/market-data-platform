import argparse

from mdp.database.candle_repository import fetch_market_candles
from mdp.features.basic_features import add_basic_candle_features


FEATURE_COLUMNS_TO_DISPLAY = [
    "timestamp_utc",
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "close_return",
    "candle_range",
    "candle_body",
    "upper_wick",
    "lower_wick",
    "body_to_range_ratio",
    "close_position_in_range",
    "rolling_volatility_5",
    "rolling_volatility_20",
    "rolling_high_20",
    "rolling_low_20",
]


def parse_arguments():
    """
    Parses command-line arguments for basic feature generation.
    """
    parser = argparse.ArgumentParser(
        description="Build basic candle features from market candles stored in MySQL."
    )

    parser.add_argument(
        "--symbol",
        required=True,
        help='Market symbol, for example: "Step Index" or "Volatility 75 Index".',
    )

    parser.add_argument(
        "--timeframe",
        required=True,
        help="Timeframe, for example: M1, M5, M15, H1, H4, D1.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of recent candles to use for feature generation.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    symbol = args.symbol
    timeframe = args.timeframe.upper()
    limit = args.limit

    candles_df = fetch_market_candles(
        symbol=symbol,
        timeframe=timeframe,
        limit=limit,
    )

    features_df = add_basic_candle_features(candles_df)

    print("Basic candle features created successfully.")
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Input candles: {len(candles_df)}")
    print(f"Output rows: {len(features_df)}")
    print(f"Output columns: {len(features_df.columns)}")

    print("\nLast 10 rows with selected features:")
    print(features_df[FEATURE_COLUMNS_TO_DISPLAY].tail(10))

    print("\nMissing values in selected features:")
    print(features_df[FEATURE_COLUMNS_TO_DISPLAY].isna().sum())


if __name__ == "__main__":
    main()