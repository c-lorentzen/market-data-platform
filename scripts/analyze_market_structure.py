import argparse

from mdp.database.candle_repository import fetch_market_candles
from mdp.features.basic_features import add_basic_candle_features
from mdp.structure.market_structure import (
    add_market_structure_features,
    summarize_market_structure,
)


COLUMNS_TO_DISPLAY = [
    "timestamp_utc",
    "close_price",
    "close_change_lookback",
    "close_position_in_rolling_range_20",
    "market_structure",
    "market_structure_score",
]


def parse_arguments():
    """
    Parses command-line arguments for basic market structure analysis.
    """
    parser = argparse.ArgumentParser(
        description="Analyze basic market structure from market candles stored in MySQL."
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
        help="Number of recent candles to analyze.",
    )

    parser.add_argument(
        "--lookback",
        type=int,
        default=20,
        help="Number of candles used to evaluate recent structure.",
    )

    return parser.parse_args()


def print_market_structure_summary(summary: dict) -> None:
    """
    Prints a readable market structure summary.
    """
    print("Market Structure Summary")
    print("=" * 35)

    print(f"Rows analyzed: {summary['row_count']}")
    print(f"Latest timestamp: {summary['latest_timestamp']}")
    print(f"Latest close: {summary['latest_close']}")
    print(f"Latest structure: {summary['latest_structure']}")
    print(f"Latest structure score: {summary['latest_structure_score']}")

    print("\nStructure counts:")
    for structure, count in summary["structure_counts"].items():
        print(f"- {structure}: {count}")

    print("\nStructure percentages:")
    for structure, percentage in summary["structure_percentages"].items():
        print(f"- {structure}: {percentage}%")


def main():
    args = parse_arguments()

    symbol = args.symbol
    timeframe = args.timeframe.upper()
    limit = args.limit
    lookback = args.lookback

    candles_df = fetch_market_candles(
        symbol=symbol,
        timeframe=timeframe,
        limit=limit,
    )

    features_df = add_basic_candle_features(candles_df)

    structure_df = add_market_structure_features(
        features_df=features_df,
        lookback=lookback,
    )

    summary = summarize_market_structure(structure_df)

    print_market_structure_summary(summary)

    print("\nLast 15 rows:")
    print(structure_df[COLUMNS_TO_DISPLAY].tail(15))


if __name__ == "__main__":
    main()