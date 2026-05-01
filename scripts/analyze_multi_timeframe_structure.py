import argparse

from mdp.database.candle_repository import fetch_market_candles
from mdp.features.basic_features import add_basic_candle_features
from mdp.structure.market_structure import (
    add_market_structure_features,
    summarize_market_structure,
)


def parse_arguments():
    """
    Parses command-line arguments for multi-timeframe market structure analysis.
    """
    parser = argparse.ArgumentParser(
        description="Analyze market structure across multiple timeframes."
    )

    parser.add_argument(
        "--symbol",
        required=True,
        help='Market symbol, for example: "Step Index" or "Volatility 75 Index".',
    )

    parser.add_argument(
        "--timeframes",
        nargs="+",
        required=True,
        help="List of timeframes, for example: M15 H1 H4.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=300,
        help="Number of recent candles to analyze per timeframe.",
    )

    parser.add_argument(
        "--lookback",
        type=int,
        default=20,
        help="Number of candles used to evaluate recent structure.",
    )

    return parser.parse_args()


def analyze_timeframe(
    symbol: str,
    timeframe: str,
    limit: int,
    lookback: int,
) -> dict:
    """
    Runs the full structure analysis pipeline for one symbol and one timeframe.
    """
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

    return {
        "timeframe": timeframe,
        "rows_analyzed": summary["row_count"],
        "latest_timestamp": summary.get("latest_timestamp"),
        "latest_close": summary.get("latest_close"),
        "latest_structure": summary.get("latest_structure"),
        "latest_structure_score": summary.get("latest_structure_score"),
        "structure_percentages": summary.get("structure_percentages", {}),
    }


def print_multi_timeframe_report(results: list[dict]) -> None:
    """
    Prints a readable multi-timeframe market structure report.
    """
    print("Multi-Timeframe Market Structure Report")
    print("=" * 50)

    for result in results:
        print(f"\nTimeframe: {result['timeframe']}")
        print("-" * 30)
        print(f"Rows analyzed: {result['rows_analyzed']}")
        print(f"Latest timestamp: {result['latest_timestamp']}")
        print(f"Latest close: {result['latest_close']}")
        print(f"Latest structure: {result['latest_structure']}")
        print(f"Structure score: {result['latest_structure_score']}")

        print("Structure percentages:")
        for structure, percentage in result["structure_percentages"].items():
            print(f"  - {structure}: {percentage}%")


def main():
    args = parse_arguments()

    symbol = args.symbol
    timeframes = [timeframe.upper() for timeframe in args.timeframes]
    limit = args.limit
    lookback = args.lookback

    results = []

    for timeframe in timeframes:
        result = analyze_timeframe(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
            lookback=lookback,
        )

        results.append(result)

    print_multi_timeframe_report(results)


if __name__ == "__main__":
    main()