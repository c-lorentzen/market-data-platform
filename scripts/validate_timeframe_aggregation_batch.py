import argparse

import pandas as pd

from mdp.database.candle_repository import (
    fetch_market_candles,
    fetch_market_candles_between,
)
from mdp.timeframes.alignment import (
    build_parent_time_window,
    compare_parent_to_child_aggregation,
    validate_child_timeframe_alignment,
)


def parse_arguments():
    """
    Parses command-line arguments for batch timeframe aggregation validation.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Validate whether multiple parent candles can be reconstructed "
            "from lower-timeframe child candles."
        )
    )

    parser.add_argument(
        "--symbol",
        required=True,
        help='Market symbol, for example: "Step Index".',
    )

    parser.add_argument(
        "--parent-timeframe",
        required=True,
        help="Parent timeframe, for example: H1 or H4.",
    )

    parser.add_argument(
        "--child-timeframes",
        nargs="+",
        required=True,
        help="Child timeframes, for example: M15 M5 M1.",
    )

    parser.add_argument(
        "--parent-limit",
        type=int,
        default=10,
        help="Number of recent parent candles to validate.",
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.000001,
        help="Allowed OHLC difference between parent and aggregated child candle.",
    )

    return parser.parse_args()


def validate_parent_child_pair(
    symbol: str,
    parent_candle: dict,
    parent_timeframe: str,
    child_timeframe: str,
    tolerance: float,
) -> dict:
    """
    Validates one parent candle against one child timeframe.
    """
    parent_start_time = pd.to_datetime(parent_candle["timestamp_utc"])

    parent_start, parent_end = build_parent_time_window(
        parent_start_time=parent_start_time,
        parent_timeframe=parent_timeframe,
    )

    child_df = fetch_market_candles_between(
        symbol=symbol,
        timeframe=child_timeframe,
        start_time=parent_start,
        end_time=parent_end,
    )

    alignment_report = validate_child_timeframe_alignment(
        child_candles_df=child_df,
        parent_start_time=parent_start,
        parent_timeframe=parent_timeframe,
        child_timeframe=child_timeframe,
    )

    aggregation_report = compare_parent_to_child_aggregation(
        parent_candle=parent_candle,
        child_candles_df=child_df,
        tolerance=tolerance,
    )

    return {
        "symbol": symbol,
        "parent_timeframe": parent_timeframe,
        "child_timeframe": child_timeframe,
        "parent_timestamp": parent_start,
        "parent_end": parent_end,
        "expected_child_candles": alignment_report["expected_child_candles"],
        "actual_child_candles": alignment_report["actual_child_candles"],
        "is_time_aligned": alignment_report["is_aligned"],
        "is_ohlc_match": aggregation_report["is_ohlc_match"],
        "open_difference": aggregation_report["differences"].get("open_price"),
        "high_difference": aggregation_report["differences"].get("high_price"),
        "low_difference": aggregation_report["differences"].get("low_price"),
        "close_difference": aggregation_report["differences"].get("close_price"),
    }


def summarize_batch_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes batch validation results by child timeframe.
    """
    if results_df.empty:
        return pd.DataFrame()

    summary_df = (
        results_df.groupby("child_timeframe")
        .agg(
            parent_candles_checked=("parent_timestamp", "count"),
            fully_time_aligned=("is_time_aligned", "sum"),
            ohlc_matches=("is_ohlc_match", "sum"),
            average_actual_child_candles=("actual_child_candles", "mean"),
            max_open_difference=("open_difference", "max"),
            max_high_difference=("high_difference", "max"),
            max_low_difference=("low_difference", "max"),
            max_close_difference=("close_difference", "max"),
        )
        .reset_index()
    )

    summary_df["time_alignment_rate"] = (
        summary_df["fully_time_aligned"] / summary_df["parent_candles_checked"] * 100
    )

    summary_df["ohlc_match_rate"] = (
        summary_df["ohlc_matches"] / summary_df["parent_candles_checked"] * 100
    )

    return summary_df


def main():
    args = parse_arguments()

    symbol = args.symbol
    parent_timeframe = args.parent_timeframe.upper()
    child_timeframes = [timeframe.upper() for timeframe in args.child_timeframes]

    parent_df = fetch_market_candles(
        symbol=symbol,
        timeframe=parent_timeframe,
        limit=args.parent_limit,
    )

    if parent_df.empty:
        raise ValueError(f"No parent candles found for {symbol} {parent_timeframe}.")

    results = []

    for _, parent_row in parent_df.iterrows():
        parent_candle = parent_row.to_dict()

        for child_timeframe in child_timeframes:
            result = validate_parent_child_pair(
                symbol=symbol,
                parent_candle=parent_candle,
                parent_timeframe=parent_timeframe,
                child_timeframe=child_timeframe,
                tolerance=args.tolerance,
            )

            results.append(result)

    results_df = pd.DataFrame(results)
    summary_df = summarize_batch_results(results_df)

    print("Batch Timeframe Aggregation Validation")
    print("=" * 55)
    print(f"Symbol: {symbol}")
    print(f"Parent timeframe: {parent_timeframe}")
    print(f"Parent candles checked: {len(parent_df)}")
    print(f"Child timeframes: {', '.join(child_timeframes)}")

    print("\nSummary by child timeframe:")
    print(summary_df)

    print("\nDetailed results:")
    print(results_df)


if __name__ == "__main__":
    main()