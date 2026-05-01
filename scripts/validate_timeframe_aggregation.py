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
    Parses command-line arguments for timeframe aggregation validation.
    """
    parser = argparse.ArgumentParser(
        description="Validate whether child timeframe candles reconstruct the parent OHLC candle."
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
        "--parent-timestamp",
        required=False,
        help='Optional parent candle timestamp, for example: "2026-05-01 20:00:00".',
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.000001,
        help="Allowed OHLC difference between parent and aggregated child candle.",
    )

    return parser.parse_args()


def get_parent_candle(
    symbol: str,
    parent_timeframe: str,
    parent_timestamp: str | None,
) -> dict:
    """
    Gets the parent candle either by a specific timestamp or by the latest stored candle.
    """
    if parent_timestamp:
        parent_start = pd.to_datetime(parent_timestamp)

        parent_df = fetch_market_candles_between(
            symbol=symbol,
            timeframe=parent_timeframe,
            start_time=parent_start,
            end_time=parent_start + pd.Timedelta(seconds=1),
        )
    else:
        parent_df = fetch_market_candles(
            symbol=symbol,
            timeframe=parent_timeframe,
            limit=1,
        )

    if parent_df.empty:
        raise ValueError(
            f"No parent candle found for {symbol} {parent_timeframe}."
        )

    return parent_df.iloc[-1].to_dict()


def print_aggregation_report(
    child_timeframe: str,
    alignment_report: dict,
    aggregation_report: dict,
) -> None:
    """
    Prints a readable parent-child timeframe aggregation report.
    """
    print(f"\nChild timeframe: {child_timeframe}")
    print("-" * 45)

    print(f"Parent window: {alignment_report['parent_start']} → {alignment_report['parent_end']}")
    print(f"Expected child candles: {alignment_report['expected_child_candles']}")
    print(f"Actual child candles: {alignment_report['actual_child_candles']}")
    print(f"Time aligned: {alignment_report['is_aligned']}")

    print(f"OHLC match: {aggregation_report['is_ohlc_match']}")

    print("\nParent OHLC:")
    for key, value in aggregation_report["parent"].items():
        print(f"  {key}: {value}")

    print("\nAggregated child OHLC:")
    for key, value in aggregation_report["aggregated_child"].items():
        print(f"  {key}: {value}")

    print("\nOHLC differences:")
    for key, value in aggregation_report["differences"].items():
        print(f"  {key}: {value}")


def main():
    args = parse_arguments()

    symbol = args.symbol
    parent_timeframe = args.parent_timeframe.upper()
    child_timeframes = [timeframe.upper() for timeframe in args.child_timeframes]

    parent_candle = get_parent_candle(
        symbol=symbol,
        parent_timeframe=parent_timeframe,
        parent_timestamp=args.parent_timestamp,
    )

    parent_start_time = pd.to_datetime(parent_candle["timestamp_utc"])

    parent_start, parent_end = build_parent_time_window(
        parent_start_time=parent_start_time,
        parent_timeframe=parent_timeframe,
    )

    print("Timeframe Aggregation Validation")
    print("=" * 50)
    print(f"Symbol: {symbol}")
    print(f"Parent timeframe: {parent_timeframe}")
    print(f"Parent timestamp: {parent_start_time}")
    print(f"Parent window: {parent_start} → {parent_end}")

    for child_timeframe in child_timeframes:
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
            tolerance=args.tolerance,
        )

        print_aggregation_report(
            child_timeframe=child_timeframe,
            alignment_report=alignment_report,
            aggregation_report=aggregation_report,
        )


if __name__ == "__main__":
    main()