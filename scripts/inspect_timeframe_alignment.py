import argparse

import pandas as pd

from mdp.database.candle_repository import (
    fetch_market_candles,
    fetch_market_candles_between,
)
from mdp.timeframes.alignment import (
    build_parent_time_window,
    validate_child_timeframe_alignment,
)


def parse_arguments():
    """
    Parses command-line arguments for timeframe alignment inspection.
    """
    parser = argparse.ArgumentParser(
        description="Inspect whether lower-timeframe candles align inside a parent candle."
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

    return parser.parse_args()


def get_parent_start_time(
    symbol: str,
    parent_timeframe: str,
    parent_timestamp: str | None,
) -> pd.Timestamp:
    """
    Gets the parent candle start time from user input or from the latest stored parent candle.
    """
    if parent_timestamp:
        return pd.to_datetime(parent_timestamp)

    parent_df = fetch_market_candles(
        symbol=symbol,
        timeframe=parent_timeframe,
        limit=1,
    )

    if parent_df.empty:
        raise ValueError(
            f"No parent candles found for {symbol} {parent_timeframe}."
        )

    return pd.to_datetime(parent_df.iloc[-1]["timestamp_utc"])


def print_alignment_report(report: dict) -> None:
    """
    Prints a readable alignment report.
    """
    print(f"\nChild timeframe: {report['child_timeframe']}")
    print("-" * 35)
    print(f"Parent window: {report['parent_start']} → {report['parent_end']}")
    print(f"Expected child candles: {report['expected_child_candles']}")
    print(f"Actual child candles: {report['actual_child_candles']}")
    print(f"Time sorted: {report['is_time_sorted']}")
    print(f"Aligned: {report['is_aligned']}")
    print(f"First child timestamp: {report['first_child_timestamp']}")
    print(f"Last child timestamp: {report['last_child_timestamp']}")


def main():
    args = parse_arguments()

    symbol = args.symbol
    parent_timeframe = args.parent_timeframe.upper()
    child_timeframes = [timeframe.upper() for timeframe in args.child_timeframes]

    parent_start_time = get_parent_start_time(
        symbol=symbol,
        parent_timeframe=parent_timeframe,
        parent_timestamp=args.parent_timestamp,
    )

    parent_start, parent_end = build_parent_time_window(
        parent_start_time=parent_start_time,
        parent_timeframe=parent_timeframe,
    )

    print("Timeframe Alignment Inspection")
    print("=" * 45)
    print(f"Symbol: {symbol}")
    print(f"Parent timeframe: {parent_timeframe}")
    print(f"Parent start: {parent_start}")
    print(f"Parent end: {parent_end}")

    for child_timeframe in child_timeframes:
        child_df = fetch_market_candles_between(
            symbol=symbol,
            timeframe=child_timeframe,
            start_time=parent_start,
            end_time=parent_end,
        )

        report = validate_child_timeframe_alignment(
            child_candles_df=child_df,
            parent_start_time=parent_start,
            parent_timeframe=parent_timeframe,
            child_timeframe=child_timeframe,
        )

        print_alignment_report(report)


if __name__ == "__main__":
    main()