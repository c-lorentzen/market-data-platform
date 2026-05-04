import argparse

import pandas as pd

from mdp.database.candle_repository import insert_market_candles_from_dataframe
from mdp.ingestion.mt5_connector import (
    fetch_rates,
    fetch_rates_between,
    initialize_mt5,
    shutdown_mt5,
)
from mdp.timeframes.registry import build_time_window


def parse_arguments():
    """
    Parses command-line arguments for aligned multi-timeframe MT5 ingestion.
    """
    parser = argparse.ArgumentParser(
        description="Ingest MT5 candles aligned by parent timeframe windows."
    )

    parser.add_argument(
        "--symbol",
        required=True,
        help='MT5 symbol name, for example: "Step Index".',
    )

    parser.add_argument(
        "--parent-timeframe",
        required=True,
        help="Parent timeframe, for example: H1, H4, D1.",
    )

    parser.add_argument(
        "--child-timeframes",
        nargs="+",
        default=[],
        help="Child timeframes to ingest inside each parent window, for example: M15 M5 M1.",
    )

    parser.add_argument(
        "--parent-count",
        type=int,
        default=1,
        help="Number of recent closed parent candles to ingest.",
    )

    parser.add_argument(
        "--start-time",
        default=None,
        help='Optional explicit start datetime, for example: "2026-05-01 00:00:00".',
    )

    parser.add_argument(
        "--end-time",
        default=None,
        help='Optional explicit end datetime, for example: "2026-05-04 00:00:00".',
    )

    return parser.parse_args()


def ingest_dataframe(candles_df: pd.DataFrame, label: str) -> int:
    """
    Inserts a DataFrame into MySQL and prints a compact ingestion summary.
    """
    affected_rows = insert_market_candles_from_dataframe(candles_df)

    print(f"{label}")
    print(f"  Candles fetched: {len(candles_df)}")
    print(f"  Rows inserted or updated: {affected_rows}")

    if not candles_df.empty:
        print(f"  First timestamp: {candles_df['timestamp_utc'].min()}")
        print(f"  Last timestamp: {candles_df['timestamp_utc'].max()}")

    return affected_rows


def ingest_explicit_date_range(
    symbol: str,
    timeframes: list[str],
    start_time: str,
    end_time: str,
) -> None:
    """
    Ingests all requested timeframes inside an explicit date range.
    """
    print("Date Range Multi-Timeframe Ingestion")
    print("=" * 55)
    print(f"Symbol: {symbol}")
    print(f"Start time: {start_time}")
    print(f"End time: {end_time}")
    print(f"Timeframes: {', '.join(timeframes)}")

    for timeframe in timeframes:
        candles_df = fetch_rates_between(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
        )

        ingest_dataframe(
            candles_df=candles_df,
            label=f"\nTimeframe: {timeframe}",
        )


def ingest_parent_windows(
    symbol: str,
    parent_timeframe: str,
    child_timeframes: list[str],
    parent_count: int,
) -> None:
    """
    Ingests recent closed parent candles and their aligned child timeframe candles.
    """
    print("Aligned Parent-Child Multi-Timeframe Ingestion")
    print("=" * 55)
    print(f"Symbol: {symbol}")
    print(f"Parent timeframe: {parent_timeframe}")
    print(f"Child timeframes: {', '.join(child_timeframes) if child_timeframes else 'None'}")
    print(f"Parent candles requested: {parent_count}")

    parent_df = fetch_rates(
        symbol=symbol,
        timeframe=parent_timeframe,
        count=parent_count,
        start_pos=1,
    )

    ingest_dataframe(
        candles_df=parent_df,
        label=f"\nParent timeframe: {parent_timeframe}",
    )

    if parent_df.empty:
        return

    for _, parent_row in parent_df.iterrows():
        parent_start_time = parent_row["timestamp_utc"]

        parent_start, parent_end = build_time_window(
            start_time=parent_start_time,
            timeframe=parent_timeframe,
        )

        print("\nParent window")
        print("-" * 35)
        print(f"{parent_start} → {parent_end}")

        for child_timeframe in child_timeframes:
            child_df = fetch_rates_between(
                symbol=symbol,
                timeframe=child_timeframe,
                start_time=parent_start,
                end_time=parent_end,
            )

            ingest_dataframe(
                candles_df=child_df,
                label=f"Child timeframe: {child_timeframe}",
            )


def main():
    args = parse_arguments()

    symbol = args.symbol
    parent_timeframe = args.parent_timeframe.upper()
    child_timeframes = [timeframe.upper() for timeframe in args.child_timeframes]

    initialize_mt5()

    try:
        if args.start_time or args.end_time:
            if not args.start_time or not args.end_time:
                raise ValueError("Both --start-time and --end-time are required for date range mode.")

            timeframes = [parent_timeframe] + child_timeframes

            ingest_explicit_date_range(
                symbol=symbol,
                timeframes=timeframes,
                start_time=args.start_time,
                end_time=args.end_time,
            )

        else:
            ingest_parent_windows(
                symbol=symbol,
                parent_timeframe=parent_timeframe,
                child_timeframes=child_timeframes,
                parent_count=args.parent_count,
            )

    finally:
        shutdown_mt5()
        print("\nMetaTrader 5 connection closed.")


if __name__ == "__main__":
    main()