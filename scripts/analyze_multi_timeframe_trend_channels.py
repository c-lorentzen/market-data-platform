import argparse
from pathlib import Path

import pandas as pd

from mdp.structure.multi_timeframe_trend_channels import (
    analyze_multi_timeframe_channels,
    parse_timeframes,
    summarize_alignment,
)


def parse_arguments():
    """
    Parses command-line options for scale-aware multi-timeframe trend channel analysis.
    """
    parser = argparse.ArgumentParser(
        description="Analyze scale-aware active rolling trend channels across multiple timeframes."
    )

    parser.add_argument(
        "--symbol",
        required=True,
        help='Market symbol, for example: "Step Index" or "Volatility 75 Index".',
    )
    parser.add_argument(
        "--timeframes",
        default="D1,H4,H1,M15,M5,M1",
        help='Comma or space-separated timeframes, for example: "H1,M15,M5,M1".',
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=300,
        help="Number of candles to load per timeframe.",
    )

    parser.add_argument(
        "--end-time",
        default=None,
        help="Optional historical cutoff timestamp. Only candles at or before this time are used.",
    )
    parser.add_argument(
        "--include-forming-candle",
        action="store_true",
        help="Allow the candle currently forming at --end-time. Default uses only fully closed candles.",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=20,
        help="Rolling step size used when active-only is disabled.",
    )
    parser.add_argument(
        "--top-n-per-timeframe",
        type=int,
        default=3,
        help="Maximum valid channel candidates to keep per timeframe.",
    )
    parser.add_argument(
        "--include-historical",
        action="store_true",
        help="Include older rolling windows instead of only latest active windows.",
    )
    parser.add_argument(
        "--recent-window",
        type=int,
        default=None,
        help="Optional override for adaptive recent market-stage window.",
    )
    parser.add_argument(
        "--atr-window",
        type=int,
        default=None,
        help="Optional override for adaptive ATR calculation window.",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save summary and candidate tables as CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/multi_timeframe_channels",
        help="Folder used when saving CSV outputs.",
    )

    return parser.parse_args()


def print_alignment_report(alignment_summary: dict) -> None:
    """
    Prints the high-level alignment summary.
    """
    print("\nAlignment Summary")
    print("-" * 90)

    # Print each alignment field in a simple readable format.
    for key, value in alignment_summary.items():
        print(f"{key}: {value}")


def print_summary_table(summary_df: pd.DataFrame) -> None:
    """
    Prints the compact timeframe summary table.
    """
    print("\nScale-Aware Timeframe Summary")
    print("-" * 90)

    if summary_df.empty:
        print("No timeframe summary rows were created.")
        return

    display_columns = [
        "timeframe",
        "purpose",
        "status",
        "scale_status",
        "candles_analyzed",
        "raw_channels",
        "valid_channels",
        "volatility_stage",
        "best_channel_role",
        "best_channel_type",
        "best_current_state",
        "channel_bias",
        "best_window_size",
        "best_rolling_quality_score",
        "best_r_squared",
        "best_containment_rate",
        "best_channel_width",
        "adaptive_channel_min_width",
        "adaptive_channel_max_width",
        "best_width_to_range_ratio",
        "latest_close",
        "latest_timestamp",
    ]

    existing_columns = [column for column in display_columns if column in summary_df.columns]
    print(summary_df[existing_columns])


def print_adaptive_threshold_table(summary_df: pd.DataFrame) -> None:
    """
    Prints adaptive thresholds used by each timeframe.
    """
    print("\nAdaptive Thresholds Used")
    print("-" * 90)

    if summary_df.empty:
        print("No adaptive threshold rows were created.")
        return

    display_columns = [
        "timeframe",
        "purpose",
        "volatility_stage",
        "adaptive_channel_min_width",
        "adaptive_channel_max_width",
        "adaptive_breakout_threshold",
        "adaptive_retest_tolerance",
        "adaptive_stop_buffer",
        "adaptive_min_target_distance",
    ]

    existing_columns = [column for column in display_columns if column in summary_df.columns]
    print(summary_df[existing_columns])


def print_candidate_table(candidates_df: pd.DataFrame) -> None:
    """
    Prints the ranked valid channel candidates.
    """
    print("\nValid Scale-Aware Channel Candidates")
    print("-" * 90)

    if candidates_df.empty:
        print("No valid channel candidates passed the adaptive timeframe filters.")
        return

    display_columns = [
        "timeframe",
        "purpose",
        "rolling_channel_id",
        "window_size",
        "channel_role",
        "rolling_quality_score",
        "channel_type",
        "current_state",
        "r_squared",
        "containment_rate",
        "channel_width",
        "adaptive_channel_min_width",
        "adaptive_channel_max_width",
        "channel_width_usage_ratio",
        "width_to_range_ratio",
        "current_position_ratio",
        "current_lower",
        "current_upper",
    ]

    existing_columns = [column for column in display_columns if column in candidates_df.columns]
    print(candidates_df[existing_columns])


def save_results(
    symbol: str,
    summary_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """
    Saves multi-timeframe analysis tables to CSV files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Keep filenames clean and easy to identify.
    clean_symbol = symbol.lower().replace(" ", "_")
    summary_path = output_path / f"{clean_symbol}_scale_aware_mtf_channel_summary.csv"
    candidates_path = output_path / f"{clean_symbol}_scale_aware_mtf_channel_candidates.csv"

    summary_df.to_csv(summary_path, index=False)
    candidates_df.to_csv(candidates_path, index=False)

    print("\nSaved results:")
    print(f"Summary: {summary_path}")
    print(f"Candidates: {candidates_path}")


def main():
    args = parse_arguments()

    symbol = args.symbol
    timeframes = parse_timeframes(args.timeframes)
    active_only = not args.include_historical

    # Run the multi-timeframe channel analysis using adaptive scale values.
    summary_df, candidates_df = analyze_multi_timeframe_channels(
        symbol=symbol,
        timeframes=timeframes,
        limit=args.limit,
        step_size=args.step_size,
        top_n_per_timeframe=args.top_n_per_timeframe,
        active_only=active_only,
        recent_window=args.recent_window,
        atr_window=args.atr_window,
        end_time=args.end_time,
        closed_candles_only=not args.include_forming_candle,
    )

    # Build a compact alignment summary for macro-to-micro interpretation.
    alignment_summary = summarize_alignment(summary_df)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 360)

    print("Scale-Aware Multi-Timeframe Trend Channel Analysis Report")
    print("=" * 90)
    print(f"Symbol: {symbol}")
    print(f"Timeframes: {timeframes}")
    print(f"Limit per timeframe: {args.limit}")
    print(f"Active only: {active_only}")
    print(f"Recent window override: {args.recent_window}")
    print(f"ATR window override: {args.atr_window}")
    print(f"Analysis cutoff: {args.end_time}")
    print(f"Closed candles only: {not args.include_forming_candle}")

    print_alignment_report(alignment_summary)
    print_summary_table(summary_df)
    print_adaptive_threshold_table(summary_df)
    print_candidate_table(candidates_df)

    if args.save_results:
        # Save tables only when the user explicitly requests outputs.
        save_results(
            symbol=symbol,
            summary_df=summary_df,
            candidates_df=candidates_df,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
