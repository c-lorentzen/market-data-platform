import argparse
from pathlib import Path

import pandas as pd

from mdp.backtesting.trend_channel_outcome_dataset import (
    build_historical_trend_channel_outcome_dataset,
)


def parse_arguments():
    """
    Parses command-line options for historical trend-channel outcome labeling.
    """
    parser = argparse.ArgumentParser(
        description="Build a cutoff-aware historical SL/TP outcome dataset for trend-channel trade plans."
    )

    parser.add_argument(
        "--symbol",
        required=True,
        help='Market symbol, for example: "Step Index" or "Volatility 75 Index".',
    )
    parser.add_argument(
        "--timeframes",
        default="H1,M1",
        help='Comma or space-separated timeframes, for example: "H1,M15,M5,M1".',
    )
    parser.add_argument(
        "--analysis-limit",
        type=int,
        default=300,
        help="Number of past candles used by the analysis pipeline at each cutoff.",
    )
    parser.add_argument(
        "--dataset-limit",
        type=int,
        default=2000,
        help="Number of entry-timeframe candles used to create historical cutoff times.",
    )
    parser.add_argument(
        "--start-time",
        default=None,
        help="Optional dataset start timestamp for the entry timeframe.",
    )
    parser.add_argument(
        "--end-time",
        default=None,
        help="Optional dataset end timestamp for the entry timeframe.",
    )
    parser.add_argument(
        "--warmup-candles",
        type=int,
        default=300,
        help="Minimum past entry-timeframe candles required before the first cutoff.",
    )
    parser.add_argument(
        "--horizon-candles",
        type=int,
        default=120,
        help="Future entry-timeframe candles used to label SL/TP outcomes.",
    )
    parser.add_argument(
        "--cutoff-step-candles",
        type=int,
        default=20,
        help="Spacing between historical cutoff points.",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=20,
        help="Rolling channel step size used when historical windows are included.",
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
        help="Include older rolling channel windows instead of only latest active windows.",
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
        "--event-lookback",
        type=int,
        default=40,
        help="Recent entry-timeframe candles used to detect breakout events.",
    )
    parser.add_argument(
        "--max-retest-candles",
        type=int,
        default=20,
        help="Maximum candles after breakout to search for a retest.",
    )
    parser.add_argument(
        "--max-false-breakout-candles",
        type=int,
        default=10,
        help="Maximum candles after breakout to classify a false breakout.",
    )
    parser.add_argument(
        "--max-confirmation-candles",
        type=int,
        default=20,
        help="Maximum candles after retest to look for continuation confirmation.",
    )
    parser.add_argument(
        "--min-rejection-wick-ratio",
        type=float,
        default=0.35,
        help="Minimum wick ratio required for direct retest rejection classification.",
    )
    parser.add_argument(
        "--continuation-multiplier",
        type=float,
        default=2.0,
        help="Continuation distance multiplier relative to adaptive breakout threshold.",
    )
    parser.add_argument(
        "--minimum-rr",
        type=float,
        default=1.50,
        help="Minimum RR required for target availability.",
    )
    parser.add_argument(
        "--actionable-only",
        action="store_true",
        help="Only keep cutoffs where a valid trade plan was produced.",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save outcome, summary, and cutoff schedule CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/trend_channel_outcomes",
        help="Folder where historical outcome CSV files are saved.",
    )

    return parser.parse_args()


def print_report(symbol: str, timeframes: str, args, outcomes_df, summary_df, cutoff_schedule_df):
    """
    Prints a readable historical outcome dataset report.
    """
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)

    print("Historical Trend Channel Outcome Dataset Report")
    print("=" * 90)
    print(f"Symbol: {symbol}")
    print(f"Timeframes: {timeframes}")
    print(f"Analysis limit per cutoff: {args.analysis_limit}")
    print(f"Dataset limit: {args.dataset_limit}")
    print(f"Warmup candles: {args.warmup_candles}")
    print(f"Horizon candles: {args.horizon_candles}")
    print(f"Cutoff step candles: {args.cutoff_step_candles}")
    print(f"Actionable only: {args.actionable_only}")
    print()

    print("Dataset Summary")
    print("-" * 90)
    print(summary_df)
    print()

    print("Cutoff Schedule Preview")
    print("-" * 90)
    if cutoff_schedule_df.empty:
        print("No cutoff schedule was created.")
    else:
        print(cutoff_schedule_df.head(10))
    print()

    print("Outcome Rows Preview")
    print("-" * 90)
    if outcomes_df.empty:
        print("No outcome rows were created.")
    else:
        preview_columns = [
            "analysis_cutoff",
            "plan_state",
            "plan_direction",
            "event_state",
            "outcome_state",
            "first_terminal_outcome",
            "best_target_hit",
            "best_target_rr_hit",
            "stop_hit",
            "tp2_hit_before_stop",
            "tp3_hit_before_stop",
            "max_favorable_r",
            "max_adverse_r",
            "label_quality",
        ]
        available_columns = [column for column in preview_columns if column in outcomes_df.columns]
        print(outcomes_df[available_columns].head(20))


def save_results(output_dir: str, symbol: str, outcomes_df, summary_df, cutoff_schedule_df):
    """
    Saves the generated historical dataset as CSV files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    safe_symbol = symbol.lower().replace(" ", "_")

    outcomes_file = output_path / f"{safe_symbol}_trend_channel_outcomes.csv"
    summary_file = output_path / f"{safe_symbol}_trend_channel_outcome_summary.csv"
    schedule_file = output_path / f"{safe_symbol}_trend_channel_cutoff_schedule.csv"

    outcomes_df.to_csv(outcomes_file, index=False)
    summary_df.to_csv(summary_file, index=False)
    cutoff_schedule_df.to_csv(schedule_file, index=False)

    print()
    print("Saved files")
    print("-" * 90)
    print(f"Outcomes: {outcomes_file}")
    print(f"Summary: {summary_file}")
    print(f"Cutoff schedule: {schedule_file}")


def main():
    """
    Runs cutoff-aware historical outcome labeling from the command line.
    """
    args = parse_arguments()

    outcomes_df, summary_df, cutoff_schedule_df = build_historical_trend_channel_outcome_dataset(
        symbol=args.symbol,
        timeframes=args.timeframes,
        analysis_limit=args.analysis_limit,
        dataset_limit=args.dataset_limit,
        start_time=args.start_time,
        end_time=args.end_time,
        warmup_candles=args.warmup_candles,
        horizon_candles=args.horizon_candles,
        cutoff_step_candles=args.cutoff_step_candles,
        step_size=args.step_size,
        top_n_per_timeframe=args.top_n_per_timeframe,
        active_only=not args.include_historical,
        recent_window=args.recent_window,
        atr_window=args.atr_window,
        event_lookback=args.event_lookback,
        max_retest_candles=args.max_retest_candles,
        max_false_breakout_candles=args.max_false_breakout_candles,
        max_confirmation_candles=args.max_confirmation_candles,
        min_rejection_wick_ratio=args.min_rejection_wick_ratio,
        continuation_multiplier=args.continuation_multiplier,
        minimum_rr=args.minimum_rr,
        include_non_actionable=not args.actionable_only,
    )

    print_report(
        symbol=args.symbol,
        timeframes=args.timeframes,
        args=args,
        outcomes_df=outcomes_df,
        summary_df=summary_df,
        cutoff_schedule_df=cutoff_schedule_df,
    )

    if args.save_results:
        save_results(
            output_dir=args.output_dir,
            symbol=args.symbol,
            outcomes_df=outcomes_df,
            summary_df=summary_df,
            cutoff_schedule_df=cutoff_schedule_df,
        )


if __name__ == "__main__":
    main()
