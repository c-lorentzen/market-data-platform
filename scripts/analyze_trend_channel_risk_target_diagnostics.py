import argparse
from pathlib import Path

import pandas as pd

from mdp.backtesting.trend_channel_risk_target_diagnostics import run_risk_target_diagnostics


def parse_arguments():
    """
    Parses command-line options for SL/TP diagnostic analysis.
    """
    parser = argparse.ArgumentParser(
        description="Analyze historical SL/TP efficiency for trend-channel trade plans."
    )

    parser.add_argument("--symbol", required=True, help='Market symbol, for example: "Step Index".')
    parser.add_argument("--timeframes", default="H1,M1", help='Comma-separated timeframes, for example: "H1,M1".')
    parser.add_argument("--analysis-limit", type=int, default=300, help="Past candles used at each cutoff.")
    parser.add_argument("--dataset-limit", type=int, default=2000, help="Entry-timeframe candles used to create cutoffs.")
    parser.add_argument("--start-time", default=None, help="Optional dataset start timestamp.")
    parser.add_argument("--end-time", default=None, help="Optional dataset end timestamp.")
    parser.add_argument("--warmup-candles", type=int, default=300, help="Warmup candles before first cutoff.")
    parser.add_argument("--horizon-candles", type=int, default=120, help="Future candles used for outcome labeling.")
    parser.add_argument("--cutoff-step-candles", type=int, default=20, help="Spacing between historical cutoffs.")
    parser.add_argument("--step-size", type=int, default=20, help="Rolling channel step size.")
    parser.add_argument("--top-n-per-timeframe", type=int, default=3, help="Valid channel candidates per timeframe.")
    parser.add_argument("--include-historical", action="store_true", help="Include older rolling channel windows.")
    parser.add_argument("--recent-window", type=int, default=None, help="Override adaptive recent market-stage window.")
    parser.add_argument("--atr-window", type=int, default=None, help="Override adaptive ATR window.")
    parser.add_argument("--event-lookback", type=int, default=40, help="Recent candles used to detect events.")
    parser.add_argument("--max-retest-candles", type=int, default=20, help="Max candles after breakout to search for retest.")
    parser.add_argument("--max-false-breakout-candles", type=int, default=10, help="Max candles for false-breakout detection.")
    parser.add_argument("--max-confirmation-candles", type=int, default=20, help="Max candles after retest for continuation.")
    parser.add_argument("--min-rejection-wick-ratio", type=float, default=0.35, help="Minimum wick ratio for retest rejection.")
    parser.add_argument("--continuation-multiplier", type=float, default=2.0, help="Continuation distance multiplier.")
    parser.add_argument("--minimum-rr", type=float, default=1.50, help="Minimum RR for target availability.")
    parser.add_argument("--actionable-only", action="store_true", help="Exclude non-actionable cutoffs from raw outcome rows.")
    parser.add_argument("--save-results", action="store_true", help="Save diagnostic CSV files.")
    parser.add_argument("--output-dir", default="outputs/trend_channel_risk_diagnostics", help="Output folder for CSV files.")

    return parser.parse_args()


def print_report(args, outcomes_df, outcome_summary_df, cutoff_schedule_df, diagnostic_summary_df, grouped_df, detail_df):
    """
    Prints a readable SL/TP diagnostic report.
    """
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 260)

    print("Trend Channel Risk/Target Diagnostic Report")
    print("=" * 90)
    print(f"Symbol: {args.symbol}")
    print(f"Timeframes: {args.timeframes}")
    print(f"Analysis limit per cutoff: {args.analysis_limit}")
    print(f"Dataset limit: {args.dataset_limit}")
    print(f"Warmup candles: {args.warmup_candles}")
    print(f"Horizon candles: {args.horizon_candles}")
    print(f"Cutoff step candles: {args.cutoff_step_candles}")
    print(f"Minimum RR: {args.minimum_rr}")
    print()

    print("Original Outcome Dataset Summary")
    print("-" * 90)
    print(outcome_summary_df)
    print()

    print("SL/TP Diagnostic Summary")
    print("-" * 90)
    print(diagnostic_summary_df)
    print()

    print("Grouped Diagnostics by Event / Plan Type")
    print("-" * 90)
    if grouped_df.empty:
        print("No grouped actionable diagnostics were available.")
    else:
        print(grouped_df)
    print()

    print("Actionable Rows Detail Preview")
    print("-" * 90)
    if detail_df.empty:
        print("No actionable rows were available.")
    else:
        print(detail_df.head(25))
    print()

    print("Cutoff Schedule Preview")
    print("-" * 90)
    if cutoff_schedule_df.empty:
        print("No cutoff schedule was created.")
    else:
        print(cutoff_schedule_df.head(5))


def save_results(args, outcomes_df, outcome_summary_df, cutoff_schedule_df, diagnostic_summary_df, grouped_df, detail_df):
    """
    Saves raw outcome rows and diagnostic tables for later review.
    """
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    safe_symbol = args.symbol.lower().replace(" ", "_")

    files = {
        "outcomes": output_path / f"{safe_symbol}_risk_diagnostic_outcomes.csv",
        "outcome_summary": output_path / f"{safe_symbol}_risk_diagnostic_outcome_summary.csv",
        "cutoff_schedule": output_path / f"{safe_symbol}_risk_diagnostic_cutoff_schedule.csv",
        "diagnostic_summary": output_path / f"{safe_symbol}_risk_diagnostic_summary.csv",
        "grouped_diagnostics": output_path / f"{safe_symbol}_risk_diagnostic_grouped.csv",
        "actionable_detail": output_path / f"{safe_symbol}_risk_diagnostic_actionable_detail.csv",
    }

    outcomes_df.to_csv(files["outcomes"], index=False)
    outcome_summary_df.to_csv(files["outcome_summary"], index=False)
    cutoff_schedule_df.to_csv(files["cutoff_schedule"], index=False)
    diagnostic_summary_df.to_csv(files["diagnostic_summary"], index=False)
    grouped_df.to_csv(files["grouped_diagnostics"], index=False)
    detail_df.to_csv(files["actionable_detail"], index=False)

    print()
    print("Saved files")
    print("-" * 90)
    for label, path in files.items():
        print(f"{label}: {path}")


def main():
    """
    Runs SL/TP diagnostics from the command line.
    """
    args = parse_arguments()

    outcomes_df, outcome_summary_df, cutoff_schedule_df, diagnostic_summary_df, grouped_df, detail_df = run_risk_target_diagnostics(
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
        args=args,
        outcomes_df=outcomes_df,
        outcome_summary_df=outcome_summary_df,
        cutoff_schedule_df=cutoff_schedule_df,
        diagnostic_summary_df=diagnostic_summary_df,
        grouped_df=grouped_df,
        detail_df=detail_df,
    )

    if args.save_results:
        save_results(
            args=args,
            outcomes_df=outcomes_df,
            outcome_summary_df=outcome_summary_df,
            cutoff_schedule_df=cutoff_schedule_df,
            diagnostic_summary_df=diagnostic_summary_df,
            grouped_df=grouped_df,
            detail_df=detail_df,
        )


if __name__ == "__main__":
    main()
