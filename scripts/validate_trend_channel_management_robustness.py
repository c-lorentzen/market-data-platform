import argparse
from pathlib import Path

import pandas as pd

from mdp.backtesting.trend_channel_management_robustness import validate_management_policy_robustness


def parse_arguments():
    """
    Parses command-line options for lightweight/cached management robustness validation.
    """
    parser = argparse.ArgumentParser(
        description="Validate trend-channel trade-management robustness with cached cutoff analysis."
    )

    parser.add_argument("--symbol", required=True, help='Market symbol, for example: "Step Index".')
    parser.add_argument("--timeframes", default="H1,M1", help='Comma-separated timeframes, for example: "H1,M1".')
    parser.add_argument("--analysis-limit", type=int, default=200, help="Past candles used by each cutoff analysis.")
    parser.add_argument("--dataset-limit", type=int, default=1200, help="Base entry-timeframe candles used for validation.")
    parser.add_argument("--start-time", default=None, help="Optional dataset start timestamp.")
    parser.add_argument("--end-time", default=None, help="Optional dataset end timestamp.")
    parser.add_argument("--warmup-candles", type=int, default=200, help="Past candles required before first cutoff.")
    parser.add_argument("--horizon-candles", type=int, default=80, help="Base future candles used for outcome simulation.")
    parser.add_argument("--step-size", type=int, default=20, help="Rolling channel step size.")
    parser.add_argument("--top-n-per-timeframe", type=int, default=3, help="Valid channel candidates kept per timeframe.")
    parser.add_argument("--include-historical", action="store_true", help="Include older rolling windows instead of active-only windows.")
    parser.add_argument("--recent-window", type=int, default=None, help="Optional adaptive recent-window override.")
    parser.add_argument("--atr-window", type=int, default=None, help="Optional ATR-window override.")
    parser.add_argument("--event-lookback", type=int, default=40, help="Recent candles used to detect events.")
    parser.add_argument("--max-retest-candles", type=int, default=20, help="Maximum candles used to search for retest.")
    parser.add_argument("--max-false-breakout-candles", type=int, default=10, help="Maximum candles used to detect false breakout.")
    parser.add_argument("--max-confirmation-candles", type=int, default=20, help="Maximum candles used to confirm continuation.")
    parser.add_argument("--min-rejection-wick-ratio", type=float, default=0.35, help="Minimum rejection wick ratio.")
    parser.add_argument("--continuation-multiplier", type=float, default=2.0, help="Continuation threshold multiplier.")
    parser.add_argument("--minimum-rr", type=float, default=1.50, help="Minimum RR required for target availability.")
    parser.add_argument("--max-cutoffs", type=int, default=30, help="Maximum historical cutoffs per validation profile.")
    parser.add_argument("--max-validation-runs", type=int, default=3, help="Maximum validation profiles to run.")
    parser.add_argument("--max-policies", type=int, default=4, help="Maximum management policies to compare.")
    parser.add_argument("--quick", action="store_true", help="Use a small smoke-test workload.")
    parser.add_argument("--no-horizon-close", action="store_true", help="Do not mark remaining size at the horizon close.")
    parser.add_argument("--save-results", action="store_true", help="Save robustness CSV files.")
    parser.add_argument("--output-dir", default="outputs/trend_channel_management_robustness", help="Folder where CSV files are saved.")
    return parser.parse_args()


def apply_quick_mode(args):
    """
    Shrinks workload for quick validation on slower laptops.
    """
    if not args.quick:
        return args

    args.dataset_limit = min(args.dataset_limit, 800)
    args.analysis_limit = min(args.analysis_limit, 180)
    args.warmup_candles = min(args.warmup_candles, 180)
    args.horizon_candles = min(args.horizon_candles, 60)
    args.max_cutoffs = min(args.max_cutoffs, 20)
    args.max_validation_runs = min(args.max_validation_runs, 2)
    args.max_policies = min(args.max_policies, 3)
    return args


def save_results(
    output_dir: str,
    symbol: str,
    robustness_df: pd.DataFrame,
    validation_summary_df: pd.DataFrame,
    best_by_run_df: pd.DataFrame,
    workload_df: pd.DataFrame,
):
    """
    Saves robustness outputs as CSV files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    safe_symbol = symbol.lower().replace(" ", "_").replace("/", "_")
    robustness_df.to_csv(output_path / f"{safe_symbol}_management_robustness_summary.csv", index=False)
    validation_summary_df.to_csv(output_path / f"{safe_symbol}_management_validation_policy_runs.csv", index=False)
    best_by_run_df.to_csv(output_path / f"{safe_symbol}_management_best_by_run.csv", index=False)
    workload_df.to_csv(output_path / f"{safe_symbol}_management_workload.csv", index=False)


def print_report(
    args,
    robustness_df: pd.DataFrame,
    validation_summary_df: pd.DataFrame,
    best_by_run_df: pd.DataFrame,
    workload_df: pd.DataFrame,
):
    """
    Prints a readable robustness report.
    """
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 280)

    print("Trend Channel Management Robustness Validation Report v3")
    print("=" * 90)
    print(f"Symbol: {args.symbol}")
    print(f"Timeframes: {args.timeframes}")
    print(f"Base dataset limit: {args.dataset_limit}")
    print(f"Analysis limit per cutoff: {args.analysis_limit}")
    print(f"Base horizon candles: {args.horizon_candles}")
    print(f"Max cutoffs per profile: {args.max_cutoffs}")
    print(f"Max validation runs: {args.max_validation_runs}")
    print(f"Max policies: {args.max_policies}")
    print(f"Minimum RR: {args.minimum_rr}")
    print(f"Active only: {not args.include_historical}")
    print(f"Quick mode: {args.quick}")
    print("Cached design: strategy pipeline is built once per cutoff and reused across policies.")
    print()

    print("Workload / Efficiency Summary")
    print("-" * 90)
    if workload_df.empty:
        print("No workload summary was produced.")
    else:
        workload_columns = [
            "validation_name",
            "dataset_limit",
            "horizon_candles",
            "cutoff_step_candles",
            "original_cutoff_count",
            "used_cutoff_count",
            "valid_trade_plan_count",
            "policy_count",
            "estimated_full_pipeline_calls_saved",
            "elapsed_seconds_total",
        ]
        print(workload_df[[column for column in workload_columns if column in workload_df.columns]])
    print()

    print("Robustness Ranking Summary")
    print("-" * 90)
    if robustness_df.empty:
        print("No robustness summary was produced.")
    else:
        print(robustness_df)
    print()

    print("Best Policy by Validation Run")
    print("-" * 90)
    if best_by_run_df.empty:
        print("No validation-run winners were produced.")
    else:
        print(best_by_run_df)
    print()

    print("Validation Run Detail Preview")
    print("-" * 90)
    if validation_summary_df.empty:
        print("No validation details were produced.")
    else:
        detail_columns = [
            "validation_name",
            "validation_config_signature",
            "validation_dataset_limit",
            "validation_horizon_candles",
            "validation_cutoff_step_candles",
            "policy_name",
            "is_best_policy_for_run",
            "policy_score",
            "managed_trades",
            "positive_exit_rate",
            "average_managed_r",
            "median_managed_r",
            "loss_exit_count",
            "improved_vs_original_rate",
            "policy_quality_state",
        ]
        available_columns = [column for column in detail_columns if column in validation_summary_df.columns]
        print(validation_summary_df[available_columns].head(30))
    print()

    print("Interpretation Hint")
    print("-" * 90)
    if robustness_df.empty:
        print("No conclusion available.")
    else:
        top_state = robustness_df.iloc[0].get("robustness_state")
        top_policy = robustness_df.iloc[0].get("policy_name")
        print(f"Top policy: {top_policy}")
        print(f"Robustness state: {top_state}")
        print("If average R is still negative, do not lock the policy as final. Continue with setup filtering or entry refinement.")


def main():
    """
    Runs cached management robustness validation.
    """
    args = apply_quick_mode(parse_arguments())

    robustness_df, validation_summary_df, best_by_run_df, workload_df = validate_management_policy_robustness(
        symbol=args.symbol,
        timeframes=args.timeframes,
        analysis_limit=args.analysis_limit,
        dataset_limit=args.dataset_limit,
        start_time=args.start_time,
        end_time=args.end_time,
        warmup_candles=args.warmup_candles,
        horizon_candles=args.horizon_candles,
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
        exit_at_horizon_close=not args.no_horizon_close,
        max_cutoffs=args.max_cutoffs,
        max_validation_runs=args.max_validation_runs,
        max_policies=args.max_policies,
    )

    print_report(args, robustness_df, validation_summary_df, best_by_run_df, workload_df)

    if args.save_results:
        save_results(args.output_dir, args.symbol, robustness_df, validation_summary_df, best_by_run_df, workload_df)
        print()
        print(f"Saved results to: {args.output_dir}")


if __name__ == "__main__":
    main()
