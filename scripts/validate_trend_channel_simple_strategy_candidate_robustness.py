from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mdp.backtesting.trend_channel_simple_strategy_candidate_robustness import (
    SimpleStrategyCandidateRobustnessConfig,
    run_simple_strategy_candidate_robustness,
)


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for simple strategy candidate robustness validation.
    """
    parser = argparse.ArgumentParser(
        description="Validate simple trend-channel strategy candidates across several samples."
    )
    parser.add_argument("--symbol", required=True, help="Market symbol, for example: Step Index")
    parser.add_argument("--timeframes", default="H1,M1", help="Comma-separated timeframes, for example: H1,M1")
    parser.add_argument("--dataset-limit", type=int, default=1600, help="Base number of entry-timeframe candles to build cutoffs from")
    parser.add_argument("--analysis-limit", type=int, default=220, help="Candles used by each cutoff-aware strategy analysis")
    parser.add_argument("--warmup-candles", type=int, default=220, help="Minimum historical candles before the first cutoff")
    parser.add_argument("--horizon-candles", type=int, default=80, help="Future candles used to label each candidate")
    parser.add_argument("--cutoff-step-candles", type=int, default=20, help="Step size between historical cutoffs")
    parser.add_argument("--max-cutoffs", type=int, default=35, help="Maximum recent cutoffs to test per profile")
    parser.add_argument("--max-validation-runs", type=int, default=3, help="Maximum validation profiles to run")
    parser.add_argument("--management-policy-name", default="targetless_scalp_manage_only", help="Management policy to simulate")
    parser.add_argument("--minimum-rr", type=float, default=1.50, help="Minimum RR required for target availability")
    parser.add_argument("--min-trades-for-candidate", type=int, default=8, help="Minimum trades before a candidate can be considered in one profile")
    parser.add_argument("--min-total-trades-for-robust-candidate", type=int, default=20, help="Minimum total trades before a candidate can be considered robust")
    parser.add_argument("--quick", action="store_true", help="Use a smaller laptop-friendly validation set")
    parser.add_argument("--inactive-too", action="store_true", help="Include inactive channel contexts when building the strategy pipeline")
    parser.add_argument("--save-results", action="store_true", help="Save output CSV files under outputs/trend_channel_simple_strategy_candidate_robustness")
    return parser.parse_args()


def print_dataframe(title: str, df: pd.DataFrame, max_rows: int = 20) -> None:
    """
    Prints a DataFrame with readable column output.
    """
    print(f"\n{title}")
    print("-" * 90)
    if df.empty:
        print("Empty DataFrame")
        return

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        280,
        "display.max_colwidth",
        95,
    ):
        print(df.head(max_rows))


def main() -> None:
    """
    Runs robustness validation and prints the report.
    """
    args = parse_args()
    config = SimpleStrategyCandidateRobustnessConfig(
        symbol=args.symbol,
        timeframes=args.timeframes,
        analysis_limit=args.analysis_limit,
        dataset_limit=args.dataset_limit,
        warmup_candles=args.warmup_candles,
        horizon_candles=args.horizon_candles,
        cutoff_step_candles=args.cutoff_step_candles,
        max_cutoffs=args.max_cutoffs,
        max_validation_runs=args.max_validation_runs,
        management_policy_name=args.management_policy_name,
        minimum_rr=args.minimum_rr,
        min_trades_for_candidate=args.min_trades_for_candidate,
        min_total_trades_for_robust_candidate=args.min_total_trades_for_robust_candidate,
        active_only=not args.inactive_too,
        quick=args.quick,
    )

    robustness_df, combined_summary_df, workload_df, preview_df = run_simple_strategy_candidate_robustness(config)

    print("Trend Channel Simple Strategy Candidate Robustness Report")
    print("=" * 90)
    print(f"Symbol: {args.symbol}")
    print(f"Timeframes: {args.timeframes}")
    print(f"Base dataset limit: {args.dataset_limit}")
    print(f"Analysis limit per cutoff: {args.analysis_limit}")
    print(f"Base horizon candles: {args.horizon_candles}")
    print(f"Cutoff step candles: {args.cutoff_step_candles}")
    print(f"Max cutoffs per profile: {args.max_cutoffs}")
    print(f"Max validation runs: {args.max_validation_runs}")
    print(f"Management policy: {args.management_policy_name}")
    print(f"Minimum RR: {args.minimum_rr}")
    print(f"Quick mode: {args.quick}")
    print("Cached design: each profile builds entry variants once, then candidate rules are reused.")

    print_dataframe("Workload / Efficiency Summary", workload_df, max_rows=10)
    print_dataframe("Robust Candidate Ranking", robustness_df, max_rows=20)

    if not combined_summary_df.empty:
        best_by_run_df = (
            combined_summary_df.sort_values(["validation_name", "candidate_score"], ascending=[True, False])
            .groupby("validation_name", as_index=False)
            .head(1)
        )
        ready_df = robustness_df[robustness_df["robustness_state"] == "robust_candidate_needs_walk_forward_validation"]
        promising_df = robustness_df[robustness_df["robustness_state"].str.contains("promising", na=False)]
        rejected_df = robustness_df[robustness_df["robustness_state"].str.contains("not_robust|insufficient", na=False)]

        print_dataframe("Best Candidate by Validation Run", best_by_run_df, max_rows=10)
        print_dataframe("Robust Candidates Ready for Walk-Forward", ready_df, max_rows=10)
        print_dataframe("Promising but Not Profitable / Not Robust Yet", promising_df, max_rows=10)
        print_dataframe("Rejected or Insufficient Robust Candidates", rejected_df, max_rows=20)

    print_dataframe("Best Candidate Preview Rows", preview_df, max_rows=20)

    print("\nInterpretation Hint")
    print("-" * 90)
    print("Do not lock a candidate unless it stays positive across multiple distinct validation profiles.")
    print("If the best candidate is only barely positive or sample size is small, validate harder before coding it as final strategy logic.")

    if args.save_results:
        safe_symbol = args.symbol.lower().replace(" ", "_").replace("/", "_")
        output_dir = Path("outputs/trend_channel_simple_strategy_candidate_robustness")
        output_dir.mkdir(parents=True, exist_ok=True)
        robustness_df.to_csv(output_dir / f"{safe_symbol}_simple_strategy_candidate_robustness.csv", index=False)
        combined_summary_df.to_csv(output_dir / f"{safe_symbol}_simple_strategy_candidate_validation_runs.csv", index=False)
        workload_df.to_csv(output_dir / f"{safe_symbol}_simple_strategy_candidate_robustness_workload.csv", index=False)
        preview_df.to_csv(output_dir / f"{safe_symbol}_simple_strategy_candidate_robustness_preview.csv", index=False)
        print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
