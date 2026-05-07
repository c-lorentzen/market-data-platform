from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mdp.backtesting.trend_channel_candidate_failure_diagnostics import (
    CandidateFailureDiagnosticsConfig,
    run_candidate_failure_diagnostics,
)


def parse_args() -> argparse.Namespace:
    """
    Parses CLI arguments for candidate failure diagnostics.
    """
    parser = argparse.ArgumentParser(
        description="Diagnose losing folds and failure drivers for a simple trend-channel strategy candidate."
    )
    parser.add_argument("--symbol", required=True, help="Market symbol, for example: Step Index")
    parser.add_argument("--timeframes", default="H1,M1", help="Comma-separated timeframes, for example: H1,M1")
    parser.add_argument("--dataset-limit", type=int, default=3200, help="Entry-timeframe candles used to build historical cutoffs")
    parser.add_argument("--analysis-limit", type=int, default=220, help="Candles used by each cutoff-aware analysis")
    parser.add_argument("--warmup-candles", type=int, default=220, help="Historical candles required before the first cutoff")
    parser.add_argument("--horizon-candles", type=int, default=100, help="Future candles used to simulate managed outcome")
    parser.add_argument("--cutoff-step-candles", type=int, default=20, help="Step size between historical cutoffs")
    parser.add_argument("--max-cutoffs", type=int, default=70, help="Maximum cutoffs for laptop-friendly validation")
    parser.add_argument("--fold-count", type=int, default=3, help="Number of chronological folds")
    parser.add_argument("--management-policy-name", default="targetless_scalp_manage_only", help="Management policy used by the selected candidate")
    parser.add_argument("--minimum-rr", type=float, default=1.50, help="Minimum RR required for target availability")
    parser.add_argument("--candidate-name", default="hybrid_retest_now_continuation_confirm2", help="Candidate strategy to diagnose")
    parser.add_argument("--quick", action="store_true", help="Use smaller laptop-friendly settings")
    parser.add_argument("--inactive-too", action="store_true", help="Include inactive channel contexts")
    parser.add_argument("--save-results", action="store_true", help="Save CSV files under outputs/trend_channel_candidate_failure_diagnostics")
    return parser.parse_args()


def print_dataframe(title: str, df: pd.DataFrame, max_rows: int = 20) -> None:
    """
    Prints a DataFrame in a readable terminal format.
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
        110,
    ):
        print(df.head(max_rows))


def main() -> None:
    """
    Runs the candidate failure diagnostic report.
    """
    args = parse_args()
    config = CandidateFailureDiagnosticsConfig(
        symbol=args.symbol,
        timeframes=args.timeframes,
        analysis_limit=args.analysis_limit,
        dataset_limit=args.dataset_limit,
        warmup_candles=args.warmup_candles,
        horizon_candles=args.horizon_candles,
        cutoff_step_candles=args.cutoff_step_candles,
        max_cutoffs=args.max_cutoffs,
        fold_count=args.fold_count,
        management_policy_name=args.management_policy_name,
        minimum_rr=args.minimum_rr,
        candidate_name=args.candidate_name,
        active_only=not args.inactive_too,
        quick=args.quick,
    )

    workload_df, fold_table_df, fold_summary_df, feature_summary_df, losing_preview_df = run_candidate_failure_diagnostics(config)

    print("Trend Channel Candidate Failure Diagnostics Report")
    print("=" * 90)
    print(f"Symbol: {args.symbol}")
    print(f"Timeframes: {args.timeframes}")
    print(f"Candidate: {args.candidate_name}")
    print(f"Dataset limit: {args.dataset_limit}")
    print(f"Analysis limit per cutoff: {args.analysis_limit}")
    print(f"Horizon candles: {args.horizon_candles}")
    print(f"Cutoff step candles: {args.cutoff_step_candles}")
    print(f"Max cutoffs: {args.max_cutoffs}")
    print(f"Fold count: {args.fold_count}")
    print(f"Management policy: {args.management_policy_name}")
    print(f"Quick mode: {args.quick}")
    print("Cached design: build entry dataset once, then diagnose selected candidate failures.")

    print_dataframe("Workload / Efficiency Summary", workload_df, max_rows=10)
    print_dataframe("Chronological Fold Table", fold_table_df, max_rows=10)
    print_dataframe("Fold Performance Summary", fold_summary_df, max_rows=10)

    if not feature_summary_df.empty:
        failure_df = feature_summary_df[feature_summary_df["diagnostic_note"] == "possible_failure_driver"]
        keep_df = feature_summary_df[feature_summary_df["diagnostic_note"] == "possible_keep_context"]
        neutral_df = feature_summary_df[feature_summary_df["diagnostic_note"] == "neutral_or_small_sample"]
        print_dataframe("Possible Failure Drivers", failure_df, max_rows=20)
        print_dataframe("Possible Keep Contexts", keep_df, max_rows=15)
        print_dataframe("Neutral / Small-Sample Contexts", neutral_df, max_rows=15)
    else:
        print_dataframe("Feature Diagnostics", feature_summary_df, max_rows=20)

    print_dataframe("Losing Trade Preview", losing_preview_df, max_rows=20)

    print("\nInterpretation Hint")
    print("-" * 90)
    print("Use this report to refine scenario/event definitions, not to force a profitable result.")
    print("If losses concentrate in one event/scenario/entry context, test a simple avoid-or-confirm rule next.")

    if args.save_results:
        safe_symbol = args.symbol.lower().replace(" ", "_").replace("/", "_")
        output_dir = Path("outputs/trend_channel_candidate_failure_diagnostics")
        output_dir.mkdir(parents=True, exist_ok=True)
        workload_df.to_csv(output_dir / f"{safe_symbol}_candidate_failure_workload.csv", index=False)
        fold_table_df.to_csv(output_dir / f"{safe_symbol}_candidate_failure_folds.csv", index=False)
        fold_summary_df.to_csv(output_dir / f"{safe_symbol}_candidate_failure_fold_summary.csv", index=False)
        feature_summary_df.to_csv(output_dir / f"{safe_symbol}_candidate_failure_features.csv", index=False)
        losing_preview_df.to_csv(output_dir / f"{safe_symbol}_candidate_failure_losers.csv", index=False)
        print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
