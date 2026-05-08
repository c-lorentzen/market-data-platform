from __future__ import annotations

import argparse
import pandas as pd

from mdp.backtesting.trend_channel_candidate_refinement_validation import (
    CandidateRefinementValidationConfig,
    run_candidate_refinement_validation,
)


def _print_section(title: str) -> None:
    """
    Prints a consistent terminal section header.
    """
    print("\n" + title)
    print("-" * 90)


def _print_df(df: pd.DataFrame, max_rows: int = 20) -> None:
    """
    Prints DataFrames without hiding columns.
    """
    if df.empty:
        print("Empty DataFrame")
        return

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        240,
        "display.max_colwidth",
        90,
        "display.max_rows",
        max_rows,
    ):
        print(df.head(max_rows))


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for targeted refinement validation.
    """
    parser = argparse.ArgumentParser(description="Validate targeted trend-channel candidate refinements.")
    parser.add_argument("--symbol", required=True, help="Market symbol, for example 'Step Index'.")
    parser.add_argument("--timeframes", default="H1,M1", help="Comma-separated structure/entry timeframes.")
    parser.add_argument("--dataset-limit", type=int, default=3200, help="Candles loaded for historical replay.")
    parser.add_argument("--analysis-limit", type=int, default=220, help="Candles used per cutoff analysis.")
    parser.add_argument("--warmup-candles", type=int, default=220, help="Warmup candles before the first cutoff.")
    parser.add_argument("--horizon-candles", type=int, default=100, help="Future candles used to simulate outcomes.")
    parser.add_argument("--cutoff-step-candles", type=int, default=20, help="Spacing between historical cutoff points.")
    parser.add_argument("--max-cutoffs", type=int, default=70, help="Maximum cutoff points to analyze.")
    parser.add_argument("--fold-count", type=int, default=3, help="Chronological fold count.")
    parser.add_argument("--management-policy-name", default="targetless_scalp_manage_only", help="Management policy used for outcomes.")
    parser.add_argument("--minimum-rr", type=float, default=1.5, help="Minimum RR required by upstream trade plans.")
    parser.add_argument("--min-trades-for-candidate", type=int, default=8, help="Minimum trades before a rule can be considered a candidate.")
    parser.add_argument("--include-inactive", action="store_true", help="Include inactive channel contexts.")
    parser.add_argument("--quick", action="store_true", help="Run a smaller smoke test.")
    return parser.parse_args()


def main() -> None:
    """
    Runs the refinement validation report.
    """
    args = parse_args()
    config = CandidateRefinementValidationConfig(
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
        min_trades_for_candidate=args.min_trades_for_candidate,
        active_only=not args.include_inactive,
        quick=args.quick,
    )

    dataset_df, summary_df, robust_df, fold_df, workload_df, preview_df = run_candidate_refinement_validation(config)

    print("Trend Channel Candidate Refinement Validation Report")
    print("=" * 90)
    print(f"Symbol: {config.symbol}")
    print(f"Timeframes: {config.timeframes}")
    print(f"Dataset limit: {config.dataset_limit}")
    print(f"Analysis limit per cutoff: {config.analysis_limit}")
    print(f"Horizon candles: {config.horizon_candles}")
    print(f"Cutoff step candles: {config.cutoff_step_candles}")
    print(f"Max cutoffs: {config.max_cutoffs}")
    print(f"Fold count: {config.fold_count}")
    print(f"Management policy: {config.management_policy_name}")
    print(f"Minimum RR: {config.minimum_rr}")
    print(f"Quick mode: {config.quick}")
    print("Cached design: entry dataset is built once, then refinement rules and folds reuse it.")

    _print_section("Workload / Efficiency Summary")
    _print_df(workload_df)

    _print_section("Refinement Rule Ranking")
    _print_df(summary_df)

    _print_section("Walk-Forward Robustness Ranking")
    _print_df(robust_df)

    ready_df = robust_df[robust_df.get("robustness_state", "") == "strong_refinement_candidate_needs_out_of_sample"] if not robust_df.empty else pd.DataFrame()
    weak_df = robust_df[robust_df.get("robustness_state", "") == "weak_positive_refinement_validate_harder"] if not robust_df.empty else pd.DataFrame()
    rejected_df = robust_df[
        ~robust_df.get("robustness_state", pd.Series(dtype="object")).isin(
            ["strong_refinement_candidate_needs_out_of_sample", "weak_positive_refinement_validate_harder"]
        )
    ] if not robust_df.empty else pd.DataFrame()

    _print_section("Strong Refinement Candidates")
    _print_df(ready_df)

    _print_section("Weak Positive Refinements / Validate Harder")
    _print_df(weak_df)

    _print_section("Rejected or Insufficient Refinements")
    _print_df(rejected_df)

    _print_section("Fold-Level Refinement Detail Preview")
    _print_df(fold_df, max_rows=30)

    _print_section("Best Refinement Trade Preview")
    _print_df(preview_df, max_rows=30)

    _print_section("Interpretation Hint")
    print("Only keep a refinement if it improves average R, median R, and fold stability without collapsing trade count.")
    print("If all refinements remain weak, improve scenario/event definitions instead of stacking more filters.")


if __name__ == "__main__":
    main()
