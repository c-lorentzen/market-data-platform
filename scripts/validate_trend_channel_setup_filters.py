from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mdp.backtesting.trend_channel_setup_filter_validation import (
    SetupFilterValidationConfig,
    run_setup_filter_validation,
)


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for setup-filter validation.
    """
    parser = argparse.ArgumentParser(
        description="Validate trend-channel setup-filter hypotheses on cached managed outcomes."
    )
    parser.add_argument("--symbol", required=True, help="Market symbol, for example 'Step Index'.")
    parser.add_argument("--timeframes", default="H1,M1", help="Comma-separated MTF list, for example H1,M1.")
    parser.add_argument("--dataset-limit", type=int, default=1200)
    parser.add_argument("--analysis-limit", type=int, default=200)
    parser.add_argument("--warmup-candles", type=int, default=200)
    parser.add_argument("--horizon-candles", type=int, default=80)
    parser.add_argument("--cutoff-step-candles", type=int, default=20)
    parser.add_argument("--max-cutoffs", type=int, default=30)
    parser.add_argument("--minimum-rr", type=float, default=1.5)
    parser.add_argument("--min-trades-for-candidate", type=int, default=8)
    parser.add_argument(
        "--management-policy-name",
        default="targetless_scalp_manage_only",
        help="Management policy to use when measuring setup outcomes.",
    )
    parser.add_argument("--save-results", action="store_true", help="Save validation CSV files.")
    return parser.parse_args()


def print_dataframe(title: str, df: pd.DataFrame, max_rows: int = 20) -> None:
    """
    Prints DataFrames consistently without hiding important columns.
    """
    print("\n" + title)
    print("-" * 90)
    if df.empty:
        print("Empty DataFrame")
        return

    with pd.option_context("display.max_columns", None, "display.width", 260, "display.max_rows", max_rows):
        print(df.head(max_rows))


def main() -> None:
    """
    Runs setup-filter validation and optionally saves CSV outputs.
    """
    args = parse_args()
    config = SetupFilterValidationConfig(
        symbol=args.symbol,
        timeframes=args.timeframes,
        analysis_limit=args.analysis_limit,
        dataset_limit=args.dataset_limit,
        warmup_candles=args.warmup_candles,
        horizon_candles=args.horizon_candles,
        cutoff_step_candles=args.cutoff_step_candles,
        max_cutoffs=args.max_cutoffs,
        management_policy_name=args.management_policy_name,
        minimum_rr=args.minimum_rr,
        min_trades_for_candidate=args.min_trades_for_candidate,
    )

    validation_df, dataset_df, workload_df = run_setup_filter_validation(config)

    print("Trend Channel Setup Filter Validation Report")
    print("=" * 90)
    print(f"Symbol: {args.symbol}")
    print(f"Timeframes: {args.timeframes}")
    print(f"Dataset limit: {args.dataset_limit}")
    print(f"Analysis limit per cutoff: {args.analysis_limit}")
    print(f"Horizon candles: {args.horizon_candles}")
    print(f"Cutoff step candles: {args.cutoff_step_candles}")
    print(f"Max cutoffs: {args.max_cutoffs}")
    print(f"Management policy: {args.management_policy_name}")
    print(f"Minimum RR: {args.minimum_rr}")
    print(f"Min trades for candidate: {args.min_trades_for_candidate}")
    print("Cached design: setup dataset is built once, then all filter rules are tested on that same dataset.")

    print_dataframe("Workload / Efficiency Summary", workload_df, max_rows=10)
    print_dataframe("Setup Filter Validation Ranking", validation_df, max_rows=20)

    if not validation_df.empty:
        candidate_df = validation_df[validation_df["validation_state"] == "candidate_keep_filter_needs_larger_validation"]
        hypothesis_df = validation_df[validation_df["validation_state"] == "hypothesis_only_insufficient_sample"]
        rejected_df = validation_df[validation_df["validation_state"].str.contains("not_profitable|reject", na=False)]

        print_dataframe("Candidate Keep Rules", candidate_df, max_rows=10)
        print_dataframe("Hypothesis Rules With Insufficient Sample", hypothesis_df, max_rows=10)
        print_dataframe("Rejected / Not Profitable Rules", rejected_df, max_rows=10)

    preview_columns = [
        "analysis_cutoff",
        "scenario_state",
        "event_state",
        "entry_mode",
        "risk_ratio_bucket",
        "best_rr_bucket",
        "target_plan_state",
        "managed_exit_state",
        "managed_realized_r",
    ]
    existing_columns = [column for column in preview_columns if column in dataset_df.columns]
    print_dataframe("Cached Setup Dataset Preview", dataset_df[existing_columns] if existing_columns else dataset_df, max_rows=15)

    print("\nInterpretation Hint")
    print("-" * 90)
    print("Do not lock a setup filter unless it has enough trades, positive average R, positive median R, and stable validation.")
    print("If no candidate rule appears, continue with entry refinement instead of forcing filters.")

    if args.save_results:
        safe_symbol = args.symbol.lower().replace(" ", "_").replace("/", "_")
        output_dir = Path("outputs/trend_channel_setup_filters")
        output_dir.mkdir(parents=True, exist_ok=True)
        validation_df.to_csv(output_dir / f"{safe_symbol}_setup_filter_validation.csv", index=False)
        dataset_df.to_csv(output_dir / f"{safe_symbol}_setup_filter_validation_dataset.csv", index=False)
        workload_df.to_csv(output_dir / f"{safe_symbol}_setup_filter_validation_workload.csv", index=False)
        print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
