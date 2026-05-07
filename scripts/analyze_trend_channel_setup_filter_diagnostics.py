from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mdp.backtesting.trend_channel_setup_filter_diagnostics import (
    SetupFilterDiagnosticConfig,
    run_setup_filter_diagnostics,
)


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for setup-filter diagnostics.
    """
    parser = argparse.ArgumentParser(
        description="Analyze which trend-channel setups should be kept, watched, or avoided."
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
    parser.add_argument("--min-group-trades", type=int, default=5)
    parser.add_argument(
        "--management-policy-name",
        default="targetless_scalp_manage_only",
        help="Management policy to use when measuring setup outcomes.",
    )
    parser.add_argument("--save-results", action="store_true", help="Save diagnostic CSV files.")
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

    with pd.option_context("display.max_columns", None, "display.width", 240, "display.max_rows", max_rows):
        print(df.head(max_rows))


def main() -> None:
    """
    Runs setup-filter diagnostics and optionally saves CSV outputs.
    """
    args = parse_args()
    config = SetupFilterDiagnosticConfig(
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
        min_group_trades=args.min_group_trades,
    )

    dataset_df, summary_df, group_df, workload_df = run_setup_filter_diagnostics(config)

    print("Trend Channel Setup Filter Diagnostic Report")
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

    print_dataframe("Workload / Efficiency Summary", workload_df, max_rows=10)
    print_dataframe("Setup Filter Dataset Summary", summary_df, max_rows=10)

    if not group_df.empty:
        avoid_df = group_df[group_df["filter_recommendation"] == "candidate_avoid_filter"]
        keep_df = group_df[group_df["filter_recommendation"] == "candidate_keep_filter"]
        neutral_df = group_df[group_df["filter_recommendation"] == "neutral_watch_filter"]
        insufficient_df = group_df[group_df["filter_recommendation"] == "insufficient_sample"]

        print_dataframe("Candidate Keep Filters", keep_df.sort_values("average_managed_r", ascending=False), max_rows=15)
        print_dataframe("Candidate Avoid Filters", avoid_df.sort_values("average_managed_r"), max_rows=15)
        print_dataframe("Neutral Watch Filters", neutral_df.sort_values("trades", ascending=False), max_rows=15)
        print_dataframe("Insufficient-Sample Groups Preview", insufficient_df.sort_values("trades", ascending=False), max_rows=10)

    detail_columns = [
        "analysis_cutoff",
        "plan_direction",
        "scenario_state",
        "event_state",
        "entry_mode",
        "risk_ratio_bucket",
        "best_rr_bucket",
        "event_quality_bucket",
        "managed_exit_state",
        "managed_realized_r",
        "partial_exit_hit",
        "breakeven_or_profit_lock_activated",
    ]
    existing_columns = [column for column in detail_columns if column in dataset_df.columns]
    print_dataframe("Diagnostic Trade Rows Preview", dataset_df[existing_columns] if existing_columns else dataset_df, max_rows=20)

    print("\nInterpretation Hint")
    print("-" * 90)
    print("If average R is still negative, do not call the strategy profitable.")
    print("Use candidate keep/avoid groups only as hypotheses until larger validation confirms them.")

    if args.save_results:
        safe_symbol = args.symbol.lower().replace(" ", "_").replace("/", "_")
        output_dir = Path("outputs/trend_channel_setup_filters")
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_df.to_csv(output_dir / f"{safe_symbol}_setup_filter_dataset.csv", index=False)
        summary_df.to_csv(output_dir / f"{safe_symbol}_setup_filter_summary.csv", index=False)
        group_df.to_csv(output_dir / f"{safe_symbol}_setup_filter_groups.csv", index=False)
        workload_df.to_csv(output_dir / f"{safe_symbol}_setup_filter_workload.csv", index=False)
        print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
