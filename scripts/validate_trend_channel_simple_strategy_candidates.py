from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mdp.backtesting.trend_channel_simple_strategy_candidate_validation import (
    SimpleStrategyCandidateValidationConfig,
    run_simple_strategy_candidate_validation,
)


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for simple strategy candidate validation.
    """
    parser = argparse.ArgumentParser(
        description="Validate simple complete trend-channel strategy candidates."
    )
    parser.add_argument("--symbol", required=True, help="Market symbol, for example: Step Index")
    parser.add_argument("--timeframes", default="H1,M1", help="Comma-separated timeframes, for example: H1,M1")
    parser.add_argument("--dataset-limit", type=int, default=1200, help="Number of entry-timeframe candles to build cutoffs from")
    parser.add_argument("--analysis-limit", type=int, default=220, help="Candles used by each cutoff-aware strategy analysis")
    parser.add_argument("--warmup-candles", type=int, default=220, help="Minimum historical candles before the first cutoff")
    parser.add_argument("--horizon-candles", type=int, default=80, help="Future candles used to label each candidate")
    parser.add_argument("--cutoff-step-candles", type=int, default=20, help="Step size between historical cutoffs")
    parser.add_argument("--max-cutoffs", type=int, default=40, help="Maximum recent cutoffs to test")
    parser.add_argument("--management-policy-name", default="targetless_scalp_manage_only", help="Management policy to simulate")
    parser.add_argument("--minimum-rr", type=float, default=1.50, help="Minimum RR required for target availability")
    parser.add_argument("--min-trades-for-candidate", type=int, default=8, help="Minimum trades before a candidate can be considered")
    parser.add_argument("--inactive-too", action="store_true", help="Include inactive channel contexts when building the strategy pipeline")
    parser.add_argument("--save-results", action="store_true", help="Save output CSV files under outputs/trend_channel_simple_strategy_candidates")
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
        260,
        "display.max_colwidth",
        95,
    ):
        print(df.head(max_rows))


def main() -> None:
    """
    Runs simple strategy candidate validation and prints the report.
    """
    args = parse_args()
    config = SimpleStrategyCandidateValidationConfig(
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
        active_only=not args.inactive_too,
    )

    dataset_df, summary_df, workload_df, preview_df = run_simple_strategy_candidate_validation(config)

    print("Trend Channel Simple Strategy Candidate Validation Report")
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
    print("Cached design: entry variants are built once, then complete candidate rules are tested.")

    print_dataframe("Workload / Efficiency Summary", workload_df, max_rows=10)
    print_dataframe("Simple Strategy Candidate Ranking", summary_df, max_rows=20)

    if not summary_df.empty:
        candidate_df = summary_df[summary_df["validation_state"] == "candidate_simple_strategy_needs_larger_validation"]
        hypothesis_df = summary_df[summary_df["validation_state"] == "hypothesis_only_insufficient_sample"]
        rejected_df = summary_df[summary_df["validation_state"].str.contains("not_profitable|reject", na=False)]
        print_dataframe("Candidate Simple Strategies", candidate_df, max_rows=10)
        print_dataframe("Hypothesis Simple Strategies With Insufficient Sample", hypothesis_df, max_rows=10)
        print_dataframe("Rejected / Not Profitable Simple Strategies", rejected_df, max_rows=10)

    print_dataframe("Best Candidate Trade Preview", preview_df, max_rows=20)

    print("\nInterpretation Hint")
    print("-" * 90)
    print("Do not lock a simple strategy candidate unless it has enough trades, positive average R, positive median R, and stable validation.")
    print("If every candidate stays negative, improve the scenario/event definition instead of adding random complexity.")

    if args.save_results:
        safe_symbol = args.symbol.lower().replace(" ", "_").replace("/", "_")
        output_dir = Path("outputs/trend_channel_simple_strategy_candidates")
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_dir / f"{safe_symbol}_simple_strategy_candidate_summary.csv", index=False)
        dataset_df.to_csv(output_dir / f"{safe_symbol}_simple_strategy_candidate_dataset.csv", index=False)
        workload_df.to_csv(output_dir / f"{safe_symbol}_simple_strategy_candidate_workload.csv", index=False)
        preview_df.to_csv(output_dir / f"{safe_symbol}_simple_strategy_candidate_preview.csv", index=False)
        print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
