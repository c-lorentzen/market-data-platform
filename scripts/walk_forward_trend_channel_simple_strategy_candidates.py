from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mdp.backtesting.trend_channel_simple_strategy_walk_forward import (
    SimpleStrategyWalkForwardConfig,
    run_simple_strategy_walk_forward_validation,
)


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for chronological simple strategy validation.
    """
    parser = argparse.ArgumentParser(
        description="Run chronological walk-forward validation for simple trend-channel strategy candidates."
    )
    parser.add_argument("--symbol", required=True, help="Market symbol, for example: Step Index")
    parser.add_argument("--timeframes", default="H1,M1", help="Comma-separated timeframes, for example: H1,M1")
    parser.add_argument("--dataset-limit", type=int, default=3200, help="Number of entry-timeframe candles used to build cutoff history")
    parser.add_argument("--analysis-limit", type=int, default=220, help="Candles used by each cutoff-aware analysis")
    parser.add_argument("--warmup-candles", type=int, default=220, help="Historical candles required before the first cutoff")
    parser.add_argument("--horizon-candles", type=int, default=100, help="Future candles used to label each candidate")
    parser.add_argument("--cutoff-step-candles", type=int, default=20, help="Step size between historical cutoffs")
    parser.add_argument("--max-cutoffs", type=int, default=70, help="Maximum cutoffs to build before splitting into folds")
    parser.add_argument("--fold-count", type=int, default=3, help="Number of chronological folds")
    parser.add_argument("--management-policy-name", default="targetless_scalp_manage_only", help="Management policy to simulate")
    parser.add_argument("--minimum-rr", type=float, default=1.50, help="Minimum RR required for target availability")
    parser.add_argument("--min-trades-per-fold", type=int, default=3, help="Minimum selected trades required per fold")
    parser.add_argument("--min-total-trades-for-candidate", type=int, default=12, help="Minimum total trades required before a candidate can be considered")
    parser.add_argument("--candidate-name", default=None, help="Optional single candidate to validate")
    parser.add_argument("--quick", action="store_true", help="Use smaller laptop-friendly settings")
    parser.add_argument("--inactive-too", action="store_true", help="Include inactive channel contexts")
    parser.add_argument("--save-results", action="store_true", help="Save CSV files under outputs/trend_channel_simple_strategy_walk_forward")
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
        95,
    ):
        print(df.head(max_rows))


def main() -> None:
    """
    Runs the walk-forward validation report.
    """
    args = parse_args()
    config = SimpleStrategyWalkForwardConfig(
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
        min_trades_per_fold=args.min_trades_per_fold,
        min_total_trades_for_candidate=args.min_total_trades_for_candidate,
        fold_count=args.fold_count,
        active_only=not args.inactive_too,
        candidate_name=args.candidate_name,
        quick=args.quick,
    )

    summary_df, fold_results_df, fold_table_df, workload_df, preview_df = run_simple_strategy_walk_forward_validation(config)

    print("Trend Channel Simple Strategy Walk-Forward Validation Report")
    print("=" * 90)
    print(f"Symbol: {args.symbol}")
    print(f"Timeframes: {args.timeframes}")
    print(f"Dataset limit: {args.dataset_limit}")
    print(f"Analysis limit per cutoff: {args.analysis_limit}")
    print(f"Horizon candles: {args.horizon_candles}")
    print(f"Cutoff step candles: {args.cutoff_step_candles}")
    print(f"Max cutoffs: {args.max_cutoffs}")
    print(f"Fold count: {args.fold_count}")
    print(f"Management policy: {args.management_policy_name}")
    print(f"Minimum RR: {args.minimum_rr}")
    print(f"Quick mode: {args.quick}")
    print("Cached design: entry variants are built once, then split into chronological folds.")

    print_dataframe("Workload / Efficiency Summary", workload_df, max_rows=10)
    print_dataframe("Chronological Fold Table", fold_table_df, max_rows=10)
    print_dataframe("Walk-Forward Candidate Ranking", summary_df, max_rows=20)

    if not summary_df.empty:
        promising_df = summary_df[summary_df["validation_state"].str.contains("promising", na=False)]
        rejected_df = summary_df[summary_df["validation_state"].str.contains("do_not_lock|unstable", na=False)]
        print_dataframe("Promising Walk-Forward Candidates", promising_df, max_rows=10)
        print_dataframe("Rejected / Not Ready Candidates", rejected_df, max_rows=20)

    print_dataframe("Fold-Level Candidate Results", fold_results_df, max_rows=30)
    print_dataframe("Best Candidate Trade Preview", preview_df, max_rows=20)

    print("\nInterpretation Hint")
    print("-" * 90)
    print("Do not lock a candidate unless it stays positive across chronological folds with enough trades.")
    print("If the best candidate is barely positive or fails a fold, keep refining the scenario/event definition.")

    if args.save_results:
        safe_symbol = args.symbol.lower().replace(" ", "_").replace("/", "_")
        output_dir = Path("outputs/trend_channel_simple_strategy_walk_forward")
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_dir / f"{safe_symbol}_simple_strategy_walk_forward_summary.csv", index=False)
        fold_results_df.to_csv(output_dir / f"{safe_symbol}_simple_strategy_walk_forward_folds.csv", index=False)
        fold_table_df.to_csv(output_dir / f"{safe_symbol}_simple_strategy_walk_forward_fold_table.csv", index=False)
        workload_df.to_csv(output_dir / f"{safe_symbol}_simple_strategy_walk_forward_workload.csv", index=False)
        preview_df.to_csv(output_dir / f"{safe_symbol}_simple_strategy_walk_forward_preview.csv", index=False)
        print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
