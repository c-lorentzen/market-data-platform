"""Command-line runner for Step Index online learning filter diagnostics."""

from __future__ import annotations

import argparse
import time

from mdp.backtesting.trend_channel_online_learning_filter import (
    OnlineLearningFilterConfig,
    run_online_learning_filter,
)


def _print_section(title: str) -> None:
    """Print a clear report section title."""

    print()
    print(title)
    print("-" * 90)


def main() -> None:
    """Run the diagnostic from an ML-ready dataset CSV."""

    parser = argparse.ArgumentParser(
        description="Analyze a no-lookahead online learning filter from a Step Index ML-ready dataset."
    )
    parser.add_argument("--ml-dataset-path", required=True, help="Path to the exported ML-ready dataset CSV.")
    parser.add_argument("--tag", default="step_online_learning_filter", help="Tag used for output CSV names.")
    parser.add_argument("--min-history-trades", type=int, default=20)
    parser.add_argument("--min-context-trades", type=int, default=4)
    parser.add_argument("--prior-strength", type=float, default=6.0)
    parser.add_argument("--min-estimated-avg-r", type=float, default=0.03)
    parser.add_argument("--min-estimated-positive-rate", type=float, default=0.58)
    parser.add_argument("--max-estimated-full-loss-rate", type=float, default=0.25)
    parser.add_argument("--max-trades-per-day", type=int, default=None)
    args = parser.parse_args()

    started_at = time.perf_counter()

    config = OnlineLearningFilterConfig(
        ml_dataset_path=args.ml_dataset_path,
        tag=args.tag,
        min_history_trades=args.min_history_trades,
        min_context_trades=args.min_context_trades,
        prior_strength=args.prior_strength,
        min_estimated_avg_r=args.min_estimated_avg_r,
        min_estimated_positive_rate=args.min_estimated_positive_rate,
        max_estimated_full_loss_rate=args.max_estimated_full_loss_rate,
        max_trades_per_day=args.max_trades_per_day,
    )

    result = run_online_learning_filter(config)
    elapsed_seconds = time.perf_counter() - started_at
    result.workload_df["elapsed_seconds_total"] = round(elapsed_seconds, 3)

    print("Step Index Online Learning Filter Diagnostic Report")
    print("=" * 90)
    print(f"ML dataset path: {args.ml_dataset_path}")
    print(f"Tag: {args.tag}")
    print(f"Elapsed seconds: {elapsed_seconds:.3f}")
    print("Design: loads exported ML dataset once; no strategy pipeline rebuild.")
    print("Learning rule: no-lookahead proxy using only prior rows before each decision.")

    _print_section("Workload Summary")
    print(result.workload_df.to_string(index=False))

    _print_section("Baseline vs Online Filter Summary")
    print(result.summary_df.to_string(index=False))

    _print_section("Daily Comparison")
    print(result.daily_df.to_string(index=False))

    _print_section("Decision Preview")
    preview_columns = [
        "analysis_cutoff",
        "trade_date",
        "hour_utc",
        "decision_state",
        "kept_by_online_filter",
        "estimated_avg_r",
        "estimated_positive_rate",
        "estimated_full_loss_rate",
        "skip_reasons",
        "label_managed_r",
        "scenario_state",
        "event_state",
        "entry_mode",
        "target_plan_state",
        "online_best_rr_bucket",
        "online_risk_ratio_bucket",
    ]
    preview_columns = [column for column in preview_columns if column in result.decision_df.columns]
    print(result.decision_df[preview_columns].head(60).to_string(index=False))

    _print_section("Context Edge Preview - Diagnostic Only")
    if result.context_edge_df.empty:
        print("No context edges available.")
    else:
        print(result.context_edge_df.head(30).to_string(index=False))

    _print_section("Saved CSV Outputs")
    for name, path in result.output_paths.items():
        print(f"{name}: {path}")

    print()
    print("Interpretation Hint")
    print("-" * 90)
    print(
        "If the online filter improves average R and reduces full-loss rate while keeping enough trades, "
        "it becomes a bridge toward ML. If it skips profitable trades or turns negative, do not use it."
    )


if __name__ == "__main__":
    main()
