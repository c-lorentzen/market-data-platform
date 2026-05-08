"""CLI for diagnosing fold instability from cached trend-channel research files."""

from __future__ import annotations

import argparse
import time

import pandas as pd

from mdp.backtesting.trend_channel_cache_fold_instability_diagnostics import (
    diagnose_fold_instability,
)


def _print_section(title: str) -> None:
    print("\n" + title)
    print("-" * 90)


def _print_df(df: pd.DataFrame, max_rows: int = 20) -> None:
    if df is None or df.empty:
        print("Empty DataFrame")
        return
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 240,
        "display.max_colwidth", 80,
    ):
        print(df.head(max_rows))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose weak chronological folds from a saved trend-channel research cache."
    )
    parser.add_argument("--cache-path", required=True, help="Path to the saved entry research cache CSV.")
    parser.add_argument(
        "--rule-name",
        default="retest_requires_tp2_tp3",
        help="Candidate/refinement rule to diagnose.",
    )
    parser.add_argument("--fold-count", type=int, default=5, help="Number of chronological folds.")
    parser.add_argument("--max-preview-rows", type=int, default=20, help="Maximum preview rows to print.")
    args = parser.parse_args()

    start = time.perf_counter()
    result = diagnose_fold_instability(
        cache_path=args.cache_path,
        rule_name=args.rule_name,
        fold_count=args.fold_count,
    )
    elapsed = time.perf_counter() - start
    result.workload_summary["elapsed_seconds_total"] = round(elapsed, 3)

    print("Trend Channel Cache Fold Instability Diagnostics Report")
    print("=" * 90)
    print(f"Cache path: {args.cache_path}")
    print(f"Rule name: {args.rule_name}")
    print(f"Fold count: {args.fold_count}")
    print("Cached design: loads saved research cache once; no strategy pipeline rebuild.")

    _print_section("Workload / Efficiency Summary")
    _print_df(result.workload_summary)

    _print_section("Fold Performance Summary")
    _print_df(result.fold_summary, max_rows=args.max_preview_rows)

    _print_section("Candidate Avoid Contexts")
    _print_df(result.candidate_avoid_contexts, max_rows=args.max_preview_rows)

    _print_section("Candidate Keep Contexts")
    _print_df(result.candidate_keep_contexts, max_rows=args.max_preview_rows)

    _print_section("Weak Fold Trade Preview")
    preview_columns = [
        "fold_id", "fold_position", "analysis_cutoff", "entry_variant_name", "scenario_state",
        "event_state", "entry_mode", "entry_price", "risk_points", "available_targets",
        "target_plan_state", "managed_exit_state", "managed_realized_r",
    ]
    existing_columns = [column for column in preview_columns if column in result.weak_fold_trades.columns]
    _print_df(result.weak_fold_trades[existing_columns] if existing_columns else result.weak_fold_trades, max_rows=args.max_preview_rows)

    _print_section("Selected Trade Preview")
    existing_columns = [column for column in preview_columns if column in result.selected_trades.columns]
    _print_df(result.selected_trades[existing_columns] if existing_columns else result.selected_trades, max_rows=args.max_preview_rows)

    print("\nInterpretation Hint")
    print("-" * 90)
    print(
        "Use this report to identify why a promising rule has negative folds. "
        "Do not create another filter unless losses cluster around a simple, repeated context."
    )


if __name__ == "__main__":
    main()
