"""CLI for feature-edge diagnostics from a saved trend-channel cache."""

from __future__ import annotations

import argparse
import time

import pandas as pd

from mdp.backtesting.trend_channel_cache_feature_edge_diagnostics import (
    FeatureEdgeConfig,
    diagnose_cache_feature_edges,
)


def _print_section(title: str) -> None:
    print("\n" + title)
    print("-" * 90)


def _print_df(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        print("Empty DataFrame")
        return
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        240,
        "display.max_colwidth",
        80,
    ):
        print(df)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose simple feature edges from a cached trend-channel research dataset."
    )
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--rule-name", default="retest_requires_tp2_tp3")
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--min-group-trades", type=int, default=3)
    parser.add_argument("--weak-r-threshold", type=float, default=0.0)
    parser.add_argument("--strong-r-threshold", type=float, default=0.10)
    args = parser.parse_args()

    start_time = time.perf_counter()
    config = FeatureEdgeConfig(
        cache_path=args.cache_path,
        rule_name=args.rule_name,
        fold_count=args.fold_count,
        min_group_trades=args.min_group_trades,
        weak_r_threshold=args.weak_r_threshold,
        strong_r_threshold=args.strong_r_threshold,
    )
    result = diagnose_cache_feature_edges(config)
    elapsed = time.perf_counter() - start_time
    result["workload"]["elapsed_seconds_total"] = round(elapsed, 3)

    print("Trend Channel Cache Feature Edge Diagnostics Report")
    print("=" * 90)
    print(f"Cache path: {args.cache_path}")
    print(f"Rule name: {args.rule_name}")
    print(f"Fold count: {args.fold_count}")
    print(f"Min group trades: {args.min_group_trades}")
    print("Cached design: loads saved cache once and groups selected trades by pre-entry features.")

    _print_section("Workload / Efficiency Summary")
    _print_df(result["workload"])

    _print_section("Fold Performance Summary")
    _print_df(result["fold_summary"])

    _print_section("Candidate Avoid / Confirm Contexts")
    _print_df(result["candidate_avoid_contexts"].head(25))

    _print_section("Candidate Keep Contexts")
    _print_df(result["candidate_keep_contexts"].head(25))

    _print_section("Neutral / Small-Sample Contexts")
    _print_df(result["neutral_or_small_sample_contexts"].head(25))

    _print_section("Selected Trade Preview")
    preview_cols = [
        col
        for col in [
            "fold_id",
            "fold_position",
            "analysis_cutoff",
            "scenario_state",
            "event_state",
            "entry_mode",
            "entry_variant_name",
            "risk_points",
            "available_targets",
            "target_plan_state",
            "managed_exit_state",
            "managed_realized_r",
        ]
        if col in result["selected_trade_preview"].columns
    ]
    _print_df(result["selected_trade_preview"][preview_cols])

    print("\nInterpretation Hint")
    print("-" * 90)
    print(
        "Only test a new refinement if the avoid context is based on pre-entry information, "
        "has enough trades, and repeats outside a single chronological fold."
    )


if __name__ == "__main__":
    main()
