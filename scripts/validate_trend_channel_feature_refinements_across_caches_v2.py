from __future__ import annotations

import argparse
import time

import pandas as pd

from mdp.backtesting.trend_channel_cache_feature_batch_validation_v2 import (
    DEFAULT_RULE_NAMES,
    FeatureBatchValidationConfig,
    validate_feature_rules_across_caches_v2,
)


def _print_section(title: str) -> None:
    print("\n" + title)
    print("-" * 90)


def _print_df(df: pd.DataFrame, max_rows: int | None = None) -> None:
    if df is None or df.empty:
        print("Empty DataFrame")
        return
    with pd.option_context("display.max_columns", None, "display.width", 240, "display.max_colwidth", 70):
        print(df.head(max_rows) if max_rows else df)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate trend-channel feature refinement rules across saved research caches, including post-target guards."
    )
    parser.add_argument("--cache-paths", required=True, help="Comma-separated cache CSV paths.")
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--rule-names", default=",".join(DEFAULT_RULE_NAMES), help="Comma-separated feature rule names.")
    parser.add_argument("--min-total-trades", type=int, default=20)
    parser.add_argument("--min-trades-per-fold", type=int, default=3)
    parser.add_argument("--preview-rows", type=int, default=40)
    args = parser.parse_args()

    cache_paths = [p.strip() for p in args.cache_paths.split(",") if p.strip()]
    rule_names = [r.strip() for r in args.rule_names.split(",") if r.strip()]

    start = time.perf_counter()
    config = FeatureBatchValidationConfig(
        cache_paths=cache_paths,
        fold_count=args.fold_count,
        rule_names=rule_names,
        min_total_trades=args.min_total_trades,
        min_trades_per_fold=args.min_trades_per_fold,
        preview_rows=args.preview_rows,
    )
    results = validate_feature_rules_across_caches_v2(config)
    elapsed = time.perf_counter() - start
    results["workload"].loc[0, "elapsed_seconds_total"] = round(elapsed, 3)

    print("Trend Channel Cached Feature Batch Validation v2 Report")
    print("=" * 90)
    print(f"Cache paths argument: {args.cache_paths}")
    print(f"Fold count: {args.fold_count}")
    print(f"Rule names: {','.join(rule_names)}")
    print("Cached design: validates feature rules from saved caches without rebuilding the strategy pipeline.")
    print("Post-target guard rules reject entries where target availability disappears after entry refinement.")

    _print_section("Workload / Efficiency Summary")
    _print_df(results["workload"])

    _print_section("Aggregate Feature Rule Summary Across Caches")
    _print_df(results["aggregate"])

    _print_section("Per-Cache Feature Rule Summary")
    _print_df(results["per_cache"])

    _print_section("Fold Detail Preview")
    _print_df(results["fold_detail"], max_rows=60)

    _print_section("Selected Trade Preview")
    _print_df(results["selected_preview"], max_rows=args.preview_rows)

    print("\nInterpretation Hint")
    print("-" * 90)
    print("Post-target guard rules are designed to avoid trades whose target plan becomes invalid after entry refinement.")
    print("Do not lock a rule unless it is positive across caches, has enough trades, and avoids severe negative folds.")


if __name__ == "__main__":
    main()
