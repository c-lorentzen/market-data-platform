"""Validate cached feature refinements across multiple research caches."""

from __future__ import annotations

import argparse
import pandas as pd

from mdp.backtesting.trend_channel_cache_feature_batch_validation import (
    FeatureBatchValidationConfig,
    validate_feature_refinements_across_caches,
)


def _parse_cache_paths(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_rule_names(value: str | None) -> list[str] | None:
    if value is None or value.strip() == "":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _print_df(title: str, df: pd.DataFrame, rows: int | None = None) -> None:
    print("\n" + title)
    print("-" * 90)
    if df.empty:
        print("Empty DataFrame")
        return
    print(df if rows is None else df.head(rows))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate feature-based trend-channel refinements across saved research caches."
    )
    parser.add_argument("--cache-paths", required=True, help="Comma-separated cache CSV paths.")
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--rule-names", default=None, help="Comma-separated feature rule names.")
    parser.add_argument("--min-total-trades", type=int, default=20)
    parser.add_argument("--min-trades-per-fold", type=int, default=3)

    args = parser.parse_args()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)

    config = FeatureBatchValidationConfig(
        cache_paths=_parse_cache_paths(args.cache_paths),
        fold_count=args.fold_count,
        rule_names=_parse_rule_names(args.rule_names),
        min_total_trades=args.min_total_trades,
        min_trades_per_fold=args.min_trades_per_fold,
    )

    result = validate_feature_refinements_across_caches(config)

    print("Trend Channel Cached Feature Batch Validation Report")
    print("=" * 90)
    print(f"Cache paths argument: {args.cache_paths}")
    print(f"Fold count: {args.fold_count}")
    print(f"Rule names: {args.rule_names or 'default_feature_refinement_rules'}")
    print("Cached design: validates feature rules from saved caches without rebuilding the strategy pipeline.")

    _print_df("Workload / Efficiency Summary", result.workload)
    _print_df("Aggregate Feature Rule Summary Across Caches", result.aggregate_summary)
    _print_df("Per-Cache Feature Rule Summary", result.per_cache_summary)
    _print_df("Fold Detail Preview", result.fold_detail, rows=40)
    _print_df("Selected Trade Preview", result.selected_preview, rows=40)

    print("\nInterpretation Hint")
    print("-" * 90)
    print(
        "Use this for cross-market feature-rule validation. Do not lock a feature rule unless it stays positive "
        "across caches with enough trades and without severe negative folds."
    )


if __name__ == "__main__":
    main()
