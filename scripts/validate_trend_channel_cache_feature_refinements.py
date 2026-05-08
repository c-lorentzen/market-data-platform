"""Validate cached feature-refinement rules for trend-channel strategy research."""

from __future__ import annotations

import argparse
import pandas as pd

from mdp.backtesting.trend_channel_cache_feature_refinement_validation import (
    validate_feature_refinements_from_cache,
)


def _parse_rule_names(value: str | None) -> list[str] | None:
    if value is None or value.strip() == "":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _print_df(title: str, df: pd.DataFrame) -> None:
    print("\n" + title)
    print("-" * 90)
    if df.empty:
        print("Empty DataFrame")
    else:
        print(df)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate feature-based refinements from a saved trend-channel entry research cache."
    )
    parser.add_argument("--cache-path", required=True, help="Path to saved entry research cache CSV.")
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--rule-names", default=None, help="Comma-separated list of rule names to test.")
    parser.add_argument("--min-total-trades", type=int, default=20)
    parser.add_argument("--min-trades-per-fold", type=int, default=3)

    args = parser.parse_args()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    result = validate_feature_refinements_from_cache(
        cache_path=args.cache_path,
        fold_count=args.fold_count,
        rule_names=_parse_rule_names(args.rule_names),
        min_total_trades=args.min_total_trades,
        min_trades_per_fold=args.min_trades_per_fold,
    )

    print("Trend Channel Cached Feature Refinement Validation Report")
    print("=" * 90)
    print(f"Cache path: {args.cache_path}")
    print(f"Fold count: {args.fold_count}")
    print(f"Rule names: {args.rule_names or 'default_feature_refinement_rules'}")
    print("Cached design: loads saved cache once; no strategy pipeline rebuild.")

    _print_df("Workload / Efficiency Summary", result.workload)
    _print_df("Feature Refinement Ranking", result.ranking)
    _print_df("Fold Detail Preview", result.fold_detail.head(30))
    _print_df("Selected Trade Preview", result.selected_preview.head(40))

    print("\nInterpretation Hint")
    print("-" * 90)
    print(
        "Use this as diagnostic validation only. Do not lock numeric bucket refinements until they survive "
        "larger caches and cross-market validation without collapsing trade count."
    )


if __name__ == "__main__":
    main()
