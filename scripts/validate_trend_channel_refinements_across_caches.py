"""CLI for validating trend-channel refinements across cached research files."""

from __future__ import annotations

import argparse
import time

import pandas as pd

from mdp.backtesting.trend_channel_cache_batch_validation import (
    BatchValidationConfig,
    validate_refinements_across_caches,
)


def _print_section(title: str) -> None:
    """Print a consistent report section divider."""

    print("\n" + title)
    print("-" * 90)


def _print_df(df: pd.DataFrame, max_rows: int = 30) -> None:
    """Print DataFrames without hiding important columns."""

    if df.empty:
        print("Empty DataFrame")
        return

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        240,
        "display.max_rows",
        max_rows,
    ):
        print(df.head(max_rows))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Validate trend-channel refinement rules across saved cache CSV files."
    )

    parser.add_argument(
        "--cache-paths",
        required=True,
        help=(
            "Comma-separated cache paths or glob pattern. "
            'Example: "outputs/trend_channel_research_cache/*entry_research_cache*.csv"'
        ),
    )
    parser.add_argument("--fold-count", type=int, default=4)
    parser.add_argument("--min-total-trades", type=int, default=20)
    parser.add_argument("--min-trades-per-fold", type=int, default=3)
    parser.add_argument(
        "--rule-names",
        default="",
        help="Optional comma-separated rule names. Leave blank for default rules.",
    )

    return parser.parse_args()


def main() -> None:
    """Run cached batch validation and print the report."""

    args = parse_args()
    started_at = time.perf_counter()

    # Convert comma-separated CLI text into clean Python lists.
    cache_paths = [path.strip() for path in args.cache_paths.split(",") if path.strip()]
    rule_names = [rule.strip() for rule in args.rule_names.split(",") if rule.strip()] or None

    config = BatchValidationConfig(
        cache_paths=cache_paths,
        fold_count=args.fold_count,
        min_total_trades=args.min_total_trades,
        min_trades_per_fold=args.min_trades_per_fold,
        rule_names=rule_names,
    )

    results = validate_refinements_across_caches(config)
    elapsed_seconds = time.perf_counter() - started_at

    print("Trend Channel Cached Batch Refinement Validation Report")
    print("=" * 90)
    print(f"Cache paths argument: {args.cache_paths}")
    print(f"Fold count: {args.fold_count}")
    print(f"Elapsed seconds: {elapsed_seconds:.3f}")
    print("Cached design: validates saved cache files without rebuilding the strategy pipeline.")

    workload_df = results["workload"].copy()
    workload_df["elapsed_seconds_total"] = round(elapsed_seconds, 3)

    _print_section("Workload / Efficiency Summary")
    _print_df(workload_df)

    _print_section("Aggregate Rule Summary Across Caches")
    _print_df(results["aggregate_summary"])

    _print_section("Per-Cache Rule Summary")
    _print_df(results["per_cache_summary"])

    _print_section("Fold Detail Preview")
    _print_df(results["fold_detail"], max_rows=40)

    print("\nInterpretation Hint")
    print("-" * 90)
    print("Use this after building one or more research caches.")
    print("Do not lock a rule unless it stays positive with enough total trades and stable folds/caches.")


if __name__ == "__main__":
    main()
