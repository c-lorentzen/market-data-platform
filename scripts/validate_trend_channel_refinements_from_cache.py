from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from mdp.backtesting.trend_channel_research_cache import (
    build_best_refinement_preview_from_cache,
    load_entry_research_cache,
    validate_refinements_from_entry_cache,
)


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for fast refinement validation from a saved cache.
    """
    parser = argparse.ArgumentParser(description="Validate trend-channel candidate refinements from a saved cache.")
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--fold-count", type=int, default=3)
    parser.add_argument("--min-trades-for-candidate", type=int, default=8)
    parser.add_argument("--rule-names", default=None, help="Optional comma-separated refinement rule names.")
    parser.add_argument("--save-results", action="store_true")
    parser.add_argument("--output-dir", default="outputs/trend_channel_research_cache")
    return parser.parse_args()


def _save_results(
    cache_path: str,
    summary_df: pd.DataFrame,
    robust_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    preview_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """
    Saves validation outputs next to the cache analysis folder.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stem = Path(cache_path).stem

    summary_df.to_csv(output_path / f"{stem}_refinement_summary.csv", index=False)
    robust_df.to_csv(output_path / f"{stem}_refinement_robustness.csv", index=False)
    fold_df.to_csv(output_path / f"{stem}_refinement_folds.csv", index=False)
    preview_df.to_csv(output_path / f"{stem}_best_refinement_preview.csv", index=False)


def main() -> None:
    """
    Loads the cache and runs cheap refinement validation without rebuilding strategy history.
    """
    args = parse_args()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 260)

    dataset_df = load_entry_research_cache(args.cache_path)
    summary_df, robust_df, fold_df = validate_refinements_from_entry_cache(
        dataset_df=dataset_df,
        fold_count=args.fold_count,
        min_trades_for_candidate=args.min_trades_for_candidate,
        rule_names=args.rule_names,
    )
    preview_df = build_best_refinement_preview_from_cache(dataset_df, robust_df)

    print("Trend Channel Cached Refinement Validation Report")
    print("=" * 90)
    print(f"Cache path: {args.cache_path}")
    print(f"Cache rows: {len(dataset_df)}")
    print(f"Unique cutoffs: {dataset_df['analysis_cutoff'].nunique() if 'analysis_cutoff' in dataset_df.columns else 0}")
    print(f"Fold count: {args.fold_count}")
    print(f"Rule names: {args.rule_names or 'default_refinement_rules'}")
    print("\nRefinement Summary")
    print("-" * 90)
    print(summary_df)
    print("\nWalk-Forward Robustness Summary")
    print("-" * 90)
    print(robust_df)
    print("\nFold Detail Preview")
    print("-" * 90)
    print(fold_df.head(30))
    print("\nBest Refinement Preview")
    print("-" * 90)
    print(preview_df.head(20))

    if args.save_results:
        _save_results(args.cache_path, summary_df, robust_df, fold_df, preview_df, args.output_dir)
        print("\nSaved result CSV files under:")
        print(args.output_dir)

    print("\nInterpretation Hint")
    print("-" * 90)
    print("This validation should run fast because it reuses the cached entry dataset.")
    print("Only lock a refinement after it has enough trades and stable positive fold performance.")


if __name__ == "__main__":
    main()
