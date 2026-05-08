from __future__ import annotations

import argparse
import pandas as pd

from mdp.backtesting.trend_channel_research_cache import (
    EntryResearchCacheConfig,
    build_entry_research_cache,
    save_entry_research_cache,
)


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for building a reusable entry research cache.
    """
    parser = argparse.ArgumentParser(description="Build reusable trend-channel entry research cache.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframes", required=True)
    parser.add_argument("--dataset-limit", type=int, default=3200)
    parser.add_argument("--analysis-limit", type=int, default=220)
    parser.add_argument("--warmup-candles", type=int, default=220)
    parser.add_argument("--horizon-candles", type=int, default=100)
    parser.add_argument("--cutoff-step-candles", type=int, default=20)
    parser.add_argument("--max-cutoffs", type=int, default=70)
    parser.add_argument("--management-policy-name", default="targetless_scalp_manage_only")
    parser.add_argument("--minimum-rr", type=float, default=1.50)
    parser.add_argument("--min-trades-for-candidate", type=int, default=8)
    parser.add_argument("--output-dir", default="outputs/trend_channel_research_cache")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--inactive-too", action="store_true")
    return parser.parse_args()


def main() -> None:
    """
    Builds and saves the cache dataset for fast repeated validation.
    """
    args = parse_args()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)

    config = EntryResearchCacheConfig(
        symbol=args.symbol,
        timeframes=args.timeframes,
        dataset_limit=args.dataset_limit,
        analysis_limit=args.analysis_limit,
        warmup_candles=args.warmup_candles,
        horizon_candles=args.horizon_candles,
        cutoff_step_candles=args.cutoff_step_candles,
        max_cutoffs=args.max_cutoffs,
        management_policy_name=args.management_policy_name,
        minimum_rr=args.minimum_rr,
        min_trades_for_candidate=args.min_trades_for_candidate,
        active_only=not args.inactive_too,
        quick=args.quick,
    )

    dataset_df, workload_df, metadata = build_entry_research_cache(config)
    paths = save_entry_research_cache(
        dataset_df=dataset_df,
        workload_df=workload_df,
        metadata=metadata,
        output_dir=args.output_dir,
        tag=args.tag,
    )

    print("Trend Channel Entry Research Cache Build Report")
    print("=" * 90)
    print(f"Symbol: {args.symbol}")
    print(f"Timeframes: {args.timeframes}")
    print(f"Quick mode: {args.quick}")
    print(f"Dataset rows: {metadata['dataset_rows']}")
    print(f"Unique analysis cutoffs: {metadata['unique_analysis_cutoffs']}")
    print(f"Unique baseline trade plans: {metadata['unique_baseline_trade_plans']}")
    print(f"Elapsed seconds: {metadata['elapsed_seconds_total_cache_build']}")
    print("\nWorkload Summary")
    print("-" * 90)
    print(workload_df)
    print("\nSaved Cache Files")
    print("-" * 90)
    print(f"Dataset CSV: {paths.dataset_path}")
    print(f"Workload CSV: {paths.workload_path}")
    print(f"Metadata JSON: {paths.metadata_path}")
    print("\nNext command idea")
    print("-" * 90)
    print(
        "python scripts\\validate_trend_channel_refinements_from_cache.py "
        f"--cache-path \"{paths.dataset_path}\" --fold-count 3"
    )


if __name__ == "__main__":
    main()
