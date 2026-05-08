from __future__ import annotations

import argparse

from mdp.backtesting.trend_channel_ml_dataset_export import (
    MLDatasetExportConfig,
    build_ml_ready_dataset,
    save_ml_ready_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a no-lookahead ML-ready dataset from a Step Index trend-channel cache."
    )
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--rule-name", default="candidate_v1_base")
    parser.add_argument("--tag", default="step_candidate_ml_dataset")
    parser.add_argument("--output-dir", default="outputs/trend_channel_ml_datasets")
    parser.add_argument("--max-trades-per-day", type=int, default=None)
    parser.add_argument("--positive-r-threshold", type=float, default=0.0)
    parser.add_argument("--full-loss-r-threshold", type=float, default=-0.95)
    args = parser.parse_args()

    config = MLDatasetExportConfig(
        cache_path=args.cache_path,
        rule_name=args.rule_name,
        output_dir=args.output_dir,
        tag=args.tag,
        max_trades_per_day=args.max_trades_per_day,
        positive_r_threshold=args.positive_r_threshold,
        full_loss_r_threshold=args.full_loss_r_threshold,
    )

    outputs = build_ml_ready_dataset(config)
    saved_paths = save_ml_ready_dataset(config)

    print("Step Index ML-Ready Dataset Export Report")
    print("=" * 90)
    print(f"Cache path: {args.cache_path}")
    print(f"Rule name: {args.rule_name}")
    print("Cached design: loads saved cache once; no strategy pipeline rebuild.")
    print("Dataset rule: one selected trade per analysis_cutoff; labels are outcome columns, not features.")
    print()

    print("Summary")
    print("-" * 90)
    print(outputs["summary"].to_string(index=False))
    print()

    print("Daily Summary")
    print("-" * 90)
    print(outputs["daily_summary"].to_string(index=False))
    print()

    print("ML Dataset Preview")
    print("-" * 90)
    print(outputs["ml_dataset"].head(20).to_string(index=False))
    print()

    print("Saved CSV Outputs")
    print("-" * 90)
    for name, path in saved_paths.items():
        print(f"{name}: {path}")
    print()
    print("Interpretation Hint")
    print("-" * 90)
    print(
        "Use this as a clean labeled dataset for future ML. Train only on feature columns and "
        "split chronologically to avoid future leakage."
    )


if __name__ == "__main__":
    main()
