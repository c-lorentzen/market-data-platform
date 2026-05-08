"""CLI for cache-based SL/TP distance diagnostics."""

from __future__ import annotations

import argparse

from mdp.backtesting.trend_channel_distance_profile import (
    DistanceProfileConfig,
    build_distance_profile,
    save_distance_profile_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SL/TP distances from a trend-channel research cache.")
    parser.add_argument("--cache-path", required=True, help="Path to the saved entry research cache CSV.")
    parser.add_argument("--rule-name", default="retest_requires_tp2_tp3", help="Candidate rule to profile.")
    parser.add_argument("--output-dir", default="outputs/trend_channel_distance_profiles")
    parser.add_argument("--tag", default="step_distance_profile")
    parser.add_argument("--preview-rows", type=int, default=20)
    args = parser.parse_args()

    config = DistanceProfileConfig(
        cache_path=args.cache_path,
        rule_name=args.rule_name,
        output_dir=args.output_dir,
        tag=args.tag,
    )

    report = build_distance_profile(config)
    saved_paths = save_distance_profile_outputs(report, config)

    print("Trend Channel Distance Profile Report")
    print("=" * 90)
    print(f"Cache path: {args.cache_path}")
    print(f"Rule name: {args.rule_name}")
    print("Cached design: loads saved cache once; no strategy pipeline rebuild.")

    print("\nWorkload Summary")
    print("-" * 90)
    print(report["workload"])

    print("\nSL / TP Distance Summary")
    print("-" * 90)
    print(report["distance_summary"])

    print("\nGrouped Distance Summary")
    print("-" * 90)
    print(report["group_summary"].head(60))

    preview_columns = [
        "analysis_cutoff",
        "entry_mode",
        "entry_variant_name",
        "entry_price",
        "stop_loss",
        "sl_distance_points",
        "best_available_rr",
        "best_target_distance_points",
        "available_targets",
        "target_plan_state",
        "managed_exit_state",
        "managed_realized_r",
    ]
    existing_preview_columns = [col for col in preview_columns if col in report["selected_trades"].columns]

    print("\nSelected Trade Distance Preview")
    print("-" * 90)
    print(report["selected_trades"][existing_preview_columns].head(args.preview_rows))

    print("\nSaved CSV Outputs")
    print("-" * 90)
    for name, path in saved_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
