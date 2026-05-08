from __future__ import annotations

import argparse
import time

from mdp.backtesting.trend_channel_htf_candidate_export import (
    CandidateExportConfig,
    export_htf_candidate_datasets,
)


def _parse_rule_names(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Step Index HTF-filtered candidate datasets from an enriched ML dataset."
    )
    parser.add_argument("--enriched-dataset-path", required=True)
    parser.add_argument(
        "--rule-names",
        default="baseline_d1_h4_m15_m5_not_against,premium_baseline_no_htf_opposition,require_baseline_current_close,candidate_v1_base",
        help="Comma-separated candidate rules to export.",
    )
    parser.add_argument("--fold-count", type=int, default=10)
    parser.add_argument("--tag", default="step_htf_candidate_export")
    args = parser.parse_args()

    started = time.perf_counter()
    config = CandidateExportConfig(
        enriched_dataset_path=args.enriched_dataset_path,
        rule_names=_parse_rule_names(args.rule_names),
        fold_count=args.fold_count,
        tag=args.tag,
    )
    results = export_htf_candidate_datasets(config)
    elapsed = time.perf_counter() - started

    print("Step Index HTF Candidate Dataset Export Report")
    print("=" * 90)
    print(f"Enriched dataset path: {args.enriched_dataset_path}")
    print(f"Rules: {', '.join(config.rule_names)}")
    print(f"Fold count: {args.fold_count}")
    print(f"Elapsed seconds: {elapsed:.3f}")
    print("Design: loads enriched HTF dataset once; exports selected pre-entry candidate profiles.")

    print("\nCandidate Summary")
    print("-" * 90)
    print(results["summary"].to_string(index=False))

    print("\nFold Summary Preview")
    print("-" * 90)
    fold_summary = results["fold_summary"]
    if fold_summary.empty:
        print("No fold rows exported.")
    else:
        print(fold_summary.head(60).to_string(index=False))

    print("\nDaily Summary Preview")
    print("-" * 90)
    daily_summary = results["daily_summary"]
    if daily_summary.empty:
        print("No daily rows exported.")
    else:
        print(daily_summary.head(60).to_string(index=False))

    print("\nSaved CSV Outputs")
    print("-" * 90)
    base = f"outputs\\trend_channel_htf_candidates\\{args.tag}"
    print(f"summary: {base}_summary.csv")
    print(f"daily_summary: {base}_daily_summary.csv")
    print(f"fold_summary: {base}_fold_summary.csv")
    print(f"all_selected_datasets: {base}_all_selected_datasets.csv")
    for rule_name in config.rule_names:
        print(f"{rule_name}: {base}_{rule_name}_selected_dataset.csv")

    print("\nInterpretation Hint")
    print("-" * 90)
    print(
        "Use the exported candidate datasets for final distance/frequency diagnostics and future ML labels. "
        "Do not treat these as live-ready until validated on more history and forward/paper tested."
    )


if __name__ == "__main__":
    main()
