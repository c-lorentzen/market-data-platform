"""Export Step Index HTF candidate datasets with rule-consistent predicates."""

from __future__ import annotations

import argparse
import time

from mdp.backtesting.trend_channel_htf_candidate_export_v3 import (
    HtfCandidateExportConfig,
    export_htf_candidate_datasets_v3,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Step Index HTF candidate datasets v3.")
    parser.add_argument("--enriched-dataset-path", required=True)
    parser.add_argument("--rule-names", required=True, help="Comma-separated rule names.")
    parser.add_argument("--fold-count", type=int, default=10)
    parser.add_argument("--tag", default="step_candidate_v1_2_htf_export_v3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rule_names = [rule.strip() for rule in args.rule_names.split(",") if rule.strip()]

    config = HtfCandidateExportConfig(
        enriched_dataset_path=args.enriched_dataset_path,
        rule_names=rule_names,
        fold_count=args.fold_count,
        tag=args.tag,
    )

    started = time.perf_counter()
    result = export_htf_candidate_datasets_v3(config)
    elapsed = time.perf_counter() - started

    print("Step Index HTF Candidate Dataset Export v3 Report")
    print("=" * 90)
    print(f"Enriched dataset path: {args.enriched_dataset_path}")
    print(f"Rules: {', '.join(rule_names)}")
    print(f"Fold count: {args.fold_count}")
    print(f"Elapsed seconds: {elapsed:.3f}")
    print("Design: loads enriched HTF dataset once; baseline HTF rules require baseline_current_close.")

    print("\nCandidate Summary")
    print("-" * 90)
    print(result["summary_df"].to_string(index=False))

    print("\nFold Summary Preview")
    print("-" * 90)
    print(result["fold_summary_df"].head(40).to_string(index=False))

    print("\nSaved CSV Outputs")
    print("-" * 90)
    for rule_name, path in result["selected_paths"].items():
        print(f"{rule_name}: {path}")
    print(f"summary: {result['summary_path']}")
    print(f"daily_summary: {result['daily_summary_path']}")
    print(f"fold_summary: {result['fold_summary_path']}")
    print(f"all_selected_datasets: {result['all_selected_path']}")

    print("\nExpected Count Sanity Check")
    print("-" * 90)
    print("baseline_d1_h4_m15_m5_not_against should match the prior HTF validation count (~46).")
    print("premium_baseline_no_htf_opposition should match the prior HTF validation count (~37).")
    print("require_baseline_current_close should remain 74, and candidate_v1_base should remain 142.")


if __name__ == "__main__":
    main()
