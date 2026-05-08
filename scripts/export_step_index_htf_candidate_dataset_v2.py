from __future__ import annotations

import argparse
import time

import pandas as pd

from mdp.backtesting.trend_channel_htf_candidate_export_v2 import (
    HTFCandidateExportConfig,
    export_htf_candidate_datasets,
)


def _parse_rule_names(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Step Index HTF candidate datasets with rule-consistent predicates."
    )
    parser.add_argument("--enriched-dataset-path", required=True)
    parser.add_argument("--rule-names", required=True)
    parser.add_argument("--fold-count", type=int, default=10)
    parser.add_argument("--tag", default="step_candidate_v1_2_htf_export_v2_1000")
    args = parser.parse_args()

    start = time.perf_counter()
    config = HTFCandidateExportConfig(
        enriched_dataset_path=args.enriched_dataset_path,
        rule_names=_parse_rule_names(args.rule_names),
        fold_count=args.fold_count,
        tag=args.tag,
    )
    output_paths = export_htf_candidate_datasets(config)
    elapsed = time.perf_counter() - start

    summary_df = pd.read_csv(output_paths["summary"])
    fold_df = pd.read_csv(output_paths["fold_summary"])
    daily_df = pd.read_csv(output_paths["daily_summary"])

    print("Step Index HTF Candidate Dataset Export v2 Report")
    print("=" * 90)
    print(f"Enriched dataset path: {args.enriched_dataset_path}")
    print(f"Rules: {args.rule_names}")
    print(f"Fold count: {args.fold_count}")
    print(f"Elapsed seconds: {elapsed:.3f}")
    print("Design: loads enriched HTF dataset once; uses shared explicit rule predicates.")
    print()

    print("Candidate Summary")
    print("-" * 90)
    print(summary_df.to_string(index=False))
    print()

    print("Fold Summary Preview")
    print("-" * 90)
    print(fold_df.head(40).to_string(index=False))
    print()

    print("Daily Summary Preview")
    print("-" * 90)
    print(daily_df.head(60).to_string(index=False))
    print()

    print("Saved CSV Outputs")
    print("-" * 90)
    for key, path in output_paths.items():
        print(f"{key}: {path}")
    print()

    print("Interpretation Hint")
    print("-" * 90)
    print(
        "Candidate counts should match the HTF validation rule counts. "
        "If they do not, audit the rule predicates before using exports for ML."
    )


if __name__ == "__main__":
    main()
