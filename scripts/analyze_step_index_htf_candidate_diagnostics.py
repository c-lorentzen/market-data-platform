"""CLI for final Step Index HTF candidate diagnostics."""

from __future__ import annotations

import argparse
import time

from mdp.backtesting.trend_channel_htf_candidate_diagnostics import (
    HTFCandidateDiagnosticsConfig,
    analyze_htf_candidate_dataset,
)


def _print_section(title: str) -> None:
    print("\n" + title)
    print("-" * 90)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a saved Step Index HTF candidate dataset."
    )
    parser.add_argument("--candidate-dataset-path", required=True)
    parser.add_argument("--fold-count", type=int, default=10)
    parser.add_argument("--tag", default="step_index_htf_candidate_diagnostics")
    args = parser.parse_args()

    started = time.perf_counter()
    config = HTFCandidateDiagnosticsConfig(
        candidate_dataset_path=args.candidate_dataset_path,
        fold_count=args.fold_count,
        tag=args.tag,
    )
    result = analyze_htf_candidate_dataset(config)
    elapsed = time.perf_counter() - started

    print("Step Index HTF Candidate Final Diagnostics Report")
    print("=" * 90)
    print(f"Candidate dataset path: {args.candidate_dataset_path}")
    print(f"Fold count: {args.fold_count}")
    print(f"Elapsed seconds: {elapsed:.3f}")
    print("Design: loads one exported candidate dataset; no strategy pipeline rebuild.")

    _print_section("Summary")
    print(result["summary"].to_string(index=False))

    _print_section("Fold Summary")
    print(result["fold_summary"].to_string(index=False))

    _print_section("Daily Summary Preview")
    print(result["daily_summary"].head(30).to_string(index=False))

    _print_section("Best / Worst Context Buckets Preview")
    bucket_df = result["bucket_summary"]
    if len(bucket_df) == 0:
        print("No context bucket columns found.")
    else:
        print("Top contexts:")
        print(bucket_df.head(15).to_string(index=False))
        print("\nWeak contexts:")
        print(bucket_df.tail(15).to_string(index=False))

    _print_section("Saved CSV Outputs")
    print(f"summary: {result['summary_path']}")
    print(f"daily_summary: {result['daily_summary_path']}")
    print(f"fold_summary: {result['fold_summary_path']}")
    print(f"bucket_summary: {result['bucket_summary_path']}")

    _print_section("Interpretation Hint")
    print(
        "Use this to decide whether Candidate v1.2 is clean enough for a strategy card "
        "and later ML labels. It is still research/paper-test status until validated on "
        "more history and forward-tested."
    )


if __name__ == "__main__":
    main()
