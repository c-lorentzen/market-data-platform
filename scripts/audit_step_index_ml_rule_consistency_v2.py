"""CLI for Step Index ML rule-consistency audit v2."""

from __future__ import annotations

import argparse
import time

from mdp.backtesting.trend_channel_rule_consistency_audit_v2 import (
    RuleAuditV2Config,
    run_rule_consistency_audit_v2,
)


def _print_section(title: str) -> None:
    """Print a clean report section header."""

    print()
    print(title)
    print("-" * 90)


def main() -> None:
    """Run the audit from command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Audit Step Index ML rule consistency with derived RR/risk buckets."
    )
    parser.add_argument("--cache-path", required=True, help="Path to the research cache CSV.")
    parser.add_argument(
        "--rule-names",
        default=(
            "candidate_v1_base,avoid_tp3_only,require_best_available_rr_high,"
            "require_risk_points_high,keep_high_rr_or_risk_high,keep_high_rr_or_bullish_pullback"
        ),
        help="Comma-separated audit rules.",
    )
    parser.add_argument("--tag", default="step_ml_rule_audit_v2", help="Output filename tag.")
    parser.add_argument(
        "--output-dir",
        default="outputs/trend_channel_rule_audits",
        help="Directory where CSV outputs are saved.",
    )

    args = parser.parse_args()

    rule_names = tuple(rule.strip() for rule in args.rule_names.split(",") if rule.strip())

    config = RuleAuditV2Config(
        cache_path=args.cache_path,
        rule_names=rule_names,
        tag=args.tag,
        output_dir=args.output_dir,
    )

    start = time.perf_counter()
    result = run_rule_consistency_audit_v2(config)
    elapsed = time.perf_counter() - start

    workload_df = result["workload_df"]
    summary_df = result["summary_df"]
    pairwise_df = result["pairwise_df"]
    violation_df = result["violation_df"]
    paths = result["paths"]

    print("Step Index ML Rule Consistency Audit v2 Report")
    print("=" * 90)
    print(f"Cache path: {args.cache_path}")
    print(f"Rule names: {','.join(rule_names)}")
    print(f"Elapsed seconds: {elapsed:.3f}")
    print("Cached design: loads saved cache once; derives missing RR/risk buckets; no strategy pipeline rebuild.")

    _print_section("Workload Summary")
    print(workload_df.to_string(index=False))

    _print_section("Rule Selection Summary")
    print(summary_df.to_string(index=False))

    _print_section("Pairwise Rule Comparison")
    print(pairwise_df.to_string(index=False))

    _print_section("Rule Violation Check")
    print(violation_df.to_string(index=False))

    _print_section("Saved CSV Outputs")
    for label, path in paths.items():
        print(f"{label}: {path}")

    _print_section("Interpretation Hint")
    print(
        "If high-RR rules now return rows, the older audit was missing derived buckets. "
        "Do not train ML until final rule selection is consistent and based only on pre-entry features."
    )


if __name__ == "__main__":
    main()
