from __future__ import annotations

import argparse
import time

from mdp.backtesting.trend_channel_rule_consistency_audit import (
    RuleConsistencyAuditConfig,
    build_rule_consistency_audit,
    save_rule_consistency_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Step Index trend-channel rule selection consistency before ML training."
    )
    parser.add_argument("--cache-path", required=True)
    parser.add_argument(
        "--rule-names",
        default="candidate_v1_base,avoid_tp3_only,require_best_available_rr_high,keep_high_rr_or_risk_high",
    )
    parser.add_argument("--tag", default="step_rule_consistency_audit")
    parser.add_argument("--output-dir", default="outputs/trend_channel_rule_audits")
    args = parser.parse_args()

    rule_names = tuple(rule.strip() for rule in args.rule_names.split(",") if rule.strip())
    config = RuleConsistencyAuditConfig(
        cache_path=args.cache_path,
        rule_names=rule_names,
        tag=args.tag,
        output_dir=args.output_dir,
    )

    started = time.perf_counter()
    outputs = build_rule_consistency_audit(config)
    elapsed = time.perf_counter() - started
    saved_paths = save_rule_consistency_audit(config, outputs)

    print("Step Index ML Rule Consistency Audit Report")
    print("=" * 90)
    print(f"Cache path: {args.cache_path}")
    print(f"Rule names: {','.join(rule_names)}")
    print(f"Elapsed seconds: {elapsed:.3f}")
    print("Cached design: loads saved cache once; no strategy pipeline rebuild.")
    print()

    print("Workload Summary")
    print("-" * 90)
    print(outputs["workload"].to_string(index=False))
    print()

    print("Rule Selection Summary")
    print("-" * 90)
    print(outputs["summary"].to_string(index=False))
    print()

    print("Pairwise Rule Comparison")
    print("-" * 90)
    print(outputs["pairwise_comparison"].to_string(index=False))
    print()

    print("Rule Violation Check")
    print("-" * 90)
    print(outputs["rule_violation_check"].to_string(index=False))
    print()

    print("Saved CSV Outputs")
    print("-" * 90)
    for name, path in saved_paths.items():
        print(f"{name}: {path}")
    print()
    print("Interpretation Hint")
    print("-" * 90)
    print(
        "Use this before ML training. If two rules produce identical final selections, treat them "
        "as equivalent after deduplication. If avoid rules still show violations, fix rule selection first."
    )


if __name__ == "__main__":
    main()
