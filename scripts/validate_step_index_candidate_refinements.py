"""Command-line runner for Step Index Candidate v1 refinement validation."""

from __future__ import annotations

import argparse

from mdp.backtesting.trend_channel_step_candidate_refinement_validation import (
    StepCandidateRefinementConfig,
    save_step_candidate_refinement_outputs,
    validate_step_candidate_refinements,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Step Index Trend Channel Candidate v1 refinements from cache."
    )
    parser.add_argument("--cache-path", required=True, help="Path to the Step Index research cache CSV.")
    parser.add_argument("--fold-count", type=int, default=7, help="Number of chronological folds.")
    parser.add_argument("--min-total-trades", type=int, default=20, help="Minimum trades needed before a rule can be considered lockable.")
    parser.add_argument("--min-trades-per-fold", type=int, default=3, help="Minimum trades per fold for fold stability.")
    parser.add_argument(
        "--rule-names",
        default="candidate_v1_base,avoid_entry_price_low,avoid_risk_points_mid,avoid_tp3_only,require_best_available_rr_high,require_risk_points_high,avoid_entry_price_low_and_risk_mid,keep_high_rr_or_risk_high,keep_high_rr_or_bullish_pullback",
        help="Comma-separated refinement rule names.",
    )
    parser.add_argument("--tag", default="step_candidate_refinement", help="Output file tag.")
    return parser.parse_args()


def print_section(title: str) -> None:
    print("\n" + title)
    print("-" * 90)


def main() -> None:
    args = parse_args()
    rule_names = tuple(name.strip() for name in args.rule_names.split(",") if name.strip())

    config = StepCandidateRefinementConfig(
        cache_path=args.cache_path,
        fold_count=args.fold_count,
        min_total_trades=args.min_total_trades,
        min_trades_per_fold=args.min_trades_per_fold,
        rule_names=rule_names,
        tag=args.tag,
    )

    results = validate_step_candidate_refinements(config)
    paths = save_step_candidate_refinement_outputs(results, config)

    print("Step Index Trend Channel Candidate Refinement Validation Report")
    print("=" * 90)
    print(f"Cache path: {args.cache_path}")
    print(f"Fold count: {args.fold_count}")
    print(f"Rule names: {','.join(rule_names)}")
    print("Cached design: loads saved cache once; no strategy pipeline rebuild.")

    print_section("Workload / Efficiency Summary")
    print(results["workload"])

    print_section("Refinement Ranking")
    print(results["summary"])

    print_section("Fold Detail Preview")
    print(results["fold_detail"].head(40))

    print_section("Selected Trade Preview")
    print(results["selected_trade_preview"].head(40))

    print_section("Saved CSV Outputs")
    for name, path in paths.items():
        print(f"{name}: {path}")

    print("\nInterpretation Hint")
    print("-" * 90)
    print("Only keep a refinement if it improves average R, keeps enough trades, and improves fold stability without using post-entry information.")


if __name__ == "__main__":
    main()
