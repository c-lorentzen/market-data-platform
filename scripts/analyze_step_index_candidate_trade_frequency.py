"""CLI for Step Index candidate trade-frequency profiling."""

from __future__ import annotations

import argparse

from mdp.backtesting.trend_channel_trade_frequency_profile import (
    TradeFrequencyProfileConfig,
    profile_trade_frequency,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze how often a Step Index trend-channel candidate appears from a saved research cache."
    )
    parser.add_argument("--cache-path", required=True, help="Path to the saved entry research cache CSV.")
    parser.add_argument(
        "--rule-name",
        default="candidate_v1_base",
        choices=[
            "candidate_v1_base",
            "avoid_tp3_only",
            "retest_rejection_only",
            "continuation_confirm2_only",
        ],
        help="Candidate rule to profile.",
    )
    parser.add_argument("--tag", default="step_candidate_frequency_profile", help="Output filename tag.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = TradeFrequencyProfileConfig(
        cache_path=args.cache_path,
        rule_name=args.rule_name,
        tag=args.tag,
    )
    results = profile_trade_frequency(config)

    print("Step Index Candidate Trade Frequency Profile Report")
    print("=" * 90)
    print(f"Cache path: {args.cache_path}")
    print(f"Rule name: {args.rule_name}")
    print("Cached design: loads saved cache once; no strategy pipeline rebuild.")

    print("\nFrequency Summary")
    print("-" * 90)
    print(results["frequency_summary"])

    print("\nDaily Trade Summary")
    print("-" * 90)
    print(results["daily_summary"])

    print("\nHourly Trade Summary")
    print("-" * 90)
    print(results["hourly_summary"].head(24))

    print("\nSelected Trade Preview")
    print("-" * 90)
    preview_columns = [
        column
        for column in [
            "analysis_cutoff",
            "entry_mode",
            "entry_variant_name",
            "scenario_state",
            "event_state",
            "risk_points",
            "available_targets",
            "target_plan_state",
            "managed_exit_state",
            "managed_realized_r",
        ]
        if column in results["selected_trades"].columns
    ]
    print(results["selected_trades"][preview_columns].head(30))

    print("\nInterpretation Hint")
    print("-" * 90)
    print(
        "Use this to estimate realistic daily opportunity frequency. A strategy does not need many trades per day; "
        "it needs enough clean, repeatable, positive-expectancy trades without forcing weak setups."
    )


if __name__ == "__main__":
    main()
