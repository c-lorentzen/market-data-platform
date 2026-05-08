"""Build a Markdown/JSON strategy card for a Step Index candidate."""

from __future__ import annotations

import argparse

from mdp.backtesting.trend_channel_strategy_card import (
    StrategyCardConfig,
    build_strategy_card,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the strategy card builder."""

    parser = argparse.ArgumentParser(
        description="Build a strategy card for a Step Index trend-channel candidate."
    )
    parser.add_argument("--candidate-dataset-path", required=True)
    parser.add_argument("--candidate-name", default="Step Index Trend Channel Candidate v1.2")
    parser.add_argument("--rule-name", default="baseline_d1_h4_m15_m5_not_against")
    parser.add_argument("--symbol", default="Step Index")
    parser.add_argument("--timeframes", default="D1,H4,H1,M15,M5,M1")
    parser.add_argument("--tag", default="step_candidate_v1_2_strategy_card_1000")
    parser.add_argument("--output-dir", default="outputs/strategy_cards")
    return parser.parse_args()


def main() -> None:
    """Build and print generated strategy card paths."""

    args = parse_args()

    # The config keeps the script simple and makes the build repeatable.
    config = StrategyCardConfig(
        candidate_dataset_path=args.candidate_dataset_path,
        candidate_name=args.candidate_name,
        rule_name=args.rule_name,
        symbol=args.symbol,
        timeframes=args.timeframes,
        tag=args.tag,
        output_dir=args.output_dir,
    )

    artifacts = build_strategy_card(config)

    print("Step Index Strategy Card Build Report")
    print("=" * 90)
    print(f"Candidate dataset path: {args.candidate_dataset_path}")
    print(f"Candidate name: {args.candidate_name}")
    print(f"Rule name: {args.rule_name}")
    print("Design: loads one candidate dataset; documents research status honestly.")
    print()
    print("Saved Strategy Card Outputs")
    print("-" * 90)
    print(f"Markdown: {artifacts.markdown_path}")
    print(f"JSON: {artifacts.json_path}")
    print(f"Summary CSV: {artifacts.summary_csv_path}")


if __name__ == "__main__":
    main()
