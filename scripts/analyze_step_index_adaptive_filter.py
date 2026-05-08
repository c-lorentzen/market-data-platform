
from __future__ import annotations

import argparse
import time

import pandas as pd

from mdp.backtesting.trend_channel_daily_adaptive_filter import (
    AdaptiveFilterConfig,
    analyze_step_candidate_adaptive_filter,
    save_outputs,
)


def _print_section(title: str) -> None:
    print("\n" + title)
    print("-" * 90)


def _print_df(df: pd.DataFrame, max_rows: int = 30) -> None:
    if df is None or df.empty:
        print("Empty DataFrame")
        return

    with pd.option_context(
        "display.max_rows",
        max_rows,
        "display.max_columns",
        None,
        "display.width",
        220,
    ):
        print(df.head(max_rows))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Step Index candidate frequency and simple no-lookahead adaptive filtering."
    )
    parser.add_argument("--cache-path", required=True, help="Path to the saved entry research cache CSV.")
    parser.add_argument(
        "--rule-name",
        default="candidate_v1_base",
        choices=[
            "candidate_v1_base",
            "avoid_tp3_only",
            "require_best_available_rr_high",
            "keep_high_rr_or_bullish_pullback",
        ],
        help="Candidate rule to analyze.",
    )
    parser.add_argument("--tag", default="step_candidate_adaptive_filter", help="Output file tag.")
    parser.add_argument(
        "--min-history-trades",
        type=int,
        default=5,
        help="Minimum previous trades needed before a feature value can be used for skipping.",
    )
    parser.add_argument(
        "--avoid-avg-r-below",
        type=float,
        default=0.0,
        help="Skip feature values whose prior average R is below this threshold.",
    )
    parser.add_argument(
        "--max-trades-per-day",
        type=int,
        default=None,
        help="Optional daily cap after filtering. Omit for no cap.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    start = time.perf_counter()

    config = AdaptiveFilterConfig(
        cache_path=args.cache_path,
        rule_name=args.rule_name,
        tag=args.tag,
        min_history_trades=args.min_history_trades,
        avoid_avg_r_below=args.avoid_avg_r_below,
        max_trades_per_day=args.max_trades_per_day,
    )

    results = analyze_step_candidate_adaptive_filter(config)
    output_paths = save_outputs(results, config)

    elapsed = round(time.perf_counter() - start, 3)

    print("Step Index Candidate Adaptive Filter Diagnostic Report")
    print("=" * 90)
    print(f"Cache path: {args.cache_path}")
    print(f"Rule name: {args.rule_name}")
    print(f"Elapsed seconds: {elapsed}")
    print("Cached design: loads saved cache once; no strategy pipeline rebuild.")
    print("Adaptive filter: no-lookahead rule-learning proxy using prior days only.")

    _print_section("Workload Summary")
    workload = results["workload"].copy()
    workload["elapsed_seconds_total"] = elapsed
    _print_df(workload)

    _print_section("Frequency Summary")
    _print_df(results["frequency_summary"])

    _print_section("Daily Summary After Deduplication")
    _print_df(results["daily_summary"], max_rows=20)

    _print_section("No-Lookahead Adaptive Filter Summary")
    _print_df(results["adaptive_summary"])

    _print_section("Adaptive Decision Preview")
    _print_df(results["adaptive_decisions"], max_rows=40)

    _print_section("Selected Deduped Trade Preview")
    preview_cols = [
        "analysis_cutoff",
        "trade_date",
        "hour_utc",
        "scenario_state",
        "event_state",
        "entry_mode",
        "target_plan_state",
        "risk_points",
        "best_available_rr_bucket",
        "managed_exit_state",
        "managed_realized_r",
    ]
    preview = results["selected_deduped_preview"]
    existing_cols = [col for col in preview_cols if col in preview.columns]
    _print_df(preview[existing_cols], max_rows=40)

    _print_section("Saved CSV Outputs")
    for name, path in output_paths.items():
        print(f"{name}: {path}")

    print("\nInterpretation Hint")
    print("-" * 90)
    print(
        "This diagnostic shows whether the strategy has enough daily opportunities and whether "
        "simple prior-day learning can skip historically weak contexts without future leakage. "
        "If the adaptive filter improves expectancy while keeping enough trades, the next step is "
        "an ML-ready labeled dataset. If it collapses trade count, keep Candidate v1 as research only."
    )


if __name__ == "__main__":
    main()
