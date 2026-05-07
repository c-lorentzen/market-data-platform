import argparse
from pathlib import Path

import pandas as pd

from mdp.backtesting.trend_channel_management_optimizer import compare_management_profiles


def parse_arguments():
    """
    Parses command-line options for comparing candle-based trade-management policies.
    """
    parser = argparse.ArgumentParser(
        description="Compare candle-based trend-channel trade-management policies."
    )

    parser.add_argument("--symbol", required=True, help='Market symbol, for example: "Step Index".')
    parser.add_argument("--timeframes", default="H1,M1", help='Comma-separated timeframes, for example: "H1,M1".')
    parser.add_argument("--analysis-limit", type=int, default=300, help="Past candles used by the analysis pipeline.")
    parser.add_argument("--dataset-limit", type=int, default=2000, help="Entry-timeframe candles used for cutoff generation.")
    parser.add_argument("--start-time", default=None, help="Optional dataset start timestamp.")
    parser.add_argument("--end-time", default=None, help="Optional dataset end timestamp.")
    parser.add_argument("--warmup-candles", type=int, default=300, help="Past candles required before the first cutoff.")
    parser.add_argument("--horizon-candles", type=int, default=120, help="Future candles used for outcome simulation.")
    parser.add_argument("--cutoff-step-candles", type=int, default=20, help="Spacing between cutoff points.")
    parser.add_argument("--step-size", type=int, default=20, help="Rolling channel step size.")
    parser.add_argument("--top-n-per-timeframe", type=int, default=3, help="Valid channel candidates kept per timeframe.")
    parser.add_argument("--include-historical", action="store_true", help="Include older rolling windows instead of active-only windows.")
    parser.add_argument("--recent-window", type=int, default=None, help="Optional adaptive recent-window override.")
    parser.add_argument("--atr-window", type=int, default=None, help="Optional ATR-window override.")
    parser.add_argument("--event-lookback", type=int, default=40, help="Recent candles used to detect events.")
    parser.add_argument("--max-retest-candles", type=int, default=20, help="Maximum candles to search for retest.")
    parser.add_argument("--max-false-breakout-candles", type=int, default=10, help="Maximum candles to detect false breakout.")
    parser.add_argument("--max-confirmation-candles", type=int, default=20, help="Maximum candles to detect continuation confirmation.")
    parser.add_argument("--min-rejection-wick-ratio", type=float, default=0.35, help="Minimum rejection wick ratio.")
    parser.add_argument("--continuation-multiplier", type=float, default=2.0, help="Continuation threshold multiplier.")
    parser.add_argument("--minimum-rr", type=float, default=1.50, help="Minimum RR required for target availability.")
    parser.add_argument("--no-horizon-close", action="store_true", help="Do not mark remaining size at the horizon close.")
    parser.add_argument("--save-results", action="store_true", help="Save policy comparison CSV files.")
    parser.add_argument("--output-dir", default="outputs/trend_channel_management_optimizer", help="Folder where CSV files are saved.")
    return parser.parse_args()


def save_results(output_dir: str, symbol: str, summary_df: pd.DataFrame, trades_df: pd.DataFrame, cutoff_schedule_df: pd.DataFrame):
    """
    Saves policy-comparison outputs as CSV files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    safe_symbol = symbol.lower().replace(" ", "_").replace("/", "_")
    summary_df.to_csv(output_path / f"{safe_symbol}_management_policy_summary.csv", index=False)
    trades_df.to_csv(output_path / f"{safe_symbol}_management_policy_trades.csv", index=False)
    cutoff_schedule_df.to_csv(output_path / f"{safe_symbol}_management_policy_cutoff_schedule.csv", index=False)


def print_report(args, summary_df: pd.DataFrame, trades_df: pd.DataFrame, cutoff_schedule_df: pd.DataFrame):
    """
    Prints a readable management-policy comparison report.
    """
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 260)

    print("Trend Channel Management Policy Optimization Report")
    print("=" * 90)
    print(f"Symbol: {args.symbol}")
    print(f"Timeframes: {args.timeframes}")
    print(f"Dataset limit: {args.dataset_limit}")
    print(f"Analysis limit per cutoff: {args.analysis_limit}")
    print(f"Horizon candles: {args.horizon_candles}")
    print(f"Cutoff step candles: {args.cutoff_step_candles}")
    print(f"Minimum RR: {args.minimum_rr}")
    print(f"Active only: {not args.include_historical}")
    print()

    print("Policy Ranking Summary")
    print("-" * 90)
    if summary_df.empty:
        print("No policy summary was produced.")
    else:
        ranking_columns = [
            "policy_name",
            "policy_score",
            "policy_quality_state",
            "managed_trades",
            "positive_exit_rate",
            "average_managed_r",
            "median_managed_r",
            "loss_exit_count",
            "partial_exit_rate",
            "breakeven_or_lock_rate",
            "trailing_activation_rate",
            "improved_vs_original_rate",
            "partial_take_profit_r",
            "move_stop_to_breakeven_r",
            "locked_profit_r",
            "trailing_start_r",
            "trailing_distance_r",
            "time_exit_candles",
            "full_target_mode",
        ]
        available_columns = [column for column in ranking_columns if column in summary_df.columns]
        print(summary_df[available_columns])
    print()

    print("Best Policy")
    print("-" * 90)
    if summary_df.empty:
        print("No best policy was selected.")
    else:
        print(summary_df.head(1).T)
    print()

    print("Managed Trades Preview for Best Policy")
    print("-" * 90)
    if trades_df.empty or summary_df.empty:
        print("No managed trades were produced.")
    else:
        best_policy_name = summary_df.iloc[0]["policy_name"]
        best_trades_df = trades_df[trades_df["policy_name"] == best_policy_name]
        preview_columns = [
            "policy_name",
            "analysis_cutoff",
            "plan_direction",
            "event_state",
            "original_outcome_state",
            "original_max_favorable_r",
            "original_max_adverse_r",
            "partial_exit_hit",
            "breakeven_or_profit_lock_activated",
            "trailing_activated",
            "managed_exit_state",
            "managed_realized_r",
            "management_improved_vs_original",
        ]
        available_columns = [column for column in preview_columns if column in best_trades_df.columns]
        print(best_trades_df[available_columns].head(20))
    print()

    print("Cutoff Schedule Preview")
    print("-" * 90)
    if cutoff_schedule_df.empty:
        print("No cutoff schedule was created.")
    else:
        print(cutoff_schedule_df.head(5))


def main():
    """
    Runs the candle-based management-policy comparison.
    """
    args = parse_arguments()

    summary_df, trades_df, cutoff_schedule_df = compare_management_profiles(
        symbol=args.symbol,
        timeframes=args.timeframes,
        analysis_limit=args.analysis_limit,
        dataset_limit=args.dataset_limit,
        start_time=args.start_time,
        end_time=args.end_time,
        warmup_candles=args.warmup_candles,
        horizon_candles=args.horizon_candles,
        cutoff_step_candles=args.cutoff_step_candles,
        step_size=args.step_size,
        top_n_per_timeframe=args.top_n_per_timeframe,
        active_only=not args.include_historical,
        recent_window=args.recent_window,
        atr_window=args.atr_window,
        event_lookback=args.event_lookback,
        max_retest_candles=args.max_retest_candles,
        max_false_breakout_candles=args.max_false_breakout_candles,
        max_confirmation_candles=args.max_confirmation_candles,
        min_rejection_wick_ratio=args.min_rejection_wick_ratio,
        continuation_multiplier=args.continuation_multiplier,
        minimum_rr=args.minimum_rr,
        exit_at_horizon_close=not args.no_horizon_close,
    )

    print_report(args, summary_df, trades_df, cutoff_schedule_df)

    if args.save_results:
        save_results(args.output_dir, args.symbol, summary_df, trades_df, cutoff_schedule_df)
        print()
        print(f"Saved results to: {args.output_dir}")


if __name__ == "__main__":
    main()
