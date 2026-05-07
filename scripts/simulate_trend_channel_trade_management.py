import argparse
from pathlib import Path

import pandas as pd

from mdp.backtesting.trend_channel_trade_management import build_trend_channel_management_dataset


def parse_arguments():
    """
    Parses command-line options for trend-channel trade-management simulation.
    """
    parser = argparse.ArgumentParser(
        description="Simulate partial exits, breakeven/profit-lock, and trailing stops for trend-channel trade plans."
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

    parser.add_argument("--partial-take-profit-r", type=float, default=0.50, help="R level where a partial scalp exit is taken.")
    parser.add_argument("--partial-exit-fraction", type=float, default=0.50, help="Position fraction closed at partial TP.")
    parser.add_argument("--move-stop-to-breakeven-r", type=float, default=0.75, help="Favorable R needed before moving stop to breakeven/profit lock.")
    parser.add_argument("--locked-profit-r", type=float, default=0.00, help="Profit locked after breakeven trigger, in R units.")
    parser.add_argument("--trailing-start-r", type=float, default=1.25, help="Favorable R needed before trailing starts.")
    parser.add_argument("--trailing-distance-r", type=float, default=0.75, help="Trailing stop distance in R units.")
    parser.add_argument("--full-target-mode", choices=["nearest", "best", "none"], default="nearest", help="Target used to close remaining size.")
    parser.add_argument("--time-exit-candles", type=int, default=None, help="Optional candle count for time-based exit.")
    parser.add_argument("--no-horizon-close", action="store_true", help="Do not mark remaining size at the horizon close.")
    parser.add_argument("--intrabar-policy", choices=["conservative", "optimistic"], default="conservative", help="How to handle stop and target touched in same candle.")

    parser.add_argument("--save-results", action="store_true", help="Save managed outcome CSV files.")
    parser.add_argument("--output-dir", default="outputs/trend_channel_management", help="Folder where managed CSV files are saved.")
    return parser.parse_args()


def save_results(output_dir: str, symbol: str, results_df: pd.DataFrame, summary_df: pd.DataFrame, cutoff_schedule_df: pd.DataFrame):
    """
    Saves managed-trade simulation outputs as CSV files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    safe_symbol = symbol.lower().replace(" ", "_").replace("/", "_")
    results_df.to_csv(output_path / f"{safe_symbol}_managed_trade_outcomes.csv", index=False)
    summary_df.to_csv(output_path / f"{safe_symbol}_managed_trade_summary.csv", index=False)
    cutoff_schedule_df.to_csv(output_path / f"{safe_symbol}_managed_cutoff_schedule.csv", index=False)


def print_report(args, results_df: pd.DataFrame, summary_df: pd.DataFrame, cutoff_schedule_df: pd.DataFrame):
    """
    Prints a readable managed-trade simulation report.
    """
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 260)

    print("Trend Channel Trade Management Simulation Report")
    print("=" * 90)
    print(f"Symbol: {args.symbol}")
    print(f"Timeframes: {args.timeframes}")
    print(f"Dataset limit: {args.dataset_limit}")
    print(f"Analysis limit per cutoff: {args.analysis_limit}")
    print(f"Horizon candles: {args.horizon_candles}")
    print(f"Cutoff step candles: {args.cutoff_step_candles}")
    print(f"Partial TP R: {args.partial_take_profit_r}")
    print(f"Partial exit fraction: {args.partial_exit_fraction}")
    print(f"Move stop trigger R: {args.move_stop_to_breakeven_r}")
    print(f"Locked profit R: {args.locked_profit_r}")
    print(f"Trailing start R: {args.trailing_start_r}")
    print(f"Trailing distance R: {args.trailing_distance_r}")
    print(f"Full target mode: {args.full_target_mode}")
    print(f"Time exit candles: {args.time_exit_candles}")
    print(f"Intrabar policy: {args.intrabar_policy}")
    print()

    print("Managed Outcome Summary")
    print("-" * 90)
    print(summary_df)
    print()

    print("Managed Rows Preview")
    print("-" * 90)
    if results_df.empty:
        print("No managed trade rows were produced.")
    else:
        preview_columns = [
            "analysis_cutoff",
            "plan_direction",
            "event_state",
            "original_outcome_state",
            "original_first_terminal_outcome",
            "original_max_favorable_r",
            "original_max_adverse_r",
            "partial_exit_hit",
            "breakeven_or_profit_lock_activated",
            "trailing_activated",
            "managed_exit_state",
            "managed_realized_r",
            "management_improved_vs_original",
        ]
        available_columns = [column for column in preview_columns if column in results_df.columns]
        print(results_df[available_columns].head(25))
    print()

    print("Cutoff Schedule Preview")
    print("-" * 90)
    if cutoff_schedule_df.empty:
        print("No cutoff schedule was created.")
    else:
        print(cutoff_schedule_df.head(5))


def main():
    """
    Runs the trade-management simulation from the command line.
    """
    args = parse_arguments()

    results_df, summary_df, cutoff_schedule_df = build_trend_channel_management_dataset(
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
        partial_take_profit_r=args.partial_take_profit_r,
        partial_exit_fraction=args.partial_exit_fraction,
        move_stop_to_breakeven_r=args.move_stop_to_breakeven_r,
        locked_profit_r=args.locked_profit_r,
        trailing_start_r=args.trailing_start_r,
        trailing_distance_r=args.trailing_distance_r,
        full_target_mode=args.full_target_mode,
        time_exit_candles=args.time_exit_candles,
        exit_at_horizon_close=not args.no_horizon_close,
        intrabar_policy=args.intrabar_policy,
    )

    print_report(args, results_df, summary_df, cutoff_schedule_df)

    if args.save_results:
        save_results(args.output_dir, args.symbol, results_df, summary_df, cutoff_schedule_df)
        print()
        print(f"Saved results to: {args.output_dir}")


if __name__ == "__main__":
    main()
