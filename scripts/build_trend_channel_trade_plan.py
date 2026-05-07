import argparse
from pathlib import Path

import pandas as pd

from mdp.structure.trend_channel_trade_plan import build_trend_channel_trade_plan


def parse_arguments():
    """
    Parses command-line options for scale-aware trade-plan building.
    """
    parser = argparse.ArgumentParser(
        description="Build a scale-aware trade plan from trend channel event analysis."
    )

    parser.add_argument(
        "--symbol",
        required=True,
        help='Market symbol, for example: "Step Index" or "Volatility 75 Index".',
    )
    parser.add_argument(
        "--timeframes",
        default="H1,M1",
        help='Comma or space-separated timeframes, for example: "H1,M15,M5,M1".',
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=300,
        help="Number of candles to load per timeframe.",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=20,
        help="Rolling step size used when historical windows are included.",
    )
    parser.add_argument(
        "--top-n-per-timeframe",
        type=int,
        default=3,
        help="Maximum valid channel candidates to keep per timeframe.",
    )
    parser.add_argument(
        "--include-historical",
        action="store_true",
        help="Include older rolling windows instead of only latest active windows.",
    )
    parser.add_argument(
        "--recent-window",
        type=int,
        default=None,
        help="Optional override for adaptive recent market-stage window.",
    )
    parser.add_argument(
        "--atr-window",
        type=int,
        default=None,
        help="Optional override for adaptive ATR calculation window.",
    )
    parser.add_argument(
        "--event-lookback",
        type=int,
        default=40,
        help="Recent entry-timeframe candles used to detect breakout events.",
    )
    parser.add_argument(
        "--max-retest-candles",
        type=int,
        default=20,
        help="Maximum candles after breakout to search for a retest.",
    )
    parser.add_argument(
        "--max-false-breakout-candles",
        type=int,
        default=10,
        help="Maximum candles after breakout to detect a failed breakout.",
    )
    parser.add_argument(
        "--max-confirmation-candles",
        type=int,
        default=20,
        help="Maximum candles after breakout to detect continuation.",
    )
    parser.add_argument(
        "--min-rejection-wick-ratio",
        type=float,
        default=0.35,
        help="Minimum rejection wick ratio used to confirm a retest rejection.",
    )
    parser.add_argument(
        "--continuation-multiplier",
        type=float,
        default=2.0,
        help="Adaptive breakout-threshold multiplier used for continuation checks.",
    )
    parser.add_argument(
        "--minimum-rr",
        type=float,
        default=1.50,
        help="Minimum reward/risk required before a target becomes available.",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save trade plan, event, scenario, summary, and candidates as CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/trend_channel_trade_plans",
        help="Folder used when saving CSV outputs.",
    )

    return parser.parse_args()


def print_trade_plan_summary(trade_plan_df: pd.DataFrame) -> None:
    """
    Prints the main trade-plan state and actionability.
    """
    print("\nScale-Aware Trade Plan Summary")
    print("-" * 90)

    if trade_plan_df.empty:
        print("No trade plan was created.")
        return

    display_columns = [
        "plan_state",
        "plan_direction",
        "plan_actionability",
        "entry_timeframe",
        "entry_mode",
        "risk_efficiency_state",
        "event_state",
        "event_quality_score",
        "available_target_count",
        "available_targets",
        "best_available_target",
        "best_available_rr",
        "target_plan_state",
        "suggested_management_style",
    ]

    existing_columns = [column for column in display_columns if column in trade_plan_df.columns]
    print(trade_plan_df[existing_columns])


def print_trade_plan_levels(trade_plan_df: pd.DataFrame) -> None:
    """
    Prints entry, SL, TP, and RR levels.
    """
    print("\nScale-Aware Trade Plan Levels")
    print("-" * 90)

    if trade_plan_df.empty:
        print("No trade-plan levels were created.")
        return

    display_columns = [
        "latest_close",
        "entry_price",
        "stop_loss",
        "risk_points",
        "risk_points_ratio",
        "tp1_price",
        "tp1_rr",
        "tp1_available",
        "tp2_price",
        "tp2_rr",
        "tp2_available",
        "tp3_price",
        "tp3_rr",
        "tp3_available",
        "minimum_rr",
    ]

    existing_columns = [column for column in display_columns if column in trade_plan_df.columns]
    print(trade_plan_df[existing_columns])


def print_probability_placeholder(trade_plan_df: pd.DataFrame) -> None:
    """
    Prints probability-model status without pretending probabilities exist yet.
    """
    print("\nProbability Model Status")
    print("-" * 90)

    if trade_plan_df.empty:
        print("No trade plan exists, so probability estimation is not available yet.")
        return

    display_columns = [
        "available_targets",
        "probability_model_status",
        "notes",
    ]

    existing_columns = [column for column in display_columns if column in trade_plan_df.columns]
    print(trade_plan_df[existing_columns])


def print_event_context(event_df: pd.DataFrame) -> None:
    """
    Prints the source event used by the trade-plan builder.
    """
    print("\nSource Event Context")
    print("-" * 90)

    if event_df.empty:
        print("No event context was available.")
        return

    display_columns = [
        "scenario_state",
        "event_state",
        "actionability",
        "event_direction",
        "event_timeframe",
        "event_channel_role",
        "breakout_detected",
        "retest_detected",
        "retest_rejected",
        "continuation_after_retest_confirmed",
        "risk_efficiency_state",
    ]

    existing_columns = [column for column in display_columns if column in event_df.columns]
    print(event_df[existing_columns])


def save_outputs(
    output_dir: str,
    symbol: str,
    trade_plan_df: pd.DataFrame,
    event_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
) -> None:
    """
    Saves trade-plan tables as CSV files for later review.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Keep filenames simple and safe for Windows paths.
    safe_symbol = symbol.lower().replace(" ", "_").replace("/", "_")

    trade_plan_df.to_csv(output_path / f"{safe_symbol}_trade_plan.csv", index=False)
    event_df.to_csv(output_path / f"{safe_symbol}_event.csv", index=False)
    scenario_df.to_csv(output_path / f"{safe_symbol}_scenario.csv", index=False)
    summary_df.to_csv(output_path / f"{safe_symbol}_timeframe_summary.csv", index=False)
    candidates_df.to_csv(output_path / f"{safe_symbol}_channel_candidates.csv", index=False)

    print(f"\nSaved trade-plan outputs to: {output_path}")


def main() -> None:
    """
    Runs the command-line trade-plan builder.
    """
    args = parse_arguments()

    trade_plan_df, event_df, scenario_df, summary_df, candidates_df = build_trend_channel_trade_plan(
        symbol=args.symbol,
        timeframes=args.timeframes,
        limit=args.limit,
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
    )

    print("Scale-Aware Trend Channel Trade Plan Report")
    print("=" * 90)
    print(f"Symbol: {args.symbol}")
    print(f"Timeframes: {args.timeframes}")
    print(f"Limit per timeframe: {args.limit}")
    print(f"Active only: {not args.include_historical}")
    print(f"Minimum RR: {args.minimum_rr}")
    print(f"Recent window override: {args.recent_window}")
    print(f"ATR window override: {args.atr_window}")

    print_trade_plan_summary(trade_plan_df)
    print_trade_plan_levels(trade_plan_df)
    print_probability_placeholder(trade_plan_df)
    print_event_context(event_df)

    if args.save_results:
        save_outputs(
            output_dir=args.output_dir,
            symbol=args.symbol,
            trade_plan_df=trade_plan_df,
            event_df=event_df,
            scenario_df=scenario_df,
            summary_df=summary_df,
            candidates_df=candidates_df,
        )


if __name__ == "__main__":
    main()
