import argparse
from pathlib import Path

import pandas as pd

from mdp.structure.trend_channel_events import analyze_trend_channel_events


def parse_arguments():
    """
    Parses command-line options for event analysis.
    """
    parser = argparse.ArgumentParser(
        description="Analyze scale-aware trend channel breakout, retest, and false-breakout events."
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
        "--end-time",
        default=None,
        help="Optional historical cutoff timestamp. Only candles at or before this time are used.",
    )
    parser.add_argument(
        "--include-forming-candle",
        action="store_true",
        help="Allow the candle currently forming at --end-time. Default uses only fully closed candles.",
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
        help="Minimum TP2 reward/risk required before actionability becomes trade-plan ready.",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save event, scenario, summary, and candidate tables as CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/trend_channel_events",
        help="Folder used when saving CSV outputs.",
    )

    return parser.parse_args()


def print_event_summary(event_df: pd.DataFrame) -> None:
    """
    Prints compact event classification results.
    """
    print("\nScale-Aware Event Summary")
    print("-" * 90)

    if event_df.empty:
        print("No event row was created.")
        return

    display_columns = [
        "scenario_state",
        "event_state",
        "actionability",
        "event_direction",
        "event_timeframe",
        "event_channel_role",
        "event_channel_type",
        "event_channel_current_state",
        "breakout_detected",
        "breakout_timestamp",
        "retest_detected",
        "retest_timestamp",
        "retest_rejected",
        "retest_rejection_timestamp",
        "false_breakout_detected",
        "breakout_impulse_confirmed",
        "breakout_impulse_timestamp",
        "continuation_after_retest_confirmed",
        "continuation_after_retest_timestamp",
        "event_quality_score",
    ]

    existing_columns = [column for column in display_columns if column in event_df.columns]
    print(event_df[existing_columns])


def print_event_levels(event_df: pd.DataFrame) -> None:
    """
    Prints adaptive event levels and suggested trade-plan reference levels.
    """
    print("\nScale-Aware Event Levels")
    print("-" * 90)

    if event_df.empty:
        print("No event levels were created.")
        return

    display_columns = [
        "latest_close",
        "event_boundary_price",
        "event_trigger_price",
        "event_retest_lower",
        "event_retest_upper",
        "event_invalidation_price",
        "suggested_entry_mode",
        "suggested_entry_price",
        "suggested_stop_loss",
        "suggested_tp1",
        "suggested_tp2",
        "suggested_rr_tp1",
        "suggested_rr_tp2",
        "risk_efficiency_state",
    ]

    existing_columns = [column for column in display_columns if column in event_df.columns]
    print(event_df[existing_columns])


def print_adaptive_thresholds(event_df: pd.DataFrame) -> None:
    """
    Prints adaptive thresholds used by the event detector.
    """
    print("\nAdaptive Thresholds Used by Event Detector")
    print("-" * 90)

    if event_df.empty:
        print("No adaptive threshold row was created.")
        return

    display_columns = [
        "adaptive_breakout_threshold",
        "adaptive_retest_tolerance",
        "adaptive_stop_buffer",
        "adaptive_min_target_distance",
        "latest_timestamp",
        "notes",
    ]

    existing_columns = [column for column in display_columns if column in event_df.columns]
    print(event_df[existing_columns])


def print_scenario_context(scenario_df: pd.DataFrame) -> None:
    """
    Prints the scenario context that the event detector consumed.
    """
    print("\nScenario Context Used")
    print("-" * 90)

    if scenario_df.empty:
        print("No scenario context was available.")
        return

    display_columns = [
        "alignment_state",
        "scenario_state",
        "actionability",
        "scenario_direction",
        "structure_timeframe",
        "entry_timeframe",
        "structure_bias",
        "entry_bias",
        "bearish_trigger_price",
        "bullish_trigger_price",
        "bearish_invalidation_price",
        "bullish_invalidation_price",
    ]

    existing_columns = [column for column in display_columns if column in scenario_df.columns]
    print(scenario_df[existing_columns])


def print_candidate_context(candidates_df: pd.DataFrame) -> None:
    """
    Prints valid channel candidates available to the event detector.
    """
    print("\nValid Channel Candidates Available")
    print("-" * 90)

    if candidates_df.empty:
        print("No valid channel candidates were available.")
        return

    display_columns = [
        "timeframe",
        "rolling_channel_id",
        "window_size",
        "channel_role",
        "rolling_quality_score",
        "channel_type",
        "current_state",
        "r_squared",
        "containment_rate",
        "channel_width",
        "current_lower",
        "current_upper",
    ]

    existing_columns = [column for column in display_columns if column in candidates_df.columns]
    print(candidates_df[existing_columns])


def save_results(
    symbol: str,
    event_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """
    Saves event analysis outputs as CSV files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Keep filenames predictable for later reports and dashboards.
    clean_symbol = symbol.lower().replace(" ", "_")
    event_path = output_path / f"{clean_symbol}_trend_channel_events.csv"
    scenario_path = output_path / f"{clean_symbol}_trend_channel_event_scenario.csv"
    summary_path = output_path / f"{clean_symbol}_trend_channel_event_timeframes.csv"
    candidates_path = output_path / f"{clean_symbol}_trend_channel_event_candidates.csv"

    event_df.to_csv(event_path, index=False)
    scenario_df.to_csv(scenario_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    candidates_df.to_csv(candidates_path, index=False)

    print("\nSaved results:")
    print(f"Events: {event_path}")
    print(f"Scenario: {scenario_path}")
    print(f"Timeframes: {summary_path}")
    print(f"Candidates: {candidates_path}")


def main():
    args = parse_arguments()
    active_only = not args.include_historical

    # Run scenario analysis first, then classify measurable event states from that context.
    event_df, scenario_df, summary_df, candidates_df = analyze_trend_channel_events(
        symbol=args.symbol,
        timeframes=args.timeframes,
        limit=args.limit,
        step_size=args.step_size,
        top_n_per_timeframe=args.top_n_per_timeframe,
        active_only=active_only,
        recent_window=args.recent_window,
        atr_window=args.atr_window,
        end_time=args.end_time,
        closed_candles_only=not args.include_forming_candle,
        event_lookback=args.event_lookback,
        max_retest_candles=args.max_retest_candles,
        max_false_breakout_candles=args.max_false_breakout_candles,
        max_confirmation_candles=args.max_confirmation_candles,
        min_rejection_wick_ratio=args.min_rejection_wick_ratio,
        continuation_multiplier=args.continuation_multiplier,
        minimum_rr=args.minimum_rr,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 520)

    print("Scale-Aware Trend Channel Event Report")
    print("=" * 90)
    print(f"Symbol: {args.symbol}")
    print(f"Timeframes: {args.timeframes}")
    print(f"Limit per timeframe: {args.limit}")
    print(f"Active only: {active_only}")
    print(f"Event lookback: {args.event_lookback}")
    print(f"Minimum RR: {args.minimum_rr}")
    print(f"Recent window override: {args.recent_window}")
    print(f"ATR window override: {args.atr_window}")
    print(f"Analysis cutoff: {args.end_time}")
    print(f"Closed candles only: {not args.include_forming_candle}")

    print_event_summary(event_df)
    print_event_levels(event_df)
    print_adaptive_thresholds(event_df)
    print_scenario_context(scenario_df)
    print_candidate_context(candidates_df)

    if args.save_results:
        # Save only when the user explicitly requests generated outputs.
        save_results(
            symbol=args.symbol,
            event_df=event_df,
            scenario_df=scenario_df,
            summary_df=summary_df,
            candidates_df=candidates_df,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
