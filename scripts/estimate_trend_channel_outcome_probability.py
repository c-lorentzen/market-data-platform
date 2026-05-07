import argparse
from pathlib import Path

import pandas as pd

from mdp.structure.trend_channel_outcome_probability import estimate_trend_channel_outcome_probability


def parse_arguments():
    """
    Parses command-line options for trend-channel outcome probability estimation.
    """
    parser = argparse.ArgumentParser(
        description="Estimate scale-aware SL/TP probability proxies from a trend-channel trade plan."
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
        "--rr-decay",
        type=float,
        default=0.45,
        help="Controls how strongly farther RR targets reduce hit-probability proxy.",
    )
    parser.add_argument(
        "--stop-bias-multiplier",
        type=float,
        default=1.25,
        help="Controls how much event uncertainty contributes to stop probability.",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save probability, trade plan, event, scenario, summary, and candidates as CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/trend_channel_probabilities",
        help="Folder used when saving CSV outputs.",
    )

    return parser.parse_args()


def configure_display() -> None:
    """
    Makes pandas print full rows so terminal output is easier to inspect.
    """
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_colwidth", 120)


def format_percent_columns(probability_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a readable copy with probability and weight values shown as percentages.
    """
    readable_df = probability_df.copy()

    percent_columns = [
        "stop_loss_probability",
        "tp1_probability",
        "tp2_probability",
        "tp3_probability",
        "tp1_weight",
        "tp2_weight",
        "tp3_weight",
        "probability_confidence_score",
    ]

    # Convert decimals to rounded percentages for terminal readability.
    for column in percent_columns:
        if column in readable_df.columns:
            readable_df[column] = readable_df[column].apply(
                lambda value: None if pd.isna(value) else round(float(value) * 100.0, 2)
            )

    return readable_df


def print_probability_summary(probability_df: pd.DataFrame) -> None:
    """
    Prints the main probability state and expected value.
    """
    print("\nScale-Aware Outcome Probability Summary")
    print("-" * 90)

    if probability_df.empty:
        print("No probability estimate was created.")
        return

    display_df = format_percent_columns(probability_df)
    display_columns = [
        "probability_state",
        "estimation_method",
        "plan_direction",
        "event_state",
        "available_targets",
        "best_probability_target",
        "best_expected_value_target",
        "weighted_expected_r",
        "probability_confidence_score",
        "probability_model_status",
    ]

    existing_columns = [column for column in display_columns if column in display_df.columns]
    print(display_df[existing_columns])


def print_target_probabilities(probability_df: pd.DataFrame) -> None:
    """
    Prints SL and available TP probabilities.
    """
    print("\nSL / TP Probability Breakdown")
    print("-" * 90)

    if probability_df.empty:
        print("No probability breakdown was created.")
        return

    display_df = format_percent_columns(probability_df)
    display_columns = [
        "entry_price",
        "stop_loss",
        "risk_points",
        "stop_loss_probability",
        "tp1_price",
        "tp1_rr",
        "tp1_available",
        "tp1_probability",
        "tp2_price",
        "tp2_rr",
        "tp2_available",
        "tp2_probability",
        "tp3_price",
        "tp3_rr",
        "tp3_available",
        "tp3_probability",
    ]

    existing_columns = [column for column in display_columns if column in display_df.columns]
    print(display_df[existing_columns])


def print_target_weights(probability_df: pd.DataFrame) -> None:
    """
    Prints scale-out weights only for available targets.
    """
    print("\nWeighted Target Allocation Proxy")
    print("-" * 90)

    if probability_df.empty:
        print("No target weights were created.")
        return

    display_df = format_percent_columns(probability_df)
    display_columns = [
        "target_plan_state",
        "tp1_weight",
        "tp2_weight",
        "tp3_weight",
        "notes",
    ]

    existing_columns = [column for column in display_columns if column in display_df.columns]
    print(display_df[existing_columns])


def print_source_trade_plan(trade_plan_df: pd.DataFrame) -> None:
    """
    Prints the source trade plan used by the probability estimator.
    """
    print("\nSource Trade Plan Used")
    print("-" * 90)

    if trade_plan_df.empty:
        print("No source trade plan was available.")
        return

    display_columns = [
        "plan_state",
        "plan_direction",
        "plan_actionability",
        "entry_mode",
        "risk_efficiency_state",
        "target_plan_state",
        "available_targets",
        "best_available_target",
        "best_available_rr",
    ]

    existing_columns = [column for column in display_columns if column in trade_plan_df.columns]
    print(trade_plan_df[existing_columns])


def save_outputs(
    output_dir: str,
    symbol: str,
    probability_df: pd.DataFrame,
    trade_plan_df: pd.DataFrame,
    event_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
) -> None:
    """
    Saves probability and source analysis outputs as CSV files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Keep filenames simple and safe for Windows paths.
    safe_symbol = symbol.lower().replace(" ", "_").replace("/", "_")

    probability_df.to_csv(output_path / f"{safe_symbol}_outcome_probability.csv", index=False)
    trade_plan_df.to_csv(output_path / f"{safe_symbol}_trade_plan.csv", index=False)
    event_df.to_csv(output_path / f"{safe_symbol}_event.csv", index=False)
    scenario_df.to_csv(output_path / f"{safe_symbol}_scenario.csv", index=False)
    summary_df.to_csv(output_path / f"{safe_symbol}_timeframe_summary.csv", index=False)
    candidates_df.to_csv(output_path / f"{safe_symbol}_channel_candidates.csv", index=False)

    print(f"\nSaved probability outputs to: {output_path}")


def main() -> None:
    """
    Runs the command-line probability estimator.
    """
    configure_display()
    args = parse_arguments()

    probability_df, trade_plan_df, event_df, scenario_df, summary_df, candidates_df = estimate_trend_channel_outcome_probability(
        symbol=args.symbol,
        timeframes=args.timeframes,
        limit=args.limit,
        step_size=args.step_size,
        top_n_per_timeframe=args.top_n_per_timeframe,
        active_only=not args.include_historical,
        recent_window=args.recent_window,
        atr_window=args.atr_window,
        end_time=args.end_time,
        event_lookback=args.event_lookback,
        max_retest_candles=args.max_retest_candles,
        max_false_breakout_candles=args.max_false_breakout_candles,
        max_confirmation_candles=args.max_confirmation_candles,
        min_rejection_wick_ratio=args.min_rejection_wick_ratio,
        continuation_multiplier=args.continuation_multiplier,
        minimum_rr=args.minimum_rr,
        rr_decay=args.rr_decay,
        stop_bias_multiplier=args.stop_bias_multiplier,
    )

    print("Scale-Aware Trend Channel Outcome Probability Report")
    print("=" * 90)
    print(f"Symbol: {args.symbol}")
    print(f"Timeframes: {args.timeframes}")
    print(f"Limit per timeframe: {args.limit}")
    print(f"Active only: {not args.include_historical}")
    print(f"Minimum RR: {args.minimum_rr}")
    print(f"RR decay: {args.rr_decay}")
    print(f"Stop bias multiplier: {args.stop_bias_multiplier}")
    print(f"Recent window override: {args.recent_window}")
    print(f"ATR window override: {args.atr_window}")
    print(f"Analysis cutoff: {args.end_time}")

    print_probability_summary(probability_df)
    print_target_probabilities(probability_df)
    print_target_weights(probability_df)
    print_source_trade_plan(trade_plan_df)

    if args.save_results:
        save_outputs(
            output_dir=args.output_dir,
            symbol=args.symbol,
            probability_df=probability_df,
            trade_plan_df=trade_plan_df,
            event_df=event_df,
            scenario_df=scenario_df,
            summary_df=summary_df,
            candidates_df=candidates_df,
        )


if __name__ == "__main__":
    main()
