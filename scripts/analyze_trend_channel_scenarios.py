import argparse
from pathlib import Path

import pandas as pd

from mdp.structure.trend_channel_scenarios import analyze_trend_channel_scenario


def parse_arguments():
    """
    Parses command-line options for trend channel scenario analysis.
    """
    parser = argparse.ArgumentParser(
        description="Analyze scale-aware trend channel scenarios across multiple timeframes."
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
        help="Rolling step size used when active-only is disabled.",
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
        "--save-results",
        action="store_true",
        help="Save scenario, summary, and candidate tables as CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/trend_channel_scenarios",
        help="Folder used when saving CSV outputs.",
    )

    return parser.parse_args()


def print_scenario_table(scenario_df: pd.DataFrame) -> None:
    """
    Prints the compact scenario summary.
    """
    print("\nScale-Aware Scenario Summary")
    print("-" * 90)

    if scenario_df.empty:
        print("No scenario row was created.")
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
        "structure_channel_type",
        "entry_channel_type",
        "structure_current_state",
        "entry_current_state",
        "structure_current_position_ratio",
        "entry_current_position_ratio",
        "bearish_trigger_price",
        "bullish_trigger_price",
        "bearish_invalidation_price",
        "bullish_invalidation_price",
    ]

    existing_columns = [column for column in display_columns if column in scenario_df.columns]
    print(scenario_df[existing_columns])


def print_scale_levels_table(scenario_df: pd.DataFrame) -> None:
    """
    Prints adaptive retest, buffer, and target distances.
    """
    print("\nScale-Aware Scenario Levels")
    print("-" * 90)

    if scenario_df.empty:
        print("No scenario level row was created.")
        return

    display_columns = [
        "entry_retest_lower",
        "entry_retest_upper",
        "structure_retest_lower",
        "structure_retest_upper",
        "entry_stop_buffer",
        "structure_stop_buffer",
        "entry_min_target_distance",
        "structure_min_target_distance",
        "notes",
    ]

    existing_columns = [column for column in display_columns if column in scenario_df.columns]
    print(scenario_df[existing_columns])


def print_timeframe_summary(summary_df: pd.DataFrame) -> None:
    """
    Prints the timeframe channel context used by the scenario engine.
    """
    print("\nTimeframe Context Used")
    print("-" * 90)

    if summary_df.empty:
        print("No timeframe context rows were created.")
        return

    display_columns = [
        "timeframe",
        "purpose",
        "status",
        "volatility_stage",
        "channel_bias",
        "best_channel_role",
        "best_channel_type",
        "best_current_state",
        "best_window_size",
        "best_rolling_quality_score",
        "best_r_squared",
        "best_channel_width",
        "adaptive_channel_min_width",
        "adaptive_channel_max_width",
        "latest_close",
        "latest_timestamp",
    ]

    existing_columns = [column for column in display_columns if column in summary_df.columns]
    print(summary_df[existing_columns])


def print_candidate_table(candidates_df: pd.DataFrame) -> None:
    """
    Prints the valid candidates that were available to the scenario engine.
    """
    print("\nValid Candidates Used")
    print("-" * 90)

    if candidates_df.empty:
        print("No valid candidates were available.")
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
        "channel_width_usage_ratio",
        "current_lower",
        "current_upper",
    ]

    existing_columns = [column for column in display_columns if column in candidates_df.columns]
    print(candidates_df[existing_columns])


def save_results(
    symbol: str,
    scenario_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """
    Saves scenario analysis outputs as CSV files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Keep filenames predictable so reports can consume them later.
    clean_symbol = symbol.lower().replace(" ", "_")
    scenario_path = output_path / f"{clean_symbol}_trend_channel_scenario.csv"
    summary_path = output_path / f"{clean_symbol}_trend_channel_scenario_timeframes.csv"
    candidates_path = output_path / f"{clean_symbol}_trend_channel_scenario_candidates.csv"

    scenario_df.to_csv(scenario_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    candidates_df.to_csv(candidates_path, index=False)

    print("\nSaved results:")
    print(f"Scenario: {scenario_path}")
    print(f"Timeframes: {summary_path}")
    print(f"Candidates: {candidates_path}")


def main():
    args = parse_arguments()
    active_only = not args.include_historical

    # Run the scenario engine after scale-aware MTF channel analysis.
    scenario_df, summary_df, candidates_df = analyze_trend_channel_scenario(
        symbol=args.symbol,
        timeframes=args.timeframes,
        limit=args.limit,
        step_size=args.step_size,
        top_n_per_timeframe=args.top_n_per_timeframe,
        active_only=active_only,
        recent_window=args.recent_window,
        atr_window=args.atr_window,
        end_time=args.end_time,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 420)

    print("Scale-Aware Trend Channel Scenario Report")
    print("=" * 90)
    print(f"Symbol: {args.symbol}")
    print(f"Timeframes: {args.timeframes}")
    print(f"Limit per timeframe: {args.limit}")
    print(f"Active only: {active_only}")
    print(f"Recent window override: {args.recent_window}")
    print(f"ATR window override: {args.atr_window}")
    print(f"Analysis cutoff: {args.end_time}")

    print_scenario_table(scenario_df)
    print_scale_levels_table(scenario_df)
    print_timeframe_summary(summary_df)
    print_candidate_table(candidates_df)

    if args.save_results:
        # Save only when the user explicitly requests generated outputs.
        save_results(
            symbol=args.symbol,
            scenario_df=scenario_df,
            summary_df=summary_df,
            candidates_df=candidates_df,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
