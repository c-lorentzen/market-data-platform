import argparse
from pathlib import Path

import pandas as pd

from mdp.database.candle_loader import load_market_candles
from mdp.structure.market_scale import (
    build_market_scale_table,
    parse_timeframes,
)


def parse_arguments():
    """
    Parses command-line options for adaptive market scale analysis.
    """
    parser = argparse.ArgumentParser(
        description="Analyze adaptive market scale values across timeframes."
    )

    parser.add_argument(
        "--symbol",
        required=True,
        help='Market symbol, for example: "Step Index" or "Volatility 75 Index".',
    )
    parser.add_argument(
        "--timeframes",
        default="D1,H4,H1,M15,M5,M1",
        help='Comma or space-separated timeframes, for example: "H1,M1".',
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=300,
        help="Number of candles to load per timeframe.",
    )
    parser.add_argument(
        "--recent-window",
        type=int,
        default=None,
        help="Optional override for the recent market-stage window.",
    )
    parser.add_argument(
        "--atr-window",
        type=int,
        default=None,
        help="Optional override for the ATR calculation window.",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save the scale table as a CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/market_scale",
        help="Folder used when saving CSV outputs.",
    )

    return parser.parse_args()


def load_timeframe_candles(symbol: str, timeframes: list[str], limit: int) -> dict[str, pd.DataFrame]:
    """
    Loads candle data for each requested timeframe.
    """
    timeframe_to_candles: dict[str, pd.DataFrame] = {}

    # Load each timeframe independently so missing data does not stop the full report.
    for timeframe in timeframes:
        candles_df = load_market_candles(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
        )
        timeframe_to_candles[timeframe] = candles_df

    return timeframe_to_candles


def print_scale_summary(scale_df: pd.DataFrame) -> None:
    """
    Prints a compact scale summary across all analyzed timeframes.
    """
    print("\nAdaptive Scale Summary")
    print("-" * 90)

    if scale_df.empty:
        print("No scale rows were created.")
        return

    display_columns = [
        "timeframe",
        "purpose",
        "status",
        "candles_analyzed",
        "latest_close",
        "volatility_stage",
        "recent_median_candle_range",
        "recent_atr",
        "volatility_expansion_ratio",
        "channel_min_width",
        "channel_max_width",
        "channel_max_width_ratio",
    ]

    existing_columns = [column for column in display_columns if column in scale_df.columns]
    print(scale_df[existing_columns])


def print_scale_thresholds(scale_df: pd.DataFrame) -> None:
    """
    Prints practical thresholds that future modules can reuse.
    """
    print("\nAdaptive Thresholds for Future Modules")
    print("-" * 90)

    if scale_df.empty:
        print("No adaptive thresholds available.")
        return

    display_columns = [
        "timeframe",
        "purpose",
        "support_resistance_zone_width",
        "order_block_zone_buffer",
        "breakout_threshold",
        "retest_tolerance",
        "stop_buffer",
        "min_target_distance",
        "suggested_min_r_squared",
        "suggested_min_containment_rate",
    ]

    existing_columns = [column for column in display_columns if column in scale_df.columns]
    print(scale_df[existing_columns])


def save_results(symbol: str, scale_df: pd.DataFrame, output_dir: str) -> None:
    """
    Saves the adaptive scale table as a CSV file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Keep the file name clean and symbol-specific.
    clean_symbol = symbol.lower().replace(" ", "_")
    scale_path = output_path / f"{clean_symbol}_market_scale.csv"

    scale_df.to_csv(scale_path, index=False)

    print("\nSaved results:")
    print(f"Scale table: {scale_path}")


def main():
    args = parse_arguments()

    symbol = args.symbol
    timeframes = parse_timeframes(args.timeframes)

    # Load all requested timeframe datasets from the database.
    timeframe_to_candles = load_timeframe_candles(
        symbol=symbol,
        timeframes=timeframes,
        limit=args.limit,
    )

    # Calculate adaptive scale values for every timeframe independently.
    scale_df = build_market_scale_table(
        timeframe_to_candles=timeframe_to_candles,
        recent_window=args.recent_window,
        atr_window=args.atr_window,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 320)

    print("Adaptive Market Scale Report")
    print("=" * 90)
    print(f"Symbol: {symbol}")
    print(f"Timeframes: {timeframes}")
    print(f"Limit per timeframe: {args.limit}")
    print(f"Recent window override: {args.recent_window}")
    print(f"ATR window override: {args.atr_window}")

    print_scale_summary(scale_df)
    print_scale_thresholds(scale_df)

    if args.save_results:
        # Save outputs only when explicitly requested.
        save_results(
            symbol=symbol,
            scale_df=scale_df,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
