import argparse
from pathlib import Path

import pandas as pd

from mdp.database.candle_repository import fetch_market_candles
from mdp.structure.support_resistance import detect_support_resistance_zones
from mdp.visualization.price_charts import plot_candlestick_with_levels


def parse_arguments():
    """
    Parses command-line arguments for support/resistance chart generation.
    """
    parser = argparse.ArgumentParser(
        description="Create a candlestick chart with support/resistance zones."
    )

    parser.add_argument(
        "--symbol",
        required=True,
        help='Market symbol, for example: "Step Index".',
    )

    parser.add_argument(
        "--timeframe",
        required=True,
        help="Timeframe, for example: M15, H1, H4.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=300,
        help="Number of recent candles to analyze.",
    )

    parser.add_argument(
        "--left-window",
        type=int,
        default=3,
        help="Number of candles to the left of a pivot.",
    )

    parser.add_argument(
        "--right-window",
        type=int,
        default=3,
        help="Number of candles to the right of a pivot.",
    )

    parser.add_argument(
        "--tolerance-percent",
        type=float,
        default=0.001,
        help="Price distance tolerance used to group nearby levels.",
    )

    parser.add_argument(
        "--min-touches",
        type=int,
        default=2,
        help="Minimum touches required for a level to be plotted.",
    )

    parser.add_argument(
        "--max-levels",
        type=int,
        default=10,
        help="Maximum number of levels to plot.",
    )

    return parser.parse_args()


def build_output_path(symbol: str, timeframe: str) -> Path:
    """
    Builds a clean output path for the generated chart image.
    """
    clean_symbol = symbol.lower().replace(" ", "_")
    clean_timeframe = timeframe.lower()

    return (
        Path("outputs")
        / "charts"
        / f"{clean_symbol}_{clean_timeframe}_support_resistance.png"
    )


def main():
    args = parse_arguments()

    symbol = args.symbol
    timeframe = args.timeframe.upper()

    candles_df = fetch_market_candles(
        symbol=symbol,
        timeframe=timeframe,
        limit=args.limit,
    )

    if candles_df.empty:
        raise ValueError(f"No candles found for {symbol} {timeframe}.")

    levels_df = detect_support_resistance_zones(
        candles_df=candles_df,
        left_window=args.left_window,
        right_window=args.right_window,
        tolerance_percent=args.tolerance_percent,
        min_touches=args.min_touches,
    )

    output_path = build_output_path(symbol, timeframe)

    plot_candlestick_with_levels(
        candles_df=candles_df,
        levels_df=levels_df,
        symbol=symbol,
        timeframe=timeframe,
        output_path=output_path,
        max_levels=args.max_levels,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)

    print("Support / Resistance chart created successfully.")
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Candles plotted: {len(candles_df)}")
    print(f"Levels plotted: {min(len(levels_df), args.max_levels)}")
    print(f"Output file: {output_path}")

    print("\nLevels used:")
    print(levels_df.head(args.max_levels))


if __name__ == "__main__":
    main()