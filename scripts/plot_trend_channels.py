import argparse
from pathlib import Path

import pandas as pd

from mdp.database.candle_loader import load_market_candles
from mdp.structure.trend_channels import detect_swing_pivots, detect_trend_channels
from mdp.visualization.trend_channel_charts import plot_trend_channel_chart


def parse_arguments():
    """
    Parses command-line arguments for trend channel chart generation.
    """
    parser = argparse.ArgumentParser(
        description="Create a candlestick chart with detected trend channels."
    )

    parser.add_argument(
        "--symbol",
        required=True,
        help='Market symbol, for example: "Step Index" or "Volatility 75 Index".',
    )

    parser.add_argument(
        "--timeframe",
        required=True,
        help="Timeframe, for example: M1, M5, M15, H1, H4, D1.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=300,
        help="Number of recent candles to analyze when no date range is provided.",
    )

    parser.add_argument(
        "--start-time",
        default=None,
        help='Start datetime, for example: "2026-05-04 21:00:00".',
    )

    parser.add_argument(
        "--end-time",
        default=None,
        help='End datetime, for example: "2026-05-04 22:00:00".',
    )

    parser.add_argument(
        "--lookback-days",
        type=int,
        default=None,
        help="Calendar lookback in days.",
    )

    parser.add_argument(
        "--lookback-weeks",
        type=int,
        default=None,
        help="Calendar lookback in weeks.",
    )

    parser.add_argument(
        "--lookback-months",
        type=int,
        default=None,
        help="Calendar-aware lookback in months.",
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
        "--envelope-quantile",
        type=float,
        default=0.90,
        help="Quantile used to estimate channel width from opposite pivots.",
    )

    parser.add_argument(
        "--tolerance-range-multiplier",
        type=float,
        default=0.50,
        help="Multiplier applied to median candle range for channel tolerance.",
    )

    parser.add_argument(
        "--min-anchor-pivots",
        type=int,
        default=3,
        help="Minimum anchor pivots required to build a channel candidate.",
    )

    parser.add_argument(
        "--max-channels",
        type=int,
        default=2,
        help="Maximum number of ranked channels to draw.",
    )

    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional custom output path for the generated PNG chart.",
    )

    return parser.parse_args()


def build_output_path(symbol: str, timeframe: str) -> Path:
    """
    Builds a clean default output path for the trend channel chart.
    """
    # Convert names into a safe file-name format.
    clean_symbol = symbol.lower().replace(" ", "_")
    clean_timeframe = timeframe.lower()

    return (
        Path("outputs")
        / "charts"
        / f"{clean_symbol}_{clean_timeframe}_trend_channels.png"
    )


def main():
    args = parse_arguments()

    symbol = args.symbol
    timeframe = args.timeframe.upper()

    # Load candles using the same flexible loader used by other project scripts.
    candles_df = load_market_candles(
        symbol=symbol,
        timeframe=timeframe,
        limit=args.limit,
        start_time=args.start_time,
        end_time=args.end_time,
        lookback_days=args.lookback_days,
        lookback_weeks=args.lookback_weeks,
        lookback_months=args.lookback_months,
    )

    if candles_df.empty:
        raise ValueError(f"No candles found for {symbol} {timeframe}.")

    # Detect pivots so the chart can show the anchor structure visually.
    pivots_df = detect_swing_pivots(
        candles_df=candles_df,
        left_window=args.left_window,
        right_window=args.right_window,
    )

    # Detect trend channels using the current v1 mathematical detector.
    channels_df = detect_trend_channels(
        candles_df=candles_df,
        left_window=args.left_window,
        right_window=args.right_window,
        envelope_quantile=args.envelope_quantile,
        tolerance_range_multiplier=args.tolerance_range_multiplier,
        min_anchor_pivots=args.min_anchor_pivots,
    )

    # Use a default output path unless the user provides a custom one.
    if args.output_path is None:
        output_path = build_output_path(symbol=symbol, timeframe=timeframe)
    else:
        output_path = Path(args.output_path)

    # Create and save the visualization.
    plot_trend_channel_chart(
        candles_df=candles_df,
        pivots_df=pivots_df,
        channels_df=channels_df,
        symbol=symbol,
        timeframe=timeframe,
        output_path=output_path,
        max_channels=args.max_channels,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    print("Trend channel chart created successfully.")
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Candles plotted: {len(candles_df)}")
    print(f"Pivots plotted: {len(pivots_df)}")
    print(f"Channels detected: {len(channels_df)}")
    print(f"Channels plotted: {min(len(channels_df), args.max_channels)}")
    print(f"Output file: {output_path}")

    print("\nDetected channels:")
    print(channels_df)


if __name__ == "__main__":
    main()
