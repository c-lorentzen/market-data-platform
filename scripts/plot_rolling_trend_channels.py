import argparse
from pathlib import Path

import pandas as pd

from mdp.database.candle_loader import load_market_candles
from mdp.structure.rolling_trend_channels import (
    detect_rolling_trend_channels,
    parse_window_sizes,
    summarize_rolling_channels,
)
from mdp.structure.trend_channels import detect_swing_pivots
from mdp.visualization.rolling_trend_channel_charts import plot_rolling_trend_channel_chart


def parse_arguments():
    """
    Parses command-line arguments for rolling trend channel visualization.
    """
    parser = argparse.ArgumentParser(
        description="Plot rolling trend channels on a candlestick chart."
    )

    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", required=True)

    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--start-time", default=None)
    parser.add_argument("--end-time", default=None)
    parser.add_argument("--lookback-days", type=int, default=None)
    parser.add_argument("--lookback-weeks", type=int, default=None)
    parser.add_argument("--lookback-months", type=int, default=None)

    parser.add_argument(
        "--window-sizes",
        default="60,80,120,180",
        help='Comma-separated rolling window sizes, for example: "60,80,120".',
    )
    parser.add_argument("--step-size", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--max-channels", type=int, default=4)

    parser.add_argument("--left-window", type=int, default=3)
    parser.add_argument("--right-window", type=int, default=3)
    parser.add_argument("--envelope-quantile", type=float, default=0.90)
    parser.add_argument("--tolerance-range-multiplier", type=float, default=0.50)
    parser.add_argument("--min-anchor-pivots", type=int, default=3)

    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=0.0,
        help="Minimum original channel quality score required before ranking.",
    )
    parser.add_argument(
        "--min-rolling-quality-score",
        type=float,
        default=0.0,
        help="Minimum final rolling score required before plotting.",
    )
    parser.add_argument(
        "--min-r-squared",
        type=float,
        default=0.0,
        help="Minimum R-squared required after rolling detection. Useful for cleaner charts.",
    )
    parser.add_argument(
        "--max-channel-width-ratio",
        type=float,
        default=0.002,
        help="Maximum width ratio used to score narrow vs wide channels.",
    )
    parser.add_argument(
        "--min-width-to-range-ratio",
        type=float,
        default=1.0,
        help="Minimum channel width divided by median candle range for entry efficiency.",
    )
    parser.add_argument(
        "--max-width-to-range-ratio",
        type=float,
        default=8.0,
        help="Maximum channel width divided by median candle range for entry efficiency.",
    )
    parser.add_argument(
        "--recency-decay-candles",
        type=int,
        default=60,
        help="Number of candles after which older windows lose most recency score.",
    )
    parser.add_argument(
        "--active-only",
        action="store_true",
        help="Only evaluate the latest window for each window size. Best for entry refinement.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/charts",
        help="Folder where the chart image will be saved.",
    )

    return parser.parse_args()


def build_output_path(symbol: str, timeframe: str, output_dir: str) -> Path:
    """
    Builds a clean output image path for the rolling channel chart.
    """
    # Make the filename safe and readable.
    clean_symbol = symbol.lower().replace(" ", "_")
    clean_timeframe = timeframe.lower()

    return Path(output_dir) / f"{clean_symbol}_{clean_timeframe}_rolling_trend_channels.png"


def main():
    args = parse_arguments()

    symbol = args.symbol
    timeframe = args.timeframe.upper()

    # Load market candles using the shared project loader.
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

    # Parse rolling window sizes from the command line.
    window_sizes = parse_window_sizes(args.window_sizes)

    # Detect pivots for visual reference on the chart.
    pivots_df = detect_swing_pivots(
        candles_df=candles_df,
        left_window=args.left_window,
        right_window=args.right_window,
    )

    # Detect active or historical rolling trend channel candidates.
    rolling_channels_df = detect_rolling_trend_channels(
        candles_df=candles_df,
        window_sizes=window_sizes,
        step_size=args.step_size,
        top_n=args.top_n,
        left_window=args.left_window,
        right_window=args.right_window,
        envelope_quantile=args.envelope_quantile,
        tolerance_range_multiplier=args.tolerance_range_multiplier,
        min_anchor_pivots=args.min_anchor_pivots,
        min_quality_score=args.min_quality_score,
        min_rolling_quality_score=args.min_rolling_quality_score,
        max_channel_width_ratio=args.max_channel_width_ratio,
        min_width_to_range_ratio=args.min_width_to_range_ratio,
        max_width_to_range_ratio=args.max_width_to_range_ratio,
        recency_decay_candles=args.recency_decay_candles,
        active_only=args.active_only,
    )

    # Optionally remove weak-fit channels from the chart without changing the detector module yet.
    if args.min_r_squared > 0 and not rolling_channels_df.empty:
        rolling_channels_df = rolling_channels_df[
            rolling_channels_df["r_squared"].astype(float) >= args.min_r_squared
        ].reset_index(drop=True)

    # Rebuild IDs after filtering so the chart and report are easier to read.
    if not rolling_channels_df.empty:
        rolling_channels_df["rolling_channel_id"] = rolling_channels_df.index + 1

    output_path = build_output_path(
        symbol=symbol,
        timeframe=timeframe,
        output_dir=args.output_dir,
    )

    # Create the rolling channel candlestick chart.
    plot_rolling_trend_channel_chart(
        candles_df=candles_df,
        pivots_df=pivots_df,
        rolling_channels_df=rolling_channels_df,
        symbol=symbol,
        timeframe=timeframe,
        output_path=output_path,
        max_channels=args.max_channels,
    )

    # Print a terminal report so the chart can be verified with numbers.
    summary = summarize_rolling_channels(rolling_channels_df)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 320)

    print("Rolling Trend Channel Chart Report")
    print("=" * 80)
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Candles plotted: {len(candles_df)}")
    print(f"Pivots plotted: {len(pivots_df)}")
    print(f"Window sizes: {window_sizes}")
    print(f"Active only: {args.active_only}")
    print(f"Channels plotted: {min(len(rolling_channels_df), args.max_channels)}")
    print(f"Output file: {output_path}")

    print("\nSummary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nRolling channels used for chart:")
    print(rolling_channels_df)


if __name__ == "__main__":
    main()
