import argparse
from pathlib import Path

import pandas as pd

from mdp.database.candle_loader import load_market_candles
from mdp.structure.rolling_trend_channels import (
    detect_rolling_trend_channels,
    parse_window_sizes,
    summarize_rolling_channels,
)


def parse_arguments():
    """
    Parses command-line arguments for rolling trend channel detection.
    """
    parser = argparse.ArgumentParser(
        description="Detect trend channels across rolling candle windows."
    )

    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", required=True)

    parser.add_argument("--limit", type=int, default=600)
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
        help="Minimum final rolling score required after ranking components are combined.",
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
        "--save-results",
        action="store_true",
        help="Save rolling channel candidates to a CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/rolling_channels",
        help="Output folder used when --save-results is enabled.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    symbol = args.symbol
    timeframe = args.timeframe.upper()

    # Load candles from the project database using the shared loader.
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

    # Convert the command-line string into real integer window sizes.
    window_sizes = parse_window_sizes(args.window_sizes)

    # Detect ranked channels across several rolling/segmented windows.
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

    # Build a compact decision-oriented summary.
    summary = summarize_rolling_channels(rolling_channels_df)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 320)

    print("Rolling Trend Channel Detection Report")
    print("=" * 80)
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Candles analyzed: {len(candles_df)}")
    print(f"Window sizes: {window_sizes}")
    print(f"Step size: {args.step_size}")
    print(f"Top N returned: {args.top_n}")
    print(f"Active only: {args.active_only}")

    print("\nSummary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nRolling channels detected:")
    print(rolling_channels_df)

    if args.save_results:
        # Create the output folder only when saving is requested.
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build a clean filename from symbol and timeframe.
        clean_symbol = symbol.lower().replace(" ", "_")
        output_path = output_dir / f"{clean_symbol}_{timeframe.lower()}_rolling_channels.csv"

        rolling_channels_df.to_csv(output_path, index=False)
        print(f"\nSaved rolling channel results to: {output_path}")


if __name__ == "__main__":
    main()
