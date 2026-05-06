import argparse

import pandas as pd

from mdp.database.candle_loader import load_market_candles
from mdp.structure.trend_channels import detect_swing_pivots, detect_trend_channels


def parse_arguments():
    """
    Parses command-line arguments for trend channel detection.
    """
    parser = argparse.ArgumentParser(
        description="Detect trend channels from OHLC market candles."
    )

    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", required=True)

    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--start-time", default=None)
    parser.add_argument("--end-time", default=None)
    parser.add_argument("--lookback-days", type=int, default=None)
    parser.add_argument("--lookback-weeks", type=int, default=None)
    parser.add_argument("--lookback-months", type=int, default=None)

    parser.add_argument("--left-window", type=int, default=3)
    parser.add_argument("--right-window", type=int, default=3)
    parser.add_argument("--envelope-quantile", type=float, default=0.90)
    parser.add_argument("--tolerance-range-multiplier", type=float, default=0.50)
    parser.add_argument("--min-anchor-pivots", type=int, default=3)

    return parser.parse_args()


def main():
    args = parse_arguments()

    symbol = args.symbol
    timeframe = args.timeframe.upper()

    # Load market candles using either limit or date range.
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

    # Detect swing pivots for diagnostics.
    pivots_df = detect_swing_pivots(
        candles_df=candles_df,
        left_window=args.left_window,
        right_window=args.right_window,
    )

    # Detect trend channel candidates.
    channels_df = detect_trend_channels(
        candles_df=candles_df,
        left_window=args.left_window,
        right_window=args.right_window,
        envelope_quantile=args.envelope_quantile,
        tolerance_range_multiplier=args.tolerance_range_multiplier,
        min_anchor_pivots=args.min_anchor_pivots,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    print("Trend Channel Detection Report")
    print("=" * 70)
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Candles analyzed: {len(candles_df)}")
    print(f"Pivots detected: {len(pivots_df)}")

    if not pivots_df.empty:
        print("\nPivot counts:")
        print(pivots_df["pivot_type"].value_counts())

    print("\nChannels detected:")
    print(channels_df)


if __name__ == "__main__":
    main()