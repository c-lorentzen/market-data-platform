import argparse

import pandas as pd

from mdp.database.candle_loader import load_market_candles
from mdp.structure.support_resistance import (
    detect_support_resistance_zones,
    find_nearest_levels,
)


def parse_arguments():
    """
    Parses command-line arguments for support/resistance detection.
    """
    parser = argparse.ArgumentParser(
        description="Detect support and resistance zones from historical candles."
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
        "--tolerance-percent",
        type=float,
        default=0.001,
        help="Price distance tolerance used to group nearby levels.",
    )

    parser.add_argument(
        "--min-touches",
        type=int,
        default=2,
        help="Minimum touches required for a level to be returned.",
    )

    parser.add_argument(
        "--max-zone-width-points",
        type=float,
        default=None,
        help="Optional maximum half-width of a zone in price points.",
    )

    parser.add_argument(
        "--auto-zone-width",
        action="store_true",
        help="Automatically calculate zone width based on candle range.",
    )

    parser.add_argument(
        "--zone-width-multiplier",
        type=float,
        default=0.75,
        help="Multiplier applied to the median candle range when using auto zone width.",
    )

    parser.add_argument(
        "--min-zone-width-points",
        type=float,
        default=None,
        help="Optional minimum half-width of a zone in price points.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    symbol = args.symbol
    timeframe = args.timeframe.upper()

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

    levels_df = detect_support_resistance_zones(
        candles_df=candles_df,
        left_window=args.left_window,
        right_window=args.right_window,
        tolerance_percent=args.tolerance_percent,
        min_touches=args.min_touches,
        max_zone_width_points=args.max_zone_width_points,
        auto_zone_width=args.auto_zone_width,
        zone_width_multiplier=args.zone_width_multiplier,
        min_zone_width_points=args.min_zone_width_points,
    )

    current_price = float(candles_df.iloc[-1]["close_price"])

    nearest_levels = find_nearest_levels(
        levels_df=levels_df,
        current_price=current_price,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)

    print("Support / Resistance Detection Report")
    print("=" * 50)
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Candles analyzed: {len(candles_df)}")
    print(f"Current price: {current_price}")
    print(f"Levels detected: {len(levels_df)}")

    print("\nDetected levels:")
    print(levels_df)

    print("\nNearest support:")
    print(nearest_levels["nearest_support"])

    print("\nNearest resistance:")
    print(nearest_levels["nearest_resistance"])


if __name__ == "__main__":
    main()