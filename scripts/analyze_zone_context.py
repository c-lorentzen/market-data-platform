import argparse

import pandas as pd

from mdp.database.candle_repository import fetch_market_candles
from mdp.structure.support_resistance import detect_support_resistance_zones
from mdp.structure.zone_validation import evaluate_support_resistance_zones
from mdp.structure.zone_context import (
    add_zone_context,
    build_zone_decision_snapshot,
)


def parse_arguments():
    """
    Parses command-line arguments for zone context analysis.
    """
    parser = argparse.ArgumentParser(
        description="Analyze nearest evaluated support/resistance zones around the current price."
    )

    parser.add_argument(
        "--symbol",
        required=True,
        help='Market symbol, for example: "Step Index".',
    )

    parser.add_argument(
        "--timeframe",
        required=True,
        help="Timeframe, for example: M1, M5, M15, H1.",
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
        help="Minimum pivot touches required for a level.",
    )

    parser.add_argument(
        "--reaction-window",
        type=int,
        default=5,
        help="How many future candles to inspect after a touch.",
    )

    parser.add_argument(
        "--cooldown",
        type=int,
        default=3,
        help="How many candles to skip after counting a touch.",
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

    candles_df = fetch_market_candles(
        symbol=symbol,
        timeframe=timeframe,
        limit=args.limit,
    )

    if candles_df.empty:
        raise ValueError(f"No candles found for {symbol} {timeframe}.")

    current_price = float(candles_df.iloc[-1]["close_price"])

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

    evaluated_levels_df = evaluate_support_resistance_zones(
        candles_df=candles_df,
        levels_df=levels_df,
        reaction_window=args.reaction_window,
        cooldown=args.cooldown,
    )

    context_df = add_zone_context(
        evaluated_levels_df=evaluated_levels_df,
        current_price=current_price,
    )

    snapshot = build_zone_decision_snapshot(
        context_df=context_df,
        current_price=current_price,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    print("Zone Context Analysis")
    print("=" * 60)
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Current price: {current_price}")
    print(f"Candles analyzed: {len(candles_df)}")
    print(f"Zones evaluated: {len(context_df)}")

    print("\nDecision snapshot:")
    print(f"Stronger side: {snapshot['stronger_side']}")

    print("\nNearest support:")
    print(snapshot["nearest_support"])

    print("\nNearest resistance:")
    print(snapshot["nearest_resistance"])

    print("\nZones containing current price:")
    print(snapshot["zones_containing_price"])

    print("\nContext table:")
    print(context_df)


if __name__ == "__main__":
    main()