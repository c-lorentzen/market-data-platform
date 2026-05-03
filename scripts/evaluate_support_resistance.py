import argparse

import pandas as pd

from mdp.database.candle_repository import fetch_market_candles
from mdp.structure.support_resistance import detect_support_resistance_zones
from mdp.structure.zone_validation import evaluate_support_resistance_zones


def parse_arguments():
    """
    Parses command-line arguments for support/resistance evaluation.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate historical respect/break behavior of support/resistance zones."
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
        help="Minimum touches required for a level to be returned.",
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

    levels_df = detect_support_resistance_zones(
        candles_df=candles_df,
        left_window=args.left_window,
        right_window=args.right_window,
        tolerance_percent=args.tolerance_percent,
        min_touches=args.min_touches,
    )

    evaluation_df = evaluate_support_resistance_zones(
        candles_df=candles_df,
        levels_df=levels_df,
        reaction_window=args.reaction_window,
        cooldown=args.cooldown,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("Support / Resistance Historical Evaluation")
    print("=" * 60)
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Candles analyzed: {len(candles_df)}")
    print(f"Levels detected: {len(levels_df)}")
    print(f"Reaction window: {args.reaction_window}")
    print(f"Cooldown: {args.cooldown}")

    print("\nEvaluation results:")
    print(evaluation_df)


if __name__ == "__main__":
    main()