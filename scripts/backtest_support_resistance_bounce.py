import argparse

import pandas as pd

from mdp.backtesting.engine import run_backtest, summarize_backtest
from mdp.database.candle_loader import load_market_candles
from mdp.structure.support_resistance import detect_support_resistance_zones
from mdp.structure.zone_validation import evaluate_support_resistance_zones
from mdp.structure.zone_context import add_zone_context
from mdp.strategies.support_resistance_bounce import (
    build_support_resistance_bounce_signals,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Backtest a simple support/resistance bounce strategy."
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
    parser.add_argument("--tolerance-percent", type=float, default=0.001)
    parser.add_argument("--min-touches", type=int, default=2)

    parser.add_argument("--reaction-window", type=int, default=5)
    parser.add_argument("--cooldown", type=int, default=3)

    parser.add_argument("--auto-zone-width", action="store_true")
    parser.add_argument("--zone-width-multiplier", type=float, default=0.75)
    parser.add_argument("--min-zone-width-points", type=float, default=None)
    parser.add_argument("--max-zone-width-points", type=float, default=None)

    parser.add_argument("--min-respect-rate", type=float, default=0.55)
    parser.add_argument("--reward-risk-ratio", type=float, default=1.5)

    parser.add_argument(
        "--min-risk-points",
        type=float,
        default=0.0,
        help="Minimum allowed distance between entry and stop loss in price points.",
    )

    parser.add_argument("--max-holding-candles", type=int, default=20)

    parser.add_argument(
        "--allow-overlapping-trades",
        action="store_true",
        help="Allow multiple trades to be open at the same time.",
    )

    parser.add_argument(
        "--confirmation-mode",
        choices=["touch", "rejection_close"],
        default="rejection_close",
        help="Signal confirmation mode used by the strategy.",
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

    evaluated_levels_df = evaluate_support_resistance_zones(
        candles_df=candles_df,
        levels_df=levels_df,
        reaction_window=args.reaction_window,
        cooldown=args.cooldown,
    )

    current_price = float(candles_df.iloc[-1]["close_price"])

    context_df = add_zone_context(
        evaluated_levels_df=evaluated_levels_df,
        current_price=current_price,
    )

    # Build strategy signals using support/resistance zones and confirmation rules.
    signals_df = build_support_resistance_bounce_signals(
        candles_df=candles_df,
        context_df=context_df,
        min_respect_rate=args.min_respect_rate,
        reward_risk_ratio=args.reward_risk_ratio,
        min_risk_points=args.min_risk_points,
        confirmation_mode=args.confirmation_mode,
    )

    trades_df = run_backtest(
        candles_df=candles_df,
        signals_df=signals_df,
        max_holding_candles=args.max_holding_candles,
        allow_overlapping_trades=args.allow_overlapping_trades,
    )

    summary = summarize_backtest(trades_df)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    print("Support / Resistance Bounce Backtest")
    print("=" * 60)
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Candles analyzed: {len(candles_df)}")
    print(f"Levels detected: {len(levels_df)}")
    print(f"Levels evaluated: {len(evaluated_levels_df)}")
    print(f"Signals generated: {len(signals_df)}")
    print(f"Trades simulated: {len(trades_df)}")

    print("\nBacktest summary:")
    for key, value in summary.items():
        print(f"- {key}: {value}")

    print("\nSignals:")
    print(signals_df)

    print("\nTrades:")
    print(trades_df)


if __name__ == "__main__":
    main()