import argparse
from pathlib import Path

import pandas as pd

from mdp.backtesting.engine import run_backtest, summarize_backtest
from mdp.database.candle_loader import load_market_candles
from mdp.strategies.support_resistance_bounce import (
    build_support_resistance_bounce_signals,
)
from mdp.structure.support_resistance import detect_support_resistance_zones
from mdp.structure.zone_context import add_zone_context
from mdp.structure.zone_validation import evaluate_support_resistance_zones
from mdp.visualization.backtest_charts import plot_backtest_trades


def parse_arguments():
    """
    Parses command-line arguments for plotting a support/resistance bounce backtest.
    """
    parser = argparse.ArgumentParser(
        description="Plot a support/resistance bounce backtest on a candlestick chart."
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
    parser.add_argument("--min-risk-points", type=float, default=0.0)
    parser.add_argument("--max-holding-candles", type=int, default=20)

    parser.add_argument(
        "--confirmation-mode",
        choices=["touch", "rejection_close"],
        default="rejection_close",
        help="Signal confirmation mode used by the strategy.",
    )

    parser.add_argument(
        "--allow-overlapping-trades",
        action="store_true",
        help="Allow multiple trades to be open at the same time.",
    )

    parser.add_argument(
        "--max-levels",
        type=int,
        default=10,
        help="Maximum number of evaluated zones to draw.",
    )

    return parser.parse_args()


def build_output_path(symbol: str, timeframe: str) -> Path:
    """
    Builds a clean output path for the backtest chart.
    """
    # Make the symbol safe for filenames.
    clean_symbol = symbol.lower().replace(" ", "_")

    # Make the timeframe safe for filenames.
    clean_timeframe = timeframe.lower()

    return (
        Path("outputs")
        / "charts"
        / f"{clean_symbol}_{clean_timeframe}_sr_bounce_backtest.png"
    )


def main():
    args = parse_arguments()

    symbol = args.symbol
    timeframe = args.timeframe.upper()

    # Load candles by either recent limit or explicit date range.
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

    # Detect support/resistance zones from the selected candle sample.
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

    # Evaluate how the detected zones behaved historically.
    evaluated_levels_df = evaluate_support_resistance_zones(
        candles_df=candles_df,
        levels_df=levels_df,
        reaction_window=args.reaction_window,
        cooldown=args.cooldown,
    )

    # Use the latest close price to add current-price context to each zone.
    current_price = float(candles_df.iloc[-1]["close_price"])

    context_df = add_zone_context(
        evaluated_levels_df=evaluated_levels_df,
        current_price=current_price,
    )

    # Build strategy signals using the selected strategy filters.
    signals_df = build_support_resistance_bounce_signals(
        candles_df=candles_df,
        context_df=context_df,
        min_respect_rate=args.min_respect_rate,
        reward_risk_ratio=args.reward_risk_ratio,
        min_risk_points=args.min_risk_points,
        confirmation_mode=args.confirmation_mode,
    )

    # Run the backtest engine on the generated signals.
    trades_df = run_backtest(
        candles_df=candles_df,
        signals_df=signals_df,
        max_holding_candles=args.max_holding_candles,
        allow_overlapping_trades=args.allow_overlapping_trades,
    )

    # Summarize results for quick terminal review.
    summary = summarize_backtest(trades_df)

    output_path = build_output_path(symbol, timeframe)

    # Draw candles, zones, entries, exits, SL, and TP into one chart.
    plot_backtest_trades(
        candles_df=candles_df,
        levels_df=evaluated_levels_df,
        trades_df=trades_df,
        symbol=symbol,
        timeframe=timeframe,
        output_path=output_path,
        max_levels=args.max_levels,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    print("Support / Resistance Bounce Backtest Chart")
    print("=" * 60)
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Candles plotted: {len(candles_df)}")
    print(f"Signals generated: {len(signals_df)}")
    print(f"Trades plotted: {len(trades_df)}")
    print(f"Output file: {output_path}")

    print("\nBacktest summary:")
    for key, value in summary.items():
        print(f"- {key}: {value}")

    print("\nTrades plotted:")
    print(trades_df)


if __name__ == "__main__":
    main()