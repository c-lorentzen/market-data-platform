import argparse

import pandas as pd

from mdp.backtesting.walk_forward import (
    run_walk_forward_backtest,
    summarize_walk_forward_results,
)
from mdp.database.candle_loader import load_market_candles
from mdp.strategies.support_resistance_bounce import (
    build_support_resistance_bounce_signals,
)
from mdp.structure.support_resistance import detect_support_resistance_zones
from mdp.structure.zone_context import add_zone_context
from mdp.structure.zone_validation import evaluate_support_resistance_zones


def parse_arguments():
    """
    Parses command-line arguments for walk-forward SR bounce backtesting.
    """
    parser = argparse.ArgumentParser(
        description="Run walk-forward validation for the support/resistance bounce strategy."
    )

    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", required=True)

    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--start-time", default=None)
    parser.add_argument("--end-time", default=None)
    parser.add_argument("--lookback-days", type=int, default=None)
    parser.add_argument("--lookback-weeks", type=int, default=None)
    parser.add_argument("--lookback-months", type=int, default=None)

    parser.add_argument("--train-window-candles", type=int, default=180)
    parser.add_argument("--test-window-candles", type=int, default=60)
    parser.add_argument("--step-size-candles", type=int, default=60)

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
    parser.add_argument("--min-risk-points", type=float, default=0.5)
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
        help="Allow overlapping trades during the walk-forward test.",
    )

    return parser.parse_args()


def build_sr_bounce_signals_for_window(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    window_id: int,
    left_window: int,
    right_window: int,
    tolerance_percent: float,
    min_touches: int,
    reaction_window: int,
    cooldown: int,
    auto_zone_width: bool,
    zone_width_multiplier: float,
    min_zone_width_points: float | None,
    max_zone_width_points: float | None,
    min_respect_rate: float,
    reward_risk_ratio: float,
    min_risk_points: float,
    confirmation_mode: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Builds SR bounce signals for one walk-forward window.

    Important:
    - Zones are detected only from train_df.
    - Signals are generated only on test_df.
    """
    # Detect support/resistance levels using only historical training data.
    levels_df = detect_support_resistance_zones(
        candles_df=train_df,
        left_window=left_window,
        right_window=right_window,
        tolerance_percent=tolerance_percent,
        min_touches=min_touches,
        max_zone_width_points=max_zone_width_points,
        auto_zone_width=auto_zone_width,
        zone_width_multiplier=zone_width_multiplier,
        min_zone_width_points=min_zone_width_points,
    )

    # If no zones are detected, this window cannot generate signals.
    if levels_df.empty:
        diagnostics = {
            "levels_detected": 0,
            "levels_evaluated": 0,
        }
        return pd.DataFrame(), diagnostics

    # Evaluate zone behavior using only the same training window.
    evaluated_levels_df = evaluate_support_resistance_zones(
        candles_df=train_df,
        levels_df=levels_df,
        reaction_window=reaction_window,
        cooldown=cooldown,
    )

    # If no zones survive evaluation, this window cannot generate signals.
    if evaluated_levels_df.empty:
        diagnostics = {
            "levels_detected": len(levels_df),
            "levels_evaluated": 0,
        }
        return pd.DataFrame(), diagnostics

    # Use the latest train close as the context price before the test window starts.
    current_price = float(train_df.iloc[-1]["close_price"])

    # Build contextual zone features without using future test outcomes.
    context_df = add_zone_context(
        evaluated_levels_df=evaluated_levels_df,
        current_price=current_price,
    )

    # Generate strategy signals on the future test window only.
    signals_df = build_support_resistance_bounce_signals(
        candles_df=test_df,
        context_df=context_df,
        min_respect_rate=min_respect_rate,
        reward_risk_ratio=reward_risk_ratio,
        min_risk_points=min_risk_points,
        confirmation_mode=confirmation_mode,
    )

    # Store useful diagnostics for this window.
    diagnostics = {
        "levels_detected": len(levels_df),
        "levels_evaluated": len(evaluated_levels_df),
    }

    return signals_df, diagnostics


def main():
    args = parse_arguments()

    symbol = args.symbol
    timeframe = args.timeframe.upper()

    # Load candles by limit or by explicit date range.
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

    # Collect all strategy parameters in one dictionary.
    strategy_kwargs = {
        "left_window": args.left_window,
        "right_window": args.right_window,
        "tolerance_percent": args.tolerance_percent,
        "min_touches": args.min_touches,
        "reaction_window": args.reaction_window,
        "cooldown": args.cooldown,
        "auto_zone_width": args.auto_zone_width,
        "zone_width_multiplier": args.zone_width_multiplier,
        "min_zone_width_points": args.min_zone_width_points,
        "max_zone_width_points": args.max_zone_width_points,
        "min_respect_rate": args.min_respect_rate,
        "reward_risk_ratio": args.reward_risk_ratio,
        "min_risk_points": args.min_risk_points,
        "confirmation_mode": args.confirmation_mode,
    }

    # Run walk-forward validation using the SR bounce signal builder.
    results = run_walk_forward_backtest(
        candles_df=candles_df,
        build_signals_for_window=build_sr_bounce_signals_for_window,
        train_window_candles=args.train_window_candles,
        test_window_candles=args.test_window_candles,
        step_size_candles=args.step_size_candles,
        max_holding_candles=args.max_holding_candles,
        allow_overlapping_trades=args.allow_overlapping_trades,
        strategy_kwargs=strategy_kwargs,
    )

    windows_df = results["windows_df"]
    signals_df = results["signals_df"]
    trades_df = results["trades_df"]

    # Summarize out-of-sample walk-forward performance.
    summary = summarize_walk_forward_results(
        trades_df=trades_df,
        windows_df=windows_df,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)

    print("Walk-Forward Support / Resistance Bounce Backtest")
    print("=" * 70)
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Candles loaded: {len(candles_df)}")
    print(f"Train window candles: {args.train_window_candles}")
    print(f"Test window candles: {args.test_window_candles}")
    print(f"Step size candles: {args.step_size_candles}")

    print("\nWalk-forward summary:")
    for key, value in summary.items():
        print(f"- {key}: {value}")

    print("\nWindow diagnostics:")
    print(windows_df)

    print("\nSignals:")
    print(signals_df)

    print("\nTrades:")
    print(trades_df)


if __name__ == "__main__":
    main()