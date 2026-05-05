from collections.abc import Callable
from typing import Any

import pandas as pd

from mdp.backtesting.engine import run_backtest, summarize_backtest


def build_walk_forward_windows(
    candles_df: pd.DataFrame,
    train_window_candles: int,
    test_window_candles: int,
    step_size_candles: int,
) -> list[dict]:
    """
    Builds walk-forward train/test windows from a candle DataFrame.
    """
    # Validate that all window sizes are positive.
    if train_window_candles <= 0:
        raise ValueError("train_window_candles must be greater than 0.")

    if test_window_candles <= 0:
        raise ValueError("test_window_candles must be greater than 0.")

    if step_size_candles <= 0:
        raise ValueError("step_size_candles must be greater than 0.")

    # Copy and sort candles so each window follows chronological order.
    market_df = candles_df.copy()
    market_df["timestamp_utc"] = pd.to_datetime(market_df["timestamp_utc"])
    market_df = market_df.sort_values("timestamp_utc").reset_index(drop=True)

    windows = []
    window_id = 1
    start_index = 0

    # Keep building windows while a full train + test block fits in the dataset.
    while start_index + train_window_candles + test_window_candles <= len(market_df):
        train_start_index = start_index
        train_end_index = start_index + train_window_candles

        test_start_index = train_end_index
        test_end_index = test_start_index + test_window_candles

        # Slice the train and test data for this window.
        train_df = market_df.iloc[train_start_index:train_end_index].copy()
        test_df = market_df.iloc[test_start_index:test_end_index].copy()

        windows.append(
            {
                "window_id": window_id,
                "train_start": train_df["timestamp_utc"].min(),
                "train_end": train_df["timestamp_utc"].max(),
                "test_start": test_df["timestamp_utc"].min(),
                "test_end": test_df["timestamp_utc"].max(),
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "train_df": train_df,
                "test_df": test_df,
            }
        )

        # Move forward by the selected step size.
        start_index += step_size_candles
        window_id += 1

    return windows


def run_walk_forward_backtest(
    candles_df: pd.DataFrame,
    build_signals_for_window: Callable[..., tuple[pd.DataFrame, dict[str, Any]]],
    train_window_candles: int,
    test_window_candles: int,
    step_size_candles: int,
    max_holding_candles: int = 20,
    allow_overlapping_trades: bool = False,
    strategy_kwargs: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Runs a generic walk-forward backtest.

    The strategy logic is injected through build_signals_for_window,
    so this engine can support many different strategies later.
    """
    # Use an empty dictionary when no strategy parameters are provided.
    if strategy_kwargs is None:
        strategy_kwargs = {}

    # Build chronological train/test windows.
    windows = build_walk_forward_windows(
        candles_df=candles_df,
        train_window_candles=train_window_candles,
        test_window_candles=test_window_candles,
        step_size_candles=step_size_candles,
    )

    window_rows = []
    all_signals = []
    all_trades = []

    for window in windows:
        window_id = window["window_id"]
        train_df = window["train_df"]
        test_df = window["test_df"]

        # Ask the selected strategy to build signals using train data and test data.
        signals_df, diagnostics = build_signals_for_window(
            train_df=train_df,
            test_df=test_df,
            window_id=window_id,
            **strategy_kwargs,
        )

        # Run the normal backtest engine only on the test window.
        trades_df = run_backtest(
            candles_df=test_df,
            signals_df=signals_df,
            max_holding_candles=max_holding_candles,
            allow_overlapping_trades=allow_overlapping_trades,
        )

        # Add window_id to signals so we can trace where each signal came from.
        if signals_df is not None and not signals_df.empty:
            signals_df = signals_df.copy()
            signals_df["window_id"] = window_id
            all_signals.append(signals_df)

        # Add window_id to trades so we can analyze performance by window.
        if trades_df is not None and not trades_df.empty:
            trades_df = trades_df.copy()
            trades_df["window_id"] = window_id
            all_trades.append(trades_df)

        # Store compact window diagnostics for reporting.
        window_rows.append(
            {
                "window_id": window_id,
                "train_start": window["train_start"],
                "train_end": window["train_end"],
                "test_start": window["test_start"],
                "test_end": window["test_end"],
                "train_rows": window["train_rows"],
                "test_rows": window["test_rows"],
                "signals_generated": 0 if signals_df is None else len(signals_df),
                "trades_simulated": 0 if trades_df is None else len(trades_df),
                **diagnostics,
            }
        )

    # Convert window diagnostics into a DataFrame.
    windows_df = pd.DataFrame(window_rows)

    # Combine all signals from all windows.
    signals_output_df = (
        pd.concat(all_signals, ignore_index=True)
        if all_signals
        else pd.DataFrame()
    )

    # Combine all trades from all windows.
    trades_output_df = (
        pd.concat(all_trades, ignore_index=True)
        if all_trades
        else pd.DataFrame()
    )

    return {
        "windows_df": windows_df,
        "signals_df": signals_output_df,
        "trades_df": trades_output_df,
    }

def summarize_walk_forward_results(
    trades_df: pd.DataFrame,
    windows_df: pd.DataFrame,
) -> dict:
    """
    Summarizes walk-forward performance across all windows.

    This separates:
    - all walk-forward windows
    - windows that actually generated trades
    - profitable active windows
    """
    # Reuse the normal backtest summary for trade-level metrics.
    summary = summarize_backtest(trades_df)

    # Count how many walk-forward windows were created.
    total_windows = 0 if windows_df.empty else len(windows_df)

    # Store total window count in the final summary.
    summary["total_windows"] = total_windows

    # If there are no windows, there is nothing else to summarize.
    if windows_df.empty:
        summary["active_trade_windows"] = 0
        summary["no_trade_windows"] = 0
        summary["active_window_rate"] = None
        summary["profitable_active_windows"] = 0
        summary["profitable_active_window_rate"] = None
        summary["profitable_all_window_rate"] = None
        summary["average_r_per_active_window"] = None
        return summary

    # Count windows where at least one trade was simulated.
    active_trade_windows = int((windows_df["trades_simulated"] > 0).sum())

    # Count windows where no trades happened.
    no_trade_windows = total_windows - active_trade_windows

    # Calculate how often the strategy was active.
    active_window_rate = active_trade_windows / total_windows

    # Store activity metrics in the summary.
    summary["active_trade_windows"] = active_trade_windows
    summary["no_trade_windows"] = no_trade_windows
    summary["active_window_rate"] = active_window_rate

    # If there are no trades, profitability metrics cannot be calculated.
    if trades_df.empty:
        summary["profitable_active_windows"] = 0
        summary["profitable_active_window_rate"] = None
        summary["profitable_all_window_rate"] = None
        summary["average_r_per_active_window"] = None
        return summary

    # Use only trades with valid R multiples for performance calculations.
    closed_df = trades_df[trades_df["r_multiple"].notna()].copy()

    # If no trades have R results, profitability metrics cannot be calculated.
    if closed_df.empty:
        summary["profitable_active_windows"] = 0
        summary["profitable_active_window_rate"] = None
        summary["profitable_all_window_rate"] = None
        summary["average_r_per_active_window"] = None
        return summary

    # Calculate total R per active walk-forward window.
    window_r_df = (
        closed_df.groupby("window_id")["r_multiple"]
        .sum()
        .reset_index(name="window_total_r")
    )

    # Count active windows with positive total R.
    profitable_active_windows = int((window_r_df["window_total_r"] > 0).sum())

    # Calculate profitability among active windows only.
    profitable_active_window_rate = (
        profitable_active_windows / active_trade_windows
        if active_trade_windows > 0
        else None
    )

    # Calculate profitability across all windows, including no-trade windows.
    profitable_all_window_rate = (
        profitable_active_windows / total_windows
        if total_windows > 0
        else None
    )

    # Store window-level profitability metrics.
    summary["profitable_active_windows"] = profitable_active_windows
    summary["profitable_active_window_rate"] = profitable_active_window_rate
    summary["profitable_all_window_rate"] = profitable_all_window_rate
    summary["average_r_per_active_window"] = window_r_df["window_total_r"].mean()

    return summary