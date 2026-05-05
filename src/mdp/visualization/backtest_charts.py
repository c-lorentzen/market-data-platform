from pathlib import Path

import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
from matplotlib.patches import Rectangle


def _find_nearest_x_position(chart_index: pd.DatetimeIndex, timestamp) -> int | None:
    """
    Finds the closest x-axis position for a timestamp in the candle chart.
    """
    # Convert the incoming timestamp into a pandas Timestamp.
    target_time = pd.to_datetime(timestamp)

    # Return None if the timestamp is missing.
    if pd.isna(target_time):
        return None

    # Find the candle timestamp closest to the requested trade timestamp.
    nearest_position = chart_index.get_indexer([target_time], method="nearest")[0]

    # Return None if pandas could not find a valid nearby timestamp.
    if nearest_position == -1:
        return None

    return int(nearest_position)


def _draw_trade_box(
    ax,
    entry_x: int,
    exit_x: int,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    direction: str,
) -> None:
    """
    Draws risk and reward rectangles for a single trade.
    """
    # Make sure the box has visible width even if entry and exit are close.
    box_width = max(exit_x - entry_x, 0.8)

    if direction == "long":
        # For longs, reward is above entry and risk is below entry.
        reward_bottom = entry_price
        reward_height = take_profit - entry_price

        risk_bottom = stop_loss
        risk_height = entry_price - stop_loss

    else:
        # For shorts, reward is below entry and risk is above entry.
        reward_bottom = take_profit
        reward_height = entry_price - take_profit

        risk_bottom = entry_price
        risk_height = stop_loss - entry_price

    # Draw the reward area.
    if reward_height > 0:
        ax.add_patch(
            Rectangle(
                (entry_x, reward_bottom),
                box_width,
                reward_height,
                alpha=0.22,
                facecolor="green",
                edgecolor="green",
                linewidth=0.8,
                zorder=2,
            )
        )

    # Draw the risk area.
    if risk_height > 0:
        ax.add_patch(
            Rectangle(
                (entry_x, risk_bottom),
                box_width,
                risk_height,
                alpha=0.22,
                facecolor="red",
                edgecolor="red",
                linewidth=0.8,
                zorder=2,
            )
        )


def plot_backtest_trades(
    candles_df: pd.DataFrame,
    levels_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    output_path: str | Path,
    max_levels: int = 10,
) -> None:
    """
    Creates a candlestick chart with support/resistance zones and backtest trades.
    """
    # Stop early if there are no candles to plot.
    if candles_df.empty:
        raise ValueError("Cannot plot backtest chart because candles_df is empty.")

    # Stop early if there are no trades to visualize.
    if trades_df.empty:
        raise ValueError("Cannot plot backtest chart because trades_df is empty.")

    required_candle_columns = [
        "timestamp_utc",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
    ]

    required_trade_columns = [
        "entry_timestamp",
        "direction",
        "entry_price",
        "stop_loss",
        "take_profit",
        "exit_timestamp",
        "exit_price",
        "outcome",
        "r_multiple",
    ]

    # Check that all required candle columns exist.
    missing_candle_columns = [
        column for column in required_candle_columns
        if column not in candles_df.columns
    ]

    if missing_candle_columns:
        raise ValueError(
            f"Missing required candle columns: {missing_candle_columns}"
        )

    # Check that all required trade columns exist.
    missing_trade_columns = [
        column for column in required_trade_columns
        if column not in trades_df.columns
    ]

    if missing_trade_columns:
        raise ValueError(
            f"Missing required trade columns: {missing_trade_columns}"
        )

    # Copy candles so the original DataFrame is not modified.
    chart_df = candles_df[required_candle_columns].copy()

    # Convert timestamps and prices into plotting-friendly types.
    chart_df["timestamp_utc"] = pd.to_datetime(chart_df["timestamp_utc"])
    chart_df["open_price"] = chart_df["open_price"].astype(float)
    chart_df["high_price"] = chart_df["high_price"].astype(float)
    chart_df["low_price"] = chart_df["low_price"].astype(float)
    chart_df["close_price"] = chart_df["close_price"].astype(float)

    # Sort candles so x-axis positions match time order.
    chart_df = chart_df.sort_values("timestamp_utc").reset_index(drop=True)

    # Rename columns to the format expected by mplfinance.
    chart_df = chart_df.rename(
        columns={
            "timestamp_utc": "Date",
            "open_price": "Open",
            "high_price": "High",
            "low_price": "Low",
            "close_price": "Close",
        }
    )

    # Use the Date column as the chart index.
    chart_df = chart_df.set_index("Date")

    # Copy trades so we can safely convert timestamps and numeric values.
    trades = trades_df.copy()
    trades["entry_timestamp"] = pd.to_datetime(trades["entry_timestamp"])
    trades["exit_timestamp"] = pd.to_datetime(trades["exit_timestamp"])

    # Convert trade price columns to numeric values.
    for column in ["entry_price", "stop_loss", "take_profit", "exit_price"]:
        trades[column] = pd.to_numeric(trades[column], errors="coerce")

    output_path = Path(output_path)

    # Create output folder if it does not exist.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use a simple custom style so candles and trade boxes are easier to see.
    market_colors = mpf.make_marketcolors(
        up="green",
        down="red",
        edge="inherit",
        wick="black",
        volume="inherit",
    )

    chart_style = mpf.make_mpf_style(
        base_mpf_style="yahoo",
        marketcolors=market_colors,
        gridstyle="-",
    )

    # Create the base candlestick chart.
    fig, axes = mpf.plot(
        chart_df,
        type="candle",
        style=chart_style,
        title=f"{symbol} Backtest Trades - {timeframe}",
        ylabel="Price",
        volume=False,
        tight_layout=True,
        returnfig=True,
        warn_too_much_data=len(chart_df) + 1,
    )

    main_ax = axes[0]

    # Plot support/resistance zones if they exist.
    if levels_df is not None and not levels_df.empty:
        levels_to_plot = (
            levels_df.sort_values(
                ["tested_touches", "respect_rate"],
                ascending=False,
                na_position="last",
            )
            .head(max_levels)
            .copy()
        )

        for _, level in levels_to_plot.iterrows():
            # Draw the support/resistance zone as a horizontal shaded area.
            main_ax.axhspan(
                float(level["zone_low"]),
                float(level["zone_high"]),
                alpha=0.10,
                color="steelblue",
                zorder=1,
            )

            # Build a compact zone label.
            respect_rate = level.get("respect_rate")
            respect_text = "N/A" if pd.isna(respect_rate) else f"{respect_rate:.2f}"

            label = (
                f"{level['level_type']} | "
                f"tested: {int(level['tested_touches'])} | "
                f"respect: {respect_text}"
            )

            # Place zone label on the left side of the chart.
            main_ax.text(
                0.01,
                float(level["level_price"]),
                label,
                transform=main_ax.get_yaxis_transform(),
                fontsize=8,
                verticalalignment="center",
                zorder=6,
            )

    # Draw each trade with risk/reward boxes and clear markers.
    for trade_number, (_, trade) in enumerate(trades.iterrows(), start=1):
        entry_time = trade["entry_timestamp"]
        exit_time = trade["exit_timestamp"]

        # Convert trade timestamps to mplfinance x-axis positions.
        entry_x = _find_nearest_x_position(chart_df.index, entry_time)
        exit_x = _find_nearest_x_position(chart_df.index, exit_time)

        # Skip trades that cannot be mapped to the chart x-axis.
        if entry_x is None or exit_x is None:
            continue

        entry_price = float(trade["entry_price"])
        stop_loss = float(trade["stop_loss"])
        take_profit = float(trade["take_profit"])
        exit_price = float(trade["exit_price"])

        direction = trade["direction"]
        outcome = trade["outcome"]

        # Draw the TradingView-style risk/reward box.
        _draw_trade_box(
            ax=main_ax,
            entry_x=entry_x,
            exit_x=exit_x,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            direction=direction,
        )

        # Choose entry marker based on direction.
        entry_marker = "^" if direction == "long" else "v"

        # Plot entry point with a clear marker.
        main_ax.scatter(
            entry_x,
            entry_price,
            marker=entry_marker,
            s=90,
            color="black",
            edgecolor="white",
            linewidth=0.8,
            zorder=8,
        )

        # Plot exit point with a clear X marker.
        main_ax.scatter(
            exit_x,
            exit_price,
            marker="x",
            s=90,
            color="black",
            linewidth=1.5,
            zorder=8,
        )

        # Draw a dotted path from entry to exit.
        main_ax.plot(
            [entry_x, exit_x],
            [entry_price, exit_price],
            linestyle=":",
            linewidth=1.0,
            color="black",
            alpha=0.8,
            zorder=7,
        )

        # Draw SL and TP lines only across the trade duration.
        main_ax.hlines(
            stop_loss,
            xmin=entry_x,
            xmax=exit_x,
            linestyles="dashed",
            linewidth=1.0,
            color="red",
            alpha=0.9,
            zorder=6,
        )

        main_ax.hlines(
            take_profit,
            xmin=entry_x,
            xmax=exit_x,
            linestyles="dashed",
            linewidth=1.0,
            color="green",
            alpha=0.9,
            zorder=6,
        )

        # Build a compact label for the trade result.
        r_multiple = trade["r_multiple"]
        r_text = "N/A" if pd.isna(r_multiple) else f"{r_multiple:.2f}R"

        trade_label = f"#{trade_number} {outcome} {r_text}"

        # Place the label near the entry.
        main_ax.text(
            entry_x,
            entry_price,
            trade_label,
            fontsize=7,
            verticalalignment="bottom",
            color="black",
            zorder=9,
        )

    # Save the final chart.
    fig.savefig(output_path, dpi=160)
    plt.close(fig)