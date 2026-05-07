from pathlib import Path

import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd

from mdp.structure.trend_channels import prepare_channel_candles


def _validate_candle_columns(candles_df: pd.DataFrame) -> None:
    """
    Validates that the candle DataFrame has the OHLC columns required for plotting.
    """
    required_columns = [
        "timestamp_utc",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
    ]

    # Find missing columns before the chart is created.
    missing_columns = [
        column for column in required_columns if column not in candles_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required candle columns for rolling channel plotting: {missing_columns}"
        )


def _validate_rolling_channel_columns(rolling_channels_df: pd.DataFrame) -> None:
    """
    Validates that rolling channel rows have the values needed for chart reconstruction.
    """
    if rolling_channels_df.empty:
        return

    required_columns = [
        "rolling_channel_id",
        "window_size",
        "window_start_index",
        "window_end_index",
        "channel_role",
        "rolling_quality_score",
        "channel_type",
        "slope",
        "intercept",
        "channel_width",
        "containment_rate",
        "r_squared",
        "current_state",
    ]

    # Check the output from rolling_trend_channels.py before drawing lines.
    missing_columns = [
        column for column in required_columns if column not in rolling_channels_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required rolling channel columns for plotting: {missing_columns}"
        )


def _prepare_candlestick_dataframe(candles_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepares candles for channel reconstruction and mplfinance candlestick plotting.
    """
    # Reuse the same preparation function as the channel detector so indexes match.
    market_df = prepare_channel_candles(candles_df)

    if market_df.empty:
        raise ValueError("Cannot plot rolling channels because candles_df is empty.")

    # Keep only the columns expected by mplfinance.
    chart_df = market_df[
        ["timestamp_utc", "open_price", "high_price", "low_price", "close_price"]
    ].copy()

    # Rename columns to the standard mplfinance names.
    chart_df = chart_df.rename(
        columns={
            "timestamp_utc": "Date",
            "open_price": "Open",
            "high_price": "High",
            "low_price": "Low",
            "close_price": "Close",
        }
    )

    # Use timestamps as the chart index.
    chart_df = chart_df.set_index("Date")

    return market_df, chart_df


def _calculate_rolling_channel_boundaries(
    market_df: pd.DataFrame,
    channel: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Reconstructs rolling channel boundaries only inside the channel's own window.
    """
    window_start_index = int(channel["window_start_index"])
    window_end_index = int(channel["window_end_index"])

    # Select only the candles used by this rolling channel candidate.
    window_df = market_df.iloc[window_start_index:window_end_index].copy()

    if window_df.empty:
        raise ValueError("Rolling channel window is empty and cannot be plotted.")

    slope = float(channel["slope"])
    intercept = float(channel["intercept"])
    channel_width = float(channel["channel_width"])
    channel_type = str(channel["channel_type"])

    # Rolling detection resets candle_index inside each window, so convert global index to local index.
    local_candle_index = window_df["candle_index"].astype(float) - float(window_start_index)
    base_line = slope * local_candle_index + intercept

    if channel_type == "ascending":
        # Ascending channels use the pivot-low line as the lower boundary.
        lower_line = base_line
        upper_line = base_line + channel_width
    elif channel_type == "descending":
        # Descending channels use the pivot-high line as the upper boundary.
        upper_line = base_line
        lower_line = base_line - channel_width
    else:
        raise ValueError(
            f"Unsupported channel_type '{channel_type}'. Expected 'ascending' or 'descending'."
        )

    return window_df["candle_index"].astype(int), lower_line, upper_line


def _draw_pivots(ax, pivots_df: pd.DataFrame) -> None:
    """
    Draws pivot highs and pivot lows on the candlestick chart.
    """
    if pivots_df.empty:
        return

    required_columns = ["candle_index", "pivot_type", "pivot_price"]

    # Validate pivot data before plotting markers.
    missing_columns = [
        column for column in required_columns if column not in pivots_df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required pivot columns: {missing_columns}")

    pivot_highs = pivots_df[pivots_df["pivot_type"] == "high"].copy()
    pivot_lows = pivots_df[pivots_df["pivot_type"] == "low"].copy()

    if not pivot_highs.empty:
        # Plot swing highs with downward markers.
        ax.scatter(
            pivot_highs["candle_index"].astype(int),
            pivot_highs["pivot_price"].astype(float),
            marker="v",
            s=28,
            label="Pivot highs",
            zorder=4,
        )

    if not pivot_lows.empty:
        # Plot swing lows with upward markers.
        ax.scatter(
            pivot_lows["candle_index"].astype(int),
            pivot_lows["pivot_price"].astype(float),
            marker="^",
            s=28,
            label="Pivot lows",
            zorder=4,
        )


def _draw_rolling_channels(
    ax,
    market_df: pd.DataFrame,
    rolling_channels_df: pd.DataFrame,
    max_channels: int,
) -> None:
    """
    Draws ranked rolling channels over their actual candle windows.
    """
    if rolling_channels_df.empty:
        return

    # Plot the highest-ranked rolling candidates first.
    channels_to_plot = (
        rolling_channels_df.sort_values("rolling_quality_score", ascending=False)
        .head(max_channels)
        .copy()
    )

    for channel_number, (_, channel) in enumerate(channels_to_plot.iterrows(), start=1):
        x_values, lower_line, upper_line = _calculate_rolling_channel_boundaries(
            market_df=market_df,
            channel=channel,
        )

        rolling_channel_id = int(channel["rolling_channel_id"])
        window_size = int(channel["window_size"])
        channel_type = str(channel["channel_type"])
        channel_role = str(channel["channel_role"])
        rolling_quality_score = float(channel["rolling_quality_score"])
        current_state = str(channel["current_state"])

        # Make the strongest channel easier to inspect.
        line_width = 2.0 if channel_number == 1 else 1.3
        alpha = 0.95 if channel_number == 1 else 0.75

        label_prefix = (
            f"R{rolling_channel_id} {channel_type} W{window_size} | "
            f"RQ {rolling_quality_score:.2f} | {channel_role} | {current_state}"
        )

        # Draw lower and upper rolling channel boundaries only across the selected window.
        ax.plot(
            x_values.to_numpy(),
            lower_line.astype(float).to_numpy(),
            linewidth=line_width,
            alpha=alpha,
            label=f"{label_prefix} lower",
            zorder=3,
        )

        ax.plot(
            x_values.to_numpy(),
            upper_line.astype(float).to_numpy(),
            linewidth=line_width,
            linestyle="--",
            alpha=alpha,
            label=f"{label_prefix} upper",
            zorder=3,
        )

        # Fill only the strongest channel more clearly to avoid clutter.
        fill_alpha = 0.10 if channel_number == 1 else 0.04
        ax.fill_between(
            x_values.to_numpy(),
            lower_line.astype(float).to_numpy(),
            upper_line.astype(float).to_numpy(),
            alpha=fill_alpha,
            zorder=1,
        )


def _build_summary_text(rolling_channels_df: pd.DataFrame) -> str:
    """
    Builds a compact text summary for the best rolling channel.
    """
    if rolling_channels_df.empty:
        return "No valid rolling trend channel detected"

    best_channel = rolling_channels_df.sort_values(
        "rolling_quality_score",
        ascending=False,
    ).iloc[0]

    channel_type = str(best_channel["channel_type"])
    channel_role = str(best_channel["channel_role"])
    current_state = str(best_channel["current_state"])
    window_size = int(best_channel["window_size"])
    rolling_quality_score = float(best_channel["rolling_quality_score"])
    quality_score = float(best_channel.get("quality_score", 0.0))
    containment_rate = float(best_channel.get("containment_rate", 0.0))
    r_squared = float(best_channel.get("r_squared", 0.0))
    channel_width = float(best_channel.get("channel_width", 0.0))
    width_to_range_ratio = float(best_channel.get("width_to_range_ratio", 0.0))

    # Keep the annotation focused on the decision metrics.
    return (
        f"Best rolling channel: {channel_type}\n"
        f"Role: {channel_role}\n"
        f"Window size: {window_size}\n"
        f"Rolling Q: {rolling_quality_score:.2f}\n"
        f"Base Q: {quality_score:.2f}\n"
        f"Containment: {containment_rate:.2f}\n"
        f"R-squared: {r_squared:.2f}\n"
        f"Width: {channel_width:.2f}\n"
        f"Width/range: {width_to_range_ratio:.2f}\n"
        f"State: {current_state}"
    )


def plot_rolling_trend_channel_chart(
    candles_df: pd.DataFrame,
    pivots_df: pd.DataFrame,
    rolling_channels_df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    output_path: str | Path,
    max_channels: int = 4,
) -> None:
    """
    Creates and saves a candlestick chart with active rolling trend channels.
    """
    # Validate inputs before creating the figure.
    _validate_candle_columns(candles_df)
    _validate_rolling_channel_columns(rolling_channels_df)

    if max_channels <= 0:
        raise ValueError("max_channels must be greater than 0.")

    # Prepare candles for mplfinance and channel boundary reconstruction.
    market_df, chart_df = _prepare_candlestick_dataframe(candles_df)

    output_path = Path(output_path)

    # Create the output folder only when needed.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use the same simple project chart style used in previous visualizations.
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

    # Create the base candlestick chart first.
    fig, axes = mpf.plot(
        chart_df,
        type="candle",
        style=chart_style,
        title=f"{symbol} Rolling Trend Channels - {timeframe}",
        ylabel="Price",
        volume=False,
        tight_layout=True,
        returnfig=True,
        warn_too_much_data=len(chart_df) + 1,
    )

    main_ax = axes[0]

    # Draw pivots and rolling channel candidates over the candles.
    _draw_pivots(ax=main_ax, pivots_df=pivots_df)
    _draw_rolling_channels(
        ax=main_ax,
        market_df=market_df,
        rolling_channels_df=rolling_channels_df,
        max_channels=max_channels,
    )

    # Add the latest close as a reference level.
    current_close = float(market_df.iloc[-1]["close_price"])
    main_ax.axhline(
        current_close,
        linestyle=":",
        linewidth=1.0,
        alpha=0.8,
        label=f"Current close {current_close:.2f}",
    )

    # Add the best-channel diagnostic box.
    main_ax.text(
        0.01,
        0.98,
        _build_summary_text(rolling_channels_df),
        transform=main_ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox={"boxstyle": "round", "alpha": 0.18},
    )

    # Show labels for pivots, rolling boundaries, and latest close.
    main_ax.legend(loc="best", fontsize=6)

    # Save the final image.
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
