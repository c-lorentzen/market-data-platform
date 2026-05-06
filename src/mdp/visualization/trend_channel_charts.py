from pathlib import Path

import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd

from mdp.structure.trend_channels import prepare_channel_candles


def _validate_candle_columns(candles_df: pd.DataFrame) -> None:
    """
    Validates that the input candle DataFrame has the required OHLC columns.
    """
    required_columns = [
        "timestamp_utc",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
    ]

    # Find missing columns before trying to plot the chart.
    missing_columns = [
        column for column in required_columns if column not in candles_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required candle columns for trend channel plotting: {missing_columns}"
        )


def _validate_channel_columns(channels_df: pd.DataFrame) -> None:
    """
    Validates that detected channel rows include the values needed for plotting.
    """
    required_columns = [
        "channel_id",
        "channel_type",
        "slope",
        "intercept",
        "channel_width",
        "quality_score",
        "containment_rate",
        "current_state",
    ]

    # Do not require channel columns when no channel was detected.
    if channels_df.empty:
        return

    # Check for missing channel metrics before drawing channel boundaries.
    missing_columns = [
        column for column in required_columns if column not in channels_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required channel columns for trend channel plotting: {missing_columns}"
        )


def _prepare_candlestick_dataframe(candles_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepares candles for both trend-channel math and mplfinance plotting.
    """
    # Reuse the same preparation function as the detector so indexes match.
    market_df = prepare_channel_candles(candles_df)

    # Stop early if there is no market data after preparation.
    if market_df.empty:
        raise ValueError("Cannot plot trend channels because candles_df is empty.")

    # Keep only the columns needed by mplfinance.
    chart_df = market_df[
        ["timestamp_utc", "open_price", "high_price", "low_price", "close_price"]
    ].copy()

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

    # Use the timestamp as the chart index.
    chart_df = chart_df.set_index("Date")

    return market_df, chart_df


def _calculate_channel_boundary_values(
    market_df: pd.DataFrame,
    channel: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    Reconstructs lower and upper channel boundaries from stored channel metrics.
    """
    slope = float(channel["slope"])
    intercept = float(channel["intercept"])
    channel_width = float(channel["channel_width"])
    channel_type = str(channel["channel_type"])

    # Rebuild the regression base line using the same candle_index used in detection.
    base_line = slope * market_df["candle_index"].astype(float) + intercept

    if channel_type == "ascending":
        # Ascending channels use the pivot-low regression line as the lower boundary.
        lower_line = base_line
        upper_line = base_line + channel_width
    elif channel_type == "descending":
        # Descending channels use the pivot-high regression line as the upper boundary.
        upper_line = base_line
        lower_line = base_line - channel_width
    else:
        raise ValueError(
            f"Unsupported channel_type '{channel_type}'. Expected 'ascending' or 'descending'."
        )

    return lower_line, upper_line


def _draw_pivots(ax, pivots_df: pd.DataFrame) -> None:
    """
    Draws pivot highs and pivot lows on the candlestick chart.
    """
    # Skip pivot drawing if no pivots were detected.
    if pivots_df.empty:
        return

    required_columns = ["candle_index", "pivot_type", "pivot_price"]

    # Validate pivot columns before plotting markers.
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
            s=35,
            label="Pivot highs",
            zorder=4,
        )

    if not pivot_lows.empty:
        # Plot swing lows with upward markers.
        ax.scatter(
            pivot_lows["candle_index"].astype(int),
            pivot_lows["pivot_price"].astype(float),
            marker="^",
            s=35,
            label="Pivot lows",
            zorder=4,
        )


def _draw_channel_boundaries(
    ax,
    market_df: pd.DataFrame,
    channels_df: pd.DataFrame,
    max_channels: int,
) -> None:
    """
    Draws the detected trend channel boundaries on the chart.
    """
    # Stop early when the detector found no valid channel.
    if channels_df.empty:
        return

    # Plot the highest-ranked channels first.
    channels_to_plot = (
        channels_df.sort_values("quality_score", ascending=False)
        .head(max_channels)
        .copy()
    )

    x_values = market_df["candle_index"].astype(int).to_numpy()

    for channel_number, (_, channel) in enumerate(channels_to_plot.iterrows(), start=1):
        lower_line, upper_line = _calculate_channel_boundary_values(
            market_df=market_df,
            channel=channel,
        )

        channel_id = int(channel["channel_id"])
        channel_type = str(channel["channel_type"])
        quality_score = float(channel["quality_score"])
        containment_rate = float(channel["containment_rate"])

        # Make the strongest channel slightly thicker than secondary candidates.
        line_width = 1.7 if channel_number == 1 else 1.2

        label_prefix = (
            f"Channel {channel_id} {channel_type} | "
            f"Q {quality_score:.2f} | containment {containment_rate:.2f}"
        )

        # Draw lower and upper channel boundaries.
        ax.plot(
            x_values,
            lower_line.astype(float).to_numpy(),
            linewidth=line_width,
            label=f"{label_prefix} lower",
            zorder=3,
        )

        ax.plot(
            x_values,
            upper_line.astype(float).to_numpy(),
            linewidth=line_width,
            linestyle="--",
            label=f"{label_prefix} upper",
            zorder=3,
        )

        # Lightly fill the channel area to make the structure easier to inspect.
        ax.fill_between(
            x_values,
            lower_line.astype(float).to_numpy(),
            upper_line.astype(float).to_numpy(),
            alpha=0.08,
            zorder=1,
        )


def _build_summary_text(channels_df: pd.DataFrame) -> str:
    """
    Builds a compact text summary for the chart annotation box.
    """
    # Show a clear message when the detector found no valid channel.
    if channels_df.empty:
        return "No valid trend channel detected"

    primary_channel = channels_df.sort_values("quality_score", ascending=False).iloc[0]

    channel_type = str(primary_channel["channel_type"])
    quality_score = float(primary_channel["quality_score"])
    containment_rate = float(primary_channel["containment_rate"])
    current_state = str(primary_channel["current_state"])
    r_squared = float(primary_channel.get("r_squared", 0.0))
    total_touch_count = int(primary_channel.get("total_touch_count", 0))

    # Keep the label short so it does not cover too much chart space.
    return (
        f"Primary channel: {channel_type}\n"
        f"Quality score: {quality_score:.2f}\n"
        f"Containment rate: {containment_rate:.2f}\n"
        f"R-squared: {r_squared:.2f}\n"
        f"Total touches: {total_touch_count}\n"
        f"Current state: {current_state}"
    )


def plot_trend_channel_chart(
    candles_df: pd.DataFrame,
    pivots_df: pd.DataFrame,
    channels_df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    output_path: str | Path,
    max_channels: int = 2,
) -> None:
    """
    Creates and saves a candlestick chart with trend channels and swing pivots.
    """
    # Validate inputs before doing any plotting work.
    _validate_candle_columns(candles_df)
    _validate_channel_columns(channels_df)

    if max_channels <= 0:
        raise ValueError("max_channels must be greater than 0.")

    # Prepare candles for mplfinance and channel boundary reconstruction.
    market_df, chart_df = _prepare_candlestick_dataframe(candles_df)

    output_path = Path(output_path)

    # Create the output folder if it does not exist yet.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use the existing project style: simple candlesticks with readable grid lines.
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
        title=f"{symbol} Trend Channel Detection - {timeframe}",
        ylabel="Price",
        volume=False,
        tight_layout=True,
        returnfig=True,
        warn_too_much_data=len(chart_df) + 1,
    )

    main_ax = axes[0]

    # Draw pivots and channels on top of the candle chart.
    _draw_pivots(ax=main_ax, pivots_df=pivots_df)
    _draw_channel_boundaries(
        ax=main_ax,
        market_df=market_df,
        channels_df=channels_df,
        max_channels=max_channels,
    )

    # Add the current close as a reference line.
    current_close = float(market_df.iloc[-1]["close_price"])
    main_ax.axhline(
        current_close,
        linestyle=":",
        linewidth=1.0,
        alpha=0.8,
        label=f"Current close {current_close:.2f}",
    )

    # Add a compact diagnostic label directly inside the chart.
    main_ax.text(
        0.01,
        0.98,
        _build_summary_text(channels_df),
        transform=main_ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox={"boxstyle": "round", "alpha": 0.18},
    )

    # Show labels for pivots, boundaries, and current close.
    main_ax.legend(loc="best", fontsize=7)

    # Save the final chart image.
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
