from pathlib import Path

import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd

def plot_close_price(
    candles_df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    output_path: str | Path,
) -> None:
    """
    Creates and saves a line chart of close prices over time.
    """
    if candles_df.empty:
        raise ValueError("Cannot plot close price because the DataFrame is empty.")

    required_columns = ["timestamp_utc", "close_price"]
    missing_columns = [
        column for column in required_columns if column not in candles_df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns for plotting: {missing_columns}")

    chart_df = candles_df.copy()
    chart_df["timestamp_utc"] = pd.to_datetime(chart_df["timestamp_utc"])
    chart_df["close_price"] = chart_df["close_price"].astype(float)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(chart_df["timestamp_utc"], chart_df["close_price"])

    plt.title(f"{symbol} Close Price - {timeframe}")
    plt.xlabel("Time UTC")
    plt.ylabel("Close Price")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()

def plot_candlestick_chart(
    candles_df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    output_path: str | Path,
) -> None:
    """
    Creates and saves a candlestick chart using OHLC market candle data.
    """
    if candles_df.empty:
        raise ValueError("Cannot plot candlestick chart because the DataFrame is empty.")

    required_columns = [
        "timestamp_utc",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
    ]

    missing_columns = [
        column for column in required_columns if column not in candles_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns for candlestick plotting: {missing_columns}"
        )

    chart_df = candles_df[required_columns].copy()

    chart_df["timestamp_utc"] = pd.to_datetime(chart_df["timestamp_utc"])
    chart_df["open_price"] = chart_df["open_price"].astype(float)
    chart_df["high_price"] = chart_df["high_price"].astype(float)
    chart_df["low_price"] = chart_df["low_price"].astype(float)
    chart_df["close_price"] = chart_df["close_price"].astype(float)

    chart_df = chart_df.rename(
        columns={
            "timestamp_utc": "Date",
            "open_price": "Open",
            "high_price": "High",
            "low_price": "Low",
            "close_price": "Close",
        }
    )

    chart_df = chart_df.set_index("Date")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mpf.plot(
        chart_df,
        type="candle",
        title=f"{symbol} Candlestick Chart - {timeframe}",
        ylabel="Price",
        volume=False,
        tight_layout=True,
        savefig=str(output_path),
    )

def plot_candlestick_with_levels(
    candles_df: pd.DataFrame,
    levels_df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    output_path: str | Path,
    max_levels: int = 10,
) -> None:
    """
    Creates and saves a candlestick chart with support/resistance zones.
    """
    if candles_df.empty:
        raise ValueError("Cannot plot candlestick chart because the DataFrame is empty.")

    required_candle_columns = [
        "timestamp_utc",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
    ]

    missing_candle_columns = [
        column for column in required_candle_columns
        if column not in candles_df.columns
    ]

    if missing_candle_columns:
        raise ValueError(
            f"Missing required candle columns for plotting: {missing_candle_columns}"
        )

    required_level_columns = [
        "level_type",
        "level_price",
        "touch_count",
        "validation_score",
        "zone_low",
        "zone_high",
    ]

    missing_level_columns = [
        column for column in required_level_columns
        if column not in levels_df.columns
    ]

    if missing_level_columns:
        raise ValueError(
            f"Missing required level columns for plotting: {missing_level_columns}"
        )

    chart_df = candles_df[required_candle_columns].copy()

    chart_df["timestamp_utc"] = pd.to_datetime(chart_df["timestamp_utc"])
    chart_df["open_price"] = chart_df["open_price"].astype(float)
    chart_df["high_price"] = chart_df["high_price"].astype(float)
    chart_df["low_price"] = chart_df["low_price"].astype(float)
    chart_df["close_price"] = chart_df["close_price"].astype(float)

    chart_df = chart_df.rename(
        columns={
            "timestamp_utc": "Date",
            "open_price": "Open",
            "high_price": "High",
            "low_price": "Low",
            "close_price": "Close",
        }
    )

    chart_df = chart_df.set_index("Date")

    levels_to_plot = (
        levels_df.sort_values(
            ["touch_count", "validation_score"],
            ascending=False,
        )
        .head(max_levels)
        .copy()
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = mpf.plot(
        chart_df,
        type="candle",
        title=f"{symbol} Support / Resistance Zones - {timeframe}",
        ylabel="Price",
        volume=False,
        tight_layout=True,
        returnfig=True,
    )

    main_ax = axes[0]

    for _, level in levels_to_plot.iterrows():
        zone_low = float(level["zone_low"])
        zone_high = float(level["zone_high"])
        level_price = float(level["level_price"])
        level_type = level["level_type"]
        touch_count = int(level["touch_count"])
        validation_score = float(level["validation_score"])

        main_ax.axhspan(
            zone_low,
            zone_high,
            alpha=0.12,
        )

        label = (
            f"{level_type} | touches: {touch_count} | "
            f"score: {validation_score:.2f}"
        )

        main_ax.text(
            0.01,
            level_price,
            label,
            transform=main_ax.get_yaxis_transform(),
            fontsize=8,
            verticalalignment="center",
        )

    fig.savefig(output_path)
    plt.close(fig)