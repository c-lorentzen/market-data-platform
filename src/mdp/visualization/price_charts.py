from pathlib import Path

import matplotlib.pyplot as plt
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