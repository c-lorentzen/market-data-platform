from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_equity_curve_chart(
    equity_curve_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Saves a cumulative R equity curve chart.
    """
    # Convert the output path into a Path object.
    output_path = Path(output_path)

    # Create the output folder if it does not exist.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Stop early if there is no equity data to chart.
    if equity_curve_df.empty:
        return

    # Create one clean chart for cumulative R.
    plt.figure(figsize=(10, 5))

    # Plot cumulative R by trade number.
    plt.plot(
        equity_curve_df["trade_number"],
        equity_curve_df["cumulative_r"],
        marker="o",
        linewidth=2,
    )

    # Add a zero reference line to separate positive and negative performance.
    plt.axhline(0, linestyle="--", linewidth=1)

    # Add titles and labels for client readability.
    plt.title("Equity Curve - Cumulative R")
    plt.xlabel("Trade Number")
    plt.ylabel("Cumulative R")

    # Make the chart easier to read.
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the chart image.
    plt.savefig(output_path, dpi=160)

    # Close the figure so memory does not build up.
    plt.close()


def save_drawdown_chart(
    equity_curve_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Saves a drawdown chart in R units.
    """
    # Convert the output path into a Path object.
    output_path = Path(output_path)

    # Create the output folder if it does not exist.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Stop early if there is no equity data to chart.
    if equity_curve_df.empty:
        return

    # Create one clean chart for drawdown.
    plt.figure(figsize=(10, 5))

    # Plot drawdown by trade number.
    plt.plot(
        equity_curve_df["trade_number"],
        equity_curve_df["drawdown_r"],
        marker="o",
        linewidth=2,
    )

    # Add a zero reference line.
    plt.axhline(0, linestyle="--", linewidth=1)

    # Add titles and labels for client readability.
    plt.title("Drawdown Curve - R Units")
    plt.xlabel("Trade Number")
    plt.ylabel("Drawdown R")

    # Make the chart easier to read.
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the chart image.
    plt.savefig(output_path, dpi=160)

    # Close the figure so memory does not build up.
    plt.close()


def save_trade_distribution_chart(
    trade_distribution_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Saves a bar chart showing trade outcome distribution by R bucket.
    """
    # Convert the output path into a Path object.
    output_path = Path(output_path)

    # Create the output folder if it does not exist.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Stop early if there is no distribution data to chart.
    if trade_distribution_df.empty:
        return

    # Create a more readable label for each bucket.
    chart_df = trade_distribution_df.copy()
    chart_df["bucket_label"] = chart_df["bucket"].str.replace("_", " ").str.title()

    # Create one clean bar chart.
    plt.figure(figsize=(11, 5))

    # Plot trade count by bucket.
    plt.bar(
        chart_df["bucket_label"],
        chart_df["trade_count"],
    )

    # Add titles and labels for client readability.
    plt.title("Trade Distribution by R Multiple")
    plt.xlabel("R Multiple Bucket")
    plt.ylabel("Number of Trades")

    # Rotate labels so they do not overlap.
    plt.xticks(rotation=25, ha="right")

    # Make the chart easier to read.
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    # Save the chart image.
    plt.savefig(output_path, dpi=160)

    # Close the figure so memory does not build up.
    plt.close()


def save_window_activity_chart(
    windows_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Saves a chart showing signals and trades per walk-forward window.
    """
    # Convert the output path into a Path object.
    output_path = Path(output_path)

    # Create the output folder if it does not exist.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Stop early if there is no window data to chart.
    if windows_df.empty:
        return

    # Create one clean chart for window activity.
    plt.figure(figsize=(10, 5))

    # Plot signals generated per window.
    plt.plot(
        windows_df["window_id"],
        windows_df["signals_generated"],
        marker="o",
        linewidth=2,
        label="Signals",
    )

    # Plot trades simulated per window.
    plt.plot(
        windows_df["window_id"],
        windows_df["trades_simulated"],
        marker="o",
        linewidth=2,
        label="Trades",
    )

    # Add titles and labels for client readability.
    plt.title("Walk-Forward Activity by Window")
    plt.xlabel("Window ID")
    plt.ylabel("Count")

    # Add legend so the client can distinguish signals from trades.
    plt.legend()

    # Make the chart easier to read.
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the chart image.
    plt.savefig(output_path, dpi=160)

    # Close the figure so memory does not build up.
    plt.close()