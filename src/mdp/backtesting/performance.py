import math

import pandas as pd


def prepare_closed_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares trades with valid R multiples for performance analysis.
    """
    # Return an empty DataFrame if there are no trades to analyze.
    if trades_df.empty:
        return pd.DataFrame()

    # Copy the DataFrame so we do not modify the original trades.
    closed_df = trades_df.copy()

    # Convert r_multiple to numeric values so calculations are safe.
    closed_df["r_multiple"] = pd.to_numeric(
        closed_df["r_multiple"],
        errors="coerce",
    )

    # Keep only trades with a valid R multiple.
    closed_df = closed_df[closed_df["r_multiple"].notna()].copy()

    # Return early if no closed trades remain.
    if closed_df.empty:
        return pd.DataFrame()

    # Convert entry timestamps so trades can be sorted chronologically.
    closed_df["entry_timestamp"] = pd.to_datetime(
        closed_df["entry_timestamp"],
        errors="coerce",
    )

    # Sort trades in the order they happened.
    closed_df = closed_df.sort_values("entry_timestamp").reset_index(drop=True)

    # Add a simple trade number for reporting and charting.
    closed_df["trade_number"] = closed_df.index + 1

    return closed_df


def build_equity_curve(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds an equity curve using cumulative R multiples.
    """
    # Prepare only trades with valid R multiples.
    closed_df = prepare_closed_trades(trades_df)

    # Return a standard empty structure if there are no closed trades.
    if closed_df.empty:
        return pd.DataFrame(
            columns=[
                "trade_number",
                "entry_timestamp",
                "outcome",
                "r_multiple",
                "cumulative_r",
                "equity_peak_r",
                "drawdown_r",
            ]
        )

    # Calculate cumulative R over time.
    closed_df["cumulative_r"] = closed_df["r_multiple"].cumsum()

    # Track the best equity value reached so far.
    closed_df["equity_peak_r"] = closed_df["cumulative_r"].cummax()

    # Calculate drawdown as the distance from the current equity to the peak.
    closed_df["drawdown_r"] = (
        closed_df["cumulative_r"] - closed_df["equity_peak_r"]
    )

    return closed_df[
        [
            "trade_number",
            "entry_timestamp",
            "outcome",
            "r_multiple",
            "cumulative_r",
            "equity_peak_r",
            "drawdown_r",
        ]
    ].copy()


def calculate_streaks(r_values: pd.Series) -> dict:
    """
    Calculates longest winning and losing streaks.
    """
    # Start streak counters at zero.
    current_win_streak = 0
    current_loss_streak = 0
    longest_win_streak = 0
    longest_loss_streak = 0

    # Walk through each R result one trade at a time.
    for r_value in r_values:
        # Positive R counts as a winning/profitable trade.
        if r_value > 0:
            current_win_streak += 1
            current_loss_streak = 0

        # Negative R counts as a losing trade.
        elif r_value < 0:
            current_loss_streak += 1
            current_win_streak = 0

        # Zero R breaks both win and loss streaks.
        else:
            current_win_streak = 0
            current_loss_streak = 0

        # Update the maximum streaks seen so far.
        longest_win_streak = max(longest_win_streak, current_win_streak)
        longest_loss_streak = max(longest_loss_streak, current_loss_streak)

    return {
        "longest_win_streak": longest_win_streak,
        "longest_loss_streak": longest_loss_streak,
    }


def build_trade_distribution(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a simple distribution table for R multiple outcomes.
    """
    # Prepare only trades with valid R multiples.
    closed_df = prepare_closed_trades(trades_df)

    # Return an empty distribution if there are no trades.
    if closed_df.empty:
        return pd.DataFrame(
            columns=["bucket", "trade_count", "percentage"]
        )

    # Define human-readable R multiple buckets.
    buckets = [
        ("big_loss_r_lte_minus_1", closed_df["r_multiple"] <= -1),
        ("small_loss_between_minus_1_and_0", (closed_df["r_multiple"] > -1) & (closed_df["r_multiple"] < 0)),
        ("breakeven_r_equal_0", closed_df["r_multiple"] == 0),
        ("small_win_between_0_and_1", (closed_df["r_multiple"] > 0) & (closed_df["r_multiple"] < 1)),
        ("medium_win_between_1_and_2", (closed_df["r_multiple"] >= 1) & (closed_df["r_multiple"] < 2)),
        ("big_win_r_gte_2", closed_df["r_multiple"] >= 2),
    ]

    rows = []

    # Count how many trades fall into each bucket.
    for bucket_name, condition in buckets:
        trade_count = int(condition.sum())

        rows.append(
            {
                "bucket": bucket_name,
                "trade_count": trade_count,
                "percentage": trade_count / len(closed_df),
            }
        )

    return pd.DataFrame(rows)


def calculate_performance_summary(trades_df: pd.DataFrame) -> dict:
    """
    Calculates professional performance metrics from backtest trades.
    """
    # Prepare trades that have valid R results.
    closed_df = prepare_closed_trades(trades_df)

    # Return empty-safe metrics if there are no closed trades.
    if closed_df.empty:
        return {
            "closed_trades": 0,
            "total_r": None,
            "average_r": None,
            "median_r": None,
            "std_r": None,
            "max_win_r": None,
            "max_loss_r": None,
            "profit_factor": None,
            "expectancy_r": None,
            "average_win_r": None,
            "average_loss_r": None,
            "payoff_ratio": None,
            "profitable_rate": None,
            "tp_sl_win_rate": None,
            "max_drawdown_r": None,
            "return_to_drawdown_ratio": None,
            "sharpe_like_r": None,
            "longest_win_streak": 0,
            "longest_loss_streak": 0,
        }

    # Store R values in one variable for easier calculations.
    r_values = closed_df["r_multiple"]

    # Split positive and negative trades.
    winning_r = r_values[r_values > 0]
    losing_r = r_values[r_values < 0]

    # Calculate total gain and total loss in R units.
    gross_profit_r = winning_r.sum()
    gross_loss_r = abs(losing_r.sum())

    # Profit factor compares gross profit against gross loss.
    if gross_loss_r > 0:
        profit_factor = gross_profit_r / gross_loss_r
    elif gross_profit_r > 0:
        profit_factor = math.inf
    else:
        profit_factor = None

    # Calculate average win and average loss.
    average_win_r = winning_r.mean() if not winning_r.empty else None
    average_loss_r = losing_r.mean() if not losing_r.empty else None

    # Payoff ratio compares average win size to average loss size.
    if average_win_r is not None and average_loss_r is not None and average_loss_r != 0:
        payoff_ratio = average_win_r / abs(average_loss_r)
    else:
        payoff_ratio = None

    # Calculate percentage of trades with positive R.
    profitable_rate = int((r_values > 0).sum()) / len(closed_df)

    # TP/SL win rate only counts trades that hit TP or SL.
    tp_trades = int((closed_df["outcome"] == "win").sum())
    sl_trades = int((closed_df["outcome"] == "loss").sum())
    tp_sl_total = tp_trades + sl_trades

    if tp_sl_total > 0:
        tp_sl_win_rate = tp_trades / tp_sl_total
    else:
        tp_sl_win_rate = None

    # Build equity curve so we can calculate drawdown.
    equity_curve_df = build_equity_curve(closed_df)

    # Max drawdown is the worst negative drawdown value.
    max_drawdown_r = equity_curve_df["drawdown_r"].min()

    # Total R is the final cumulative performance.
    total_r = r_values.sum()

    # Return-to-drawdown compares total return against worst drawdown.
    if max_drawdown_r < 0:
        return_to_drawdown_ratio = total_r / abs(max_drawdown_r)
    else:
        return_to_drawdown_ratio = None

    # Sharpe-like metric in R units, not annualized.
    std_r = r_values.std(ddof=1)

    if std_r is not None and std_r > 0:
        sharpe_like_r = r_values.mean() / std_r * math.sqrt(len(r_values))
    else:
        sharpe_like_r = None

    # Calculate win/loss streaks.
    streaks = calculate_streaks(r_values)

    return {
        "closed_trades": len(closed_df),
        "total_r": total_r,
        "average_r": r_values.mean(),
        "median_r": r_values.median(),
        "std_r": std_r,
        "max_win_r": r_values.max(),
        "max_loss_r": r_values.min(),
        "gross_profit_r": gross_profit_r,
        "gross_loss_r": gross_loss_r,
        "profit_factor": profit_factor,
        "expectancy_r": r_values.mean(),
        "average_win_r": average_win_r,
        "average_loss_r": average_loss_r,
        "payoff_ratio": payoff_ratio,
        "profitable_rate": profitable_rate,
        "tp_sl_win_rate": tp_sl_win_rate,
        "max_drawdown_r": max_drawdown_r,
        "return_to_drawdown_ratio": return_to_drawdown_ratio,
        "sharpe_like_r": sharpe_like_r,
        "longest_win_streak": streaks["longest_win_streak"],
        "longest_loss_streak": streaks["longest_loss_streak"],
    }


def build_performance_report(trades_df: pd.DataFrame) -> dict:
    """
    Builds all performance outputs from a trades DataFrame.
    """
    # Calculate the main performance metrics.
    performance_summary = calculate_performance_summary(trades_df)

    # Build the cumulative R equity curve.
    equity_curve_df = build_equity_curve(trades_df)

    # Build the R multiple distribution table.
    trade_distribution_df = build_trade_distribution(trades_df)

    return {
        "performance_summary": performance_summary,
        "equity_curve_df": equity_curve_df,
        "trade_distribution_df": trade_distribution_df,
    }