import pandas as pd


REQUIRED_SIGNAL_COLUMNS = [
    "timestamp_utc",
    "entry_price",
    "stop_loss",
    "take_profit",
    "direction",
]


def validate_signals(signals_df: pd.DataFrame) -> None:
    """
    Validates that a strategy signal DataFrame has the required structure.
    """
    missing_columns = [
        column for column in REQUIRED_SIGNAL_COLUMNS
        if column not in signals_df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required signal columns: {missing_columns}")

    valid_directions = {"long", "short"}

    invalid_directions = set(signals_df["direction"].dropna()) - valid_directions

    if invalid_directions:
        raise ValueError(f"Invalid signal directions: {invalid_directions}")


def simulate_trade(
    future_candles_df: pd.DataFrame,
    direction: str,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
) -> dict:
    """
    Simulates one trade after entry using future candles.

    The trade is closed when either stop_loss or take_profit is reached first.
    """
    if future_candles_df.empty:
        return {
            "outcome": "unresolved",
            "exit_price": None,
            "exit_timestamp": None,
            "r_multiple": None,
        }

    risk = abs(entry_price - stop_loss)

    if risk == 0:
        return {
            "outcome": "invalid",
            "exit_price": None,
            "exit_timestamp": None,
            "r_multiple": None,
        }

    for _, candle in future_candles_df.iterrows():
        candle_time = candle["timestamp_utc"]
        candle_high = float(candle["high_price"])
        candle_low = float(candle["low_price"])

        if direction == "long":
            stop_hit = candle_low <= stop_loss
            target_hit = candle_high >= take_profit

            if stop_hit and target_hit:
                return {
                    "outcome": "ambiguous",
                    "exit_price": None,
                    "exit_timestamp": candle_time,
                    "r_multiple": None,
                }

            if stop_hit:
                return {
                    "outcome": "loss",
                    "exit_price": stop_loss,
                    "exit_timestamp": candle_time,
                    "r_multiple": -1.0,
                }

            if target_hit:
                reward = take_profit - entry_price
                return {
                    "outcome": "win",
                    "exit_price": take_profit,
                    "exit_timestamp": candle_time,
                    "r_multiple": reward / risk,
                }

        elif direction == "short":
            stop_hit = candle_high >= stop_loss
            target_hit = candle_low <= take_profit

            if stop_hit and target_hit:
                return {
                    "outcome": "ambiguous",
                    "exit_price": None,
                    "exit_timestamp": candle_time,
                    "r_multiple": None,
                }

            if stop_hit:
                return {
                    "outcome": "loss",
                    "exit_price": stop_loss,
                    "exit_timestamp": candle_time,
                    "r_multiple": -1.0,
                }

            if target_hit:
                reward = entry_price - take_profit
                return {
                    "outcome": "win",
                    "exit_price": take_profit,
                    "exit_timestamp": candle_time,
                    "r_multiple": reward / risk,
                }

    # If neither TP nor SL was reached, close the trade at the last available candle.
    # This is a time-based exit, not a true unresolved trade.
    last_candle = future_candles_df.iloc[-1]
    exit_price = float(last_candle["close_price"])

    # Calculate the realized R multiple based on the final close price.
    if direction == "long":
        r_multiple = (exit_price - entry_price) / risk
    else:
        r_multiple = (entry_price - exit_price) / risk

    return {
        "outcome": "time_exit",
        "exit_price": exit_price,
        "exit_timestamp": last_candle["timestamp_utc"],
        "r_multiple": r_multiple,
    }


def run_backtest(
    candles_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    max_holding_candles: int = 20,
    allow_overlapping_trades: bool = False,
) -> pd.DataFrame:
    """
    Runs a simple candle-by-candle backtest from a strategy signal DataFrame.
    """
    if candles_df.empty:
        raise ValueError("candles_df is empty.")

    if signals_df.empty:
        return pd.DataFrame()

    validate_signals(signals_df)

    market_df = candles_df.copy()
    market_df["timestamp_utc"] = pd.to_datetime(market_df["timestamp_utc"])
    market_df = market_df.sort_values("timestamp_utc").reset_index(drop=True)

    price_columns = [
        "open_price",
        "high_price",
        "low_price",
        "close_price",
    ]

    for column in price_columns:
        market_df[column] = market_df[column].astype(float)

    signals = signals_df.copy()
    signals["timestamp_utc"] = pd.to_datetime(signals["timestamp_utc"])
    signals = signals.sort_values("timestamp_utc").reset_index(drop=True)

    trades = []
    next_allowed_entry_time = None

    for _, signal in signals.iterrows():
        signal_time = signal["timestamp_utc"]

        # Ignore upcoming signals while the previous trade still open
        if (
            not allow_overlapping_trades
            and next_allowed_entry_time is not None
            and signal_time <= next_allowed_entry_time
        ):
            continue

        matching_indexes = market_df.index[
            market_df["timestamp_utc"] == signal_time
        ].tolist()

        if not matching_indexes:
            continue

        signal_index = matching_indexes[0]

        future_start = signal_index + 1
        future_end = min(
            future_start + max_holding_candles,
            len(market_df),
        )

        future_candles_df = market_df.iloc[future_start:future_end].copy()

        trade_result = simulate_trade(
            future_candles_df=future_candles_df,
            direction=signal["direction"],
            entry_price=float(signal["entry_price"]),
            stop_loss=float(signal["stop_loss"]),
            take_profit=float(signal["take_profit"]),
        )

        # When overlapping trades are disabled, block new entries until this trade is finished.
        # This applies to wins, losses, and unresolved trades with a timeout timestamp.
        if (
            not allow_overlapping_trades
            and trade_result["exit_timestamp"] is not None
        ):
            next_allowed_entry_time = pd.to_datetime(trade_result["exit_timestamp"])

        trades.append(
            {
                "entry_timestamp": signal_time,
                "direction": signal["direction"],
                "entry_price": float(signal["entry_price"]),
                "stop_loss": float(signal["stop_loss"]),
                "take_profit": float(signal["take_profit"]),
                "exit_timestamp": trade_result["exit_timestamp"],
                "exit_price": trade_result["exit_price"],
                "outcome": trade_result["outcome"],
                "r_multiple": trade_result["r_multiple"],
            }
        )

    return pd.DataFrame(trades)


def summarize_backtest(trades_df: pd.DataFrame) -> dict:
    """
    Summarizes basic backtest results.

    Wins and losses mean TP/SL outcomes.
    Time exits are trades closed because max_holding_candles was reached.
    """
    if trades_df.empty:
        return {
            "total_trades": 0,
            "closed_trades": 0,
            "wins": 0,
            "losses": 0,
            "time_exits": 0,
            "tp_sl_win_rate": None,
            "profitable_rate": None,
            "average_r": None,
            "total_r": None,
            "unresolved_trades": 0,
            "ambiguous_trades": 0,
            "invalid_trades": 0,
        }

    # Use only trades that have a real R multiple for performance metrics.
    closed_df = trades_df[trades_df["r_multiple"].notna()].copy()

    total_trades = len(trades_df)
    closed_trades = len(closed_df)

    # Count classic TP/SL outcomes.
    wins = int((trades_df["outcome"] == "win").sum())
    losses = int((trades_df["outcome"] == "loss").sum())
    time_exits = int((trades_df["outcome"] == "time_exit").sum())

    # TP/SL win rate ignores time exits because they did not hit TP or SL.
    tp_sl_trades = wins + losses
    tp_sl_win_rate = None

    if tp_sl_trades > 0:
        tp_sl_win_rate = wins / tp_sl_trades

    profitable_rate = None

    # Profitable rate uses all closed trades, including time exits.
    if closed_trades > 0:
        profitable_trades = int((closed_df["r_multiple"] > 0).sum())
        profitable_rate = profitable_trades / closed_trades

    return {
        "total_trades": total_trades,
        "closed_trades": closed_trades,
        "wins": wins,
        "losses": losses,
        "time_exits": time_exits,
        "tp_sl_win_rate": tp_sl_win_rate,
        "profitable_rate": profitable_rate,
        "average_r": closed_df["r_multiple"].mean() if not closed_df.empty else None,
        "total_r": closed_df["r_multiple"].sum() if not closed_df.empty else None,
        "unresolved_trades": int((trades_df["outcome"] == "unresolved").sum()),
        "ambiguous_trades": int((trades_df["outcome"] == "ambiguous").sum()),
        "invalid_trades": int((trades_df["outcome"] == "invalid").sum()),
    }