from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from mdp.backtesting.trend_channel_outcome_dataset import (
    build_cutoff_schedule,
    infer_entry_timeframe,
    load_future_candles,
    normalize_timeframes,
    safe_float,
    safe_text,
    simulate_trade_outcome,
)
from mdp.structure.trend_channel_trade_plan import build_trend_channel_trade_plan


@dataclass(frozen=True)
class ManagedTrendChannelOutcome:
    """
    Stores one historical trade-management simulation result.
    """

    symbol: str
    timeframes: str
    analysis_cutoff: Any
    entry_timeframe: str | None
    entry_timestamp: Any
    plan_state: str | None
    plan_direction: str | None
    event_state: str | None
    entry_price: float | None
    original_stop_loss: float | None
    original_risk_points: float | None
    available_targets: str | None
    best_available_target: str | None
    best_available_rr: float | None
    original_outcome_state: str
    original_first_terminal_outcome: str | None
    original_max_favorable_r: float | None
    original_max_adverse_r: float | None
    management_profile: str
    partial_take_profit_r: float
    partial_exit_fraction: float
    move_stop_to_breakeven_r: float
    locked_profit_r: float
    trailing_start_r: float
    trailing_distance_r: float
    full_target_mode: str
    selected_full_target: str | None
    selected_full_target_rr: float | None
    partial_exit_hit: bool
    partial_exit_timestamp: Any
    breakeven_or_profit_lock_activated: bool
    trailing_activated: bool
    managed_exit_state: str
    managed_exit_timestamp: Any
    managed_realized_r: float | None
    managed_remaining_fraction: float
    managed_stop_price_at_exit: float | None
    managed_candles_to_exit: int | None
    managed_positive_exit: bool | None
    management_improved_vs_original: bool | None
    label_quality: str
    notes: str


def _directional_profit_r(direction: str, entry_price: float, exit_price: float, risk_points: float) -> float:
    """
    Converts an exit price into realized R for one unit of position size.
    """
    if direction == "bearish":
        return (entry_price - exit_price) / risk_points

    return (exit_price - entry_price) / risk_points


def _price_at_r(direction: str, entry_price: float, risk_points: float, r_value: float) -> float:
    """
    Converts an R distance into an absolute price level.
    """
    if direction == "bearish":
        return entry_price - (risk_points * r_value)

    return entry_price + (risk_points * r_value)


def _stop_price_at_r(direction: str, entry_price: float, risk_points: float, r_value: float) -> float:
    """
    Converts a locked-profit R into a stop price.
    """
    if direction == "bearish":
        return entry_price - (risk_points * r_value)

    return entry_price + (risk_points * r_value)


def _target_hit(direction: str, high_price: float, low_price: float, target_price: float) -> bool:
    """
    Checks whether a target price was touched by the current candle.
    """
    if direction == "bearish":
        return low_price <= target_price

    return high_price >= target_price


def _stop_hit(direction: str, high_price: float, low_price: float, stop_price: float) -> bool:
    """
    Checks whether the active stop price was touched by the current candle.
    """
    if direction == "bearish":
        return high_price >= stop_price

    return low_price <= stop_price


def _favorable_r_from_candle(direction: str, entry_price: float, risk_points: float, high_price: float, low_price: float) -> float:
    """
    Calculates the best favorable movement reached inside a candle.
    """
    if direction == "bearish":
        return (entry_price - low_price) / risk_points

    return (high_price - entry_price) / risk_points


def _select_full_target(trade_plan_row: pd.Series, full_target_mode: str) -> tuple[str | None, float | None, float | None]:
    """
    Selects which available target should close the remaining position.
    """
    available_targets: list[tuple[str, float, float]] = []

    for target_label in ["tp1", "tp2", "tp3"]:
        # Only use targets that the trade-plan module marked as available.
        if bool(trade_plan_row.get(f"{target_label}_available")):
            target_price = safe_float(trade_plan_row.get(f"{target_label}_price"))
            target_rr = safe_float(trade_plan_row.get(f"{target_label}_rr"))
            if target_price is not None and target_rr is not None:
                available_targets.append((target_label.upper(), target_price, target_rr))

    if not available_targets or full_target_mode == "none":
        return None, None, None

    if full_target_mode == "best":
        selected = max(available_targets, key=lambda item: item[2])
    else:
        # The nearest target is the first realistic full exit for conservative management.
        selected = min(available_targets, key=lambda item: item[2])

    return selected


def simulate_managed_trade_outcome(
    trade_plan_row: pd.Series,
    future_candles: pd.DataFrame,
    partial_take_profit_r: float = 0.50,
    partial_exit_fraction: float = 0.50,
    move_stop_to_breakeven_r: float = 0.75,
    locked_profit_r: float = 0.00,
    trailing_start_r: float = 1.25,
    trailing_distance_r: float = 0.75,
    full_target_mode: str = "nearest",
    time_exit_candles: int | None = None,
    exit_at_horizon_close: bool = True,
    intrabar_policy: str = "conservative",
) -> dict[str, Any]:
    """
    Simulates partial exit, breakeven/profit-lock, trailing stop, and optional time exit.
    """
    direction = safe_text(trade_plan_row.get("plan_direction"))
    entry_price = safe_float(trade_plan_row.get("entry_price"))
    stop_loss = safe_float(trade_plan_row.get("stop_loss"))
    risk_points = safe_float(trade_plan_row.get("risk_points"))

    if direction not in {"bullish", "bearish"} or entry_price is None or stop_loss is None or risk_points is None or risk_points <= 0:
        return {
            "managed_exit_state": "invalid_trade_plan_levels",
            "managed_exit_timestamp": None,
            "managed_realized_r": None,
            "managed_remaining_fraction": 1.0,
            "managed_stop_price_at_exit": None,
            "managed_candles_to_exit": None,
            "partial_exit_hit": False,
            "partial_exit_timestamp": None,
            "breakeven_or_profit_lock_activated": False,
            "trailing_activated": False,
            "selected_full_target": None,
            "selected_full_target_rr": None,
            "label_quality": "invalid",
            "notes": "Trade-plan levels were invalid, so management could not be simulated.",
        }

    partial_exit_fraction = max(0.0, min(float(partial_exit_fraction), 1.0))
    partial_price = _price_at_r(direction, entry_price, risk_points, partial_take_profit_r)
    lock_stop_price = _stop_price_at_r(direction, entry_price, risk_points, locked_profit_r)
    target_label, target_price, target_rr = _select_full_target(trade_plan_row, full_target_mode)

    active_stop_price = stop_loss
    remaining_fraction = 1.0
    realized_r = 0.0
    partial_exit_hit = False
    partial_exit_timestamp = None
    breakeven_or_profit_lock_activated = False
    trailing_activated = False
    managed_exit_state = "open_horizon"
    managed_exit_timestamp = None
    managed_candles_to_exit = None

    # Walk forward candle-by-candle so the management rules respect event order.
    for future_index, candle in future_candles.iterrows():
        high_price = safe_float(candle.get("high_price"))
        low_price = safe_float(candle.get("low_price"))
        close_price = safe_float(candle.get("close_price"))
        candle_time = candle.get("timestamp_utc")
        if high_price is None or low_price is None or close_price is None:
            continue

        stop_touched = _stop_hit(direction, high_price, low_price, active_stop_price)
        partial_touched = _target_hit(direction, high_price, low_price, partial_price)
        full_target_touched = target_price is not None and _target_hit(direction, high_price, low_price, target_price)

        # Conservative mode assumes the stop happened first when OHLC cannot prove intrabar order.
        if intrabar_policy == "conservative" and stop_touched and (partial_touched or full_target_touched):
            exit_r = _directional_profit_r(direction, entry_price, active_stop_price, risk_points)
            realized_r += remaining_fraction * exit_r
            remaining_fraction = 0.0
            managed_exit_state = "managed_stop_first_conservative_same_candle"
            managed_exit_timestamp = candle_time
            managed_candles_to_exit = int(future_index + 1)
            break

        # Take a partial profit when the trade reaches the configured small-profit threshold.
        if not partial_exit_hit and partial_touched and partial_exit_fraction > 0:
            realized_r += partial_exit_fraction * partial_take_profit_r
            remaining_fraction = round(remaining_fraction - partial_exit_fraction, 10)
            partial_exit_hit = True
            partial_exit_timestamp = candle_time

        # After enough favorable movement, protect the remaining position at breakeven or small locked profit.
        favorable_r = _favorable_r_from_candle(direction, entry_price, risk_points, high_price, low_price)
        if favorable_r >= move_stop_to_breakeven_r:
            breakeven_or_profit_lock_activated = True
            if direction == "bearish":
                active_stop_price = min(active_stop_price, lock_stop_price)
            else:
                active_stop_price = max(active_stop_price, lock_stop_price)

        # Trail the stop only after the trade has already moved enough in the favorable direction.
        if favorable_r >= trailing_start_r:
            trailing_activated = True
            if direction == "bearish":
                candidate_stop = low_price + (trailing_distance_r * risk_points)
                active_stop_price = min(active_stop_price, candidate_stop)
            else:
                candidate_stop = high_price - (trailing_distance_r * risk_points)
                active_stop_price = max(active_stop_price, candidate_stop)

        # Close the remaining position at the selected full target when it is reached.
        if full_target_touched and remaining_fraction > 0:
            realized_r += remaining_fraction * (target_rr if target_rr is not None else 0.0)
            remaining_fraction = 0.0
            managed_exit_state = f"managed_{target_label.lower()}_target_exit" if target_label else "managed_target_exit"
            managed_exit_timestamp = candle_time
            managed_candles_to_exit = int(future_index + 1)
            break

        # Re-check the active stop after partial/breakeven/trailing updates.
        if _stop_hit(direction, high_price, low_price, active_stop_price) and remaining_fraction > 0:
            exit_r = _directional_profit_r(direction, entry_price, active_stop_price, risk_points)
            realized_r += remaining_fraction * exit_r
            remaining_fraction = 0.0
            if exit_r > 0:
                managed_exit_state = "managed_profit_locked_stop_exit"
            elif exit_r == 0:
                managed_exit_state = "managed_breakeven_stop_exit"
            else:
                managed_exit_state = "managed_loss_stop_exit"
            managed_exit_timestamp = candle_time
            managed_candles_to_exit = int(future_index + 1)
            break

        # Scalping systems may close the remaining position after a fixed holding time.
        if time_exit_candles is not None and int(future_index + 1) >= int(time_exit_candles):
            exit_r = _directional_profit_r(direction, entry_price, close_price, risk_points)
            realized_r += remaining_fraction * exit_r
            remaining_fraction = 0.0
            managed_exit_state = "managed_time_exit"
            managed_exit_timestamp = candle_time
            managed_candles_to_exit = int(future_index + 1)
            break

    # If nothing closed the trade, optionally mark the remaining position at the last known close.
    if remaining_fraction > 0 and exit_at_horizon_close and not future_candles.empty:
        last_candle = future_candles.iloc[-1]
        last_close = safe_float(last_candle.get("close_price"))
        if last_close is not None:
            exit_r = _directional_profit_r(direction, entry_price, last_close, risk_points)
            realized_r += remaining_fraction * exit_r
            remaining_fraction = 0.0
            managed_exit_state = "managed_horizon_close_exit"
            managed_exit_timestamp = last_candle.get("timestamp_utc")
            managed_candles_to_exit = len(future_candles)

    if managed_exit_state == "open_horizon":
        label_quality = "open_horizon"
        notes = "The managed trade stayed open within the available horizon."
        managed_positive_exit = None
    else:
        label_quality = "clean"
        managed_positive_exit = realized_r > 0
        notes = "Managed trade was simulated with partial exit, breakeven/profit lock, and trailing rules."

    return {
        "managed_exit_state": managed_exit_state,
        "managed_exit_timestamp": managed_exit_timestamp,
        "managed_realized_r": round(realized_r, 6),
        "managed_remaining_fraction": remaining_fraction,
        "managed_stop_price_at_exit": active_stop_price,
        "managed_candles_to_exit": managed_candles_to_exit,
        "partial_exit_hit": partial_exit_hit,
        "partial_exit_timestamp": partial_exit_timestamp,
        "breakeven_or_profit_lock_activated": breakeven_or_profit_lock_activated,
        "trailing_activated": trailing_activated,
        "selected_full_target": target_label,
        "selected_full_target_rr": target_rr,
        "managed_positive_exit": managed_positive_exit,
        "label_quality": label_quality,
        "notes": notes,
    }


def summarize_managed_outcomes(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a compact summary of managed-trade performance.
    """
    if results_df.empty:
        return pd.DataFrame(
            [
                {
                    "managed_trades": 0,
                    "positive_exit_count": 0,
                    "loss_exit_count": 0,
                    "breakeven_exit_count": 0,
                    "positive_exit_rate": 0.0,
                    "average_managed_r": None,
                    "median_managed_r": None,
                    "partial_exit_rate": 0.0,
                    "breakeven_or_lock_rate": 0.0,
                    "trailing_activation_rate": 0.0,
                    "improved_vs_original_rate": 0.0,
                    "management_state": "no_data",
                }
            ]
        )

    managed_r = pd.to_numeric(results_df["managed_realized_r"], errors="coerce")
    valid_r = managed_r.dropna()
    managed_trades = len(results_df)
    positive_count = int((valid_r > 0).sum())
    loss_count = int((valid_r < 0).sum())
    breakeven_count = int((valid_r == 0).sum())
    partial_count = int(results_df["partial_exit_hit"].fillna(False).sum())
    lock_count = int(results_df["breakeven_or_profit_lock_activated"].fillna(False).sum())
    trail_count = int(results_df["trailing_activated"].fillna(False).sum())
    improved_count = int(results_df["management_improved_vs_original"].fillna(False).sum())

    positive_rate = positive_count / managed_trades if managed_trades else 0.0
    improved_rate = improved_count / managed_trades if managed_trades else 0.0

    if managed_trades < 30:
        management_state = "small_sample_directional_only"
    elif positive_rate >= 0.55 and valid_r.mean() > 0:
        management_state = "management_rules_promising"
    elif positive_rate < 0.45 or valid_r.mean() <= 0:
        management_state = "management_rules_need_refinement"
    else:
        management_state = "management_rules_mixed"

    return pd.DataFrame(
        [
            {
                "managed_trades": managed_trades,
                "positive_exit_count": positive_count,
                "loss_exit_count": loss_count,
                "breakeven_exit_count": breakeven_count,
                "positive_exit_rate": round(positive_rate, 6),
                "average_managed_r": None if valid_r.empty else round(float(valid_r.mean()), 6),
                "median_managed_r": None if valid_r.empty else round(float(valid_r.median()), 6),
                "partial_exit_rate": round(partial_count / managed_trades, 6) if managed_trades else 0.0,
                "breakeven_or_lock_rate": round(lock_count / managed_trades, 6) if managed_trades else 0.0,
                "trailing_activation_rate": round(trail_count / managed_trades, 6) if managed_trades else 0.0,
                "improved_vs_original_rate": round(improved_rate, 6),
                "management_state": management_state,
            }
        ]
    )


def build_trend_channel_management_dataset(
    symbol: str,
    timeframes: str | list[str] | tuple[str, ...],
    analysis_limit: int = 300,
    dataset_limit: int = 2000,
    start_time: str | None = None,
    end_time: str | None = None,
    warmup_candles: int = 300,
    horizon_candles: int = 120,
    cutoff_step_candles: int = 20,
    step_size: int = 20,
    top_n_per_timeframe: int = 3,
    active_only: bool = True,
    recent_window: int | None = None,
    atr_window: int | None = None,
    event_lookback: int = 40,
    max_retest_candles: int = 20,
    max_false_breakout_candles: int = 10,
    max_confirmation_candles: int = 20,
    min_rejection_wick_ratio: float = 0.35,
    continuation_multiplier: float = 2.0,
    minimum_rr: float = 1.50,
    partial_take_profit_r: float = 0.50,
    partial_exit_fraction: float = 0.50,
    move_stop_to_breakeven_r: float = 0.75,
    locked_profit_r: float = 0.00,
    trailing_start_r: float = 1.25,
    trailing_distance_r: float = 0.75,
    full_target_mode: str = "nearest",
    time_exit_candles: int | None = None,
    exit_at_horizon_close: bool = True,
    intrabar_policy: str = "conservative",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Replays historical trade plans and tests active management rules on future candles.
    """
    timeframe_label = normalize_timeframes(timeframes)
    entry_timeframe = infer_entry_timeframe(timeframe_label)

    cutoff_schedule_df = build_cutoff_schedule(
        symbol=symbol,
        entry_timeframe=entry_timeframe,
        dataset_limit=dataset_limit,
        start_time=start_time,
        end_time=end_time,
        warmup_candles=warmup_candles,
        horizon_candles=horizon_candles,
        cutoff_step_candles=cutoff_step_candles,
    )

    if cutoff_schedule_df.empty:
        return pd.DataFrame(), summarize_managed_outcomes(pd.DataFrame()), cutoff_schedule_df

    result_rows: list[dict[str, Any]] = []
    management_profile = "partial_breakeven_trailing_v1"

    for _, cutoff_row in cutoff_schedule_df.iterrows():
        analysis_cutoff = cutoff_row["analysis_cutoff"]

        try:
            trade_plan_df, event_df, scenario_df, _, _ = build_trend_channel_trade_plan(
                symbol=symbol,
                timeframes=timeframe_label,
                limit=analysis_limit,
                step_size=step_size,
                top_n_per_timeframe=top_n_per_timeframe,
                active_only=active_only,
                recent_window=recent_window,
                atr_window=atr_window,
                event_lookback=event_lookback,
                max_retest_candles=max_retest_candles,
                max_false_breakout_candles=max_false_breakout_candles,
                max_confirmation_candles=max_confirmation_candles,
                min_rejection_wick_ratio=min_rejection_wick_ratio,
                continuation_multiplier=continuation_multiplier,
                minimum_rr=minimum_rr,
                end_time=str(analysis_cutoff),
                closed_candles_only=True,
            )
        except Exception:  # noqa: BLE001 - one failed cutoff should not stop the simulation.
            continue

        if trade_plan_df.empty:
            continue

        trade_plan_row = trade_plan_df.iloc[0]
        plan_state = safe_text(trade_plan_row.get("plan_state"))
        if not plan_state or not plan_state.startswith("valid_trade_plan"):
            continue

        entry_timestamp = safe_text(trade_plan_row.get("latest_timestamp"))
        row_entry_timeframe = safe_text(trade_plan_row.get("entry_timeframe")) or entry_timeframe
        future_candles = load_future_candles(
            symbol=symbol,
            timeframe=row_entry_timeframe,
            entry_timestamp=entry_timestamp,
            horizon_candles=horizon_candles,
        )

        original_outcome = simulate_trade_outcome(
            trade_plan_row=trade_plan_row,
            future_candles=future_candles,
            horizon_candles=horizon_candles,
        )
        managed_outcome = simulate_managed_trade_outcome(
            trade_plan_row=trade_plan_row,
            future_candles=future_candles,
            partial_take_profit_r=partial_take_profit_r,
            partial_exit_fraction=partial_exit_fraction,
            move_stop_to_breakeven_r=move_stop_to_breakeven_r,
            locked_profit_r=locked_profit_r,
            trailing_start_r=trailing_start_r,
            trailing_distance_r=trailing_distance_r,
            full_target_mode=full_target_mode,
            time_exit_candles=time_exit_candles,
            exit_at_horizon_close=exit_at_horizon_close,
            intrabar_policy=intrabar_policy,
        )

        original_r = original_outcome.get("best_target_rr_hit")
        if original_outcome.get("first_terminal_outcome") == "stop_loss_hit":
            original_r = -1.0
        elif original_outcome.get("first_terminal_outcome") is None:
            original_r = original_outcome.get("max_favorable_r")

        managed_r = managed_outcome.get("managed_realized_r")
        management_improved = None
        if managed_r is not None and original_r is not None:
            management_improved = float(managed_r) > float(original_r)

        event_row = event_df.iloc[0] if not event_df.empty else pd.Series(dtype="object")
        outcome = ManagedTrendChannelOutcome(
            symbol=symbol,
            timeframes=timeframe_label,
            analysis_cutoff=analysis_cutoff,
            entry_timeframe=row_entry_timeframe,
            entry_timestamp=entry_timestamp,
            plan_state=plan_state,
            plan_direction=safe_text(trade_plan_row.get("plan_direction")),
            event_state=safe_text(event_row.get("event_state")),
            entry_price=safe_float(trade_plan_row.get("entry_price")),
            original_stop_loss=safe_float(trade_plan_row.get("stop_loss")),
            original_risk_points=safe_float(trade_plan_row.get("risk_points")),
            available_targets=safe_text(trade_plan_row.get("available_targets")),
            best_available_target=safe_text(trade_plan_row.get("best_available_target")),
            best_available_rr=safe_float(trade_plan_row.get("best_available_rr")),
            original_outcome_state=safe_text(original_outcome.get("outcome_state")) or "unknown",
            original_first_terminal_outcome=safe_text(original_outcome.get("first_terminal_outcome")),
            original_max_favorable_r=safe_float(original_outcome.get("max_favorable_r")),
            original_max_adverse_r=safe_float(original_outcome.get("max_adverse_r")),
            management_profile=management_profile,
            partial_take_profit_r=partial_take_profit_r,
            partial_exit_fraction=partial_exit_fraction,
            move_stop_to_breakeven_r=move_stop_to_breakeven_r,
            locked_profit_r=locked_profit_r,
            trailing_start_r=trailing_start_r,
            trailing_distance_r=trailing_distance_r,
            full_target_mode=full_target_mode,
            selected_full_target=safe_text(managed_outcome.get("selected_full_target")),
            selected_full_target_rr=safe_float(managed_outcome.get("selected_full_target_rr")),
            partial_exit_hit=bool(managed_outcome.get("partial_exit_hit")),
            partial_exit_timestamp=managed_outcome.get("partial_exit_timestamp"),
            breakeven_or_profit_lock_activated=bool(managed_outcome.get("breakeven_or_profit_lock_activated")),
            trailing_activated=bool(managed_outcome.get("trailing_activated")),
            managed_exit_state=safe_text(managed_outcome.get("managed_exit_state")) or "unknown",
            managed_exit_timestamp=managed_outcome.get("managed_exit_timestamp"),
            managed_realized_r=safe_float(managed_outcome.get("managed_realized_r")),
            managed_remaining_fraction=safe_float(managed_outcome.get("managed_remaining_fraction")) or 0.0,
            managed_stop_price_at_exit=safe_float(managed_outcome.get("managed_stop_price_at_exit")),
            managed_candles_to_exit=managed_outcome.get("managed_candles_to_exit"),
            managed_positive_exit=managed_outcome.get("managed_positive_exit"),
            management_improved_vs_original=management_improved,
            label_quality=safe_text(managed_outcome.get("label_quality")) or "unknown",
            notes=safe_text(managed_outcome.get("notes")) or "",
        )
        result_rows.append(asdict(outcome))

    results_df = pd.DataFrame(result_rows)
    summary_df = summarize_managed_outcomes(results_df)
    return results_df, summary_df, cutoff_schedule_df
