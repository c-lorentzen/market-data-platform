from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from mdp.database.candle_loader import load_market_candles
from mdp.database.candle_loader_asof import get_timeframe_seconds
from mdp.structure.trend_channel_trade_plan import build_trend_channel_trade_plan


@dataclass(frozen=True)
class TrendChannelHistoricalOutcome:
    """
    Stores one cutoff-aware historical trade-plan outcome label.
    """

    symbol: str
    timeframes: str
    analysis_cutoff: Any
    entry_timeframe: str | None
    entry_timestamp: Any
    plan_state: str | None
    plan_direction: str | None
    event_state: str | None
    scenario_state: str | None
    plan_actionability: str | None
    entry_mode: str | None
    entry_price: float | None
    stop_loss: float | None
    risk_points: float | None
    risk_points_ratio: float | None
    tp1_price: float | None
    tp1_rr: float | None
    tp1_available: bool
    tp2_price: float | None
    tp2_rr: float | None
    tp2_available: bool
    tp3_price: float | None
    tp3_rr: float | None
    tp3_available: bool
    available_targets: str | None
    best_available_target: str | None
    best_available_rr: float | None
    event_quality_score: float | None
    risk_efficiency_state: str | None
    target_plan_state: str | None
    outcome_state: str
    first_terminal_outcome: str | None
    first_terminal_timestamp: Any
    first_target_hit: str | None
    first_target_timestamp: Any
    best_target_hit: str | None
    best_target_rr_hit: float | None
    stop_hit: bool
    stop_hit_timestamp: Any
    tp1_hit_before_stop: bool | None
    tp1_hit_timestamp: Any
    tp2_hit_before_stop: bool | None
    tp2_hit_timestamp: Any
    tp3_hit_before_stop: bool | None
    tp3_hit_timestamp: Any
    max_favorable_r: float | None
    max_adverse_r: float | None
    candles_to_first_terminal: int | None
    future_candles_analyzed: int
    horizon_candles: int
    label_quality: str
    notes: str


def safe_float(value: Any) -> float | None:
    """
    Converts numeric values safely while treating missing values as None.
    """
    # Pandas uses NaN for missing numeric cells, so this avoids spreading NaN values.
    if value is None or pd.isna(value):
        return None

    return float(value)


def safe_text(value: Any) -> str | None:
    """
    Converts text values safely while treating missing values as None.
    """
    if value is None or pd.isna(value):
        return None

    return str(value)


def normalize_timeframes(timeframes: str | list[str] | tuple[str, ...]) -> str:
    """
    Converts a timeframe list into a stable comma-separated label.
    """
    if isinstance(timeframes, str):
        return ",".join(part.strip().upper() for part in timeframes.replace(" ", ",").split(",") if part.strip())

    return ",".join(str(part).strip().upper() for part in timeframes if str(part).strip())


def infer_entry_timeframe(timeframes: str | list[str] | tuple[str, ...]) -> str:
    """
    Uses the smallest timeframe in the input as the entry/outcome timeframe.
    """
    ordered_timeframes = ["D1", "H4", "H1", "M15", "M5", "M1"]
    parsed_timeframes = normalize_timeframes(timeframes).split(",")

    # Walk backward through the ordered list so lower timeframes are preferred.
    for timeframe in reversed(ordered_timeframes):
        if timeframe in parsed_timeframes:
            return timeframe

    # Fall back to the last provided timeframe if the list is unusual.
    return parsed_timeframes[-1]


def calculate_analysis_cutoff_for_closed_candle(candle_open_time: Any, timeframe: str) -> pd.Timestamp:
    """
    Returns the analysis cutoff that makes this candle the latest fully closed candle.
    """
    timeframe_seconds = get_timeframe_seconds(timeframe)
    if timeframe_seconds is None:
        raise ValueError(f"Unsupported timeframe for cutoff schedule: {timeframe}")

    # The as-of loader subtracts one candle when closed_candles_only=True.
    # Therefore, pass the next candle-open time so this candle is included as closed.
    return pd.to_datetime(candle_open_time) + pd.Timedelta(seconds=timeframe_seconds)


def build_cutoff_schedule(
    symbol: str,
    entry_timeframe: str,
    dataset_limit: int = 2000,
    start_time: str | None = None,
    end_time: str | None = None,
    warmup_candles: int = 300,
    horizon_candles: int = 120,
    cutoff_step_candles: int = 20,
) -> pd.DataFrame:
    """
    Builds historical cutoff times with enough warmup candles before and future candles after.
    """
    candles = load_market_candles(
        symbol=symbol,
        timeframe=entry_timeframe,
        limit=dataset_limit,
        start_time=start_time,
        end_time=end_time,
    )

    if candles.empty:
        return pd.DataFrame()

    candles = candles.sort_values("timestamp_utc").reset_index(drop=True)
    timeframe_seconds = get_timeframe_seconds(entry_timeframe)
    if timeframe_seconds is None:
        raise ValueError(f"Unsupported entry timeframe: {entry_timeframe}")

    cutoff_rows: list[dict[str, Any]] = []
    last_allowed_index = len(candles) - horizon_candles - 1

    # Start after the warmup window so every cutoff has enough past data for analysis.
    for candle_index in range(warmup_candles - 1, last_allowed_index + 1, cutoff_step_candles):
        latest_closed_open = candles.loc[candle_index, "timestamp_utc"]
        analysis_cutoff = calculate_analysis_cutoff_for_closed_candle(
            candle_open_time=latest_closed_open,
            timeframe=entry_timeframe,
        )

        cutoff_rows.append(
            {
                "cutoff_id": len(cutoff_rows) + 1,
                "entry_timeframe": entry_timeframe,
                "latest_closed_open": latest_closed_open,
                "analysis_cutoff": analysis_cutoff,
                "candle_index": candle_index,
                "future_start_time": pd.to_datetime(latest_closed_open)
                + pd.Timedelta(seconds=timeframe_seconds),
                "future_end_time": pd.to_datetime(latest_closed_open)
                + pd.Timedelta(seconds=timeframe_seconds * horizon_candles),
            }
        )

    return pd.DataFrame(cutoff_rows)


def load_future_candles(
    symbol: str,
    timeframe: str,
    entry_timestamp: Any,
    horizon_candles: int,
) -> pd.DataFrame:
    """
    Loads candles after the entry candle so outcomes are measured only using future data.
    """
    timeframe_seconds = get_timeframe_seconds(timeframe)
    if timeframe_seconds is None:
        raise ValueError(f"Unsupported timeframe for future outcome loading: {timeframe}")

    entry_timestamp = pd.to_datetime(entry_timestamp)
    future_start = entry_timestamp + pd.Timedelta(seconds=timeframe_seconds)
    future_end = entry_timestamp + pd.Timedelta(seconds=timeframe_seconds * horizon_candles)

    future_candles = load_market_candles(
        symbol=symbol,
        timeframe=timeframe,
        start_time=str(future_start),
        end_time=str(future_end),
    )

    if future_candles.empty:
        return pd.DataFrame()

    return future_candles.sort_values("timestamp_utc").reset_index(drop=True)


def calculate_trade_excursions(
    direction: str | None,
    entry_price: float | None,
    risk_points: float | None,
    future_candles: pd.DataFrame,
) -> tuple[float | None, float | None]:
    """
    Calculates max favorable/adverse movement in R units over the outcome horizon.
    """
    if direction not in {"bullish", "bearish"} or entry_price is None or risk_points is None or risk_points <= 0:
        return None, None

    max_favorable_points = 0.0
    max_adverse_points = 0.0

    for _, candle in future_candles.iterrows():
        high_price = safe_float(candle.get("high_price"))
        low_price = safe_float(candle.get("low_price"))
        if high_price is None or low_price is None:
            continue

        if direction == "bullish":
            favorable_points = high_price - entry_price
            adverse_points = entry_price - low_price
        else:
            favorable_points = entry_price - low_price
            adverse_points = high_price - entry_price

        max_favorable_points = max(max_favorable_points, favorable_points)
        max_adverse_points = max(max_adverse_points, adverse_points)

    return max_favorable_points / risk_points, max_adverse_points / risk_points


def get_target_price(trade_plan_row: pd.Series, target_label: str) -> float | None:
    """
    Reads a target price from a trade-plan row.
    """
    return safe_float(trade_plan_row.get(f"{target_label.lower()}_price"))


def get_target_rr(trade_plan_row: pd.Series, target_label: str) -> float | None:
    """
    Reads a target RR value from a trade-plan row.
    """
    return safe_float(trade_plan_row.get(f"{target_label.lower()}_rr"))


def is_target_available(trade_plan_row: pd.Series, target_label: str) -> bool:
    """
    Checks whether a target passed the trade-plan availability filter.
    """
    return bool(trade_plan_row.get(f"{target_label.lower()}_available"))


def simulate_trade_outcome(
    trade_plan_row: pd.Series,
    future_candles: pd.DataFrame,
    horizon_candles: int,
) -> dict[str, Any]:
    """
    Labels whether SL or available TP levels were reached first after the entry candle.
    """
    direction = safe_text(trade_plan_row.get("plan_direction"))
    entry_price = safe_float(trade_plan_row.get("entry_price"))
    stop_loss = safe_float(trade_plan_row.get("stop_loss"))
    risk_points = safe_float(trade_plan_row.get("risk_points"))

    if direction not in {"bullish", "bearish"} or entry_price is None or stop_loss is None or risk_points is None or risk_points <= 0:
        return {
            "outcome_state": "invalid_trade_plan_levels",
            "first_terminal_outcome": None,
            "first_terminal_timestamp": None,
            "first_target_hit": None,
            "first_target_timestamp": None,
            "best_target_hit": None,
            "best_target_rr_hit": None,
            "stop_hit": False,
            "stop_hit_timestamp": None,
            "tp1_hit_before_stop": None,
            "tp1_hit_timestamp": None,
            "tp2_hit_before_stop": None,
            "tp2_hit_timestamp": None,
            "tp3_hit_before_stop": None,
            "tp3_hit_timestamp": None,
            "max_favorable_r": None,
            "max_adverse_r": None,
            "candles_to_first_terminal": None,
            "future_candles_analyzed": len(future_candles),
            "label_quality": "invalid",
            "notes": "Trade-plan levels were invalid, so no outcome label was created.",
        }

    target_labels = ["tp1", "tp2", "tp3"]
    target_prices = {label: get_target_price(trade_plan_row, label) for label in target_labels}
    target_rrs = {label: get_target_rr(trade_plan_row, label) for label in target_labels}
    target_available = {label: is_target_available(trade_plan_row, label) for label in target_labels}

    stop_hit_timestamp = None
    first_terminal_outcome = None
    first_terminal_timestamp = None
    first_target_hit = None
    first_target_timestamp = None
    best_target_hit = None
    best_target_rr_hit = None
    candles_to_first_terminal = None

    target_hit_timestamps: dict[str, Any] = {"tp1": None, "tp2": None, "tp3": None}
    target_hit_before_stop: dict[str, bool | None] = {
        label: None if not target_available[label] else False for label in target_labels
    }

    for future_index, candle in future_candles.iterrows():
        high_price = safe_float(candle.get("high_price"))
        low_price = safe_float(candle.get("low_price"))
        candle_time = candle.get("timestamp_utc")
        if high_price is None or low_price is None:
            continue

        if direction == "bullish":
            stop_hit_this_candle = low_price <= stop_loss
            targets_hit_this_candle = [
                label
                for label in target_labels
                if target_available[label]
                and target_prices[label] is not None
                and high_price >= target_prices[label]
            ]
        else:
            stop_hit_this_candle = high_price >= stop_loss
            targets_hit_this_candle = [
                label
                for label in target_labels
                if target_available[label]
                and target_prices[label] is not None
                and low_price <= target_prices[label]
            ]

        # Record every target reached before the stop is observed.
        if stop_hit_timestamp is None:
            for label in targets_hit_this_candle:
                if target_hit_timestamps[label] is None:
                    target_hit_timestamps[label] = candle_time
                    target_hit_before_stop[label] = True

        if stop_hit_this_candle and stop_hit_timestamp is None:
            stop_hit_timestamp = candle_time

        if first_terminal_outcome is None and (stop_hit_this_candle or targets_hit_this_candle):
            candles_to_first_terminal = int(future_index + 1)

            if stop_hit_this_candle and targets_hit_this_candle:
                # OHLC data cannot reveal intrabar order, so mark this as ambiguous.
                first_terminal_outcome = "ambiguous_stop_and_target_same_candle"
                first_terminal_timestamp = candle_time
                first_target_hit = targets_hit_this_candle[0]
                first_target_timestamp = candle_time
                best_target_hit = max(
                    targets_hit_this_candle,
                    key=lambda label: target_rrs[label] if target_rrs[label] is not None else -999,
                )
                best_target_rr_hit = target_rrs.get(best_target_hit)
                break

            if targets_hit_this_candle:
                # The nearest available target is treated as the first target hit.
                first_target_hit = targets_hit_this_candle[0]
                first_target_timestamp = candle_time
                best_target_hit = max(
                    targets_hit_this_candle,
                    key=lambda label: target_rrs[label] if target_rrs[label] is not None else -999,
                )
                best_target_rr_hit = target_rrs.get(best_target_hit)
                first_terminal_outcome = f"{first_target_hit}_hit_before_stop"
                first_terminal_timestamp = candle_time
                break

            first_terminal_outcome = "stop_loss_hit"
            first_terminal_timestamp = candle_time
            break

    max_favorable_r, max_adverse_r = calculate_trade_excursions(
        direction=direction,
        entry_price=entry_price,
        risk_points=risk_points,
        future_candles=future_candles,
    )

    if first_terminal_outcome is None:
        outcome_state = "no_terminal_outcome_within_horizon"
        label_quality = "open_horizon"
        notes = "Neither stop loss nor available targets were reached within the future horizon."
    elif first_terminal_outcome.startswith("ambiguous"):
        outcome_state = "ambiguous_terminal_outcome"
        label_quality = "ambiguous"
        notes = "Stop and target were touched in the same candle, so intrabar order is unknown."
    elif first_terminal_outcome == "stop_loss_hit":
        outcome_state = "stop_loss_first"
        label_quality = "clean"
        notes = "Stop loss was reached before any available target."
    else:
        outcome_state = "target_first"
        label_quality = "clean"
        notes = "An available target was reached before the stop loss."

    # If the stop was never hit, any target reached is still marked as before-stop.
    for label in target_labels:
        if not target_available[label]:
            target_hit_before_stop[label] = None
        elif target_hit_timestamps[label] is not None and stop_hit_timestamp is None:
            target_hit_before_stop[label] = True

    return {
        "outcome_state": outcome_state,
        "first_terminal_outcome": first_terminal_outcome,
        "first_terminal_timestamp": first_terminal_timestamp,
        "first_target_hit": first_target_hit,
        "first_target_timestamp": first_target_timestamp,
        "best_target_hit": best_target_hit,
        "best_target_rr_hit": best_target_rr_hit,
        "stop_hit": stop_hit_timestamp is not None,
        "stop_hit_timestamp": stop_hit_timestamp,
        "tp1_hit_before_stop": target_hit_before_stop["tp1"],
        "tp1_hit_timestamp": target_hit_timestamps["tp1"],
        "tp2_hit_before_stop": target_hit_before_stop["tp2"],
        "tp2_hit_timestamp": target_hit_timestamps["tp2"],
        "tp3_hit_before_stop": target_hit_before_stop["tp3"],
        "tp3_hit_timestamp": target_hit_timestamps["tp3"],
        "max_favorable_r": max_favorable_r,
        "max_adverse_r": max_adverse_r,
        "candles_to_first_terminal": candles_to_first_terminal,
        "future_candles_analyzed": len(future_candles),
        "label_quality": label_quality,
        "notes": notes,
    }


def build_non_actionable_outcome_row(
    symbol: str,
    timeframes: str,
    analysis_cutoff: Any,
    trade_plan_df: pd.DataFrame,
    event_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    horizon_candles: int,
    reason: str,
) -> dict[str, Any]:
    """
    Builds a dataset row for cutoffs that did not produce an actionable trade plan.
    """
    trade_plan_row = trade_plan_df.iloc[0] if not trade_plan_df.empty else pd.Series(dtype="object")
    event_row = event_df.iloc[0] if not event_df.empty else pd.Series(dtype="object")
    scenario_row = scenario_df.iloc[0] if not scenario_df.empty else pd.Series(dtype="object")

    # Non-actionable rows are still useful to measure signal frequency.
    outcome = TrendChannelHistoricalOutcome(
        symbol=symbol,
        timeframes=timeframes,
        analysis_cutoff=analysis_cutoff,
        entry_timeframe=safe_text(trade_plan_row.get("entry_timeframe")),
        entry_timestamp=safe_text(trade_plan_row.get("latest_timestamp")),
        plan_state=safe_text(trade_plan_row.get("plan_state")),
        plan_direction=safe_text(trade_plan_row.get("plan_direction")),
        event_state=safe_text(event_row.get("event_state")),
        scenario_state=safe_text(scenario_row.get("scenario_state")),
        plan_actionability=safe_text(trade_plan_row.get("plan_actionability")),
        entry_mode=safe_text(trade_plan_row.get("entry_mode")),
        entry_price=safe_float(trade_plan_row.get("entry_price")),
        stop_loss=safe_float(trade_plan_row.get("stop_loss")),
        risk_points=safe_float(trade_plan_row.get("risk_points")),
        risk_points_ratio=safe_float(trade_plan_row.get("risk_points_ratio")),
        tp1_price=safe_float(trade_plan_row.get("tp1_price")),
        tp1_rr=safe_float(trade_plan_row.get("tp1_rr")),
        tp1_available=bool(trade_plan_row.get("tp1_available")) if not trade_plan_row.empty else False,
        tp2_price=safe_float(trade_plan_row.get("tp2_price")),
        tp2_rr=safe_float(trade_plan_row.get("tp2_rr")),
        tp2_available=bool(trade_plan_row.get("tp2_available")) if not trade_plan_row.empty else False,
        tp3_price=safe_float(trade_plan_row.get("tp3_price")),
        tp3_rr=safe_float(trade_plan_row.get("tp3_rr")),
        tp3_available=bool(trade_plan_row.get("tp3_available")) if not trade_plan_row.empty else False,
        available_targets=safe_text(trade_plan_row.get("available_targets")),
        best_available_target=safe_text(trade_plan_row.get("best_available_target")),
        best_available_rr=safe_float(trade_plan_row.get("best_available_rr")),
        event_quality_score=safe_float(trade_plan_row.get("event_quality_score")),
        risk_efficiency_state=safe_text(trade_plan_row.get("risk_efficiency_state")),
        target_plan_state=safe_text(trade_plan_row.get("target_plan_state")),
        outcome_state="not_actionable",
        first_terminal_outcome=None,
        first_terminal_timestamp=None,
        first_target_hit=None,
        first_target_timestamp=None,
        best_target_hit=None,
        best_target_rr_hit=None,
        stop_hit=False,
        stop_hit_timestamp=None,
        tp1_hit_before_stop=None,
        tp1_hit_timestamp=None,
        tp2_hit_before_stop=None,
        tp2_hit_timestamp=None,
        tp3_hit_before_stop=None,
        tp3_hit_timestamp=None,
        max_favorable_r=None,
        max_adverse_r=None,
        candles_to_first_terminal=None,
        future_candles_analyzed=0,
        horizon_candles=horizon_candles,
        label_quality="not_actionable",
        notes=reason,
    )

    return asdict(outcome)


def build_actionable_outcome_row(
    symbol: str,
    timeframes: str,
    analysis_cutoff: Any,
    trade_plan_df: pd.DataFrame,
    event_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    horizon_candles: int,
) -> dict[str, Any]:
    """
    Simulates future candles for one actionable trade plan and returns a labeled row.
    """
    trade_plan_row = trade_plan_df.iloc[0]
    event_row = event_df.iloc[0] if not event_df.empty else pd.Series(dtype="object")
    scenario_row = scenario_df.iloc[0] if not scenario_df.empty else pd.Series(dtype="object")

    entry_timeframe = safe_text(trade_plan_row.get("entry_timeframe")) or infer_entry_timeframe(timeframes)
    entry_timestamp = safe_text(trade_plan_row.get("latest_timestamp"))

    future_candles = load_future_candles(
        symbol=symbol,
        timeframe=entry_timeframe,
        entry_timestamp=entry_timestamp,
        horizon_candles=horizon_candles,
    )

    outcome_data = simulate_trade_outcome(
        trade_plan_row=trade_plan_row,
        future_candles=future_candles,
        horizon_candles=horizon_candles,
    )

    outcome = TrendChannelHistoricalOutcome(
        symbol=symbol,
        timeframes=timeframes,
        analysis_cutoff=analysis_cutoff,
        entry_timeframe=entry_timeframe,
        entry_timestamp=entry_timestamp,
        plan_state=safe_text(trade_plan_row.get("plan_state")),
        plan_direction=safe_text(trade_plan_row.get("plan_direction")),
        event_state=safe_text(event_row.get("event_state")),
        scenario_state=safe_text(scenario_row.get("scenario_state")),
        plan_actionability=safe_text(trade_plan_row.get("plan_actionability")),
        entry_mode=safe_text(trade_plan_row.get("entry_mode")),
        entry_price=safe_float(trade_plan_row.get("entry_price")),
        stop_loss=safe_float(trade_plan_row.get("stop_loss")),
        risk_points=safe_float(trade_plan_row.get("risk_points")),
        risk_points_ratio=safe_float(trade_plan_row.get("risk_points_ratio")),
        tp1_price=safe_float(trade_plan_row.get("tp1_price")),
        tp1_rr=safe_float(trade_plan_row.get("tp1_rr")),
        tp1_available=bool(trade_plan_row.get("tp1_available")),
        tp2_price=safe_float(trade_plan_row.get("tp2_price")),
        tp2_rr=safe_float(trade_plan_row.get("tp2_rr")),
        tp2_available=bool(trade_plan_row.get("tp2_available")),
        tp3_price=safe_float(trade_plan_row.get("tp3_price")),
        tp3_rr=safe_float(trade_plan_row.get("tp3_rr")),
        tp3_available=bool(trade_plan_row.get("tp3_available")),
        available_targets=safe_text(trade_plan_row.get("available_targets")),
        best_available_target=safe_text(trade_plan_row.get("best_available_target")),
        best_available_rr=safe_float(trade_plan_row.get("best_available_rr")),
        event_quality_score=safe_float(trade_plan_row.get("event_quality_score")),
        risk_efficiency_state=safe_text(trade_plan_row.get("risk_efficiency_state")),
        target_plan_state=safe_text(trade_plan_row.get("target_plan_state")),
        outcome_state=outcome_data["outcome_state"],
        first_terminal_outcome=outcome_data["first_terminal_outcome"],
        first_terminal_timestamp=outcome_data["first_terminal_timestamp"],
        first_target_hit=outcome_data["first_target_hit"],
        first_target_timestamp=outcome_data["first_target_timestamp"],
        best_target_hit=outcome_data["best_target_hit"],
        best_target_rr_hit=outcome_data["best_target_rr_hit"],
        stop_hit=outcome_data["stop_hit"],
        stop_hit_timestamp=outcome_data["stop_hit_timestamp"],
        tp1_hit_before_stop=outcome_data["tp1_hit_before_stop"],
        tp1_hit_timestamp=outcome_data["tp1_hit_timestamp"],
        tp2_hit_before_stop=outcome_data["tp2_hit_before_stop"],
        tp2_hit_timestamp=outcome_data["tp2_hit_timestamp"],
        tp3_hit_before_stop=outcome_data["tp3_hit_before_stop"],
        tp3_hit_timestamp=outcome_data["tp3_hit_timestamp"],
        max_favorable_r=outcome_data["max_favorable_r"],
        max_adverse_r=outcome_data["max_adverse_r"],
        candles_to_first_terminal=outcome_data["candles_to_first_terminal"],
        future_candles_analyzed=outcome_data["future_candles_analyzed"],
        horizon_candles=horizon_candles,
        label_quality=outcome_data["label_quality"],
        notes=outcome_data["notes"],
    )

    return asdict(outcome)


def summarize_outcome_dataset(outcomes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes signal frequency and outcome quality for the generated dataset.
    """
    if outcomes_df.empty:
        return pd.DataFrame(
            [
                {
                    "total_cutoffs": 0,
                    "actionable_plans": 0,
                    "clean_labels": 0,
                    "target_first_count": 0,
                    "stop_first_count": 0,
                    "ambiguous_count": 0,
                    "open_horizon_count": 0,
                    "actionable_rate": 0.0,
                    "target_first_rate": 0.0,
                    "stop_first_rate": 0.0,
                }
            ]
        )

    total_cutoffs = len(outcomes_df)
    actionable_mask = outcomes_df["outcome_state"] != "not_actionable"
    actionable_count = int(actionable_mask.sum())
    clean_count = int((outcomes_df["label_quality"] == "clean").sum())
    target_first_count = int((outcomes_df["outcome_state"] == "target_first").sum())
    stop_first_count = int((outcomes_df["outcome_state"] == "stop_loss_first").sum())
    ambiguous_count = int((outcomes_df["outcome_state"] == "ambiguous_terminal_outcome").sum())
    open_horizon_count = int((outcomes_df["outcome_state"] == "no_terminal_outcome_within_horizon").sum())

    return pd.DataFrame(
        [
            {
                "total_cutoffs": total_cutoffs,
                "actionable_plans": actionable_count,
                "clean_labels": clean_count,
                "target_first_count": target_first_count,
                "stop_first_count": stop_first_count,
                "ambiguous_count": ambiguous_count,
                "open_horizon_count": open_horizon_count,
                "actionable_rate": actionable_count / total_cutoffs if total_cutoffs else 0.0,
                "target_first_rate": target_first_count / clean_count if clean_count else 0.0,
                "stop_first_rate": stop_first_count / clean_count if clean_count else 0.0,
            }
        ]
    )


def build_historical_trend_channel_outcome_dataset(
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
    include_non_actionable: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Replays many historical cutoffs and labels future SL/TP outcomes.
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
        return pd.DataFrame(), summarize_outcome_dataset(pd.DataFrame()), cutoff_schedule_df

    outcome_rows: list[dict[str, Any]] = []

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
        except Exception as error:  # noqa: BLE001 - dataset generation should keep moving across cutoffs.
            outcome_rows.append(
                {
                    "symbol": symbol,
                    "timeframes": timeframe_label,
                    "analysis_cutoff": analysis_cutoff,
                    "outcome_state": "pipeline_error",
                    "label_quality": "error",
                    "notes": f"Pipeline failed at this cutoff: {error}",
                }
            )
            continue

        plan_state = safe_text(trade_plan_df.iloc[0].get("plan_state")) if not trade_plan_df.empty else None

        if not plan_state or not plan_state.startswith("valid_trade_plan"):
            if include_non_actionable:
                outcome_rows.append(
                    build_non_actionable_outcome_row(
                        symbol=symbol,
                        timeframes=timeframe_label,
                        analysis_cutoff=analysis_cutoff,
                        trade_plan_df=trade_plan_df,
                        event_df=event_df,
                        scenario_df=scenario_df,
                        horizon_candles=horizon_candles,
                        reason="No actionable trade plan existed at this historical cutoff.",
                    )
                )
            continue

        outcome_rows.append(
            build_actionable_outcome_row(
                symbol=symbol,
                timeframes=timeframe_label,
                analysis_cutoff=analysis_cutoff,
                trade_plan_df=trade_plan_df,
                event_df=event_df,
                scenario_df=scenario_df,
                horizon_candles=horizon_candles,
            )
        )

    outcomes_df = pd.DataFrame(outcome_rows)
    summary_df = summarize_outcome_dataset(outcomes_df)

    return outcomes_df, summary_df, cutoff_schedule_df
