from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from mdp.database.candle_loader import load_market_candles
from mdp.database.candle_loader_asof import load_market_candles_as_of
from mdp.structure.trend_channels import prepare_channel_candles
from mdp.structure.trend_channel_scenarios import analyze_trend_channel_scenario


@dataclass(frozen=True)
class TrendChannelEventAnalysis:
    """
    Stores scale-aware breakout, retest, false-breakout, and sequence context.
    """

    symbol: str
    timeframes: str
    scenario_state: str | None
    scenario_direction: str
    actionability: str
    event_state: str
    event_direction: str | None
    event_timeframe: str | None
    event_channel_role: str | None
    event_channel_type: str | None
    event_channel_current_state: str | None
    breakout_detected: bool
    breakout_timestamp: Any
    retest_detected: bool
    retest_timestamp: Any
    retest_rejected: bool
    retest_rejection_timestamp: Any
    false_breakout_detected: bool
    false_breakout_timestamp: Any
    breakout_impulse_confirmed: bool
    breakout_impulse_timestamp: Any
    continuation_after_retest_confirmed: bool
    continuation_after_retest_timestamp: Any
    event_boundary_price: float | None
    event_trigger_price: float | None
    event_retest_lower: float | None
    event_retest_upper: float | None
    event_invalidation_price: float | None
    suggested_entry_mode: str | None
    suggested_entry_price: float | None
    suggested_stop_loss: float | None
    suggested_tp1: float | None
    suggested_tp2: float | None
    suggested_rr_tp1: float | None
    suggested_rr_tp2: float | None
    risk_efficiency_state: str
    adaptive_breakout_threshold: float | None
    adaptive_retest_tolerance: float | None
    adaptive_stop_buffer: float | None
    adaptive_min_target_distance: float | None
    latest_close: float | None
    latest_timestamp: Any
    event_quality_score: float | None
    notes: str


def safe_float(value: Any) -> float | None:
    """
    Converts numeric values safely while treating NaN as missing.
    """
    # Pandas frequently returns NaN for missing numeric cells.
    if value is None or pd.isna(value):
        return None

    return float(value)


def safe_text(value: Any) -> str | None:
    """
    Converts text values safely while treating NaN as missing.
    """
    if value is None or pd.isna(value):
        return None

    return str(value)


def get_candle_price(candle: pd.Series | None, column_name: str) -> float | None:
    """
    Reads one OHLC value from a candle safely.
    """
    if candle is None:
        return None

    return safe_float(candle.get(column_name))


def infer_event_direction(scenario_row: pd.Series | None) -> str | None:
    """
    Infers the event direction from the scenario output.
    """
    if scenario_row is None:
        return None

    scenario_direction = safe_text(scenario_row.get("scenario_direction")) or ""
    scenario_state = safe_text(scenario_row.get("scenario_state")) or ""

    # Watch states still carry directional intent even when they are not final signals.
    combined_text = f"{scenario_direction} {scenario_state}".lower()

    if "bearish" in combined_text:
        return "bearish"

    if "bullish" in combined_text:
        return "bullish"

    return None


def get_timeframe_candidates(candidates_df: pd.DataFrame, timeframe: str | None) -> pd.DataFrame:
    """
    Returns candidate channels for one timeframe.
    """
    if candidates_df.empty or timeframe is None or "timeframe" not in candidates_df.columns:
        return pd.DataFrame()

    return candidates_df[candidates_df["timeframe"] == timeframe].copy()


def sort_candidates(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts candidates using quality-first logic.
    """
    if candidates_df.empty:
        return candidates_df

    sort_columns = [
        column
        for column in ["rolling_quality_score", "r_squared", "containment_rate"]
        if column in candidates_df.columns
    ]

    if not sort_columns:
        return candidates_df

    return candidates_df.sort_values(by=sort_columns, ascending=[False] * len(sort_columns))


def pick_event_candidate(
    candidates_df: pd.DataFrame,
    entry_timeframe: str | None,
    event_direction: str | None,
) -> pd.Series | None:
    """
    Picks the channel most relevant to breakout/retest event detection.
    """
    timeframe_candidates = get_timeframe_candidates(candidates_df, entry_timeframe)

    if timeframe_candidates.empty:
        return None

    # For bearish events, a broken ascending channel is the cleanest micro-breakdown evidence.
    if event_direction == "bearish":
        breakout_df = timeframe_candidates[
            (timeframe_candidates["channel_role"] == "active_breakout_context")
            & (timeframe_candidates["channel_type"] == "ascending")
            & (timeframe_candidates["current_state"] == "below_channel")
        ].copy()
        if not breakout_df.empty:
            return sort_candidates(breakout_df).iloc[0]

    # For bullish events, a broken descending channel is the cleanest micro-breakout evidence.
    if event_direction == "bullish":
        breakout_df = timeframe_candidates[
            (timeframe_candidates["channel_role"] == "active_breakout_context")
            & (timeframe_candidates["channel_type"] == "descending")
            & (timeframe_candidates["current_state"] == "above_channel")
        ].copy()
        if not breakout_df.empty:
            return sort_candidates(breakout_df).iloc[0]

    # Fall back to the strongest active entry channel when no broken micro-channel exists.
    entry_df = timeframe_candidates[
        timeframe_candidates["channel_role"] == "active_entry_context"
    ].copy()
    if not entry_df.empty:
        return sort_candidates(entry_df).iloc[0]

    return sort_candidates(timeframe_candidates).iloc[0]


def calculate_base_event_levels(
    event_candidate: pd.Series | None,
    event_direction: str | None,
    scenario_row: pd.Series | None,
) -> dict[str, float | None]:
    """
    Builds adaptive breakout, retest, invalidation, and base target distances.
    """
    if event_candidate is None or event_direction is None:
        return {
            "event_boundary_price": None,
            "event_trigger_price": None,
            "event_retest_lower": None,
            "event_retest_upper": None,
            "event_invalidation_price": None,
            "raw_micro_stop_loss": None,
            "adaptive_breakout_threshold": None,
            "adaptive_retest_tolerance": None,
            "adaptive_stop_buffer": None,
            "adaptive_min_target_distance": None,
            "structure_min_target_distance": None,
        }

    current_lower = safe_float(event_candidate.get("current_lower"))
    current_upper = safe_float(event_candidate.get("current_upper"))
    breakout_threshold = safe_float(event_candidate.get("adaptive_breakout_threshold")) or 0.0
    retest_tolerance = safe_float(event_candidate.get("adaptive_retest_tolerance")) or 0.0
    stop_buffer = safe_float(event_candidate.get("adaptive_stop_buffer")) or 0.0
    min_target_distance = safe_float(event_candidate.get("adaptive_min_target_distance")) or 0.0
    structure_min_target = safe_float(scenario_row.get("structure_min_target_distance")) if scenario_row is not None else None

    if event_direction == "bearish":
        # Bearish events break below the lower channel boundary.
        boundary_price = current_lower
        trigger_price = current_lower - breakout_threshold if current_lower is not None else None
        retest_lower = current_lower - retest_tolerance if current_lower is not None else None
        retest_upper = current_lower + retest_tolerance if current_lower is not None else None
        raw_micro_stop = current_upper + stop_buffer if current_upper is not None else None
        scenario_invalidation = safe_float(scenario_row.get("bearish_invalidation_price")) if scenario_row is not None else None
    else:
        # Bullish events break above the upper channel boundary.
        boundary_price = current_upper
        trigger_price = current_upper + breakout_threshold if current_upper is not None else None
        retest_lower = current_upper - retest_tolerance if current_upper is not None else None
        retest_upper = current_upper + retest_tolerance if current_upper is not None else None
        raw_micro_stop = current_lower - stop_buffer if current_lower is not None else None
        scenario_invalidation = safe_float(scenario_row.get("bullish_invalidation_price")) if scenario_row is not None else None

    return {
        "event_boundary_price": boundary_price,
        "event_trigger_price": trigger_price,
        "event_retest_lower": retest_lower,
        "event_retest_upper": retest_upper,
        "event_invalidation_price": scenario_invalidation or raw_micro_stop,
        "raw_micro_stop_loss": raw_micro_stop,
        "adaptive_breakout_threshold": breakout_threshold,
        "adaptive_retest_tolerance": retest_tolerance,
        "adaptive_stop_buffer": stop_buffer,
        "adaptive_min_target_distance": min_target_distance,
        "structure_min_target_distance": structure_min_target,
    }


def pick_stop_loss(
    direction: str | None,
    entry_price: float | None,
    raw_micro_stop: float | None,
    scenario_invalidation: float | None,
) -> float | None:
    """
    Caps the stop loss with structure invalidation when it gives a valid tighter risk.
    """
    if direction is None or entry_price is None:
        return None

    valid_stops: list[float] = []

    if direction == "bearish":
        # Bearish stops must be above the entry price.
        for candidate_stop in [raw_micro_stop, scenario_invalidation]:
            if candidate_stop is not None and candidate_stop > entry_price:
                valid_stops.append(candidate_stop)
        return min(valid_stops) if valid_stops else None

    # Bullish stops must be below the entry price.
    for candidate_stop in [raw_micro_stop, scenario_invalidation]:
        if candidate_stop is not None and candidate_stop < entry_price:
            valid_stops.append(candidate_stop)
    return max(valid_stops) if valid_stops else None


def calculate_rr(
    direction: str | None,
    entry_price: float | None,
    stop_loss: float | None,
    take_profit: float | None,
) -> float | None:
    """
    Calculates reward/risk for a proposed level set.
    """
    if direction is None or entry_price is None or stop_loss is None or take_profit is None:
        return None

    if direction == "bearish":
        risk = stop_loss - entry_price
        reward = entry_price - take_profit
    else:
        risk = entry_price - stop_loss
        reward = take_profit - entry_price

    # Invalid or zero risk makes the level set unusable.
    if risk <= 0 or reward <= 0:
        return None

    return float(reward / risk)


def find_first_breakout_candle(
    market_df: pd.DataFrame,
    direction: str | None,
    trigger_price: float | None,
    event_lookback: int,
) -> pd.Series | None:
    """
    Finds the first recent closed candle that breaks the directional trigger.
    """
    if direction is None or trigger_price is None or market_df.empty:
        return None

    recent_df = market_df.tail(event_lookback).copy()

    if direction == "bearish":
        breakout_df = recent_df[recent_df["close_price"] <= trigger_price].copy()
    else:
        breakout_df = recent_df[recent_df["close_price"] >= trigger_price].copy()

    if breakout_df.empty:
        return None

    # Use the first breakout in the lookback window so sequence logic can scan after it.
    return breakout_df.iloc[0]


def get_candles_after(
    market_df: pd.DataFrame,
    anchor_candle: pd.Series | None,
    max_candles: int,
    include_anchor: bool = False,
) -> pd.DataFrame:
    """
    Returns candles after an anchor candle while preserving sequence order.
    """
    if anchor_candle is None or market_df.empty:
        return pd.DataFrame()

    anchor_index = int(anchor_candle["candle_index"])
    if include_anchor:
        return market_df[market_df["candle_index"] >= anchor_index].head(max_candles).copy()

    return market_df[market_df["candle_index"] > anchor_index].head(max_candles).copy()


def find_retest_candle(
    market_df: pd.DataFrame,
    breakout_candle: pd.Series | None,
    retest_lower: float | None,
    retest_upper: float | None,
    max_retest_candles: int,
) -> pd.Series | None:
    """
    Finds the first candle after breakout that touches the broken-boundary retest zone.
    """
    if breakout_candle is None or retest_lower is None or retest_upper is None:
        return None

    after_breakout_df = get_candles_after(
        market_df=market_df,
        anchor_candle=breakout_candle,
        max_candles=max_retest_candles,
        include_anchor=False,
    )

    if after_breakout_df.empty:
        return None

    # A retest is any overlap between the candle range and the adaptive retest zone.
    retest_df = after_breakout_df[
        (after_breakout_df["high_price"] >= retest_lower)
        & (after_breakout_df["low_price"] <= retest_upper)
    ].copy()

    if retest_df.empty:
        return None

    return retest_df.iloc[0]


def find_false_breakout_candle(
    market_df: pd.DataFrame,
    direction: str | None,
    breakout_candle: pd.Series | None,
    retest_lower: float | None,
    retest_upper: float | None,
    max_false_breakout_candles: int,
) -> pd.Series | None:
    """
    Checks whether price returned back inside the broken channel soon after breakout.
    """
    if direction is None or breakout_candle is None or retest_lower is None or retest_upper is None:
        return None

    after_breakout_df = get_candles_after(
        market_df=market_df,
        anchor_candle=breakout_candle,
        max_candles=max_false_breakout_candles,
        include_anchor=False,
    )

    if after_breakout_df.empty:
        return None

    # A bearish breakdown fails when price closes back above the retest zone.
    if direction == "bearish":
        false_breakout_df = after_breakout_df[after_breakout_df["close_price"] > retest_upper].copy()
    else:
        false_breakout_df = after_breakout_df[after_breakout_df["close_price"] < retest_lower].copy()

    if false_breakout_df.empty:
        return None

    return false_breakout_df.iloc[0]


def calculate_wick_ratio(candle: pd.Series | None, direction: str | None) -> float | None:
    """
    Calculates rejection wick ratio in the expected trade direction.
    """
    if candle is None or direction is None:
        return None

    high_price = float(candle["high_price"])
    low_price = float(candle["low_price"])
    open_price = float(candle["open_price"])
    close_price = float(candle["close_price"])

    candle_range = high_price - low_price
    if candle_range <= 0:
        return None

    candle_body_high = max(open_price, close_price)
    candle_body_low = min(open_price, close_price)

    # Bearish rejection should leave an upper wick near the retest area.
    if direction == "bearish":
        wick_size = high_price - candle_body_high
    else:
        wick_size = candle_body_low - low_price

    return float(max(wick_size, 0.0) / candle_range)


def check_retest_rejection(
    retest_candle: pd.Series | None,
    direction: str | None,
    retest_lower: float | None,
    retest_upper: float | None,
    min_rejection_wick_ratio: float,
) -> bool:
    """
    Confirms whether the retest candle rejected the broken boundary.
    """
    if retest_candle is None or direction is None or retest_lower is None or retest_upper is None:
        return False

    close_price = float(retest_candle["close_price"])
    wick_ratio = calculate_wick_ratio(retest_candle, direction)

    if wick_ratio is None:
        return False

    if direction == "bearish":
        # A bearish retest should close back below the retest zone with an upper wick.
        return close_price < retest_lower and wick_ratio >= min_rejection_wick_ratio

    # A bullish retest should close back above the retest zone with a lower wick.
    return close_price > retest_upper and wick_ratio >= min_rejection_wick_ratio


def find_continuation_candle(
    market_df: pd.DataFrame,
    direction: str | None,
    anchor_candle: pd.Series | None,
    boundary_price: float | None,
    breakout_threshold: float | None,
    min_target_distance: float | None,
    continuation_multiplier: float,
    max_confirmation_candles: int,
    include_anchor: bool,
) -> pd.Series | None:
    """
    Finds the first candle that moves far enough beyond the broken boundary.
    """
    if direction is None or anchor_candle is None or boundary_price is None or market_df.empty:
        return None

    threshold = float(breakout_threshold or 0.0)
    min_target = float(min_target_distance or 0.0)

    # Use a scale-aware continuation distance that is meaningful but not equal to full TP1.
    continuation_distance = max(threshold * continuation_multiplier, min_target * 0.50)

    after_anchor_df = get_candles_after(
        market_df=market_df,
        anchor_candle=anchor_candle,
        max_candles=max_confirmation_candles,
        include_anchor=include_anchor,
    )

    if after_anchor_df.empty:
        return None

    if direction == "bearish":
        continuation_df = after_anchor_df[after_anchor_df["close_price"] <= boundary_price - continuation_distance].copy()
    else:
        continuation_df = after_anchor_df[after_anchor_df["close_price"] >= boundary_price + continuation_distance].copy()

    if continuation_df.empty:
        return None

    return continuation_df.iloc[0]


def build_suggested_trade_levels(
    direction: str | None,
    base_levels: dict[str, float | None],
    event_state: str,
    breakout_candle: pd.Series | None,
    retest_candle: pd.Series | None,
    breakout_impulse_candle: pd.Series | None,
    continuation_after_retest_candle: pd.Series | None,
) -> dict[str, float | str | None]:
    """
    Builds a conservative suggested level set after event classification.
    """
    if direction is None:
        return {
            "suggested_entry_mode": None,
            "suggested_entry_price": None,
            "suggested_stop_loss": None,
            "suggested_tp1": None,
            "suggested_tp2": None,
        }

    trigger_price = base_levels["event_trigger_price"]
    min_target = base_levels["adaptive_min_target_distance"] or 0.0
    structure_target = base_levels["structure_min_target_distance"] or (min_target * 2.0)

    # Choose an entry anchor based on the strongest confirmed event stage.
    entry_mode = "trigger_reference"
    entry_price = trigger_price

    if "retest_rejection_confirmed" in event_state and retest_candle is not None:
        entry_mode = "retest_rejection_close"
        entry_price = get_candle_price(retest_candle, "close_price")
    elif "continuation_after_retest_confirmed" in event_state and continuation_after_retest_candle is not None:
        entry_mode = "continuation_after_retest_close"
        entry_price = get_candle_price(continuation_after_retest_candle, "close_price")
    elif "breakout_impulse_confirmed" in event_state and breakout_impulse_candle is not None:
        entry_mode = "breakout_impulse_close"
        entry_price = get_candle_price(breakout_impulse_candle, "close_price")
    elif breakout_candle is not None:
        entry_mode = "breakout_trigger_close"
        entry_price = get_candle_price(breakout_candle, "close_price") or trigger_price

    raw_stop = base_levels["raw_micro_stop_loss"]
    scenario_invalidation = base_levels["event_invalidation_price"]
    stop_loss = pick_stop_loss(
        direction=direction,
        entry_price=safe_float(entry_price),
        raw_micro_stop=raw_stop,
        scenario_invalidation=scenario_invalidation,
    )

    if entry_price is None:
        return {
            "suggested_entry_mode": entry_mode,
            "suggested_entry_price": None,
            "suggested_stop_loss": stop_loss,
            "suggested_tp1": None,
            "suggested_tp2": None,
        }

    if direction == "bearish":
        tp1 = float(entry_price) - float(min_target)
        tp2 = float(entry_price) - float(structure_target)
    else:
        tp1 = float(entry_price) + float(min_target)
        tp2 = float(entry_price) + float(structure_target)

    return {
        "suggested_entry_mode": entry_mode,
        "suggested_entry_price": float(entry_price),
        "suggested_stop_loss": stop_loss,
        "suggested_tp1": tp1,
        "suggested_tp2": tp2,
    }


def classify_risk_efficiency(rr_tp1: float | None, rr_tp2: float | None, minimum_rr: float) -> str:
    """
    Classifies whether the suggested levels are efficient enough for a future trade plan.
    """
    if rr_tp1 is None and rr_tp2 is None:
        return "no_valid_rr"

    if rr_tp2 is not None and rr_tp2 >= minimum_rr:
        if rr_tp1 is not None and rr_tp1 >= 1.0:
            return "efficient_tp1_and_tp2"
        return "efficient_tp2_only"

    return "risk_reward_not_efficient_yet"


def calculate_event_quality_score(
    event_candidate: pd.Series | None,
    breakout_detected: bool,
    retest_detected: bool,
    retest_rejected: bool,
    false_breakout_detected: bool,
    breakout_impulse_confirmed: bool,
    continuation_after_retest_confirmed: bool,
) -> float | None:
    """
    Scores the event quality using channel quality and event confirmation evidence.
    """
    if event_candidate is None:
        return None

    rolling_quality = safe_float(event_candidate.get("rolling_quality_score")) or 0.0
    r_squared = safe_float(event_candidate.get("r_squared")) or 0.0
    containment_rate = safe_float(event_candidate.get("containment_rate")) or 0.0

    # Start with the channel evidence before adding event-sequence confirmation points.
    score = (0.45 * rolling_quality) + (0.25 * r_squared) + (0.20 * containment_rate)

    if breakout_detected:
        score += 0.05
    if retest_detected:
        score += 0.03
    if retest_rejected:
        score += 0.08
    if breakout_impulse_confirmed:
        score += 0.04
    if continuation_after_retest_confirmed:
        score += 0.08
    if false_breakout_detected:
        score -= 0.25

    return float(max(0.0, min(score, 1.0)))


def classify_event_state(
    event_direction: str | None,
    breakout_detected: bool,
    retest_detected: bool,
    retest_rejected: bool,
    false_breakout_detected: bool,
    breakout_impulse_confirmed: bool,
    continuation_after_retest_confirmed: bool,
    risk_efficiency_state: str,
    minimum_rr: float,
) -> tuple[str, str, str]:
    """
    Converts event flags into a final event state and actionability label.
    """
    if event_direction is None:
        return (
            "no_directional_scenario",
            "no_trade_insufficient_scenario_context",
            "No clear bullish or bearish scenario direction was available.",
        )

    if not breakout_detected:
        return (
            f"{event_direction}_trigger_pending",
            "wait_for_scale_aware_breakout_trigger",
            "The scenario exists, but price has not closed beyond the adaptive trigger yet.",
        )

    if false_breakout_detected:
        return (
            f"{event_direction}_false_breakout_detected",
            "avoid_trade_or_wait_for_new_structure",
            "Price broke the trigger but returned inside the channel, so the breakout is not reliable yet.",
        )

    if retest_rejected:
        if risk_efficiency_state.startswith("efficient"):
            return (
                f"{event_direction}_retest_rejection_confirmed",
                "build_trade_plan_from_confirmed_retest",
                "Price retested the broken boundary, rejected it, and the projected risk/reward is usable.",
            )
        return (
            f"{event_direction}_retest_rejection_confirmed_but_rr_weak",
            "wait_for_better_price_or_wider_target",
            f"Retest rejection exists, but projected risk/reward is below the minimum RR target of {minimum_rr}.",
        )

    if continuation_after_retest_confirmed:
        if risk_efficiency_state.startswith("efficient"):
            return (
                f"{event_direction}_continuation_after_retest_confirmed",
                "build_trade_plan_from_retest_continuation",
                "Price retested the broken boundary, then continued in the scenario direction with usable risk/reward.",
            )
        return (
            f"{event_direction}_continuation_after_retest_confirmed_but_rr_weak",
            "wait_for_better_price_or_wider_target",
            f"Continuation after retest exists, but projected risk/reward is below the minimum RR target of {minimum_rr}.",
        )

    if retest_detected:
        return (
            f"{event_direction}_retest_detected_waiting_rejection",
            "wait_for_retest_rejection_or_new_continuation",
            "Price returned to the broken boundary zone, but rejection is not confirmed yet.",
        )

    if breakout_impulse_confirmed:
        if risk_efficiency_state.startswith("efficient"):
            return (
                f"{event_direction}_breakout_impulse_confirmed",
                "watch_continuation_or_wait_for_retest_entry",
                "Price moved far enough beyond the broken boundary, but no retest has confirmed yet.",
            )
        return (
            f"{event_direction}_breakout_impulse_confirmed_but_rr_weak",
            "wait_for_retest_or_better_price",
            f"Breakout impulse exists, but projected risk/reward is below the minimum RR target of {minimum_rr}.",
        )

    return (
        f"{event_direction}_breakout_detected_waiting_retest_or_impulse",
        "wait_for_retest_or_impulse_confirmation",
        "Price broke the adaptive trigger, but retest or impulse confirmation is still pending.",
    )


def analyze_trend_channel_events(
    symbol: str,
    timeframes: str | list[str] | tuple[str, ...],
    limit: int = 300,
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
    end_time: str | None = None,
    closed_candles_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Runs scenario analysis, then classifies sequence-aware breakout/retest events.
    """
    scenario_df, summary_df, candidates_df = analyze_trend_channel_scenario(
        symbol=symbol,
        timeframes=timeframes,
        limit=limit,
        step_size=step_size,
        top_n_per_timeframe=top_n_per_timeframe,
        active_only=active_only,
        recent_window=recent_window,
        atr_window=atr_window,
        end_time=end_time,
        closed_candles_only=closed_candles_only,
    )

    if scenario_df.empty:
        return pd.DataFrame(), scenario_df, summary_df, candidates_df

    scenario_row = scenario_df.iloc[0]
    event_direction = infer_event_direction(scenario_row)
    entry_timeframe = safe_text(scenario_row.get("entry_timeframe"))
    event_candidate = pick_event_candidate(candidates_df, entry_timeframe, event_direction)

    # Load entry timeframe candles because event detection belongs to the execution/refinement layer.
    if entry_timeframe and end_time is not None:
        candles_df = load_market_candles_as_of(
            symbol=symbol,
            timeframe=entry_timeframe,
            end_time=end_time,
            limit=limit,
            closed_candles_only=closed_candles_only,
        )
    elif entry_timeframe:
        candles_df = load_market_candles(symbol=symbol, timeframe=entry_timeframe, limit=limit)
    else:
        candles_df = pd.DataFrame()
    market_df = prepare_channel_candles(candles_df) if not candles_df.empty else pd.DataFrame()

    base_levels = calculate_base_event_levels(
        event_candidate=event_candidate,
        event_direction=event_direction,
        scenario_row=scenario_row,
    )

    breakout_candle = find_first_breakout_candle(
        market_df=market_df,
        direction=event_direction,
        trigger_price=base_levels["event_trigger_price"],
        event_lookback=event_lookback,
    )
    breakout_detected = breakout_candle is not None

    retest_candle = find_retest_candle(
        market_df=market_df,
        breakout_candle=breakout_candle,
        retest_lower=base_levels["event_retest_lower"],
        retest_upper=base_levels["event_retest_upper"],
        max_retest_candles=max_retest_candles,
    )
    retest_detected = retest_candle is not None

    false_breakout_candle = find_false_breakout_candle(
        market_df=market_df,
        direction=event_direction,
        breakout_candle=breakout_candle,
        retest_lower=base_levels["event_retest_lower"],
        retest_upper=base_levels["event_retest_upper"],
        max_false_breakout_candles=max_false_breakout_candles,
    )
    false_breakout_detected = false_breakout_candle is not None

    retest_rejected = check_retest_rejection(
        retest_candle=retest_candle,
        direction=event_direction,
        retest_lower=base_levels["event_retest_lower"],
        retest_upper=base_levels["event_retest_upper"],
        min_rejection_wick_ratio=min_rejection_wick_ratio,
    )

    # This captures an impulse after breakout, even before any retest happens.
    breakout_impulse_candle = find_continuation_candle(
        market_df=market_df,
        direction=event_direction,
        anchor_candle=breakout_candle,
        boundary_price=base_levels["event_boundary_price"],
        breakout_threshold=base_levels["adaptive_breakout_threshold"],
        min_target_distance=base_levels["adaptive_min_target_distance"],
        continuation_multiplier=continuation_multiplier,
        max_confirmation_candles=max_confirmation_candles,
        include_anchor=True,
    )
    breakout_impulse_confirmed = breakout_impulse_candle is not None

    # This is stricter: continuation must happen after the retest candle.
    continuation_after_retest_candle = find_continuation_candle(
        market_df=market_df,
        direction=event_direction,
        anchor_candle=retest_candle,
        boundary_price=base_levels["event_boundary_price"],
        breakout_threshold=base_levels["adaptive_breakout_threshold"],
        min_target_distance=base_levels["adaptive_min_target_distance"],
        continuation_multiplier=continuation_multiplier,
        max_confirmation_candles=max_confirmation_candles,
        include_anchor=False,
    )
    continuation_after_retest_confirmed = continuation_after_retest_candle is not None

    # Build suggested levels before final actionability so risk quality can influence the label.
    provisional_trade_levels = build_suggested_trade_levels(
        direction=event_direction,
        base_levels=base_levels,
        event_state="retest_rejection_confirmed" if retest_rejected else "breakout_impulse_confirmed",
        breakout_candle=breakout_candle,
        retest_candle=retest_candle,
        breakout_impulse_candle=breakout_impulse_candle,
        continuation_after_retest_candle=continuation_after_retest_candle,
    )

    suggested_rr_tp1 = calculate_rr(
        direction=event_direction,
        entry_price=safe_float(provisional_trade_levels["suggested_entry_price"]),
        stop_loss=safe_float(provisional_trade_levels["suggested_stop_loss"]),
        take_profit=safe_float(provisional_trade_levels["suggested_tp1"]),
    )
    suggested_rr_tp2 = calculate_rr(
        direction=event_direction,
        entry_price=safe_float(provisional_trade_levels["suggested_entry_price"]),
        stop_loss=safe_float(provisional_trade_levels["suggested_stop_loss"]),
        take_profit=safe_float(provisional_trade_levels["suggested_tp2"]),
    )
    risk_efficiency_state = classify_risk_efficiency(
        rr_tp1=suggested_rr_tp1,
        rr_tp2=suggested_rr_tp2,
        minimum_rr=minimum_rr,
    )

    event_state, actionability, event_note = classify_event_state(
        event_direction=event_direction,
        breakout_detected=breakout_detected,
        retest_detected=retest_detected,
        retest_rejected=retest_rejected,
        false_breakout_detected=false_breakout_detected,
        breakout_impulse_confirmed=breakout_impulse_confirmed,
        continuation_after_retest_confirmed=continuation_after_retest_confirmed,
        risk_efficiency_state=risk_efficiency_state,
        minimum_rr=minimum_rr,
    )

    # Rebuild levels using the final event state, so entry mode matches the event label.
    trade_levels = build_suggested_trade_levels(
        direction=event_direction,
        base_levels=base_levels,
        event_state=event_state,
        breakout_candle=breakout_candle,
        retest_candle=retest_candle,
        breakout_impulse_candle=breakout_impulse_candle,
        continuation_after_retest_candle=continuation_after_retest_candle,
    )

    suggested_rr_tp1 = calculate_rr(
        direction=event_direction,
        entry_price=safe_float(trade_levels["suggested_entry_price"]),
        stop_loss=safe_float(trade_levels["suggested_stop_loss"]),
        take_profit=safe_float(trade_levels["suggested_tp1"]),
    )
    suggested_rr_tp2 = calculate_rr(
        direction=event_direction,
        entry_price=safe_float(trade_levels["suggested_entry_price"]),
        stop_loss=safe_float(trade_levels["suggested_stop_loss"]),
        take_profit=safe_float(trade_levels["suggested_tp2"]),
    )
    risk_efficiency_state = classify_risk_efficiency(
        rr_tp1=suggested_rr_tp1,
        rr_tp2=suggested_rr_tp2,
        minimum_rr=minimum_rr,
    )

    latest_row = market_df.iloc[-1] if not market_df.empty else None
    latest_close = safe_float(latest_row.get("close_price")) if latest_row is not None else None
    latest_timestamp = latest_row.get("timestamp_utc") if latest_row is not None else None

    event_quality_score = calculate_event_quality_score(
        event_candidate=event_candidate,
        breakout_detected=breakout_detected,
        retest_detected=retest_detected,
        retest_rejected=retest_rejected,
        false_breakout_detected=false_breakout_detected,
        breakout_impulse_confirmed=breakout_impulse_confirmed,
        continuation_after_retest_confirmed=continuation_after_retest_confirmed,
    )

    event_analysis = TrendChannelEventAnalysis(
        symbol=symbol,
        timeframes=safe_text(scenario_row.get("timeframes")) or "",
        scenario_state=safe_text(scenario_row.get("scenario_state")),
        scenario_direction=safe_text(scenario_row.get("scenario_direction")) or "neutral_or_unclear",
        actionability=actionability,
        event_state=event_state,
        event_direction=event_direction,
        event_timeframe=entry_timeframe,
        event_channel_role=safe_text(event_candidate.get("channel_role")) if event_candidate is not None else None,
        event_channel_type=safe_text(event_candidate.get("channel_type")) if event_candidate is not None else None,
        event_channel_current_state=safe_text(event_candidate.get("current_state")) if event_candidate is not None else None,
        breakout_detected=breakout_detected,
        breakout_timestamp=breakout_candle.get("timestamp_utc") if breakout_candle is not None else None,
        retest_detected=retest_detected,
        retest_timestamp=retest_candle.get("timestamp_utc") if retest_candle is not None else None,
        retest_rejected=retest_rejected,
        retest_rejection_timestamp=retest_candle.get("timestamp_utc") if retest_rejected and retest_candle is not None else None,
        false_breakout_detected=false_breakout_detected,
        false_breakout_timestamp=false_breakout_candle.get("timestamp_utc") if false_breakout_candle is not None else None,
        breakout_impulse_confirmed=breakout_impulse_confirmed,
        breakout_impulse_timestamp=breakout_impulse_candle.get("timestamp_utc") if breakout_impulse_candle is not None else None,
        continuation_after_retest_confirmed=continuation_after_retest_confirmed,
        continuation_after_retest_timestamp=continuation_after_retest_candle.get("timestamp_utc") if continuation_after_retest_candle is not None else None,
        event_boundary_price=base_levels["event_boundary_price"],
        event_trigger_price=base_levels["event_trigger_price"],
        event_retest_lower=base_levels["event_retest_lower"],
        event_retest_upper=base_levels["event_retest_upper"],
        event_invalidation_price=base_levels["event_invalidation_price"],
        suggested_entry_mode=safe_text(trade_levels["suggested_entry_mode"]),
        suggested_entry_price=safe_float(trade_levels["suggested_entry_price"]),
        suggested_stop_loss=safe_float(trade_levels["suggested_stop_loss"]),
        suggested_tp1=safe_float(trade_levels["suggested_tp1"]),
        suggested_tp2=safe_float(trade_levels["suggested_tp2"]),
        suggested_rr_tp1=suggested_rr_tp1,
        suggested_rr_tp2=suggested_rr_tp2,
        risk_efficiency_state=risk_efficiency_state,
        adaptive_breakout_threshold=base_levels["adaptive_breakout_threshold"],
        adaptive_retest_tolerance=base_levels["adaptive_retest_tolerance"],
        adaptive_stop_buffer=base_levels["adaptive_stop_buffer"],
        adaptive_min_target_distance=base_levels["adaptive_min_target_distance"],
        latest_close=latest_close,
        latest_timestamp=latest_timestamp,
        event_quality_score=event_quality_score,
        notes=event_note,
    )

    event_df = pd.DataFrame([asdict(event_analysis)])

    return event_df, scenario_df, summary_df, candidates_df
