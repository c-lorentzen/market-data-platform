from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from mdp.structure.trend_channel_events import analyze_trend_channel_events


@dataclass(frozen=True)
class TrendChannelTradePlan:
    """
    Stores one deterministic, scale-aware trade plan candidate.
    """

    symbol: str
    timeframes: str
    plan_state: str
    plan_direction: str | None
    plan_actionability: str
    event_state: str | None
    event_actionability: str | None
    event_quality_score: float | None
    risk_efficiency_state: str | None
    entry_timeframe: str | None
    entry_mode: str | None
    entry_price: float | None
    stop_loss: float | None
    risk_points: float | None
    risk_points_ratio: float | None
    tp1_price: float | None
    tp1_rr: float | None
    tp1_available: bool
    tp1_role: str
    tp2_price: float | None
    tp2_rr: float | None
    tp2_available: bool
    tp2_role: str
    tp3_price: float | None
    tp3_rr: float | None
    tp3_available: bool
    tp3_role: str
    available_target_count: int
    available_targets: str
    best_available_target: str | None
    best_available_rr: float | None
    target_plan_state: str
    suggested_management_style: str
    minimum_rr: float
    latest_close: float | None
    latest_timestamp: Any
    probability_model_status: str
    notes: str


def safe_float(value: Any) -> float | None:
    """
    Converts a value to float while treating missing values as None.
    """
    # Pandas uses NaN for missing values, so we guard against it here.
    if value is None or pd.isna(value):
        return None

    return float(value)


def safe_text(value: Any) -> str | None:
    """
    Converts a value to text while treating missing values as None.
    """
    if value is None or pd.isna(value):
        return None

    return str(value)


def calculate_rr(
    direction: str | None,
    entry_price: float | None,
    stop_loss: float | None,
    take_profit: float | None,
) -> float | None:
    """
    Calculates reward/risk for long or short plans.
    """
    if direction is None or entry_price is None or stop_loss is None or take_profit is None:
        return None

    if direction == "bearish":
        risk = stop_loss - entry_price
        reward = entry_price - take_profit
    else:
        risk = entry_price - stop_loss
        reward = take_profit - entry_price

    # The trade plan is invalid if risk or reward is not positive.
    if risk <= 0 or reward <= 0:
        return None

    return float(reward / risk)


def calculate_risk_points(
    direction: str | None,
    entry_price: float | None,
    stop_loss: float | None,
) -> float | None:
    """
    Calculates the absolute stop distance in price points.
    """
    if direction is None or entry_price is None or stop_loss is None:
        return None

    if direction == "bearish":
        risk_points = stop_loss - entry_price
    else:
        risk_points = entry_price - stop_loss

    if risk_points <= 0:
        return None

    return float(risk_points)


def is_directional_target_valid(
    direction: str | None,
    entry_price: float | None,
    take_profit: float | None,
) -> bool:
    """
    Checks whether a target is placed in the correct trade direction.
    """
    if direction is None or entry_price is None or take_profit is None:
        return False

    # Bearish targets must be below entry; bullish targets must be above entry.
    if direction == "bearish":
        return take_profit < entry_price

    return take_profit > entry_price


def classify_target_availability(
    target_rr: float | None,
    minimum_rr: float,
    direction: str | None,
    entry_price: float | None,
    target_price: float | None,
) -> bool:
    """
    Marks a target as tradable only when it is directional and meets minimum RR.
    """
    if not is_directional_target_valid(direction, entry_price, target_price):
        return False

    if target_rr is None:
        return False

    return target_rr >= minimum_rr


def get_best_structure_target(
    candidates_df: pd.DataFrame,
    structure_timeframe: str | None,
    direction: str | None,
) -> float | None:
    """
    Uses the higher-timeframe channel opposite boundary as an optional TP3.
    """
    if candidates_df.empty or structure_timeframe is None or direction is None:
        return None

    if "timeframe" not in candidates_df.columns:
        return None

    structure_candidates = candidates_df[candidates_df["timeframe"] == structure_timeframe].copy()
    if structure_candidates.empty:
        return None

    # Prefer the strongest structure candidate because TP3 should use the main structure.
    sort_columns = [
        column
        for column in ["rolling_quality_score", "r_squared", "containment_rate"]
        if column in structure_candidates.columns
    ]
    if sort_columns:
        structure_candidates = structure_candidates.sort_values(
            by=sort_columns,
            ascending=[False] * len(sort_columns),
        )

    best_structure = structure_candidates.iloc[0]

    # A bearish plan targets the lower side of the structure; bullish targets the upper side.
    if direction == "bearish":
        return safe_float(best_structure.get("current_lower"))

    return safe_float(best_structure.get("current_upper"))


def summarize_available_targets(
    target_flags: dict[str, bool],
    target_rrs: dict[str, float | None],
) -> tuple[int, str, str | None, float | None]:
    """
    Builds compact target availability labels.
    """
    available_labels = [label for label, is_available in target_flags.items() if is_available]

    if not available_labels:
        return 0, "none", None, None

    # Pick the available target with the highest RR as the best target.
    best_label = max(
        available_labels,
        key=lambda label: target_rrs.get(label) if target_rrs.get(label) is not None else -1.0,
    )
    best_rr = target_rrs.get(best_label)

    return len(available_labels), ",".join(available_labels), best_label, best_rr


def classify_plan_state(
    event_actionability: str | None,
    event_state: str | None,
    available_target_count: int,
    risk_points: float | None,
) -> tuple[str, str, str]:
    """
    Converts event and target context into a final trade-plan state.
    """
    actionability = event_actionability or ""
    event_state_text = event_state or ""

    if not actionability.startswith("build_trade_plan"):
        return (
            "no_trade_plan_event_not_ready",
            "wait_for_event_confirmation",
            "The event detector has not produced a trade-plan-ready event yet.",
        )

    if risk_points is None:
        return (
            "no_trade_plan_invalid_risk",
            "wait_for_valid_entry_and_stop",
            "The event exists, but entry and stop do not form a valid risk profile.",
        )

    if available_target_count == 0:
        return (
            "watch_only_no_efficient_target",
            "wait_for_better_target_or_entry",
            "The event exists, but no target meets the minimum RR requirement.",
        )

    if available_target_count == 1:
        return (
            "valid_trade_plan_single_target",
            "trade_plan_ready_single_target",
            "One efficient target is available. The plan should not force extra targets.",
        )

    if "continuation_after_retest" in event_state_text:
        return (
            "valid_trade_plan_retest_continuation_multi_target",
            "trade_plan_ready_after_retest_continuation",
            "The event confirmed continuation after retest and multiple efficient targets are available.",
        )

    return (
        "valid_trade_plan_multi_target",
        "trade_plan_ready_multi_target",
        "Multiple efficient targets are available for this event.",
    )


def classify_target_plan_state(
    tp1_available: bool,
    tp2_available: bool,
    tp3_available: bool,
) -> str:
    """
    Builds a readable target state from target flags.
    """
    if tp1_available and tp2_available and tp3_available:
        return "tp1_tp2_tp3_available"
    if tp1_available and tp2_available:
        return "tp1_tp2_available"
    if tp2_available and tp3_available:
        return "tp2_tp3_available"
    if tp1_available:
        return "tp1_only"
    if tp2_available:
        return "tp2_only"
    if tp3_available:
        return "tp3_only"
    return "no_efficient_target_available"


def choose_management_style(target_plan_state: str) -> str:
    """
    Suggests a simple target management style without executing trades.
    """
    if target_plan_state == "no_efficient_target_available":
        return "no_trade"
    if target_plan_state.endswith("only"):
        return "single_target_plan"
    return "scale_out_across_available_targets"


def build_trade_plan_from_event(
    event_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    minimum_rr: float,
) -> pd.DataFrame:
    """
    Converts one confirmed event row into one deterministic trade plan row.
    """
    if event_df.empty:
        return pd.DataFrame()

    event_row = event_df.iloc[0]
    scenario_row = scenario_df.iloc[0] if not scenario_df.empty else pd.Series(dtype="object")

    direction = safe_text(event_row.get("event_direction"))
    entry_price = safe_float(event_row.get("suggested_entry_price"))
    stop_loss = safe_float(event_row.get("suggested_stop_loss"))
    tp1_price = safe_float(event_row.get("suggested_tp1"))
    tp2_price = safe_float(event_row.get("suggested_tp2"))

    # TP3 is optional and comes from the higher-timeframe structure boundary.
    structure_timeframe = safe_text(scenario_row.get("structure_timeframe"))
    tp3_price = get_best_structure_target(
        candidates_df=candidates_df,
        structure_timeframe=structure_timeframe,
        direction=direction,
    )

    risk_points = calculate_risk_points(direction, entry_price, stop_loss)
    risk_points_ratio = None
    if risk_points is not None and entry_price not in [None, 0]:
        risk_points_ratio = float(risk_points / entry_price)

    tp1_rr = calculate_rr(direction, entry_price, stop_loss, tp1_price)
    tp2_rr = calculate_rr(direction, entry_price, stop_loss, tp2_price)
    tp3_rr = calculate_rr(direction, entry_price, stop_loss, tp3_price)

    tp1_available = classify_target_availability(tp1_rr, minimum_rr, direction, entry_price, tp1_price)
    tp2_available = classify_target_availability(tp2_rr, minimum_rr, direction, entry_price, tp2_price)
    tp3_available = classify_target_availability(tp3_rr, minimum_rr, direction, entry_price, tp3_price)

    target_flags = {"TP1": tp1_available, "TP2": tp2_available, "TP3": tp3_available}
    target_rrs = {"TP1": tp1_rr, "TP2": tp2_rr, "TP3": tp3_rr}
    available_target_count, available_targets, best_target, best_rr = summarize_available_targets(
        target_flags=target_flags,
        target_rrs=target_rrs,
    )

    plan_state, plan_actionability, notes = classify_plan_state(
        event_actionability=safe_text(event_row.get("actionability")),
        event_state=safe_text(event_row.get("event_state")),
        available_target_count=available_target_count,
        risk_points=risk_points,
    )

    target_plan_state = classify_target_plan_state(tp1_available, tp2_available, tp3_available)
    management_style = choose_management_style(target_plan_state)

    trade_plan = TrendChannelTradePlan(
        symbol=safe_text(event_row.get("symbol")) or "",
        timeframes=safe_text(event_row.get("timeframes")) or "",
        plan_state=plan_state,
        plan_direction=direction,
        plan_actionability=plan_actionability,
        event_state=safe_text(event_row.get("event_state")),
        event_actionability=safe_text(event_row.get("actionability")),
        event_quality_score=safe_float(event_row.get("event_quality_score")),
        risk_efficiency_state=safe_text(event_row.get("risk_efficiency_state")),
        entry_timeframe=safe_text(event_row.get("event_timeframe")),
        entry_mode=safe_text(event_row.get("suggested_entry_mode")),
        entry_price=entry_price,
        stop_loss=stop_loss,
        risk_points=risk_points,
        risk_points_ratio=risk_points_ratio,
        tp1_price=tp1_price,
        tp1_rr=tp1_rr,
        tp1_available=tp1_available,
        tp1_role="entry_timeframe_min_target",
        tp2_price=tp2_price,
        tp2_rr=tp2_rr,
        tp2_available=tp2_available,
        tp2_role="structure_scale_target",
        tp3_price=tp3_price,
        tp3_rr=tp3_rr,
        tp3_available=tp3_available,
        tp3_role="higher_timeframe_opposite_channel_boundary",
        available_target_count=available_target_count,
        available_targets=available_targets,
        best_available_target=best_target,
        best_available_rr=best_rr,
        target_plan_state=target_plan_state,
        suggested_management_style=management_style,
        minimum_rr=minimum_rr,
        latest_close=safe_float(event_row.get("latest_close")),
        latest_timestamp=event_row.get("latest_timestamp"),
        probability_model_status="not_estimated_yet_requires_historical_outcome_model",
        notes=notes,
    )

    return pd.DataFrame([asdict(trade_plan)])


def build_trend_channel_trade_plan(
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Runs event detection and converts the result into a scale-aware trade plan.
    """
    event_df, scenario_df, summary_df, candidates_df = analyze_trend_channel_events(
        symbol=symbol,
        timeframes=timeframes,
        limit=limit,
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
        end_time=end_time,
        closed_candles_only=closed_candles_only,
    )

    # Build the trade plan only after the event module has created its event row.
    trade_plan_df = build_trade_plan_from_event(
        event_df=event_df,
        scenario_df=scenario_df,
        candidates_df=candidates_df,
        minimum_rr=minimum_rr,
    )

    return trade_plan_df, event_df, scenario_df, summary_df, candidates_df
