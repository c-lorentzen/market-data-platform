from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from mdp.structure.trend_channel_trade_plan import build_trend_channel_trade_plan


@dataclass(frozen=True)
class TrendChannelOutcomeProbability:
    """
    Stores a scale-aware probability proxy for one trend-channel trade plan.
    """

    symbol: str
    timeframes: str
    probability_state: str
    estimation_method: str
    plan_state: str | None
    plan_direction: str | None
    event_state: str | None
    event_quality_score: float | None
    target_plan_state: str | None
    available_targets: str
    entry_price: float | None
    stop_loss: float | None
    risk_points: float | None
    tp1_price: float | None
    tp1_rr: float | None
    tp1_available: bool
    tp1_probability: float | None
    tp1_weight: float | None
    tp2_price: float | None
    tp2_rr: float | None
    tp2_available: bool
    tp2_probability: float | None
    tp2_weight: float | None
    tp3_price: float | None
    tp3_rr: float | None
    tp3_available: bool
    tp3_probability: float | None
    tp3_weight: float | None
    stop_loss_probability: float | None
    best_probability_target: str | None
    best_expected_value_target: str | None
    weighted_expected_r: float | None
    probability_confidence_score: float | None
    probability_model_status: str
    notes: str


def safe_float(value: Any) -> float | None:
    """
    Converts values to float while treating missing values as None.
    """
    # Pandas represents missing cells as NaN, so this prevents NaN from spreading.
    if value is None or pd.isna(value):
        return None

    return float(value)


def safe_text(value: Any) -> str | None:
    """
    Converts values to text while treating missing values as None.
    """
    if value is None or pd.isna(value):
        return None

    return str(value)


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """
    Keeps a decimal value inside a safe probability range.
    """
    return float(max(minimum, min(maximum, value)))


def calculate_target_raw_score(
    event_quality_score: float,
    target_rr: float | None,
    target_role_multiplier: float,
    rr_decay: float,
) -> float:
    """
    Builds a raw target score before normalizing probabilities.
    """
    if target_rr is None or target_rr <= 0:
        return 0.0

    # Farther targets usually have lower hit probability, so RR decays the score.
    rr_distance_score = 1.0 / (1.0 + (target_rr * rr_decay))

    return float(event_quality_score * target_role_multiplier * rr_distance_score)


def calculate_stop_raw_score(
    event_quality_score: float,
    risk_points_ratio: float | None,
    stop_bias_multiplier: float,
) -> float:
    """
    Builds a raw stop-loss score before normalizing probabilities.
    """
    # Strong event quality lowers the stop-loss score, but never removes risk completely.
    uncertainty_score = max(0.05, 1.0 - event_quality_score)

    # Larger relative risk deserves a small penalty because the plan needs more room to survive.
    risk_ratio_penalty = 0.0
    if risk_points_ratio is not None:
        risk_ratio_penalty = min(0.15, risk_points_ratio * 50.0)

    return float((uncertainty_score * stop_bias_multiplier) + risk_ratio_penalty)


def normalize_scores(raw_scores: dict[str, float]) -> dict[str, float | None]:
    """
    Converts raw scores into decimal probabilities that sum to 1.0.
    """
    total_score = sum(score for score in raw_scores.values() if score > 0)
    if total_score <= 0:
        return {key: None for key in raw_scores}

    return {key: float(score / total_score) if score > 0 else None for key, score in raw_scores.items()}


def calculate_weighted_expected_r(
    probabilities: dict[str, float | None],
    target_rrs: dict[str, float | None],
) -> float | None:
    """
    Calculates probability-weighted expected R for the available plan.
    """
    stop_probability = probabilities.get("SL")
    if stop_probability is None:
        return None

    # Hitting the stop is modeled as -1R.
    expected_r = -1.0 * stop_probability

    for target_label in ["TP1", "TP2", "TP3"]:
        probability = probabilities.get(target_label)
        rr_value = target_rrs.get(target_label)
        if probability is not None and rr_value is not None:
            expected_r += probability * rr_value

    return float(expected_r)


def calculate_target_weights(
    probabilities: dict[str, float | None],
    target_rrs: dict[str, float | None],
) -> dict[str, float | None]:
    """
    Calculates scale-out weights from each available target's expected contribution.
    """
    contributions: dict[str, float] = {}

    for target_label in ["TP1", "TP2", "TP3"]:
        probability = probabilities.get(target_label)
        rr_value = target_rrs.get(target_label)
        if probability is not None and rr_value is not None and rr_value > 0:
            contributions[target_label] = probability * rr_value

    total_contribution = sum(contributions.values())
    if total_contribution <= 0:
        return {"TP1": None, "TP2": None, "TP3": None}

    # Weights only apply to available targets, not to the stop-loss.
    return {
        target_label: float(contribution / total_contribution)
        for target_label, contribution in contributions.items()
    } | {
        target_label: None
        for target_label in ["TP1", "TP2", "TP3"]
        if target_label not in contributions
    }


def pick_best_probability_target(probabilities: dict[str, float | None]) -> str | None:
    """
    Selects the available TP with the highest hit-probability proxy.
    """
    target_probabilities = {
        label: probability
        for label, probability in probabilities.items()
        if label.startswith("TP") and probability is not None
    }
    if not target_probabilities:
        return None

    return max(target_probabilities, key=lambda label: target_probabilities[label])


def pick_best_expected_value_target(
    probabilities: dict[str, float | None],
    target_rrs: dict[str, float | None],
) -> str | None:
    """
    Selects the available TP with the highest probability times RR contribution.
    """
    expected_values: dict[str, float] = {}

    for target_label in ["TP1", "TP2", "TP3"]:
        probability = probabilities.get(target_label)
        rr_value = target_rrs.get(target_label)
        if probability is not None and rr_value is not None:
            expected_values[target_label] = probability * rr_value

    if not expected_values:
        return None

    return max(expected_values, key=lambda label: expected_values[label])


def classify_probability_state(
    trade_plan_row: pd.Series,
    weighted_expected_r: float | None,
    stop_loss_probability: float | None,
) -> tuple[str, str]:
    """
    Classifies whether the probability proxy supports the trade plan.
    """
    plan_state = safe_text(trade_plan_row.get("plan_state")) or ""
    available_targets = safe_text(trade_plan_row.get("available_targets")) or "none"

    if not plan_state.startswith("valid_trade_plan"):
        return (
            "probability_not_available_no_valid_trade_plan",
            "No valid trade plan exists, so outcome probabilities are not calculated.",
        )

    if available_targets == "none":
        return (
            "probability_not_available_no_targets",
            "The plan has no efficient targets, so target probabilities are not useful yet.",
        )

    if weighted_expected_r is None or stop_loss_probability is None:
        return (
            "probability_proxy_unavailable",
            "The plan exists, but the probability proxy could not be calculated.",
        )

    if weighted_expected_r > 0 and stop_loss_probability < 0.50:
        return (
            "positive_probability_proxy",
            "The probability proxy favors the available target set over the stop outcome.",
        )

    return (
        "weak_probability_proxy",
        "The probability proxy does not strongly favor the target set yet.",
    )


def estimate_probability_from_trade_plan(
    trade_plan_df: pd.DataFrame,
    rr_decay: float = 0.45,
    stop_bias_multiplier: float = 1.25,
) -> pd.DataFrame:
    """
    Estimates SL/TP probability proxies from one scale-aware trade plan.
    """
    if trade_plan_df.empty:
        return pd.DataFrame()

    trade_plan_row = trade_plan_df.iloc[0]

    event_quality_score = safe_float(trade_plan_row.get("event_quality_score"))
    if event_quality_score is None:
        event_quality_score = 0.50

    # Keep event quality bounded because it is used as a probability-like confidence input.
    event_quality_score = clamp(event_quality_score, 0.05, 0.98)

    target_rrs = {
        "TP1": safe_float(trade_plan_row.get("tp1_rr")),
        "TP2": safe_float(trade_plan_row.get("tp2_rr")),
        "TP3": safe_float(trade_plan_row.get("tp3_rr")),
    }

    target_available = {
        "TP1": bool(trade_plan_row.get("tp1_available")),
        "TP2": bool(trade_plan_row.get("tp2_available")),
        "TP3": bool(trade_plan_row.get("tp3_available")),
    }

    # Only available targets get probability scores; unavailable targets stay blank.
    raw_scores = {
        "SL": calculate_stop_raw_score(
            event_quality_score=event_quality_score,
            risk_points_ratio=safe_float(trade_plan_row.get("risk_points_ratio")),
            stop_bias_multiplier=stop_bias_multiplier,
        )
    }

    role_multipliers = {
        "TP1": 0.90,
        "TP2": 1.00,
        "TP3": 0.75,
    }

    for target_label in ["TP1", "TP2", "TP3"]:
        if target_available[target_label]:
            raw_scores[target_label] = calculate_target_raw_score(
                event_quality_score=event_quality_score,
                target_rr=target_rrs[target_label],
                target_role_multiplier=role_multipliers[target_label],
                rr_decay=rr_decay,
            )
        else:
            raw_scores[target_label] = 0.0

    probabilities = normalize_scores(raw_scores)
    weighted_expected_r = calculate_weighted_expected_r(
        probabilities=probabilities,
        target_rrs=target_rrs,
    )
    target_weights = calculate_target_weights(
        probabilities=probabilities,
        target_rrs=target_rrs,
    )

    best_probability_target = pick_best_probability_target(probabilities)
    best_expected_value_target = pick_best_expected_value_target(probabilities, target_rrs)

    probability_state, notes = classify_probability_state(
        trade_plan_row=trade_plan_row,
        weighted_expected_r=weighted_expected_r,
        stop_loss_probability=probabilities.get("SL"),
    )

    # Confidence is stronger when event quality is high and multiple targets are available.
    available_target_count = int(trade_plan_row.get("available_target_count") or 0)
    probability_confidence_score = clamp(
        (event_quality_score * 0.75) + (min(available_target_count, 3) / 3.0 * 0.25),
        0.0,
        1.0,
    )

    probability_row = TrendChannelOutcomeProbability(
        symbol=safe_text(trade_plan_row.get("symbol")) or "",
        timeframes=safe_text(trade_plan_row.get("timeframes")) or "",
        probability_state=probability_state,
        estimation_method="scale_aware_outcome_probability_proxy_v1_not_historical_yet",
        plan_state=safe_text(trade_plan_row.get("plan_state")),
        plan_direction=safe_text(trade_plan_row.get("plan_direction")),
        event_state=safe_text(trade_plan_row.get("event_state")),
        event_quality_score=safe_float(trade_plan_row.get("event_quality_score")),
        target_plan_state=safe_text(trade_plan_row.get("target_plan_state")),
        available_targets=safe_text(trade_plan_row.get("available_targets")) or "none",
        entry_price=safe_float(trade_plan_row.get("entry_price")),
        stop_loss=safe_float(trade_plan_row.get("stop_loss")),
        risk_points=safe_float(trade_plan_row.get("risk_points")),
        tp1_price=safe_float(trade_plan_row.get("tp1_price")),
        tp1_rr=target_rrs["TP1"],
        tp1_available=target_available["TP1"],
        tp1_probability=probabilities.get("TP1"),
        tp1_weight=target_weights.get("TP1"),
        tp2_price=safe_float(trade_plan_row.get("tp2_price")),
        tp2_rr=target_rrs["TP2"],
        tp2_available=target_available["TP2"],
        tp2_probability=probabilities.get("TP2"),
        tp2_weight=target_weights.get("TP2"),
        tp3_price=safe_float(trade_plan_row.get("tp3_price")),
        tp3_rr=target_rrs["TP3"],
        tp3_available=target_available["TP3"],
        tp3_probability=probabilities.get("TP3"),
        tp3_weight=target_weights.get("TP3"),
        stop_loss_probability=probabilities.get("SL"),
        best_probability_target=best_probability_target,
        best_expected_value_target=best_expected_value_target,
        weighted_expected_r=weighted_expected_r,
        probability_confidence_score=probability_confidence_score,
        probability_model_status="proxy_only_until_historical_similarity_backtest_is_added",
        notes=notes,
    )

    return pd.DataFrame([asdict(probability_row)])


def estimate_trend_channel_outcome_probability(
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
    rr_decay: float = 0.45,
    stop_bias_multiplier: float = 1.25,
    end_time: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Builds a trade plan and estimates probability proxies for its available targets.
    """
    trade_plan_df, event_df, scenario_df, summary_df, candidates_df = build_trend_channel_trade_plan(
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
    )

    # Estimate only after the trade-plan builder decides which targets are available.
    probability_df = estimate_probability_from_trade_plan(
        trade_plan_df=trade_plan_df,
        rr_decay=rr_decay,
        stop_bias_multiplier=stop_bias_multiplier,
    )

    return probability_df, trade_plan_df, event_df, scenario_df, summary_df, candidates_df
