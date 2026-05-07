from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd

from mdp.backtesting.trend_channel_outcome_dataset import (
    build_historical_trend_channel_outcome_dataset,
)


@dataclass(frozen=True)
class RiskTargetDiagnosticSummary:
    """
    Stores one high-level diagnostic row for SL/TP efficiency.
    """

    total_rows: int
    actionable_plans: int
    clean_labels: int
    target_first_count: int
    stop_first_count: int
    open_horizon_count: int
    actionable_rate: float
    target_first_rate_clean: float
    stop_first_rate_clean: float
    open_horizon_rate_actionable: float
    median_max_favorable_r: float | None
    p75_max_favorable_r: float | None
    p90_max_favorable_r: float | None
    median_max_adverse_r: float | None
    p75_max_adverse_r: float | None
    p90_max_adverse_r: float | None
    median_nearest_target_rr: float | None
    p75_nearest_target_rr: float | None
    recommended_stop_multiplier_p75: float | None
    recommended_stop_multiplier_p90: float | None
    estimated_tp2_rr_after_p75_stop: float | None
    estimated_tp3_rr_after_p75_stop: float | None
    stop_efficiency_state: str
    target_efficiency_state: str
    sample_quality_state: str
    suggested_next_action: str


def _safe_float(value: Any) -> float | None:
    """
    Converts values to float without spreading NaN values through the report.
    """
    if value is None or pd.isna(value):
        return None
    return float(value)


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    """
    Calculates rates safely when the denominator may be zero.
    """
    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _quantile(series: pd.Series, q: float) -> float | None:
    """
    Returns a quantile from valid numeric values only.
    """
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.quantile(q))


def _median(series: pd.Series) -> float | None:
    """
    Returns the median from valid numeric values only.
    """
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.median())


def add_nearest_target_columns(outcomes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds nearest/farthest available target RR columns for diagnostic comparisons.
    """
    if outcomes_df.empty:
        return outcomes_df.copy()

    df = outcomes_df.copy()
    nearest_values: list[float | None] = []
    farthest_values: list[float | None] = []

    for _, row in df.iterrows():
        # Keep only targets that the trade plan marked as available.
        target_rrs: list[float] = []
        for target in ["tp1", "tp2", "tp3"]:
            available = bool(row.get(f"{target}_available"))
            rr_value = _safe_float(row.get(f"{target}_rr"))
            if available and rr_value is not None:
                target_rrs.append(rr_value)

        if target_rrs:
            nearest_values.append(min(target_rrs))
            farthest_values.append(max(target_rrs))
        else:
            nearest_values.append(None)
            farthest_values.append(None)

    df["nearest_available_target_rr"] = nearest_values
    df["farthest_available_target_rr"] = farthest_values
    return df


def classify_sample_quality(actionable_plans: int, clean_labels: int) -> str:
    """
    Classifies whether the dataset is large enough to trust the diagnostics.
    """
    if actionable_plans < 30 or clean_labels < 20:
        return "small_sample_directional_only"
    if actionable_plans < 100 or clean_labels < 60:
        return "medium_sample_use_with_caution"
    return "large_enough_for_initial_statistics"


def classify_stop_efficiency(
    stop_first_rate_clean: float,
    p75_adverse: float | None,
    p90_adverse: float | None,
) -> str:
    """
    Classifies whether the current SL placement appears too tight, acceptable, or uncertain.
    """
    if p75_adverse is None or p90_adverse is None:
        return "not_enough_adverse_data"

    # If most clean outcomes are stops and the 75th percentile exceeds 1R, the SL may be too tight or entries may be late.
    if stop_first_rate_clean >= 0.70 and p75_adverse > 1.0:
        return "stop_or_entry_needs_refinement_high_stop_first_rate"

    if p75_adverse < 0.50 and stop_first_rate_clean < 0.40:
        return "stop_may_be_looser_than_needed"

    return "stop_range_reasonable_but_needs_more_data"


def classify_target_efficiency(
    target_first_rate_clean: float,
    open_horizon_rate_actionable: float,
    p75_favorable: float | None,
    median_nearest_target_rr: float | None,
) -> str:
    """
    Classifies whether targets seem too ambitious for the observed favorable movement.
    """
    if p75_favorable is None or median_nearest_target_rr is None:
        return "not_enough_target_data"

    # If clean target hits are zero but many trades move favorably, target logic may be too ambitious or management needs partial exits.
    if target_first_rate_clean == 0.0 and p75_favorable >= 1.0:
        return "targets_or_management_need_refinement_no_clean_target_hits"

    if open_horizon_rate_actionable >= 0.30 and p75_favorable < median_nearest_target_rr:
        return "targets_may_be_too_far_for_current_horizon"

    return "target_range_reasonable_but_needs_more_data"


def build_risk_target_diagnostic_tables(
    outcomes_df: pd.DataFrame,
    minimum_rr: float = 1.50,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Builds summary, grouped diagnostics, and detailed actionable rows.
    """
    if outcomes_df.empty:
        empty_summary = pd.DataFrame(
            [
                asdict(
                    RiskTargetDiagnosticSummary(
                        total_rows=0,
                        actionable_plans=0,
                        clean_labels=0,
                        target_first_count=0,
                        stop_first_count=0,
                        open_horizon_count=0,
                        actionable_rate=0.0,
                        target_first_rate_clean=0.0,
                        stop_first_rate_clean=0.0,
                        open_horizon_rate_actionable=0.0,
                        median_max_favorable_r=None,
                        p75_max_favorable_r=None,
                        p90_max_favorable_r=None,
                        median_max_adverse_r=None,
                        p75_max_adverse_r=None,
                        p90_max_adverse_r=None,
                        median_nearest_target_rr=None,
                        p75_nearest_target_rr=None,
                        recommended_stop_multiplier_p75=None,
                        recommended_stop_multiplier_p90=None,
                        estimated_tp2_rr_after_p75_stop=None,
                        estimated_tp3_rr_after_p75_stop=None,
                        stop_efficiency_state="no_data",
                        target_efficiency_state="no_data",
                        sample_quality_state="no_data",
                        suggested_next_action="Load more historical candles before changing strategy rules.",
                    )
                )
            ]
        )
        return empty_summary, pd.DataFrame(), pd.DataFrame()

    df = add_nearest_target_columns(outcomes_df)
    actionable_df = df[df["outcome_state"] != "not_actionable"].copy()
    clean_df = df[df["label_quality"] == "clean"].copy()

    total_rows = len(df)
    actionable_plans = len(actionable_df)
    clean_labels = len(clean_df)
    target_first_count = int((df["outcome_state"] == "target_first").sum())
    stop_first_count = int((df["outcome_state"] == "stop_loss_first").sum())
    open_horizon_count = int((df["outcome_state"] == "no_terminal_outcome_within_horizon").sum())

    p75_adverse = _quantile(actionable_df.get("max_adverse_r", pd.Series(dtype="float64")), 0.75)
    p90_adverse = _quantile(actionable_df.get("max_adverse_r", pd.Series(dtype="float64")), 0.90)
    p75_favorable = _quantile(actionable_df.get("max_favorable_r", pd.Series(dtype="float64")), 0.75)
    p90_favorable = _quantile(actionable_df.get("max_favorable_r", pd.Series(dtype="float64")), 0.90)

    median_nearest_target_rr = _median(actionable_df.get("nearest_available_target_rr", pd.Series(dtype="float64")))
    p75_nearest_target_rr = _quantile(actionable_df.get("nearest_available_target_rr", pd.Series(dtype="float64")), 0.75)

    target_first_rate_clean = _safe_rate(target_first_count, clean_labels)
    stop_first_rate_clean = _safe_rate(stop_first_count, clean_labels)
    open_horizon_rate_actionable = _safe_rate(open_horizon_count, actionable_plans)

    # Use adverse quantiles as diagnostic multipliers, not immediate strategy replacements.
    recommended_stop_multiplier_p75 = None if p75_adverse is None else round(max(1.0, p75_adverse), 4)
    recommended_stop_multiplier_p90 = None if p90_adverse is None else round(max(1.0, p90_adverse), 4)

    median_tp2_rr = _median(actionable_df[actionable_df.get("tp2_available", False) == True].get("tp2_rr", pd.Series(dtype="float64")))
    median_tp3_rr = _median(actionable_df[actionable_df.get("tp3_available", False) == True].get("tp3_rr", pd.Series(dtype="float64")))

    # If the stop is widened by the p75 adverse multiplier, RR compresses by the same multiplier.
    if recommended_stop_multiplier_p75 and recommended_stop_multiplier_p75 > 0:
        estimated_tp2_rr_after_p75_stop = None if median_tp2_rr is None else round(median_tp2_rr / recommended_stop_multiplier_p75, 4)
        estimated_tp3_rr_after_p75_stop = None if median_tp3_rr is None else round(median_tp3_rr / recommended_stop_multiplier_p75, 4)
    else:
        estimated_tp2_rr_after_p75_stop = None
        estimated_tp3_rr_after_p75_stop = None

    stop_efficiency_state = classify_stop_efficiency(
        stop_first_rate_clean=stop_first_rate_clean,
        p75_adverse=p75_adverse,
        p90_adverse=p90_adverse,
    )
    target_efficiency_state = classify_target_efficiency(
        target_first_rate_clean=target_first_rate_clean,
        open_horizon_rate_actionable=open_horizon_rate_actionable,
        p75_favorable=p75_favorable,
        median_nearest_target_rr=median_nearest_target_rr,
    )
    sample_quality_state = classify_sample_quality(actionable_plans, clean_labels)

    if sample_quality_state == "small_sample_directional_only":
        suggested_next_action = "Do not optimize yet. Run a larger dataset, then compare SL variants and target-management variants."
    elif "refinement" in stop_efficiency_state or "refinement" in target_efficiency_state:
        suggested_next_action = "Create variant tests for structure-capped SL, volatility-expanded SL, break-even rules, and partial target management."
    else:
        suggested_next_action = "Keep current logic and validate with a larger walk-forward sample."

    summary = RiskTargetDiagnosticSummary(
        total_rows=total_rows,
        actionable_plans=actionable_plans,
        clean_labels=clean_labels,
        target_first_count=target_first_count,
        stop_first_count=stop_first_count,
        open_horizon_count=open_horizon_count,
        actionable_rate=_safe_rate(actionable_plans, total_rows),
        target_first_rate_clean=target_first_rate_clean,
        stop_first_rate_clean=stop_first_rate_clean,
        open_horizon_rate_actionable=open_horizon_rate_actionable,
        median_max_favorable_r=_median(actionable_df.get("max_favorable_r", pd.Series(dtype="float64"))),
        p75_max_favorable_r=p75_favorable,
        p90_max_favorable_r=p90_favorable,
        median_max_adverse_r=_median(actionable_df.get("max_adverse_r", pd.Series(dtype="float64"))),
        p75_max_adverse_r=p75_adverse,
        p90_max_adverse_r=p90_adverse,
        median_nearest_target_rr=median_nearest_target_rr,
        p75_nearest_target_rr=p75_nearest_target_rr,
        recommended_stop_multiplier_p75=recommended_stop_multiplier_p75,
        recommended_stop_multiplier_p90=recommended_stop_multiplier_p90,
        estimated_tp2_rr_after_p75_stop=estimated_tp2_rr_after_p75_stop,
        estimated_tp3_rr_after_p75_stop=estimated_tp3_rr_after_p75_stop,
        stop_efficiency_state=stop_efficiency_state,
        target_efficiency_state=target_efficiency_state,
        sample_quality_state=sample_quality_state,
        suggested_next_action=suggested_next_action,
    )

    summary_df = pd.DataFrame([asdict(summary)])

    group_columns = ["plan_direction", "event_state", "risk_efficiency_state", "target_plan_state"]
    existing_group_columns = [column for column in group_columns if column in actionable_df.columns]
    if existing_group_columns and not actionable_df.empty:
        grouped_df = (
            actionable_df.groupby(existing_group_columns, dropna=False)
            .agg(
                trades=("outcome_state", "count"),
                clean_labels=("label_quality", lambda values: int((values == "clean").sum())),
                target_first_count=("outcome_state", lambda values: int((values == "target_first").sum())),
                stop_first_count=("outcome_state", lambda values: int((values == "stop_loss_first").sum())),
                open_horizon_count=("outcome_state", lambda values: int((values == "no_terminal_outcome_within_horizon").sum())),
                median_max_favorable_r=("max_favorable_r", "median"),
                p75_max_favorable_r=("max_favorable_r", lambda values: pd.to_numeric(values, errors="coerce").quantile(0.75)),
                median_max_adverse_r=("max_adverse_r", "median"),
                p75_max_adverse_r=("max_adverse_r", lambda values: pd.to_numeric(values, errors="coerce").quantile(0.75)),
                median_nearest_target_rr=("nearest_available_target_rr", "median"),
            )
            .reset_index()
        )

        # Add rates after aggregation for readability.
        grouped_df["stop_first_rate_clean"] = grouped_df.apply(
            lambda row: _safe_rate(row["stop_first_count"], row["clean_labels"]), axis=1
        )
        grouped_df["target_first_rate_clean"] = grouped_df.apply(
            lambda row: _safe_rate(row["target_first_count"], row["clean_labels"]), axis=1
        )
    else:
        grouped_df = pd.DataFrame()

    detail_columns = [
        "analysis_cutoff",
        "plan_state",
        "plan_direction",
        "event_state",
        "outcome_state",
        "first_terminal_outcome",
        "available_targets",
        "nearest_available_target_rr",
        "best_available_rr",
        "risk_points",
        "risk_points_ratio",
        "max_favorable_r",
        "max_adverse_r",
        "label_quality",
    ]
    existing_detail_columns = [column for column in detail_columns if column in actionable_df.columns]
    detail_df = actionable_df[existing_detail_columns].copy() if existing_detail_columns else pd.DataFrame()

    return summary_df, grouped_df, detail_df


def run_risk_target_diagnostics(
    symbol: str,
    timeframes: str,
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Builds the historical dataset and then summarizes SL/TP diagnostic behavior.
    """
    outcomes_df, outcome_summary_df, cutoff_schedule_df = build_historical_trend_channel_outcome_dataset(
        symbol=symbol,
        timeframes=timeframes,
        analysis_limit=analysis_limit,
        dataset_limit=dataset_limit,
        start_time=start_time,
        end_time=end_time,
        warmup_candles=warmup_candles,
        horizon_candles=horizon_candles,
        cutoff_step_candles=cutoff_step_candles,
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
        include_non_actionable=include_non_actionable,
    )

    diagnostic_summary_df, grouped_df, detail_df = build_risk_target_diagnostic_tables(
        outcomes_df=outcomes_df,
        minimum_rr=minimum_rr,
    )

    return outcomes_df, outcome_summary_df, cutoff_schedule_df, diagnostic_summary_df, grouped_df, detail_df
