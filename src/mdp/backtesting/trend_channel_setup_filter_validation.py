from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from mdp.backtesting.trend_channel_setup_filter_diagnostics import (
    SetupFilterDiagnosticConfig,
    build_setup_filter_diagnostic_dataset,
)


@dataclass(frozen=True)
class SetupFilterRule:
    """
    Stores one deterministic setup-filter rule to test against historical managed outcomes.
    """

    rule_name: str
    rule_description: str
    filter_fn: Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class SetupFilterValidationConfig:
    """
    Stores setup-filter validation settings.
    """

    symbol: str
    timeframes: str
    analysis_limit: int = 200
    dataset_limit: int = 1200
    warmup_candles: int = 200
    horizon_candles: int = 80
    cutoff_step_candles: int = 20
    max_cutoffs: int | None = 30
    management_policy_name: str = "targetless_scalp_manage_only"
    minimum_rr: float = 1.50
    min_trades_for_candidate: int = 8
    active_only: bool = True


def _series_text(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Returns a safe lowercase string series for filter comparisons.
    """
    if column not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype="object")
    return df[column].fillna("").astype(str).str.strip().str.lower()


def _series_bool(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Returns a safe boolean series for target availability checks.
    """
    if column not in df.columns:
        return pd.Series([False] * len(df), index=df.index, dtype="bool")
    values = df[column]
    if values.dtype == bool:
        return values.fillna(False)
    return values.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def _series_float(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Returns a numeric series while treating bad values as NaN.
    """
    if column not in df.columns:
        return pd.Series([float("nan")] * len(df), index=df.index, dtype="float")
    return pd.to_numeric(df[column], errors="coerce")


def _all_rows(df: pd.DataFrame) -> pd.Series:
    """
    Keeps every row as the baseline rule.
    """
    return pd.Series([True] * len(df), index=df.index, dtype="bool")


def build_default_setup_filter_rules() -> list[SetupFilterRule]:
    """
    Builds deterministic setup-filter hypotheses from the current diagnostics.
    """
    return [
        SetupFilterRule(
            rule_name="baseline_all_valid_trade_plans",
            rule_description="Keep every valid trade plan. This is the comparison baseline.",
            filter_fn=_all_rows,
        ),
        SetupFilterRule(
            rule_name="avoid_continuation_after_retest_close",
            rule_description="Skip continuation-after-retest entries because diagnostics showed weak R in the recent sample.",
            filter_fn=lambda df: _series_text(df, "entry_mode") != "continuation_after_retest_close",
        ),
        SetupFilterRule(
            rule_name="retest_rejection_only",
            rule_description="Keep only entries confirmed by retest-rejection close.",
            filter_fn=lambda df: _series_text(df, "entry_mode") == "retest_rejection_close",
        ),
        SetupFilterRule(
            rule_name="avoid_rr_over_5",
            rule_description="Skip very distant target plans because high RR over 5 may be too ambitious for scalp management.",
            filter_fn=lambda df: _series_text(df, "best_rr_bucket") != "rr_over_5",
        ),
        SetupFilterRule(
            rule_name="require_tp2_tp3_available",
            rule_description="Keep only plans with TP2 and TP3 available because this group was the strongest neutral watch candidate.",
            filter_fn=lambda df: _series_text(df, "target_plan_state") == "tp2_tp3_available",
        ),
        SetupFilterRule(
            rule_name="avoid_tiny_risk_ratio",
            rule_description="Skip tiny-risk setups that may be too tight for candle-level confirmation.",
            filter_fn=lambda df: _series_text(df, "risk_ratio_bucket") != "tiny_risk_ratio",
        ),
        SetupFilterRule(
            rule_name="retest_rejection_and_tp2_tp3",
            rule_description="Keep only retest-rejection entries with TP2 and TP3 available.",
            filter_fn=lambda df: (_series_text(df, "entry_mode") == "retest_rejection_close")
            & (_series_text(df, "target_plan_state") == "tp2_tp3_available"),
        ),
        SetupFilterRule(
            rule_name="tp2_tp3_without_rr_over_5",
            rule_description="Keep TP2/TP3 plans but avoid the most distant RR-over-5 target bucket.",
            filter_fn=lambda df: (_series_text(df, "target_plan_state") == "tp2_tp3_available")
            & (_series_text(df, "best_rr_bucket") != "rr_over_5"),
        ),
        SetupFilterRule(
            rule_name="tp2_available_normal_or_wide_risk",
            rule_description="Keep TP2-available setups with normal or wide risk, avoiding tiny stops and very-wide risk.",
            filter_fn=lambda df: _series_bool(df, "tp2_available")
            & _series_text(df, "risk_ratio_bucket").isin({"normal_risk_ratio", "wide_risk_ratio"}),
        ),
    ]


def _summarize_filtered_dataset(
    dataset_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    rule: SetupFilterRule,
    min_trades_for_candidate: int,
) -> dict[str, Any]:
    """
    Converts one filtered result into comparable strategy metrics.
    """
    baseline_trades = len(dataset_df)
    kept_trades = len(filtered_df)
    coverage_rate = kept_trades / baseline_trades if baseline_trades else 0.0

    managed_r = _series_float(filtered_df, "managed_realized_r").dropna()
    positive_rate = float((managed_r > 0).mean()) if not managed_r.empty else 0.0
    loss_rate = float((managed_r < 0).mean()) if not managed_r.empty else 0.0
    average_r = float(managed_r.mean()) if not managed_r.empty else 0.0
    median_r = float(managed_r.median()) if not managed_r.empty else 0.0
    total_r = float(managed_r.sum()) if not managed_r.empty else 0.0

    partial_rate = float(filtered_df.get("partial_exit_hit", pd.Series(dtype="bool")).fillna(False).mean()) if kept_trades else 0.0
    lock_rate = float(filtered_df.get("breakeven_or_profit_lock_activated", pd.Series(dtype="bool")).fillna(False).mean()) if kept_trades else 0.0

    # Penalize tiny samples so a one-trade lucky filter does not look like a real strategy.
    sample_penalty = max(0.0, (min_trades_for_candidate - kept_trades) * 8.0)
    filter_score = (average_r * 100.0) + (positive_rate * 35.0) + (median_r * 25.0) - sample_penalty

    if kept_trades == 0:
        validation_state = "reject_no_trades_kept"
    elif kept_trades < min_trades_for_candidate:
        validation_state = "hypothesis_only_insufficient_sample"
    elif average_r > 0 and positive_rate >= 0.55 and median_r > 0:
        validation_state = "candidate_keep_filter_needs_larger_validation"
    elif average_r <= 0:
        validation_state = "not_profitable_filter_do_not_lock"
    else:
        validation_state = "mixed_filter_needs_more_data"

    return {
        "rule_name": rule.rule_name,
        "rule_description": rule.rule_description,
        "baseline_trades": baseline_trades,
        "kept_trades": kept_trades,
        "coverage_rate": round(coverage_rate, 6),
        "positive_exit_rate": round(positive_rate, 6),
        "loss_exit_rate": round(loss_rate, 6),
        "average_managed_r": round(average_r, 6),
        "median_managed_r": round(median_r, 6),
        "total_managed_r": round(total_r, 6),
        "partial_exit_rate": round(partial_rate, 6),
        "breakeven_or_lock_rate": round(lock_rate, 6),
        "filter_score": round(filter_score, 6),
        "validation_state": validation_state,
    }


def validate_setup_filters_on_dataset(
    dataset_df: pd.DataFrame,
    min_trades_for_candidate: int = 8,
) -> pd.DataFrame:
    """
    Applies multiple setup-filter hypotheses to the same cached diagnostic dataset.
    """
    if dataset_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for rule in build_default_setup_filter_rules():
        try:
            keep_mask = rule.filter_fn(dataset_df).fillna(False).astype(bool)
        except Exception:  # noqa: BLE001 - a bad rule should not break the full report.
            keep_mask = pd.Series([False] * len(dataset_df), index=dataset_df.index, dtype="bool")

        filtered_df = dataset_df[keep_mask].copy()
        rows.append(
            _summarize_filtered_dataset(
                dataset_df=dataset_df,
                filtered_df=filtered_df,
                rule=rule,
                min_trades_for_candidate=min_trades_for_candidate,
            )
        )

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        return result_df

    return result_df.sort_values(
        by=["validation_state", "filter_score", "kept_trades"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def run_setup_filter_validation(
    config: SetupFilterValidationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Builds one cached setup dataset and validates filter hypotheses against it.
    """
    diagnostic_config = SetupFilterDiagnosticConfig(
        symbol=config.symbol,
        timeframes=config.timeframes,
        analysis_limit=config.analysis_limit,
        dataset_limit=config.dataset_limit,
        warmup_candles=config.warmup_candles,
        horizon_candles=config.horizon_candles,
        cutoff_step_candles=config.cutoff_step_candles,
        max_cutoffs=config.max_cutoffs,
        management_policy_name=config.management_policy_name,
        minimum_rr=config.minimum_rr,
        active_only=config.active_only,
    )

    # Build the expensive cutoff dataset once, then test all filter rules cheaply.
    dataset_df, workload_df = build_setup_filter_diagnostic_dataset(diagnostic_config)
    validation_df = validate_setup_filters_on_dataset(
        dataset_df=dataset_df,
        min_trades_for_candidate=config.min_trades_for_candidate,
    )
    return validation_df, dataset_df, workload_df
