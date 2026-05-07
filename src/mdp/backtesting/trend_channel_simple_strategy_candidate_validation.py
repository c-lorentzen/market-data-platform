from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import pandas as pd

from mdp.backtesting.trend_channel_entry_refinement_validation import (
    EntryRefinementValidationConfig,
    build_entry_refinement_dataset,
)


@dataclass(frozen=True)
class SimpleStrategyCandidateValidationConfig:
    """
    Stores settings for validating simple complete strategy candidates.
    """

    symbol: str
    timeframes: str
    analysis_limit: int = 220
    dataset_limit: int = 1200
    warmup_candles: int = 220
    horizon_candles: int = 80
    cutoff_step_candles: int = 20
    max_cutoffs: int | None = 40
    management_policy_name: str = "targetless_scalp_manage_only"
    minimum_rr: float = 1.50
    min_trades_for_candidate: int = 8
    active_only: bool = True


@dataclass(frozen=True)
class SimpleStrategyCandidate:
    """
    Describes one simple rule set that combines setup filter + entry timing.
    """

    candidate_name: str
    candidate_description: str


def build_default_simple_strategy_candidates() -> list[SimpleStrategyCandidate]:
    """
    Builds a small market-agnostic candidate list.
    """
    return [
        SimpleStrategyCandidate(
            candidate_name="baseline_all_current_close",
            candidate_description="Use every valid trade plan with the original current-close entry.",
        ),
        SimpleStrategyCandidate(
            candidate_name="delay_2_all",
            candidate_description="Use every valid trade plan, but wait two candles before entering.",
        ),
        SimpleStrategyCandidate(
            candidate_name="confirm_2_all",
            candidate_description="Use only valid trade plans that pass two-candle directional confirmation.",
        ),
        SimpleStrategyCandidate(
            candidate_name="retest_rejection_current_only",
            candidate_description="Enter only retest-rejection setups at the current close.",
        ),
        SimpleStrategyCandidate(
            candidate_name="retest_rejection_tp2_tp3_current",
            candidate_description="Enter only retest-rejection setups when TP2 and TP3 are both available.",
        ),
        SimpleStrategyCandidate(
            candidate_name="continuation_confirm2_only",
            candidate_description="Enter continuation-after-retest setups only if two-candle confirmation passes.",
        ),
        SimpleStrategyCandidate(
            candidate_name="hybrid_retest_now_continuation_confirm2",
            candidate_description="Enter retest-rejection setups immediately; require two-candle confirmation for continuation setups.",
        ),
        SimpleStrategyCandidate(
            candidate_name="hybrid_tp2_tp3_retest_or_confirm2",
            candidate_description="Hybrid candidate, but keep only setups where TP2 and TP3 are both available.",
        ),
        SimpleStrategyCandidate(
            candidate_name="baseline_avoid_rr_over_5",
            candidate_description="Use baseline entries but avoid very distant best-RR setups above 5R.",
        ),
    ]


def _safe_float(value: Any) -> float | None:
    """
    Converts a value to float without throwing an exception.
    """
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _entered_rows(dataset_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns rows where the entry variant actually entered a trade.
    """
    if dataset_df.empty or "entered_trade" not in dataset_df.columns:
        return pd.DataFrame()
    return dataset_df[dataset_df["entered_trade"].fillna(False) == True].copy()  # noqa: E712 - pandas comparison is intentional.


def _variant_rows(dataset_df: pd.DataFrame, variant_name: str) -> pd.DataFrame:
    """
    Returns entered rows for one entry variant.
    """
    entered_df = _entered_rows(dataset_df)
    if entered_df.empty:
        return entered_df
    return entered_df[entered_df["entry_variant_name"] == variant_name].copy()


def _contains_text(series: pd.Series, text: str) -> pd.Series:
    """
    Checks whether a text column contains a phrase.
    """
    return series.fillna("").astype(str).str.contains(text, case=False, regex=False)


def _best_rr_at_most(series: pd.Series, max_rr: float) -> pd.Series:
    """
    Checks whether best available RR is below a maximum value.
    """
    return series.apply(lambda value: (_safe_float(value) is not None) and (_safe_float(value) <= max_rr))


def _deduplicate_by_cutoff(candidate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps one selected candidate trade per cutoff.
    """
    if candidate_df.empty or "analysis_cutoff" not in candidate_df.columns:
        return candidate_df.reset_index(drop=True)

    # One strategy candidate should not enter multiple variants at the same cutoff.
    return (
        candidate_df.sort_values(["analysis_cutoff", "entry_variant_name"])
        .drop_duplicates(subset=["analysis_cutoff"], keep="first")
        .reset_index(drop=True)
    )


def select_candidate_rows(dataset_df: pd.DataFrame, candidate_name: str) -> pd.DataFrame:
    """
    Selects rows matching one complete simple strategy candidate.
    """
    baseline_df = _variant_rows(dataset_df, "baseline_current_close")
    delay2_df = _variant_rows(dataset_df, "delay_2_close")
    confirm2_df = _variant_rows(dataset_df, "confirm_2_directional_close")

    if candidate_name == "baseline_all_current_close":
        return _deduplicate_by_cutoff(baseline_df)

    if candidate_name == "delay_2_all":
        return _deduplicate_by_cutoff(delay2_df)

    if candidate_name == "confirm_2_all":
        return _deduplicate_by_cutoff(confirm2_df)

    if candidate_name == "retest_rejection_current_only":
        selected = baseline_df[baseline_df.get("entry_mode", "") == "retest_rejection_close"].copy()
        return _deduplicate_by_cutoff(selected)

    if candidate_name == "retest_rejection_tp2_tp3_current":
        selected = baseline_df[
            (baseline_df.get("entry_mode", "") == "retest_rejection_close")
            & _contains_text(baseline_df.get("target_plan_state", pd.Series(dtype="object")), "tp2_tp3_available")
        ].copy()
        return _deduplicate_by_cutoff(selected)

    if candidate_name == "continuation_confirm2_only":
        selected = confirm2_df[confirm2_df.get("entry_mode", "") == "continuation_after_retest_close"].copy()
        return _deduplicate_by_cutoff(selected)

    if candidate_name == "hybrid_retest_now_continuation_confirm2":
        retest_now = baseline_df[baseline_df.get("entry_mode", "") == "retest_rejection_close"].copy()
        continuation_confirmed = confirm2_df[confirm2_df.get("entry_mode", "") == "continuation_after_retest_close"].copy()
        selected = pd.concat([retest_now, continuation_confirmed], ignore_index=True)
        return _deduplicate_by_cutoff(selected)

    if candidate_name == "hybrid_tp2_tp3_retest_or_confirm2":
        hybrid_df = select_candidate_rows(dataset_df, "hybrid_retest_now_continuation_confirm2")
        selected = hybrid_df[_contains_text(hybrid_df.get("target_plan_state", pd.Series(dtype="object")), "tp2_tp3_available")].copy()
        return _deduplicate_by_cutoff(selected)

    if candidate_name == "baseline_avoid_rr_over_5":
        selected = baseline_df[_best_rr_at_most(baseline_df.get("best_available_rr", pd.Series(dtype="float")), 5.0)].copy()
        return _deduplicate_by_cutoff(selected)

    raise ValueError(f"Unknown candidate_name: {candidate_name}")


def summarize_simple_strategy_candidates(
    dataset_df: pd.DataFrame,
    candidates: list[SimpleStrategyCandidate] | None = None,
    min_trades_for_candidate: int = 8,
) -> pd.DataFrame:
    """
    Summarizes performance for each complete strategy candidate.
    """
    if dataset_df.empty:
        return pd.DataFrame()

    candidates = candidates or build_default_simple_strategy_candidates()
    baseline_attempts = int(_variant_rows(dataset_df, "baseline_current_close")["analysis_cutoff"].nunique())
    rows: list[dict[str, Any]] = []

    for candidate in candidates:
        selected_df = select_candidate_rows(dataset_df, candidate.candidate_name)
        kept_trades = len(selected_df)
        coverage_rate = kept_trades / baseline_attempts if baseline_attempts else 0.0
        managed_r = pd.to_numeric(selected_df.get("managed_realized_r"), errors="coerce").dropna()

        positive_rate = float((managed_r > 0).mean()) if not managed_r.empty else 0.0
        loss_rate = float((managed_r < 0).mean()) if not managed_r.empty else 0.0
        average_r = float(managed_r.mean()) if not managed_r.empty else 0.0
        median_r = float(managed_r.median()) if not managed_r.empty else 0.0
        total_r = float(managed_r.sum()) if not managed_r.empty else 0.0
        partial_rate = float(selected_df.get("partial_exit_hit", pd.Series(dtype="bool")).fillna(False).mean()) if kept_trades else 0.0
        lock_rate = float(selected_df.get("breakeven_or_profit_lock_activated", pd.Series(dtype="bool")).fillna(False).mean()) if kept_trades else 0.0

        # The score rewards simple, profitable, sufficiently covered candidates.
        sample_penalty = max(0, min_trades_for_candidate - kept_trades) * 8.0
        low_coverage_penalty = 10.0 if 0 < coverage_rate < 0.25 else 0.0
        candidate_score = (
            average_r * 100.0
            + median_r * 25.0
            + positive_rate * 30.0
            + coverage_rate * 10.0
            - loss_rate * 15.0
            - sample_penalty
            - low_coverage_penalty
        )

        if kept_trades == 0:
            validation_state = "reject_no_trades_selected"
        elif kept_trades < min_trades_for_candidate:
            validation_state = "hypothesis_only_insufficient_sample"
        elif average_r > 0 and median_r > 0 and positive_rate >= 0.55:
            validation_state = "candidate_simple_strategy_needs_larger_validation"
        elif average_r <= 0:
            validation_state = "not_profitable_candidate_do_not_lock"
        else:
            validation_state = "neutral_candidate_watch_only"

        rows.append(
            {
                "candidate_name": candidate.candidate_name,
                "candidate_description": candidate.candidate_description,
                "baseline_attempts": baseline_attempts,
                "kept_trades": kept_trades,
                "coverage_rate": coverage_rate,
                "positive_exit_rate": positive_rate,
                "loss_exit_rate": loss_rate,
                "average_managed_r": average_r,
                "median_managed_r": median_r,
                "total_managed_r": total_r,
                "partial_exit_rate": partial_rate,
                "breakeven_or_lock_rate": lock_rate,
                "candidate_score": candidate_score,
                "validation_state": validation_state,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["validation_state", "candidate_score"],
        ascending=[True, False],
    ).reset_index(drop=True)


def build_candidate_preview(dataset_df: pd.DataFrame, candidate_name: str) -> pd.DataFrame:
    """
    Builds a preview of selected trades for one candidate.
    """
    selected_df = select_candidate_rows(dataset_df, candidate_name)
    preview_columns = [
        "analysis_cutoff",
        "entry_variant_name",
        "scenario_state",
        "event_state",
        "entry_mode",
        "entry_price",
        "risk_points",
        "available_targets",
        "target_plan_state",
        "managed_exit_state",
        "managed_realized_r",
    ]
    existing_columns = [column for column in preview_columns if column in selected_df.columns]
    return selected_df[existing_columns].reset_index(drop=True) if existing_columns else selected_df.reset_index(drop=True)


def run_simple_strategy_candidate_validation(
    config: SimpleStrategyCandidateValidationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Builds the entry-variant dataset once and validates simple complete candidates.
    """
    start_time = perf_counter()

    # Reuse the entry-refinement dataset builder so the expensive strategy replay is cached once.
    entry_config = EntryRefinementValidationConfig(
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
        min_trades_for_candidate=config.min_trades_for_candidate,
        active_only=config.active_only,
    )
    dataset_df, workload_df = build_entry_refinement_dataset(entry_config)
    candidates = build_default_simple_strategy_candidates()
    summary_df = summarize_simple_strategy_candidates(
        dataset_df=dataset_df,
        candidates=candidates,
        min_trades_for_candidate=config.min_trades_for_candidate,
    )

    best_candidate_name = None
    if not summary_df.empty:
        best_candidate_name = str(summary_df.iloc[0].get("candidate_name"))
    preview_df = build_candidate_preview(dataset_df, best_candidate_name) if best_candidate_name else pd.DataFrame()

    workload_df = workload_df.copy()
    workload_df["simple_candidate_count"] = len(candidates)
    workload_df["elapsed_seconds_total_candidate_validation"] = round(perf_counter() - start_time, 3)
    workload_df["cached_design"] = "entry_variants_built_once_then_candidate_rules_tested"

    return dataset_df, summary_df, workload_df, preview_df
