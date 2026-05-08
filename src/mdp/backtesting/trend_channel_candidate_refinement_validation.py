from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import pandas as pd

from mdp.backtesting.trend_channel_entry_refinement_validation import (
    EntryRefinementValidationConfig,
    build_entry_refinement_dataset,
)
from mdp.backtesting.trend_channel_simple_strategy_candidate_validation import select_candidate_rows
from mdp.backtesting.trend_channel_simple_strategy_walk_forward import build_chronological_fold_table


@dataclass(frozen=True)
class CandidateRefinementValidationConfig:
    """
    Stores settings for testing small refinements to the best simple trend-channel candidate.
    """

    symbol: str
    timeframes: str
    analysis_limit: int = 220
    dataset_limit: int = 3200
    warmup_candles: int = 220
    horizon_candles: int = 100
    cutoff_step_candles: int = 20
    max_cutoffs: int | None = 70
    fold_count: int = 3
    management_policy_name: str = "targetless_scalp_manage_only"
    minimum_rr: float = 1.50
    min_trades_for_candidate: int = 8
    active_only: bool = True
    quick: bool = False


@dataclass(frozen=True)
class CandidateRefinementRule:
    """
    Describes one small refinement rule applied to the current hybrid candidate.
    """

    rule_name: str
    rule_description: str


def build_default_refinement_rules() -> list[CandidateRefinementRule]:
    """
    Builds simple, market-agnostic refinements inspired by failure diagnostics.
    """
    return [
        CandidateRefinementRule(
            rule_name="hybrid_reference",
            rule_description="Current hybrid candidate: retest-rejection enters now; continuation entries require 2-candle confirmation.",
        ),
        CandidateRefinementRule(
            rule_name="avoid_context_retest_tp2_only",
            rule_description="Skip bearish-context retest-rejection setups when only TP2 is available; this targets the strongest full-loss context.",
        ),
        CandidateRefinementRule(
            rule_name="retest_requires_tp2_tp3",
            rule_description="Retest-rejection entries require TP2 and TP3 availability; continuation entries keep the 2-candle confirmation rule.",
        ),
        CandidateRefinementRule(
            rule_name="avoid_all_tp2_only",
            rule_description="Skip all selected hybrid trades where only TP2 is available.",
        ),
        CandidateRefinementRule(
            rule_name="require_tp2_tp3_all",
            rule_description="Keep only selected hybrid trades with TP2 and TP3 available.",
        ),
        CandidateRefinementRule(
            rule_name="avoid_bearish_context_alignment",
            rule_description="Skip bearish-context-alignment setups and keep pullback/mixed contexts only.",
        ),
        CandidateRefinementRule(
            rule_name="keep_pullback_or_tp2_tp3",
            rule_description="Keep pullback scenarios or any setup with TP2 and TP3 available.",
        ),
    ]


def _contains_text(series: pd.Series, text: str) -> pd.Series:
    """
    Checks if a string column contains a target phrase.
    """
    return series.fillna("").astype(str).str.contains(text, case=False, regex=False)


def _equals_text(series: pd.Series, text: str) -> pd.Series:
    """
    Checks if a string column exactly matches the target text.
    """
    return series.fillna("").astype(str) == text


def _add_compact_keys(candidate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds compact feature keys used by the refinement filters.
    """
    if candidate_df.empty:
        return candidate_df.copy()

    result_df = candidate_df.copy()
    result_df["scenario_event_key"] = (
        result_df.get("scenario_state", pd.Series(dtype="object")).fillna("unknown").astype(str)
        + "|"
        + result_df.get("event_state", pd.Series(dtype="object")).fillna("unknown").astype(str)
    )
    result_df["entry_target_key"] = (
        result_df.get("entry_mode", pd.Series(dtype="object")).fillna("unknown").astype(str)
        + "|"
        + result_df.get("target_plan_state", pd.Series(dtype="object")).fillna("unknown").astype(str)
    )
    return result_df


def _deduplicate_by_cutoff(candidate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps one selected trade per historical cutoff.
    """
    if candidate_df.empty or "analysis_cutoff" not in candidate_df.columns:
        return candidate_df.reset_index(drop=True)

    # Each strategy candidate should only produce one trade at a cutoff.
    return (
        candidate_df.sort_values(["analysis_cutoff", "entry_variant_name"])
        .drop_duplicates(subset=["analysis_cutoff"], keep="first")
        .reset_index(drop=True)
    )


def select_refined_candidate_rows(dataset_df: pd.DataFrame, rule_name: str) -> pd.DataFrame:
    """
    Selects rows for one refined candidate rule.
    """
    hybrid_df = select_candidate_rows(dataset_df, "hybrid_retest_now_continuation_confirm2")
    hybrid_df = _add_compact_keys(hybrid_df)

    if hybrid_df.empty:
        return hybrid_df

    target_state = hybrid_df.get("target_plan_state", pd.Series(dtype="object"))
    scenario_state = hybrid_df.get("scenario_state", pd.Series(dtype="object"))
    event_state = hybrid_df.get("event_state", pd.Series(dtype="object"))
    entry_mode = hybrid_df.get("entry_mode", pd.Series(dtype="object"))

    # This is the current candidate and acts as the benchmark.
    if rule_name == "hybrid_reference":
        return _deduplicate_by_cutoff(hybrid_df)

    # This directly tests the strongest observed failure cluster without changing other setups.
    if rule_name == "avoid_context_retest_tp2_only":
        failure_context = (
            _equals_text(scenario_state, "bearish_context_alignment")
            & _equals_text(event_state, "bearish_retest_rejection_confirmed")
            & _equals_text(entry_mode, "retest_rejection_close")
            & _equals_text(target_state, "tp2_only")
        )
        return _deduplicate_by_cutoff(hybrid_df[~failure_context].copy())

    # Retest entries are allowed only if broader target structure is available.
    if rule_name == "retest_requires_tp2_tp3":
        retest_tp2_only = _equals_text(entry_mode, "retest_rejection_close") & ~_contains_text(target_state, "tp2_tp3")
        return _deduplicate_by_cutoff(hybrid_df[~retest_tp2_only].copy())

    # This checks whether TP2-only plans are generally weak for this candidate.
    if rule_name == "avoid_all_tp2_only":
        return _deduplicate_by_cutoff(hybrid_df[~_equals_text(target_state, "tp2_only")].copy())

    # This is stricter and tests whether only multi-target structure should be traded.
    if rule_name == "require_tp2_tp3_all":
        return _deduplicate_by_cutoff(hybrid_df[_contains_text(target_state, "tp2_tp3")].copy())

    # This checks whether the bearish-context-alignment scenario is hurting the edge.
    if rule_name == "avoid_bearish_context_alignment":
        return _deduplicate_by_cutoff(hybrid_df[~_equals_text(scenario_state, "bearish_context_alignment")].copy())

    # This keeps either pullback context or better target structure, without requiring both.
    if rule_name == "keep_pullback_or_tp2_tp3":
        keep_mask = _contains_text(scenario_state, "pullback") | _contains_text(target_state, "tp2_tp3")
        return _deduplicate_by_cutoff(hybrid_df[keep_mask].copy())

    raise ValueError(f"Unknown refinement rule: {rule_name}")


def _summarize_selected_rows(
    selected_df: pd.DataFrame,
    baseline_attempts: int,
    min_trades_for_candidate: int,
) -> dict[str, Any]:
    """
    Computes core performance metrics for selected trades.
    """
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

    sample_penalty = max(0, min_trades_for_candidate - kept_trades) * 8.0
    low_coverage_penalty = 10.0 if 0 < coverage_rate < 0.25 else 0.0
    rule_score = (
        average_r * 120.0
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
    elif average_r >= 0.10 and median_r > 0 and positive_rate >= 0.60:
        validation_state = "promising_refinement_needs_walk_forward"
    elif average_r > 0 and median_r > 0 and positive_rate >= 0.55:
        validation_state = "weak_positive_refinement_validate_harder"
    elif average_r <= 0:
        validation_state = "not_profitable_refinement_do_not_lock"
    else:
        validation_state = "neutral_refinement_watch_only"

    return {
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
        "refinement_score": rule_score,
        "validation_state": validation_state,
    }


def summarize_refinement_rules(
    dataset_df: pd.DataFrame,
    rules: list[CandidateRefinementRule] | None = None,
    min_trades_for_candidate: int = 8,
) -> pd.DataFrame:
    """
    Summarizes every targeted refinement rule on the same cached dataset.
    """
    if dataset_df.empty:
        return pd.DataFrame()

    rules = rules or build_default_refinement_rules()
    baseline_attempts = int(
        select_candidate_rows(dataset_df, "baseline_all_current_close")["analysis_cutoff"].nunique()
    )
    rows: list[dict[str, Any]] = []

    for rule in rules:
        selected_df = select_refined_candidate_rows(dataset_df, rule.rule_name)
        row = _summarize_selected_rows(selected_df, baseline_attempts, min_trades_for_candidate)
        row["rule_name"] = rule.rule_name
        row["rule_description"] = rule.rule_description
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["validation_state", "refinement_score"],
        ascending=[True, False],
    ).reset_index(drop=True)


def _assign_folds(selected_df: pd.DataFrame, fold_table_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns every selected trade to a chronological fold.
    """
    if selected_df.empty or fold_table_df.empty:
        return selected_df.copy()

    result_df = selected_df.copy()
    result_df["analysis_cutoff"] = pd.to_datetime(result_df["analysis_cutoff"])
    result_df["fold_id"] = pd.NA
    result_df["fold_position"] = pd.NA

    # Each fold owns a contiguous time range from the selected historical cutoffs.
    for _, fold_row in fold_table_df.iterrows():
        start_time = pd.to_datetime(fold_row["fold_start_cutoff"])
        end_time = pd.to_datetime(fold_row["fold_end_cutoff"])
        mask = (result_df["analysis_cutoff"] >= start_time) & (result_df["analysis_cutoff"] <= end_time)
        result_df.loc[mask, "fold_id"] = int(fold_row["fold_id"])
        result_df.loc[mask, "fold_position"] = fold_row["fold_position"]

    return result_df


def summarize_refinement_walk_forward(
    dataset_df: pd.DataFrame,
    fold_count: int,
    rules: list[CandidateRefinementRule] | None = None,
    min_trades_for_candidate: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tests each refinement rule across chronological folds.
    """
    if dataset_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    rules = rules or build_default_refinement_rules()
    fold_table_df = build_chronological_fold_table(dataset_df, fold_count=fold_count)
    fold_rows: list[dict[str, Any]] = []
    robust_rows: list[dict[str, Any]] = []

    for rule in rules:
        selected_df = select_refined_candidate_rows(dataset_df, rule.rule_name)
        selected_df = _assign_folds(selected_df, fold_table_df)

        fold_average_values: list[float] = []
        fold_median_values: list[float] = []
        positive_average_folds = 0
        positive_median_folds = 0
        folds_with_min_trades = 0

        for _, fold_row in fold_table_df.iterrows():
            fold_id = int(fold_row["fold_id"])
            fold_df = selected_df[selected_df.get("fold_id") == fold_id].copy()
            metrics = _summarize_selected_rows(
                fold_df,
                baseline_attempts=max(1, int((pd.to_datetime(dataset_df["analysis_cutoff"]) >= pd.to_datetime(fold_row["fold_start_cutoff"])).sum())),
                min_trades_for_candidate=1,
            )
            fold_average_values.append(metrics["average_managed_r"])
            fold_median_values.append(metrics["median_managed_r"])
            positive_average_folds += int(metrics["average_managed_r"] > 0)
            positive_median_folds += int(metrics["median_managed_r"] > 0)
            folds_with_min_trades += int(metrics["kept_trades"] >= 2)

            fold_rows.append(
                {
                    "rule_name": rule.rule_name,
                    "rule_description": rule.rule_description,
                    "fold_id": fold_id,
                    "fold_position": fold_row["fold_position"],
                    "fold_start_cutoff": fold_row["fold_start_cutoff"],
                    "fold_end_cutoff": fold_row["fold_end_cutoff"],
                    **metrics,
                }
            )

        selected_summary = _summarize_selected_rows(
            selected_df,
            baseline_attempts=int(select_candidate_rows(dataset_df, "baseline_all_current_close")["analysis_cutoff"].nunique()),
            min_trades_for_candidate=min_trades_for_candidate,
        )
        fold_count_safe = max(1, len(fold_table_df))
        positive_average_r_fold_rate = positive_average_folds / fold_count_safe
        positive_median_r_fold_rate = positive_median_folds / fold_count_safe
        folds_with_min_trades_rate = folds_with_min_trades / fold_count_safe
        min_average_r = min(fold_average_values) if fold_average_values else 0.0
        max_average_r = max(fold_average_values) if fold_average_values else 0.0

        if selected_summary["kept_trades"] < min_trades_for_candidate:
            robustness_state = "insufficient_total_trades_do_not_lock"
        elif min_average_r > 0 and positive_average_r_fold_rate == 1.0 and selected_summary["average_managed_r"] >= 0.10:
            robustness_state = "strong_refinement_candidate_needs_out_of_sample"
        elif selected_summary["average_managed_r"] > 0 and positive_average_r_fold_rate >= 0.67:
            robustness_state = "weak_positive_refinement_validate_harder"
        else:
            robustness_state = "not_robust_do_not_lock"

        robust_rows.append(
            {
                "rule_name": rule.rule_name,
                "rule_description": rule.rule_description,
                "validation_folds": len(fold_table_df),
                "kept_trades": selected_summary["kept_trades"],
                "coverage_rate": selected_summary["coverage_rate"],
                "positive_exit_rate": selected_summary["positive_exit_rate"],
                "loss_exit_rate": selected_summary["loss_exit_rate"],
                "average_managed_r": selected_summary["average_managed_r"],
                "median_managed_r": selected_summary["median_managed_r"],
                "total_managed_r": selected_summary["total_managed_r"],
                "min_fold_average_r": min_average_r,
                "max_fold_average_r": max_average_r,
                "positive_average_r_fold_rate": positive_average_r_fold_rate,
                "positive_median_r_fold_rate": positive_median_r_fold_rate,
                "folds_with_min_trades_rate": folds_with_min_trades_rate,
                "refinement_score": selected_summary["refinement_score"],
                "robustness_state": robustness_state,
            }
        )

    robust_df = pd.DataFrame(robust_rows).sort_values(
        ["robustness_state", "refinement_score"],
        ascending=[True, False],
    ).reset_index(drop=True)
    fold_df = pd.DataFrame(fold_rows).reset_index(drop=True)
    return robust_df, fold_df


def build_refinement_preview(dataset_df: pd.DataFrame, rule_name: str) -> pd.DataFrame:
    """
    Builds a compact preview for the best refinement rule.
    """
    selected_df = select_refined_candidate_rows(dataset_df, rule_name)
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


def _build_entry_dataset(config: CandidateRefinementValidationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds the expensive entry-variant dataset once.
    """
    # Quick mode keeps the same logic path but avoids heavy laptop runs.
    dataset_limit = min(config.dataset_limit, 1400) if config.quick else config.dataset_limit
    analysis_limit = min(config.analysis_limit, 200) if config.quick else config.analysis_limit
    warmup_candles = min(config.warmup_candles, 200) if config.quick else config.warmup_candles
    horizon_candles = min(config.horizon_candles, 80) if config.quick else config.horizon_candles
    max_cutoffs = min(config.max_cutoffs or 35, 35) if config.quick else config.max_cutoffs

    entry_config = EntryRefinementValidationConfig(
        symbol=config.symbol,
        timeframes=config.timeframes,
        analysis_limit=analysis_limit,
        dataset_limit=dataset_limit,
        warmup_candles=warmup_candles,
        horizon_candles=horizon_candles,
        cutoff_step_candles=config.cutoff_step_candles,
        max_cutoffs=max_cutoffs,
        management_policy_name=config.management_policy_name,
        minimum_rr=config.minimum_rr,
        min_trades_for_candidate=config.min_trades_for_candidate,
        active_only=config.active_only,
    )
    return build_entry_refinement_dataset(entry_config)


def run_candidate_refinement_validation(
    config: CandidateRefinementValidationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Runs targeted refinement validation using one cached entry dataset.
    """
    start_time = perf_counter()

    dataset_df, workload_df = _build_entry_dataset(config)
    rules = build_default_refinement_rules()
    summary_df = summarize_refinement_rules(
        dataset_df=dataset_df,
        rules=rules,
        min_trades_for_candidate=config.min_trades_for_candidate,
    )
    robust_df, fold_df = summarize_refinement_walk_forward(
        dataset_df=dataset_df,
        fold_count=config.fold_count,
        rules=rules,
        min_trades_for_candidate=config.min_trades_for_candidate,
    )

    best_rule_name = None
    if not robust_df.empty:
        best_rule_name = str(robust_df.iloc[0].get("rule_name"))
    preview_df = build_refinement_preview(dataset_df, best_rule_name) if best_rule_name else pd.DataFrame()

    workload_df = workload_df.copy()
    workload_df["refinement_rule_count"] = len(rules)
    workload_df["fold_count"] = config.fold_count
    workload_df["elapsed_seconds_total_refinement_validation"] = round(perf_counter() - start_time, 3)
    workload_df["cached_design"] = "entry_dataset_built_once_then_refinement_rules_and_folds_reused"

    return dataset_df, summary_df, robust_df, fold_df, workload_df, preview_df
