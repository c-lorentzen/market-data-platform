from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import pandas as pd

from mdp.backtesting.trend_channel_entry_refinement_validation import (
    EntryRefinementValidationConfig,
    build_entry_refinement_dataset,
)
from mdp.backtesting.trend_channel_simple_strategy_candidate_validation import (
    SimpleStrategyCandidate,
    build_default_simple_strategy_candidates,
    select_candidate_rows,
)


@dataclass(frozen=True)
class SimpleStrategyWalkForwardConfig:
    """
    Stores settings for chronological walk-forward validation of simple strategy candidates.
    """

    symbol: str
    timeframes: str
    analysis_limit: int = 220
    dataset_limit: int = 3200
    warmup_candles: int = 220
    horizon_candles: int = 100
    cutoff_step_candles: int = 20
    max_cutoffs: int | None = 70
    management_policy_name: str = "targetless_scalp_manage_only"
    minimum_rr: float = 1.50
    min_trades_per_fold: int = 3
    min_total_trades_for_candidate: int = 12
    fold_count: int = 3
    active_only: bool = True
    candidate_name: str | None = None
    quick: bool = False


def _safe_mean(series: pd.Series) -> float:
    """
    Returns the mean of a numeric series while safely handling empty values.
    """
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else 0.0


def _safe_median(series: pd.Series) -> float:
    """
    Returns the median of a numeric series while safely handling empty values.
    """
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.median()) if not numeric.empty else 0.0


def _safe_sum(series: pd.Series) -> float:
    """
    Returns the sum of a numeric series while safely handling empty values.
    """
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.sum()) if not numeric.empty else 0.0


def _candidate_list(config: SimpleStrategyWalkForwardConfig) -> list[SimpleStrategyCandidate]:
    """
    Returns the candidates to evaluate. A single candidate can be selected for faster focused tests.
    """
    candidates = build_default_simple_strategy_candidates()
    if config.candidate_name:
        candidates = [candidate for candidate in candidates if candidate.candidate_name == config.candidate_name]
    if not candidates:
        raise ValueError(f"Unknown candidate_name: {config.candidate_name}")
    return candidates


def _build_entry_dataset(config: SimpleStrategyWalkForwardConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds the expensive cutoff-aware entry dataset once so every fold and candidate can reuse it.
    """
    # Quick mode keeps laptop tests small without changing the validation logic.
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
        min_trades_for_candidate=config.min_total_trades_for_candidate,
        active_only=config.active_only,
    )
    return build_entry_refinement_dataset(entry_config)


def build_chronological_fold_table(dataset_df: pd.DataFrame, fold_count: int) -> pd.DataFrame:
    """
    Splits unique analysis cutoffs into non-overlapping chronological folds.
    """
    if dataset_df.empty or "analysis_cutoff" not in dataset_df.columns:
        return pd.DataFrame()

    cutoffs = pd.Series(pd.to_datetime(dataset_df["analysis_cutoff"].dropna().unique())).sort_values().reset_index(drop=True)
    if cutoffs.empty:
        return pd.DataFrame()

    fold_count = max(1, min(int(fold_count), len(cutoffs)))
    rows: list[dict[str, Any]] = []

    # Use integer boundaries so folds are deterministic and non-overlapping.
    for fold_index in range(fold_count):
        start_idx = int(round(fold_index * len(cutoffs) / fold_count))
        end_idx = int(round((fold_index + 1) * len(cutoffs) / fold_count))
        fold_cutoffs = cutoffs.iloc[start_idx:end_idx]
        if fold_cutoffs.empty:
            continue
        rows.append(
            {
                "fold_id": fold_index + 1,
                "fold_start_cutoff": fold_cutoffs.iloc[0],
                "fold_end_cutoff": fold_cutoffs.iloc[-1],
                "fold_cutoff_count": int(len(fold_cutoffs)),
                "fold_position": "earliest" if fold_index == 0 else ("latest" if fold_index == fold_count - 1 else "middle"),
            }
        )

    return pd.DataFrame(rows)


def _filter_dataset_to_fold(dataset_df: pd.DataFrame, fold_row: pd.Series) -> pd.DataFrame:
    """
    Keeps only rows whose cutoff belongs to one fold.
    """
    if dataset_df.empty:
        return dataset_df.copy()

    cutoff_times = pd.to_datetime(dataset_df["analysis_cutoff"])
    start_time = pd.to_datetime(fold_row["fold_start_cutoff"])
    end_time = pd.to_datetime(fold_row["fold_end_cutoff"])
    mask = (cutoff_times >= start_time) & (cutoff_times <= end_time)
    return dataset_df.loc[mask].copy()


def _summarize_candidate_rows(candidate_df: pd.DataFrame) -> dict[str, Any]:
    """
    Summarizes performance for already-selected candidate trades.
    """
    kept_trades = int(len(candidate_df))
    managed_r = pd.to_numeric(candidate_df.get("managed_realized_r"), errors="coerce").dropna()
    positive_exit_rate = float((managed_r > 0).mean()) if not managed_r.empty else 0.0
    loss_exit_rate = float((managed_r < 0).mean()) if not managed_r.empty else 0.0

    return {
        "kept_trades": kept_trades,
        "positive_exit_rate": positive_exit_rate,
        "loss_exit_rate": loss_exit_rate,
        "average_managed_r": float(managed_r.mean()) if not managed_r.empty else 0.0,
        "median_managed_r": float(managed_r.median()) if not managed_r.empty else 0.0,
        "total_managed_r": float(managed_r.sum()) if not managed_r.empty else 0.0,
        "partial_exit_rate": float(candidate_df.get("partial_exit_hit", pd.Series(dtype="bool")).fillna(False).mean()) if kept_trades else 0.0,
        "breakeven_or_lock_rate": float(candidate_df.get("breakeven_or_profit_lock_activated", pd.Series(dtype="bool")).fillna(False).mean()) if kept_trades else 0.0,
    }


def build_candidate_fold_results(
    dataset_df: pd.DataFrame,
    fold_table_df: pd.DataFrame,
    candidates: list[SimpleStrategyCandidate],
) -> pd.DataFrame:
    """
    Evaluates every candidate on every chronological fold.
    """
    if dataset_df.empty or fold_table_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for _, fold_row in fold_table_df.iterrows():
        fold_dataset_df = _filter_dataset_to_fold(dataset_df, fold_row)
        baseline_attempts = int(
            fold_dataset_df[
                (fold_dataset_df.get("entry_variant_name") == "baseline_current_close")
                & (fold_dataset_df.get("entered_trade").fillna(False) == True)  # noqa: E712 - pandas comparison is intentional.
            ]["analysis_cutoff"].nunique()
        )

        for candidate in candidates:
            selected_df = select_candidate_rows(fold_dataset_df, candidate.candidate_name)
            summary = _summarize_candidate_rows(selected_df)
            coverage_rate = summary["kept_trades"] / baseline_attempts if baseline_attempts else 0.0

            rows.append(
                {
                    "fold_id": int(fold_row["fold_id"]),
                    "fold_position": fold_row["fold_position"],
                    "fold_start_cutoff": fold_row["fold_start_cutoff"],
                    "fold_end_cutoff": fold_row["fold_end_cutoff"],
                    "fold_cutoff_count": int(fold_row["fold_cutoff_count"]),
                    "candidate_name": candidate.candidate_name,
                    "candidate_description": candidate.candidate_description,
                    "baseline_attempts": baseline_attempts,
                    "coverage_rate": coverage_rate,
                    **summary,
                }
            )

    return pd.DataFrame(rows)


def summarize_walk_forward_candidates(
    fold_results_df: pd.DataFrame,
    config: SimpleStrategyWalkForwardConfig,
) -> pd.DataFrame:
    """
    Aggregates fold performance and assigns an honest walk-forward readiness state.
    """
    if fold_results_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for candidate_name, group_df in fold_results_df.groupby("candidate_name"):
        fold_count = int(group_df["fold_id"].nunique())
        total_kept_trades = int(pd.to_numeric(group_df["kept_trades"], errors="coerce").fillna(0).sum())
        mean_kept_trades = _safe_mean(group_df["kept_trades"])
        min_fold_trades = int(pd.to_numeric(group_df["kept_trades"], errors="coerce").fillna(0).min())
        mean_average_r = _safe_mean(group_df["average_managed_r"])
        min_average_r = float(pd.to_numeric(group_df["average_managed_r"], errors="coerce").min())
        max_average_r = float(pd.to_numeric(group_df["average_managed_r"], errors="coerce").max())
        mean_median_r = _safe_mean(group_df["median_managed_r"])
        mean_positive_rate = _safe_mean(group_df["positive_exit_rate"])
        mean_loss_rate = _safe_mean(group_df["loss_exit_rate"])
        positive_average_r_fold_rate = float((pd.to_numeric(group_df["average_managed_r"], errors="coerce") > 0).mean())
        positive_median_r_fold_rate = float((pd.to_numeric(group_df["median_managed_r"], errors="coerce") > 0).mean())
        folds_with_min_trades_rate = float((pd.to_numeric(group_df["kept_trades"], errors="coerce") >= config.min_trades_per_fold).mean())

        # Penalize candidates that only work in one fold or have tiny fold counts.
        sample_penalty = max(0, config.min_total_trades_for_candidate - total_kept_trades) * 2.5
        fold_sample_penalty = max(0, config.min_trades_per_fold - min_fold_trades) * 5.0
        walk_forward_score = (
            mean_average_r * 140.0
            + mean_median_r * 40.0
            + mean_positive_rate * 35.0
            + positive_average_r_fold_rate * 35.0
            + positive_median_r_fold_rate * 15.0
            + folds_with_min_trades_rate * 20.0
            - mean_loss_rate * 25.0
            - sample_penalty
            - fold_sample_penalty
        )

        if total_kept_trades < config.min_total_trades_for_candidate:
            validation_state = "insufficient_total_trades_do_not_lock"
        elif min_fold_trades < config.min_trades_per_fold:
            validation_state = "insufficient_trades_per_fold_do_not_lock"
        elif mean_average_r > 0 and min_average_r > 0 and mean_median_r > 0 and positive_average_r_fold_rate >= 0.67:
            validation_state = "walk_forward_candidate_promising_needs_larger_data"
        elif mean_average_r > 0 and mean_median_r > 0:
            validation_state = "directionally_promising_but_unstable"
        else:
            validation_state = "not_walk_forward_robust_do_not_lock"

        rows.append(
            {
                "candidate_name": candidate_name,
                "fold_count": fold_count,
                "total_kept_trades": total_kept_trades,
                "mean_kept_trades_per_fold": round(mean_kept_trades, 4),
                "min_fold_trades": min_fold_trades,
                "mean_positive_exit_rate": mean_positive_rate,
                "mean_loss_exit_rate": mean_loss_rate,
                "mean_average_managed_r": mean_average_r,
                "min_average_managed_r": min_average_r,
                "max_average_managed_r": max_average_r,
                "mean_median_managed_r": mean_median_r,
                "positive_average_r_fold_rate": positive_average_r_fold_rate,
                "positive_median_r_fold_rate": positive_median_r_fold_rate,
                "folds_with_min_trades_rate": folds_with_min_trades_rate,
                "walk_forward_score": walk_forward_score,
                "validation_state": validation_state,
            }
        )

    return pd.DataFrame(rows).sort_values("walk_forward_score", ascending=False).reset_index(drop=True)


def build_best_candidate_preview(dataset_df: pd.DataFrame, best_candidate_name: str | None) -> pd.DataFrame:
    """
    Builds a compact preview of the selected trades for the best candidate.
    """
    if not best_candidate_name or dataset_df.empty:
        return pd.DataFrame()

    selected_df = select_candidate_rows(dataset_df, best_candidate_name)
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


def run_simple_strategy_walk_forward_validation(
    config: SimpleStrategyWalkForwardConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Runs chronological fold validation from one cached entry-variant dataset.
    """
    start_time = perf_counter()

    # Build the expensive strategy replay once, then reuse the dataset across folds and candidates.
    dataset_df, workload_df = _build_entry_dataset(config)
    candidates = _candidate_list(config)
    fold_table_df = build_chronological_fold_table(dataset_df, config.fold_count)
    fold_results_df = build_candidate_fold_results(dataset_df, fold_table_df, candidates)
    summary_df = summarize_walk_forward_candidates(fold_results_df, config)

    best_candidate_name = str(summary_df.iloc[0]["candidate_name"]) if not summary_df.empty else None
    preview_df = build_best_candidate_preview(dataset_df, best_candidate_name)

    workload_df = workload_df.copy()
    workload_df["fold_count"] = int(len(fold_table_df))
    workload_df["candidate_count"] = int(len(candidates))
    workload_df["elapsed_seconds_total_walk_forward"] = round(perf_counter() - start_time, 3)
    workload_df["cached_design"] = "entry_variants_built_once_then_chronological_folds_tested"

    return summary_df, fold_results_df, fold_table_df, workload_df, preview_df
