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
class CandidateFailureDiagnosticsConfig:
    """
    Stores settings for diagnosing why a simple trend-channel candidate fails in specific folds.
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
    candidate_name: str = "hybrid_retest_now_continuation_confirm2"
    active_only: bool = True
    quick: bool = False


def _safe_mean(series: pd.Series) -> float:
    """
    Returns a safe numeric mean for compact diagnostics.
    """
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else 0.0


def _safe_median(series: pd.Series) -> float:
    """
    Returns a safe numeric median for compact diagnostics.
    """
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.median()) if not numeric.empty else 0.0


def _build_entry_dataset(config: CandidateFailureDiagnosticsConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds the expensive cutoff-aware entry dataset once, then reuses it for all diagnostics.
    """
    # Quick mode keeps laptop runs small while preserving the same logic path.
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
        min_trades_for_candidate=1,
        active_only=config.active_only,
    )
    return build_entry_refinement_dataset(entry_config)


def _assign_folds(candidate_df: pd.DataFrame, fold_table_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds fold information to every selected candidate trade.
    """
    if candidate_df.empty or fold_table_df.empty:
        return candidate_df.copy()

    result_df = candidate_df.copy()
    result_df["analysis_cutoff"] = pd.to_datetime(result_df["analysis_cutoff"])
    result_df["fold_id"] = pd.NA
    result_df["fold_position"] = pd.NA

    # Assign every trade to the chronological fold that owns its cutoff timestamp.
    for _, fold_row in fold_table_df.iterrows():
        start_time = pd.to_datetime(fold_row["fold_start_cutoff"])
        end_time = pd.to_datetime(fold_row["fold_end_cutoff"])
        mask = (result_df["analysis_cutoff"] >= start_time) & (result_df["analysis_cutoff"] <= end_time)
        result_df.loc[mask, "fold_id"] = int(fold_row["fold_id"])
        result_df.loc[mask, "fold_position"] = fold_row["fold_position"]

    return result_df


def _add_result_columns(candidate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds simple win/loss and risk diagnostic columns.
    """
    if candidate_df.empty:
        return candidate_df.copy()

    result_df = candidate_df.copy()
    managed_r = pd.to_numeric(result_df.get("managed_realized_r"), errors="coerce")
    result_df["result_bucket"] = "flat_or_missing"
    result_df.loc[managed_r > 0, "result_bucket"] = "positive_exit"
    result_df.loc[managed_r < 0, "result_bucket"] = "negative_exit"
    result_df["is_positive_exit"] = managed_r > 0
    result_df["is_negative_exit"] = managed_r < 0

    # A full-loss marker helps us separate small managed losses from true stop damage.
    result_df["is_full_or_near_full_loss"] = managed_r <= -0.90

    # Build compact keys that are easy to inspect in terminal output.
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


def summarize_fold_performance(candidate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes candidate performance by chronological fold.
    """
    if candidate_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for (fold_id, fold_position), group_df in candidate_df.groupby(["fold_id", "fold_position"], dropna=False):
        managed_r = pd.to_numeric(group_df.get("managed_realized_r"), errors="coerce").dropna()
        rows.append(
            {
                "fold_id": fold_id,
                "fold_position": fold_position,
                "trades": int(len(group_df)),
                "positive_exit_rate": float((managed_r > 0).mean()) if not managed_r.empty else 0.0,
                "loss_exit_rate": float((managed_r < 0).mean()) if not managed_r.empty else 0.0,
                "average_managed_r": float(managed_r.mean()) if not managed_r.empty else 0.0,
                "median_managed_r": float(managed_r.median()) if not managed_r.empty else 0.0,
                "total_managed_r": float(managed_r.sum()) if not managed_r.empty else 0.0,
                "full_or_near_full_loss_count": int(group_df.get("is_full_or_near_full_loss", pd.Series(dtype="bool")).fillna(False).sum()),
            }
        )

    return pd.DataFrame(rows).sort_values("fold_id").reset_index(drop=True)


def summarize_failure_features(candidate_df: pd.DataFrame, min_group_trades: int = 1) -> pd.DataFrame:
    """
    Groups selected trades by key features to find what is concentrated in losing folds.
    """
    if candidate_df.empty:
        return pd.DataFrame()

    feature_columns = [
        "fold_position",
        "scenario_state",
        "event_state",
        "entry_mode",
        "entry_variant_name",
        "target_plan_state",
        "risk_ratio_bucket",
        "best_rr_bucket",
        "available_targets",
        "managed_exit_state",
        "scenario_event_key",
        "entry_target_key",
    ]

    rows: list[dict[str, Any]] = []
    for feature_name in feature_columns:
        if feature_name not in candidate_df.columns:
            continue

        for feature_value, group_df in candidate_df.groupby(feature_name, dropna=False):
            managed_r = pd.to_numeric(group_df.get("managed_realized_r"), errors="coerce").dropna()
            trades = int(len(group_df))
            if trades < min_group_trades:
                continue

            average_r = float(managed_r.mean()) if not managed_r.empty else 0.0
            loss_rate = float((managed_r < 0).mean()) if not managed_r.empty else 0.0
            full_loss_count = int(group_df.get("is_full_or_near_full_loss", pd.Series(dtype="bool")).fillna(False).sum())

            if average_r < 0 and full_loss_count > 0:
                diagnostic_note = "possible_failure_driver"
            elif average_r > 0 and loss_rate <= 0.35:
                diagnostic_note = "possible_keep_context"
            else:
                diagnostic_note = "neutral_or_small_sample"

            rows.append(
                {
                    "feature_name": feature_name,
                    "feature_value": feature_value,
                    "trades": trades,
                    "positive_exit_rate": float((managed_r > 0).mean()) if not managed_r.empty else 0.0,
                    "loss_exit_rate": loss_rate,
                    "average_managed_r": average_r,
                    "median_managed_r": float(managed_r.median()) if not managed_r.empty else 0.0,
                    "total_managed_r": float(managed_r.sum()) if not managed_r.empty else 0.0,
                    "full_or_near_full_loss_count": full_loss_count,
                    "diagnostic_note": diagnostic_note,
                }
            )

    if not rows:
        return pd.DataFrame()

    # Worst groups come first so the terminal shows the likely failure drivers immediately.
    return pd.DataFrame(rows).sort_values(
        ["diagnostic_note", "average_managed_r", "trades"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def build_losing_trade_preview(candidate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the losing selected trades with the most useful context columns.
    """
    if candidate_df.empty:
        return pd.DataFrame()

    managed_r = pd.to_numeric(candidate_df.get("managed_realized_r"), errors="coerce")
    losing_df = candidate_df[managed_r < 0].copy()
    preview_columns = [
        "fold_id",
        "fold_position",
        "analysis_cutoff",
        "entry_variant_name",
        "scenario_state",
        "event_state",
        "entry_mode",
        "entry_price",
        "risk_points",
        "available_targets",
        "target_plan_state",
        "risk_ratio_bucket",
        "best_rr_bucket",
        "managed_exit_state",
        "managed_realized_r",
    ]
    existing_columns = [column for column in preview_columns if column in losing_df.columns]
    return losing_df[existing_columns].sort_values("managed_realized_r").reset_index(drop=True) if existing_columns else losing_df.reset_index(drop=True)


def run_candidate_failure_diagnostics(
    config: CandidateFailureDiagnosticsConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Runs fold-level and feature-level diagnostics for one candidate strategy.
    """
    start_time = perf_counter()

    # Build once, then select and diagnose the candidate without extra strategy replays.
    dataset_df, workload_df = _build_entry_dataset(config)
    fold_table_df = build_chronological_fold_table(dataset_df, config.fold_count)
    selected_df = select_candidate_rows(dataset_df, config.candidate_name)
    selected_df = _assign_folds(selected_df, fold_table_df)
    selected_df = _add_result_columns(selected_df)

    fold_summary_df = summarize_fold_performance(selected_df)
    feature_summary_df = summarize_failure_features(selected_df)
    losing_preview_df = build_losing_trade_preview(selected_df)

    workload_df = workload_df.copy()
    workload_df["candidate_name"] = config.candidate_name
    workload_df["selected_candidate_trades"] = int(len(selected_df))
    workload_df["fold_count"] = int(len(fold_table_df))
    workload_df["elapsed_seconds_total_failure_diagnostics"] = round(perf_counter() - start_time, 3)
    workload_df["cached_design"] = "entry_dataset_built_once_then_candidate_failure_diagnostics"

    return workload_df, fold_table_df, fold_summary_df, feature_summary_df, losing_preview_df
