"""Batch validation for cached trend-channel feature refinements.

This module validates feature-refinement rules across multiple saved entry
research caches without rebuilding the strategy pipeline. It is designed for
cross-market checks such as Step Index + Volatility 10 Index.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from mdp.backtesting.trend_channel_cache_feature_refinement_validation import (
    validate_feature_refinements_from_cache,
)


@dataclass(frozen=True)
class FeatureBatchValidationConfig:
    """Configuration for validating feature refinements across saved caches."""

    cache_paths: list[str]
    fold_count: int = 5
    rule_names: list[str] | None = None
    min_total_trades: int = 20
    min_trades_per_fold: int = 3


@dataclass(frozen=True)
class FeatureBatchValidationResult:
    """Container for cross-cache feature-refinement validation outputs."""

    workload: pd.DataFrame
    aggregate_summary: pd.DataFrame
    per_cache_summary: pd.DataFrame
    fold_detail: pd.DataFrame
    selected_preview: pd.DataFrame


def _split_cache_paths(cache_paths: str | Iterable[str]) -> list[str]:
    """Normalize comma-separated or iterable cache paths into a clean list."""
    if isinstance(cache_paths, str):
        return [item.strip() for item in cache_paths.split(",") if item.strip()]
    return [str(item).strip() for item in cache_paths if str(item).strip()]


def _cache_label(path: str | Path) -> str:
    """Create a short readable label for each cache file."""
    return Path(path).stem


def _aggregate_state(row: pd.Series, min_total_trades: int) -> str:
    """Return an honest cross-cache state for one feature rule."""
    if int(row.get("total_kept_trades", 0)) < min_total_trades:
        return "insufficient_total_trades_do_not_lock"
    if float(row.get("mean_average_managed_r", 0.0)) <= 0 or float(row.get("mean_median_managed_r", 0.0)) <= 0:
        return "not_profitable_do_not_lock"
    if float(row.get("cache_positive_average_r_rate", 0.0)) < 1.0:
        return "positive_but_cache_instability_validate_harder"
    if float(row.get("min_cache_average_r", 0.0)) < -0.10:
        return "positive_but_negative_cache_validate_harder"
    if float(row.get("mean_folds_with_min_trades_rate", 0.0)) < 0.80:
        return "positive_but_fold_sample_weak_validate_harder"
    return "candidate_ready_for_larger_multi_market_validation"


def _aggregate_score(row: pd.Series) -> float:
    """Score feature rules while rewarding cross-cache stability."""
    return float(
        (float(row.get("mean_average_managed_r", 0.0)) * 100.0)
        + (float(row.get("mean_median_managed_r", 0.0)) * 35.0)
        + (float(row.get("mean_positive_exit_rate", 0.0)) * 20.0)
        + (float(row.get("cache_positive_average_r_rate", 0.0)) * 15.0)
        + (float(row.get("mean_folds_with_min_trades_rate", 0.0)) * 8.0)
        - (max(0.0, -float(row.get("min_cache_average_r", 0.0))) * 35.0)
    )


def validate_feature_refinements_across_caches(
    config: FeatureBatchValidationConfig,
) -> FeatureBatchValidationResult:
    """Validate feature-refinement rules across multiple saved cache files.

    Student-style note: this function reuses cache CSVs only. It does not call
    MT5, MySQL, or the full trend-channel strategy pipeline.
    """
    import time

    start = time.perf_counter()
    cache_paths = _split_cache_paths(config.cache_paths)
    if not cache_paths:
        raise ValueError("At least one cache path is required.")

    workload_rows: list[pd.DataFrame] = []
    per_cache_rows: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []
    preview_frames: list[pd.DataFrame] = []

    for cache_path in cache_paths:
        result = validate_feature_refinements_from_cache(
            cache_path=cache_path,
            fold_count=config.fold_count,
            rule_names=config.rule_names,
            min_total_trades=config.min_total_trades,
            min_trades_per_fold=config.min_trades_per_fold,
        )

        label = _cache_label(cache_path)

        workload = result.workload.copy()
        workload.insert(0, "cache_label", label)
        workload["cache_path"] = str(cache_path)
        workload_rows.append(workload)

        summary = result.ranking.copy()
        summary.insert(0, "cache_label", label)
        summary["cache_path"] = str(cache_path)
        per_cache_rows.append(summary)

        fold_detail = result.fold_detail.copy()
        if not fold_detail.empty:
            fold_detail.insert(0, "cache_label", label)
            fold_detail["cache_path"] = str(cache_path)
            fold_frames.append(fold_detail)

        preview = result.selected_preview.copy()
        if not preview.empty:
            preview.insert(0, "cache_label", label)
            preview["cache_path"] = str(cache_path)
            preview_frames.append(preview)

    per_cache_summary = pd.concat(per_cache_rows, ignore_index=True) if per_cache_rows else pd.DataFrame()
    fold_detail = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    selected_preview = pd.concat(preview_frames, ignore_index=True) if preview_frames else pd.DataFrame()

    aggregate_rows: list[dict[str, object]] = []
    if not per_cache_summary.empty:
        for rule_name, group in per_cache_summary.groupby("rule_name", dropna=False):
            aggregate_rows.append({
                "rule_name": rule_name,
                "cache_count": int(group["cache_label"].nunique()),
                "total_kept_trades": int(group["kept_trades"].sum()),
                "mean_kept_trades": float(group["kept_trades"].mean()),
                "mean_positive_exit_rate": float(group["positive_exit_rate"].mean()),
                "mean_loss_exit_rate": float(group["loss_exit_rate"].mean()),
                "mean_average_managed_r": float(group["average_managed_r"].mean()),
                "median_average_managed_r": float(group["average_managed_r"].median()),
                "min_cache_average_r": float(group["average_managed_r"].min()),
                "max_cache_average_r": float(group["average_managed_r"].max()),
                "mean_median_managed_r": float(group["median_managed_r"].mean()),
                "cache_positive_average_r_rate": float((group["average_managed_r"] > 0).mean()),
                "cache_positive_median_r_rate": float((group["median_managed_r"] > 0).mean()),
                "mean_positive_average_r_fold_rate": float(group.get("positive_average_r_fold_rate", pd.Series(dtype=float)).mean()) if "positive_average_r_fold_rate" in group else 0.0,
                "mean_folds_with_min_trades_rate": float(group.get("folds_with_min_trades_rate", pd.Series(dtype=float)).mean()) if "folds_with_min_trades_rate" in group else 0.0,
            })

    aggregate_summary = pd.DataFrame(aggregate_rows)
    if not aggregate_summary.empty:
        aggregate_summary["aggregate_state"] = aggregate_summary.apply(
            lambda row: _aggregate_state(row, min_total_trades=config.min_total_trades), axis=1
        )
        aggregate_summary["aggregate_score"] = aggregate_summary.apply(_aggregate_score, axis=1)
        aggregate_summary = aggregate_summary.sort_values(
            by=["aggregate_state", "aggregate_score", "mean_average_managed_r"],
            ascending=[True, False, False],
        ).reset_index(drop=True)

    workload = pd.concat(workload_rows, ignore_index=True) if workload_rows else pd.DataFrame()
    elapsed = time.perf_counter() - start
    batch_workload = pd.DataFrame([{
        "cache_file_count": len(cache_paths),
        "rule_count": len(config.rule_names or []),
        "fold_count": config.fold_count,
        "min_total_trades": config.min_total_trades,
        "min_trades_per_fold": config.min_trades_per_fold,
        "cache_design": "validate_feature_rules_across_saved_caches_without_rebuilding_pipeline",
        "elapsed_seconds_total": round(elapsed, 3),
    }])

    workload = pd.concat([batch_workload, workload], ignore_index=True, sort=False)

    return FeatureBatchValidationResult(
        workload=workload,
        aggregate_summary=aggregate_summary,
        per_cache_summary=per_cache_summary,
        fold_detail=fold_detail,
        selected_preview=selected_preview,
    )
