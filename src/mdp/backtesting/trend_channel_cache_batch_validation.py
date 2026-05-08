"""Batch validation for cached trend-channel refinement rules.

This module validates simple refinement rules across one or more saved
entry-research cache CSV files. It is designed to avoid rebuilding the
full strategy pipeline when we only need to compare candidate rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BatchValidationConfig:
    """Configuration for validating cached strategy refinements."""

    cache_paths: list[str]
    fold_count: int = 4
    min_total_trades: int = 20
    min_trades_per_fold: int = 3
    rule_names: list[str] | None = None


DEFAULT_RULE_NAMES = [
    "hybrid_reference",
    "retest_requires_tp2_tp3",
    "avoid_context_retest_tp2_only",
    "avoid_bearish_context_alignment",
]


def expand_cache_paths(cache_paths: Iterable[str]) -> list[str]:
    """Expand file paths and glob patterns into concrete cache CSV paths."""

    expanded_paths: list[str] = []

    for raw_path in cache_paths:
        # Let the user pass several paths separated by commas from the CLI.
        for path_part in str(raw_path).split(","):
            path_part = path_part.strip().strip('"')
            if not path_part:
                continue

            # Glob patterns make it easy to validate every cache in a folder.
            matches = sorted(str(path) for path in Path().glob(path_part))
            if matches:
                expanded_paths.extend(matches)
            else:
                expanded_paths.append(path_part)

    # Remove duplicates while preserving order.
    unique_paths = list(dict.fromkeys(expanded_paths))
    return unique_paths


def load_cache_dataset(cache_path: str) -> pd.DataFrame:
    """Load one saved entry-research cache CSV."""

    path = Path(cache_path)

    if not path.exists():
        raise FileNotFoundError(f"Cache file does not exist: {cache_path}")

    dataset_df = pd.read_csv(path)

    # Convert cutoff time to datetime so chronological folds are reliable.
    if "analysis_cutoff" in dataset_df.columns:
        dataset_df["analysis_cutoff"] = pd.to_datetime(dataset_df["analysis_cutoff"])

    # Keep a source column so later summaries can identify where rows came from.
    dataset_df["cache_source"] = str(path)
    return dataset_df


def _series_contains(series: pd.Series, text: str) -> pd.Series:
    """Return a safe boolean mask for string containment checks."""

    return series.fillna("").astype(str).str.contains(text, case=False, regex=False)


def _build_hybrid_reference_mask(dataset_df: pd.DataFrame) -> pd.Series:
    """Select the current hybrid candidate from all cached entry variants."""

    # Retest-rejection entries use the immediate/current close.
    retest_now = (
        (dataset_df["entry_mode"] == "retest_rejection_close")
        & (dataset_df["entry_variant_name"] == "baseline_current_close")
    )

    # Continuation entries require two-candle directional confirmation.
    continuation_confirm2 = (
        (dataset_df["entry_mode"] == "continuation_after_retest_close")
        & (dataset_df["entry_variant_name"] == "confirm_2_directional_close")
    )

    # Only actual entered trades should be considered.
    entered = dataset_df.get("entered_trade", True)
    if not isinstance(entered, pd.Series):
        entered = pd.Series(True, index=dataset_df.index)

    return (retest_now | continuation_confirm2) & entered.fillna(False).astype(bool)


def apply_refinement_rule(dataset_df: pd.DataFrame, rule_name: str) -> pd.DataFrame:
    """Apply one refinement rule and return the selected trades."""

    hybrid_mask = _build_hybrid_reference_mask(dataset_df)
    selected_df = dataset_df.loc[hybrid_mask].copy()

    if selected_df.empty:
        return selected_df

    if rule_name == "hybrid_reference":
        return selected_df

    if rule_name == "retest_requires_tp2_tp3":
        # Retest entries must have TP2 and TP3 available.
        retest_rows = selected_df["entry_mode"] == "retest_rejection_close"
        has_tp2_tp3 = _series_contains(selected_df["available_targets"], "TP2,TP3")
        keep_mask = (~retest_rows) | has_tp2_tp3
        return selected_df.loc[keep_mask].copy()

    if rule_name == "avoid_context_retest_tp2_only":
        # Target the specific weak cluster found in failure diagnostics.
        weak_context = (
            (selected_df["scenario_state"] == "bearish_context_alignment")
            & (selected_df["entry_mode"] == "retest_rejection_close")
            & (selected_df["target_plan_state"] == "tp2_only")
        )
        return selected_df.loc[~weak_context].copy()

    if rule_name == "avoid_bearish_context_alignment":
        # Keep non-bearish-context-alignment selected hybrid trades.
        keep_mask = selected_df["scenario_state"] != "bearish_context_alignment"
        return selected_df.loc[keep_mask].copy()

    raise ValueError(f"Unknown refinement rule: {rule_name}")


def _safe_mean(series: pd.Series) -> float:
    """Return a clean float mean with zero for empty data."""

    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
    if numeric_series.empty:
        return 0.0
    return float(numeric_series.mean())


def _safe_median(series: pd.Series) -> float:
    """Return a clean float median with zero for empty data."""

    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
    if numeric_series.empty:
        return 0.0
    return float(numeric_series.median())


def summarize_rule_trades(
    trades_df: pd.DataFrame,
    *,
    baseline_attempts: int,
    rule_name: str,
    fold_count: int,
    min_total_trades: int,
    min_trades_per_fold: int,
) -> tuple[dict, pd.DataFrame]:
    """Summarize a selected set of rule trades and its chronological folds."""

    if trades_df.empty:
        empty_summary = {
            "rule_name": rule_name,
            "baseline_attempts": baseline_attempts,
            "kept_trades": 0,
            "coverage_rate": 0.0,
            "positive_exit_rate": 0.0,
            "loss_exit_rate": 0.0,
            "average_managed_r": 0.0,
            "median_managed_r": 0.0,
            "total_managed_r": 0.0,
            "min_fold_average_r": 0.0,
            "positive_average_r_fold_rate": 0.0,
            "positive_median_r_fold_rate": 0.0,
            "folds_with_min_trades_rate": 0.0,
            "validation_state": "reject_no_trades_selected",
        }
        return empty_summary, pd.DataFrame()

    trades_df = trades_df.sort_values("analysis_cutoff").reset_index(drop=True)
    realized_r = pd.to_numeric(trades_df["managed_realized_r"], errors="coerce").fillna(0.0)

    kept_trades = int(len(trades_df))
    positive_exit_rate = float((realized_r > 0).mean())
    loss_exit_rate = float((realized_r < 0).mean())
    average_managed_r = float(realized_r.mean())
    median_managed_r = float(realized_r.median())
    total_managed_r = float(realized_r.sum())

    # Split rows into chronological folds using row order after sorting.
    fold_ids = np.array_split(np.arange(kept_trades), max(1, fold_count))
    fold_rows: list[dict] = []

    for fold_index, row_indexes in enumerate(fold_ids, start=1):
        if len(row_indexes) == 0:
            continue

        fold_df = trades_df.iloc[row_indexes]
        fold_r = pd.to_numeric(fold_df["managed_realized_r"], errors="coerce").fillna(0.0)

        fold_rows.append(
            {
                "rule_name": rule_name,
                "fold_id": fold_index,
                "fold_start_cutoff": fold_df["analysis_cutoff"].min(),
                "fold_end_cutoff": fold_df["analysis_cutoff"].max(),
                "kept_trades": int(len(fold_df)),
                "positive_exit_rate": float((fold_r > 0).mean()),
                "loss_exit_rate": float((fold_r < 0).mean()),
                "average_managed_r": float(fold_r.mean()),
                "median_managed_r": float(fold_r.median()),
                "total_managed_r": float(fold_r.sum()),
            }
        )

    fold_df = pd.DataFrame(fold_rows)

    if fold_df.empty:
        min_fold_average_r = 0.0
        positive_average_r_fold_rate = 0.0
        positive_median_r_fold_rate = 0.0
        folds_with_min_trades_rate = 0.0
    else:
        min_fold_average_r = float(fold_df["average_managed_r"].min())
        positive_average_r_fold_rate = float((fold_df["average_managed_r"] > 0).mean())
        positive_median_r_fold_rate = float((fold_df["median_managed_r"] > 0).mean())
        folds_with_min_trades_rate = float((fold_df["kept_trades"] >= min_trades_per_fold).mean())

    if kept_trades < min_total_trades:
        validation_state = "insufficient_total_trades_do_not_lock"
    elif average_managed_r <= 0 or median_managed_r <= 0:
        validation_state = "not_profitable_do_not_lock"
    elif min_fold_average_r < 0:
        validation_state = "positive_but_fold_instability_validate_harder"
    elif positive_average_r_fold_rate < 0.75:
        validation_state = "weak_fold_consistency_validate_harder"
    else:
        validation_state = "candidate_ready_for_larger_out_of_sample_validation"

    summary = {
        "rule_name": rule_name,
        "baseline_attempts": baseline_attempts,
        "kept_trades": kept_trades,
        "coverage_rate": float(kept_trades / baseline_attempts) if baseline_attempts else 0.0,
        "positive_exit_rate": positive_exit_rate,
        "loss_exit_rate": loss_exit_rate,
        "average_managed_r": average_managed_r,
        "median_managed_r": median_managed_r,
        "total_managed_r": total_managed_r,
        "min_fold_average_r": min_fold_average_r,
        "positive_average_r_fold_rate": positive_average_r_fold_rate,
        "positive_median_r_fold_rate": positive_median_r_fold_rate,
        "folds_with_min_trades_rate": folds_with_min_trades_rate,
        "validation_state": validation_state,
    }

    return summary, fold_df


def validate_cache_file(
    cache_path: str,
    *,
    rule_names: list[str],
    fold_count: int,
    min_total_trades: int,
    min_trades_per_fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate refinement rules for one cache CSV."""

    dataset_df = load_cache_dataset(cache_path)
    baseline_attempts = int(_build_hybrid_reference_mask(dataset_df).sum())

    summaries: list[dict] = []
    fold_details: list[pd.DataFrame] = []

    for rule_name in rule_names:
        selected_df = apply_refinement_rule(dataset_df, rule_name)
        summary, folds_df = summarize_rule_trades(
            selected_df,
            baseline_attempts=baseline_attempts,
            rule_name=rule_name,
            fold_count=fold_count,
            min_total_trades=min_total_trades,
            min_trades_per_fold=min_trades_per_fold,
        )

        summary["cache_path"] = cache_path
        summary["cache_rows"] = int(len(dataset_df))
        summary["unique_cutoffs"] = int(dataset_df["analysis_cutoff"].nunique())
        summaries.append(summary)

        if not folds_df.empty:
            folds_df["cache_path"] = cache_path
            fold_details.append(folds_df)

    summary_df = pd.DataFrame(summaries)
    fold_detail_df = pd.concat(fold_details, ignore_index=True) if fold_details else pd.DataFrame()

    return summary_df, fold_detail_df


def aggregate_cache_summaries(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate rule performance across multiple cache files."""

    if summary_df.empty:
        return pd.DataFrame()

    grouped = summary_df.groupby("rule_name", dropna=False)

    aggregate_df = grouped.agg(
        cache_count=("cache_path", "nunique"),
        total_kept_trades=("kept_trades", "sum"),
        mean_kept_trades=("kept_trades", "mean"),
        mean_positive_exit_rate=("positive_exit_rate", "mean"),
        mean_average_managed_r=("average_managed_r", "mean"),
        median_average_managed_r=("average_managed_r", "median"),
        min_cache_average_r=("average_managed_r", "min"),
        max_cache_average_r=("average_managed_r", "max"),
        mean_median_managed_r=("median_managed_r", "mean"),
        cache_positive_average_r_rate=("average_managed_r", lambda s: float((s > 0).mean())),
        cache_positive_median_r_rate=("median_managed_r", lambda s: float((s > 0).mean())),
    ).reset_index()

    def classify(row: pd.Series) -> str:
        # Use conservative states so we do not lock weak or tiny samples.
        if row["total_kept_trades"] < 20:
            return "insufficient_total_trades_do_not_lock"
        if row["mean_average_managed_r"] <= 0 or row["mean_median_managed_r"] <= 0:
            return "not_profitable_do_not_lock"
        if row["min_cache_average_r"] < 0:
            return "positive_but_cache_instability_validate_harder"
        if row["cache_positive_average_r_rate"] < 0.75:
            return "weak_cache_consistency_validate_harder"
        return "candidate_ready_for_bigger_market_validation"

    aggregate_df["aggregate_state"] = aggregate_df.apply(classify, axis=1)

    # Higher average R and more trades should rank higher.
    aggregate_df["aggregate_score"] = (
        aggregate_df["mean_average_managed_r"] * 100.0
        + aggregate_df["mean_median_managed_r"] * 50.0
        + aggregate_df["mean_positive_exit_rate"] * 20.0
        + np.minimum(aggregate_df["total_kept_trades"], 50) * 0.25
    )

    return aggregate_df.sort_values("aggregate_score", ascending=False).reset_index(drop=True)


def validate_refinements_across_caches(config: BatchValidationConfig) -> dict[str, pd.DataFrame]:
    """Validate rule performance across one or more cache CSV files."""

    cache_paths = expand_cache_paths(config.cache_paths)
    if not cache_paths:
        raise ValueError("No cache paths were provided.")

    rule_names = config.rule_names or DEFAULT_RULE_NAMES

    all_summaries: list[pd.DataFrame] = []
    all_folds: list[pd.DataFrame] = []

    for cache_path in cache_paths:
        summary_df, fold_df = validate_cache_file(
            cache_path,
            rule_names=rule_names,
            fold_count=config.fold_count,
            min_total_trades=config.min_total_trades,
            min_trades_per_fold=config.min_trades_per_fold,
        )
        all_summaries.append(summary_df)

        if not fold_df.empty:
            all_folds.append(fold_df)

    per_cache_summary_df = pd.concat(all_summaries, ignore_index=True)
    fold_detail_df = pd.concat(all_folds, ignore_index=True) if all_folds else pd.DataFrame()
    aggregate_summary_df = aggregate_cache_summaries(per_cache_summary_df)

    workload_df = pd.DataFrame(
        [
            {
                "cache_file_count": len(cache_paths),
                "rule_count": len(rule_names),
                "fold_count": config.fold_count,
                "min_total_trades": config.min_total_trades,
                "min_trades_per_fold": config.min_trades_per_fold,
                "cache_design": "validate_saved_cache_files_without_rebuilding_strategy_pipeline",
            }
        ]
    )

    return {
        "workload": workload_df,
        "aggregate_summary": aggregate_summary_df,
        "per_cache_summary": per_cache_summary_df,
        "fold_detail": fold_detail_df,
    }
