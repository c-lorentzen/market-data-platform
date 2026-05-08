"""Diagnostics for fold instability in cached trend-channel research datasets.

This module intentionally works from a saved entry research cache so we can
inspect candidate weaknesses without rebuilding the expensive strategy pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FoldInstabilityDiagnosticsResult:
    """Container for diagnostic tables produced from a cached research file."""

    selected_trades: pd.DataFrame
    fold_summary: pd.DataFrame
    weak_fold_trades: pd.DataFrame
    feature_diagnostics: pd.DataFrame
    candidate_avoid_contexts: pd.DataFrame
    candidate_keep_contexts: pd.DataFrame
    workload_summary: pd.DataFrame


def load_cache(cache_path: str | Path) -> pd.DataFrame:
    """Load a saved trend-channel entry research cache."""
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"Cache file not found: {path}")

    df = pd.read_csv(path)

    # Convert dates when available so chronological folds sort correctly.
    if "analysis_cutoff" in df.columns:
        df["analysis_cutoff"] = pd.to_datetime(df["analysis_cutoff"], errors="coerce")

    return df


def _safe_contains(series: pd.Series, text: str) -> pd.Series:
    """Case-insensitive contains helper that treats missing values as False."""
    return series.astype(str).str.contains(text, case=False, na=False)


def apply_refinement_rule(df: pd.DataFrame, rule_name: str) -> pd.DataFrame:
    """Select trades matching one simple candidate/refinement rule.

    The rules mirror the research-cache validation logic in a lightweight way.
    They are kept explicit so future rules can be reviewed easily.
    """
    required = {"entry_mode", "entry_variant_name", "target_plan_state", "available_targets"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Cache is missing required columns for rule selection: {sorted(missing)}")

    is_retest = df["entry_mode"].eq("retest_rejection_close")
    is_continuation = df["entry_mode"].eq("continuation_after_retest_close")
    is_baseline_entry = df["entry_variant_name"].eq("baseline_current_close")
    is_confirm2_entry = df["entry_variant_name"].eq("confirm_2_directional_close")
    has_tp2_tp3 = df["target_plan_state"].eq("tp2_tp3_available") | (
        _safe_contains(df["available_targets"], "TP2") & _safe_contains(df["available_targets"], "TP3")
    )

    hybrid_mask = (is_retest & is_baseline_entry) | (is_continuation & is_confirm2_entry)

    if rule_name == "hybrid_reference":
        mask = hybrid_mask
    elif rule_name == "retest_requires_tp2_tp3":
        # Retest entries need stronger target availability; continuation entries keep confirm2.
        mask = (is_retest & is_baseline_entry & has_tp2_tp3) | (is_continuation & is_confirm2_entry)
    elif rule_name == "avoid_context_retest_tp2_only":
        scenario = df.get("scenario_state", pd.Series("", index=df.index)).astype(str)
        is_weak_context = scenario.eq("bearish_context_alignment") & is_retest & df["target_plan_state"].eq("tp2_only")
        mask = hybrid_mask & ~is_weak_context
    elif rule_name == "avoid_all_tp2_only":
        mask = hybrid_mask & ~df["target_plan_state"].eq("tp2_only")
    elif rule_name == "require_tp2_tp3_all":
        mask = hybrid_mask & has_tp2_tp3
    elif rule_name == "avoid_bearish_context_alignment":
        scenario = df.get("scenario_state", pd.Series("", index=df.index)).astype(str)
        mask = hybrid_mask & ~scenario.eq("bearish_context_alignment")
    elif rule_name == "keep_pullback_or_tp2_tp3":
        scenario = df.get("scenario_state", pd.Series("", index=df.index)).astype(str)
        is_pullback = _safe_contains(scenario, "pullback")
        mask = hybrid_mask & (is_pullback | has_tp2_tp3)
    else:
        raise ValueError(
            "Unknown rule_name. Supported: hybrid_reference, retest_requires_tp2_tp3, "
            "avoid_context_retest_tp2_only, avoid_all_tp2_only, require_tp2_tp3_all, "
            "avoid_bearish_context_alignment, keep_pullback_or_tp2_tp3"
        )

    selected = df.loc[mask].copy()
    selected = selected.sort_values("analysis_cutoff").reset_index(drop=True)
    return selected


def assign_chronological_folds(selected: pd.DataFrame, fold_count: int) -> pd.DataFrame:
    """Assign selected trades into chronological folds with roughly equal sizes."""
    if selected.empty:
        return selected.assign(fold_id=pd.Series(dtype=int), fold_position=pd.Series(dtype=str))

    fold_count = max(1, int(fold_count))
    fold_ids = np.array_split(np.arange(len(selected)), min(fold_count, len(selected)))

    selected = selected.copy()
    selected["fold_id"] = np.nan
    for idx, fold_index_array in enumerate(fold_ids, start=1):
        selected.loc[fold_index_array, "fold_id"] = idx

    selected["fold_id"] = selected["fold_id"].astype(int)

    # Label folds for easier reading.
    max_fold = selected["fold_id"].max()
    def _position(fid: int) -> str:
        if fid == 1:
            return "earliest"
        if fid == max_fold:
            return "latest"
        return "middle"

    selected["fold_position"] = selected["fold_id"].map(_position)
    return selected


def summarize_folds(selected: pd.DataFrame) -> pd.DataFrame:
    """Build fold-level performance summary."""
    if selected.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for fold_id, group in selected.groupby("fold_id", sort=True):
        realized = pd.to_numeric(group["managed_realized_r"], errors="coerce").dropna()
        losses = realized < 0
        rows.append(
            {
                "fold_id": int(fold_id),
                "fold_position": group["fold_position"].iloc[0],
                "fold_start_cutoff": group["analysis_cutoff"].min(),
                "fold_end_cutoff": group["analysis_cutoff"].max(),
                "kept_trades": int(len(realized)),
                "positive_exit_rate": float((realized > 0).mean()) if len(realized) else 0.0,
                "loss_exit_rate": float(losses.mean()) if len(realized) else 0.0,
                "average_managed_r": float(realized.mean()) if len(realized) else 0.0,
                "median_managed_r": float(realized.median()) if len(realized) else 0.0,
                "total_managed_r": float(realized.sum()) if len(realized) else 0.0,
                "full_or_near_full_loss_count": int((realized <= -0.75).sum()),
            }
        )

    fold_summary = pd.DataFrame(rows)
    fold_summary["fold_state"] = np.where(
        fold_summary["average_managed_r"] > 0,
        "positive_fold",
        "weak_or_negative_fold",
    )
    return fold_summary


def _summarize_group(group: pd.DataFrame) -> dict:
    """Summarize one feature/context group."""
    realized = pd.to_numeric(group["managed_realized_r"], errors="coerce").dropna()
    return {
        "trades": int(len(realized)),
        "positive_exit_rate": float((realized > 0).mean()) if len(realized) else 0.0,
        "loss_exit_rate": float((realized < 0).mean()) if len(realized) else 0.0,
        "average_managed_r": float(realized.mean()) if len(realized) else 0.0,
        "median_managed_r": float(realized.median()) if len(realized) else 0.0,
        "total_managed_r": float(realized.sum()) if len(realized) else 0.0,
        "full_or_near_full_loss_count": int((realized <= -0.75).sum()) if len(realized) else 0,
    }


def build_feature_diagnostics(
    selected: pd.DataFrame,
    feature_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Summarize performance by feature/context columns."""
    if selected.empty:
        return pd.DataFrame()

    default_features = [
        "fold_position",
        "scenario_state",
        "event_state",
        "entry_mode",
        "entry_variant_name",
        "target_plan_state",
        "available_targets",
        "managed_exit_state",
        "risk_ratio_bucket",
        "best_rr_bucket",
    ]
    features = list(feature_columns or default_features)
    features = [column for column in features if column in selected.columns]

    rows: list[dict] = []
    for feature in features:
        for value, group in selected.groupby(feature, dropna=False):
            stats = _summarize_group(group)
            stats.update({"feature_name": feature, "feature_value": value})
            rows.append(stats)

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    # Rank weak contexts first, then strong contexts.
    result["diagnostic_state"] = np.select(
        [
            (result["trades"] >= 2) & (result["average_managed_r"] < 0),
            (result["trades"] >= 2) & (result["average_managed_r"] > 0),
        ],
        ["possible_avoid_context", "possible_keep_context"],
        default="small_sample_context",
    )
    result = result.sort_values(
        ["diagnostic_state", "average_managed_r", "trades"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    return result


def diagnose_fold_instability(
    cache_path: str | Path,
    rule_name: str = "retest_requires_tp2_tp3",
    fold_count: int = 5,
) -> FoldInstabilityDiagnosticsResult:
    """Run fold-instability diagnostics from a saved research cache."""
    cache_df = load_cache(cache_path)
    selected = apply_refinement_rule(cache_df, rule_name)
    selected = assign_chronological_folds(selected, fold_count=fold_count)
    fold_summary = summarize_folds(selected)

    weak_fold_ids = []
    if not fold_summary.empty:
        weak_fold_ids = fold_summary.loc[fold_summary["average_managed_r"] <= 0, "fold_id"].tolist()

    weak_fold_trades = selected.loc[selected["fold_id"].isin(weak_fold_ids)].copy() if weak_fold_ids else pd.DataFrame()
    feature_diagnostics = build_feature_diagnostics(selected)
    candidate_avoid_contexts = feature_diagnostics.loc[
        feature_diagnostics.get("diagnostic_state", pd.Series(dtype=str)).eq("possible_avoid_context")
    ].copy()
    candidate_keep_contexts = feature_diagnostics.loc[
        feature_diagnostics.get("diagnostic_state", pd.Series(dtype=str)).eq("possible_keep_context")
    ].copy()

    workload_summary = pd.DataFrame(
        [
            {
                "cache_path": str(cache_path),
                "cache_rows": int(len(cache_df)),
                "unique_cutoffs": int(cache_df["analysis_cutoff"].nunique()) if "analysis_cutoff" in cache_df else 0,
                "rule_name": rule_name,
                "selected_trades": int(len(selected)),
                "fold_count_requested": int(fold_count),
                "fold_count_used": int(selected["fold_id"].nunique()) if "fold_id" in selected else 0,
                "weak_fold_count": int(len(weak_fold_ids)),
                "cache_design": "load_saved_cache_once_then_diagnose_fold_instability",
            }
        ]
    )

    return FoldInstabilityDiagnosticsResult(
        selected_trades=selected,
        fold_summary=fold_summary,
        weak_fold_trades=weak_fold_trades,
        feature_diagnostics=feature_diagnostics,
        candidate_avoid_contexts=candidate_avoid_contexts,
        candidate_keep_contexts=candidate_keep_contexts,
        workload_summary=workload_summary,
    )
