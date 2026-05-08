"""Feature-edge diagnostics for cached trend-channel research datasets.

This module is intentionally lightweight: it reads an existing entry research
cache and tests whether a selected candidate/refinement has obvious pre-entry
feature clusters that explain wins/losses. It does not rebuild the strategy
pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


DEFAULT_RULE_NAMES = [
    "hybrid_reference",
    "retest_requires_tp2_tp3",
    "avoid_context_retest_tp2_only",
]


@dataclass(frozen=True)
class FeatureEdgeConfig:
    cache_path: str
    rule_name: str = "retest_requires_tp2_tp3"
    fold_count: int = 5
    min_group_trades: int = 3
    weak_r_threshold: float = 0.0
    strong_r_threshold: float = 0.10


def _normalize_string_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return _normalize_string_series(series).str.lower().isin({"true", "1", "yes", "y"})


def _require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Cache is missing required columns: {missing}")


def _select_entered_rows(df: pd.DataFrame) -> pd.DataFrame:
    # Student-style comment: only entered trades can be scored as outcomes.
    if "entered_trade" in df.columns:
        return df[_bool_series(df["entered_trade"])].copy()
    if "entry_status" in df.columns:
        return df[_normalize_string_series(df["entry_status"]).eq("entered")].copy()
    return df[df["managed_realized_r"].notna()].copy()


def _apply_rule(df: pd.DataFrame, rule_name: str) -> pd.DataFrame:
    """Apply the same simple refinement rules used by cached validation."""
    _require_columns(
        df,
        [
            "event_state",
            "entry_variant_name",
            "entry_mode",
            "target_plan_state",
            "available_targets",
            "scenario_state",
        ],
    )

    event_state = _normalize_string_series(df["event_state"])
    entry_variant = _normalize_string_series(df["entry_variant_name"])
    entry_mode = _normalize_string_series(df["entry_mode"])
    target_plan_state = _normalize_string_series(df["target_plan_state"])
    available_targets = _normalize_string_series(df["available_targets"])
    scenario_state = _normalize_string_series(df["scenario_state"])

    # Student-style comment: this is the current hybrid candidate logic.
    retest_now = event_state.str.contains("retest_rejection", case=False, regex=False) & entry_variant.eq(
        "baseline_current_close"
    )
    continuation_confirm2 = event_state.str.contains(
        "continuation_after_retest", case=False, regex=False
    ) & entry_variant.eq("confirm_2_directional_close")
    hybrid = retest_now | continuation_confirm2

    tp2_tp3_available = target_plan_state.eq("tp2_tp3_available") | available_targets.eq("TP2,TP3")
    bearish_context_retest_tp2_only = (
        scenario_state.eq("bearish_context_alignment")
        & entry_mode.eq("retest_rejection_close")
        & target_plan_state.eq("tp2_only")
    )

    if rule_name == "hybrid_reference":
        mask = hybrid
    elif rule_name == "retest_requires_tp2_tp3":
        # Retest entries need stronger target structure; continuation entries stay hybrid-confirmed.
        mask = (retest_now & tp2_tp3_available) | continuation_confirm2
    elif rule_name == "avoid_context_retest_tp2_only":
        mask = hybrid & ~bearish_context_retest_tp2_only
    elif rule_name == "require_tp2_tp3_all":
        mask = hybrid & tp2_tp3_available
    elif rule_name == "avoid_all_tp2_only":
        mask = hybrid & ~target_plan_state.eq("tp2_only")
    elif rule_name == "avoid_bearish_context_alignment":
        mask = hybrid & ~scenario_state.eq("bearish_context_alignment")
    elif rule_name == "keep_pullback_or_tp2_tp3":
        pullback_context = scenario_state.str.contains("pullback", case=False, regex=False)
        mask = hybrid & (pullback_context | tp2_tp3_available)
    else:
        raise ValueError(f"Unsupported rule_name: {rule_name}")

    return df[mask].copy()


def _assign_chronological_folds(df: pd.DataFrame, fold_count: int) -> pd.DataFrame:
    if df.empty:
        df = df.copy()
        df["fold_id"] = []
        df["fold_position"] = []
        return df

    out = df.copy()
    if "analysis_cutoff" in out.columns:
        out["analysis_cutoff"] = pd.to_datetime(out["analysis_cutoff"], errors="coerce")
        out = out.sort_values("analysis_cutoff")
    else:
        out = out.reset_index(drop=True)

    folds = np.array_split(np.arange(len(out)), max(1, fold_count))
    out["fold_id"] = 0
    out["fold_position"] = "middle"
    for idx, fold_indices in enumerate(folds, start=1):
        if len(fold_indices) == 0:
            continue
        out.iloc[fold_indices, out.columns.get_loc("fold_id")] = idx
        if idx == 1:
            position = "earliest"
        elif idx == len(folds):
            position = "latest"
        else:
            position = "middle"
        out.iloc[fold_indices, out.columns.get_loc("fold_position")] = position
    return out


def _summarize_outcomes(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "trades": 0,
            "positive_exit_rate": 0.0,
            "loss_exit_rate": 0.0,
            "average_managed_r": 0.0,
            "median_managed_r": 0.0,
            "total_managed_r": 0.0,
            "full_or_near_full_loss_count": 0,
        }

    r = pd.to_numeric(df["managed_realized_r"], errors="coerce").fillna(0.0)
    trades = len(df)
    return {
        "trades": int(trades),
        "positive_exit_rate": float((r > 0).mean()),
        "loss_exit_rate": float((r < 0).mean()),
        "average_managed_r": float(r.mean()),
        "median_managed_r": float(r.median()),
        "total_managed_r": float(r.sum()),
        "full_or_near_full_loss_count": int((r <= -0.90).sum()),
    }


def _bucket_numeric(series: pd.Series, column_name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() < 4:
        return pd.Series(["unknown"] * len(series), index=series.index)

    # Student-style comment: quantile buckets adapt to the cache distribution instead of fixed values.
    labels = [f"{column_name}_low", f"{column_name}_mid", f"{column_name}_high"]
    try:
        return pd.qcut(numeric.rank(method="first"), q=3, labels=labels).astype(str)
    except ValueError:
        return pd.cut(numeric, bins=3, labels=labels, include_lowest=True).astype(str)


def _feature_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "scenario_state",
        "event_state",
        "entry_mode",
        "entry_variant_name",
        "target_plan_state",
        "available_targets",
        "managed_exit_state",
        "fold_position",
    ]
    optional = [
        "risk_ratio_bucket",
        "best_rr_bucket",
        "channel_bias",
        "risk_efficiency_state",
        "plan_direction",
        "event_timeframe",
    ]
    return [column for column in preferred + optional if column in df.columns]


def _numeric_bucket_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        "entry_price",
        "risk_points",
        "tp1_rr",
        "tp2_rr",
        "tp3_rr",
        "best_available_rr",
        "event_quality_score",
        "probability_confidence_score",
    ]
    return [column for column in candidates if column in df.columns]


def _group_diagnostics(df: pd.DataFrame, config: FeatureEdgeConfig) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    for column in _numeric_bucket_columns(work):
        bucket_column = f"{column}_bucket"
        work[bucket_column] = _bucket_numeric(work[column], column)

    feature_columns = _feature_columns(work) + [
        f"{column}_bucket" for column in _numeric_bucket_columns(work)
    ]

    rows: list[dict] = []
    for feature in feature_columns:
        for value, group in work.groupby(feature, dropna=False):
            summary = _summarize_outcomes(group)
            if summary["trades"] < config.min_group_trades:
                state = "small_sample_watch_only"
            elif summary["average_managed_r"] < config.weak_r_threshold:
                state = "candidate_avoid_or_confirm_context"
            elif summary["average_managed_r"] >= config.strong_r_threshold:
                state = "candidate_keep_context"
            else:
                state = "neutral_context"

            rows.append(
                {
                    "feature_name": feature,
                    "feature_value": value,
                    **summary,
                    "diagnostic_state": state,
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    # Student-style comment: show the most useful contexts first.
    result["abs_average_r"] = result["average_managed_r"].abs()
    result = result.sort_values(
        ["diagnostic_state", "trades", "abs_average_r"],
        ascending=[True, False, False],
    ).drop(columns=["abs_average_r"])
    return result.reset_index(drop=True)


def diagnose_cache_feature_edges(config: FeatureEdgeConfig) -> dict[str, pd.DataFrame]:
    cache_path = Path(config.cache_path)
    dataset_df = pd.read_csv(cache_path)
    _require_columns(dataset_df, ["managed_realized_r"])

    entered_df = _select_entered_rows(dataset_df)
    selected_df = _apply_rule(entered_df, config.rule_name)
    selected_df = selected_df[pd.to_numeric(selected_df["managed_realized_r"], errors="coerce").notna()].copy()
    selected_df = _assign_chronological_folds(selected_df, config.fold_count)

    workload_df = pd.DataFrame(
        [
            {
                "cache_path": str(cache_path),
                "cache_rows": len(dataset_df),
                "entered_rows": len(entered_df),
                "selected_trades": len(selected_df),
                "rule_name": config.rule_name,
                "fold_count": config.fold_count,
                "min_group_trades": config.min_group_trades,
                "cache_design": "load_saved_cache_once_then_bucket_feature_edges",
            }
        ]
    )

    fold_rows = []
    for fold_id, group in selected_df.groupby("fold_id"):
        summary = _summarize_outcomes(group)
        fold_rows.append(
            {
                "fold_id": int(fold_id),
                "fold_position": group["fold_position"].iloc[0],
                "fold_start_cutoff": group["analysis_cutoff"].min() if "analysis_cutoff" in group else None,
                "fold_end_cutoff": group["analysis_cutoff"].max() if "analysis_cutoff" in group else None,
                **summary,
                "fold_state": "weak_or_negative_fold"
                if summary["average_managed_r"] < config.weak_r_threshold
                else "positive_fold",
            }
        )
    fold_df = pd.DataFrame(fold_rows)

    diagnostics_df = _group_diagnostics(selected_df, config)
    avoid_df = diagnostics_df[diagnostics_df["diagnostic_state"].eq("candidate_avoid_or_confirm_context")].copy()
    keep_df = diagnostics_df[diagnostics_df["diagnostic_state"].eq("candidate_keep_context")].copy()
    neutral_df = diagnostics_df[~diagnostics_df["diagnostic_state"].isin(["candidate_avoid_or_confirm_context", "candidate_keep_context"])].copy()

    return {
        "workload": workload_df,
        "fold_summary": fold_df,
        "candidate_avoid_contexts": avoid_df,
        "candidate_keep_contexts": keep_df,
        "neutral_or_small_sample_contexts": neutral_df,
        "selected_trade_preview": selected_df.head(30).copy(),
    }
