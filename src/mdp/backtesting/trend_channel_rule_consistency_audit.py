"""Audit Step Index trend-channel rule selection consistency before ML training.

This module works from a saved research cache. It is intentionally lightweight:
it does not rebuild the strategy pipeline, does not reread candles, and does not
use future outcomes as features. The purpose is to verify that candidate rules
used by different diagnostics select the same intended trades before exporting
ML datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from itertools import combinations

import pandas as pd

from mdp.backtesting.trend_channel_ml_dataset_export import (
    apply_step_candidate_rule,
    dedupe_one_trade_per_cutoff,
)


@dataclass(frozen=True)
class RuleConsistencyAuditConfig:
    cache_path: str
    rule_names: tuple[str, ...] = (
        "candidate_v1_base",
        "avoid_tp3_only",
        "require_best_available_rr_high",
        "keep_high_rr_or_risk_high",
    )
    output_dir: str = "outputs/trend_channel_rule_audits"
    tag: str = "step_rule_consistency_audit"


def _safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _safe_string(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([""] * len(df), index=df.index)
    return df[column].fillna("").astype(str)


def _selected_summary(rule_name: str, raw_df: pd.DataFrame, deduped_df: pd.DataFrame) -> dict:
    managed = _safe_numeric(deduped_df.get("managed_realized_r", pd.Series(dtype=float)))
    full_loss = managed <= -0.95
    positive = managed > 0

    target_counts = (
        _safe_string(deduped_df, "target_plan_state")
        .value_counts(dropna=False)
        .to_dict()
        if len(deduped_df)
        else {}
    )
    entry_counts = (
        _safe_string(deduped_df, "entry_variant_name")
        .value_counts(dropna=False)
        .to_dict()
        if len(deduped_df)
        else {}
    )

    return {
        "rule_name": rule_name,
        "raw_selected_rows": len(raw_df),
        "deduped_selected_rows": len(deduped_df),
        "unique_analysis_cutoffs": deduped_df["analysis_cutoff"].nunique() if "analysis_cutoff" in deduped_df.columns else 0,
        "positive_exit_rate": float(positive.mean()) if len(deduped_df) else 0.0,
        "full_loss_rate": float(full_loss.mean()) if len(deduped_df) else 0.0,
        "average_managed_r": float(managed.mean()) if len(deduped_df) else 0.0,
        "median_managed_r": float(managed.median()) if len(deduped_df) else 0.0,
        "total_managed_r": float(managed.sum()) if len(deduped_df) else 0.0,
        "target_plan_distribution": str(target_counts),
        "entry_variant_distribution": str(entry_counts),
    }


def _make_identity(df: pd.DataFrame) -> pd.Series:
    """Create a deterministic identity for selected rows to compare rules."""
    cols = [
        "analysis_cutoff",
        "entry_variant_name",
        "entry_mode",
        "target_plan_state",
        "available_targets",
        "risk_points",
        "managed_realized_r",
    ]
    present = [col for col in cols if col in df.columns]
    if not present:
        return pd.Series([], dtype=str)
    return df[present].astype(str).agg("|".join, axis=1)


def build_rule_consistency_audit(config: RuleConsistencyAuditConfig) -> dict[str, pd.DataFrame]:
    cache_path = Path(config.cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache path does not exist: {cache_path}")

    cache_df = pd.read_csv(cache_path)
    selected_by_rule: dict[str, pd.DataFrame] = {}
    raw_by_rule: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict] = []

    for rule_name in config.rule_names:
        raw_df = apply_step_candidate_rule(cache_df, rule_name)
        deduped_df = dedupe_one_trade_per_cutoff(raw_df)
        raw_by_rule[rule_name] = raw_df
        selected_by_rule[rule_name] = deduped_df
        summary_rows.append(_selected_summary(rule_name, raw_df, deduped_df))

    summary_df = pd.DataFrame(summary_rows)

    comparison_rows: list[dict] = []
    for left, right in combinations(config.rule_names, 2):
        left_df = selected_by_rule[left]
        right_df = selected_by_rule[right]
        left_cutoffs = set(pd.to_datetime(left_df.get("analysis_cutoff", pd.Series(dtype=str))).astype(str))
        right_cutoffs = set(pd.to_datetime(right_df.get("analysis_cutoff", pd.Series(dtype=str))).astype(str))
        left_ids = set(_make_identity(left_df))
        right_ids = set(_make_identity(right_df))
        comparison_rows.append(
            {
                "left_rule": left,
                "right_rule": right,
                "same_cutoff_count": len(left_cutoffs & right_cutoffs),
                "left_only_cutoffs": len(left_cutoffs - right_cutoffs),
                "right_only_cutoffs": len(right_cutoffs - left_cutoffs),
                "same_selected_row_count": len(left_ids & right_ids),
                "left_only_selected_rows": len(left_ids - right_ids),
                "right_only_selected_rows": len(right_ids - left_ids),
                "identical_final_selection": left_ids == right_ids,
                "diagnostic_note": (
                    "same_final_rows_despite_possible_raw_difference"
                    if left_ids == right_ids and len(raw_by_rule[left]) != len(raw_by_rule[right])
                    else "different_final_selection_or_same_raw_behavior"
                ),
            }
        )
    comparison_df = pd.DataFrame(comparison_rows)

    # Flag rows where TP3-only survived final selection for rules that should avoid it.
    violation_rows: list[dict] = []
    for rule_name, selected_df in selected_by_rule.items():
        target_plan = _safe_string(selected_df, "target_plan_state").str.lower()
        targets = _safe_string(selected_df, "available_targets").str.upper()
        tp3_only_mask = target_plan.eq("tp3_available") | targets.eq("TP3")
        violation_rows.append(
            {
                "rule_name": rule_name,
                "final_selected_rows": len(selected_df),
                "tp3_only_final_rows": int(tp3_only_mask.sum()),
                "tp3_only_violation_state": (
                    "unexpected_tp3_only_rows" if "avoid_tp3" in rule_name and int(tp3_only_mask.sum()) > 0 else "ok"
                ),
            }
        )
    violation_df = pd.DataFrame(violation_rows)

    workload_df = pd.DataFrame(
        [
            {
                "cache_path": str(cache_path),
                "cache_rows": len(cache_df),
                "rule_count": len(config.rule_names),
                "cache_design": "load_saved_cache_once_then_audit_rule_consistency",
            }
        ]
    )

    return {
        "workload": workload_df,
        "summary": summary_df,
        "pairwise_comparison": comparison_df,
        "rule_violation_check": violation_df,
    }


def save_rule_consistency_audit(config: RuleConsistencyAuditConfig, outputs: dict[str, pd.DataFrame]) -> dict[str, str]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    for name, df in outputs.items():
        path = output_dir / f"{config.tag}_{name}.csv"
        df.to_csv(path, index=False)
        saved[name] = str(path)
    return saved
