"""HTF candidate dataset export v2.

This module intentionally keeps the higher-timeframe candidate rule predicates
in one place so the export step matches the validation step.

All rules must use only pre-entry columns that already exist in the enriched
dataset. Outcome columns such as label_managed_r are used only for summaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HTFCandidateExportConfig:
    enriched_dataset_path: str
    rule_names: list[str]
    fold_count: int = 10
    tag: str = "step_htf_candidate_export_v2"
    output_dir: str = "outputs/trend_channel_htf_candidates"


def _safe_bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a boolean series; missing columns become False.

    Student note: the enriched dataset should contain these columns, but this
    helper makes the script safer if a column is missing during early tests.
    """
    if column not in df.columns:
        return pd.Series(False, index=df.index)

    series = df[column]

    if series.dtype == bool:
        return series.fillna(False)

    # Student note: CSV files may reload booleans as strings.
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )


def _not_against(df: pd.DataFrame, timeframes: Iterable[str]) -> pd.Series:
    """Return rows where the selected HTF bias is not against the plan."""
    mask = pd.Series(True, index=df.index)

    for timeframe in timeframes:
        column = f"htf_{timeframe.lower()}_bias_against_plan"
        mask &= ~_safe_bool_series(df, column)

    return mask


def apply_htf_candidate_rule(df: pd.DataFrame, rule_name: str) -> pd.DataFrame:
    """Apply one candidate rule using only pre-entry features.

    Student note: these predicates are intentionally explicit. This prevents the
    exporter from silently using a looser rule than the validator.
    """
    if rule_name == "candidate_v1_base":
        mask = pd.Series(True, index=df.index)

    elif rule_name == "require_baseline_current_close":
        mask = df.get("entry_variant_name", "").astype(str).eq("baseline_current_close")

    elif rule_name == "baseline_no_h4_opposition":
        mask = _not_against(df, ["H4"])

    elif rule_name == "baseline_no_d1_h4_opposition":
        mask = _not_against(df, ["D1", "H4"])

    elif rule_name == "baseline_d1_h4_m15_m5_not_against":
        # Main Candidate v1.2 rule:
        # D1, H4, M15, and M5 are not allowed to oppose the trade direction.
        # H1 is intentionally not included here because the prior validation
        # tested this exact rule name separately from the stricter premium rule.
        mask = _not_against(df, ["D1", "H4", "M15", "M5"])

    elif rule_name == "premium_baseline_no_htf_opposition":
        # Premium mode: no configured HTF context is against the plan.
        mask = _not_against(df, ["D1", "H4", "H1", "M15", "M5"])

    else:
        raise ValueError(f"Unknown HTF candidate rule: {rule_name}")

    selected = df.loc[mask].copy()
    selected["htf_candidate_rule"] = rule_name
    return selected


def _metric_summary(df: pd.DataFrame, rule_name: str) -> dict:
    """Create summary metrics for a selected candidate dataset."""
    if df.empty:
        return {
            "rule_name": rule_name,
            "trades": 0,
            "unique_days": 0,
            "avg_trades_per_day": 0.0,
            "positive_exit_rate": 0.0,
            "full_loss_rate": 0.0,
            "average_managed_r": 0.0,
            "median_managed_r": 0.0,
            "total_managed_r": 0.0,
            "average_risk_points": np.nan,
            "median_risk_points": np.nan,
            "average_best_available_rr": np.nan,
            "median_best_available_rr": np.nan,
            "average_best_target_distance": np.nan,
            "median_best_target_distance": np.nan,
        }

    labels = df["label_managed_r"].astype(float)
    unique_days = int(df["trade_date"].nunique()) if "trade_date" in df.columns else 0

    return {
        "rule_name": rule_name,
        "trades": int(len(df)),
        "unique_days": unique_days,
        "avg_trades_per_day": float(len(df) / unique_days) if unique_days else 0.0,
        "positive_exit_rate": float((labels > 0).mean()),
        "full_loss_rate": float((labels <= -0.95).mean()),
        "average_managed_r": float(labels.mean()),
        "median_managed_r": float(labels.median()),
        "total_managed_r": float(labels.sum()),
        "average_risk_points": float(df["risk_points"].mean()) if "risk_points" in df.columns else np.nan,
        "median_risk_points": float(df["risk_points"].median()) if "risk_points" in df.columns else np.nan,
        "average_best_available_rr": float(df["best_available_rr"].mean()) if "best_available_rr" in df.columns else np.nan,
        "median_best_available_rr": float(df["best_available_rr"].median()) if "best_available_rr" in df.columns else np.nan,
        "average_best_target_distance": float(df["best_target_distance"].mean()) if "best_target_distance" in df.columns else np.nan,
        "median_best_target_distance": float(df["best_target_distance"].median()) if "best_target_distance" in df.columns else np.nan,
    }


def _daily_summary(df: pd.DataFrame, rule_name: str) -> pd.DataFrame:
    """Summarize one selected candidate by trade_date."""
    if df.empty or "trade_date" not in df.columns:
        return pd.DataFrame()

    rows = []
    for trade_date, group in df.groupby("trade_date", sort=True):
        labels = group["label_managed_r"].astype(float)
        rows.append({
            "rule_name": rule_name,
            "trade_date": trade_date,
            "trades": int(len(group)),
            "positive_exit_rate": float((labels > 0).mean()),
            "full_loss_rate": float((labels <= -0.95).mean()),
            "average_managed_r": float(labels.mean()),
            "median_managed_r": float(labels.median()),
            "total_managed_r": float(labels.sum()),
        })
    return pd.DataFrame(rows)


def _fold_summary(df: pd.DataFrame, rule_name: str, fold_count: int) -> pd.DataFrame:
    """Create chronological fold summary for one candidate."""
    if df.empty:
        return pd.DataFrame()

    ordered = df.sort_values("analysis_cutoff").reset_index(drop=True)
    index_folds = np.array_split(np.arange(len(ordered)), fold_count)

    rows = []
    for fold_id, indices in enumerate(index_folds, start=1):
        if len(indices) == 0:
            continue

        group = ordered.iloc[indices].copy()
        labels = group["label_managed_r"].astype(float)

        rows.append({
            "rule_name": rule_name,
            "fold_id": fold_id,
            "trades": int(len(group)),
            "unique_days": int(group["trade_date"].nunique()) if "trade_date" in group.columns else 0,
            "fold_start_cutoff": str(group["analysis_cutoff"].min()),
            "fold_end_cutoff": str(group["analysis_cutoff"].max()),
            "positive_exit_rate": float((labels > 0).mean()),
            "full_loss_rate": float((labels <= -0.95).mean()),
            "average_managed_r": float(labels.mean()),
            "median_managed_r": float(labels.median()),
            "total_managed_r": float(labels.sum()),
        })

    return pd.DataFrame(rows)


def export_htf_candidate_datasets(config: HTFCandidateExportConfig) -> dict[str, str]:
    """Export selected datasets for each HTF candidate rule."""
    input_path = Path(config.enriched_dataset_path)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if "analysis_cutoff" in df.columns:
        df["analysis_cutoff"] = pd.to_datetime(df["analysis_cutoff"], errors="coerce")
    if "trade_date" not in df.columns and "analysis_cutoff" in df.columns:
        df["trade_date"] = df["analysis_cutoff"].dt.date.astype(str)

    summary_rows = []
    daily_frames = []
    fold_frames = []
    selected_frames = []
    output_paths: dict[str, str] = {}

    for rule_name in config.rule_names:
        selected = apply_htf_candidate_rule(df, rule_name)

        summary_rows.append(_metric_summary(selected, rule_name))
        daily = _daily_summary(selected, rule_name)
        fold = _fold_summary(selected, rule_name, config.fold_count)

        if not daily.empty:
            daily_frames.append(daily)
        if not fold.empty:
            fold_frames.append(fold)
        selected_frames.append(selected)

        selected_path = output_dir / f"{config.tag}_{rule_name}_selected_dataset.csv"
        selected.to_csv(selected_path, index=False)
        output_paths[rule_name] = str(selected_path)

    summary_df = pd.DataFrame(summary_rows)
    daily_df = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    fold_df = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    all_selected_df = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()

    summary_path = output_dir / f"{config.tag}_summary.csv"
    daily_path = output_dir / f"{config.tag}_daily_summary.csv"
    fold_path = output_dir / f"{config.tag}_fold_summary.csv"
    all_selected_path = output_dir / f"{config.tag}_all_selected_datasets.csv"

    summary_df.to_csv(summary_path, index=False)
    daily_df.to_csv(daily_path, index=False)
    fold_df.to_csv(fold_path, index=False)
    all_selected_df.to_csv(all_selected_path, index=False)

    output_paths["summary"] = str(summary_path)
    output_paths["daily_summary"] = str(daily_path)
    output_paths["fold_summary"] = str(fold_path)
    output_paths["all_selected_datasets"] = str(all_selected_path)
    return output_paths
