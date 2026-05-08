"""Diagnostics for exported HTF trend-channel candidate datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HTFCandidateDiagnosticsConfig:
    """Configuration for a saved candidate-dataset diagnostic run."""

    candidate_dataset_path: str
    fold_count: int = 10
    tag: str = "htf_candidate_diagnostics"
    output_dir: str = "outputs/trend_channel_htf_candidate_diagnostics"


def _safe_name(value: str) -> str:
    """Make a filesystem-safe name from a user-provided tag or rule name."""
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9_\-]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_") or "candidate"


def _rate(series: pd.Series) -> float:
    """Return the mean of a boolean series as a float rate."""
    if len(series) == 0:
        return float("nan")
    return float(series.mean())


def _get_full_loss_mask(df: pd.DataFrame) -> pd.Series:
    """Use the existing full-loss label if available; otherwise infer it from R."""
    if "label_full_or_near_full_loss" in df.columns:
        return df["label_full_or_near_full_loss"].astype(bool)
    return df["label_managed_r"] <= -0.95


def _derive_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add common diagnostic columns without changing original label meaning."""
    df = df.copy()

    # Student-style note: normalize dates so daily summaries group correctly.
    if "analysis_cutoff" in df.columns:
        df["analysis_cutoff"] = pd.to_datetime(df["analysis_cutoff"], errors="coerce")
    if "trade_date" not in df.columns and "analysis_cutoff" in df.columns:
        df["trade_date"] = df["analysis_cutoff"].dt.date.astype(str)

    # Student-style note: compute target distance when the export does not already include it.
    if "best_target_distance" not in df.columns:
        if {"risk_points", "best_available_rr"}.issubset(df.columns):
            df["best_target_distance"] = df["risk_points"] * df["best_available_rr"]
        else:
            df["best_target_distance"] = np.nan

    # Student-style note: keep labels separate from pre-entry feature columns.
    df["diagnostic_positive_r"] = df["label_managed_r"] > 0
    df["diagnostic_full_loss"] = _get_full_loss_mask(df)
    return df


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build one-row candidate summary statistics."""
    unique_days = df["trade_date"].nunique() if "trade_date" in df.columns else np.nan
    trades = len(df)
    avg_trades_per_day = trades / unique_days if unique_days else np.nan

    summary = {
        "trades": trades,
        "unique_days": unique_days,
        "avg_trades_per_day": avg_trades_per_day,
        "positive_exit_rate": _rate(df["diagnostic_positive_r"]),
        "full_loss_rate": _rate(df["diagnostic_full_loss"]),
        "average_managed_r": df["label_managed_r"].mean(),
        "median_managed_r": df["label_managed_r"].median(),
        "total_managed_r": df["label_managed_r"].sum(),
        "average_risk_points": df.get("risk_points", pd.Series(dtype=float)).mean(),
        "median_risk_points": df.get("risk_points", pd.Series(dtype=float)).median(),
        "average_best_available_rr": df.get("best_available_rr", pd.Series(dtype=float)).mean(),
        "median_best_available_rr": df.get("best_available_rr", pd.Series(dtype=float)).median(),
        "average_best_target_distance": df["best_target_distance"].mean(),
        "median_best_target_distance": df["best_target_distance"].median(),
    }

    # Student-style note: drawdown is based on cumulative R in chronological order.
    if "analysis_cutoff" in df.columns:
        ordered = df.sort_values("analysis_cutoff")
    else:
        ordered = df.copy()
    cumulative_r = ordered["label_managed_r"].cumsum()
    running_peak = cumulative_r.cummax()
    drawdown = cumulative_r - running_peak
    summary["max_drawdown_r"] = drawdown.min() if len(drawdown) else np.nan
    summary["ending_cumulative_r"] = cumulative_r.iloc[-1] if len(cumulative_r) else np.nan
    return pd.DataFrame([summary])


def _daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize performance by trade date."""
    if "trade_date" not in df.columns:
        return pd.DataFrame()

    grouped = df.groupby("trade_date", dropna=False)
    rows = []
    for trade_date, group in grouped:
        rows.append(
            {
                "trade_date": trade_date,
                "trades": len(group),
                "positive_exit_rate": _rate(group["diagnostic_positive_r"]),
                "full_loss_rate": _rate(group["diagnostic_full_loss"]),
                "average_managed_r": group["label_managed_r"].mean(),
                "median_managed_r": group["label_managed_r"].median(),
                "total_managed_r": group["label_managed_r"].sum(),
            }
        )
    return pd.DataFrame(rows)


def _fold_summary(df: pd.DataFrame, fold_count: int) -> pd.DataFrame:
    """Split rows chronologically and summarize each fold."""
    if len(df) == 0:
        return pd.DataFrame()

    if "analysis_cutoff" in df.columns:
        ordered = df.sort_values("analysis_cutoff").reset_index(drop=True)
    else:
        ordered = df.reset_index(drop=True)

    rows = []
    for fold_id, fold_indices in enumerate(np.array_split(np.arange(len(ordered)), fold_count), start=1):
        if len(fold_indices) == 0:
            continue
        group = ordered.iloc[fold_indices]
        rows.append(
            {
                "fold_id": fold_id,
                "trades": len(group),
                "unique_days": group["trade_date"].nunique() if "trade_date" in group.columns else np.nan,
                "fold_start_cutoff": group["analysis_cutoff"].min() if "analysis_cutoff" in group.columns else np.nan,
                "fold_end_cutoff": group["analysis_cutoff"].max() if "analysis_cutoff" in group.columns else np.nan,
                "positive_exit_rate": _rate(group["diagnostic_positive_r"]),
                "full_loss_rate": _rate(group["diagnostic_full_loss"]),
                "average_managed_r": group["label_managed_r"].mean(),
                "median_managed_r": group["label_managed_r"].median(),
                "total_managed_r": group["label_managed_r"].sum(),
            }
        )
    return pd.DataFrame(rows)


def _bucket_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize important categorical pre-entry columns when they exist."""
    candidate_columns = [
        "scenario_state",
        "event_state",
        "entry_variant_name",
        "target_plan_state",
        "htf_d1_bias",
        "htf_h4_bias",
        "htf_h1_bias",
        "htf_m15_bias",
        "htf_m5_bias",
    ]

    rows = []
    for column in candidate_columns:
        if column not in df.columns:
            continue
        for value, group in df.groupby(column, dropna=False):
            rows.append(
                {
                    "feature_name": column,
                    "feature_value": value,
                    "trades": len(group),
                    "positive_exit_rate": _rate(group["diagnostic_positive_r"]),
                    "full_loss_rate": _rate(group["diagnostic_full_loss"]),
                    "average_managed_r": group["label_managed_r"].mean(),
                    "median_managed_r": group["label_managed_r"].median(),
                    "total_managed_r": group["label_managed_r"].sum(),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["average_managed_r", "trades"], ascending=[False, False])


def analyze_htf_candidate_dataset(config: HTFCandidateDiagnosticsConfig) -> dict[str, pd.DataFrame | str]:
    """Run final diagnostics on one exported HTF candidate dataset."""
    dataset_path = Path(config.candidate_dataset_path)
    df = pd.read_csv(dataset_path)
    df = _derive_columns(df)

    summary_df = _summary(df)
    daily_df = _daily_summary(df)
    fold_df = _fold_summary(df, config.fold_count)
    bucket_df = _bucket_summary(df)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = _safe_name(config.tag)

    summary_path = output_dir / f"{tag}_summary.csv"
    daily_path = output_dir / f"{tag}_daily_summary.csv"
    fold_path = output_dir / f"{tag}_fold_summary.csv"
    bucket_path = output_dir / f"{tag}_bucket_summary.csv"

    summary_df.to_csv(summary_path, index=False)
    daily_df.to_csv(daily_path, index=False)
    fold_df.to_csv(fold_path, index=False)
    bucket_df.to_csv(bucket_path, index=False)

    return {
        "summary": summary_df,
        "daily_summary": daily_df,
        "fold_summary": fold_df,
        "bucket_summary": bucket_df,
        "summary_path": str(summary_path),
        "daily_summary_path": str(daily_path),
        "fold_summary_path": str(fold_path),
        "bucket_summary_path": str(bucket_path),
    }
