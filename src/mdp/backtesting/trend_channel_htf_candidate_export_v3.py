"""HTF candidate dataset export v3.

This module exports selected Step Index trend-channel candidate profiles from an
already enriched higher-timeframe dataset.

Important design choice:
- The HTF candidate names intentionally require the baseline_current_close entry
  variant when the rule name starts with "baseline_" or "premium_baseline".
- This is what should make the exported counts match the HTF validation counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class HtfCandidateExportConfig:
    enriched_dataset_path: str
    rule_names: list[str]
    fold_count: int = 10
    tag: str = "htf_candidate_export_v3"
    output_dir: str = "outputs/trend_channel_htf_candidates"


def _to_bool(series: pd.Series) -> pd.Series:
    """Convert boolean-like columns safely."""
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _base_mask(df: pd.DataFrame) -> pd.Series:
    return pd.Series(True, index=df.index)


def _baseline_current_close_mask(df: pd.DataFrame) -> pd.Series:
    _require_columns(df, ["entry_variant_name"])
    return df["entry_variant_name"].astype(str).eq("baseline_current_close")


def _not_against_mask(df: pd.DataFrame, timeframes: list[str]) -> pd.Series:
    columns = [f"htf_{timeframe.lower()}_bias_against_plan" for timeframe in timeframes]
    _require_columns(df, columns)
    mask = pd.Series(True, index=df.index)
    for column in columns:
        mask = mask & (~_to_bool(df[column]))
    return mask


def apply_htf_candidate_rule(df: pd.DataFrame, rule_name: str) -> pd.DataFrame:
    """Apply one named HTF candidate rule using explicit no-lookahead columns."""
    rule_name = rule_name.strip()

    if rule_name == "candidate_v1_base":
        mask = _base_mask(df)
    elif rule_name == "require_baseline_current_close":
        mask = _baseline_current_close_mask(df)
    elif rule_name == "baseline_no_h4_opposition":
        mask = _baseline_current_close_mask(df) & _not_against_mask(df, ["H4"])
    elif rule_name == "baseline_no_d1_h4_opposition":
        mask = _baseline_current_close_mask(df) & _not_against_mask(df, ["D1", "H4"])
    elif rule_name == "baseline_d1_h4_m15_m5_not_against":
        # This is Candidate v1.2: baseline current-close entry, with D1/H4/M15/M5
        # not fighting the planned direction. H1 is intentionally excluded because
        # H1 is already part of the original trend-channel structure context.
        mask = _baseline_current_close_mask(df) & _not_against_mask(df, ["D1", "H4", "M15", "M5"])
    elif rule_name == "premium_baseline_no_htf_opposition":
        # Premium mode: baseline current-close entry, with all available HTFs not
        # against the planned direction, including H1.
        mask = _baseline_current_close_mask(df) & _not_against_mask(df, ["D1", "H4", "H1", "M15", "M5"])
    else:
        raise ValueError(f"Unknown HTF candidate rule: {rule_name}")

    selected_df = df.loc[mask].copy()
    selected_df["rule_name"] = rule_name
    return selected_df


def _safe_mean(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.mean())


def _safe_median(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.median())


def _prepare_distance_columns(df: pd.DataFrame) -> pd.DataFrame:
    prepared_df = df.copy()
    if "best_target_distance" not in prepared_df.columns:
        if {"risk_points", "best_available_rr"}.issubset(prepared_df.columns):
            prepared_df["best_target_distance"] = (
                pd.to_numeric(prepared_df["risk_points"], errors="coerce")
                * pd.to_numeric(prepared_df["best_available_rr"], errors="coerce")
            )
    return prepared_df


def _summarize(selected_df: pd.DataFrame, rule_name: str) -> dict[str, object]:
    selected_df = _prepare_distance_columns(selected_df)
    trades = len(selected_df)
    positive = pd.to_numeric(selected_df.get("label_positive_r"), errors="coerce")
    full_loss = pd.to_numeric(selected_df.get("label_full_or_near_full_loss"), errors="coerce")
    managed_r = pd.to_numeric(selected_df.get("label_managed_r"), errors="coerce")
    trade_dates = pd.to_datetime(selected_df.get("trade_date"), errors="coerce")

    return {
        "rule_name": rule_name,
        "trades": trades,
        "unique_days": int(trade_dates.dt.date.nunique()) if trades else 0,
        "avg_trades_per_day": float(trades / trade_dates.dt.date.nunique()) if trades and trade_dates.dt.date.nunique() else 0.0,
        "positive_exit_rate": _safe_mean(positive),
        "full_loss_rate": _safe_mean(full_loss),
        "average_managed_r": _safe_mean(managed_r),
        "median_managed_r": _safe_median(managed_r),
        "total_managed_r": float(managed_r.sum()) if trades else 0.0,
        "average_risk_points": _safe_mean(selected_df.get("risk_points", pd.Series(dtype=float))),
        "median_risk_points": _safe_median(selected_df.get("risk_points", pd.Series(dtype=float))),
        "average_best_available_rr": _safe_mean(selected_df.get("best_available_rr", pd.Series(dtype=float))),
        "median_best_available_rr": _safe_median(selected_df.get("best_available_rr", pd.Series(dtype=float))),
        "average_best_target_distance": _safe_mean(selected_df.get("best_target_distance", pd.Series(dtype=float))),
        "median_best_target_distance": _safe_median(selected_df.get("best_target_distance", pd.Series(dtype=float))),
    }


def _build_fold_summary(selected_df: pd.DataFrame, rule_name: str, fold_count: int) -> pd.DataFrame:
    if selected_df.empty:
        return pd.DataFrame()

    sorted_df = selected_df.sort_values("analysis_cutoff").reset_index(drop=True)
    fold_ids = pd.qcut(sorted_df.index, q=min(fold_count, len(sorted_df)), labels=False, duplicates="drop") + 1
    sorted_df["fold_id"] = fold_ids

    rows: list[dict[str, object]] = []
    for fold_id, fold_df in sorted_df.groupby("fold_id", sort=True):
        managed_r = pd.to_numeric(fold_df["label_managed_r"], errors="coerce")
        positive = pd.to_numeric(fold_df["label_positive_r"], errors="coerce")
        full_loss = pd.to_numeric(fold_df["label_full_or_near_full_loss"], errors="coerce")
        trade_dates = pd.to_datetime(fold_df["trade_date"], errors="coerce")
        rows.append({
            "rule_name": rule_name,
            "fold_id": int(fold_id),
            "trades": len(fold_df),
            "unique_days": int(trade_dates.dt.date.nunique()),
            "fold_start_cutoff": fold_df["analysis_cutoff"].min(),
            "fold_end_cutoff": fold_df["analysis_cutoff"].max(),
            "positive_exit_rate": _safe_mean(positive),
            "full_loss_rate": _safe_mean(full_loss),
            "average_managed_r": _safe_mean(managed_r),
            "median_managed_r": _safe_median(managed_r),
            "total_managed_r": float(managed_r.sum()),
        })
    return pd.DataFrame(rows)


def _build_daily_summary(selected_df: pd.DataFrame, rule_name: str) -> pd.DataFrame:
    if selected_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for trade_date, day_df in selected_df.groupby("trade_date", sort=True):
        managed_r = pd.to_numeric(day_df["label_managed_r"], errors="coerce")
        positive = pd.to_numeric(day_df["label_positive_r"], errors="coerce")
        full_loss = pd.to_numeric(day_df["label_full_or_near_full_loss"], errors="coerce")
        rows.append({
            "rule_name": rule_name,
            "trade_date": trade_date,
            "trades": len(day_df),
            "positive_exit_rate": _safe_mean(positive),
            "full_loss_rate": _safe_mean(full_loss),
            "average_managed_r": _safe_mean(managed_r),
            "median_managed_r": _safe_median(managed_r),
            "total_managed_r": float(managed_r.sum()),
        })
    return pd.DataFrame(rows)


def export_htf_candidate_datasets_v3(config: HtfCandidateExportConfig) -> dict[str, object]:
    enriched_path = Path(config.enriched_dataset_path)
    dataset_df = pd.read_csv(enriched_path)
    dataset_df = _prepare_distance_columns(dataset_df)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    fold_frames: list[pd.DataFrame] = []
    daily_frames: list[pd.DataFrame] = []
    selected_frames: list[pd.DataFrame] = []
    selected_paths: dict[str, str] = {}

    for rule_name in config.rule_names:
        selected_df = apply_htf_candidate_rule(dataset_df, rule_name)
        summary_rows.append(_summarize(selected_df, rule_name))
        fold_frames.append(_build_fold_summary(selected_df, rule_name, config.fold_count))
        daily_frames.append(_build_daily_summary(selected_df, rule_name))
        selected_frames.append(selected_df)

        selected_path = output_dir / f"{config.tag}_{rule_name}_selected_dataset.csv"
        selected_df.to_csv(selected_path, index=False)
        selected_paths[rule_name] = str(selected_path)

    summary_df = pd.DataFrame(summary_rows)
    fold_summary_df = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    daily_summary_df = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    all_selected_df = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()

    summary_path = output_dir / f"{config.tag}_summary.csv"
    fold_summary_path = output_dir / f"{config.tag}_fold_summary.csv"
    daily_summary_path = output_dir / f"{config.tag}_daily_summary.csv"
    all_selected_path = output_dir / f"{config.tag}_all_selected_datasets.csv"

    summary_df.to_csv(summary_path, index=False)
    fold_summary_df.to_csv(fold_summary_path, index=False)
    daily_summary_df.to_csv(daily_summary_path, index=False)
    all_selected_df.to_csv(all_selected_path, index=False)

    return {
        "summary_df": summary_df,
        "fold_summary_df": fold_summary_df,
        "daily_summary_df": daily_summary_df,
        "selected_paths": selected_paths,
        "summary_path": str(summary_path),
        "fold_summary_path": str(fold_summary_path),
        "daily_summary_path": str(daily_summary_path),
        "all_selected_path": str(all_selected_path),
    }
