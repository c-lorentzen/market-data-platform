"""Utilities to export a no-lookahead ML-ready dataset from trend-channel research caches.

This module intentionally works from a saved cache CSV so it does not rebuild the full
strategy pipeline. It selects one candidate row per analysis cutoff using only pre-entry
features, then adds supervised labels from the already-simulated managed outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MLDatasetExportConfig:
    cache_path: str
    rule_name: str = "candidate_v1_base"
    output_dir: str = "outputs/trend_channel_ml_datasets"
    tag: str = "step_candidate_ml_dataset"
    max_trades_per_day: int | None = None
    positive_r_threshold: float = 0.0
    full_loss_r_threshold: float = -0.95


def _safe_series(df: pd.DataFrame, column: str, default: object = np.nan) -> pd.Series:
    """Return a column if it exists; otherwise return a default-valued Series."""
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)


def _string_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a lowercase string column for robust rule checks."""
    return _safe_series(df, column, "").fillna("").astype(str).str.lower()


def _numeric_series(df: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    """Return a numeric column and coerce invalid values to NaN/default."""
    values = pd.to_numeric(_safe_series(df, column, default), errors="coerce")
    if not np.isnan(default):
        values = values.fillna(default)
    return values


def apply_step_candidate_rule(df: pd.DataFrame, rule_name: str) -> pd.DataFrame:
    """Apply Step Index candidate rules using only pre-entry fields."""
    data = df.copy()

    entered = _safe_series(data, "entered_trade", True)
    if entered.dtype == object:
        entered_mask = entered.astype(str).str.lower().isin(["true", "1", "yes", "entered"])
    else:
        entered_mask = entered.astype(bool)

    target_plan = _string_series(data, "target_plan_state")
    targets = _string_series(data, "available_targets")
    event_state = _string_series(data, "event_state")
    entry_mode = _string_series(data, "entry_mode")
    rr_bucket = _string_series(data, "best_available_rr_bucket")
    risk_bucket = _string_series(data, "risk_points_bucket")
    scenario_state = _string_series(data, "scenario_state")

    # Candidate v1 base: the existing refined trend-channel candidate.
    # It keeps rows that still have a usable target plan and are entered.
    base_mask = entered_mask & target_plan.ne("") & ~target_plan.str.contains("no_available|unknown|none", regex=True)

    # This condition follows the strategy idea already tested in the project:
    # retest-rejection entries need TP2+TP3, while continuation entries may pass if entered by confirmation.
    retest_event = event_state.str.contains("retest_rejection") | entry_mode.str.contains("retest_rejection")
    has_tp2_tp3 = target_plan.str.contains("tp2_tp3|tp1_tp2_tp3") | targets.str.contains("tp2,tp3|tp1,tp2,tp3")
    continuation_entry = entry_mode.str.contains("continuation_after_retest")
    base_mask &= ((retest_event & has_tp2_tp3) | continuation_entry)

    if rule_name == "candidate_v1_base":
        mask = base_mask
    elif rule_name == "avoid_tp3_only":
        mask = base_mask & ~target_plan.eq("tp3_available") & ~targets.eq("tp3")
    elif rule_name == "require_best_available_rr_high":
        mask = base_mask & rr_bucket.eq("best_available_rr_high")
    elif rule_name == "require_risk_points_high":
        mask = base_mask & risk_bucket.eq("risk_points_high")
    elif rule_name == "keep_high_rr_or_risk_high":
        mask = base_mask & (rr_bucket.eq("best_available_rr_high") | risk_bucket.eq("risk_points_high"))
    elif rule_name == "keep_high_rr_or_bullish_pullback":
        mask = base_mask & (rr_bucket.eq("best_available_rr_high") | scenario_state.eq("bullish_structure_bearish_entry_pullback"))
    else:
        raise ValueError(f"Unknown rule_name: {rule_name}")

    return data.loc[mask].copy()


def add_pre_entry_selection_score(df: pd.DataFrame) -> pd.DataFrame:
    """Rank duplicate rows per cutoff using only pre-entry information."""
    data = df.copy()

    rr = _numeric_series(data, "_best_available_rr")
    if rr.isna().all():
        rr = _numeric_series(data, "best_available_rr")
    risk = _numeric_series(data, "risk_points")
    event_quality = _numeric_series(data, "event_quality_score", 0.0).fillna(0.0)

    target_plan = _string_series(data, "target_plan_state")
    target_score = np.select(
        [
            target_plan.str.contains("tp1_tp2_tp3"),
            target_plan.str.contains("tp2_tp3"),
            target_plan.str.contains("tp2_only"),
            target_plan.str.contains("tp3_available"),
        ],
        [4.0, 3.0, 2.0, 1.0],
        default=0.0,
    )

    # The score intentionally avoids managed_realized_r or any future outcome.
    # It prefers usable target structure, decent RR, stronger event quality, and not-too-large risk.
    risk_penalty = risk.rank(pct=True, ascending=True).fillna(0.5) * 0.25
    rr_score = rr.clip(lower=0, upper=20).fillna(0) / 20.0
    data["pre_entry_selection_score"] = target_score + rr_score + event_quality - risk_penalty
    return data


def dedupe_one_trade_per_cutoff(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one deterministic candidate per analysis cutoff using pre-entry score."""
    data = add_pre_entry_selection_score(df)
    if "analysis_cutoff" not in data.columns:
        raise ValueError("Cache must include an analysis_cutoff column for deduplication.")

    data["analysis_cutoff"] = pd.to_datetime(data["analysis_cutoff"], errors="coerce")
    data = data.sort_values(
        ["analysis_cutoff", "pre_entry_selection_score"],
        ascending=[True, False],
        kind="mergesort",
    )
    return data.drop_duplicates(subset=["analysis_cutoff"], keep="first").reset_index(drop=True)


def apply_max_trades_per_day(df: pd.DataFrame, max_trades_per_day: int | None) -> pd.DataFrame:
    """Optionally cap the number of selected trades per day by pre-entry score."""
    if max_trades_per_day is None or max_trades_per_day <= 0:
        return df.copy()
    data = df.copy()
    data["trade_date"] = pd.to_datetime(data["analysis_cutoff"]).dt.date
    data = data.sort_values(
        ["trade_date", "pre_entry_selection_score", "analysis_cutoff"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    data = data.groupby("trade_date", group_keys=False).head(max_trades_per_day)
    return data.sort_values("analysis_cutoff").reset_index(drop=True)


def build_ml_ready_dataset(config: MLDatasetExportConfig) -> dict[str, pd.DataFrame]:
    """Build summary tables and an ML-ready labeled dataset from a cache file."""
    cache_path = Path(config.cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache path does not exist: {cache_path}")

    cache_df = pd.read_csv(cache_path)
    selected_raw = apply_step_candidate_rule(cache_df, config.rule_name)
    selected_deduped = dedupe_one_trade_per_cutoff(selected_raw)
    selected_final = apply_max_trades_per_day(selected_deduped, config.max_trades_per_day)

    # Add labels from the already-simulated outcome. These are labels, not features.
    data = selected_final.copy()
    data["analysis_cutoff"] = pd.to_datetime(data["analysis_cutoff"], errors="coerce")
    data["trade_date"] = data["analysis_cutoff"].dt.date
    data["hour_utc"] = data["analysis_cutoff"].dt.hour
    managed_r = _numeric_series(data, "managed_realized_r", 0.0).fillna(0.0)
    data["label_positive_r"] = (managed_r > config.positive_r_threshold).astype(int)
    data["label_full_or_near_full_loss"] = (managed_r <= config.full_loss_r_threshold).astype(int)
    data["label_managed_r"] = managed_r

    # Mark likely feature columns available before entry. The model should not train on outcome columns.
    candidate_feature_cols = [
        "scenario_state",
        "event_state",
        "entry_mode",
        "entry_variant_name",
        "plan_direction",
        "target_plan_state",
        "available_targets",
        "risk_points",
        "risk_points_ratio",
        "_best_available_rr",
        "best_available_rr",
        "best_available_rr_bucket",
        "risk_points_bucket",
        "event_quality_score",
        "hour_utc",
        "pre_entry_selection_score",
    ]
    feature_cols = [col for col in candidate_feature_cols if col in data.columns]
    label_cols = ["label_positive_r", "label_full_or_near_full_loss", "label_managed_r"]
    id_cols = [col for col in ["analysis_cutoff", "trade_date"] if col in data.columns]
    output_cols = id_cols + feature_cols + label_cols
    ml_dataset = data[output_cols].copy()

    summary = pd.DataFrame(
        [
            {
                "cache_path": str(cache_path),
                "cache_rows": len(cache_df),
                "rule_name": config.rule_name,
                "raw_selected_rows": len(selected_raw),
                "deduped_selected_rows": len(selected_deduped),
                "final_selected_rows": len(data),
                "unique_days": data["trade_date"].nunique() if len(data) else 0,
                "avg_trades_per_day": data.groupby("trade_date").size().mean() if len(data) else 0.0,
                "positive_exit_rate": data["label_positive_r"].mean() if len(data) else 0.0,
                "full_loss_rate": data["label_full_or_near_full_loss"].mean() if len(data) else 0.0,
                "average_managed_r": data["label_managed_r"].mean() if len(data) else 0.0,
                "median_managed_r": data["label_managed_r"].median() if len(data) else 0.0,
                "total_managed_r": data["label_managed_r"].sum() if len(data) else 0.0,
                "max_trades_per_day": config.max_trades_per_day,
            }
        ]
    )

    daily = (
        data.groupby("trade_date", dropna=False)
        .agg(
            trades=("analysis_cutoff", "count"),
            positive_exit_rate=("label_positive_r", "mean"),
            full_loss_rate=("label_full_or_near_full_loss", "mean"),
            average_managed_r=("label_managed_r", "mean"),
            median_managed_r=("label_managed_r", "median"),
            total_managed_r=("label_managed_r", "sum"),
        )
        .reset_index()
        if len(data)
        else pd.DataFrame()
    )

    return {
        "summary": summary,
        "daily_summary": daily,
        "ml_dataset": ml_dataset,
        "selected_trades": data,
    }


def save_ml_ready_dataset(config: MLDatasetExportConfig) -> dict[str, str]:
    """Build and save CSV outputs for ML preparation."""
    outputs = build_ml_ready_dataset(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_rule = config.rule_name.replace(" ", "_").lower()
    prefix = output_dir / f"{config.tag}_{safe_rule}"
    paths: dict[str, str] = {}
    for name, df in outputs.items():
        path = f"{prefix}_{name}.csv"
        df.to_csv(path, index=False)
        paths[name] = path
    return paths
