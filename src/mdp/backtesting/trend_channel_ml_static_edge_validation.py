"""Static edge validation for Step Index trend-channel ML-ready datasets.

This module validates simple pre-entry context rules from an exported ML-ready
candidate dataset. It does not rebuild the strategy pipeline and does not use
post-entry outcome columns as features for selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd


OUTCOME_COLUMNS = {
    "label_positive_r",
    "label_full_or_near_full_loss",
    "label_managed_r",
}


@dataclass(frozen=True)
class StaticEdgeValidationConfig:
    ml_dataset_path: str
    rule_names: tuple[str, ...] = (
        "candidate_v1_base",
        "require_retest_rejection",
        "require_bullish_structure_bearish_entry_pullback",
        "require_baseline_current_close",
        "avoid_tp2_only",
        "keep_retest_or_bullish_pullback",
        "keep_best_rr_high_or_retest",
        "avoid_hour_17",
    )
    fold_count: int = 7
    min_total_trades: int = 30
    min_trades_per_fold: int = 3
    tag: str = "step_ml_static_edges"
    output_dir: str = "outputs/trend_channel_ml_static_edges"


def _normalize_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _derive_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Add stable buckets when the exported dataset lacks bucket columns."""
    out = df.copy()

    if "best_available_rr_bucket" not in out.columns and "online_best_rr_bucket" in out.columns:
        out["best_available_rr_bucket"] = out["online_best_rr_bucket"]

    if "risk_points_bucket" not in out.columns and "online_risk_ratio_bucket" in out.columns:
        out["risk_points_bucket"] = out["online_risk_ratio_bucket"]

    if "best_available_rr_bucket" not in out.columns and "best_available_rr" in out.columns:
        rr = pd.to_numeric(out["best_available_rr"], errors="coerce")
        # Quantile buckets are computed from pre-entry numeric RR only.
        q1, q2 = rr.quantile([0.33, 0.66]).tolist()
        out["best_available_rr_bucket"] = pd.cut(
            rr,
            bins=[float("-inf"), q1, q2, float("inf")],
            labels=["best_rr_low", "best_rr_mid", "best_rr_high"],
            include_lowest=True,
        ).astype("object").fillna("best_rr_unknown")

    if "risk_points_bucket" not in out.columns and "risk_points" in out.columns:
        risk = pd.to_numeric(out["risk_points"], errors="coerce")
        q1, q2 = risk.quantile([0.33, 0.66]).tolist()
        out["risk_points_bucket"] = pd.cut(
            risk,
            bins=[float("-inf"), q1, q2, float("inf")],
            labels=["risk_points_low", "risk_points_mid", "risk_points_high"],
            include_lowest=True,
        ).astype("object").fillna("risk_unknown")

    return out


def load_ml_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "analysis_cutoff" not in df.columns:
        raise ValueError("ML dataset must include analysis_cutoff.")
    if "label_managed_r" not in df.columns:
        raise ValueError("ML dataset must include label_managed_r.")

    df = df.copy()
    df["analysis_cutoff"] = pd.to_datetime(df["analysis_cutoff"], errors="coerce")
    if "trade_date" not in df.columns:
        df["trade_date"] = df["analysis_cutoff"].dt.date.astype(str)
    if "hour_utc" not in df.columns:
        df["hour_utc"] = df["analysis_cutoff"].dt.hour

    df = df.sort_values("analysis_cutoff").reset_index(drop=True)
    return _derive_buckets(df)


def _rule_mask(df: pd.DataFrame, rule_name: str) -> pd.Series:
    scenario = _normalize_text(df.get("scenario_state", pd.Series(index=df.index, dtype=str)))
    event = _normalize_text(df.get("event_state", pd.Series(index=df.index, dtype=str)))
    entry_mode = _normalize_text(df.get("entry_mode", pd.Series(index=df.index, dtype=str)))
    entry_variant = _normalize_text(df.get("entry_variant_name", pd.Series(index=df.index, dtype=str)))
    target_plan = _normalize_text(df.get("target_plan_state", pd.Series(index=df.index, dtype=str)))
    rr_bucket = _normalize_text(df.get("best_available_rr_bucket", pd.Series(index=df.index, dtype=str)))
    hour = pd.to_numeric(df.get("hour_utc", pd.Series(index=df.index, dtype=float)), errors="coerce")

    base = pd.Series(True, index=df.index)

    if rule_name == "candidate_v1_base":
        return base
    if rule_name == "require_retest_rejection":
        return event.eq("bearish_retest_rejection_confirmed") | entry_mode.eq("retest_rejection_close")
    if rule_name == "require_bullish_structure_bearish_entry_pullback":
        return scenario.eq("bullish_structure_bearish_entry_pullback")
    if rule_name == "require_baseline_current_close":
        return entry_variant.eq("baseline_current_close")
    if rule_name == "avoid_tp2_only":
        return ~target_plan.eq("tp2_only")
    if rule_name == "keep_retest_or_bullish_pullback":
        return (
            event.eq("bearish_retest_rejection_confirmed")
            | entry_mode.eq("retest_rejection_close")
            | scenario.eq("bullish_structure_bearish_entry_pullback")
        )
    if rule_name == "keep_best_rr_high_or_retest":
        return rr_bucket.str.contains("high", case=False, na=False) | event.eq("bearish_retest_rejection_confirmed")
    if rule_name == "avoid_hour_17":
        return hour.ne(17)
    if rule_name == "diagnostic_good_hours_1_5_9_14":
        # Diagnostic only: time-of-day filters are prone to overfitting on small samples.
        return hour.isin([1, 5, 9, 14])

    raise ValueError(f"Unknown static edge rule: {rule_name}")


def _summarize(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {
            "rule_name": label,
            "trades": 0,
            "unique_days": 0,
            "avg_trades_per_day": 0.0,
            "positive_exit_rate": 0.0,
            "full_loss_rate": 0.0,
            "average_managed_r": 0.0,
            "median_managed_r": 0.0,
            "total_managed_r": 0.0,
        }

    r = pd.to_numeric(df["label_managed_r"], errors="coerce")
    pos = pd.to_numeric(df.get("label_positive_r", (r > 0).astype(int)), errors="coerce")
    full_loss = pd.to_numeric(
        df.get("label_full_or_near_full_loss", (r <= -0.95).astype(int)), errors="coerce"
    )
    unique_days = df["trade_date"].nunique()
    return {
        "rule_name": label,
        "trades": int(len(df)),
        "unique_days": int(unique_days),
        "avg_trades_per_day": float(len(df) / unique_days) if unique_days else 0.0,
        "positive_exit_rate": float(pos.mean()) if len(df) else 0.0,
        "full_loss_rate": float(full_loss.mean()) if len(df) else 0.0,
        "average_managed_r": float(r.mean()) if len(df) else 0.0,
        "median_managed_r": float(r.median()) if len(df) else 0.0,
        "total_managed_r": float(r.sum()) if len(df) else 0.0,
    }


def _fold_summary(df: pd.DataFrame, rule_name: str, fold_count: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    ordered = df.sort_values("analysis_cutoff").reset_index(drop=True).copy()
    fold_count = max(1, min(fold_count, len(ordered)))
    ordered["fold_id"] = (ordered.index * fold_count // len(ordered)) + 1

    rows = []
    for fold_id, fold_df in ordered.groupby("fold_id", sort=True):
        summary = _summarize(fold_df, rule_name)
        summary.update(
            {
                "fold_id": int(fold_id),
                "fold_start_cutoff": fold_df["analysis_cutoff"].min(),
                "fold_end_cutoff": fold_df["analysis_cutoff"].max(),
            }
        )
        rows.append(summary)
    return pd.DataFrame(rows)


def validate_static_edges(config: StaticEdgeValidationConfig) -> dict[str, pd.DataFrame]:
    dataset = load_ml_dataset(config.ml_dataset_path)

    summary_rows = []
    fold_frames = []
    selected_frames = []

    for rule_name in config.rule_names:
        selected = dataset[_rule_mask(dataset, rule_name)].copy()
        summary = _summarize(selected, rule_name)
        folds = _fold_summary(selected, rule_name, config.fold_count)

        if not folds.empty:
            min_fold_avg = float(folds["average_managed_r"].min())
            positive_fold_rate = float((folds["average_managed_r"] > 0).mean())
            folds_with_min_trade_rate = float((folds["trades"] >= config.min_trades_per_fold).mean())
        else:
            min_fold_avg = 0.0
            positive_fold_rate = 0.0
            folds_with_min_trade_rate = 0.0

        validation_state = "candidate_static_edge_validate_harder"
        if summary["trades"] < config.min_total_trades:
            validation_state = "insufficient_total_trades_do_not_lock"
        elif summary["average_managed_r"] <= 0 or summary["median_managed_r"] <= 0:
            validation_state = "not_profitable_do_not_lock"
        elif positive_fold_rate < 0.75 or min_fold_avg < -0.25:
            validation_state = "positive_but_fold_instability_validate_harder"
        else:
            validation_state = "promising_static_edge_validate_on_larger_data"

        summary.update(
            {
                "min_fold_average_r": min_fold_avg,
                "positive_average_r_fold_rate": positive_fold_rate,
                "folds_with_min_trades_rate": folds_with_min_trade_rate,
                "validation_state": validation_state,
            }
        )
        summary_rows.append(summary)

        if not folds.empty:
            fold_frames.append(folds)
        preview_cols = [
            c
            for c in [
                "analysis_cutoff",
                "trade_date",
                "hour_utc",
                "scenario_state",
                "event_state",
                "entry_mode",
                "entry_variant_name",
                "target_plan_state",
                "available_targets",
                "best_available_rr",
                "best_available_rr_bucket",
                "risk_points",
                "risk_points_bucket",
                "label_managed_r",
            ]
            if c in selected.columns
        ]
        selected_preview = selected[preview_cols].copy().head(40)
        selected_preview.insert(0, "rule_name", rule_name)
        selected_frames.append(selected_preview)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["validation_state", "average_managed_r"], ascending=[True, False]
    )
    fold_df = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    selected_preview_df = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()

    return {
        "summary": summary_df,
        "fold_detail": fold_df,
        "selected_preview": selected_preview_df,
    }


def save_static_edge_outputs(config: StaticEdgeValidationConfig, outputs: dict[str, pd.DataFrame]) -> dict[str, str]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    for name, df in outputs.items():
        path = output_dir / f"{config.tag}_{name}.csv"
        df.to_csv(path, index=False)
        saved[name] = str(path)
    return saved
