"""Export higher-timeframe filtered Step Index candidate datasets.

This module works from the already enriched HTF dataset. It does not rebuild
trend channels, scenarios, events, or trade management. That keeps this step fast
and makes it safe for repeated research iterations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


OUTPUT_DIR = Path("outputs/trend_channel_htf_candidates")


@dataclass(frozen=True)
class CandidateExportConfig:
    """Configuration for exporting one or more HTF candidate datasets."""

    enriched_dataset_path: str
    rule_names: tuple[str, ...]
    fold_count: int = 10
    tag: str = "step_htf_candidate_export"
    output_dir: Path = OUTPUT_DIR


def _parse_bool_series(series: pd.Series) -> pd.Series:
    """Convert common CSV boolean values into real booleans."""
    # CSV files may load booleans as strings, so normalize them safely.
    if series.dtype == bool:
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def _ensure_required_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise a clear error when the enriched dataset is missing expected fields."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in enriched dataset: {missing}")


def load_enriched_dataset(path: str | Path) -> pd.DataFrame:
    """Load the enriched dataset and normalize key columns for filtering."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Enriched dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    _ensure_required_columns(
        df,
        [
            "analysis_cutoff",
            "trade_date",
            "plan_direction",
            "entry_variant_name",
            "label_managed_r",
        ],
    )

    # Dates are parsed here so sorting/folding is chronological and reproducible.
    df["analysis_cutoff"] = pd.to_datetime(df["analysis_cutoff"], errors="coerce")
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.date
    df = df.dropna(subset=["analysis_cutoff"]).sort_values("analysis_cutoff").reset_index(drop=True)

    # Normalize boolean HTF opposition columns if they exist.
    for column in [
        "htf_d1_bias_against_plan",
        "htf_h4_bias_against_plan",
        "htf_h1_bias_against_plan",
        "htf_m15_bias_against_plan",
        "htf_m5_bias_against_plan",
    ]:
        if column in df.columns:
            df[column] = _parse_bool_series(df[column])

    return df


def apply_htf_candidate_rule(df: pd.DataFrame, rule_name: str) -> pd.DataFrame:
    """Apply one pre-entry HTF candidate rule.

    Rules must only use columns available before entry. Outcome columns are never
    used for filtering.
    """
    rule = rule_name.strip()
    selected = df.copy()

    if rule == "candidate_v1_base":
        return selected

    if rule == "require_baseline_current_close":
        return selected[selected["entry_variant_name"].astype(str) == "baseline_current_close"].copy()

    if rule == "baseline_no_h4_opposition":
        _ensure_required_columns(selected, ["htf_h4_bias_against_plan"])
        return selected[~selected["htf_h4_bias_against_plan"]].copy()

    if rule == "baseline_no_d1_h4_opposition":
        _ensure_required_columns(selected, ["htf_d1_bias_against_plan", "htf_h4_bias_against_plan"])
        mask = ~selected["htf_d1_bias_against_plan"] & ~selected["htf_h4_bias_against_plan"]
        return selected[mask].copy()

    if rule == "baseline_d1_h4_m15_m5_not_against":
        needed = [
            "htf_d1_bias_against_plan",
            "htf_h4_bias_against_plan",
            "htf_m15_bias_against_plan",
            "htf_m5_bias_against_plan",
        ]
        _ensure_required_columns(selected, needed)
        mask = pd.Series(True, index=selected.index)
        for column in needed:
            mask &= ~selected[column]
        return selected[mask].copy()

    if rule == "premium_baseline_no_htf_opposition":
        needed = [
            "htf_d1_bias_against_plan",
            "htf_h4_bias_against_plan",
            "htf_h1_bias_against_plan",
            "htf_m15_bias_against_plan",
            "htf_m5_bias_against_plan",
        ]
        _ensure_required_columns(selected, needed)
        mask = pd.Series(True, index=selected.index)
        for column in needed:
            mask &= ~selected[column]
        return selected[mask].copy()

    if rule == "baseline_h4_or_m15_aligned":
        _ensure_required_columns(selected, ["htf_h4_bias", "htf_m15_bias", "plan_direction"])
        plan = selected["plan_direction"].astype(str).str.lower()
        h4 = selected["htf_h4_bias"].astype(str).str.lower()
        m15 = selected["htf_m15_bias"].astype(str).str.lower()
        mask = (h4 == plan) | (m15 == plan)
        return selected[mask].copy()

    raise ValueError(f"Unknown HTF candidate rule: {rule_name}")


def _summarize_rule(rule_name: str, selected: pd.DataFrame) -> dict[str, object]:
    """Build a one-row performance summary for a selected candidate dataset."""
    if selected.empty:
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
            "average_risk_points": None,
            "median_risk_points": None,
            "average_best_available_rr": None,
            "median_best_available_rr": None,
            "average_best_target_distance": None,
            "median_best_target_distance": None,
        }

    managed_r = pd.to_numeric(selected["label_managed_r"], errors="coerce")
    positive = managed_r > 0
    # Treat -1R and below as full/near-full loss for consistency with prior diagnostics.
    full_loss = managed_r <= -0.99

    summary: dict[str, object] = {
        "rule_name": rule_name,
        "trades": int(len(selected)),
        "unique_days": int(selected["trade_date"].nunique()),
        "avg_trades_per_day": float(len(selected) / max(selected["trade_date"].nunique(), 1)),
        "positive_exit_rate": float(positive.mean()),
        "full_loss_rate": float(full_loss.mean()),
        "average_managed_r": float(managed_r.mean()),
        "median_managed_r": float(managed_r.median()),
        "total_managed_r": float(managed_r.sum()),
    }

    # Add distance profile columns when they exist in the enriched dataset.
    if "risk_points" in selected.columns:
        risk_points = pd.to_numeric(selected["risk_points"], errors="coerce")
        summary["average_risk_points"] = float(risk_points.mean())
        summary["median_risk_points"] = float(risk_points.median())
    else:
        risk_points = pd.Series(dtype=float)
        summary["average_risk_points"] = None
        summary["median_risk_points"] = None

    if "best_available_rr" in selected.columns:
        best_rr = pd.to_numeric(selected["best_available_rr"], errors="coerce")
        summary["average_best_available_rr"] = float(best_rr.mean())
        summary["median_best_available_rr"] = float(best_rr.median())
        if not risk_points.empty:
            best_target_distance = risk_points * best_rr
            summary["average_best_target_distance"] = float(best_target_distance.mean())
            summary["median_best_target_distance"] = float(best_target_distance.median())
        else:
            summary["average_best_target_distance"] = None
            summary["median_best_target_distance"] = None
    else:
        summary["average_best_available_rr"] = None
        summary["median_best_available_rr"] = None
        summary["average_best_target_distance"] = None
        summary["median_best_target_distance"] = None

    return summary


def _summarize_daily(rule_name: str, selected: pd.DataFrame) -> pd.DataFrame:
    """Summarize performance by trade date."""
    if selected.empty:
        return pd.DataFrame()

    rows = []
    for trade_date, group in selected.groupby("trade_date", sort=True):
        managed_r = pd.to_numeric(group["label_managed_r"], errors="coerce")
        rows.append(
            {
                "rule_name": rule_name,
                "trade_date": trade_date,
                "trades": int(len(group)),
                "positive_exit_rate": float((managed_r > 0).mean()),
                "full_loss_rate": float((managed_r <= -0.99).mean()),
                "average_managed_r": float(managed_r.mean()),
                "median_managed_r": float(managed_r.median()),
                "total_managed_r": float(managed_r.sum()),
            }
        )
    return pd.DataFrame(rows)


def _summarize_folds(rule_name: str, selected: pd.DataFrame, fold_count: int) -> pd.DataFrame:
    """Create chronological fold summaries for stability checks."""
    if selected.empty:
        return pd.DataFrame()

    selected = selected.sort_values("analysis_cutoff").reset_index(drop=True)
    # qcut can fail with tiny datasets, so use array_split-style indexing instead.
    indices = pd.Series(range(len(selected)))
    fold_ids = (indices * fold_count // max(len(selected), 1)) + 1
    fold_ids = fold_ids.clip(upper=fold_count)
    selected = selected.copy()
    selected["fold_id"] = fold_ids.values

    rows = []
    for fold_id, group in selected.groupby("fold_id", sort=True):
        managed_r = pd.to_numeric(group["label_managed_r"], errors="coerce")
        rows.append(
            {
                "rule_name": rule_name,
                "fold_id": int(fold_id),
                "trades": int(len(group)),
                "unique_days": int(group["trade_date"].nunique()),
                "fold_start_cutoff": group["analysis_cutoff"].min(),
                "fold_end_cutoff": group["analysis_cutoff"].max(),
                "positive_exit_rate": float((managed_r > 0).mean()),
                "full_loss_rate": float((managed_r <= -0.99).mean()),
                "average_managed_r": float(managed_r.mean()),
                "median_managed_r": float(managed_r.median()),
                "total_managed_r": float(managed_r.sum()),
            }
        )
    return pd.DataFrame(rows)


def export_htf_candidate_datasets(config: CandidateExportConfig) -> dict[str, pd.DataFrame]:
    """Export selected candidate datasets and their summaries."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    source_df = load_enriched_dataset(config.enriched_dataset_path)

    summary_rows = []
    daily_frames = []
    fold_frames = []
    selected_frames = []

    for rule_name in config.rule_names:
        selected = apply_htf_candidate_rule(source_df, rule_name)
        selected = selected.sort_values("analysis_cutoff").reset_index(drop=True)
        selected.insert(0, "candidate_rule_name", rule_name)

        summary_rows.append(_summarize_rule(rule_name, selected))
        daily_frames.append(_summarize_daily(rule_name, selected))
        fold_frames.append(_summarize_folds(rule_name, selected, config.fold_count))
        selected_frames.append(selected)

        selected_path = config.output_dir / f"{config.tag}_{rule_name}_selected_dataset.csv"
        selected.to_csv(selected_path, index=False)

    summary_df = pd.DataFrame(summary_rows)
    daily_df = pd.concat([frame for frame in daily_frames if not frame.empty], ignore_index=True) if daily_frames else pd.DataFrame()
    fold_df = pd.concat([frame for frame in fold_frames if not frame.empty], ignore_index=True) if fold_frames else pd.DataFrame()
    selected_all_df = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()

    summary_df.to_csv(config.output_dir / f"{config.tag}_summary.csv", index=False)
    daily_df.to_csv(config.output_dir / f"{config.tag}_daily_summary.csv", index=False)
    fold_df.to_csv(config.output_dir / f"{config.tag}_fold_summary.csv", index=False)
    selected_all_df.to_csv(config.output_dir / f"{config.tag}_all_selected_datasets.csv", index=False)

    return {
        "summary": summary_df,
        "daily_summary": daily_df,
        "fold_summary": fold_df,
        "selected_datasets": selected_all_df,
    }
