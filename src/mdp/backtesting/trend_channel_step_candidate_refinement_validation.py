"""
Step Index Trend Channel Candidate Refinement Validation.

This module validates simple pre-entry refinements for the Step Index Trend
Channel Candidate v1 using an existing research cache CSV. It does not rebuild
any strategy pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StepCandidateRefinementConfig:
    """Configuration for cache-based refinement validation."""

    cache_path: str
    fold_count: int = 7
    min_total_trades: int = 20
    min_trades_per_fold: int = 3
    rule_names: tuple[str, ...] = (
        "candidate_v1_base",
        "avoid_entry_price_low",
        "avoid_risk_points_mid",
        "avoid_tp3_only",
        "require_best_available_rr_high",
        "require_risk_points_high",
        "avoid_entry_price_low_and_risk_mid",
        "keep_high_rr_or_risk_high",
        "keep_high_rr_or_bullish_pullback",
    )
    tag: str = "step_candidate_refinement"
    output_dir: str = "outputs/trend_channel_step_candidate_refinements"


RULE_DESCRIPTIONS: dict[str, str] = {
    "candidate_v1_base": (
        "Step Index Candidate v1: retest-rejection entries use baseline close "
        "and require TP2+TP3; continuation entries require two-candle confirmation."
    ),
    "avoid_entry_price_low": "Candidate v1 excluding low entry-price bucket setups.",
    "avoid_risk_points_mid": "Candidate v1 excluding mid risk-points bucket setups.",
    "avoid_tp3_only": "Candidate v1 excluding TP3-only target plans.",
    "require_best_available_rr_high": "Candidate v1 keeping only high best-available-RR setups.",
    "require_risk_points_high": "Candidate v1 keeping only high risk-points bucket setups.",
    "avoid_entry_price_low_and_risk_mid": (
        "Candidate v1 excluding low entry-price bucket and mid risk-points bucket setups."
    ),
    "keep_high_rr_or_risk_high": (
        "Candidate v1 keeping setups with high best-available-RR or high risk-points bucket."
    ),
    "keep_high_rr_or_bullish_pullback": (
        "Candidate v1 keeping high best-available-RR setups or bullish-structure bearish-entry pullbacks."
    ),
}


def _as_bool_series(series: pd.Series) -> pd.Series:
    """Convert mixed boolean/text values into a clean boolean Series."""
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes", "entered"})


def _contains_text(series: pd.Series, text: str) -> pd.Series:
    """Safely check whether a text fragment exists in a possibly missing Series."""
    return series.fillna("").astype(str).str.contains(text, case=False, regex=False)


def load_step_candidate_cache(cache_path: str | Path) -> pd.DataFrame:
    """Load a saved trend-channel research cache."""
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    dataset_df = pd.read_csv(path)
    if "analysis_cutoff" in dataset_df.columns:
        dataset_df["analysis_cutoff"] = pd.to_datetime(dataset_df["analysis_cutoff"], errors="coerce")

    return dataset_df


def select_candidate_v1_base(dataset_df: pd.DataFrame) -> pd.DataFrame:
    """Select the base Step Index Candidate v1 rows from the research cache."""
    df = dataset_df.copy()

    # Only rows representing actual entered variants can be evaluated.
    entered_mask = pd.Series(True, index=df.index)
    if "entered_trade" in df.columns:
        entered_mask = _as_bool_series(df["entered_trade"])
    if "entry_status" in df.columns:
        entered_mask &= df["entry_status"].fillna("").astype(str).str.lower().eq("entered")

    entry_mode = df.get("entry_mode", pd.Series("", index=df.index)).fillna("").astype(str)
    variant = df.get("entry_variant_name", pd.Series("", index=df.index)).fillna("").astype(str)
    available_targets = df.get("available_targets", pd.Series("", index=df.index))

    # Retest rejection entries are cleaner when TP2 and TP3 are both available.
    retest_mask = (
        entry_mode.eq("retest_rejection_close")
        & variant.eq("baseline_current_close")
        & _contains_text(available_targets, "TP2")
        & _contains_text(available_targets, "TP3")
    )

    # Continuation entries must use the two-candle directional confirmation variant.
    continuation_mask = (
        entry_mode.eq("continuation_after_retest_close")
        & variant.eq("confirm_2_directional_close")
    )

    selected_df = df.loc[entered_mask & (retest_mask | continuation_mask)].copy()
    selected_df = selected_df.sort_values("analysis_cutoff", kind="mergesort").reset_index(drop=True)
    return selected_df


def apply_step_refinement_rule(candidate_df: pd.DataFrame, rule_name: str) -> pd.DataFrame:
    """Apply one simple pre-entry refinement rule to Candidate v1 rows."""
    if rule_name not in RULE_DESCRIPTIONS:
        raise ValueError(f"Unknown Step Candidate refinement rule: {rule_name}")

    df = candidate_df.copy()
    if rule_name == "candidate_v1_base":
        return df

    risk_bucket = df.get("risk_points_bucket", pd.Series("", index=df.index)).fillna("").astype(str)
    rr_bucket = df.get("best_available_rr_bucket", pd.Series("", index=df.index)).fillna("").astype(str)
    entry_price_bucket = df.get("entry_price_bucket", pd.Series("", index=df.index)).fillna("").astype(str)
    target_plan_state = df.get("target_plan_state", pd.Series("", index=df.index)).fillna("").astype(str)
    scenario_state = df.get("scenario_state", pd.Series("", index=df.index)).fillna("").astype(str)

    if rule_name == "avoid_entry_price_low":
        mask = ~entry_price_bucket.eq("entry_price_low")
    elif rule_name == "avoid_risk_points_mid":
        mask = ~risk_bucket.eq("risk_points_mid")
    elif rule_name == "avoid_tp3_only":
        mask = ~target_plan_state.eq("tp3_available")
    elif rule_name == "require_best_available_rr_high":
        mask = rr_bucket.eq("best_available_rr_high")
    elif rule_name == "require_risk_points_high":
        mask = risk_bucket.eq("risk_points_high")
    elif rule_name == "avoid_entry_price_low_and_risk_mid":
        mask = ~entry_price_bucket.eq("entry_price_low") & ~risk_bucket.eq("risk_points_mid")
    elif rule_name == "keep_high_rr_or_risk_high":
        mask = rr_bucket.eq("best_available_rr_high") | risk_bucket.eq("risk_points_high")
    elif rule_name == "keep_high_rr_or_bullish_pullback":
        mask = rr_bucket.eq("best_available_rr_high") | scenario_state.eq("bullish_structure_bearish_entry_pullback")
    else:
        mask = pd.Series(True, index=df.index)

    return df.loc[mask].copy().reset_index(drop=True)


def _make_chronological_folds(selected_df: pd.DataFrame, fold_count: int) -> list[pd.DataFrame]:
    """Split selected rows into chronological folds with approximately equal trade counts."""
    if selected_df.empty:
        return []
    ordered_df = selected_df.sort_values("analysis_cutoff", kind="mergesort").reset_index(drop=True)
    fold_indices = np.array_split(np.arange(len(ordered_df)), max(1, fold_count))
    return [ordered_df.iloc[idx].copy() for idx in fold_indices if len(idx) > 0]


def summarize_rule(rule_df: pd.DataFrame, rule_name: str, fold_count: int, min_trades_per_fold: int) -> tuple[dict, pd.DataFrame]:
    """Summarize one rule overall and by chronological folds."""
    realized = pd.to_numeric(rule_df.get("managed_realized_r", pd.Series(dtype=float)), errors="coerce")
    kept_trades = int(realized.notna().sum())

    fold_rows: list[dict] = []
    for fold_id, fold_df in enumerate(_make_chronological_folds(rule_df, fold_count), start=1):
        fold_realized = pd.to_numeric(fold_df.get("managed_realized_r", pd.Series(dtype=float)), errors="coerce").dropna()
        fold_rows.append(
            {
                "rule_name": rule_name,
                "fold_id": fold_id,
                "fold_start_cutoff": fold_df["analysis_cutoff"].min() if "analysis_cutoff" in fold_df.columns else pd.NaT,
                "fold_end_cutoff": fold_df["analysis_cutoff"].max() if "analysis_cutoff" in fold_df.columns else pd.NaT,
                "kept_trades": int(len(fold_realized)),
                "positive_exit_rate": float((fold_realized > 0).mean()) if len(fold_realized) else 0.0,
                "loss_exit_rate": float((fold_realized < 0).mean()) if len(fold_realized) else 0.0,
                "average_managed_r": float(fold_realized.mean()) if len(fold_realized) else 0.0,
                "median_managed_r": float(fold_realized.median()) if len(fold_realized) else 0.0,
                "total_managed_r": float(fold_realized.sum()) if len(fold_realized) else 0.0,
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    positive_average_r_fold_rate = float((fold_df["average_managed_r"] > 0).mean()) if not fold_df.empty else 0.0
    positive_median_r_fold_rate = float((fold_df["median_managed_r"] > 0).mean()) if not fold_df.empty else 0.0
    folds_with_min_trades_rate = float((fold_df["kept_trades"] >= min_trades_per_fold).mean()) if not fold_df.empty else 0.0
    min_fold_average_r = float(fold_df["average_managed_r"].min()) if not fold_df.empty else 0.0

    summary = {
        "rule_name": rule_name,
        "rule_description": RULE_DESCRIPTIONS[rule_name],
        "kept_trades": kept_trades,
        "positive_exit_rate": float((realized > 0).mean()) if kept_trades else 0.0,
        "loss_exit_rate": float((realized < 0).mean()) if kept_trades else 0.0,
        "average_managed_r": float(realized.mean()) if kept_trades else 0.0,
        "median_managed_r": float(realized.median()) if kept_trades else 0.0,
        "total_managed_r": float(realized.sum()) if kept_trades else 0.0,
        "min_fold_average_r": min_fold_average_r,
        "positive_average_r_fold_rate": positive_average_r_fold_rate,
        "positive_median_r_fold_rate": positive_median_r_fold_rate,
        "folds_with_min_trades_rate": folds_with_min_trades_rate,
    }
    return summary, fold_df


def _assign_validation_state(summary: dict, min_total_trades: int) -> str:
    """Assign a conservative validation state label."""
    if summary["kept_trades"] < min_total_trades:
        return "insufficient_total_trades_do_not_lock"
    if summary["average_managed_r"] <= 0:
        return "not_profitable_do_not_lock"
    if summary["min_fold_average_r"] < 0 or summary["positive_average_r_fold_rate"] < 0.8:
        return "positive_but_fold_instability_validate_harder"
    return "candidate_refinement_validated_for_larger_step_test"


def validate_step_candidate_refinements(config: StepCandidateRefinementConfig) -> dict[str, pd.DataFrame]:
    """Validate simple Step Index candidate refinements from cache."""
    import time

    start = time.perf_counter()
    dataset_df = load_step_candidate_cache(config.cache_path)
    candidate_df = select_candidate_v1_base(dataset_df)

    summary_rows: list[dict] = []
    fold_frames: list[pd.DataFrame] = []
    preview_frames: list[pd.DataFrame] = []

    for rule_name in config.rule_names:
        rule_df = apply_step_refinement_rule(candidate_df, rule_name)
        summary, fold_df = summarize_rule(
            rule_df=rule_df,
            rule_name=rule_name,
            fold_count=config.fold_count,
            min_trades_per_fold=config.min_trades_per_fold,
        )
        summary["validation_state"] = _assign_validation_state(summary, config.min_total_trades)
        summary_rows.append(summary)
        fold_frames.append(fold_df)
        if not rule_df.empty:
            preview_cols = [
                col
                for col in [
                    "analysis_cutoff",
                    "scenario_state",
                    "event_state",
                    "entry_mode",
                    "entry_variant_name",
                    "risk_points",
                    "risk_points_bucket",
                    "entry_price_bucket",
                    "available_targets",
                    "target_plan_state",
                    "best_available_rr_bucket",
                    "managed_exit_state",
                    "managed_realized_r",
                ]
                if col in rule_df.columns
            ]
            preview = rule_df[preview_cols].head(15).copy()
            preview.insert(0, "rule_name", rule_name)
            preview_frames.append(preview)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["validation_state", "average_managed_r", "kept_trades"],
        ascending=[True, False, False],
    )
    fold_detail_df = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    preview_df = pd.concat(preview_frames, ignore_index=True) if preview_frames else pd.DataFrame()

    workload_df = pd.DataFrame(
        [
            {
                "cache_path": str(config.cache_path),
                "cache_rows": len(dataset_df),
                "candidate_v1_rows": len(candidate_df),
                "rule_count": len(config.rule_names),
                "fold_count": config.fold_count,
                "elapsed_seconds_total": round(time.perf_counter() - start, 3),
                "cache_design": "load_saved_cache_once_then_validate_step_candidate_refinements",
            }
        ]
    )

    return {
        "workload": workload_df,
        "summary": summary_df,
        "fold_detail": fold_detail_df,
        "selected_trade_preview": preview_df,
    }


def save_step_candidate_refinement_outputs(results: dict[str, pd.DataFrame], config: StepCandidateRefinementConfig) -> dict[str, Path]:
    """Save validation outputs to CSV files."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_tag = config.tag.lower().replace(" ", "_")
    paths: dict[str, Path] = {}
    for name, df in results.items():
        path = output_dir / f"{safe_tag}_{name}.csv"
        df.to_csv(path, index=False)
        paths[name] = path
    return paths
