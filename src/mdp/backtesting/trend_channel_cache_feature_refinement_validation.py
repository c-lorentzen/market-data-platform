"""Cached feature-refinement validation for trend-channel strategy research.

This module validates simple, pre-entry feature refinements from a saved
entry research cache. It is intentionally lightweight: the expensive strategy
pipeline is not rebuilt here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureRefinementResult:
    workload: pd.DataFrame
    ranking: pd.DataFrame
    fold_detail: pd.DataFrame
    selected_preview: pd.DataFrame


def load_entry_research_cache(cache_path: str | Path) -> pd.DataFrame:
    """Load a saved entry research cache CSV.

    Student-style note: this function only reads the cached file. It does not
    call MT5, MySQL, or the full strategy pipeline.
    """
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"Cache file not found: {path}")

    df = pd.read_csv(path)
    if "analysis_cutoff" in df.columns:
        df["analysis_cutoff"] = pd.to_datetime(df["analysis_cutoff"], errors="coerce")
    return df


def _truthy_series(series: pd.Series) -> pd.Series:
    """Convert flexible true/false cache values to a boolean mask."""
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y", "entered"})


def _entered_mask(df: pd.DataFrame) -> pd.Series:
    """Return rows that represent real entered trades."""
    if "entered_trade" in df.columns:
        return _truthy_series(df["entered_trade"])
    if "entry_status" in df.columns:
        return df["entry_status"].astype(str).str.lower().eq("entered")
    return pd.Series(True, index=df.index)


def _contains_text(df: pd.DataFrame, column: str, pattern: str) -> pd.Series:
    """Safely check whether a text column contains a pattern."""
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return df[column].astype(str).str.contains(pattern, case=False, na=False, regex=True)


def _equals_text(df: pd.DataFrame, column: str, value: str) -> pd.Series:
    """Safely check whether a text column equals a value."""
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return df[column].astype(str).str.lower().eq(value.lower())


def _available_targets_include_tp2_tp3(df: pd.DataFrame) -> pd.Series:
    """Identify rows where both TP2 and TP3 are available."""
    target_state_ok = _equals_text(df, "target_plan_state", "tp2_tp3_available")

    if "available_targets" not in df.columns:
        return target_state_ok

    targets = df["available_targets"].astype(str).str.replace(" ", "", regex=False)
    targets_ok = targets.str.contains("TP2", na=False) & targets.str.contains("TP3", na=False)
    return target_state_ok | targets_ok


def _best_available_rr(df: pd.DataFrame) -> pd.Series:
    """Return best available RR if present, otherwise estimate from TP RR columns."""
    for col in ["best_available_rr", "best_rr", "available_best_rr"]:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")

    rr_cols = [col for col in ["tp1_rr", "tp2_rr", "tp3_rr"] if col in df.columns]
    if rr_cols:
        return df[rr_cols].apply(pd.to_numeric, errors="coerce").max(axis=1)

    # Student-style note: NaN means this cache does not have RR fields.
    return pd.Series(np.nan, index=df.index)


def _bucket_numeric(series: pd.Series, prefix: str) -> pd.Series:
    """Create low/mid/high buckets using 33rd and 67th percentiles."""
    values = pd.to_numeric(series, errors="coerce")
    result = pd.Series(f"{prefix}_unknown", index=series.index, dtype="object")

    clean = values.dropna()
    if clean.empty or clean.nunique() < 3:
        return result

    q33 = clean.quantile(0.3333)
    q67 = clean.quantile(0.6667)

    result.loc[values <= q33] = f"{prefix}_low"
    result.loc[(values > q33) & (values < q67)] = f"{prefix}_mid"
    result.loc[values >= q67] = f"{prefix}_high"
    return result


def prepare_feature_refinement_dataset(cache_df: pd.DataFrame) -> pd.DataFrame:
    """Add helper fields used by feature-refinement rules."""
    df = cache_df.copy()
    df["_entered"] = _entered_mask(df)
    df["_is_retest_rejection"] = _contains_text(df, "event_state", "retest_rejection")
    df["_is_continuation"] = _contains_text(df, "event_state", "continuation_after_retest")
    df["_is_baseline_entry"] = _equals_text(df, "entry_variant_name", "baseline_current_close")
    df["_is_confirm2_entry"] = _equals_text(df, "entry_variant_name", "confirm_2_directional_close")
    df["_has_tp2_tp3"] = _available_targets_include_tp2_tp3(df)

    # Current best simple hybrid rule.
    df["_hybrid_reference"] = df["_entered"] & (
        (df["_is_retest_rejection"] & df["_is_baseline_entry"])
        | (df["_is_continuation"] & df["_is_confirm2_entry"])
    )

    # Current strongest refinement from earlier tests.
    df["_retest_requires_tp2_tp3"] = df["_hybrid_reference"] & (
        (~df["_is_retest_rejection"]) | df["_has_tp2_tp3"]
    )

    df["_best_available_rr"] = _best_available_rr(df)
    if "risk_points" in df.columns:
        df["risk_points_bucket"] = _bucket_numeric(df["risk_points"], "risk_points")
    else:
        df["risk_points_bucket"] = "risk_points_unknown"

    df["best_available_rr_bucket"] = _bucket_numeric(df["_best_available_rr"], "best_available_rr")
    return df


def _rule_masks(df: pd.DataFrame) -> dict[str, tuple[str, pd.Series]]:
    """Return available refinement rule masks."""
    reference = df["_retest_requires_tp2_tp3"]
    high_rr = df["best_available_rr_bucket"].eq("best_available_rr_high")
    mid_rr = df["best_available_rr_bucket"].eq("best_available_rr_mid")
    high_risk = df["risk_points_bucket"].eq("risk_points_high")
    mid_risk = df["risk_points_bucket"].eq("risk_points_mid")

    return {
        "retest_requires_tp2_tp3": (
            "Current refined candidate: retest-rejection requires TP2+TP3; continuation keeps 2-candle confirmation.",
            reference,
        ),
        "require_best_available_rr_high": (
            "Keep refined candidate trades only when best available RR is in the high bucket.",
            reference & high_rr,
        ),
        "avoid_best_available_rr_mid": (
            "Keep refined candidate trades except those with best available RR in the mid bucket.",
            reference & (~mid_rr),
        ),
        "require_risk_points_high": (
            "Keep refined candidate trades only when risk_points is in the high bucket.",
            reference & high_risk,
        ),
        "avoid_risk_points_mid": (
            "Keep refined candidate trades except those with risk_points in the mid bucket.",
            reference & (~mid_risk),
        ),
        "require_high_rr_or_high_risk": (
            "Keep refined candidate trades when high RR or high risk_points is present.",
            reference & (high_rr | high_risk),
        ),
        "require_high_rr_and_high_risk": (
            "Keep refined candidate trades only when both high RR and high risk_points are present.",
            reference & high_rr & high_risk,
        ),
    }


def _assign_chronological_folds(df: pd.DataFrame, fold_count: int) -> pd.DataFrame:
    """Assign fold ids based on chronological cutoff order."""
    output = df.copy()
    output["fold_id"] = np.nan
    output["fold_position"] = "unknown"

    cutoffs = output["analysis_cutoff"].dropna().sort_values().unique()
    if len(cutoffs) == 0:
        output["fold_id"] = 1
        output["fold_position"] = "single"
        return output

    folds = np.array_split(cutoffs, max(1, min(fold_count, len(cutoffs))))
    for idx, fold_cutoffs in enumerate(folds, start=1):
        if len(fold_cutoffs) == 0:
            continue
        mask = output["analysis_cutoff"].isin(fold_cutoffs)
        output.loc[mask, "fold_id"] = idx
        if idx == 1:
            position = "earliest"
        elif idx == len(folds):
            position = "latest"
        else:
            position = "middle"
        output.loc[mask, "fold_position"] = position

    output["fold_id"] = output["fold_id"].fillna(1).astype(int)
    return output


def _summarize_trades(df: pd.DataFrame) -> dict[str, float | int]:
    """Calculate performance metrics for selected trades."""
    if df.empty or "managed_realized_r" not in df.columns:
        return {
            "kept_trades": 0,
            "positive_exit_rate": 0.0,
            "loss_exit_rate": 0.0,
            "average_managed_r": 0.0,
            "median_managed_r": 0.0,
            "total_managed_r": 0.0,
            "partial_exit_rate": 0.0,
            "breakeven_or_lock_rate": 0.0,
        }

    r = pd.to_numeric(df["managed_realized_r"], errors="coerce").fillna(0.0)
    exit_state = df.get("managed_exit_state", pd.Series("", index=df.index)).astype(str)
    return {
        "kept_trades": int(len(df)),
        "positive_exit_rate": float((r > 0).mean()) if len(r) else 0.0,
        "loss_exit_rate": float((r < 0).mean()) if len(r) else 0.0,
        "average_managed_r": float(r.mean()) if len(r) else 0.0,
        "median_managed_r": float(r.median()) if len(r) else 0.0,
        "total_managed_r": float(r.sum()) if len(r) else 0.0,
        "partial_exit_rate": float(exit_state.str.contains("time_exit|profit|breakeven|lock|same_candle", case=False, na=False).mean()) if len(exit_state) else 0.0,
        "breakeven_or_lock_rate": float(exit_state.str.contains("breakeven|lock", case=False, na=False).mean()) if len(exit_state) else 0.0,
    }


def _score(summary: dict[str, float | int], baseline_attempts: int, min_total_trades: int) -> float:
    """Score refinements while penalizing tiny samples."""
    kept = float(summary["kept_trades"])
    coverage = kept / baseline_attempts if baseline_attempts else 0.0
    sample_factor = min(1.0, kept / max(1, min_total_trades))
    return float(
        (summary["average_managed_r"] * 100.0)
        + (summary["median_managed_r"] * 35.0)
        + (summary["positive_exit_rate"] * 20.0)
        + (coverage * 10.0)
        - ((1.0 - sample_factor) * 15.0)
    )


def _state(summary: dict[str, float | int], fold_summary: pd.DataFrame, min_total_trades: int, min_trades_per_fold: int) -> str:
    """Return honest validation state for a refinement."""
    kept = int(summary["kept_trades"])
    if kept == 0:
        return "reject_no_trades_selected"
    if kept < min_total_trades:
        return "insufficient_total_trades_do_not_lock"
    if summary["average_managed_r"] <= 0 or summary["median_managed_r"] <= 0:
        return "not_profitable_do_not_lock"

    if fold_summary.empty:
        return "positive_needs_fold_validation"

    positive_fold_rate = float((fold_summary["average_managed_r"] > 0).mean())
    folds_with_min = float((fold_summary["kept_trades"] >= min_trades_per_fold).mean())
    min_fold_avg = float(fold_summary["average_managed_r"].min())

    if positive_fold_rate >= 0.8 and folds_with_min >= 0.8 and min_fold_avg > -0.15:
        return "candidate_ready_for_larger_cross_market_validation"
    return "positive_but_fold_instability_validate_harder"


def validate_feature_refinements_from_cache(
    cache_path: str | Path,
    fold_count: int = 5,
    rule_names: Iterable[str] | None = None,
    min_total_trades: int = 20,
    min_trades_per_fold: int = 3,
) -> FeatureRefinementResult:
    """Validate feature-based refinements from an existing research cache."""
    import time

    start = time.perf_counter()
    raw_df = load_entry_research_cache(cache_path)
    df = prepare_feature_refinement_dataset(raw_df)
    df = _assign_chronological_folds(df, fold_count=fold_count)

    rule_map = _rule_masks(df)
    selected_rule_names = list(rule_names) if rule_names else list(rule_map.keys())

    baseline_attempts = int(df["_retest_requires_tp2_tp3"].sum())
    summaries: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    preview_frames: list[pd.DataFrame] = []

    for name in selected_rule_names:
        if name not in rule_map:
            continue
        description, mask = rule_map[name]
        trades = df.loc[mask].copy()
        summary = _summarize_trades(trades)

        fold_summary_rows = []
        for fold_id, fold_df in trades.groupby("fold_id"):
            fold_summary = _summarize_trades(fold_df)
            fold_row = {
                "rule_name": name,
                "fold_id": int(fold_id),
                "fold_position": str(fold_df["fold_position"].iloc[0]) if len(fold_df) else "unknown",
                "fold_start_cutoff": fold_df["analysis_cutoff"].min() if len(fold_df) else pd.NaT,
                "fold_end_cutoff": fold_df["analysis_cutoff"].max() if len(fold_df) else pd.NaT,
                **fold_summary,
            }
            fold_rows.append(fold_row)
            fold_summary_rows.append(fold_row)

        fold_summary_df = pd.DataFrame(fold_summary_rows)
        score = _score(summary, baseline_attempts=baseline_attempts, min_total_trades=min_total_trades)
        state = _state(summary, fold_summary_df, min_total_trades=min_total_trades, min_trades_per_fold=min_trades_per_fold)

        summaries.append({
            "rule_name": name,
            "rule_description": description,
            "baseline_attempts": baseline_attempts,
            "kept_trades": summary["kept_trades"],
            "coverage_rate": float(summary["kept_trades"] / baseline_attempts) if baseline_attempts else 0.0,
            "positive_exit_rate": summary["positive_exit_rate"],
            "loss_exit_rate": summary["loss_exit_rate"],
            "average_managed_r": summary["average_managed_r"],
            "median_managed_r": summary["median_managed_r"],
            "total_managed_r": summary["total_managed_r"],
            "partial_exit_rate": summary["partial_exit_rate"],
            "breakeven_or_lock_rate": summary["breakeven_or_lock_rate"],
            "min_fold_average_r": float(fold_summary_df["average_managed_r"].min()) if not fold_summary_df.empty else 0.0,
            "positive_average_r_fold_rate": float((fold_summary_df["average_managed_r"] > 0).mean()) if not fold_summary_df.empty else 0.0,
            "positive_median_r_fold_rate": float((fold_summary_df["median_managed_r"] > 0).mean()) if not fold_summary_df.empty else 0.0,
            "folds_with_min_trades_rate": float((fold_summary_df["kept_trades"] >= min_trades_per_fold).mean()) if not fold_summary_df.empty else 0.0,
            "refinement_score": score,
            "validation_state": state,
        })

        if trades.empty is False:
            preview_cols = [col for col in [
                "analysis_cutoff", "scenario_state", "event_state", "entry_mode", "entry_variant_name",
                "risk_points", "risk_points_bucket", "_best_available_rr", "best_available_rr_bucket",
                "available_targets", "target_plan_state", "managed_exit_state", "managed_realized_r",
            ] if col in trades.columns]
            preview = trades[preview_cols].copy().head(10)
            preview.insert(0, "rule_name", name)
            preview_frames.append(preview)

    ranking = pd.DataFrame(summaries).sort_values(
        by=["validation_state", "refinement_score", "average_managed_r"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    fold_detail = pd.DataFrame(fold_rows)
    selected_preview = pd.concat(preview_frames, ignore_index=True) if preview_frames else pd.DataFrame()

    elapsed = time.perf_counter() - start
    workload = pd.DataFrame([{
        "cache_path": str(cache_path),
        "cache_rows": int(len(raw_df)),
        "entered_rows": int(_entered_mask(raw_df).sum()) if len(raw_df) else 0,
        "baseline_attempts": baseline_attempts,
        "rule_count": len(selected_rule_names),
        "fold_count": fold_count,
        "min_total_trades": min_total_trades,
        "min_trades_per_fold": min_trades_per_fold,
        "cache_design": "load_saved_cache_once_then_validate_feature_refinements",
        "elapsed_seconds_total": round(elapsed, 3),
    }])

    return FeatureRefinementResult(
        workload=workload,
        ranking=ranking,
        fold_detail=fold_detail,
        selected_preview=selected_preview,
    )
