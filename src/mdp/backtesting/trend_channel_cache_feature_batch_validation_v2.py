"""Cached cross-market feature-rule validation for trend-channel strategy candidates.

This module is intentionally cache-first for efficiency. It does not rebuild the
trend-channel pipeline. It reads one or more saved entry research cache CSV files,
applies market-agnostic candidate/refinement rules, then compares results by cache
and chronological folds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_RULE_NAMES = [
    "retest_requires_tp2_tp3",
    "require_high_rr_or_high_risk",
    "require_best_available_rr_high",
    "avoid_best_available_rr_mid",
    "retest_requires_tp2_tp3_post_target_guard",
    "high_rr_or_high_risk_post_target_guard",
    "avoid_unknown_rr_post_target_guard",
    "hybrid_reference",
]


@dataclass(frozen=True)
class FeatureBatchValidationConfig:
    cache_paths: list[str]
    fold_count: int = 5
    rule_names: list[str] | None = None
    min_total_trades: int = 20
    min_trades_per_fold: int = 3
    preview_rows: int = 40


def _safe_str_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype="string")
    return df[column].fillna("").astype(str)


def _safe_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def _has_valid_post_refinement_target(df: pd.DataFrame) -> pd.Series:
    """Return True when the selected entry still has an actionable target set.

    Some entry variants can invalidate the original target plan after the entry is
    delayed or refined. Those rows usually show target_plan_state like
    'no_available_target_after_entry_refinement' or missing available_targets.
    This guard is market-agnostic and prevents the rule from trading setups whose
    target logic no longer exists after the chosen entry variant.
    """
    target_state = _safe_str_series(df, "target_plan_state").str.lower()
    targets = _safe_str_series(df, "available_targets").str.lower()

    invalid_state = (
        target_state.eq("")
        | target_state.str.contains("no_available_target", na=False)
        | target_state.str.contains("none", na=False)
        | target_state.str.contains("nan", na=False)
    )
    invalid_targets = (
        targets.eq("")
        | targets.eq("nan")
        | targets.eq("none")
    )
    return ~(invalid_state | invalid_targets)


def _hybrid_reference_mask(df: pd.DataFrame) -> pd.Series:
    """Base simple candidate.

    - Retest-rejection entries use the baseline current close.
    - Continuation-after-retest entries require two-candle directional close.
    """
    entry_mode = _safe_str_series(df, "entry_mode")
    variant = _safe_str_series(df, "entry_variant_name")

    retest_now = entry_mode.eq("retest_rejection_close") & variant.eq("baseline_current_close")
    continuation_confirmed = (
        entry_mode.eq("continuation_after_retest_close")
        & variant.eq("confirm_2_directional_close")
    )
    return retest_now | continuation_confirmed


def _retest_requires_tp2_tp3_mask(df: pd.DataFrame) -> pd.Series:
    """Refined candidate used in prior tests.

    Retest-rejection entries require TP2+TP3 availability. Continuation entries keep
    two-candle confirmation logic.
    """
    hybrid = _hybrid_reference_mask(df)
    entry_mode = _safe_str_series(df, "entry_mode")
    target_state = _safe_str_series(df, "target_plan_state").str.lower()
    targets = _safe_str_series(df, "available_targets").str.upper()

    is_retest = entry_mode.eq("retest_rejection_close")
    has_tp2_tp3 = target_state.str.contains("tp2_tp3", na=False) | targets.str.contains("TP2,TP3", na=False)
    return hybrid & ((~is_retest) | has_tp2_tp3)


def _rule_descriptions() -> dict[str, str]:
    return {
        "hybrid_reference": "Retest-rejection enters now; continuation entries require two-candle directional confirmation.",
        "retest_requires_tp2_tp3": "Retest-rejection entries require TP2 and TP3 availability; continuation entries keep confirmation logic.",
        "require_high_rr_or_high_risk": "Keep refined candidate trades when best_available_rr or risk_points bucket is high.",
        "require_best_available_rr_high": "Keep refined candidate trades only when best_available_rr bucket is high.",
        "avoid_best_available_rr_mid": "Keep refined candidate trades except those with best_available_rr bucket equal to mid.",
        "retest_requires_tp2_tp3_post_target_guard": "Base refined candidate plus guard: selected entry must still have valid post-refinement targets.",
        "high_rr_or_high_risk_post_target_guard": "High RR/high risk candidate plus guard: selected entry must still have valid post-refinement targets.",
        "avoid_unknown_rr_post_target_guard": "Base refined candidate excluding unknown RR and entries with no valid post-refinement targets.",
    }


def apply_feature_rule(df: pd.DataFrame, rule_name: str) -> pd.DataFrame:
    """Apply a feature refinement rule and return selected rows."""
    base = _retest_requires_tp2_tp3_mask(df)
    hybrid = _hybrid_reference_mask(df)
    rr_bucket = _safe_str_series(df, "best_available_rr_bucket")
    risk_bucket = _safe_str_series(df, "risk_points_bucket")
    valid_targets = _has_valid_post_refinement_target(df)

    if rule_name == "hybrid_reference":
        mask = hybrid
    elif rule_name == "retest_requires_tp2_tp3":
        mask = base
    elif rule_name == "require_high_rr_or_high_risk":
        mask = base & (rr_bucket.eq("best_available_rr_high") | risk_bucket.eq("risk_points_high"))
    elif rule_name == "require_best_available_rr_high":
        mask = base & rr_bucket.eq("best_available_rr_high")
    elif rule_name == "avoid_best_available_rr_mid":
        mask = base & ~rr_bucket.eq("best_available_rr_mid")
    elif rule_name == "retest_requires_tp2_tp3_post_target_guard":
        mask = base & valid_targets
    elif rule_name == "high_rr_or_high_risk_post_target_guard":
        mask = base & (rr_bucket.eq("best_available_rr_high") | risk_bucket.eq("risk_points_high")) & valid_targets
    elif rule_name == "avoid_unknown_rr_post_target_guard":
        mask = base & valid_targets & ~rr_bucket.eq("best_available_rr_unknown")
    else:
        raise ValueError(f"Unknown feature refinement rule: {rule_name}")

    selected = df.loc[mask].copy()
    selected["rule_name"] = rule_name
    selected["rule_description"] = _rule_descriptions().get(rule_name, "")
    return selected


def _assign_folds(selected_df: pd.DataFrame, fold_count: int) -> pd.DataFrame:
    if selected_df.empty:
        selected_df = selected_df.copy()
        selected_df["fold_id"] = pd.Series(dtype="int64")
        selected_df["fold_position"] = pd.Series(dtype="object")
        return selected_df

    df = selected_df.copy()
    df["analysis_cutoff"] = pd.to_datetime(df["analysis_cutoff"], errors="coerce")
    df = df.sort_values("analysis_cutoff").reset_index(drop=True)

    actual_folds = max(1, min(fold_count, len(df)))
    fold_ids = np.array_split(np.arange(len(df)), actual_folds)
    df["fold_id"] = 0
    for idx, row_positions in enumerate(fold_ids, start=1):
        if len(row_positions) > 0:
            df.loc[row_positions, "fold_id"] = idx

    def position_name(fid: int) -> str:
        if actual_folds == 1:
            return "single"
        if fid == 1:
            return "earliest"
        if fid == actual_folds:
            return "latest"
        return "middle"

    df["fold_position"] = df["fold_id"].apply(position_name)
    return df


def _summarize_rows(df: pd.DataFrame) -> dict[str, float | int | str]:
    if df.empty:
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

    r = _safe_numeric_series(df, "managed_realized_r")
    exit_state = _safe_str_series(df, "managed_exit_state")
    kept = len(df)
    positive = (r > 0).mean()
    loss = (r < 0).mean()
    partial = exit_state.str.contains("partial", case=False, na=False).mean()
    be_lock = (
        exit_state.str.contains("breakeven", case=False, na=False)
        | exit_state.str.contains("profit_locked", case=False, na=False)
    ).mean()
    return {
        "kept_trades": int(kept),
        "positive_exit_rate": float(positive),
        "loss_exit_rate": float(loss),
        "average_managed_r": float(r.mean()),
        "median_managed_r": float(r.median()),
        "total_managed_r": float(r.sum()),
        "partial_exit_rate": float(partial),
        "breakeven_or_lock_rate": float(be_lock),
    }


def _fold_summary(selected_df: pd.DataFrame, fold_count: int) -> pd.DataFrame:
    folded = _assign_folds(selected_df, fold_count)
    if folded.empty:
        return pd.DataFrame()

    rows = []
    for fold_id, group in folded.groupby("fold_id", sort=True):
        summary = _summarize_rows(group)
        rows.append(
            {
                "fold_id": int(fold_id),
                "fold_position": group["fold_position"].iloc[0],
                "fold_start_cutoff": group["analysis_cutoff"].min(),
                "fold_end_cutoff": group["analysis_cutoff"].max(),
                **summary,
            }
        )
    return pd.DataFrame(rows)


def validate_cache_file(
    cache_path: str,
    rule_names: Iterable[str],
    fold_count: int,
    min_total_trades: int,
    min_trades_per_fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = Path(cache_path)
    df = pd.read_csv(path)
    cache_label = path.stem

    per_rule_rows = []
    fold_frames = []
    selected_frames = []
    baseline_attempts = int(df.get("analysis_cutoff", pd.Series(dtype=object)).nunique())

    for rule_name in rule_names:
        selected = apply_feature_rule(df, rule_name)
        selected["cache_label"] = cache_label
        selected["cache_path"] = str(path)
        selected_frames.append(selected)

        fold_df = _fold_summary(selected, fold_count)
        if not fold_df.empty:
            fold_df["rule_name"] = rule_name
            fold_df["rule_description"] = _rule_descriptions().get(rule_name, "")
            fold_df["cache_label"] = cache_label
            fold_df["cache_path"] = str(path)
            fold_frames.append(fold_df)

        summary = _summarize_rows(selected)
        if fold_df.empty:
            min_fold_avg = 0.0
            positive_avg_fold_rate = 0.0
            positive_median_fold_rate = 0.0
            folds_with_min_trades_rate = 0.0
        else:
            min_fold_avg = float(fold_df["average_managed_r"].min())
            positive_avg_fold_rate = float((fold_df["average_managed_r"] > 0).mean())
            positive_median_fold_rate = float((fold_df["median_managed_r"] > 0).mean())
            folds_with_min_trades_rate = float((fold_df["kept_trades"] >= min_trades_per_fold).mean())

        if summary["kept_trades"] < min_total_trades:
            state = "insufficient_total_trades_do_not_lock"
        elif summary["average_managed_r"] <= 0:
            state = "not_profitable_do_not_lock"
        elif min_fold_avg < 0:
            state = "positive_but_fold_instability_validate_harder"
        elif positive_avg_fold_rate < 1:
            state = "positive_but_fold_instability_validate_harder"
        else:
            state = "candidate_ready_for_bigger_market_validation"

        score = (
            summary["average_managed_r"] * 100
            + summary["median_managed_r"] * 35
            + summary["positive_exit_rate"] * 25
            - summary["loss_exit_rate"] * 10
            + positive_avg_fold_rate * 10
            + positive_median_fold_rate * 5
        )

        per_rule_rows.append(
            {
                "cache_label": cache_label,
                "cache_path": str(path),
                "rule_name": rule_name,
                "rule_description": _rule_descriptions().get(rule_name, ""),
                "baseline_attempts": baseline_attempts,
                **summary,
                "min_fold_average_r": min_fold_avg,
                "positive_average_r_fold_rate": positive_avg_fold_rate,
                "positive_median_r_fold_rate": positive_median_fold_rate,
                "folds_with_min_trades_rate": folds_with_min_trades_rate,
                "refinement_score": float(score),
                "validation_state": state,
            }
        )

    summary_df = pd.DataFrame(per_rule_rows)
    fold_all_df = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    selected_all_df = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    return summary_df, fold_all_df, selected_all_df


def _aggregate(summary_df: pd.DataFrame, min_total_trades: int) -> pd.DataFrame:
    rows = []
    for rule_name, group in summary_df.groupby("rule_name", sort=False):
        total_kept = int(group["kept_trades"].sum())
        mean_avg = float(group["average_managed_r"].mean())
        median_avg = float(group["average_managed_r"].median())
        min_cache_avg = float(group["average_managed_r"].min())
        cache_positive_avg_rate = float((group["average_managed_r"] > 0).mean())
        cache_positive_median_rate = float((group["median_managed_r"] > 0).mean())
        mean_positive_exit_rate = float(group["positive_exit_rate"].mean())
        mean_loss_exit_rate = float(group["loss_exit_rate"].mean())
        mean_median_r = float(group["median_managed_r"].mean())
        mean_fold_rate = float(group["positive_average_r_fold_rate"].mean())
        mean_min_trade_fold_rate = float(group["folds_with_min_trades_rate"].mean())

        if total_kept < min_total_trades:
            state = "insufficient_total_trades_do_not_lock"
        elif mean_avg <= 0:
            state = "not_profitable_do_not_lock"
        elif cache_positive_avg_rate < 1:
            state = "positive_but_cache_instability_validate_harder"
        elif mean_fold_rate < 1:
            state = "positive_but_fold_instability_validate_harder"
        else:
            state = "candidate_ready_for_bigger_market_validation"

        aggregate_score = (
            mean_avg * 100
            + mean_median_r * 35
            + mean_positive_exit_rate * 25
            - mean_loss_exit_rate * 10
            + cache_positive_avg_rate * 15
            + mean_fold_rate * 10
        )

        rows.append(
            {
                "rule_name": rule_name,
                "cache_count": int(group["cache_label"].nunique()),
                "total_kept_trades": total_kept,
                "mean_kept_trades": float(group["kept_trades"].mean()),
                "mean_positive_exit_rate": mean_positive_exit_rate,
                "mean_loss_exit_rate": mean_loss_exit_rate,
                "mean_average_managed_r": mean_avg,
                "median_average_managed_r": median_avg,
                "min_cache_average_r": min_cache_avg,
                "max_cache_average_r": float(group["average_managed_r"].max()),
                "mean_median_managed_r": mean_median_r,
                "cache_positive_average_r_rate": cache_positive_avg_rate,
                "cache_positive_median_r_rate": cache_positive_median_rate,
                "mean_positive_average_r_fold_rate": mean_fold_rate,
                "mean_folds_with_min_trades_rate": mean_min_trade_fold_rate,
                "aggregate_state": state,
                "aggregate_score": float(aggregate_score),
            }
        )
    return pd.DataFrame(rows).sort_values("aggregate_score", ascending=False).reset_index(drop=True)


def validate_feature_rules_across_caches_v2(config: FeatureBatchValidationConfig) -> dict[str, pd.DataFrame]:
    rule_names = config.rule_names or DEFAULT_RULE_NAMES
    summary_frames = []
    fold_frames = []
    selected_frames = []
    workload_rows = [
        {
            "cache_file_count": len(config.cache_paths),
            "rule_count": len(rule_names),
            "fold_count": config.fold_count,
            "min_total_trades": config.min_total_trades,
            "min_trades_per_fold": config.min_trades_per_fold,
            "cache_design": "validate_feature_rules_across_saved_caches_with_post_target_guards",
        }
    ]

    for cache_path in config.cache_paths:
        summary_df, fold_df, selected_df = validate_cache_file(
            cache_path=cache_path,
            rule_names=rule_names,
            fold_count=config.fold_count,
            min_total_trades=config.min_total_trades,
            min_trades_per_fold=config.min_trades_per_fold,
        )
        summary_frames.append(summary_df)
        if not fold_df.empty:
            fold_frames.append(fold_df)
        if not selected_df.empty:
            selected_frames.append(selected_df)
        workload_rows.append(
            {
                "cache_label": Path(cache_path).stem,
                "cache_path": cache_path,
                "cache_rows": int(pd.read_csv(cache_path, nrows=0).shape[0]) if False else int(pd.read_csv(cache_path).shape[0]),
                "rule_count": len(rule_names),
                "fold_count": config.fold_count,
                "min_total_trades": config.min_total_trades,
                "min_trades_per_fold": config.min_trades_per_fold,
                "cache_design": "load_saved_cache_once_then_validate_feature_rules_v2",
            }
        )

    per_cache = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    fold_detail = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    selected = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    aggregate = _aggregate(per_cache, config.min_total_trades) if not per_cache.empty else pd.DataFrame()
    workload = pd.DataFrame(workload_rows)

    return {
        "workload": workload,
        "aggregate": aggregate,
        "per_cache": per_cache.sort_values(["cache_label", "refinement_score"], ascending=[True, False]).reset_index(drop=True) if not per_cache.empty else per_cache,
        "fold_detail": fold_detail,
        "selected_preview": selected.head(config.preview_rows) if not selected.empty else selected,
    }
