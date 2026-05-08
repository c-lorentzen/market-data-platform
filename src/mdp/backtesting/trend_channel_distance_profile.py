"""Distance-profile diagnostics for cached trend-channel research rows.

This module is intentionally cache-first: it reads an existing research cache CSV
and computes SL/TP distance statistics without rebuilding the strategy pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DistanceProfileConfig:
    """Configuration for cache-based distance diagnostics."""

    cache_path: str
    rule_name: str = "retest_requires_tp2_tp3"
    output_dir: str = "outputs/trend_channel_distance_profiles"
    tag: str = "step_distance_profile"


def _as_bool(series: pd.Series) -> pd.Series:
    """Convert mixed bool/string columns into a clean boolean mask."""
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def _safe_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric column when present; otherwise return NaNs."""
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[column], errors="coerce")


def _contains(df: pd.DataFrame, column: str, text: str) -> pd.Series:
    """Case-insensitive contains helper that safely handles missing columns."""
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return df[column].astype(str).str.lower().str.contains(text.lower(), na=False)


def load_entry_cache(cache_path: str) -> pd.DataFrame:
    """Load an entry research cache from CSV."""
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"Cache path does not exist: {cache_path}")
    return pd.read_csv(path)


def add_numeric_buckets_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Add rough low/mid/high buckets when older caches do not contain them."""
    result = df.copy()

    if "risk_points_bucket" not in result.columns and "risk_points" in result.columns:
        risk = _safe_numeric(result, "risk_points")
        q1 = risk.quantile(1 / 3)
        q2 = risk.quantile(2 / 3)
        result["risk_points_bucket"] = np.select(
            [risk <= q1, risk <= q2, risk > q2],
            ["risk_points_low", "risk_points_mid", "risk_points_high"],
            default="risk_points_unknown",
        )

    if "best_available_rr_bucket" not in result.columns and "best_available_rr" in result.columns:
        rr = _safe_numeric(result, "best_available_rr")
        q1 = rr.quantile(1 / 3)
        q2 = rr.quantile(2 / 3)
        result["best_available_rr_bucket"] = np.select(
            [rr <= q1, rr <= q2, rr > q2],
            ["best_available_rr_low", "best_available_rr_mid", "best_available_rr_high"],
            default="best_available_rr_unknown",
        )

    return result


def select_candidate_trades(dataset_df: pd.DataFrame, rule_name: str) -> pd.DataFrame:
    """Select trades using the current simple trend-channel candidate rules."""
    df = add_numeric_buckets_if_missing(dataset_df)

    entered_mask = _as_bool(df.get("entered_trade", pd.Series(True, index=df.index)))
    entered_df = df.loc[entered_mask].copy()

    retest_mask = _contains(entered_df, "entry_mode", "retest_rejection") & _contains(
        entered_df, "entry_variant_name", "baseline_current_close"
    )
    continuation_mask = _contains(entered_df, "entry_mode", "continuation_after_retest") & _contains(
        entered_df, "entry_variant_name", "confirm_2_directional_close"
    )

    tp2_tp3_mask = _contains(entered_df, "target_plan_state", "tp2_tp3") | _contains(
        entered_df, "target_plan_state", "tp1_tp2_tp3"
    )

    if rule_name == "hybrid_reference":
        selected = entered_df.loc[retest_mask | continuation_mask].copy()
    elif rule_name in {"retest_requires_tp2_tp3", "step_index_candidate_v1"}:
        selected = entered_df.loc[(retest_mask & tp2_tp3_mask) | continuation_mask].copy()
    elif rule_name == "retest_requires_tp2_tp3_post_target_guard":
        valid_target_mask = ~_contains(entered_df, "target_plan_state", "no_available_target")
        selected = entered_df.loc[((retest_mask & tp2_tp3_mask) | continuation_mask) & valid_target_mask].copy()
    elif rule_name == "require_high_rr_or_high_risk":
        base = entered_df.loc[(retest_mask & tp2_tp3_mask) | continuation_mask].copy()
        high_rr = _contains(base, "best_available_rr_bucket", "high")
        high_risk = _contains(base, "risk_points_bucket", "high")
        selected = base.loc[high_rr | high_risk].copy()
    else:
        raise ValueError(
            "Unknown rule_name. Supported: hybrid_reference, retest_requires_tp2_tp3, "
            "retest_requires_tp2_tp3_post_target_guard, require_high_rr_or_high_risk"
        )

    return selected.sort_values("analysis_cutoff") if "analysis_cutoff" in selected.columns else selected


def add_distance_columns(selected_df: pd.DataFrame) -> pd.DataFrame:
    """Add SL and TP distance columns in price points and ratios."""
    result = selected_df.copy()

    entry = _safe_numeric(result, "entry_price")
    stop = _safe_numeric(result, "stop_loss")
    risk_points = _safe_numeric(result, "risk_points")

    # Use the strategy risk column first because it already represents entry-to-SL distance.
    result["sl_distance_points"] = risk_points.where(risk_points.notna(), (entry - stop).abs())
    result["sl_distance_ratio"] = result["sl_distance_points"] / entry.replace(0, np.nan)
    result["sl_distance_pct"] = result["sl_distance_ratio"] * 100

    best_rr = _safe_numeric(result, "best_available_rr")
    result["best_target_distance_points"] = result["sl_distance_points"] * best_rr
    result["best_target_distance_ratio"] = result["best_target_distance_points"] / entry.replace(0, np.nan)
    result["best_target_distance_pct"] = result["best_target_distance_ratio"] * 100

    # If exact target prices exist, compute exact target distances too.
    for target_name in ["tp1", "tp2", "tp3"]:
        price_col = f"{target_name}_price"
        if price_col in result.columns:
            result[f"{target_name}_distance_points"] = (_safe_numeric(result, price_col) - entry).abs()
            result[f"{target_name}_distance_ratio"] = result[f"{target_name}_distance_points"] / entry.replace(0, np.nan)
            result[f"{target_name}_distance_pct"] = result[f"{target_name}_distance_ratio"] * 100

    return result


def _summarize_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Summarize numeric columns with trading-friendly distribution stats."""
    rows: List[Dict[str, float | str | int]] = []
    for column in columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "metric": column,
                "count": int(values.count()),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "p25": float(values.quantile(0.25)),
                "p75": float(values.quantile(0.75)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def build_distance_profile(config: DistanceProfileConfig) -> Dict[str, pd.DataFrame]:
    """Build the full distance-profile report from a saved cache."""
    dataset_df = load_entry_cache(config.cache_path)
    selected_df = select_candidate_trades(dataset_df, config.rule_name)
    selected_df = add_distance_columns(selected_df)

    distance_columns = [
        "sl_distance_points",
        "sl_distance_pct",
        "best_available_rr",
        "best_target_distance_points",
        "best_target_distance_pct",
        "managed_realized_r",
        "risk_points",
        "risk_points_ratio",
    ]
    for target_name in ["tp1", "tp2", "tp3"]:
        distance_columns.extend([f"{target_name}_distance_points", f"{target_name}_distance_pct"])

    summary_df = _summarize_numeric(selected_df, distance_columns)

    workload_df = pd.DataFrame(
        [
            {
                "cache_path": config.cache_path,
                "cache_rows": len(dataset_df),
                "selected_trades": len(selected_df),
                "rule_name": config.rule_name,
                "cache_design": "load_saved_cache_once_then_compute_sl_tp_distance_profile",
            }
        ]
    )

    group_columns = [
        "entry_mode",
        "entry_variant_name",
        "target_plan_state",
        "available_targets",
        "managed_exit_state",
    ]
    group_rows = []
    for column in group_columns:
        if column not in selected_df.columns:
            continue
        grouped = selected_df.groupby(column, dropna=False)
        for value, group in grouped:
            group_rows.append(
                {
                    "feature_name": column,
                    "feature_value": value,
                    "trades": len(group),
                    "avg_sl_distance_points": pd.to_numeric(group["sl_distance_points"], errors="coerce").mean(),
                    "avg_best_target_distance_points": pd.to_numeric(
                        group["best_target_distance_points"], errors="coerce"
                    ).mean(),
                    "avg_best_available_rr": pd.to_numeric(group.get("best_available_rr"), errors="coerce").mean()
                    if "best_available_rr" in group.columns
                    else np.nan,
                    "avg_managed_r": pd.to_numeric(group.get("managed_realized_r"), errors="coerce").mean()
                    if "managed_realized_r" in group.columns
                    else np.nan,
                }
            )
    group_summary_df = pd.DataFrame(group_rows)

    return {
        "workload": workload_df,
        "distance_summary": summary_df,
        "group_summary": group_summary_df,
        "selected_trades": selected_df,
    }


def save_distance_profile_outputs(report: Dict[str, pd.DataFrame], config: DistanceProfileConfig) -> Dict[str, str]:
    """Save report tables to CSV and return their paths."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_tag = config.tag.lower().replace(" ", "_")
    paths: Dict[str, str] = {}
    for name, df in report.items():
        output_path = output_dir / f"{safe_tag}_{config.rule_name}_{name}.csv"
        df.to_csv(output_path, index=False)
        paths[name] = str(output_path)
    return paths
