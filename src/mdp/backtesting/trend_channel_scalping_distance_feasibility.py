"""Cache-based scalping-distance feasibility diagnostics for trend-channel trades.

This module is intentionally lightweight: it reads an existing research cache CSV
and checks whether the current selected trades naturally fit smaller scalping
profiles such as 3-point SL / 6-point TP.

Important: fixed SL/TP simulations here are conservative proxy checks based on
stored max favorable/adverse movement. Exact candle-path order requires a future
candle replay module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScalpingFeasibilityConfig:
    """Configuration for cache-based scalping-distance diagnostics."""

    cache_path: str
    rule_name: str = "retest_requires_tp2_tp3"
    output_dir: str = "outputs/trend_channel_scalping_profiles"
    tag: str = "step_scalping_feasibility"
    small_sl_points: float = 3.0
    small_tp_points: float = 6.0
    medium_sl_points: float = 4.0
    medium_tp_points: float = 8.0
    wide_sl_points: float = 5.0
    wide_tp_points: float = 10.0


def _safe_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric series if a column exists; otherwise return NaNs."""
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[column], errors="coerce")


def _as_bool(series: pd.Series) -> pd.Series:
    """Convert mixed bool/string values into a clean boolean mask."""
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def _contains(df: pd.DataFrame, column: str, text: str) -> pd.Series:
    """Safely run a case-insensitive contains check."""
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return df[column].astype(str).str.lower().str.contains(text.lower(), na=False)


def _best_rr_column(df: pd.DataFrame) -> pd.Series:
    """Support both public and internal best-RR column names."""
    if "best_available_rr" in df.columns:
        return _safe_numeric(df, "best_available_rr")
    if "_best_available_rr" in df.columns:
        return _safe_numeric(df, "_best_available_rr")
    return pd.Series(np.nan, index=df.index)


def load_entry_cache(cache_path: str) -> pd.DataFrame:
    """Load the saved entry research cache."""
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"Cache path does not exist: {cache_path}")
    return pd.read_csv(path)


def add_buckets_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple low/mid/high buckets when older caches do not include them."""
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

    if "best_available_rr_bucket" not in result.columns:
        rr = _best_rr_column(result)
        q1 = rr.quantile(1 / 3)
        q2 = rr.quantile(2 / 3)
        result["best_available_rr_bucket"] = np.select(
            [rr <= q1, rr <= q2, rr > q2],
            ["best_available_rr_low", "best_available_rr_mid", "best_available_rr_high"],
            default="best_available_rr_unknown",
        )

    return result


def select_candidate_trades(dataset_df: pd.DataFrame, rule_name: str) -> pd.DataFrame:
    """Select trades using known trend-channel candidate rules."""
    df = add_buckets_if_missing(dataset_df)

    entered_mask = _as_bool(df.get("entered_trade", pd.Series(True, index=df.index)))
    entered_df = df.loc[entered_mask].copy()

    # Retest-rejection entries use the immediate/baseline close.
    retest_mask = _contains(entered_df, "entry_mode", "retest_rejection") & _contains(
        entered_df, "entry_variant_name", "baseline_current_close"
    )

    # Continuation entries require the 2-candle directional confirmation variant.
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

    if "analysis_cutoff" in selected.columns:
        selected = selected.sort_values("analysis_cutoff")
    return selected


def add_distance_columns(selected_df: pd.DataFrame) -> pd.DataFrame:
    """Add SL, target, MFE, and MAE distances in Step Index points."""
    result = selected_df.copy()
    entry = _safe_numeric(result, "entry_price")
    stop = _safe_numeric(result, "stop_loss")
    risk = _safe_numeric(result, "risk_points")

    result["sl_distance_points"] = risk.where(risk.notna(), (entry - stop).abs())
    result["best_available_rr"] = _best_rr_column(result)
    result["best_target_distance_points"] = result["sl_distance_points"] * result["best_available_rr"]

    # Convert stored max favorable/adverse R into points for fixed-distance proxy checks.
    result["max_favorable_points"] = _safe_numeric(result, "original_max_favorable_r") * result["sl_distance_points"]
    result["max_adverse_points"] = _safe_numeric(result, "original_max_adverse_r") * result["sl_distance_points"]

    return result


def _summarize_series(values: pd.Series) -> Dict[str, float | int]:
    """Return compact distribution stats for a numeric series."""
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {"count": 0, "mean": np.nan, "median": np.nan, "p25": np.nan, "p75": np.nan, "min": np.nan, "max": np.nan}
    return {
        "count": int(clean.count()),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "p25": float(clean.quantile(0.25)),
        "p75": float(clean.quantile(0.75)),
        "min": float(clean.min()),
        "max": float(clean.max()),
    }


def _profile_summary(df: pd.DataFrame, profile_name: str, mask: pd.Series) -> Dict[str, float | str | int]:
    """Summarize a distance-filter profile from the current selected trades."""
    subset = df.loc[mask].copy()
    realized_r = pd.to_numeric(subset.get("managed_realized_r", pd.Series(dtype=float)), errors="coerce")
    candles_to_exit = pd.to_numeric(subset.get("managed_candles_to_exit", pd.Series(dtype=float)), errors="coerce")

    return {
        "profile_name": profile_name,
        "kept_trades": int(len(subset)),
        "coverage_rate": float(len(subset) / len(df)) if len(df) else 0.0,
        "avg_sl_distance_points": float(pd.to_numeric(subset.get("sl_distance_points"), errors="coerce").mean()) if len(subset) else np.nan,
        "median_sl_distance_points": float(pd.to_numeric(subset.get("sl_distance_points"), errors="coerce").median()) if len(subset) else np.nan,
        "avg_best_target_distance_points": float(pd.to_numeric(subset.get("best_target_distance_points"), errors="coerce").mean()) if len(subset) else np.nan,
        "median_best_target_distance_points": float(pd.to_numeric(subset.get("best_target_distance_points"), errors="coerce").median()) if len(subset) else np.nan,
        "avg_best_available_rr": float(pd.to_numeric(subset.get("best_available_rr"), errors="coerce").mean()) if len(subset) else np.nan,
        "positive_exit_rate": float((realized_r > 0).mean()) if len(realized_r.dropna()) else np.nan,
        "average_managed_r": float(realized_r.mean()) if len(realized_r.dropna()) else np.nan,
        "median_managed_r": float(realized_r.median()) if len(realized_r.dropna()) else np.nan,
        "avg_managed_candles_to_exit": float(candles_to_exit.mean()) if len(candles_to_exit.dropna()) else np.nan,
        "median_managed_candles_to_exit": float(candles_to_exit.median()) if len(candles_to_exit.dropna()) else np.nan,
    }


def _simulate_fixed_sl_tp_proxy(df: pd.DataFrame, sl_points: float, tp_points: float) -> pd.DataFrame:
    """Conservatively estimate fixed SL/TP behavior from stored MFE/MAE.

    If both TP and SL were reachable during the horizon, the proxy assigns the
    stop first. This is conservative because the cache does not preserve the
    exact intrahorizon path ordering.
    """
    result = df.copy()
    mfe = pd.to_numeric(result["max_favorable_points"], errors="coerce")
    mae = pd.to_numeric(result["max_adverse_points"], errors="coerce")

    tp_hit = mfe >= tp_points
    sl_hit = mae >= sl_points

    fixed_r = pd.Series(0.0, index=result.index)
    state = pd.Series("no_fixed_terminal_within_horizon", index=result.index, dtype="object")

    # Conservative same-horizon handling: if both are possible, count stop first.
    both_hit = tp_hit & sl_hit
    fixed_r.loc[both_hit] = -1.0
    state.loc[both_hit] = "both_reached_count_stop_first_conservative"

    tp_only = tp_hit & ~sl_hit
    fixed_r.loc[tp_only] = tp_points / sl_points
    state.loc[tp_only] = "fixed_tp_hit"

    sl_only = sl_hit & ~tp_hit
    fixed_r.loc[sl_only] = -1.0
    state.loc[sl_only] = "fixed_sl_hit"

    result["fixed_sl_points"] = sl_points
    result["fixed_tp_points"] = tp_points
    result["fixed_rr"] = tp_points / sl_points
    result["fixed_proxy_state"] = state
    result["fixed_proxy_r"] = fixed_r
    result["fixed_tp_reached"] = tp_hit
    result["fixed_sl_reached"] = sl_hit
    return result


def _fixed_proxy_summary(df: pd.DataFrame, profile_name: str, sl_points: float, tp_points: float) -> Dict[str, float | str | int]:
    """Summarize a fixed SL/TP proxy profile."""
    simulated = _simulate_fixed_sl_tp_proxy(df, sl_points, tp_points)
    fixed_r = pd.to_numeric(simulated["fixed_proxy_r"], errors="coerce")
    return {
        "profile_name": profile_name,
        "fixed_sl_points": sl_points,
        "fixed_tp_points": tp_points,
        "fixed_rr": tp_points / sl_points,
        "trades_tested": int(len(simulated)),
        "tp_reached_rate": float(_as_bool(simulated["fixed_tp_reached"]).mean()) if len(simulated) else np.nan,
        "sl_reached_rate": float(_as_bool(simulated["fixed_sl_reached"]).mean()) if len(simulated) else np.nan,
        "average_fixed_proxy_r": float(fixed_r.mean()) if len(fixed_r.dropna()) else np.nan,
        "median_fixed_proxy_r": float(fixed_r.median()) if len(fixed_r.dropna()) else np.nan,
        "positive_fixed_proxy_rate": float((fixed_r > 0).mean()) if len(fixed_r.dropna()) else np.nan,
        "no_terminal_rate": float((simulated["fixed_proxy_state"] == "no_fixed_terminal_within_horizon").mean()) if len(simulated) else np.nan,
        "conservative_both_hit_rate": float((simulated["fixed_proxy_state"] == "both_reached_count_stop_first_conservative").mean()) if len(simulated) else np.nan,
    }


def build_scalping_feasibility_report(config: ScalpingFeasibilityConfig) -> Dict[str, pd.DataFrame]:
    """Build the full scalping feasibility report from a saved cache."""
    dataset_df = load_entry_cache(config.cache_path)
    selected_df = select_candidate_trades(dataset_df, config.rule_name)
    selected_df = add_distance_columns(selected_df)

    sl = selected_df["sl_distance_points"]
    target = selected_df["best_target_distance_points"]
    candles = _safe_numeric(selected_df, "managed_candles_to_exit")

    profile_rows = [
        _profile_summary(selected_df, "current_candidate_all_trades", pd.Series(True, index=selected_df.index)),
        _profile_summary(selected_df, "current_logic_sl_le_3", sl <= 3.0),
        _profile_summary(selected_df, "current_logic_sl_le_4", sl <= 4.0),
        _profile_summary(selected_df, "current_logic_sl_2_to_4", (sl >= 2.0) & (sl <= 4.0)),
        _profile_summary(selected_df, "current_logic_target_4_to_12", (target >= 4.0) & (target <= 12.0)),
        _profile_summary(selected_df, "current_logic_sl_le_4_target_4_to_12", (sl <= 4.0) & (target >= 4.0) & (target <= 12.0)),
        _profile_summary(selected_df, "current_logic_exit_le_10_candles", candles <= 10.0),
        _profile_summary(selected_df, "current_logic_sl_le_4_or_exit_le_10", (sl <= 4.0) | (candles <= 10.0)),
    ]
    profile_summary_df = pd.DataFrame(profile_rows)

    fixed_rows = [
        _fixed_proxy_summary(selected_df, "fixed_3sl_6tp_proxy_conservative", config.small_sl_points, config.small_tp_points),
        _fixed_proxy_summary(selected_df, "fixed_4sl_8tp_proxy_conservative", config.medium_sl_points, config.medium_tp_points),
        _fixed_proxy_summary(selected_df, "fixed_5sl_10tp_proxy_conservative", config.wide_sl_points, config.wide_tp_points),
    ]
    fixed_proxy_summary_df = pd.DataFrame(fixed_rows)

    distance_distribution_rows: List[Dict[str, float | str | int]] = []
    for column in [
        "sl_distance_points",
        "best_target_distance_points",
        "best_available_rr",
        "max_favorable_points",
        "max_adverse_points",
        "managed_realized_r",
        "managed_candles_to_exit",
    ]:
        stats = _summarize_series(selected_df[column]) if column in selected_df.columns else {}
        if stats:
            distance_distribution_rows.append({"metric": column, **stats})
    distance_distribution_df = pd.DataFrame(distance_distribution_rows)

    workload_df = pd.DataFrame(
        [
            {
                "cache_path": config.cache_path,
                "cache_rows": len(dataset_df),
                "selected_trades": len(selected_df),
                "rule_name": config.rule_name,
                "cache_design": "load_saved_cache_once_then_test_scalping_distance_feasibility",
            }
        ]
    )

    preview_columns = [
        "analysis_cutoff",
        "entry_mode",
        "entry_variant_name",
        "entry_price",
        "stop_loss",
        "sl_distance_points",
        "best_available_rr",
        "best_target_distance_points",
        "max_favorable_points",
        "max_adverse_points",
        "managed_candles_to_exit",
        "managed_exit_state",
        "managed_realized_r",
    ]
    existing_preview_columns = [column for column in preview_columns if column in selected_df.columns]

    return {
        "workload": workload_df,
        "distance_distribution": distance_distribution_df,
        "scalping_profile_summary": profile_summary_df,
        "fixed_sl_tp_proxy_summary": fixed_proxy_summary_df,
        "selected_trade_preview": selected_df[existing_preview_columns].copy(),
    }


def save_scalping_feasibility_outputs(
    report: Dict[str, pd.DataFrame], config: ScalpingFeasibilityConfig
) -> Dict[str, str]:
    """Save all report tables to CSV and return paths."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_tag = config.tag.lower().replace(" ", "_")
    paths: Dict[str, str] = {}
    for name, df in report.items():
        output_path = output_dir / f"{safe_tag}_{config.rule_name}_{name}.csv"
        df.to_csv(output_path, index=False)
        paths[name] = str(output_path)
    return paths
