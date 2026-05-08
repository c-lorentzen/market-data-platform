"""Trade-frequency diagnostics for cached trend-channel candidate datasets.

This module is intentionally cache-based. It loads an already-built research cache
and measures how often a candidate rule appears over time without rebuilding the
full strategy pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class TradeFrequencyProfileConfig:
    """Configuration for cache-based trade-frequency profiling."""

    cache_path: str
    rule_name: str = "candidate_v1_base"
    tag: str = "step_candidate_frequency_profile"
    output_dir: str = "outputs/trend_channel_trade_frequency_profiles"


RULE_DESCRIPTIONS: dict[str, str] = {
    "candidate_v1_base": (
        "Step Index Candidate v1: retest-rejection entries require TP2+TP3; "
        "continuation entries require two-candle directional confirmation."
    ),
    "avoid_tp3_only": (
        "Candidate v1.1 hypothesis: same base rule, but skip entries where only TP3 is available."
    ),
    "retest_rejection_only": "Only retest-rejection rows from the base candidate.",
    "continuation_confirm2_only": "Only continuation rows that passed 2-candle directional confirmation.",
}


def _as_bool(series: pd.Series) -> pd.Series:
    """Convert common truthy values from CSVs into boolean values."""

    if series.dtype == bool:
        return series.fillna(False)

    text_values = series.astype(str).str.lower().str.strip()
    return text_values.isin({"true", "1", "yes", "y", "entered"})


def _contains_any(series: pd.Series, values: Iterable[str]) -> pd.Series:
    """Return True when a string column contains any target text."""

    text_values = series.fillna("").astype(str)
    mask = pd.Series(False, index=series.index)
    for value in values:
        mask = mask | text_values.str.contains(value, regex=False)
    return mask


def load_cache(cache_path: str | Path) -> pd.DataFrame:
    """Load a research cache and normalize timestamp columns."""

    dataset_df = pd.read_csv(cache_path)
    if "analysis_cutoff" not in dataset_df.columns:
        raise ValueError("Cache must contain an 'analysis_cutoff' column.")

    dataset_df = dataset_df.copy()
    dataset_df["analysis_cutoff"] = pd.to_datetime(dataset_df["analysis_cutoff"])
    return dataset_df


def apply_step_candidate_rule(dataset_df: pd.DataFrame, rule_name: str) -> pd.DataFrame:
    """Apply a Step Index candidate rule to an entry research cache.

    The rules use only pre-entry fields stored in the cache. This keeps the
    diagnostic compatible with honest backtesting and future ML labels.
    """

    df = dataset_df.copy()

    # Only evaluate rows that represent actually-entered entry variants.
    if "entered_trade" in df.columns:
        entered_mask = _as_bool(df["entered_trade"])
    elif "entry_status" in df.columns:
        entered_mask = df["entry_status"].astype(str).str.lower().eq("entered")
    else:
        entered_mask = pd.Series(True, index=df.index)

    entry_mode = df.get("entry_mode", pd.Series("", index=df.index)).fillna("").astype(str)
    entry_variant_name = df.get("entry_variant_name", pd.Series("", index=df.index)).fillna("").astype(str)
    target_plan_state = df.get("target_plan_state", pd.Series("", index=df.index)).fillna("").astype(str)
    available_targets = df.get("available_targets", pd.Series("", index=df.index)).fillna("").astype(str)

    # Retest-rejection rows need enough available target structure.
    retest_mask = entry_mode.eq("retest_rejection_close")
    retest_has_tp2_tp3 = (
        target_plan_state.isin({"tp2_tp3_available", "tp1_tp2_tp3_available"})
        | _contains_any(available_targets, ["TP2,TP3", "TP1,TP2,TP3"])
    )
    valid_retest = retest_mask & retest_has_tp2_tp3

    # Continuation rows must use the delayed/confirmed entry variant.
    continuation_mask = entry_mode.eq("continuation_after_retest_close")
    valid_continuation = continuation_mask & entry_variant_name.eq("confirm_2_directional_close")

    base_mask = entered_mask & (valid_retest | valid_continuation)

    if rule_name == "candidate_v1_base":
        selected_mask = base_mask
    elif rule_name == "avoid_tp3_only":
        selected_mask = base_mask & ~target_plan_state.eq("tp3_available") & ~available_targets.eq("TP3")
    elif rule_name == "retest_rejection_only":
        selected_mask = base_mask & retest_mask
    elif rule_name == "continuation_confirm2_only":
        selected_mask = base_mask & continuation_mask
    else:
        raise ValueError(f"Unknown rule_name: {rule_name}")

    selected_df = df.loc[selected_mask].copy()
    selected_df = selected_df.sort_values("analysis_cutoff").reset_index(drop=True)
    selected_df["rule_name"] = rule_name
    selected_df["rule_description"] = RULE_DESCRIPTIONS.get(rule_name, rule_name)
    return selected_df


def build_frequency_summary(selected_df: pd.DataFrame, dataset_df: pd.DataFrame, rule_name: str) -> pd.DataFrame:
    """Compute high-level trade-frequency and outcome metrics."""

    if selected_df.empty:
        return pd.DataFrame(
            [
                {
                    "rule_name": rule_name,
                    "selected_trades": 0,
                    "calendar_days_covered": 0,
                    "elapsed_days_covered": 0.0,
                    "avg_trades_per_calendar_day": 0.0,
                    "avg_trades_per_elapsed_day": 0.0,
                    "avg_hours_between_trades": None,
                    "median_hours_between_trades": None,
                    "positive_exit_rate": None,
                    "average_managed_r": None,
                    "median_managed_r": None,
                }
            ]
        )

    start_time = selected_df["analysis_cutoff"].min()
    end_time = selected_df["analysis_cutoff"].max()
    elapsed_days = max((end_time - start_time).total_seconds() / 86400.0, 1.0 / 1440.0)
    calendar_days = selected_df["analysis_cutoff"].dt.date.nunique()

    gaps_hours = selected_df["analysis_cutoff"].diff().dropna().dt.total_seconds() / 3600.0
    managed_r = pd.to_numeric(selected_df.get("managed_realized_r"), errors="coerce")

    return pd.DataFrame(
        [
            {
                "rule_name": rule_name,
                "rule_description": RULE_DESCRIPTIONS.get(rule_name, rule_name),
                "cache_rows": len(dataset_df),
                "selected_trades": len(selected_df),
                "first_trade_time": start_time,
                "last_trade_time": end_time,
                "calendar_days_covered": calendar_days,
                "elapsed_days_covered": round(elapsed_days, 4),
                "avg_trades_per_calendar_day": round(len(selected_df) / max(calendar_days, 1), 4),
                "avg_trades_per_elapsed_day": round(len(selected_df) / elapsed_days, 4),
                "avg_hours_between_trades": round(float(gaps_hours.mean()), 4) if not gaps_hours.empty else None,
                "median_hours_between_trades": round(float(gaps_hours.median()), 4) if not gaps_hours.empty else None,
                "positive_exit_rate": round(float((managed_r > 0).mean()), 6) if managed_r.notna().any() else None,
                "loss_exit_rate": round(float((managed_r < 0).mean()), 6) if managed_r.notna().any() else None,
                "average_managed_r": round(float(managed_r.mean()), 6) if managed_r.notna().any() else None,
                "median_managed_r": round(float(managed_r.median()), 6) if managed_r.notna().any() else None,
                "total_managed_r": round(float(managed_r.sum()), 6) if managed_r.notna().any() else None,
            }
        ]
    )


def build_daily_summary(selected_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize selected trades per calendar day."""

    if selected_df.empty:
        return pd.DataFrame()

    df = selected_df.copy()
    df["trade_date"] = df["analysis_cutoff"].dt.date
    managed_r = pd.to_numeric(df.get("managed_realized_r"), errors="coerce")
    df["managed_realized_r_numeric"] = managed_r

    return (
        df.groupby("trade_date", dropna=False)
        .agg(
            trades=("analysis_cutoff", "count"),
            positive_exit_rate=("managed_realized_r_numeric", lambda x: float((x > 0).mean()) if len(x) else 0.0),
            average_managed_r=("managed_realized_r_numeric", "mean"),
            median_managed_r=("managed_realized_r_numeric", "median"),
            total_managed_r=("managed_realized_r_numeric", "sum"),
            first_trade_time=("analysis_cutoff", "min"),
            last_trade_time=("analysis_cutoff", "max"),
        )
        .reset_index()
        .sort_values("trade_date")
    )


def build_hourly_summary(selected_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize selected trades by hour of day."""

    if selected_df.empty:
        return pd.DataFrame()

    df = selected_df.copy()
    df["hour_utc"] = df["analysis_cutoff"].dt.hour
    df["managed_realized_r_numeric"] = pd.to_numeric(df.get("managed_realized_r"), errors="coerce")

    return (
        df.groupby("hour_utc", dropna=False)
        .agg(
            trades=("analysis_cutoff", "count"),
            positive_exit_rate=("managed_realized_r_numeric", lambda x: float((x > 0).mean()) if len(x) else 0.0),
            average_managed_r=("managed_realized_r_numeric", "mean"),
            total_managed_r=("managed_realized_r_numeric", "sum"),
        )
        .reset_index()
        .sort_values(["trades", "average_managed_r"], ascending=[False, False])
    )


def profile_trade_frequency(config: TradeFrequencyProfileConfig) -> dict[str, pd.DataFrame]:
    """Run the complete cache-based frequency diagnostic and save CSV outputs."""

    cache_path = Path(config.cache_path)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_df = load_cache(cache_path)
    selected_df = apply_step_candidate_rule(dataset_df, config.rule_name)

    frequency_summary_df = build_frequency_summary(selected_df, dataset_df, config.rule_name)
    daily_summary_df = build_daily_summary(selected_df)
    hourly_summary_df = build_hourly_summary(selected_df)

    safe_tag = config.tag.replace(" ", "_").lower()
    safe_rule = config.rule_name.replace(" ", "_").lower()

    frequency_summary_df.to_csv(output_dir / f"{safe_tag}_{safe_rule}_frequency_summary.csv", index=False)
    daily_summary_df.to_csv(output_dir / f"{safe_tag}_{safe_rule}_daily_summary.csv", index=False)
    hourly_summary_df.to_csv(output_dir / f"{safe_tag}_{safe_rule}_hourly_summary.csv", index=False)
    selected_df.to_csv(output_dir / f"{safe_tag}_{safe_rule}_selected_trades.csv", index=False)

    return {
        "frequency_summary": frequency_summary_df,
        "daily_summary": daily_summary_df,
        "hourly_summary": hourly_summary_df,
        "selected_trades": selected_df,
    }
