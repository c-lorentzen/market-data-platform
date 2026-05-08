
"""Step Index adaptive filter diagnostics for trend-channel candidate trades.

This module is intentionally cache-based. It does not rebuild the full strategy
pipeline. It loads the saved entry research cache, applies a candidate rule,
deduplicates repeated trade candidates by cutoff, then tests whether simple
no-lookahead learning filters would have improved trade selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AdaptiveFilterConfig:
    """Configuration for the adaptive filter diagnostic."""

    cache_path: str
    rule_name: str = "candidate_v1_base"
    tag: str = "step_candidate_adaptive_filter"
    min_history_trades: int = 5
    avoid_avg_r_below: float = 0.0
    max_trades_per_day: int | None = None
    output_dir: str = "outputs/trend_channel_adaptive_filters"


def _as_bool(series: pd.Series) -> pd.Series:
    """Convert mixed bool/string columns into a clean boolean series."""

    if series.dtype == bool:
        return series.fillna(False)

    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def _parse_timestamp_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the timestamp columns needed for daily and hourly diagnostics."""

    out = df.copy()

    if "analysis_cutoff" not in out.columns:
        raise ValueError("Cache must contain an analysis_cutoff column.")

    out["analysis_cutoff"] = pd.to_datetime(out["analysis_cutoff"], errors="coerce")
    out = out.dropna(subset=["analysis_cutoff"]).copy()

    out["trade_date"] = out["analysis_cutoff"].dt.date.astype(str)
    out["hour_utc"] = out["analysis_cutoff"].dt.hour

    return out


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise a clear error if the cache is missing required columns."""

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Cache is missing required columns: {missing}")


def _target_plan_priority(value: object) -> int:
    """Rank target-plan quality using only pre-entry target availability."""

    text = str(value).strip().lower()

    if text == "tp1_tp2_tp3_available":
        return 5
    if text == "tp2_tp3_available":
        return 4
    if text == "tp2_only":
        return 3
    if text == "tp3_available":
        return 2
    if "available" in text:
        return 1
    return 0


def apply_step_candidate_rule(dataset_df: pd.DataFrame, rule_name: str) -> pd.DataFrame:
    """Apply a Step Index candidate rule to the saved cache rows.

    The rules use only information available before or at entry. They should not
    inspect managed_realized_r while selecting trades.
    """

    df = dataset_df.copy()

    if "entered_trade" in df.columns:
        df = df[_as_bool(df["entered_trade"])].copy()
    elif "entry_status" in df.columns:
        df = df[df["entry_status"].astype(str).str.lower().eq("entered")].copy()

    if df.empty:
        return df

    # Candidate v1: current refined candidate.
    target_plan = df.get("target_plan_state", pd.Series("", index=df.index)).astype(str)
    available_targets = df.get("available_targets", pd.Series("", index=df.index)).astype(str)

    is_retest = df.get("entry_mode", pd.Series("", index=df.index)).astype(str).eq(
        "retest_rejection_close"
    )
    is_continuation = df.get("entry_mode", pd.Series("", index=df.index)).astype(str).eq(
        "continuation_after_retest_close"
    )
    entry_variant = df.get("entry_variant_name", pd.Series("", index=df.index)).astype(str)

    retest_ok = is_retest & target_plan.eq("tp2_tp3_available")
    continuation_ok = is_continuation & entry_variant.eq("confirm_2_directional_close")

    selected = df[retest_ok | continuation_ok].copy()

    if rule_name == "candidate_v1_base":
        return selected

    if rule_name == "avoid_tp3_only":
        # Avoid trades where the only available target is TP3.
        no_tp3_only = ~(
            target_plan.loc[selected.index].eq("tp3_available")
            | available_targets.loc[selected.index].eq("TP3")
        )
        return selected[no_tp3_only].copy()

    if rule_name == "require_best_available_rr_high":
        rr_bucket = selected.get(
            "best_available_rr_bucket",
            selected.get("_best_available_rr_bucket", pd.Series("", index=selected.index)),
        ).astype(str)
        return selected[rr_bucket.eq("best_available_rr_high")].copy()

    if rule_name == "keep_high_rr_or_bullish_pullback":
        rr_bucket = selected.get(
            "best_available_rr_bucket",
            selected.get("_best_available_rr_bucket", pd.Series("", index=selected.index)),
        ).astype(str)
        scenario_state = selected.get("scenario_state", pd.Series("", index=selected.index)).astype(str)
        return selected[
            rr_bucket.eq("best_available_rr_high")
            | scenario_state.eq("bullish_structure_bearish_entry_pullback")
        ].copy()

    raise ValueError(f"Unknown rule_name: {rule_name}")


def dedupe_one_trade_per_cutoff(selected_df: pd.DataFrame) -> pd.DataFrame:
    """Keep one deterministic trade candidate per analysis cutoff.

    Some cache rows can contain multiple candidate rows for the same cutoff.
    Live trading should not blindly count all of them as separate independent
    opportunities. This function keeps the best pre-entry candidate per cutoff.
    """

    if selected_df.empty:
        return selected_df.copy()

    df = selected_df.copy()

    # These values are pre-entry or entry-time features, not future outcomes.
    if "_best_available_rr" not in df.columns and "best_available_rr" in df.columns:
        df["_best_available_rr"] = df["best_available_rr"]

    df["_target_plan_priority"] = df.get("target_plan_state", "").map(_target_plan_priority)
    df["_event_quality_rank"] = pd.to_numeric(
        df.get("event_quality_score", 0.0), errors="coerce"
    ).fillna(0.0)
    df["_rr_rank"] = pd.to_numeric(df.get("_best_available_rr", 0.0), errors="coerce").fillna(0.0)
    df["_risk_rank"] = pd.to_numeric(df.get("risk_points", 0.0), errors="coerce").fillna(0.0)

    # Prefer stronger target structure and quality, then lower risk as tie-breaker.
    df = df.sort_values(
        by=[
            "analysis_cutoff",
            "_target_plan_priority",
            "_event_quality_rank",
            "_rr_rank",
            "_risk_rank",
        ],
        ascending=[True, False, False, False, True],
    )

    deduped = df.drop_duplicates(subset=["analysis_cutoff"], keep="first").copy()

    cleanup = [
        "_target_plan_priority",
        "_event_quality_rank",
        "_rr_rank",
        "_risk_rank",
    ]
    return deduped.drop(columns=[c for c in cleanup if c in deduped.columns])


def build_frequency_summary(selected_df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Build an overall frequency and expectancy summary."""

    if selected_df.empty:
        return pd.DataFrame(
            [
                {
                    "label": label,
                    "trades": 0,
                    "unique_days": 0,
                    "avg_trades_per_day": 0.0,
                    "median_trades_per_day": 0.0,
                    "positive_exit_rate": 0.0,
                    "average_managed_r": 0.0,
                    "median_managed_r": 0.0,
                    "total_managed_r": 0.0,
                }
            ]
        )

    _require_columns(selected_df, ["managed_realized_r"])

    daily_counts = selected_df.groupby("trade_date").size()

    return pd.DataFrame(
        [
            {
                "label": label,
                "trades": int(len(selected_df)),
                "unique_days": int(selected_df["trade_date"].nunique()),
                "avg_trades_per_day": float(daily_counts.mean()),
                "median_trades_per_day": float(daily_counts.median()),
                "max_trades_per_day": int(daily_counts.max()),
                "positive_exit_rate": float((selected_df["managed_realized_r"] > 0).mean()),
                "average_managed_r": float(selected_df["managed_realized_r"].mean()),
                "median_managed_r": float(selected_df["managed_realized_r"].median()),
                "total_managed_r": float(selected_df["managed_realized_r"].sum()),
            }
        ]
    )


def build_daily_summary(selected_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize selected trades by day."""

    if selected_df.empty:
        return pd.DataFrame()

    return (
        selected_df.groupby("trade_date")
        .agg(
            trades=("managed_realized_r", "size"),
            positive_exit_rate=("managed_realized_r", lambda x: float((x > 0).mean())),
            average_managed_r=("managed_realized_r", "mean"),
            median_managed_r=("managed_realized_r", "median"),
            total_managed_r=("managed_realized_r", "sum"),
            first_trade_time=("analysis_cutoff", "min"),
            last_trade_time=("analysis_cutoff", "max"),
        )
        .reset_index()
    )


def _historical_feature_score(
    history_df: pd.DataFrame,
    feature_name: str,
    feature_value: object,
) -> tuple[int, float]:
    """Return previous sample count and average R for a feature value."""

    if history_df.empty or feature_name not in history_df.columns:
        return 0, np.nan

    subset = history_df[history_df[feature_name].astype(str).eq(str(feature_value))]
    if subset.empty:
        return 0, np.nan

    return int(len(subset)), float(subset["managed_realized_r"].mean())


def simulate_adaptive_feature_filter(
    selected_df: pd.DataFrame,
    feature_names: list[str],
    min_history_trades: int,
    avoid_avg_r_below: float,
    max_trades_per_day: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate a no-lookahead adaptive filter using prior days only.

    For each day, the filter checks whether the current trade has any feature
    value that was historically negative in previous days. If so, it skips that
    trade. This is a simple rule-learning proxy, not a final ML model.
    """

    if selected_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    trades = selected_df.sort_values("analysis_cutoff").copy()
    trade_dates = list(trades["trade_date"].drop_duplicates())

    decisions = []

    for trade_date in trade_dates:
        prior_df = trades[trades["trade_date"] < trade_date]
        current_day = trades[trades["trade_date"] == trade_date].copy()

        kept_count_today = 0

        for _, row in current_day.iterrows():
            skip_reasons = []

            for feature_name in feature_names:
                if feature_name not in row.index:
                    continue

                history_count, history_avg_r = _historical_feature_score(
                    prior_df,
                    feature_name,
                    row[feature_name],
                )

                if history_count >= min_history_trades and history_avg_r < avoid_avg_r_below:
                    skip_reasons.append(
                        f"{feature_name}={row[feature_name]} prior_avg_r={history_avg_r:.4f} n={history_count}"
                    )

            kept = len(skip_reasons) == 0

            if kept and max_trades_per_day is not None and kept_count_today >= max_trades_per_day:
                kept = False
                skip_reasons.append(f"max_trades_per_day={max_trades_per_day}")

            if kept:
                kept_count_today += 1

            decisions.append(
                {
                    "trade_date": trade_date,
                    "analysis_cutoff": row["analysis_cutoff"],
                    "kept_by_adaptive_filter": kept,
                    "skip_reasons": " | ".join(skip_reasons),
                    "managed_realized_r": row["managed_realized_r"],
                    "scenario_state": row.get("scenario_state", ""),
                    "event_state": row.get("event_state", ""),
                    "entry_mode": row.get("entry_mode", ""),
                    "target_plan_state": row.get("target_plan_state", ""),
                    "risk_points_bucket": row.get("risk_points_bucket", ""),
                    "best_available_rr_bucket": row.get("best_available_rr_bucket", ""),
                    "hour_utc": row.get("hour_utc", np.nan),
                }
            )

    decisions_df = pd.DataFrame(decisions)

    if decisions_df.empty:
        return decisions_df, pd.DataFrame()

    baseline_summary = build_frequency_summary(trades, "baseline_deduped")
    adaptive_summary = build_frequency_summary(
        decisions_df[decisions_df["kept_by_adaptive_filter"]].copy(),
        "adaptive_feature_filter_no_lookahead",
    )

    skipped_df = decisions_df[~decisions_df["kept_by_adaptive_filter"]]
    kept_df = decisions_df[decisions_df["kept_by_adaptive_filter"]]

    summary = pd.concat([baseline_summary, adaptive_summary], ignore_index=True)
    summary["skipped_trades"] = int(len(skipped_df))
    summary["skipped_average_r"] = (
        float(skipped_df["managed_realized_r"].mean()) if not skipped_df.empty else np.nan
    )
    summary["kept_average_r"] = (
        float(kept_df["managed_realized_r"].mean()) if not kept_df.empty else np.nan
    )

    return decisions_df, summary


def analyze_step_candidate_adaptive_filter(config: AdaptiveFilterConfig) -> dict[str, pd.DataFrame]:
    """Run frequency + no-lookahead adaptive filter diagnostics."""

    cache_path = Path(config.cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache path does not exist: {cache_path}")

    dataset_df = pd.read_csv(cache_path)
    dataset_df = _parse_timestamp_columns(dataset_df)

    selected_raw = apply_step_candidate_rule(dataset_df, config.rule_name)
    selected_raw = _parse_timestamp_columns(selected_raw) if not selected_raw.empty else selected_raw

    selected_deduped = dedupe_one_trade_per_cutoff(selected_raw)
    selected_deduped = _parse_timestamp_columns(selected_deduped) if not selected_deduped.empty else selected_deduped

    feature_names = [
        "hour_utc",
        "scenario_state",
        "event_state",
        "entry_mode",
        "target_plan_state",
        "risk_points_bucket",
        "best_available_rr_bucket",
    ]

    decisions_df, adaptive_summary = simulate_adaptive_feature_filter(
        selected_deduped,
        feature_names=feature_names,
        min_history_trades=config.min_history_trades,
        avoid_avg_r_below=config.avoid_avg_r_below,
        max_trades_per_day=config.max_trades_per_day,
    )

    workload_df = pd.DataFrame(
        [
            {
                "cache_path": str(cache_path),
                "cache_rows": int(len(dataset_df)),
                "raw_selected_rows": int(len(selected_raw)),
                "deduped_selected_rows": int(len(selected_deduped)),
                "unique_analysis_cutoffs": int(selected_deduped["analysis_cutoff"].nunique())
                if not selected_deduped.empty
                else 0,
                "rule_name": config.rule_name,
                "min_history_trades": config.min_history_trades,
                "avoid_avg_r_below": config.avoid_avg_r_below,
                "max_trades_per_day": config.max_trades_per_day,
                "cache_design": "load_saved_cache_once_then_run_unique_frequency_and_no_lookahead_filter",
            }
        ]
    )

    frequency_summary = pd.concat(
        [
            build_frequency_summary(selected_raw, "raw_rows_may_include_duplicate_cutoffs"),
            build_frequency_summary(selected_deduped, "deduped_one_trade_per_cutoff"),
        ],
        ignore_index=True,
    )

    daily_summary = build_daily_summary(selected_deduped)

    return {
        "workload": workload_df,
        "frequency_summary": frequency_summary,
        "daily_summary": daily_summary,
        "adaptive_summary": adaptive_summary,
        "adaptive_decisions": decisions_df,
        "selected_deduped_preview": selected_deduped.head(50).copy(),
    }


def save_outputs(results: dict[str, pd.DataFrame], config: AdaptiveFilterConfig) -> dict[str, str]:
    """Save diagnostic DataFrames to CSV files."""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_rule = config.rule_name.replace(" ", "_").replace("/", "_")
    base_name = f"{config.tag}_{safe_rule}"

    paths: dict[str, str] = {}

    for name, df in results.items():
        if df is None or df.empty:
            continue
        path = output_dir / f"{base_name}_{name}.csv"
        df.to_csv(path, index=False)
        paths[name] = str(path)

    return paths
