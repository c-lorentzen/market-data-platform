"""Walk-forward online learning filter diagnostics for trend-channel ML datasets.

This module works from an already exported ML-ready dataset. It does not rebuild the
strategy pipeline. The goal is to test whether previous labeled setups can help filter
future setups without using future outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OnlineLearningFilterConfig:
    """Configuration for the no-lookahead online filter diagnostic."""

    ml_dataset_path: str
    tag: str = "step_online_learning_filter"
    min_history_trades: int = 20
    min_context_trades: int = 4
    prior_strength: float = 6.0
    min_estimated_avg_r: float = 0.03
    min_estimated_positive_rate: float = 0.58
    max_estimated_full_loss_rate: float = 0.25
    max_trades_per_day: int | None = None
    output_dir: str = "outputs/trend_channel_online_learning_filters"


@dataclass(frozen=True)
class OnlineLearningFilterResult:
    """Container for all generated report tables."""

    workload_df: pd.DataFrame
    summary_df: pd.DataFrame
    daily_df: pd.DataFrame
    decision_df: pd.DataFrame
    context_edge_df: pd.DataFrame
    output_paths: dict[str, str]


# These are all pre-entry features. Outcome/label columns are intentionally excluded.
BASE_CONTEXT_FEATURES = [
    "scenario_state",
    "event_state",
    "entry_mode",
    "entry_variant_name",
    "plan_direction",
    "target_plan_state",
    "available_targets",
    "hour_bucket",
    "online_best_rr_bucket",
    "online_risk_ratio_bucket",
]


LABEL_COLUMNS = [
    "label_positive_r",
    "label_full_or_near_full_loss",
    "label_managed_r",
]


def load_ml_dataset(ml_dataset_path: str) -> pd.DataFrame:
    """Load and normalize the ML-ready dataset."""

    dataset_df = pd.read_csv(ml_dataset_path)

    if "analysis_cutoff" not in dataset_df.columns:
        raise ValueError("ML dataset must include an analysis_cutoff column.")

    if "label_managed_r" not in dataset_df.columns:
        raise ValueError("ML dataset must include a label_managed_r column.")

    dataset_df = dataset_df.copy()
    dataset_df["analysis_cutoff"] = pd.to_datetime(dataset_df["analysis_cutoff"], errors="coerce")

    if "trade_date" in dataset_df.columns:
        dataset_df["trade_date"] = pd.to_datetime(dataset_df["trade_date"], errors="coerce").dt.date
    else:
        dataset_df["trade_date"] = dataset_df["analysis_cutoff"].dt.date

    if "hour_utc" not in dataset_df.columns:
        dataset_df["hour_utc"] = dataset_df["analysis_cutoff"].dt.hour

    # Ensure labels exist, even if older exports only had managed R.
    if "label_positive_r" not in dataset_df.columns:
        dataset_df["label_positive_r"] = (pd.to_numeric(dataset_df["label_managed_r"], errors="coerce") > 0).astype(int)

    if "label_full_or_near_full_loss" not in dataset_df.columns:
        dataset_df["label_full_or_near_full_loss"] = (pd.to_numeric(dataset_df["label_managed_r"], errors="coerce") <= -0.95).astype(int)

    dataset_df["label_managed_r"] = pd.to_numeric(dataset_df["label_managed_r"], errors="coerce")
    dataset_df = dataset_df.dropna(subset=["analysis_cutoff", "label_managed_r"])
    dataset_df = dataset_df.sort_values("analysis_cutoff").reset_index(drop=True)

    # Keep exactly one row per cutoff if a file accidentally contains duplicates.
    # The selection score is pre-entry, so it is safe for deduplication.
    if dataset_df["analysis_cutoff"].duplicated().any():
        if "pre_entry_selection_score" in dataset_df.columns:
            dataset_df = (
                dataset_df.sort_values(["analysis_cutoff", "pre_entry_selection_score"], ascending=[True, False])
                .drop_duplicates(subset=["analysis_cutoff"], keep="first")
                .sort_values("analysis_cutoff")
                .reset_index(drop=True)
            )
        else:
            dataset_df = dataset_df.drop_duplicates(subset=["analysis_cutoff"], keep="first").reset_index(drop=True)

    return dataset_df


def _hour_bucket(hour: int) -> str:
    """Group UTC hours into simple time blocks."""

    if 0 <= hour <= 5:
        return "hour_00_05"
    if 6 <= hour <= 11:
        return "hour_06_11"
    if 12 <= hour <= 17:
        return "hour_12_17"
    return "hour_18_23"


def _numeric_bucket_from_prior(value: float, prior_values: pd.Series, label: str, min_values: int = 12) -> str:
    """Bucket a numeric feature using only prior values."""

    if pd.isna(value):
        return f"{label}_unknown"

    clean_prior = pd.to_numeric(prior_values, errors="coerce").dropna()

    if len(clean_prior) < min_values:
        return f"{label}_unknown"

    low_cut = clean_prior.quantile(0.33)
    high_cut = clean_prior.quantile(0.67)

    if value <= low_cut:
        return f"{label}_low"
    if value >= high_cut:
        return f"{label}_high"
    return f"{label}_mid"


def _add_online_feature_values(current_row: pd.Series, prior_df: pd.DataFrame) -> dict[str, str]:
    """Build online categorical feature values using only prior rows for buckets."""

    hour = int(current_row.get("hour_utc", 0) or 0)
    features = {
        "hour_bucket": _hour_bucket(hour),
    }

    best_rr_column = "best_available_rr"
    if best_rr_column not in current_row.index and "_best_available_rr" in current_row.index:
        best_rr_column = "_best_available_rr"

    best_rr_value = pd.to_numeric(current_row.get(best_rr_column, np.nan), errors="coerce")
    prior_best_rr = prior_df[best_rr_column] if best_rr_column in prior_df.columns else pd.Series(dtype=float)
    features["online_best_rr_bucket"] = _numeric_bucket_from_prior(
        best_rr_value,
        prior_best_rr,
        "best_rr",
    )

    risk_column = "risk_points_ratio" if "risk_points_ratio" in current_row.index else "risk_points"
    risk_value = pd.to_numeric(current_row.get(risk_column, np.nan), errors="coerce")
    prior_risk = prior_df[risk_column] if risk_column in prior_df.columns else pd.Series(dtype=float)
    features["online_risk_ratio_bucket"] = _numeric_bucket_from_prior(
        risk_value,
        prior_risk,
        "risk",
    )

    return features


def _context_stats_for_value(
    prior_df: pd.DataFrame,
    feature_name: str,
    feature_value: str,
    global_avg_r: float,
    global_positive_rate: float,
    global_full_loss_rate: float,
    min_context_trades: int,
    prior_strength: float,
) -> dict[str, float | str | int | bool]:
    """Return smoothed statistics for one feature=value context."""

    if feature_name not in prior_df.columns:
        return {
            "feature_name": feature_name,
            "feature_value": feature_value,
            "context_count": 0,
            "is_usable": False,
            "smoothed_avg_r": global_avg_r,
            "smoothed_positive_rate": global_positive_rate,
            "smoothed_full_loss_rate": global_full_loss_rate,
        }

    context_df = prior_df[prior_df[feature_name].astype(str) == str(feature_value)]
    context_count = int(len(context_df))

    if context_count == 0:
        raw_avg_r = global_avg_r
        raw_positive_rate = global_positive_rate
        raw_full_loss_rate = global_full_loss_rate
    else:
        raw_avg_r = float(context_df["label_managed_r"].mean())
        raw_positive_rate = float(context_df["label_positive_r"].mean())
        raw_full_loss_rate = float(context_df["label_full_or_near_full_loss"].mean())

    # Shrink context stats toward global stats so tiny samples cannot dominate.
    denominator = context_count + prior_strength
    smoothed_avg_r = ((context_count * raw_avg_r) + (prior_strength * global_avg_r)) / denominator
    smoothed_positive_rate = ((context_count * raw_positive_rate) + (prior_strength * global_positive_rate)) / denominator
    smoothed_full_loss_rate = ((context_count * raw_full_loss_rate) + (prior_strength * global_full_loss_rate)) / denominator

    return {
        "feature_name": feature_name,
        "feature_value": feature_value,
        "context_count": context_count,
        "is_usable": context_count >= min_context_trades,
        "raw_avg_r": raw_avg_r,
        "raw_positive_rate": raw_positive_rate,
        "raw_full_loss_rate": raw_full_loss_rate,
        "smoothed_avg_r": float(smoothed_avg_r),
        "smoothed_positive_rate": float(smoothed_positive_rate),
        "smoothed_full_loss_rate": float(smoothed_full_loss_rate),
    }


def _summarize_trades(label: str, trade_df: pd.DataFrame) -> dict[str, float | str | int]:
    """Summarize trade outcomes."""

    if trade_df.empty:
        return {
            "label": label,
            "trades": 0,
            "unique_days": 0,
            "avg_trades_per_day": 0.0,
            "positive_exit_rate": 0.0,
            "full_loss_rate": 0.0,
            "average_managed_r": 0.0,
            "median_managed_r": 0.0,
            "total_managed_r": 0.0,
        }

    daily_counts = trade_df.groupby("trade_date").size()

    return {
        "label": label,
        "trades": int(len(trade_df)),
        "unique_days": int(trade_df["trade_date"].nunique()),
        "avg_trades_per_day": float(daily_counts.mean()),
        "median_trades_per_day": float(daily_counts.median()),
        "max_trades_per_day": int(daily_counts.max()),
        "positive_exit_rate": float(trade_df["label_positive_r"].mean()),
        "full_loss_rate": float(trade_df["label_full_or_near_full_loss"].mean()),
        "average_managed_r": float(trade_df["label_managed_r"].mean()),
        "median_managed_r": float(trade_df["label_managed_r"].median()),
        "total_managed_r": float(trade_df["label_managed_r"].sum()),
    }


def _build_context_edge_table(dataset_df: pd.DataFrame) -> pd.DataFrame:
    """Create a diagnostic table of feature edges from the whole labeled dataset.

    This table is for interpretation only, not for live trading decisions.
    """

    rows = []
    feature_candidates = [
        "scenario_state",
        "event_state",
        "entry_mode",
        "entry_variant_name",
        "plan_direction",
        "target_plan_state",
        "available_targets",
        "hour_utc",
    ]

    for feature_name in feature_candidates:
        if feature_name not in dataset_df.columns:
            continue

        grouped = dataset_df.groupby(feature_name, dropna=False)
        for feature_value, group_df in grouped:
            if len(group_df) < 3:
                continue

            rows.append(
                {
                    "feature_name": feature_name,
                    "feature_value": str(feature_value),
                    "trades": int(len(group_df)),
                    "positive_exit_rate": float(group_df["label_positive_r"].mean()),
                    "full_loss_rate": float(group_df["label_full_or_near_full_loss"].mean()),
                    "average_managed_r": float(group_df["label_managed_r"].mean()),
                    "median_managed_r": float(group_df["label_managed_r"].median()),
                    "total_managed_r": float(group_df["label_managed_r"].sum()),
                }
            )

    if not rows:
        return pd.DataFrame()

    edge_df = pd.DataFrame(rows)
    edge_df = edge_df.sort_values(["average_managed_r", "trades"], ascending=[False, False]).reset_index(drop=True)
    return edge_df


def run_online_learning_filter(config: OnlineLearningFilterConfig) -> OnlineLearningFilterResult:
    """Run a no-lookahead online filter simulation on an ML-ready dataset."""

    dataset_df = load_ml_dataset(config.ml_dataset_path)

    # Add placeholders that will be filled per row using only prior rows.
    dataset_df["hour_bucket"] = dataset_df["hour_utc"].apply(lambda x: _hour_bucket(int(x or 0)))
    dataset_df["online_best_rr_bucket"] = "best_rr_unknown"
    dataset_df["online_risk_ratio_bucket"] = "risk_unknown"

    decisions = []
    kept_flags = []
    trades_taken_today: dict[object, int] = {}

    for row_position, current_row in dataset_df.iterrows():
        prior_df = dataset_df.iloc[:row_position].copy()

        online_features = _add_online_feature_values(current_row, prior_df)
        for feature_name, feature_value in online_features.items():
            dataset_df.loc[row_position, feature_name] = feature_value

        current_trade_date = current_row["trade_date"]
        trades_today = trades_taken_today.get(current_trade_date, 0)

        if len(prior_df) < config.min_history_trades:
            keep_trade = True
            decision_state = "warmup_keep_until_min_history"
            estimated_avg_r = np.nan
            estimated_positive_rate = np.nan
            estimated_full_loss_rate = np.nan
            usable_context_count = 0
            skip_reasons = ""
        elif config.max_trades_per_day is not None and trades_today >= config.max_trades_per_day:
            keep_trade = False
            decision_state = "skip_daily_trade_cap"
            estimated_avg_r = np.nan
            estimated_positive_rate = np.nan
            estimated_full_loss_rate = np.nan
            usable_context_count = 0
            skip_reasons = f"daily_trade_cap_reached={config.max_trades_per_day}"
        else:
            global_avg_r = float(prior_df["label_managed_r"].mean())
            global_positive_rate = float(prior_df["label_positive_r"].mean())
            global_full_loss_rate = float(prior_df["label_full_or_near_full_loss"].mean())

            context_rows = []
            for feature_name in BASE_CONTEXT_FEATURES:
                if feature_name in online_features:
                    feature_value = online_features[feature_name]
                else:
                    feature_value = current_row.get(feature_name, "")

                context_rows.append(
                    _context_stats_for_value(
                        prior_df=prior_df,
                        feature_name=feature_name,
                        feature_value=str(feature_value),
                        global_avg_r=global_avg_r,
                        global_positive_rate=global_positive_rate,
                        global_full_loss_rate=global_full_loss_rate,
                        min_context_trades=config.min_context_trades,
                        prior_strength=config.prior_strength,
                    )
                )

            usable_rows = [row for row in context_rows if row["is_usable"]]
            usable_context_count = len(usable_rows)

            if usable_context_count == 0:
                estimated_avg_r = global_avg_r
                estimated_positive_rate = global_positive_rate
                estimated_full_loss_rate = global_full_loss_rate
            else:
                estimated_avg_r = float(np.mean([row["smoothed_avg_r"] for row in usable_rows]))
                estimated_positive_rate = float(np.mean([row["smoothed_positive_rate"] for row in usable_rows]))
                estimated_full_loss_rate = float(np.mean([row["smoothed_full_loss_rate"] for row in usable_rows]))

            skip_reasons_list = []
            if estimated_avg_r < config.min_estimated_avg_r:
                skip_reasons_list.append(
                    f"estimated_avg_r={estimated_avg_r:.4f}<min={config.min_estimated_avg_r:.4f}"
                )
            if estimated_positive_rate < config.min_estimated_positive_rate:
                skip_reasons_list.append(
                    f"estimated_positive_rate={estimated_positive_rate:.4f}<min={config.min_estimated_positive_rate:.4f}"
                )
            if estimated_full_loss_rate > config.max_estimated_full_loss_rate:
                skip_reasons_list.append(
                    f"estimated_full_loss_rate={estimated_full_loss_rate:.4f}>max={config.max_estimated_full_loss_rate:.4f}"
                )

            keep_trade = len(skip_reasons_list) == 0
            decision_state = "kept_by_online_filter" if keep_trade else "skipped_by_online_filter"
            skip_reasons = "; ".join(skip_reasons_list)

        if keep_trade:
            trades_taken_today[current_trade_date] = trades_today + 1

        kept_flags.append(bool(keep_trade))
        decisions.append(
            {
                "analysis_cutoff": current_row["analysis_cutoff"],
                "trade_date": current_trade_date,
                "hour_utc": current_row.get("hour_utc"),
                "decision_state": decision_state,
                "kept_by_online_filter": bool(keep_trade),
                "estimated_avg_r": estimated_avg_r,
                "estimated_positive_rate": estimated_positive_rate,
                "estimated_full_loss_rate": estimated_full_loss_rate,
                "usable_context_count": usable_context_count,
                "skip_reasons": skip_reasons,
                "label_managed_r": current_row["label_managed_r"],
                "label_positive_r": current_row["label_positive_r"],
                "label_full_or_near_full_loss": current_row["label_full_or_near_full_loss"],
                "scenario_state": current_row.get("scenario_state", ""),
                "event_state": current_row.get("event_state", ""),
                "entry_mode": current_row.get("entry_mode", ""),
                "target_plan_state": current_row.get("target_plan_state", ""),
                "available_targets": current_row.get("available_targets", ""),
                "online_best_rr_bucket": dataset_df.loc[row_position, "online_best_rr_bucket"],
                "online_risk_ratio_bucket": dataset_df.loc[row_position, "online_risk_ratio_bucket"],
            }
        )

    decision_df = pd.DataFrame(decisions)
    filtered_df = dataset_df[pd.Series(kept_flags, index=dataset_df.index)].copy()
    skipped_df = dataset_df[~pd.Series(kept_flags, index=dataset_df.index)].copy()

    summary_rows = [
        _summarize_trades("baseline_all_ml_rows", dataset_df),
        _summarize_trades("online_learning_filtered_rows", filtered_df),
        _summarize_trades("skipped_rows_diagnostic_only", skipped_df),
    ]
    summary_df = pd.DataFrame(summary_rows)

    daily_rows = []
    for label, trade_df in [
        ("baseline", dataset_df),
        ("online_learning_filter", filtered_df),
    ]:
        if trade_df.empty:
            continue

        daily_grouped = trade_df.groupby("trade_date")
        for trade_date, group_df in daily_grouped:
            daily_rows.append(
                {
                    "label": label,
                    "trade_date": trade_date,
                    "trades": int(len(group_df)),
                    "positive_exit_rate": float(group_df["label_positive_r"].mean()),
                    "full_loss_rate": float(group_df["label_full_or_near_full_loss"].mean()),
                    "average_managed_r": float(group_df["label_managed_r"].mean()),
                    "median_managed_r": float(group_df["label_managed_r"].median()),
                    "total_managed_r": float(group_df["label_managed_r"].sum()),
                }
            )

    daily_df = pd.DataFrame(daily_rows)
    context_edge_df = _build_context_edge_table(dataset_df)

    workload_df = pd.DataFrame(
        [
            {
                "ml_dataset_path": config.ml_dataset_path,
                "dataset_rows": int(len(dataset_df)),
                "min_history_trades": config.min_history_trades,
                "min_context_trades": config.min_context_trades,
                "prior_strength": config.prior_strength,
                "min_estimated_avg_r": config.min_estimated_avg_r,
                "min_estimated_positive_rate": config.min_estimated_positive_rate,
                "max_estimated_full_loss_rate": config.max_estimated_full_loss_rate,
                "max_trades_per_day": config.max_trades_per_day,
                "cache_design": "load_ml_ready_dataset_once_then_no_lookahead_online_filter",
            }
        ]
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_tag = "".join(ch.lower() if ch.isalnum() else "_" for ch in config.tag).strip("_")
    output_paths = {
        "workload": str(output_dir / f"{safe_tag}_workload.csv"),
        "summary": str(output_dir / f"{safe_tag}_summary.csv"),
        "daily": str(output_dir / f"{safe_tag}_daily.csv"),
        "decisions": str(output_dir / f"{safe_tag}_decisions.csv"),
        "context_edges": str(output_dir / f"{safe_tag}_context_edges.csv"),
    }

    workload_df.to_csv(output_paths["workload"], index=False)
    summary_df.to_csv(output_paths["summary"], index=False)
    daily_df.to_csv(output_paths["daily"], index=False)
    decision_df.to_csv(output_paths["decisions"], index=False)
    context_edge_df.to_csv(output_paths["context_edges"], index=False)

    return OnlineLearningFilterResult(
        workload_df=workload_df,
        summary_df=summary_df,
        daily_df=daily_df,
        decision_df=decision_df,
        context_edge_df=context_edge_df,
        output_paths=output_paths,
    )
