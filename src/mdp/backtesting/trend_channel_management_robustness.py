from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from mdp.backtesting.trend_channel_management_optimizer import compare_management_profiles


@dataclass(frozen=True)
class ManagementValidationRun:
    """
    Stores one robustness-validation configuration.
    """

    validation_name: str
    dataset_limit: int
    horizon_candles: int
    cutoff_step_candles: int
    analysis_limit: int = 300


def build_default_validation_runs(base_dataset_limit: int = 2000, base_horizon_candles: int = 120) -> list[ManagementValidationRun]:
    """
    Creates a small set of validation windows to check whether a management policy is stable.
    """
    return [
        ManagementValidationRun(
            validation_name="baseline_recent_window",
            dataset_limit=base_dataset_limit,
            horizon_candles=base_horizon_candles,
            cutoff_step_candles=20,
        ),
        ManagementValidationRun(
            validation_name="longer_horizon_same_sample",
            dataset_limit=base_dataset_limit,
            horizon_candles=max(base_horizon_candles, 180),
            cutoff_step_candles=20,
        ),
        ManagementValidationRun(
            validation_name="wider_sample_same_horizon",
            dataset_limit=max(base_dataset_limit, 5000),
            horizon_candles=base_horizon_candles,
            cutoff_step_candles=20,
        ),
    ]


def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    Converts a value to float while protecting summary calculations from missing data.
    """
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _classify_robustness(row: pd.Series) -> str:
    """
    Classifies whether a policy is strong enough to consider for candle-based strategy rules.
    """
    validation_runs = int(row.get("validation_runs") or 0)
    total_managed_trades = int(row.get("total_managed_trades") or 0)
    mean_average_r = _safe_float(row.get("mean_average_managed_r"))
    min_average_r = _safe_float(row.get("min_average_managed_r"))
    mean_positive_rate = _safe_float(row.get("mean_positive_exit_rate"))
    positive_average_r_run_rate = _safe_float(row.get("positive_average_r_run_rate"))
    best_policy_run_rate = _safe_float(row.get("best_policy_run_rate"))

    # Keep the state conservative because management selection can easily overfit tiny samples.
    if validation_runs < 2 or total_managed_trades < 50:
        return "insufficient_sample_for_policy_selection"

    if mean_average_r > 0 and min_average_r > -0.10 and mean_positive_rate >= 0.52 and positive_average_r_run_rate >= 0.67:
        return "robust_management_candidate"

    if best_policy_run_rate >= 0.50 and mean_positive_rate >= 0.50 and mean_average_r > -0.10:
        return "directionally_promising_but_not_profitable_yet"

    if mean_average_r <= 0:
        return "not_profitable_needs_setup_filter_or_entry_refinement"

    return "mixed_management_candidate_needs_more_data"


def _score_robust_policy(row: pd.Series) -> float:
    """
    Builds a ranking score across validation runs.
    """
    mean_average_r = _safe_float(row.get("mean_average_managed_r"))
    median_average_r = _safe_float(row.get("median_average_managed_r"))
    min_average_r = _safe_float(row.get("min_average_managed_r"))
    mean_median_r = _safe_float(row.get("mean_median_managed_r"))
    mean_positive_rate = _safe_float(row.get("mean_positive_exit_rate"))
    mean_loss_rate = _safe_float(row.get("mean_loss_exit_rate"))
    best_policy_rate = _safe_float(row.get("best_policy_run_rate"))
    total_trades = _safe_float(row.get("total_managed_trades"))

    # Reward profitability, consistency, and repeated top ranking across validation runs.
    score = (
        mean_average_r * 70.0
        + median_average_r * 25.0
        + min_average_r * 20.0
        + mean_median_r * 20.0
        + mean_positive_rate * 35.0
        + best_policy_rate * 20.0
        - mean_loss_rate * 30.0
    )

    # Penalize tiny total samples so the report does not overstate confidence.
    if total_trades < 50:
        score -= 25.0
    elif total_trades < 100:
        score -= 10.0

    return round(score, 6)


def validate_management_policy_robustness(
    symbol: str,
    timeframes: str | list[str] | tuple[str, ...],
    dataset_limit: int = 2000,
    analysis_limit: int = 300,
    start_time: str | None = None,
    end_time: str | None = None,
    warmup_candles: int = 300,
    horizon_candles: int = 120,
    step_size: int = 20,
    top_n_per_timeframe: int = 3,
    active_only: bool = True,
    recent_window: int | None = None,
    atr_window: int | None = None,
    event_lookback: int = 40,
    max_retest_candles: int = 20,
    max_false_breakout_candles: int = 10,
    max_confirmation_candles: int = 20,
    min_rejection_wick_ratio: float = 0.35,
    continuation_multiplier: float = 2.0,
    minimum_rr: float = 1.50,
    exit_at_horizon_close: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Runs management optimization across several validation windows and summarizes policy stability.
    """
    validation_runs = build_default_validation_runs(
        base_dataset_limit=dataset_limit,
        base_horizon_candles=horizon_candles,
    )

    validation_summary_frames: list[pd.DataFrame] = []
    validation_trade_frames: list[pd.DataFrame] = []
    best_policy_rows: list[dict[str, Any]] = []

    for run in validation_runs:
        # Run the existing optimizer with one validation-window configuration.
        summary_df, trades_df, _cutoff_schedule_df = compare_management_profiles(
            symbol=symbol,
            timeframes=timeframes,
            analysis_limit=run.analysis_limit or analysis_limit,
            dataset_limit=run.dataset_limit,
            start_time=start_time,
            end_time=end_time,
            warmup_candles=warmup_candles,
            horizon_candles=run.horizon_candles,
            cutoff_step_candles=run.cutoff_step_candles,
            step_size=step_size,
            top_n_per_timeframe=top_n_per_timeframe,
            active_only=active_only,
            recent_window=recent_window,
            atr_window=atr_window,
            event_lookback=event_lookback,
            max_retest_candles=max_retest_candles,
            max_false_breakout_candles=max_false_breakout_candles,
            max_confirmation_candles=max_confirmation_candles,
            min_rejection_wick_ratio=min_rejection_wick_ratio,
            continuation_multiplier=continuation_multiplier,
            minimum_rr=minimum_rr,
            exit_at_horizon_close=exit_at_horizon_close,
        )

        if not summary_df.empty:
            tagged_summary_df = summary_df.copy()
            tagged_summary_df.insert(0, "validation_name", run.validation_name)
            tagged_summary_df.insert(1, "validation_dataset_limit", run.dataset_limit)
            tagged_summary_df.insert(2, "validation_horizon_candles", run.horizon_candles)
            tagged_summary_df.insert(3, "validation_cutoff_step_candles", run.cutoff_step_candles)
            validation_summary_frames.append(tagged_summary_df)

            best_row = tagged_summary_df.iloc[0].to_dict()
            best_policy_rows.append(
                {
                    "validation_name": run.validation_name,
                    "best_policy_name": best_row.get("policy_name"),
                    "best_policy_score": best_row.get("policy_score"),
                    "best_average_managed_r": best_row.get("average_managed_r"),
                    "best_median_managed_r": best_row.get("median_managed_r"),
                    "best_positive_exit_rate": best_row.get("positive_exit_rate"),
                    "best_managed_trades": best_row.get("managed_trades"),
                }
            )

        if not trades_df.empty:
            tagged_trades_df = trades_df.copy()
            tagged_trades_df.insert(0, "validation_name", run.validation_name)
            tagged_trades_df.insert(1, "validation_dataset_limit", run.dataset_limit)
            tagged_trades_df.insert(2, "validation_horizon_candles", run.horizon_candles)
            validation_trade_frames.append(tagged_trades_df)

    all_summary_df = pd.concat(validation_summary_frames, ignore_index=True) if validation_summary_frames else pd.DataFrame()
    all_trades_df = pd.concat(validation_trade_frames, ignore_index=True) if validation_trade_frames else pd.DataFrame()
    best_by_run_df = pd.DataFrame(best_policy_rows)

    if all_summary_df.empty:
        return pd.DataFrame(), all_summary_df, best_by_run_df

    # Mark which policy won each validation run so we can measure stability.
    all_summary_df["is_best_policy_for_run"] = False
    for validation_name, group_df in all_summary_df.groupby("validation_name"):
        if group_df.empty:
            continue
        best_index = group_df.index[0]
        all_summary_df.loc[best_index, "is_best_policy_for_run"] = True

    policy_group = all_summary_df.groupby("policy_name", dropna=False)
    rows: list[dict[str, Any]] = []

    for policy_name, group_df in policy_group:
        managed_trades = group_df.get("managed_trades", pd.Series(dtype=float)).fillna(0).astype(float)
        average_r = group_df.get("average_managed_r", pd.Series(dtype=float)).fillna(0).astype(float)
        median_r = group_df.get("median_managed_r", pd.Series(dtype=float)).fillna(0).astype(float)
        positive_rate = group_df.get("positive_exit_rate", pd.Series(dtype=float)).fillna(0).astype(float)
        loss_count = group_df.get("loss_exit_count", pd.Series(dtype=float)).fillna(0).astype(float)
        loss_rate = loss_count / managed_trades.replace(0, pd.NA)
        loss_rate = loss_rate.fillna(1.0).astype(float)

        row = {
            "policy_name": policy_name,
            "validation_runs": int(len(group_df)),
            "best_policy_runs": int(group_df["is_best_policy_for_run"].sum()),
            "best_policy_run_rate": round(float(group_df["is_best_policy_for_run"].mean()), 4),
            "total_managed_trades": int(managed_trades.sum()),
            "mean_managed_trades_per_run": round(float(managed_trades.mean()), 4),
            "mean_average_managed_r": round(float(average_r.mean()), 6),
            "median_average_managed_r": round(float(average_r.median()), 6),
            "min_average_managed_r": round(float(average_r.min()), 6),
            "max_average_managed_r": round(float(average_r.max()), 6),
            "mean_median_managed_r": round(float(median_r.mean()), 6),
            "mean_positive_exit_rate": round(float(positive_rate.mean()), 6),
            "mean_loss_exit_rate": round(float(loss_rate.mean()), 6),
            "positive_average_r_run_rate": round(float((average_r > 0).mean()), 4),
            "positive_median_r_run_rate": round(float((median_r > 0).mean()), 4),
        }
        row["robust_policy_score"] = _score_robust_policy(pd.Series(row))
        row["robustness_state"] = _classify_robustness(pd.Series(row))
        rows.append(row)

    robustness_df = pd.DataFrame(rows)
    robustness_df = robustness_df.sort_values(
        by=["robust_policy_score", "mean_average_managed_r", "mean_positive_exit_rate"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return robustness_df, all_summary_df, best_by_run_df
