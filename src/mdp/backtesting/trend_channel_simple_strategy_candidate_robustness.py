from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

import pandas as pd

from mdp.backtesting.trend_channel_simple_strategy_candidate_validation import (
    SimpleStrategyCandidateValidationConfig,
    run_simple_strategy_candidate_validation,
)


@dataclass(frozen=True)
class SimpleStrategyCandidateRobustnessConfig:
    """
    Stores settings for validating simple strategy candidates across several samples.
    """

    symbol: str
    timeframes: str
    analysis_limit: int = 220
    dataset_limit: int = 1600
    warmup_candles: int = 220
    horizon_candles: int = 80
    cutoff_step_candles: int = 20
    max_cutoffs: int | None = 35
    max_validation_runs: int = 3
    management_policy_name: str = "targetless_scalp_manage_only"
    minimum_rr: float = 1.50
    min_trades_for_candidate: int = 8
    min_total_trades_for_robust_candidate: int = 20
    active_only: bool = True
    quick: bool = False


@dataclass(frozen=True)
class CandidateRobustnessProfile:
    """
    Describes one validation profile for a candidate robustness check.
    """

    validation_name: str
    dataset_limit: int
    horizon_candles: int
    cutoff_step_candles: int
    max_cutoffs: int | None
    analysis_limit: int
    warmup_candles: int



def _config_signature(profile: CandidateRobustnessProfile) -> str:
    """
    Creates a compact text signature so duplicated validation configs are easy to detect.
    """
    return (
        f"dataset={profile.dataset_limit}|horizon={profile.horizon_candles}|"
        f"cutoff_step={profile.cutoff_step_candles}|analysis={profile.analysis_limit}|"
        f"warmup={profile.warmup_candles}|max_cutoffs={profile.max_cutoffs}"
    )



def build_candidate_robustness_profiles(
    config: SimpleStrategyCandidateRobustnessConfig,
) -> list[CandidateRobustnessProfile]:
    """
    Builds a small set of distinct validation profiles.
    """
    if config.quick:
        # Quick mode should finish fast on a normal laptop.
        return [
            CandidateRobustnessProfile(
                validation_name="quick_baseline",
                dataset_limit=min(config.dataset_limit, 900),
                horizon_candles=min(config.horizon_candles, 60),
                cutoff_step_candles=max(config.cutoff_step_candles, 20),
                max_cutoffs=min(config.max_cutoffs or 20, 20),
                analysis_limit=min(config.analysis_limit, 180),
                warmup_candles=min(config.warmup_candles, 180),
            ),
            CandidateRobustnessProfile(
                validation_name="quick_denser_cutoffs",
                dataset_limit=min(config.dataset_limit, 900),
                horizon_candles=min(config.horizon_candles, 60),
                cutoff_step_candles=10,
                max_cutoffs=min(config.max_cutoffs or 20, 20),
                analysis_limit=min(config.analysis_limit, 180),
                warmup_candles=min(config.warmup_candles, 180),
            ),
        ][: max(1, config.max_validation_runs)]

    wider_dataset_limit = max(config.dataset_limit, int(config.dataset_limit * 1.4))
    shorter_horizon = max(40, int(config.horizon_candles * 0.70))
    denser_cutoff_step = max(5, int(config.cutoff_step_candles / 2))

    profiles = [
        CandidateRobustnessProfile(
            validation_name="baseline_window",
            dataset_limit=config.dataset_limit,
            horizon_candles=config.horizon_candles,
            cutoff_step_candles=config.cutoff_step_candles,
            max_cutoffs=config.max_cutoffs,
            analysis_limit=config.analysis_limit,
            warmup_candles=config.warmup_candles,
        ),
        CandidateRobustnessProfile(
            validation_name="shorter_horizon_same_sample",
            dataset_limit=config.dataset_limit,
            horizon_candles=shorter_horizon,
            cutoff_step_candles=config.cutoff_step_candles,
            max_cutoffs=config.max_cutoffs,
            analysis_limit=config.analysis_limit,
            warmup_candles=config.warmup_candles,
        ),
        CandidateRobustnessProfile(
            validation_name="wider_sample_same_horizon",
            dataset_limit=wider_dataset_limit,
            horizon_candles=config.horizon_candles,
            cutoff_step_candles=config.cutoff_step_candles,
            max_cutoffs=config.max_cutoffs,
            analysis_limit=config.analysis_limit,
            warmup_candles=config.warmup_candles,
        ),
        CandidateRobustnessProfile(
            validation_name="denser_cutoffs_same_sample",
            dataset_limit=config.dataset_limit,
            horizon_candles=config.horizon_candles,
            cutoff_step_candles=denser_cutoff_step,
            max_cutoffs=config.max_cutoffs,
            analysis_limit=config.analysis_limit,
            warmup_candles=config.warmup_candles,
        ),
    ]
    return profiles[: max(1, config.max_validation_runs)]



def _candidate_validation_config(
    config: SimpleStrategyCandidateRobustnessConfig,
    profile: CandidateRobustnessProfile,
) -> SimpleStrategyCandidateValidationConfig:
    """
    Converts a robustness profile into the existing candidate validation config.
    """
    return SimpleStrategyCandidateValidationConfig(
        symbol=config.symbol,
        timeframes=config.timeframes,
        analysis_limit=profile.analysis_limit,
        dataset_limit=profile.dataset_limit,
        warmup_candles=profile.warmup_candles,
        horizon_candles=profile.horizon_candles,
        cutoff_step_candles=profile.cutoff_step_candles,
        max_cutoffs=profile.max_cutoffs,
        management_policy_name=config.management_policy_name,
        minimum_rr=config.minimum_rr,
        min_trades_for_candidate=config.min_trades_for_candidate,
        active_only=config.active_only,
    )



def _safe_mean(series: pd.Series) -> float:
    """
    Returns a numeric mean while safely handling empty values.
    """
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else 0.0



def summarize_candidate_robustness(
    combined_summary_df: pd.DataFrame,
    config: SimpleStrategyCandidateRobustnessConfig,
) -> pd.DataFrame:
    """
    Summarizes candidate performance across validation profiles.
    """
    if combined_summary_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    best_per_run_df = (
        combined_summary_df.sort_values(["validation_name", "candidate_score"], ascending=[True, False])
        .groupby("validation_name", as_index=False)
        .head(1)
    )

    for candidate_name, group_df in combined_summary_df.groupby("candidate_name"):
        validation_runs = int(group_df["validation_name"].nunique())
        best_runs = int((best_per_run_df["candidate_name"] == candidate_name).sum())
        total_kept_trades = int(pd.to_numeric(group_df.get("kept_trades"), errors="coerce").fillna(0).sum())
        mean_kept_trades = float(pd.to_numeric(group_df.get("kept_trades"), errors="coerce").fillna(0).mean())

        mean_average_r = _safe_mean(group_df.get("average_managed_r", pd.Series(dtype="float")))
        median_average_r = float(pd.to_numeric(group_df.get("average_managed_r"), errors="coerce").median())
        min_average_r = float(pd.to_numeric(group_df.get("average_managed_r"), errors="coerce").min())
        max_average_r = float(pd.to_numeric(group_df.get("average_managed_r"), errors="coerce").max())
        mean_median_r = _safe_mean(group_df.get("median_managed_r", pd.Series(dtype="float")))
        mean_positive_rate = _safe_mean(group_df.get("positive_exit_rate", pd.Series(dtype="float")))
        mean_loss_rate = _safe_mean(group_df.get("loss_exit_rate", pd.Series(dtype="float")))
        mean_coverage_rate = _safe_mean(group_df.get("coverage_rate", pd.Series(dtype="float")))

        positive_average_r_run_rate = float((pd.to_numeric(group_df.get("average_managed_r"), errors="coerce") > 0).mean())
        positive_median_r_run_rate = float((pd.to_numeric(group_df.get("median_managed_r"), errors="coerce") > 0).mean())
        ready_state_run_rate = float((group_df.get("validation_state", pd.Series(dtype="object")) == "candidate_simple_strategy_needs_larger_validation").mean())
        best_run_rate = best_runs / validation_runs if validation_runs else 0.0

        # Penalize tiny samples and negative average R so the report stays honest.
        sample_penalty = max(0, config.min_total_trades_for_robust_candidate - total_kept_trades) * 2.0
        robust_score = (
            mean_average_r * 120.0
            + mean_median_r * 35.0
            + mean_positive_rate * 35.0
            + positive_average_r_run_rate * 35.0
            + best_run_rate * 20.0
            + mean_coverage_rate * 10.0
            - mean_loss_rate * 20.0
            - sample_penalty
        )

        if total_kept_trades < config.min_total_trades_for_robust_candidate:
            robustness_state = "insufficient_total_trades_do_not_lock"
        elif mean_average_r > 0 and mean_median_r > 0 and positive_average_r_run_rate >= 0.67 and mean_positive_rate >= 0.55:
            robustness_state = "robust_candidate_needs_walk_forward_validation"
        elif mean_average_r > -0.05 and mean_median_r > 0:
            robustness_state = "directionally_promising_but_not_profitable_yet"
        else:
            robustness_state = "not_robust_do_not_lock"

        rows.append(
            {
                "candidate_name": candidate_name,
                "validation_runs": validation_runs,
                "best_candidate_runs": best_runs,
                "best_candidate_run_rate": best_run_rate,
                "total_kept_trades": total_kept_trades,
                "mean_kept_trades_per_run": round(mean_kept_trades, 4),
                "mean_coverage_rate": mean_coverage_rate,
                "mean_positive_exit_rate": mean_positive_rate,
                "mean_loss_exit_rate": mean_loss_rate,
                "mean_average_managed_r": mean_average_r,
                "median_average_managed_r": median_average_r,
                "min_average_managed_r": min_average_r,
                "max_average_managed_r": max_average_r,
                "mean_median_managed_r": mean_median_r,
                "positive_average_r_run_rate": positive_average_r_run_rate,
                "positive_median_r_run_rate": positive_median_r_run_rate,
                "ready_state_run_rate": ready_state_run_rate,
                "robust_candidate_score": robust_score,
                "robustness_state": robustness_state,
            }
        )

    return pd.DataFrame(rows).sort_values("robust_candidate_score", ascending=False).reset_index(drop=True)



def run_simple_strategy_candidate_robustness(
    config: SimpleStrategyCandidateRobustnessConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Runs simple candidate validation across several profiles.
    """
    start_time = perf_counter()
    profiles = build_candidate_robustness_profiles(config)

    summary_frames: list[pd.DataFrame] = []
    workload_rows: list[dict[str, Any]] = []
    preview_frames: list[pd.DataFrame] = []

    for profile in profiles:
        # Each profile builds the pipeline once, then tests all candidate rules from cached rows.
        validation_config = _candidate_validation_config(config, profile)
        dataset_df, summary_df, workload_df, preview_df = run_simple_strategy_candidate_validation(validation_config)

        signature = _config_signature(profile)
        summary_df = summary_df.copy()
        summary_df["validation_name"] = profile.validation_name
        summary_df["validation_config_signature"] = signature
        summary_frames.append(summary_df)

        if not preview_df.empty:
            preview_with_context = preview_df.copy()
            preview_with_context["validation_name"] = profile.validation_name
            preview_frames.append(preview_with_context)

        workload_info = {
            "validation_name": profile.validation_name,
            "validation_config_signature": signature,
            "dataset_limit": profile.dataset_limit,
            "horizon_candles": profile.horizon_candles,
            "cutoff_step_candles": profile.cutoff_step_candles,
            "max_cutoffs": profile.max_cutoffs,
            "analysis_limit": profile.analysis_limit,
            "warmup_candles": profile.warmup_candles,
            "candidate_count": int(len(summary_df)),
        }
        if not workload_df.empty:
            first_workload = workload_df.iloc[0].to_dict()
            workload_info.update(first_workload)
        workload_rows.append(workload_info)

    combined_summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    robustness_df = summarize_candidate_robustness(combined_summary_df, config)
    workload_df = pd.DataFrame(workload_rows)
    preview_df = pd.concat(preview_frames, ignore_index=True) if preview_frames else pd.DataFrame()

    workload_df["elapsed_seconds_total_robustness"] = round(perf_counter() - start_time, 3)
    workload_df["distinct_config_count"] = int(workload_df["validation_config_signature"].nunique()) if not workload_df.empty else 0

    return robustness_df, combined_summary_df, workload_df, preview_df
