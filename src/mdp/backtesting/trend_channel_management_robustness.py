from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import pandas as pd

from mdp.backtesting.trend_channel_trade_management import (
    simulate_managed_trade_outcome,
    summarize_managed_outcomes,
)
from mdp.backtesting.trend_channel_management_optimizer import (
    ManagementPolicyProfile,
    build_default_management_profiles,
)
from mdp.backtesting.trend_channel_outcome_dataset import (
    build_cutoff_schedule,
    infer_entry_timeframe,
    load_future_candles,
    normalize_timeframes,
    safe_float,
    safe_text,
    simulate_trade_outcome,
)
from mdp.structure.trend_channel_trade_plan import build_trend_channel_trade_plan


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

    @property
    def config_signature(self) -> str:
        """
        Creates a readable signature so duplicate validation configs are easy to spot.
        """
        return (
            f"dataset={self.dataset_limit}|horizon={self.horizon_candles}|"
            f"cutoff_step={self.cutoff_step_candles}|analysis={self.analysis_limit}"
        )


def _clamp_int(value: float, minimum: int) -> int:
    """
    Converts a numeric value to int while enforcing a minimum value.
    """
    return max(minimum, int(round(value)))


def build_default_validation_runs(
    base_dataset_limit: int = 2000,
    base_horizon_candles: int = 120,
    base_cutoff_step_candles: int = 20,
    analysis_limit: int = 300,
    max_validation_runs: int | None = 3,
) -> list[ManagementValidationRun]:
    """
    Creates distinct validation windows without forcing every run to be expensive.
    """
    shorter_horizon = _clamp_int(base_horizon_candles * 0.67, 40)
    wider_dataset = max(base_dataset_limit + 1000, _clamp_int(base_dataset_limit * 1.50, base_dataset_limit + 1))
    dense_cutoff_step = max(5, base_cutoff_step_candles // 2)

    candidate_runs = [
        ManagementValidationRun(
            validation_name="baseline_recent_window",
            dataset_limit=base_dataset_limit,
            horizon_candles=base_horizon_candles,
            cutoff_step_candles=base_cutoff_step_candles,
            analysis_limit=analysis_limit,
        ),
        ManagementValidationRun(
            validation_name="shorter_horizon_same_sample",
            dataset_limit=base_dataset_limit,
            horizon_candles=shorter_horizon,
            cutoff_step_candles=base_cutoff_step_candles,
            analysis_limit=analysis_limit,
        ),
        ManagementValidationRun(
            validation_name="wider_sample_same_horizon",
            dataset_limit=wider_dataset,
            horizon_candles=base_horizon_candles,
            cutoff_step_candles=base_cutoff_step_candles,
            analysis_limit=analysis_limit,
        ),
        ManagementValidationRun(
            validation_name="denser_cutoff_same_sample",
            dataset_limit=base_dataset_limit,
            horizon_candles=base_horizon_candles,
            cutoff_step_candles=dense_cutoff_step,
            analysis_limit=analysis_limit,
        ),
    ]

    unique_runs: list[ManagementValidationRun] = []
    seen_signatures: set[str] = set()
    for run in candidate_runs:
        # Keep only one copy of exact duplicate parameter sets.
        if run.config_signature in seen_signatures:
            continue
        seen_signatures.add(run.config_signature)
        unique_runs.append(run)

    if max_validation_runs is not None and max_validation_runs > 0:
        return unique_runs[:max_validation_runs]

    return unique_runs


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


def _safe_int(value: Any, default: int = 0) -> int:
    """
    Converts a value to int while protecting against missing data.
    """
    try:
        if value is None or pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _limit_cutoff_schedule(cutoff_schedule_df: pd.DataFrame, max_cutoffs: int | None) -> pd.DataFrame:
    """
    Caps cutoff count so smoke tests finish quickly on a laptop.
    """
    if cutoff_schedule_df.empty or max_cutoffs is None or max_cutoffs <= 0:
        return cutoff_schedule_df

    if len(cutoff_schedule_df) <= max_cutoffs:
        return cutoff_schedule_df

    # Use recent cutoffs because live trading cares most about recent market behavior.
    return cutoff_schedule_df.tail(max_cutoffs).reset_index(drop=True)


def _select_profiles(max_policies: int | None = 4) -> list[ManagementPolicyProfile]:
    """
    Selects a compact policy set first so validation is not brute-forced by default.
    """
    profiles = build_default_management_profiles()
    preferred_order = [
        "targetless_scalp_manage_only",
        "profit_lock_035_lock010_trail100",
        "fast_scalp_035_be050_trail100",
        "micro_scalp_025_be040_trail080",
        "conservative_scalp_050_be075_trail125",
        "fast_scalp_time_exit_30",
    ]
    profiles_by_name = {profile.policy_name: profile for profile in profiles}
    ordered_profiles = [profiles_by_name[name] for name in preferred_order if name in profiles_by_name]

    if max_policies is not None and max_policies > 0:
        return ordered_profiles[:max_policies]

    return ordered_profiles


def _score_policy(summary_row: pd.Series) -> tuple[float, str]:
    """
    Builds a deterministic ranking score for one management policy.
    """
    managed_trades = _safe_int(summary_row.get("managed_trades"))
    average_r = _safe_float(summary_row.get("average_managed_r"))
    median_r = _safe_float(summary_row.get("median_managed_r"))
    positive_rate = _safe_float(summary_row.get("positive_exit_rate"))
    improved_rate = _safe_float(summary_row.get("improved_vs_original_rate"))
    partial_rate = _safe_float(summary_row.get("partial_exit_rate"))
    loss_count = _safe_float(summary_row.get("loss_exit_count"))
    loss_rate = loss_count / managed_trades if managed_trades else 1.0

    score = (
        average_r * 60.0
        + median_r * 20.0
        + positive_rate * 25.0
        + improved_rate * 15.0
        + partial_rate * 5.0
        - loss_rate * 20.0
    )

    if managed_trades < 30:
        score -= 10.0
        quality_state = "small_sample_directional_only"
    elif average_r > 0 and positive_rate >= 0.50:
        quality_state = "candidate_policy_promising"
    elif average_r <= 0 or positive_rate < 0.45:
        quality_state = "candidate_policy_needs_refinement"
    else:
        quality_state = "candidate_policy_mixed"

    return round(score, 6), quality_state


def _classify_robustness(row: pd.Series) -> str:
    """
    Classifies whether a policy is strong enough to consider for candle-based strategy rules.
    """
    validation_runs = _safe_int(row.get("validation_runs"))
    distinct_config_count = _safe_int(row.get("distinct_config_count"))
    total_managed_trades = _safe_int(row.get("total_managed_trades"))
    mean_average_r = _safe_float(row.get("mean_average_managed_r"))
    min_average_r = _safe_float(row.get("min_average_managed_r"))
    mean_positive_rate = _safe_float(row.get("mean_positive_exit_rate"))
    positive_average_r_run_rate = _safe_float(row.get("positive_average_r_run_rate"))
    best_policy_run_rate = _safe_float(row.get("best_policy_run_rate"))

    if validation_runs < 2 or distinct_config_count < 2 or total_managed_trades < 30:
        return "insufficient_sample_for_policy_selection"

    if validation_runs < 3 or distinct_config_count < 3 or total_managed_trades < 100:
        if mean_average_r > 0 and mean_positive_rate >= 0.55:
            return "small_sample_promising_needs_larger_validation"
        return "small_sample_directional_only"

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
    positive_average_r_run_rate = _safe_float(row.get("positive_average_r_run_rate"))
    distinct_config_count = _safe_float(row.get("distinct_config_count"))
    total_trades = _safe_float(row.get("total_managed_trades"))

    score = (
        mean_average_r * 75.0
        + median_average_r * 15.0
        + min_average_r * 25.0
        + mean_median_r * 15.0
        + mean_positive_rate * 30.0
        + best_policy_rate * 20.0
        + positive_average_r_run_rate * 20.0
        - mean_loss_rate * 25.0
    )

    if distinct_config_count < 3:
        score -= 25.0
    if total_trades < 50:
        score -= 20.0
    elif total_trades < 100:
        score -= 8.0

    return round(score, 6)


def _build_output_fingerprint(row: pd.Series) -> str:
    """
    Creates a small fingerprint that helps reveal validation runs with identical outputs.
    """
    policy_name = str(row.get("policy_name"))
    managed_trades = _safe_int(row.get("managed_trades"))
    average_r = round(_safe_float(row.get("average_managed_r")), 6)
    median_r = round(_safe_float(row.get("median_managed_r")), 6)
    positive_rate = round(_safe_float(row.get("positive_exit_rate")), 6)
    return f"{policy_name}|trades={managed_trades}|avg={average_r}|med={median_r}|pos={positive_rate}"


def _build_base_trade_plans_for_run(
    *,
    symbol: str,
    timeframes: str,
    run: ManagementValidationRun,
    start_time: str | None,
    end_time: str | None,
    warmup_candles: int,
    max_cutoffs: int | None,
    step_size: int,
    top_n_per_timeframe: int,
    active_only: bool,
    recent_window: int | None,
    atr_window: int | None,
    event_lookback: int,
    max_retest_candles: int,
    max_false_breakout_candles: int,
    max_confirmation_candles: int,
    min_rejection_wick_ratio: float,
    continuation_multiplier: float,
    minimum_rr: float,
) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    """
    Builds valid trade plans once per validation run and caches future candles in memory.
    """
    timeframe_label = normalize_timeframes(timeframes)
    entry_timeframe = infer_entry_timeframe(timeframe_label)

    cutoff_schedule_df = build_cutoff_schedule(
        symbol=symbol,
        entry_timeframe=entry_timeframe,
        dataset_limit=run.dataset_limit,
        start_time=start_time,
        end_time=end_time,
        warmup_candles=warmup_candles,
        horizon_candles=run.horizon_candles,
        cutoff_step_candles=run.cutoff_step_candles,
    )
    original_cutoff_count = len(cutoff_schedule_df)
    cutoff_schedule_df = _limit_cutoff_schedule(cutoff_schedule_df, max_cutoffs=max_cutoffs)

    cached_rows: list[dict[str, Any]] = []
    failed_cutoffs = 0
    non_actionable_cutoffs = 0

    for _, cutoff_row in cutoff_schedule_df.iterrows():
        analysis_cutoff = cutoff_row["analysis_cutoff"]

        try:
            trade_plan_df, event_df, _scenario_df, _mtf_summary_df, _candidates_df = build_trend_channel_trade_plan(
                symbol=symbol,
                timeframes=timeframe_label,
                limit=run.analysis_limit,
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
                end_time=str(analysis_cutoff),
                closed_candles_only=True,
            )
        except Exception:  # noqa: BLE001 - one bad cutoff should not stop the whole validation.
            failed_cutoffs += 1
            continue

        if trade_plan_df.empty:
            non_actionable_cutoffs += 1
            continue

        trade_plan_row = trade_plan_df.iloc[0]
        plan_state = safe_text(trade_plan_row.get("plan_state"))
        if not plan_state or not plan_state.startswith("valid_trade_plan"):
            non_actionable_cutoffs += 1
            continue

        entry_timestamp = safe_text(trade_plan_row.get("latest_timestamp"))
        row_entry_timeframe = safe_text(trade_plan_row.get("entry_timeframe")) or entry_timeframe
        future_candles = load_future_candles(
            symbol=symbol,
            timeframe=row_entry_timeframe,
            entry_timestamp=entry_timestamp,
            horizon_candles=run.horizon_candles,
        )

        original_outcome = simulate_trade_outcome(
            trade_plan_row=trade_plan_row,
            future_candles=future_candles,
            horizon_candles=run.horizon_candles,
        )
        original_r = original_outcome.get("best_target_rr_hit")
        if original_outcome.get("first_terminal_outcome") == "stop_loss_hit":
            original_r = -1.0
        elif original_outcome.get("first_terminal_outcome") is None:
            original_r = original_outcome.get("max_favorable_r")

        event_row = event_df.iloc[0] if not event_df.empty else pd.Series(dtype="object")
        cached_rows.append(
            {
                "analysis_cutoff": analysis_cutoff,
                "entry_timeframe": row_entry_timeframe,
                "entry_timestamp": entry_timestamp,
                "trade_plan_row": trade_plan_row,
                "event_state": safe_text(event_row.get("event_state")),
                "future_candles": future_candles,
                "original_outcome": original_outcome,
                "original_r": original_r,
            }
        )

    workload_row = {
        "validation_name": run.validation_name,
        "validation_config_signature": run.config_signature,
        "dataset_limit": run.dataset_limit,
        "horizon_candles": run.horizon_candles,
        "cutoff_step_candles": run.cutoff_step_candles,
        "original_cutoff_count": original_cutoff_count,
        "used_cutoff_count": len(cutoff_schedule_df),
        "valid_trade_plan_count": len(cached_rows),
        "non_actionable_cutoffs": non_actionable_cutoffs,
        "failed_cutoffs": failed_cutoffs,
    }
    return cached_rows, cutoff_schedule_df, workload_row


def _simulate_cached_policy(
    *,
    symbol: str,
    timeframes: str,
    cached_rows: list[dict[str, Any]],
    profile: ManagementPolicyProfile,
    exit_at_horizon_close: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies one management policy to cached trade plans without rebuilding the strategy pipeline.
    """
    result_rows: list[dict[str, Any]] = []

    for cached in cached_rows:
        trade_plan_row = cached["trade_plan_row"]
        managed_outcome = simulate_managed_trade_outcome(
            trade_plan_row=trade_plan_row,
            future_candles=cached["future_candles"],
            partial_take_profit_r=profile.partial_take_profit_r,
            partial_exit_fraction=profile.partial_exit_fraction,
            move_stop_to_breakeven_r=profile.move_stop_to_breakeven_r,
            locked_profit_r=profile.locked_profit_r,
            trailing_start_r=profile.trailing_start_r,
            trailing_distance_r=profile.trailing_distance_r,
            full_target_mode=profile.full_target_mode,
            time_exit_candles=profile.time_exit_candles,
            exit_at_horizon_close=exit_at_horizon_close,
            intrabar_policy=profile.intrabar_policy,
        )

        managed_r = managed_outcome.get("managed_realized_r")
        original_r = cached.get("original_r")
        management_improved = None
        if managed_r is not None and original_r is not None:
            management_improved = float(managed_r) > float(original_r)

        original_outcome = cached["original_outcome"]
        result_rows.append(
            {
                "symbol": symbol,
                "timeframes": timeframes,
                "analysis_cutoff": cached.get("analysis_cutoff"),
                "entry_timeframe": cached.get("entry_timeframe"),
                "entry_timestamp": cached.get("entry_timestamp"),
                "plan_state": safe_text(trade_plan_row.get("plan_state")),
                "plan_direction": safe_text(trade_plan_row.get("plan_direction")),
                "event_state": cached.get("event_state"),
                "entry_price": safe_float(trade_plan_row.get("entry_price")),
                "original_stop_loss": safe_float(trade_plan_row.get("stop_loss")),
                "original_risk_points": safe_float(trade_plan_row.get("risk_points")),
                "available_targets": safe_text(trade_plan_row.get("available_targets")),
                "best_available_target": safe_text(trade_plan_row.get("best_available_target")),
                "best_available_rr": safe_float(trade_plan_row.get("best_available_rr")),
                "original_outcome_state": safe_text(original_outcome.get("outcome_state")),
                "original_first_terminal_outcome": safe_text(original_outcome.get("first_terminal_outcome")),
                "original_max_favorable_r": safe_float(original_outcome.get("max_favorable_r")),
                "original_max_adverse_r": safe_float(original_outcome.get("max_adverse_r")),
                "partial_exit_hit": bool(managed_outcome.get("partial_exit_hit")),
                "breakeven_or_profit_lock_activated": bool(managed_outcome.get("breakeven_or_profit_lock_activated")),
                "trailing_activated": bool(managed_outcome.get("trailing_activated")),
                "managed_exit_state": safe_text(managed_outcome.get("managed_exit_state")) or "unknown",
                "managed_realized_r": safe_float(managed_outcome.get("managed_realized_r")),
                "managed_positive_exit": managed_outcome.get("managed_positive_exit"),
                "management_improved_vs_original": management_improved,
                "label_quality": safe_text(managed_outcome.get("label_quality")) or "unknown",
                "notes": safe_text(managed_outcome.get("notes")) or "",
            }
        )

    results_df = pd.DataFrame(result_rows)
    summary_df = summarize_managed_outcomes(results_df)
    return results_df, summary_df


def _summarize_cached_profiles(
    *,
    symbol: str,
    timeframes: str,
    cached_rows: list[dict[str, Any]],
    profiles: list[ManagementPolicyProfile],
    exit_at_horizon_close: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tests all profiles against cached trade plans and returns policy summaries.
    """
    profile_summary_rows: list[dict[str, Any]] = []
    all_trade_rows: list[pd.DataFrame] = []

    for profile in profiles:
        results_df, summary_df = _simulate_cached_policy(
            symbol=symbol,
            timeframes=timeframes,
            cached_rows=cached_rows,
            profile=profile,
            exit_at_horizon_close=exit_at_horizon_close,
        )

        summary_row = summary_df.iloc[0].to_dict() if not summary_df.empty else {}
        score, policy_quality_state = _score_policy(pd.Series(summary_row))
        profile_summary_rows.append(
            {
                **asdict(profile),
                **summary_row,
                "policy_score": score,
                "policy_quality_state": policy_quality_state,
            }
        )

        if not results_df.empty:
            tagged_results_df = results_df.copy()
            tagged_results_df.insert(0, "policy_name", profile.policy_name)
            all_trade_rows.append(tagged_results_df)

    summary_by_policy_df = pd.DataFrame(profile_summary_rows)
    if not summary_by_policy_df.empty:
        summary_by_policy_df = summary_by_policy_df.sort_values("policy_score", ascending=False).reset_index(drop=True)

    all_trades_df = pd.concat(all_trade_rows, ignore_index=True) if all_trade_rows else pd.DataFrame()
    return summary_by_policy_df, all_trades_df


def validate_management_policy_robustness(
    symbol: str,
    timeframes: str | list[str] | tuple[str, ...],
    dataset_limit: int = 1200,
    analysis_limit: int = 200,
    start_time: str | None = None,
    end_time: str | None = None,
    warmup_candles: int = 200,
    horizon_candles: int = 80,
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
    max_cutoffs: int | None = 30,
    max_validation_runs: int | None = 3,
    max_policies: int | None = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Validates management robustness while caching expensive cutoff analysis.
    """
    start = perf_counter()
    timeframe_label = normalize_timeframes(timeframes)
    profiles = _select_profiles(max_policies=max_policies)
    validation_runs = build_default_validation_runs(
        base_dataset_limit=dataset_limit,
        base_horizon_candles=horizon_candles,
        base_cutoff_step_candles=20,
        analysis_limit=analysis_limit,
        max_validation_runs=max_validation_runs,
    )

    validation_summary_frames: list[pd.DataFrame] = []
    validation_trade_frames: list[pd.DataFrame] = []
    best_policy_rows: list[dict[str, Any]] = []
    workload_rows: list[dict[str, Any]] = []

    for run in validation_runs:
        cached_rows, _cutoff_schedule_df, workload_row = _build_base_trade_plans_for_run(
            symbol=symbol,
            timeframes=timeframe_label,
            run=run,
            start_time=start_time,
            end_time=end_time,
            warmup_candles=warmup_candles,
            max_cutoffs=max_cutoffs,
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
        )
        workload_row["policy_count"] = len(profiles)
        workload_row["estimated_full_pipeline_calls_saved"] = max(0, len(profiles) - 1) * workload_row["used_cutoff_count"]
        workload_rows.append(workload_row)

        summary_df, trades_df = _summarize_cached_profiles(
            symbol=symbol,
            timeframes=timeframe_label,
            cached_rows=cached_rows,
            profiles=profiles,
            exit_at_horizon_close=exit_at_horizon_close,
        )

        if not summary_df.empty:
            tagged_summary_df = summary_df.copy()
            tagged_summary_df.insert(0, "validation_name", run.validation_name)
            tagged_summary_df.insert(1, "validation_config_signature", run.config_signature)
            tagged_summary_df.insert(2, "validation_dataset_limit", run.dataset_limit)
            tagged_summary_df.insert(3, "validation_horizon_candles", run.horizon_candles)
            tagged_summary_df.insert(4, "validation_cutoff_step_candles", run.cutoff_step_candles)
            tagged_summary_df["validation_output_fingerprint"] = tagged_summary_df.apply(_build_output_fingerprint, axis=1)
            validation_summary_frames.append(tagged_summary_df)

            best_row = tagged_summary_df.iloc[0].to_dict()
            best_policy_rows.append(
                {
                    "validation_name": run.validation_name,
                    "validation_config_signature": run.config_signature,
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
            tagged_trades_df.insert(1, "validation_config_signature", run.config_signature)
            validation_trade_frames.append(tagged_trades_df)

    all_summary_df = pd.concat(validation_summary_frames, ignore_index=True) if validation_summary_frames else pd.DataFrame()
    best_by_run_df = pd.DataFrame(best_policy_rows)
    workload_df = pd.DataFrame(workload_rows)
    all_trades_df = pd.concat(validation_trade_frames, ignore_index=True) if validation_trade_frames else pd.DataFrame()

    if all_summary_df.empty:
        return pd.DataFrame(), all_summary_df, best_by_run_df, workload_df

    # Mark the first row per validation because each tagged summary is already sorted by policy score.
    all_summary_df["is_best_policy_for_run"] = False
    for _validation_name, group_df in all_summary_df.groupby("validation_name"):
        if group_df.empty:
            continue
        all_summary_df.loc[group_df.index[0], "is_best_policy_for_run"] = True

    distinct_config_count = int(all_summary_df["validation_config_signature"].nunique())
    rows: list[dict[str, Any]] = []

    for policy_name, group_df in all_summary_df.groupby("policy_name", dropna=False):
        managed_trades = pd.to_numeric(group_df.get("managed_trades"), errors="coerce").fillna(0)
        average_r = pd.to_numeric(group_df.get("average_managed_r"), errors="coerce").fillna(0)
        median_r = pd.to_numeric(group_df.get("median_managed_r"), errors="coerce").fillna(0)
        positive_rate = pd.to_numeric(group_df.get("positive_exit_rate"), errors="coerce").fillna(0)
        loss_count = pd.to_numeric(group_df.get("loss_exit_count"), errors="coerce").fillna(0)
        loss_rate = (loss_count / managed_trades.replace(0, pd.NA)).fillna(1.0).astype(float)
        output_fingerprint_count = int(group_df.get("validation_output_fingerprint", pd.Series(dtype=str)).nunique())

        row = {
            "policy_name": policy_name,
            "validation_runs": int(len(group_df)),
            "distinct_config_count": distinct_config_count,
            "distinct_output_count": output_fingerprint_count,
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

    robustness_df = pd.DataFrame(rows).sort_values(
        by=["robust_policy_score", "mean_average_managed_r", "mean_positive_exit_rate"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    elapsed_seconds = round(perf_counter() - start, 3)
    if not workload_df.empty:
        workload_df["elapsed_seconds_total"] = elapsed_seconds
        workload_df["cached_design_note"] = "strategy_pipeline_built_once_per_cutoff_then_reused_across_policies"

    return robustness_df, all_summary_df, best_by_run_df, workload_df
