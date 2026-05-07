from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import pandas as pd

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
from mdp.backtesting.trend_channel_trade_management import simulate_managed_trade_outcome
from mdp.structure.trend_channel_trade_plan import build_trend_channel_trade_plan


@dataclass(frozen=True)
class SetupFilterDiagnosticConfig:
    """
    Stores the lightweight setup-filter diagnostic settings.
    """

    symbol: str
    timeframes: str
    analysis_limit: int = 200
    dataset_limit: int = 1200
    warmup_candles: int = 200
    horizon_candles: int = 80
    cutoff_step_candles: int = 20
    max_cutoffs: int | None = 30
    management_policy_name: str = "targetless_scalp_manage_only"
    minimum_rr: float = 1.50
    active_only: bool = True
    min_group_trades: int = 5


def _safe_float(value: Any, default: float | None = None) -> float | None:
    """
    Converts a value to float without breaking diagnostics on missing data.
    """
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any) -> bool:
    """
    Converts common truthy values to bool for target availability columns.
    """
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _bucket_risk_ratio(value: Any) -> str:
    """
    Groups risk size relative to price so filters are scale-aware.
    """
    number = _safe_float(value)
    if number is None:
        return "unknown"
    if number <= 0.00050:
        return "tiny_risk_ratio"
    if number <= 0.00080:
        return "normal_risk_ratio"
    if number <= 0.00120:
        return "wide_risk_ratio"
    return "very_wide_risk_ratio"


def _bucket_rr(value: Any) -> str:
    """
    Groups the best available target RR into broad ranges.
    """
    number = _safe_float(value)
    if number is None:
        return "unknown"
    if number < 2.0:
        return "rr_under_2"
    if number < 3.0:
        return "rr_2_to_3"
    if number < 5.0:
        return "rr_3_to_5"
    return "rr_over_5"


def _bucket_event_quality(value: Any) -> str:
    """
    Groups event quality so weak scenarios can be diagnosed.
    """
    number = _safe_float(value)
    if number is None:
        return "unknown"
    if number < 0.70:
        return "low_event_quality"
    if number < 0.85:
        return "medium_event_quality"
    return "high_event_quality"


def _limit_cutoff_schedule(cutoff_schedule_df: pd.DataFrame, max_cutoffs: int | None) -> pd.DataFrame:
    """
    Uses recent cutoffs only when a max cutoff limit is provided.
    """
    if cutoff_schedule_df.empty or max_cutoffs is None or max_cutoffs <= 0:
        return cutoff_schedule_df
    if len(cutoff_schedule_df) <= max_cutoffs:
        return cutoff_schedule_df.reset_index(drop=True)

    # Recent samples matter most for current market behavior and keep the run fast.
    return cutoff_schedule_df.tail(max_cutoffs).reset_index(drop=True)


def select_management_profile(policy_name: str) -> ManagementPolicyProfile:
    """
    Selects one management policy by name from the default profile list.
    """
    profiles = build_default_management_profiles()
    for profile in profiles:
        if profile.policy_name == policy_name:
            return profile

    available = ", ".join(profile.policy_name for profile in profiles)
    raise ValueError(f"Unknown management policy '{policy_name}'. Available policies: {available}")


def _extract_event_context(event_df: pd.DataFrame) -> pd.Series:
    """
    Returns the first event row or an empty row when no event was produced.
    """
    if event_df is not None and not event_df.empty:
        return event_df.iloc[0]
    return pd.Series(dtype="object")


def _extract_scenario_context(scenario_df: pd.DataFrame) -> pd.Series:
    """
    Returns the first scenario row or an empty row when no scenario was produced.
    """
    if scenario_df is not None and not scenario_df.empty:
        return scenario_df.iloc[0]
    return pd.Series(dtype="object")


def build_setup_filter_diagnostic_dataset(
    config: SetupFilterDiagnosticConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds a compact dataset that links setup features to managed outcomes.
    """
    start_time = perf_counter()
    timeframe_label = normalize_timeframes(config.timeframes)
    entry_timeframe = infer_entry_timeframe(timeframe_label)
    policy = select_management_profile(config.management_policy_name)

    cutoff_schedule_df = build_cutoff_schedule(
        symbol=config.symbol,
        entry_timeframe=entry_timeframe,
        dataset_limit=config.dataset_limit,
        warmup_candles=config.warmup_candles,
        horizon_candles=config.horizon_candles,
        cutoff_step_candles=config.cutoff_step_candles,
    )
    original_cutoff_count = len(cutoff_schedule_df)
    cutoff_schedule_df = _limit_cutoff_schedule(cutoff_schedule_df, config.max_cutoffs)

    result_rows: list[dict[str, Any]] = []
    failed_cutoffs = 0

    for _, cutoff_row in cutoff_schedule_df.iterrows():
        analysis_cutoff = cutoff_row["analysis_cutoff"]

        try:
            # Build the full trade plan once for this historical cutoff.
            trade_plan_df, event_df, scenario_df, _, _ = build_trend_channel_trade_plan(
                symbol=config.symbol,
                timeframes=timeframe_label,
                limit=config.analysis_limit,
                active_only=config.active_only,
                minimum_rr=config.minimum_rr,
                end_time=str(analysis_cutoff),
                closed_candles_only=True,
            )
        except Exception:  # noqa: BLE001 - one bad cutoff should not stop diagnostics.
            failed_cutoffs += 1
            continue

        if trade_plan_df.empty:
            continue

        trade_plan_row = trade_plan_df.iloc[0]
        plan_state = safe_text(trade_plan_row.get("plan_state"))
        if not plan_state or not plan_state.startswith("valid_trade_plan"):
            continue

        entry_timestamp = safe_text(trade_plan_row.get("latest_timestamp"))
        row_entry_timeframe = safe_text(trade_plan_row.get("entry_timeframe")) or entry_timeframe
        future_candles = load_future_candles(
            symbol=config.symbol,
            timeframe=row_entry_timeframe,
            entry_timestamp=entry_timestamp,
            horizon_candles=config.horizon_candles,
        )

        # Simulate both original target/stop logic and active management logic.
        original_outcome = simulate_trade_outcome(
            trade_plan_row=trade_plan_row,
            future_candles=future_candles,
            horizon_candles=config.horizon_candles,
        )
        managed_outcome = simulate_managed_trade_outcome(
            trade_plan_row=trade_plan_row,
            future_candles=future_candles,
            partial_take_profit_r=policy.partial_take_profit_r,
            partial_exit_fraction=policy.partial_exit_fraction,
            move_stop_to_breakeven_r=policy.move_stop_to_breakeven_r,
            locked_profit_r=policy.locked_profit_r,
            trailing_start_r=policy.trailing_start_r,
            trailing_distance_r=policy.trailing_distance_r,
            full_target_mode=policy.full_target_mode,
            time_exit_candles=policy.time_exit_candles,
            intrabar_policy=policy.intrabar_policy,
        )

        event_row = _extract_event_context(event_df)
        scenario_row = _extract_scenario_context(scenario_df)

        managed_r = _safe_float(managed_outcome.get("managed_realized_r"))
        best_rr = _safe_float(trade_plan_row.get("best_available_rr"))
        risk_ratio = _safe_float(trade_plan_row.get("risk_points_ratio"))
        event_quality = _safe_float(event_row.get("event_quality_score"))

        result_rows.append(
            {
                "analysis_cutoff": analysis_cutoff,
                "entry_timeframe": row_entry_timeframe,
                "plan_state": plan_state,
                "plan_direction": safe_text(trade_plan_row.get("plan_direction")),
                "scenario_state": safe_text(scenario_row.get("scenario_state")),
                "event_state": safe_text(event_row.get("event_state")),
                "entry_mode": safe_text(trade_plan_row.get("entry_mode")),
                "risk_efficiency_state": safe_text(trade_plan_row.get("risk_efficiency_state")),
                "target_plan_state": safe_text(trade_plan_row.get("target_plan_state")),
                "available_targets": safe_text(trade_plan_row.get("available_targets")),
                "best_available_target": safe_text(trade_plan_row.get("best_available_target")),
                "best_available_rr": best_rr,
                "best_rr_bucket": _bucket_rr(best_rr),
                "entry_price": _safe_float(trade_plan_row.get("entry_price")),
                "stop_loss": _safe_float(trade_plan_row.get("stop_loss")),
                "risk_points": _safe_float(trade_plan_row.get("risk_points")),
                "risk_points_ratio": risk_ratio,
                "risk_ratio_bucket": _bucket_risk_ratio(risk_ratio),
                "event_quality_score": event_quality,
                "event_quality_bucket": _bucket_event_quality(event_quality),
                "tp1_available": _safe_bool(trade_plan_row.get("tp1_available")),
                "tp2_available": _safe_bool(trade_plan_row.get("tp2_available")),
                "tp3_available": _safe_bool(trade_plan_row.get("tp3_available")),
                "original_outcome_state": safe_text(original_outcome.get("outcome_state")),
                "original_first_terminal_outcome": safe_text(original_outcome.get("first_terminal_outcome")),
                "original_max_favorable_r": _safe_float(original_outcome.get("max_favorable_r")),
                "original_max_adverse_r": _safe_float(original_outcome.get("max_adverse_r")),
                "management_policy_name": policy.policy_name,
                "managed_exit_state": safe_text(managed_outcome.get("managed_exit_state")),
                "managed_realized_r": managed_r,
                "managed_positive_exit": managed_r is not None and managed_r > 0,
                "partial_exit_hit": bool(managed_outcome.get("partial_exit_hit")),
                "breakeven_or_profit_lock_activated": bool(managed_outcome.get("breakeven_or_profit_lock_activated")),
                "trailing_activated": bool(managed_outcome.get("trailing_activated")),
                "managed_candles_to_exit": managed_outcome.get("managed_candles_to_exit"),
                "managed_label_quality": safe_text(managed_outcome.get("label_quality")),
            }
        )

    dataset_df = pd.DataFrame(result_rows)
    workload_df = pd.DataFrame(
        [
            {
                "original_cutoff_count": original_cutoff_count,
                "used_cutoff_count": len(cutoff_schedule_df),
                "valid_trade_plan_count": len(dataset_df),
                "failed_cutoffs": failed_cutoffs,
                "management_policy_name": policy.policy_name,
                "elapsed_seconds_total": round(perf_counter() - start_time, 3),
            }
        ]
    )
    return dataset_df, workload_df


def _summarize_group(group_df: pd.DataFrame, feature_name: str, feature_value: str, min_group_trades: int) -> dict[str, Any]:
    """
    Summarizes one setup-feature group.
    """
    managed_r = pd.to_numeric(group_df["managed_realized_r"], errors="coerce").dropna()
    trades = len(group_df)
    positive_rate = float((managed_r > 0).mean()) if not managed_r.empty else 0.0
    average_r = float(managed_r.mean()) if not managed_r.empty else 0.0
    median_r = float(managed_r.median()) if not managed_r.empty else 0.0
    loss_rate = float((managed_r < 0).mean()) if not managed_r.empty else 0.0
    partial_rate = float(group_df["partial_exit_hit"].fillna(False).mean()) if trades else 0.0
    lock_rate = float(group_df["breakeven_or_profit_lock_activated"].fillna(False).mean()) if trades else 0.0

    if trades < min_group_trades:
        recommendation = "insufficient_sample"
    elif average_r > 0 and positive_rate >= 0.55:
        recommendation = "candidate_keep_filter"
    elif average_r <= -0.10 or positive_rate < 0.45:
        recommendation = "candidate_avoid_filter"
    else:
        recommendation = "neutral_watch_filter"

    return {
        "feature_name": feature_name,
        "feature_value": feature_value,
        "trades": trades,
        "positive_exit_rate": round(positive_rate, 6),
        "average_managed_r": round(average_r, 6),
        "median_managed_r": round(median_r, 6),
        "loss_exit_rate": round(loss_rate, 6),
        "partial_exit_rate": round(partial_rate, 6),
        "breakeven_or_lock_rate": round(lock_rate, 6),
        "filter_recommendation": recommendation,
    }


def analyze_setup_filter_groups(
    dataset_df: pd.DataFrame,
    min_group_trades: int = 5,
) -> pd.DataFrame:
    """
    Finds which setup features are associated with better or worse managed outcomes.
    """
    if dataset_df.empty:
        return pd.DataFrame()

    feature_columns = [
        "plan_direction",
        "scenario_state",
        "event_state",
        "entry_mode",
        "risk_efficiency_state",
        "target_plan_state",
        "best_rr_bucket",
        "risk_ratio_bucket",
        "event_quality_bucket",
    ]

    summary_rows: list[dict[str, Any]] = []
    for feature_column in feature_columns:
        if feature_column not in dataset_df.columns:
            continue
        for feature_value, group_df in dataset_df.groupby(feature_column, dropna=False):
            summary_rows.append(
                _summarize_group(
                    group_df=group_df,
                    feature_name=feature_column,
                    feature_value=str(feature_value),
                    min_group_trades=min_group_trades,
                )
            )

    # Add simple two-feature combinations because single features may hide interaction effects.
    combo_columns = [
        ("event_state", "target_plan_state"),
        ("event_state", "risk_ratio_bucket"),
        ("event_state", "best_rr_bucket"),
        ("entry_mode", "risk_ratio_bucket"),
    ]
    for left_column, right_column in combo_columns:
        if left_column not in dataset_df.columns or right_column not in dataset_df.columns:
            continue
        grouped = dataset_df.groupby([left_column, right_column], dropna=False)
        for (left_value, right_value), group_df in grouped:
            summary_rows.append(
                _summarize_group(
                    group_df=group_df,
                    feature_name=f"{left_column}+{right_column}",
                    feature_value=f"{left_value}|{right_value}",
                    min_group_trades=min_group_trades,
                )
            )

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        return summary_df

    return summary_df.sort_values(
        by=["filter_recommendation", "trades", "average_managed_r"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def summarize_setup_filter_dataset(dataset_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a one-row summary of the setup-filter diagnostic dataset.
    """
    if dataset_df.empty:
        return pd.DataFrame(
            [
                {
                    "diagnostic_state": "no_valid_trade_plans",
                    "managed_trades": 0,
                    "positive_exit_rate": 0.0,
                    "average_managed_r": None,
                    "median_managed_r": None,
                    "loss_exit_rate": 0.0,
                }
            ]
        )

    managed_r = pd.to_numeric(dataset_df["managed_realized_r"], errors="coerce").dropna()
    managed_trades = len(dataset_df)
    positive_rate = float((managed_r > 0).mean()) if not managed_r.empty else 0.0
    average_r = float(managed_r.mean()) if not managed_r.empty else 0.0
    median_r = float(managed_r.median()) if not managed_r.empty else 0.0
    loss_rate = float((managed_r < 0).mean()) if not managed_r.empty else 0.0

    if managed_trades < 30:
        diagnostic_state = "small_sample_directional_only"
    elif average_r > 0 and positive_rate >= 0.52:
        diagnostic_state = "setup_filtering_candidate_promising"
    elif average_r <= 0:
        diagnostic_state = "setup_filtering_needed_average_r_negative"
    else:
        diagnostic_state = "setup_filtering_mixed"

    return pd.DataFrame(
        [
            {
                "diagnostic_state": diagnostic_state,
                "managed_trades": managed_trades,
                "positive_exit_rate": round(positive_rate, 6),
                "average_managed_r": round(average_r, 6),
                "median_managed_r": round(median_r, 6),
                "loss_exit_rate": round(loss_rate, 6),
                "partial_exit_rate": round(float(dataset_df["partial_exit_hit"].fillna(False).mean()), 6),
                "breakeven_or_lock_rate": round(float(dataset_df["breakeven_or_profit_lock_activated"].fillna(False).mean()), 6),
                "trailing_activation_rate": round(float(dataset_df["trailing_activated"].fillna(False).mean()), 6),
            }
        ]
    )


def run_setup_filter_diagnostics(
    config: SetupFilterDiagnosticConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Runs the full setup-filter diagnostic workflow.
    """
    dataset_df, workload_df = build_setup_filter_diagnostic_dataset(config)
    summary_df = summarize_setup_filter_dataset(dataset_df)
    group_df = analyze_setup_filter_groups(dataset_df, min_group_trades=config.min_group_trades)
    return dataset_df, summary_df, group_df, workload_df
