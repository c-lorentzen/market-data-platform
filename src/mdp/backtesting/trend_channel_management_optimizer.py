from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from mdp.backtesting.trend_channel_trade_management import build_trend_channel_management_dataset


@dataclass(frozen=True)
class ManagementPolicyProfile:
    """
    Stores one candle-based trade-management policy to test.
    """

    policy_name: str
    partial_take_profit_r: float
    partial_exit_fraction: float
    move_stop_to_breakeven_r: float
    locked_profit_r: float
    trailing_start_r: float
    trailing_distance_r: float
    full_target_mode: str = "nearest"
    time_exit_candles: int | None = None
    intrabar_policy: str = "conservative"


def build_default_management_profiles() -> list[ManagementPolicyProfile]:
    """
    Creates a small set of realistic candle-based management profiles.
    """
    return [
        ManagementPolicyProfile(
            policy_name="conservative_scalp_050_be075_trail125",
            partial_take_profit_r=0.50,
            partial_exit_fraction=0.50,
            move_stop_to_breakeven_r=0.75,
            locked_profit_r=0.00,
            trailing_start_r=1.25,
            trailing_distance_r=0.75,
            full_target_mode="nearest",
        ),
        ManagementPolicyProfile(
            policy_name="fast_scalp_035_be050_trail100",
            partial_take_profit_r=0.35,
            partial_exit_fraction=0.50,
            move_stop_to_breakeven_r=0.50,
            locked_profit_r=0.00,
            trailing_start_r=1.00,
            trailing_distance_r=0.60,
            full_target_mode="nearest",
        ),
        ManagementPolicyProfile(
            policy_name="micro_scalp_025_be040_trail080",
            partial_take_profit_r=0.25,
            partial_exit_fraction=0.50,
            move_stop_to_breakeven_r=0.40,
            locked_profit_r=0.00,
            trailing_start_r=0.80,
            trailing_distance_r=0.50,
            full_target_mode="nearest",
        ),
        ManagementPolicyProfile(
            policy_name="profit_lock_035_lock010_trail100",
            partial_take_profit_r=0.35,
            partial_exit_fraction=0.50,
            move_stop_to_breakeven_r=0.50,
            locked_profit_r=0.10,
            trailing_start_r=1.00,
            trailing_distance_r=0.60,
            full_target_mode="nearest",
        ),
        ManagementPolicyProfile(
            policy_name="fast_scalp_time_exit_30",
            partial_take_profit_r=0.35,
            partial_exit_fraction=0.50,
            move_stop_to_breakeven_r=0.50,
            locked_profit_r=0.00,
            trailing_start_r=1.00,
            trailing_distance_r=0.60,
            full_target_mode="nearest",
            time_exit_candles=30,
        ),
        ManagementPolicyProfile(
            policy_name="targetless_scalp_manage_only",
            partial_take_profit_r=0.35,
            partial_exit_fraction=0.50,
            move_stop_to_breakeven_r=0.50,
            locked_profit_r=0.00,
            trailing_start_r=0.90,
            trailing_distance_r=0.55,
            full_target_mode="none",
            time_exit_candles=45,
        ),
    ]


def _safe_float(value: Any) -> float | None:
    """
    Converts values to float while preserving missing values as None.
    """
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _score_policy(summary_row: pd.Series) -> tuple[float, str]:
    """
    Builds a deterministic ranking score for one management policy.
    """
    managed_trades = int(summary_row.get("managed_trades") or 0)
    average_r = _safe_float(summary_row.get("average_managed_r")) or 0.0
    median_r = _safe_float(summary_row.get("median_managed_r")) or 0.0
    positive_rate = _safe_float(summary_row.get("positive_exit_rate")) or 0.0
    improved_rate = _safe_float(summary_row.get("improved_vs_original_rate")) or 0.0
    partial_rate = _safe_float(summary_row.get("partial_exit_rate")) or 0.0
    loss_count = _safe_float(summary_row.get("loss_exit_count")) or 0.0
    loss_rate = loss_count / managed_trades if managed_trades else 1.0

    # Keep the score simple so it is easy to reason about while testing.
    score = (
        average_r * 60.0
        + median_r * 20.0
        + positive_rate * 25.0
        + improved_rate * 15.0
        + partial_rate * 5.0
        - loss_rate * 20.0
    )

    # Small samples are useful for direction, but not enough for final selection.
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


def compare_management_profiles(
    symbol: str,
    timeframes: str | list[str] | tuple[str, ...],
    profiles: list[ManagementPolicyProfile] | None = None,
    analysis_limit: int = 300,
    dataset_limit: int = 2000,
    start_time: str | None = None,
    end_time: str | None = None,
    warmup_candles: int = 300,
    horizon_candles: int = 120,
    cutoff_step_candles: int = 20,
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
    Tests several candle-based management policies and ranks them.
    """
    if profiles is None:
        profiles = build_default_management_profiles()

    profile_summary_rows: list[dict[str, Any]] = []
    all_trade_rows: list[pd.DataFrame] = []
    cutoff_schedule_df = pd.DataFrame()

    for profile in profiles:
        # Reuse the existing management simulator so every profile is tested equally.
        results_df, summary_df, current_cutoff_schedule_df = build_trend_channel_management_dataset(
            symbol=symbol,
            timeframes=timeframes,
            analysis_limit=analysis_limit,
            dataset_limit=dataset_limit,
            start_time=start_time,
            end_time=end_time,
            warmup_candles=warmup_candles,
            horizon_candles=horizon_candles,
            cutoff_step_candles=cutoff_step_candles,
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

        if cutoff_schedule_df.empty:
            cutoff_schedule_df = current_cutoff_schedule_df

        summary_row = summary_df.iloc[0].to_dict() if not summary_df.empty else {}
        score, policy_quality_state = _score_policy(pd.Series(summary_row))

        # Add the profile settings beside the performance metrics for easier comparison.
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
    return summary_by_policy_df, all_trades_df, cutoff_schedule_df
