from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import pandas as pd

from mdp.backtesting.trend_channel_management_optimizer import ManagementPolicyProfile, build_default_management_profiles
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
class EntryRefinementVariant:
    """
    Stores one simple candle-based entry timing rule.
    """

    variant_name: str
    variant_description: str
    delay_candles: int = 0
    require_directional_confirmation: bool = False


@dataclass(frozen=True)
class EntryRefinementValidationConfig:
    """
    Stores settings for validating candle-based entry refinements.
    """

    symbol: str
    timeframes: str
    analysis_limit: int = 220
    dataset_limit: int = 1200
    warmup_candles: int = 220
    horizon_candles: int = 80
    cutoff_step_candles: int = 20
    max_cutoffs: int | None = 40
    management_policy_name: str = "targetless_scalp_manage_only"
    minimum_rr: float = 1.50
    min_trades_for_candidate: int = 8
    active_only: bool = True


def build_default_entry_refinement_variants() -> list[EntryRefinementVariant]:
    """
    Builds a small, market-agnostic set of entry timing variants.
    """
    return [
        EntryRefinementVariant(
            variant_name="baseline_current_close",
            variant_description="Use the current trade-plan entry exactly as generated.",
            delay_candles=0,
            require_directional_confirmation=False,
        ),
        EntryRefinementVariant(
            variant_name="delay_1_close",
            variant_description="Wait one entry-timeframe candle and enter at that candle close.",
            delay_candles=1,
            require_directional_confirmation=False,
        ),
        EntryRefinementVariant(
            variant_name="delay_2_close",
            variant_description="Wait two entry-timeframe candles and enter at that candle close.",
            delay_candles=2,
            require_directional_confirmation=False,
        ),
        EntryRefinementVariant(
            variant_name="confirm_1_directional_close",
            variant_description="Enter after one candle only if it closes in the trade direction.",
            delay_candles=1,
            require_directional_confirmation=True,
        ),
        EntryRefinementVariant(
            variant_name="confirm_2_directional_close",
            variant_description="Enter after two candles only if the delayed close is still favorable versus the original entry.",
            delay_candles=2,
            require_directional_confirmation=True,
        ),
    ]


def select_management_profile(policy_name: str) -> ManagementPolicyProfile:
    """
    Selects a management policy by name.
    """
    profiles = build_default_management_profiles()
    for profile in profiles:
        if profile.policy_name == policy_name:
            return profile

    available = ", ".join(profile.policy_name for profile in profiles)
    raise ValueError(f"Unknown management policy '{policy_name}'. Available policies: {available}")


def _limit_cutoff_schedule(cutoff_schedule_df: pd.DataFrame, max_cutoffs: int | None) -> pd.DataFrame:
    """
    Limits the historical replay to recent cutoffs when requested.
    """
    if cutoff_schedule_df.empty or max_cutoffs is None or max_cutoffs <= 0:
        return cutoff_schedule_df.reset_index(drop=True)
    if len(cutoff_schedule_df) <= max_cutoffs:
        return cutoff_schedule_df.reset_index(drop=True)

    # Recent cutoffs keep the smoke test fast and focused on the current market stage.
    return cutoff_schedule_df.tail(max_cutoffs).reset_index(drop=True)


def _entry_close_from_future_candles(future_candles: pd.DataFrame, delay_candles: int) -> tuple[float | None, Any, pd.DataFrame]:
    """
    Returns the delayed entry close and remaining future candles after that entry.
    """
    if delay_candles <= 0:
        return None, None, future_candles

    entry_index = delay_candles - 1
    if future_candles.empty or len(future_candles) <= entry_index:
        return None, None, pd.DataFrame()

    entry_candle = future_candles.iloc[entry_index]
    entry_price = safe_float(entry_candle.get("close_price"))
    entry_time = entry_candle.get("timestamp_utc")
    remaining_future = future_candles.iloc[entry_index + 1 :].reset_index(drop=True)
    return entry_price, entry_time, remaining_future


def _directional_confirmation_passed(
    direction: str | None,
    original_entry: float | None,
    delayed_entry: float | None,
) -> bool:
    """
    Checks if the delayed candle close moved in the trade direction.
    """
    if direction not in {"bullish", "bearish"} or original_entry is None or delayed_entry is None:
        return False

    if direction == "bearish":
        return delayed_entry < original_entry

    return delayed_entry > original_entry


def _target_available_from_rr(target_rr: float | None, minimum_rr: float) -> bool:
    """
    Marks a target as available when its recalculated RR passes the minimum RR.
    """
    return target_rr is not None and target_rr >= minimum_rr


def _recalculate_target_rr(
    direction: str,
    entry_price: float,
    risk_points: float,
    target_price: float | None,
) -> float | None:
    """
    Recalculates target RR after a delayed entry price changes the risk/reward profile.
    """
    if target_price is None or risk_points <= 0:
        return None

    if direction == "bearish":
        reward_points = entry_price - target_price
    else:
        reward_points = target_price - entry_price

    if reward_points <= 0:
        return None

    return reward_points / risk_points


def _build_variant_trade_plan(
    trade_plan_row: pd.Series,
    future_candles: pd.DataFrame,
    variant: EntryRefinementVariant,
    minimum_rr: float,
) -> tuple[pd.Series | None, pd.DataFrame, str]:
    """
    Creates a variant trade plan with a delayed or confirmed entry.
    """
    direction = safe_text(trade_plan_row.get("plan_direction"))
    original_entry = safe_float(trade_plan_row.get("entry_price"))
    stop_loss = safe_float(trade_plan_row.get("stop_loss"))

    if direction not in {"bullish", "bearish"} or original_entry is None or stop_loss is None:
        return None, pd.DataFrame(), "invalid_original_trade_plan"

    if variant.delay_candles <= 0:
        # Baseline uses the original entry and all future candles.
        return trade_plan_row.copy(), future_candles.copy(), "entered"

    delayed_entry, delayed_timestamp, remaining_future = _entry_close_from_future_candles(
        future_candles=future_candles,
        delay_candles=variant.delay_candles,
    )
    if delayed_entry is None or remaining_future.empty:
        return None, pd.DataFrame(), "not_enough_future_candles_for_delay"

    if variant.require_directional_confirmation and not _directional_confirmation_passed(
        direction=direction,
        original_entry=original_entry,
        delayed_entry=delayed_entry,
    ):
        return None, pd.DataFrame(), "directional_confirmation_failed"

    if direction == "bearish":
        risk_points = stop_loss - delayed_entry
    else:
        risk_points = delayed_entry - stop_loss

    if risk_points <= 0:
        return None, pd.DataFrame(), "delayed_entry_invalidates_stop"

    variant_row = trade_plan_row.copy()
    variant_row["entry_price"] = delayed_entry
    variant_row["latest_timestamp"] = delayed_timestamp
    variant_row["risk_points"] = risk_points
    variant_row["risk_points_ratio"] = risk_points / delayed_entry if delayed_entry else None
    variant_row["entry_variant_name"] = variant.variant_name

    available_targets: list[str] = []
    best_target: str | None = None
    best_rr: float | None = None

    # Recalculate target RR because the entry changed while structure targets stayed fixed.
    for target_label in ["tp1", "tp2", "tp3"]:
        price_key = f"{target_label}_price"
        rr_key = f"{target_label}_rr"
        available_key = f"{target_label}_available"

        target_price = safe_float(variant_row.get(price_key))
        target_rr = _recalculate_target_rr(
            direction=direction,
            entry_price=delayed_entry,
            risk_points=risk_points,
            target_price=target_price,
        )
        target_available = _target_available_from_rr(target_rr, minimum_rr=minimum_rr)

        variant_row[rr_key] = target_rr
        variant_row[available_key] = target_available

        if target_available:
            label = target_label.upper()
            available_targets.append(label)
            if best_rr is None or (target_rr is not None and target_rr > best_rr):
                best_target = label
                best_rr = target_rr

    variant_row["available_targets"] = ",".join(available_targets) if available_targets else ""
    variant_row["best_available_target"] = best_target
    variant_row["best_available_rr"] = best_rr

    if not available_targets:
        variant_row["target_plan_state"] = "no_available_target_after_entry_refinement"
    elif available_targets == ["TP2"]:
        variant_row["target_plan_state"] = "tp2_only"
    elif available_targets == ["TP2", "TP3"]:
        variant_row["target_plan_state"] = "tp2_tp3_available"
    else:
        variant_row["target_plan_state"] = "_".join(label.lower() for label in available_targets) + "_available"

    return variant_row, remaining_future, "entered"


def _simulate_variant(
    trade_plan_row: pd.Series,
    future_candles: pd.DataFrame,
    policy: ManagementPolicyProfile,
    horizon_candles: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Simulates original terminal outcome and managed outcome for a variant entry.
    """
    original_outcome = simulate_trade_outcome(
        trade_plan_row=trade_plan_row,
        future_candles=future_candles,
        horizon_candles=horizon_candles,
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
    return original_outcome, managed_outcome


def _safe_bool(value: Any) -> bool:
    """
    Converts a common truthy value into bool.
    """
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def build_entry_refinement_dataset(
    config: EntryRefinementValidationConfig,
    variants: list[EntryRefinementVariant] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds a cached validation dataset by testing several entry variants per cutoff.
    """
    start_time = perf_counter()
    timeframe_label = normalize_timeframes(config.timeframes)
    entry_timeframe = infer_entry_timeframe(timeframe_label)
    policy = select_management_profile(config.management_policy_name)
    variants = variants or build_default_entry_refinement_variants()

    cutoff_schedule_df = build_cutoff_schedule(
        symbol=config.symbol,
        entry_timeframe=entry_timeframe,
        dataset_limit=config.dataset_limit,
        warmup_candles=config.warmup_candles,
        horizon_candles=config.horizon_candles + 5,
        cutoff_step_candles=config.cutoff_step_candles,
    )
    original_cutoff_count = len(cutoff_schedule_df)
    cutoff_schedule_df = _limit_cutoff_schedule(cutoff_schedule_df, config.max_cutoffs)

    result_rows: list[dict[str, Any]] = []
    failed_cutoffs = 0
    valid_baseline_trade_plans = 0

    for _, cutoff_row in cutoff_schedule_df.iterrows():
        analysis_cutoff = cutoff_row["analysis_cutoff"]

        try:
            # Build the expensive strategy pipeline once per cutoff, then reuse it for all entry variants.
            trade_plan_df, event_df, scenario_df, _, _ = build_trend_channel_trade_plan(
                symbol=config.symbol,
                timeframes=timeframe_label,
                limit=config.analysis_limit,
                active_only=config.active_only,
                minimum_rr=config.minimum_rr,
                end_time=str(analysis_cutoff),
                closed_candles_only=True,
            )
        except Exception:  # noqa: BLE001 - one bad cutoff should not stop validation.
            failed_cutoffs += 1
            continue

        if trade_plan_df.empty:
            continue

        trade_plan_row = trade_plan_df.iloc[0]
        plan_state = safe_text(trade_plan_row.get("plan_state"))
        if not plan_state or not plan_state.startswith("valid_trade_plan"):
            continue

        valid_baseline_trade_plans += 1
        entry_timestamp = safe_text(trade_plan_row.get("latest_timestamp"))
        row_entry_timeframe = safe_text(trade_plan_row.get("entry_timeframe")) or entry_timeframe
        future_candles = load_future_candles(
            symbol=config.symbol,
            timeframe=row_entry_timeframe,
            entry_timestamp=entry_timestamp,
            horizon_candles=config.horizon_candles + 5,
        )
        if future_candles.empty:
            continue

        scenario_state = safe_text(scenario_df.iloc[0].get("scenario_state")) if not scenario_df.empty else None
        event_state = safe_text(event_df.iloc[0].get("event_state")) if not event_df.empty else None
        event_quality = safe_float(event_df.iloc[0].get("event_quality_score")) if not event_df.empty else None

        for variant in variants:
            variant_row, variant_future, entry_status = _build_variant_trade_plan(
                trade_plan_row=trade_plan_row,
                future_candles=future_candles,
                variant=variant,
                minimum_rr=config.minimum_rr,
            )

            if variant_row is None or variant_future.empty:
                result_rows.append(
                    {
                        "analysis_cutoff": analysis_cutoff,
                        "entry_variant_name": variant.variant_name,
                        "entry_variant_description": variant.variant_description,
                        "entry_status": entry_status,
                        "entered_trade": False,
                        "managed_realized_r": None,
                    }
                )
                continue

            original_outcome, managed_outcome = _simulate_variant(
                trade_plan_row=variant_row,
                future_candles=variant_future,
                policy=policy,
                horizon_candles=config.horizon_candles,
            )

            result_rows.append(
                {
                    "analysis_cutoff": analysis_cutoff,
                    "entry_variant_name": variant.variant_name,
                    "entry_variant_description": variant.variant_description,
                    "entry_status": entry_status,
                    "entered_trade": True,
                    "plan_direction": safe_text(variant_row.get("plan_direction")),
                    "scenario_state": scenario_state,
                    "event_state": event_state,
                    "entry_mode": safe_text(variant_row.get("entry_mode")),
                    "event_quality_score": event_quality,
                    "entry_price": safe_float(variant_row.get("entry_price")),
                    "stop_loss": safe_float(variant_row.get("stop_loss")),
                    "risk_points": safe_float(variant_row.get("risk_points")),
                    "risk_points_ratio": safe_float(variant_row.get("risk_points_ratio")),
                    "available_targets": safe_text(variant_row.get("available_targets")),
                    "best_available_target": safe_text(variant_row.get("best_available_target")),
                    "best_available_rr": safe_float(variant_row.get("best_available_rr")),
                    "target_plan_state": safe_text(variant_row.get("target_plan_state")),
                    "tp1_available": _safe_bool(variant_row.get("tp1_available")),
                    "tp2_available": _safe_bool(variant_row.get("tp2_available")),
                    "tp3_available": _safe_bool(variant_row.get("tp3_available")),
                    "original_outcome_state": safe_text(original_outcome.get("outcome_state")),
                    "original_max_favorable_r": safe_float(original_outcome.get("max_favorable_r")),
                    "original_max_adverse_r": safe_float(original_outcome.get("max_adverse_r")),
                    "managed_exit_state": safe_text(managed_outcome.get("managed_exit_state")),
                    "managed_realized_r": safe_float(managed_outcome.get("managed_realized_r")),
                    "partial_exit_hit": bool(managed_outcome.get("partial_exit_hit")),
                    "breakeven_or_profit_lock_activated": bool(managed_outcome.get("breakeven_or_profit_lock_activated")),
                    "trailing_activated": bool(managed_outcome.get("trailing_activated")),
                    "managed_candles_to_exit": managed_outcome.get("managed_candles_to_exit"),
                }
            )

    dataset_df = pd.DataFrame(result_rows)
    workload_df = pd.DataFrame(
        [
            {
                "original_cutoff_count": original_cutoff_count,
                "used_cutoff_count": len(cutoff_schedule_df),
                "valid_baseline_trade_plans": valid_baseline_trade_plans,
                "entry_variant_count": len(variants),
                "entered_variant_trade_count": int(dataset_df.get("entered_trade", pd.Series(dtype="bool")).fillna(False).sum()) if not dataset_df.empty else 0,
                "failed_cutoffs": failed_cutoffs,
                "management_policy_name": policy.policy_name,
                "estimated_full_pipeline_calls_saved": max(0, valid_baseline_trade_plans * (len(variants) - 1)),
                "elapsed_seconds_total": round(perf_counter() - start_time, 3),
            }
        ]
    )
    return dataset_df, workload_df


def summarize_entry_refinement_variants(
    dataset_df: pd.DataFrame,
    min_trades_for_candidate: int = 8,
) -> pd.DataFrame:
    """
    Summarizes which entry variant performs best on managed outcomes.
    """
    if dataset_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    grouped = dataset_df.groupby("entry_variant_name", dropna=False)
    for variant_name, group_df in grouped:
        entered_df = group_df[group_df.get("entered_trade", False) == True].copy()  # noqa: E712 - pandas comparison is intentional.
        attempted_count = len(group_df)
        entered_count = len(entered_df)
        coverage_rate = entered_count / attempted_count if attempted_count else 0.0
        managed_r = pd.to_numeric(entered_df.get("managed_realized_r"), errors="coerce").dropna()

        positive_rate = float((managed_r > 0).mean()) if not managed_r.empty else 0.0
        loss_rate = float((managed_r < 0).mean()) if not managed_r.empty else 0.0
        average_r = float(managed_r.mean()) if not managed_r.empty else 0.0
        median_r = float(managed_r.median()) if not managed_r.empty else 0.0
        total_r = float(managed_r.sum()) if not managed_r.empty else 0.0
        partial_rate = float(entered_df.get("partial_exit_hit", pd.Series(dtype="bool")).fillna(False).mean()) if entered_count else 0.0
        lock_rate = float(entered_df.get("breakeven_or_profit_lock_activated", pd.Series(dtype="bool")).fillna(False).mean()) if entered_count else 0.0

        # The score rewards positive average R, positive median R, and enough coverage.
        sample_penalty = max(0, min_trades_for_candidate - entered_count) * 8.0
        low_coverage_penalty = 10.0 if 0 < coverage_rate < 0.25 else 0.0
        entry_score = (
            average_r * 80.0
            + median_r * 25.0
            + positive_rate * 35.0
            + coverage_rate * 10.0
            - loss_rate * 15.0
            - sample_penalty
            - low_coverage_penalty
        )

        if entered_count == 0:
            validation_state = "reject_no_trades_entered"
        elif entered_count < min_trades_for_candidate:
            validation_state = "hypothesis_only_insufficient_sample"
        elif average_r > 0 and median_r > 0 and positive_rate >= 0.55:
            validation_state = "candidate_entry_refinement_needs_larger_validation"
        elif average_r <= 0:
            validation_state = "not_profitable_entry_variant_do_not_lock"
        else:
            validation_state = "mixed_entry_variant_needs_more_data"

        rows.append(
            {
                "entry_variant_name": variant_name,
                "attempted_trades": attempted_count,
                "entered_trades": entered_count,
                "coverage_rate": round(coverage_rate, 6),
                "positive_exit_rate": round(positive_rate, 6),
                "loss_exit_rate": round(loss_rate, 6),
                "average_managed_r": round(average_r, 6),
                "median_managed_r": round(median_r, 6),
                "total_managed_r": round(total_r, 6),
                "partial_exit_rate": round(partial_rate, 6),
                "breakeven_or_lock_rate": round(lock_rate, 6),
                "entry_score": round(entry_score, 6),
                "validation_state": validation_state,
            }
        )

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return summary_df

    return summary_df.sort_values(
        by=["validation_state", "entry_score", "entered_trades"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def run_entry_refinement_validation(
    config: EntryRefinementValidationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Runs the full entry-refinement validation workflow.
    """
    dataset_df, workload_df = build_entry_refinement_dataset(config)
    summary_df = summarize_entry_refinement_variants(
        dataset_df=dataset_df,
        min_trades_for_candidate=config.min_trades_for_candidate,
    )
    return dataset_df, summary_df, workload_df
