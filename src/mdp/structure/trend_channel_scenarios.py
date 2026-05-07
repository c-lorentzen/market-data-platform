from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from mdp.structure.multi_timeframe_trend_channels import (
    analyze_multi_timeframe_channels,
    classify_channel_bias,
    classify_timeframe_purpose,
    parse_timeframes,
    summarize_alignment,
)


@dataclass(frozen=True)
class TrendChannelScenario:
    """
    Stores the final scale-aware channel scenario for the analyzed timeframes.
    """

    symbol: str
    timeframes: str
    alignment_state: str
    scenario_state: str
    actionability: str
    scenario_direction: str
    structure_timeframe: str | None
    entry_timeframe: str | None
    structure_bias: str | None
    entry_bias: str | None
    structure_channel_type: str | None
    entry_channel_type: str | None
    structure_current_state: str | None
    entry_current_state: str | None
    structure_current_position_ratio: float | None
    entry_current_position_ratio: float | None
    bearish_trigger_price: float | None
    bullish_trigger_price: float | None
    bearish_invalidation_price: float | None
    bullish_invalidation_price: float | None
    entry_retest_lower: float | None
    entry_retest_upper: float | None
    structure_retest_lower: float | None
    structure_retest_upper: float | None
    entry_stop_buffer: float | None
    structure_stop_buffer: float | None
    entry_min_target_distance: float | None
    structure_min_target_distance: float | None
    notes: str


def safe_float(value: Any) -> float | None:
    """
    Converts values to float while protecting against missing values.
    """
    # Pandas uses NaN for many empty numeric cells, so treat them as missing.
    if value is None or pd.isna(value):
        return None

    return float(value)


def safe_text(value: Any) -> str | None:
    """
    Converts values to text while protecting against missing values.
    """
    if value is None or pd.isna(value):
        return None

    return str(value)


def pick_best_timeframe_summary(summary_df: pd.DataFrame, purpose: str) -> pd.Series | None:
    """
    Picks the best valid timeframe summary for one purpose.
    """
    if summary_df.empty or "purpose" not in summary_df.columns:
        return None

    purpose_df = summary_df[
        (summary_df["purpose"] == purpose)
        & (summary_df["status"] == "valid_channel")
    ].copy()

    if purpose_df.empty:
        return None

    # Prefer higher rolling quality, then cleaner line fit, then higher containment.
    purpose_df = purpose_df.sort_values(
        by=[
            "best_rolling_quality_score",
            "best_r_squared",
            "best_containment_rate",
        ],
        ascending=[False, False, False],
    )

    return purpose_df.iloc[0]


def pick_structure_summary(summary_df: pd.DataFrame) -> pd.Series | None:
    """
    Picks the best available macro/structure context row.
    """
    # Prefer structure context first because H1/M15 usually define tradable setup structure.
    structure_row = pick_best_timeframe_summary(summary_df, purpose="structure_context")
    if structure_row is not None:
        return structure_row

    # Fall back to macro context when only D1/H4 data exists.
    return pick_best_timeframe_summary(summary_df, purpose="macro_context")


def pick_entry_summary(summary_df: pd.DataFrame) -> pd.Series | None:
    """
    Picks the best available entry-refinement row.
    """
    return pick_best_timeframe_summary(summary_df, purpose="entry_refinement")


def get_timeframe_candidates(candidates_df: pd.DataFrame, timeframe: str | None) -> pd.DataFrame:
    """
    Returns valid channel candidates for one timeframe.
    """
    if candidates_df.empty or timeframe is None:
        return pd.DataFrame()

    return candidates_df[candidates_df["timeframe"] == timeframe].copy()


def pick_candidate_by_role(
    candidates_df: pd.DataFrame,
    timeframe: str | None,
    preferred_role: str,
) -> pd.Series | None:
    """
    Picks the best channel candidate for a specific role.
    """
    timeframe_candidates = get_timeframe_candidates(candidates_df, timeframe)

    if timeframe_candidates.empty:
        return None

    role_df = timeframe_candidates[timeframe_candidates["channel_role"] == preferred_role].copy()

    if role_df.empty:
        return None

    # The candidates are already ranked, but sort again to keep the function deterministic.
    role_df = role_df.sort_values(
        by=["rolling_quality_score", "r_squared", "containment_rate"],
        ascending=[False, False, False],
    )

    return role_df.iloc[0]


def pick_best_candidate(candidates_df: pd.DataFrame, timeframe: str | None) -> pd.Series | None:
    """
    Picks the strongest candidate for one timeframe.
    """
    timeframe_candidates = get_timeframe_candidates(candidates_df, timeframe)

    if timeframe_candidates.empty:
        return None

    timeframe_candidates = timeframe_candidates.sort_values(
        by=["rolling_quality_score", "r_squared", "containment_rate"],
        ascending=[False, False, False],
    )

    return timeframe_candidates.iloc[0]


def has_directional_word(bias: str | None, direction: str) -> bool:
    """
    Checks whether a channel bias contains a bullish or bearish label.
    """
    if bias is None:
        return False

    return direction in bias


def calculate_retest_zone(
    boundary_price: float | None,
    tolerance: float | None,
) -> tuple[float | None, float | None]:
    """
    Builds a retest zone around one channel boundary.
    """
    if boundary_price is None or tolerance is None:
        return None, None

    # Retests are zones, not exact single prices.
    return boundary_price - tolerance, boundary_price + tolerance


def calculate_channel_trigger_prices(
    channel_row: pd.Series | None,
) -> dict[str, float | None]:
    """
    Calculates scale-aware trigger and invalidation levels from one channel row.
    """
    if channel_row is None:
        return {
            "bearish_trigger_price": None,
            "bullish_trigger_price": None,
            "bearish_invalidation_price": None,
            "bullish_invalidation_price": None,
        }

    current_lower = safe_float(channel_row.get("current_lower"))
    current_upper = safe_float(channel_row.get("current_upper"))
    breakout_threshold = safe_float(channel_row.get("adaptive_breakout_threshold"))
    stop_buffer = safe_float(channel_row.get("adaptive_stop_buffer"))

    if breakout_threshold is None:
        breakout_threshold = 0.0

    if stop_buffer is None:
        stop_buffer = 0.0

    # A bearish trigger requires price to break below the active lower boundary by the adaptive threshold.
    bearish_trigger_price = (
        current_lower - breakout_threshold if current_lower is not None else None
    )

    # A bullish trigger requires price to break above the active upper boundary by the adaptive threshold.
    bullish_trigger_price = (
        current_upper + breakout_threshold if current_upper is not None else None
    )

    # If testing bearish continuation, price above the opposite boundary invalidates the setup context.
    bearish_invalidation_price = (
        current_upper + stop_buffer if current_upper is not None else None
    )

    # If testing bullish continuation, price below the opposite boundary invalidates the setup context.
    bullish_invalidation_price = (
        current_lower - stop_buffer if current_lower is not None else None
    )

    return {
        "bearish_trigger_price": bearish_trigger_price,
        "bullish_trigger_price": bullish_trigger_price,
        "bearish_invalidation_price": bearish_invalidation_price,
        "bullish_invalidation_price": bullish_invalidation_price,
    }


def classify_scale_aware_scenario(
    symbol: str,
    timeframes: list[str],
    summary_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
) -> TrendChannelScenario:
    """
    Converts scale-aware channel context into a deterministic scenario label.
    """
    alignment_summary = summarize_alignment(summary_df)
    alignment_state = str(alignment_summary["alignment_state"])

    structure_summary = pick_structure_summary(summary_df)
    entry_summary = pick_entry_summary(summary_df)

    structure_timeframe = safe_text(structure_summary.get("timeframe")) if structure_summary is not None else None
    entry_timeframe = safe_text(entry_summary.get("timeframe")) if entry_summary is not None else None

    structure_candidate = pick_best_candidate(candidates_df, structure_timeframe)

    # Prefer an entry-context channel for SL efficiency, but keep breakout context available.
    entry_candidate = pick_candidate_by_role(
        candidates_df=candidates_df,
        timeframe=entry_timeframe,
        preferred_role="active_entry_context",
    )
    if entry_candidate is None:
        entry_candidate = pick_best_candidate(candidates_df, entry_timeframe)

    # Find a breakout context separately because it may represent a recently broken micro-channel.
    entry_breakout_candidate = pick_candidate_by_role(
        candidates_df=candidates_df,
        timeframe=entry_timeframe,
        preferred_role="active_breakout_context",
    )

    structure_bias = safe_text(structure_summary.get("channel_bias")) if structure_summary is not None else None
    entry_bias = safe_text(entry_summary.get("channel_bias")) if entry_summary is not None else None

    structure_bearish = has_directional_word(structure_bias, "bearish")
    structure_bullish = has_directional_word(structure_bias, "bullish")
    entry_bearish = has_directional_word(entry_bias, "bearish")
    entry_bullish = has_directional_word(entry_bias, "bullish")

    # Detect whether a lower-timeframe broken channel exists even if the best entry channel is still inside.
    lower_breakout_bias = None
    if entry_breakout_candidate is not None:
        lower_breakout_bias = classify_channel_bias(
            channel_type=entry_breakout_candidate.get("channel_type"),
            current_state=entry_breakout_candidate.get("current_state"),
        )

    lower_bearish_breakdown_exists = lower_breakout_bias == "bearish_breakdown"
    lower_bullish_breakout_exists = lower_breakout_bias == "bullish_breakout"

    # Build deterministic scenario labels without calling it a trade signal.
    if structure_bearish and entry_bearish:
        scenario_state = "bearish_context_alignment"
        scenario_direction = "bearish"
        actionability = "watch_for_scale_aware_short_trigger"
        notes = "Structure and entry contexts both lean bearish. This is still a setup context, not an executed trade signal."
    elif structure_bullish and entry_bullish:
        scenario_state = "bullish_context_alignment"
        scenario_direction = "bullish"
        actionability = "watch_for_scale_aware_long_trigger"
        notes = "Structure and entry contexts both lean bullish. This is still a setup context, not an executed trade signal."
    elif structure_bearish and entry_bullish and lower_bearish_breakdown_exists:
        scenario_state = "bearish_structure_with_micro_breakdown_watch"
        scenario_direction = "bearish_watch"
        actionability = "wait_for_retest_or_continuation_confirmation"
        notes = "Structure is bearish, the best entry channel is still bullish/inside, but a lower-timeframe broken channel exists. Wait for retest or continuation confirmation."
    elif structure_bullish and entry_bearish and lower_bullish_breakout_exists:
        scenario_state = "bullish_structure_with_micro_breakout_watch"
        scenario_direction = "bullish_watch"
        actionability = "wait_for_retest_or_continuation_confirmation"
        notes = "Structure is bullish, the best entry channel is still bearish/inside, but a lower-timeframe breakout exists. Wait for retest or continuation confirmation."
    elif structure_bearish and entry_bullish:
        scenario_state = "bearish_structure_bullish_entry_pullback"
        scenario_direction = "mixed"
        actionability = "no_trade_wait_for_breakdown_or_structure_break"
        notes = "Structure is bearish but entry context is bullish. This can be a pullback inside bearish structure or an early reversal attempt."
    elif structure_bullish and entry_bearish:
        scenario_state = "bullish_structure_bearish_entry_pullback"
        scenario_direction = "mixed"
        actionability = "no_trade_wait_for_breakout_or_structure_break"
        notes = "Structure is bullish but entry context is bearish. This can be a pullback inside bullish structure or an early reversal attempt."
    else:
        scenario_state = alignment_state
        scenario_direction = "neutral_or_unclear"
        actionability = "no_trade_insufficient_directional_alignment"
        notes = "The current channel context does not provide enough directional agreement."

    # Use entry candidate for precise trigger prices because lower timeframes refine timing.
    entry_trigger_levels = calculate_channel_trigger_prices(entry_candidate)

    # Use structure candidate for broader invalidation levels when available.
    structure_trigger_levels = calculate_channel_trigger_prices(structure_candidate)

    entry_retest_tolerance = safe_float(entry_candidate.get("adaptive_retest_tolerance")) if entry_candidate is not None else None
    structure_retest_tolerance = safe_float(structure_candidate.get("adaptive_retest_tolerance")) if structure_candidate is not None else None

    # For ascending channels, the lower boundary is usually the useful retest area.
    entry_channel_type = safe_text(entry_candidate.get("channel_type")) if entry_candidate is not None else None
    structure_channel_type = safe_text(structure_candidate.get("channel_type")) if structure_candidate is not None else None

    if entry_channel_type == "ascending":
        entry_boundary_for_retest = safe_float(entry_candidate.get("current_lower"))
    else:
        entry_boundary_for_retest = safe_float(entry_candidate.get("current_upper")) if entry_candidate is not None else None

    if structure_channel_type == "ascending":
        structure_boundary_for_retest = safe_float(structure_candidate.get("current_lower")) if structure_candidate is not None else None
    else:
        structure_boundary_for_retest = safe_float(structure_candidate.get("current_upper")) if structure_candidate is not None else None

    entry_retest_lower, entry_retest_upper = calculate_retest_zone(
        boundary_price=entry_boundary_for_retest,
        tolerance=entry_retest_tolerance,
    )
    structure_retest_lower, structure_retest_upper = calculate_retest_zone(
        boundary_price=structure_boundary_for_retest,
        tolerance=structure_retest_tolerance,
    )

    return TrendChannelScenario(
        symbol=symbol,
        timeframes=",".join(timeframes),
        alignment_state=alignment_state,
        scenario_state=scenario_state,
        actionability=actionability,
        scenario_direction=scenario_direction,
        structure_timeframe=structure_timeframe,
        entry_timeframe=entry_timeframe,
        structure_bias=structure_bias,
        entry_bias=entry_bias,
        structure_channel_type=structure_channel_type,
        entry_channel_type=entry_channel_type,
        structure_current_state=safe_text(structure_candidate.get("current_state")) if structure_candidate is not None else None,
        entry_current_state=safe_text(entry_candidate.get("current_state")) if entry_candidate is not None else None,
        structure_current_position_ratio=safe_float(structure_candidate.get("current_position_ratio")) if structure_candidate is not None else None,
        entry_current_position_ratio=safe_float(entry_candidate.get("current_position_ratio")) if entry_candidate is not None else None,
        bearish_trigger_price=entry_trigger_levels["bearish_trigger_price"],
        bullish_trigger_price=entry_trigger_levels["bullish_trigger_price"],
        bearish_invalidation_price=structure_trigger_levels["bearish_invalidation_price"] or entry_trigger_levels["bearish_invalidation_price"],
        bullish_invalidation_price=structure_trigger_levels["bullish_invalidation_price"] or entry_trigger_levels["bullish_invalidation_price"],
        entry_retest_lower=entry_retest_lower,
        entry_retest_upper=entry_retest_upper,
        structure_retest_lower=structure_retest_lower,
        structure_retest_upper=structure_retest_upper,
        entry_stop_buffer=safe_float(entry_candidate.get("adaptive_stop_buffer")) if entry_candidate is not None else None,
        structure_stop_buffer=safe_float(structure_candidate.get("adaptive_stop_buffer")) if structure_candidate is not None else None,
        entry_min_target_distance=safe_float(entry_candidate.get("adaptive_min_target_distance")) if entry_candidate is not None else None,
        structure_min_target_distance=safe_float(structure_candidate.get("adaptive_min_target_distance")) if structure_candidate is not None else None,
        notes=notes,
    )


def analyze_trend_channel_scenario(
    symbol: str,
    timeframes: str | list[str] | tuple[str, ...],
    limit: int = 300,
    step_size: int = 20,
    top_n_per_timeframe: int = 3,
    active_only: bool = True,
    recent_window: int | None = None,
    atr_window: int | None = None,
    end_time: str | None = None,
    closed_candles_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Runs scale-aware MTF channel analysis and creates a scenario table.
    """
    parsed_timeframes = parse_timeframes(timeframes)

    # Reuse the scale-aware MTF analyzer so scenario logic stays separate from detection.
    summary_df, candidates_df = analyze_multi_timeframe_channels(
        symbol=symbol,
        timeframes=parsed_timeframes,
        limit=limit,
        step_size=step_size,
        top_n_per_timeframe=top_n_per_timeframe,
        active_only=active_only,
        recent_window=recent_window,
        atr_window=atr_window,
        end_time=end_time,
        closed_candles_only=closed_candles_only,
    )

    scenario = classify_scale_aware_scenario(
        symbol=symbol,
        timeframes=parsed_timeframes,
        summary_df=summary_df,
        candidates_df=candidates_df,
    )

    # Return DataFrames so scripts and reports can save or display them consistently.
    scenario_df = pd.DataFrame([asdict(scenario)])

    return scenario_df, summary_df, candidates_df
