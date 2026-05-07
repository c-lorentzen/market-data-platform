from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from mdp.database.candle_loader import load_market_candles
from mdp.database.candle_loader_asof import load_market_candles_as_of
from mdp.structure.market_scale import MarketScaleMetrics, calculate_market_scale
from mdp.structure.rolling_trend_channels import detect_rolling_trend_channels


@dataclass(frozen=True)
class TimeframeChannelProfile:
    """
    Stores base detection defaults before adaptive scale values are applied.
    """

    window_sizes: tuple[int, ...]
    min_rolling_quality_score: float
    min_quality_score: float


def parse_timeframes(timeframes: str | list[str] | tuple[str, ...]) -> list[str]:
    """
    Converts timeframe input into a clean uppercase list.
    """
    # Allow command-line input like "D1,H4,H1,M15,M5,M1".
    if isinstance(timeframes, str):
        raw_values = timeframes.replace(",", " ").split()
    else:
        raw_values = list(timeframes)

    clean_timeframes = []

    # Keep order while removing duplicate timeframe labels.
    for timeframe in raw_values:
        normalized = str(timeframe).strip().upper()
        if normalized and normalized not in clean_timeframes:
            clean_timeframes.append(normalized)

    if not clean_timeframes:
        raise ValueError("At least one timeframe is required.")

    return clean_timeframes


def classify_timeframe_purpose(timeframe: str) -> str:
    """
    Labels the role of each timeframe in the trading analysis workflow.
    """
    normalized_timeframe = timeframe.upper()

    # Higher timeframes define directional context, not entries.
    if normalized_timeframe in {"D1", "H4"}:
        return "macro_context"

    # Middle timeframes define structure and setup quality.
    if normalized_timeframe in {"H1", "M15"}:
        return "structure_context"

    # Lower timeframes refine entry timing and invalidation.
    return "entry_refinement"


def get_timeframe_channel_profile(timeframe: str) -> TimeframeChannelProfile:
    """
    Returns base window settings for the timeframe group.
    """
    normalized_timeframe = timeframe.upper()

    # Macro channels can use fewer candles because each candle already covers a large period.
    if normalized_timeframe in {"D1", "H4"}:
        return TimeframeChannelProfile(
            window_sizes=(40, 60, 80, 120),
            min_rolling_quality_score=0.55,
            min_quality_score=0.45,
        )

    # H1/M15 should describe tradable structure without becoming too sensitive.
    if normalized_timeframe in {"H1", "M15"}:
        return TimeframeChannelProfile(
            window_sizes=(60, 80, 120, 180),
            min_rolling_quality_score=0.60,
            min_quality_score=0.50,
        )

    # M5/M1 should be stricter because they are used for entry refinement.
    return TimeframeChannelProfile(
        window_sizes=(60, 80, 120, 180),
        min_rolling_quality_score=0.70,
        min_quality_score=0.55,
    )


def safe_ratio(numerator: float | None, denominator: float | None, fallback: float = 0.0) -> float:
    """
    Calculates a ratio while avoiding zero or missing values.
    """
    # Missing or zero denominators would make the threshold unreliable.
    if numerator is None or denominator is None or denominator <= 0:
        return fallback

    return float(numerator / denominator)


def build_scale_aware_detection_settings(
    timeframe: str,
    scale_metrics: MarketScaleMetrics,
) -> dict[str, Any]:
    """
    Converts adaptive market scale values into channel detection thresholds.
    """
    profile = get_timeframe_channel_profile(timeframe)

    recent_range = scale_metrics.recent_median_candle_range or scale_metrics.median_candle_range
    median_close = scale_metrics.median_close or scale_metrics.latest_close

    # Use adaptive width bounds from the market scale module.
    min_channel_width = float(scale_metrics.channel_min_width or 0.0)
    max_channel_width = float(scale_metrics.channel_max_width or 0.0)

    # Convert absolute adaptive channel width into the ratio expected by rolling detection.
    max_channel_width_ratio = safe_ratio(
        numerator=max_channel_width,
        denominator=median_close,
        fallback=0.02,
    )

    # Convert absolute adaptive width bounds into candle-range multiples.
    min_width_to_range_ratio = safe_ratio(
        numerator=min_channel_width,
        denominator=recent_range,
        fallback=1.0,
    )
    max_width_to_range_ratio = safe_ratio(
        numerator=max_channel_width,
        denominator=recent_range,
        fallback=8.0,
    )

    # Convert adaptive retest tolerance into the detector's range multiplier.
    tolerance_range_multiplier = safe_ratio(
        numerator=scale_metrics.retest_tolerance,
        denominator=recent_range,
        fallback=0.50,
    )

    # Keep tolerance settings realistic if the market scale data is unusual.
    tolerance_range_multiplier = max(0.20, min(tolerance_range_multiplier, 0.80))

    return {
        "window_sizes": list(profile.window_sizes),
        "min_quality_score": profile.min_quality_score,
        "min_rolling_quality_score": profile.min_rolling_quality_score,
        "min_channel_width": min_channel_width,
        "max_channel_width": max_channel_width,
        "max_channel_width_ratio": max_channel_width_ratio,
        "min_width_to_range_ratio": max(0.50, min_width_to_range_ratio),
        "max_width_to_range_ratio": max(1.50, max_width_to_range_ratio),
        "min_r_squared": float(scale_metrics.suggested_min_r_squared or 0.0),
        "min_containment_rate": float(scale_metrics.suggested_min_containment_rate or 0.0),
        "tolerance_range_multiplier": tolerance_range_multiplier,
    }


def classify_channel_bias(channel_type: str | None, current_state: str | None) -> str:
    """
    Converts channel type and price state into a measurable directional context label.
    """
    if channel_type is None or current_state is None:
        return "no_channel"

    channel_type = str(channel_type)
    current_state = str(current_state)

    # An ascending channel usually supports bullish continuation while price remains inside.
    if channel_type == "ascending":
        if current_state == "above_channel":
            return "bullish_breakout"
        if current_state == "below_channel":
            return "bearish_breakdown"
        return "bullish_channel_context"

    # A descending channel usually supports bearish continuation while price remains inside.
    if channel_type == "descending":
        if current_state == "above_channel":
            return "bullish_breakout"
        if current_state == "below_channel":
            return "bearish_breakdown"
        return "bearish_channel_context"

    return "neutral_channel_context"


def filter_channels_for_scale(
    rolling_channels_df: pd.DataFrame,
    settings: dict[str, Any],
) -> pd.DataFrame:
    """
    Applies adaptive scale-aware quality filters to rolling channel candidates.
    """
    if rolling_channels_df.empty:
        return rolling_channels_df

    filtered_df = rolling_channels_df.copy()

    # Keep only channels with enough combined quality for this timeframe purpose.
    filtered_df = filtered_df[
        filtered_df["rolling_quality_score"] >= settings["min_rolling_quality_score"]
    ]

    # Keep channels with enough line fit for the current timeframe scale.
    filtered_df = filtered_df[filtered_df["r_squared"] >= settings["min_r_squared"]]

    # Keep channels with enough historical containment for this timeframe purpose.
    filtered_df = filtered_df[
        filtered_df["containment_rate"] >= settings["min_containment_rate"]
    ]

    # Reject channels that are too narrow to be structurally meaningful.
    filtered_df = filtered_df[
        filtered_df["channel_width"] >= settings["min_channel_width"]
    ]

    # Reject channels that are too wide for the current timeframe market stage.
    filtered_df = filtered_df[
        filtered_df["channel_width"] <= settings["max_channel_width"]
    ]

    # Keep the ratio filters too, so scale works relative to both price and candle range.
    filtered_df = filtered_df[
        filtered_df["channel_width_ratio"] <= settings["max_channel_width_ratio"]
    ]
    filtered_df = filtered_df[
        filtered_df["width_to_range_ratio"] >= settings["min_width_to_range_ratio"]
    ]
    filtered_df = filtered_df[
        filtered_df["width_to_range_ratio"] <= settings["max_width_to_range_ratio"]
    ]

    return filtered_df.reset_index(drop=True)


def summarize_timeframe_channels(
    timeframe: str,
    candles_df: pd.DataFrame,
    raw_channels_df: pd.DataFrame,
    filtered_channels_df: pd.DataFrame,
    scale_metrics: MarketScaleMetrics | None,
    settings: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Builds one summary row for a timeframe.
    """
    purpose = classify_timeframe_purpose(timeframe)

    # Return a safe row when no candles exist for this timeframe.
    if candles_df.empty:
        return {
            "timeframe": timeframe,
            "purpose": purpose,
            "status": "no_data",
            "candles_analyzed": 0,
            "raw_channels": 0,
            "valid_channels": 0,
            "best_channel_role": None,
            "best_channel_type": None,
            "best_current_state": None,
            "channel_bias": "no_data",
            "best_window_size": None,
            "best_rolling_quality_score": None,
            "best_r_squared": None,
            "best_containment_rate": None,
            "best_channel_width": None,
            "best_channel_width_ratio": None,
            "best_width_to_range_ratio": None,
            "volatility_stage": None,
            "adaptive_channel_min_width": None,
            "adaptive_channel_max_width": None,
            "adaptive_breakout_threshold": None,
            "adaptive_retest_tolerance": None,
            "adaptive_stop_buffer": None,
            "adaptive_min_target_distance": None,
            "latest_close": None,
            "latest_timestamp": None,
        }

    latest_row = candles_df.iloc[-1]

    scale_status = scale_metrics.status if scale_metrics is not None else "no_scale"

    # Return a safe row when channels exist but none passed the adaptive filters.
    if filtered_channels_df.empty:
        return {
            "timeframe": timeframe,
            "purpose": purpose,
            "status": "no_valid_channel",
            "scale_status": scale_status,
            "candles_analyzed": int(len(candles_df)),
            "raw_channels": int(len(raw_channels_df)),
            "valid_channels": 0,
            "best_channel_role": None,
            "best_channel_type": None,
            "best_current_state": None,
            "channel_bias": "no_valid_channel",
            "best_window_size": None,
            "best_rolling_quality_score": None,
            "best_r_squared": None,
            "best_containment_rate": None,
            "best_channel_width": None,
            "best_channel_width_ratio": None,
            "best_width_to_range_ratio": None,
            "volatility_stage": getattr(scale_metrics, "volatility_stage", None),
            "adaptive_channel_min_width": getattr(scale_metrics, "channel_min_width", None),
            "adaptive_channel_max_width": getattr(scale_metrics, "channel_max_width", None),
            "adaptive_breakout_threshold": getattr(scale_metrics, "breakout_threshold", None),
            "adaptive_retest_tolerance": getattr(scale_metrics, "retest_tolerance", None),
            "adaptive_stop_buffer": getattr(scale_metrics, "stop_buffer", None),
            "adaptive_min_target_distance": getattr(scale_metrics, "min_target_distance", None),
            "latest_close": float(latest_row["close_price"]),
            "latest_timestamp": latest_row["timestamp_utc"],
        }

    # The first row is already ranked by rolling quality.
    best_channel = filtered_channels_df.iloc[0]
    channel_bias = classify_channel_bias(
        channel_type=best_channel["channel_type"],
        current_state=best_channel["current_state"],
    )

    return {
        "timeframe": timeframe,
        "purpose": purpose,
        "status": "valid_channel",
        "scale_status": scale_status,
        "candles_analyzed": int(len(candles_df)),
        "raw_channels": int(len(raw_channels_df)),
        "valid_channels": int(len(filtered_channels_df)),
        "best_channel_role": best_channel["channel_role"],
        "best_channel_type": best_channel["channel_type"],
        "best_current_state": best_channel["current_state"],
        "channel_bias": channel_bias,
        "best_window_size": int(best_channel["window_size"]),
        "best_rolling_quality_score": float(best_channel["rolling_quality_score"]),
        "best_r_squared": float(best_channel["r_squared"]),
        "best_containment_rate": float(best_channel["containment_rate"]),
        "best_channel_width": float(best_channel["channel_width"]),
        "best_channel_width_ratio": float(best_channel["channel_width_ratio"]),
        "best_width_to_range_ratio": float(best_channel["width_to_range_ratio"]),
        "volatility_stage": getattr(scale_metrics, "volatility_stage", None),
        "adaptive_channel_min_width": getattr(scale_metrics, "channel_min_width", None),
        "adaptive_channel_max_width": getattr(scale_metrics, "channel_max_width", None),
        "adaptive_breakout_threshold": getattr(scale_metrics, "breakout_threshold", None),
        "adaptive_retest_tolerance": getattr(scale_metrics, "retest_tolerance", None),
        "adaptive_stop_buffer": getattr(scale_metrics, "stop_buffer", None),
        "adaptive_min_target_distance": getattr(scale_metrics, "min_target_distance", None),
        "latest_close": float(latest_row["close_price"]),
        "latest_timestamp": latest_row["timestamp_utc"],
    }


def add_scale_columns_to_candidates(
    candidates_df: pd.DataFrame,
    scale_metrics: MarketScaleMetrics,
) -> pd.DataFrame:
    """
    Adds adaptive threshold context to candidate channel rows.
    """
    if candidates_df.empty:
        return candidates_df

    enriched_df = candidates_df.copy()

    # These columns show which adaptive scale bounds each channel was judged against.
    enriched_df["volatility_stage"] = scale_metrics.volatility_stage
    enriched_df["adaptive_channel_min_width"] = scale_metrics.channel_min_width
    enriched_df["adaptive_channel_max_width"] = scale_metrics.channel_max_width
    enriched_df["adaptive_breakout_threshold"] = scale_metrics.breakout_threshold
    enriched_df["adaptive_retest_tolerance"] = scale_metrics.retest_tolerance
    enriched_df["adaptive_stop_buffer"] = scale_metrics.stop_buffer
    enriched_df["adaptive_min_target_distance"] = scale_metrics.min_target_distance

    # This tells us how much room remains before a channel becomes too wide for the timeframe.
    if scale_metrics.channel_max_width and scale_metrics.channel_max_width > 0:
        enriched_df["channel_width_usage_ratio"] = (
            enriched_df["channel_width"] / float(scale_metrics.channel_max_width)
        )
    else:
        enriched_df["channel_width_usage_ratio"] = None

    return enriched_df


def analyze_timeframe_channels(
    symbol: str,
    timeframe: str,
    limit: int = 300,
    step_size: int = 20,
    top_n_per_timeframe: int = 3,
    active_only: bool = True,
    left_window: int = 3,
    right_window: int = 3,
    envelope_quantile: float = 0.90,
    min_anchor_pivots: int = 3,
    recent_window: int | None = None,
    atr_window: int | None = None,
    end_time: str | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Runs scale-aware rolling channel analysis for one timeframe.
    """
    normalized_timeframe = timeframe.upper()

    # Load this timeframe independently because each timeframe has its own market scale.
    if end_time is not None:
        candles_df = load_market_candles_as_of(
            symbol=symbol,
            timeframe=normalized_timeframe,
            end_time=end_time,
            limit=limit,
        )
    else:
        candles_df = load_market_candles(
            symbol=symbol,
            timeframe=normalized_timeframe,
            limit=limit,
        )

    if candles_df.empty:
        summary = summarize_timeframe_channels(
            timeframe=normalized_timeframe,
            candles_df=candles_df,
            raw_channels_df=pd.DataFrame(),
            filtered_channels_df=pd.DataFrame(),
            scale_metrics=None,
            settings=None,
        )
        return summary, pd.DataFrame()

    # Calculate adaptive timeframe scale before detecting channels.
    scale_metrics = calculate_market_scale(
        candles_df=candles_df,
        timeframe=normalized_timeframe,
        recent_window=recent_window,
        atr_window=atr_window,
    )

    if scale_metrics.status != "valid_scale":
        summary = summarize_timeframe_channels(
            timeframe=normalized_timeframe,
            candles_df=candles_df,
            raw_channels_df=pd.DataFrame(),
            filtered_channels_df=pd.DataFrame(),
            scale_metrics=scale_metrics,
            settings=None,
        )
        return summary, pd.DataFrame()

    settings = build_scale_aware_detection_settings(
        timeframe=normalized_timeframe,
        scale_metrics=scale_metrics,
    )

    # Detect active rolling channels using adaptive scale values for this timeframe.
    raw_channels_df = detect_rolling_trend_channels(
        candles_df=candles_df,
        window_sizes=settings["window_sizes"],
        step_size=step_size,
        top_n=None,
        left_window=left_window,
        right_window=right_window,
        envelope_quantile=envelope_quantile,
        tolerance_range_multiplier=settings["tolerance_range_multiplier"],
        min_anchor_pivots=min_anchor_pivots,
        min_quality_score=settings["min_quality_score"],
        min_rolling_quality_score=0.0,
        max_channel_width_ratio=settings["max_channel_width_ratio"],
        min_width_to_range_ratio=settings["min_width_to_range_ratio"],
        max_width_to_range_ratio=settings["max_width_to_range_ratio"],
        active_only=active_only,
    )

    # Apply adaptive filters after detection so the report can show raw vs valid counts.
    filtered_channels_df = filter_channels_for_scale(
        rolling_channels_df=raw_channels_df,
        settings=settings,
    )

    # Keep only the strongest valid candidates for reporting.
    if top_n_per_timeframe is not None and top_n_per_timeframe > 0:
        filtered_channels_df = filtered_channels_df.head(top_n_per_timeframe).reset_index(drop=True)

    summary = summarize_timeframe_channels(
        timeframe=normalized_timeframe,
        candles_df=candles_df,
        raw_channels_df=raw_channels_df,
        filtered_channels_df=filtered_channels_df,
        scale_metrics=scale_metrics,
        settings=settings,
    )

    # Add symbol, timeframe, purpose, and adaptive scale values to candidate rows.
    if not filtered_channels_df.empty:
        filtered_channels_df = add_scale_columns_to_candidates(
            candidates_df=filtered_channels_df,
            scale_metrics=scale_metrics,
        )
        filtered_channels_df.insert(0, "symbol", symbol)
        filtered_channels_df.insert(1, "timeframe", normalized_timeframe)
        filtered_channels_df.insert(2, "purpose", classify_timeframe_purpose(normalized_timeframe))

    return summary, filtered_channels_df


def analyze_multi_timeframe_channels(
    symbol: str,
    timeframes: str | list[str] | tuple[str, ...],
    limit: int = 300,
    step_size: int = 20,
    top_n_per_timeframe: int = 3,
    active_only: bool = True,
    recent_window: int | None = None,
    atr_window: int | None = None,
    end_time: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs scale-aware active rolling channel analysis across several timeframes.
    """
    parsed_timeframes = parse_timeframes(timeframes)
    summary_rows = []
    candidate_tables = []

    # Analyze each timeframe independently so missing data does not stop the full report.
    for timeframe in parsed_timeframes:
        summary, candidates_df = analyze_timeframe_channels(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
            step_size=step_size,
            top_n_per_timeframe=top_n_per_timeframe,
            active_only=active_only,
            recent_window=recent_window,
            atr_window=atr_window,
            end_time=end_time,
        )

        summary_rows.append(summary)

        if not candidates_df.empty:
            candidate_tables.append(candidates_df)

    summary_df = pd.DataFrame(summary_rows)

    if candidate_tables:
        candidates_df = pd.concat(candidate_tables, ignore_index=True)
    else:
        candidates_df = pd.DataFrame()

    return summary_df, candidates_df


def summarize_alignment(summary_df: pd.DataFrame) -> dict[str, Any]:
    """
    Builds a compact macro-to-micro alignment summary from timeframe channel labels.
    """
    if summary_df.empty:
        return {
            "valid_timeframes": 0,
            "macro_biases": [],
            "structure_biases": [],
            "entry_biases": [],
            "alignment_state": "no_data",
        }

    valid_df = summary_df[summary_df["status"] == "valid_channel"].copy()

    macro_biases = valid_df[valid_df["purpose"] == "macro_context"]["channel_bias"].tolist()
    structure_biases = valid_df[valid_df["purpose"] == "structure_context"]["channel_bias"].tolist()
    entry_biases = valid_df[valid_df["purpose"] == "entry_refinement"]["channel_bias"].tolist()

    bullish_count = sum("bullish" in bias for bias in valid_df["channel_bias"])
    bearish_count = sum("bearish" in bias for bias in valid_df["channel_bias"])

    # This is not a trade signal; it only describes whether timeframe contexts agree.
    if bullish_count >= 2 and bearish_count == 0:
        alignment_state = "bullish_context_alignment"
    elif bearish_count >= 2 and bullish_count == 0:
        alignment_state = "bearish_context_alignment"
    elif bullish_count > 0 and bearish_count > 0:
        alignment_state = "mixed_context"
    elif len(valid_df) > 0:
        alignment_state = "valid_but_neutral_context"
    else:
        alignment_state = "no_valid_channels"

    return {
        "valid_timeframes": int(len(valid_df)),
        "macro_biases": macro_biases,
        "structure_biases": structure_biases,
        "entry_biases": entry_biases,
        "alignment_state": alignment_state,
    }
