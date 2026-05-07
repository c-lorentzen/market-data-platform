from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TimeframeScaleProfile:
    """
    Stores default scaling multipliers for one timeframe purpose.
    """

    purpose: str
    recent_window: int
    atr_window: int
    channel_min_width_range_multiplier: float
    channel_max_width_atr_multiplier: float
    zone_width_atr_multiplier: float
    order_block_buffer_atr_multiplier: float
    breakout_threshold_atr_multiplier: float
    retest_tolerance_atr_multiplier: float
    stop_buffer_atr_multiplier: float
    min_target_atr_multiplier: float
    min_r_squared: float
    min_containment_rate: float


@dataclass(frozen=True)
class MarketScaleMetrics:
    """
    Stores volatility-aware and timeframe-aware scale values for one dataset.
    """

    timeframe: str
    purpose: str
    status: str
    candles_analyzed: int
    latest_timestamp: Any
    latest_close: float | None
    median_close: float | None
    median_candle_range: float | None
    recent_median_candle_range: float | None
    atr: float | None
    recent_atr: float | None
    atr_to_price_ratio: float | None
    range_to_price_ratio: float | None
    volatility_stage: str
    volatility_expansion_ratio: float | None
    channel_min_width: float | None
    channel_max_width: float | None
    channel_max_width_ratio: float | None
    support_resistance_zone_width: float | None
    order_block_zone_buffer: float | None
    breakout_threshold: float | None
    retest_tolerance: float | None
    stop_buffer: float | None
    min_target_distance: float | None
    suggested_min_r_squared: float | None
    suggested_min_containment_rate: float | None


def parse_timeframes(timeframes: str | list[str] | tuple[str, ...]) -> list[str]:
    """
    Converts timeframe input into a clean uppercase list.
    """
    # Allow command-line input like "D1,H4,H1,M15,M5,M1".
    if isinstance(timeframes, str):
        raw_values = timeframes.replace(",", " ").split()
    else:
        raw_values = list(timeframes)

    clean_timeframes: list[str] = []

    # Keep order while removing duplicates.
    for timeframe in raw_values:
        normalized = str(timeframe).strip().upper()
        if normalized and normalized not in clean_timeframes:
            clean_timeframes.append(normalized)

    if not clean_timeframes:
        raise ValueError("At least one timeframe is required.")

    return clean_timeframes


def classify_timeframe_purpose(timeframe: str) -> str:
    """
    Labels how one timeframe should be used in the macro-to-micro workflow.
    """
    normalized_timeframe = timeframe.upper()

    # Higher timeframes define direction and broad market context.
    if normalized_timeframe in {"D1", "H4"}:
        return "macro_context"

    # Middle timeframes define active structure and setup conditions.
    if normalized_timeframe in {"H1", "M15"}:
        return "structure_context"

    # Lower timeframes refine entry timing and invalidation.
    return "entry_refinement"


def get_timeframe_scale_profile(timeframe: str) -> TimeframeScaleProfile:
    """
    Returns scale multipliers based on the purpose of each timeframe.
    """
    purpose = classify_timeframe_purpose(timeframe)

    # Macro context should tolerate broader structures and noisier regression fits.
    if purpose == "macro_context":
        return TimeframeScaleProfile(
            purpose=purpose,
            recent_window=80,
            atr_window=14,
            channel_min_width_range_multiplier=3.0,
            channel_max_width_atr_multiplier=14.0,
            zone_width_atr_multiplier=1.25,
            order_block_buffer_atr_multiplier=0.35,
            breakout_threshold_atr_multiplier=0.45,
            retest_tolerance_atr_multiplier=0.55,
            stop_buffer_atr_multiplier=0.50,
            min_target_atr_multiplier=2.50,
            min_r_squared=0.15,
            min_containment_rate=0.65,
        )

    # Structure context should be stricter than macro but not as tight as M1.
    if purpose == "structure_context":
        return TimeframeScaleProfile(
            purpose=purpose,
            recent_window=100,
            atr_window=14,
            channel_min_width_range_multiplier=2.5,
            channel_max_width_atr_multiplier=10.0,
            zone_width_atr_multiplier=1.00,
            order_block_buffer_atr_multiplier=0.30,
            breakout_threshold_atr_multiplier=0.40,
            retest_tolerance_atr_multiplier=0.45,
            stop_buffer_atr_multiplier=0.40,
            min_target_atr_multiplier=2.00,
            min_r_squared=0.30,
            min_containment_rate=0.70,
        )

    # Entry refinement should prefer tighter structures for efficient SL placement.
    return TimeframeScaleProfile(
        purpose=purpose,
        recent_window=120,
        atr_window=14,
        channel_min_width_range_multiplier=2.0,
        channel_max_width_atr_multiplier=7.0,
        zone_width_atr_multiplier=0.80,
        order_block_buffer_atr_multiplier=0.25,
        breakout_threshold_atr_multiplier=0.35,
        retest_tolerance_atr_multiplier=0.35,
        stop_buffer_atr_multiplier=0.35,
        min_target_atr_multiplier=1.50,
        min_r_squared=0.50,
        min_containment_rate=0.75,
    )


def prepare_scale_candles(candles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates and sorts OHLC candles for scale calculations.
    """
    required_columns = [
        "timestamp_utc",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
    ]

    missing_columns = [column for column in required_columns if column not in candles_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required candle columns: {missing_columns}")

    # Work on a copy so the caller's DataFrame is not modified.
    market_df = candles_df.copy()

    # Keep candles in chronological order before calculating true range.
    market_df = market_df.sort_values("timestamp_utc").reset_index(drop=True)

    # Convert price columns to numeric values and drop invalid rows.
    for column in ["open_price", "high_price", "low_price", "close_price"]:
        market_df[column] = pd.to_numeric(market_df[column], errors="coerce")

    market_df = market_df.dropna(subset=required_columns).reset_index(drop=True)

    return market_df


def calculate_true_range(market_df: pd.DataFrame) -> pd.Series:
    """
    Calculates true range using high, low, and previous close.
    """
    # Use high-low range as the base case for the first candle.
    high_low_range = market_df["high_price"] - market_df["low_price"]

    # Compare each candle against the previous close to capture gaps if they exist.
    previous_close = market_df["close_price"].shift(1)
    high_previous_close = (market_df["high_price"] - previous_close).abs()
    low_previous_close = (market_df["low_price"] - previous_close).abs()

    true_range_df = pd.concat(
        [high_low_range, high_previous_close, low_previous_close],
        axis=1,
    )

    # The largest of the three distances is the candle's true range.
    true_range = true_range_df.max(axis=1).fillna(high_low_range)

    return true_range.clip(lower=0.0)


def classify_volatility_stage(volatility_expansion_ratio: float | None) -> str:
    """
    Classifies current volatility relative to the broader dataset.
    """
    if volatility_expansion_ratio is None:
        return "unknown"

    # Compression means recent candles are much smaller than the broader baseline.
    if volatility_expansion_ratio < 0.75:
        return "compressed"

    # Expansion means recent candles are meaningfully larger than the baseline.
    if volatility_expansion_ratio > 1.25:
        return "expanded"

    return "normal"


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    """
    Calculates a safe ratio and returns None when the denominator is invalid.
    """
    if numerator is None or denominator is None or denominator == 0:
        return None

    return float(numerator / denominator)


def calculate_market_scale(
    candles_df: pd.DataFrame,
    timeframe: str,
    recent_window: int | None = None,
    atr_window: int | None = None,
) -> MarketScaleMetrics:
    """
    Calculates flexible market scale metrics for one timeframe.
    """
    normalized_timeframe = timeframe.upper()
    profile = get_timeframe_scale_profile(normalized_timeframe)

    # Allow caller overrides, but keep profile defaults for normal usage.
    scale_recent_window = recent_window or profile.recent_window
    scale_atr_window = atr_window or profile.atr_window

    market_df = prepare_scale_candles(candles_df)

    if market_df.empty:
        return MarketScaleMetrics(
            timeframe=normalized_timeframe,
            purpose=profile.purpose,
            status="no_data",
            candles_analyzed=0,
            latest_timestamp=None,
            latest_close=None,
            median_close=None,
            median_candle_range=None,
            recent_median_candle_range=None,
            atr=None,
            recent_atr=None,
            atr_to_price_ratio=None,
            range_to_price_ratio=None,
            volatility_stage="unknown",
            volatility_expansion_ratio=None,
            channel_min_width=None,
            channel_max_width=None,
            channel_max_width_ratio=None,
            support_resistance_zone_width=None,
            order_block_zone_buffer=None,
            breakout_threshold=None,
            retest_tolerance=None,
            stop_buffer=None,
            min_target_distance=None,
            suggested_min_r_squared=None,
            suggested_min_containment_rate=None,
        )

    latest_row = market_df.iloc[-1]
    latest_close = float(latest_row["close_price"])
    median_close = float(market_df["close_price"].median())

    # Measure candle range using high-low distances.
    candle_ranges = (market_df["high_price"] - market_df["low_price"]).clip(lower=0.0)
    median_candle_range = float(candle_ranges.median())

    # Use a recent subset to adapt to the current market stage.
    recent_df = market_df.tail(min(scale_recent_window, len(market_df))).copy()
    recent_candle_ranges = (recent_df["high_price"] - recent_df["low_price"]).clip(lower=0.0)
    recent_median_candle_range = float(recent_candle_ranges.median())

    true_range = calculate_true_range(market_df)
    atr_series = true_range.rolling(window=scale_atr_window, min_periods=1).mean()

    # Current ATR captures the latest volatility estimate.
    atr = float(atr_series.iloc[-1])

    # Recent ATR smooths the latest market stage over the recent window.
    recent_atr = float(atr_series.tail(min(scale_recent_window, len(atr_series))).median())

    atr_to_price_ratio = safe_ratio(recent_atr, median_close)
    range_to_price_ratio = safe_ratio(recent_median_candle_range, median_close)
    volatility_expansion_ratio = safe_ratio(recent_median_candle_range, median_candle_range)
    volatility_stage = classify_volatility_stage(volatility_expansion_ratio)

    # Use the larger of recent range and ATR to avoid under-scaling during sharp moves.
    current_scale_unit = max(recent_median_candle_range, recent_atr)

    channel_min_width = recent_median_candle_range * profile.channel_min_width_range_multiplier
    channel_max_width = current_scale_unit * profile.channel_max_width_atr_multiplier
    channel_max_width_ratio = safe_ratio(channel_max_width, median_close)

    # Scale zones, breakouts, retests, and buffers from current volatility.
    support_resistance_zone_width = current_scale_unit * profile.zone_width_atr_multiplier
    order_block_zone_buffer = current_scale_unit * profile.order_block_buffer_atr_multiplier
    breakout_threshold = current_scale_unit * profile.breakout_threshold_atr_multiplier
    retest_tolerance = current_scale_unit * profile.retest_tolerance_atr_multiplier
    stop_buffer = current_scale_unit * profile.stop_buffer_atr_multiplier
    min_target_distance = current_scale_unit * profile.min_target_atr_multiplier

    return MarketScaleMetrics(
        timeframe=normalized_timeframe,
        purpose=profile.purpose,
        status="valid_scale",
        candles_analyzed=int(len(market_df)),
        latest_timestamp=latest_row["timestamp_utc"],
        latest_close=latest_close,
        median_close=median_close,
        median_candle_range=median_candle_range,
        recent_median_candle_range=recent_median_candle_range,
        atr=atr,
        recent_atr=recent_atr,
        atr_to_price_ratio=atr_to_price_ratio,
        range_to_price_ratio=range_to_price_ratio,
        volatility_stage=volatility_stage,
        volatility_expansion_ratio=volatility_expansion_ratio,
        channel_min_width=channel_min_width,
        channel_max_width=channel_max_width,
        channel_max_width_ratio=channel_max_width_ratio,
        support_resistance_zone_width=support_resistance_zone_width,
        order_block_zone_buffer=order_block_zone_buffer,
        breakout_threshold=breakout_threshold,
        retest_tolerance=retest_tolerance,
        stop_buffer=stop_buffer,
        min_target_distance=min_target_distance,
        suggested_min_r_squared=profile.min_r_squared,
        suggested_min_containment_rate=profile.min_containment_rate,
    )


def market_scale_to_dict(metrics: MarketScaleMetrics) -> dict[str, Any]:
    """
    Converts market scale metrics into a dictionary for reports and CSV exports.
    """
    return asdict(metrics)


def build_market_scale_table(
    timeframe_to_candles: dict[str, pd.DataFrame],
    recent_window: int | None = None,
    atr_window: int | None = None,
) -> pd.DataFrame:
    """
    Calculates scale metrics for multiple timeframes.
    """
    rows: list[dict[str, Any]] = []

    # Analyze each timeframe independently because each one has a different scale.
    for timeframe, candles_df in timeframe_to_candles.items():
        metrics = calculate_market_scale(
            candles_df=candles_df,
            timeframe=timeframe,
            recent_window=recent_window,
            atr_window=atr_window,
        )
        rows.append(market_scale_to_dict(metrics))

    return pd.DataFrame(rows)
