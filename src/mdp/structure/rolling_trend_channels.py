from __future__ import annotations

import pandas as pd

from mdp.structure.trend_channels import detect_trend_channels, prepare_channel_candles


def parse_window_sizes(window_sizes: str | list[int] | tuple[int, ...] | None) -> list[int]:
    """
    Converts window-size input into a clean list of positive integers.
    """
    # Use a practical default set for short-term and medium-term structures.
    if window_sizes is None:
        return [80, 120, 180, 240]

    # Allow command-line input like "80,120,180,240".
    if isinstance(window_sizes, str):
        parsed_sizes = [int(value.strip()) for value in window_sizes.split(",") if value.strip()]
    else:
        parsed_sizes = [int(value) for value in window_sizes]

    # Keep only valid positive sizes and remove duplicates.
    clean_sizes = sorted({value for value in parsed_sizes if value > 0})

    if not clean_sizes:
        raise ValueError("At least one positive window size is required.")

    return clean_sizes


def build_rolling_window_ranges(
    total_candles: int,
    window_size: int,
    step_size: int,
    include_latest_window: bool = True,
    active_only: bool = False,
) -> list[tuple[int, int]]:
    """
    Builds rolling window ranges as start/end row positions.
    """
    # Stop early when the requested window is larger than the dataset.
    if total_candles < window_size:
        return []

    # Validate step size so the loop can move forward safely.
    if step_size <= 0:
        raise ValueError("step_size must be greater than 0.")

    latest_start_index = total_candles - window_size
    latest_range = (latest_start_index, total_candles)

    # For entry work, only the latest window should define the active channel.
    if active_only:
        return [latest_range]

    window_ranges = []

    # Build fixed-size windows across the dataset.
    for start_index in range(0, total_candles - window_size + 1, step_size):
        end_index = start_index + window_size
        window_ranges.append((start_index, end_index))

    if include_latest_window and latest_range not in window_ranges:
        # Always include the most recent window so the active structure is checked.
        window_ranges.append(latest_range)

    return window_ranges


def calculate_channel_width_ratio(channel_width: float, median_close: float) -> float:
    """
    Calculates channel width as a fraction of median close price.
    """
    # Avoid division by zero if bad price data is provided.
    if median_close == 0:
        return 0.0

    return float(channel_width / median_close)


def calculate_median_candle_range(window_df: pd.DataFrame) -> float:
    """
    Calculates the median high-low candle range inside one window.
    """
    # Use high-low range as a simple volatility proxy for the current segment.
    candle_ranges = window_df["high_price"] - window_df["low_price"]

    # Median is more stable than average when a few candles are very large.
    median_range = float(candle_ranges.median())

    return max(median_range, 0.0)


def calculate_width_to_range_ratio(channel_width: float, median_candle_range: float) -> float:
    """
    Compares channel width against the typical candle range in the window.
    """
    # If range is zero, the ratio cannot be trusted.
    if median_candle_range <= 0:
        return 0.0

    return float(channel_width / median_candle_range)


def calculate_width_score(
    channel_width_ratio: float,
    max_channel_width_ratio: float,
) -> float:
    """
    Scores narrower channels higher while keeping the result between 0 and 1.
    """
    # A channel with invalid width should receive no width-quality credit.
    if channel_width_ratio <= 0:
        return 0.0

    # Avoid invalid max-width settings.
    if max_channel_width_ratio <= 0:
        raise ValueError("max_channel_width_ratio must be greater than 0.")

    # Wider channels lose score because they can contain too much random price action.
    raw_score = 1 - (channel_width_ratio / max_channel_width_ratio)

    return float(max(0.0, min(raw_score, 1.0)))


def calculate_entry_efficiency_score(
    width_to_range_ratio: float,
    min_width_to_range_ratio: float,
    max_width_to_range_ratio: float,
) -> float:
    """
    Scores channels that are narrow enough for entries but not unrealistically tight.
    """
    # Invalid ratios should not help the channel rank.
    if width_to_range_ratio <= 0:
        return 0.0

    if min_width_to_range_ratio <= 0 or max_width_to_range_ratio <= 0:
        raise ValueError("Width-to-range thresholds must be greater than 0.")

    if min_width_to_range_ratio >= max_width_to_range_ratio:
        raise ValueError("min_width_to_range_ratio must be smaller than max_width_to_range_ratio.")

    # Penalize channels that are too tight because they may only describe random noise.
    if width_to_range_ratio < min_width_to_range_ratio:
        return float(width_to_range_ratio / min_width_to_range_ratio)

    # Penalize channels that are too wide because they create inefficient stop placement.
    if width_to_range_ratio > max_width_to_range_ratio:
        return float(max_width_to_range_ratio / width_to_range_ratio)

    # Inside the efficient range, the channel is practical for entry refinement.
    return 1.0


def calculate_recency_score(
    window_end_index: int,
    latest_index: int,
    recency_decay_candles: int = 60,
) -> float:
    """
    Scores windows higher when they end closer to the latest candle.
    """
    # The latest completed window receives the full recency score.
    distance_from_latest = latest_index - window_end_index

    if distance_from_latest <= 0:
        return 1.0

    if recency_decay_candles <= 0:
        raise ValueError("recency_decay_candles must be greater than 0.")

    # Older windows lose score faster so stale channels do not outrank active channels.
    raw_score = 1 - (distance_from_latest / recency_decay_candles)

    return float(max(0.0, min(raw_score, 1.0)))


def classify_channel_role(is_latest_window: bool, current_state: str) -> str:
    """
    Labels whether the channel is active context or historical context.
    """
    # Latest windows are the only windows that should drive current entry decisions.
    if is_latest_window:
        if current_state in {"above_channel", "below_channel"}:
            return "active_breakout_context"
        return "active_entry_context"

    # Older windows may become useful later for broken-channel support/resistance logic.
    return "historical_context"


def calculate_rolling_channel_score(
    quality_score: float,
    recency_score: float,
    width_score: float,
    r_squared: float,
    entry_efficiency_score: float,
) -> float:
    """
    Combines channel quality, recency, width, fit quality, and entry efficiency.
    """
    # Keep R-squared between 0 and 1 because negative values should not help the score.
    fit_score = max(0.0, min(float(r_squared), 1.0))

    # Use stronger recency weighting so stale channels do not dominate entry analysis.
    rolling_score = (
        0.35 * float(quality_score)
        + 0.25 * float(recency_score)
        + 0.15 * float(width_score)
        + 0.15 * fit_score
        + 0.10 * float(entry_efficiency_score)
    )

    return float(max(0.0, min(rolling_score, 1.0)))


def detect_rolling_trend_channels(
    candles_df: pd.DataFrame,
    window_sizes: str | list[int] | tuple[int, ...] | None = None,
    step_size: int = 20,
    top_n: int | None = 10,
    left_window: int = 3,
    right_window: int = 3,
    envelope_quantile: float = 0.90,
    tolerance_range_multiplier: float = 0.50,
    min_anchor_pivots: int = 3,
    min_quality_score: float = 0.0,
    min_rolling_quality_score: float = 0.0,
    max_channel_width_ratio: float = 0.002,
    min_width_to_range_ratio: float = 1.0,
    max_width_to_range_ratio: float = 8.0,
    recency_decay_candles: int = 60,
    include_latest_window: bool = True,
    active_only: bool = False,
) -> pd.DataFrame:
    """
    Detects trend channels across multiple rolling windows.
    """
    # Prepare the full candle dataset once so all windows use clean chronological data.
    market_df = prepare_channel_candles(candles_df)

    # Return an empty table if there are no candles to analyze.
    if market_df.empty:
        return pd.DataFrame()

    # Convert input window sizes into a validated list.
    parsed_window_sizes = parse_window_sizes(window_sizes)

    rolling_rows = []
    latest_index = len(market_df) - 1

    # Scan each requested window size independently.
    for window_size in parsed_window_sizes:
        window_ranges = build_rolling_window_ranges(
            total_candles=len(market_df),
            window_size=window_size,
            step_size=step_size,
            include_latest_window=include_latest_window,
            active_only=active_only,
        )

        # Analyze each rolling segment for channel candidates.
        for start_index, end_index in window_ranges:
            window_df = market_df.iloc[start_index:end_index].copy()
            is_latest_window = end_index == len(market_df)

            # Detect channels only inside this segment.
            channels_df = detect_trend_channels(
                candles_df=window_df,
                left_window=left_window,
                right_window=right_window,
                envelope_quantile=envelope_quantile,
                tolerance_range_multiplier=tolerance_range_multiplier,
                min_anchor_pivots=min_anchor_pivots,
            )

            # Skip this segment if no valid channel exists.
            if channels_df.empty:
                continue

            median_close = float(window_df["close_price"].median())
            median_candle_range = calculate_median_candle_range(window_df)
            window_end_global_index = end_index - 1

            recency_score = calculate_recency_score(
                window_end_index=window_end_global_index,
                latest_index=latest_index,
                recency_decay_candles=recency_decay_candles,
            )

            # Store each candidate with its segment metadata.
            for _, channel_row in channels_df.iterrows():
                if float(channel_row["quality_score"]) < min_quality_score:
                    continue

                channel_width = float(channel_row["channel_width"])
                channel_width_ratio = calculate_channel_width_ratio(
                    channel_width=channel_width,
                    median_close=median_close,
                )

                width_score = calculate_width_score(
                    channel_width_ratio=channel_width_ratio,
                    max_channel_width_ratio=max_channel_width_ratio,
                )

                width_to_range_ratio = calculate_width_to_range_ratio(
                    channel_width=channel_width,
                    median_candle_range=median_candle_range,
                )

                entry_efficiency_score = calculate_entry_efficiency_score(
                    width_to_range_ratio=width_to_range_ratio,
                    min_width_to_range_ratio=min_width_to_range_ratio,
                    max_width_to_range_ratio=max_width_to_range_ratio,
                )

                rolling_score = calculate_rolling_channel_score(
                    quality_score=float(channel_row["quality_score"]),
                    recency_score=recency_score,
                    width_score=width_score,
                    r_squared=float(channel_row["r_squared"]),
                    entry_efficiency_score=entry_efficiency_score,
                )

                if rolling_score < min_rolling_quality_score:
                    continue

                current_state = str(channel_row["current_state"])
                channel_role = classify_channel_role(
                    is_latest_window=is_latest_window,
                    current_state=current_state,
                )

                rolling_rows.append(
                    {
                        "window_size": int(window_size),
                        "window_start_index": int(start_index),
                        "window_end_index": int(window_end_global_index),
                        "window_start_time": window_df.iloc[0]["timestamp_utc"],
                        "window_end_time": window_df.iloc[-1]["timestamp_utc"],
                        "window_candle_count": int(len(window_df)),
                        "is_latest_window": bool(is_latest_window),
                        "channel_role": channel_role,
                        "recency_score": recency_score,
                        "median_candle_range": median_candle_range,
                        "channel_width_ratio": channel_width_ratio,
                        "width_to_range_ratio": width_to_range_ratio,
                        "width_score": width_score,
                        "entry_efficiency_score": entry_efficiency_score,
                        "rolling_quality_score": rolling_score,
                        **channel_row.to_dict(),
                    }
                )

    # Return an empty table if no rolling candidates were found.
    if not rolling_rows:
        return pd.DataFrame()

    rolling_channels_df = pd.DataFrame(rolling_rows)

    # Rank channels by active/latest status first, then by rolling quality.
    rolling_channels_df = rolling_channels_df.sort_values(
        ["is_latest_window", "rolling_quality_score", "quality_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    # Add a new ID that is unique across all rolling candidates.
    rolling_channels_df["rolling_channel_id"] = rolling_channels_df.index + 1

    # Keep the most important ID and ranking columns at the front.
    ordered_columns = [
        "rolling_channel_id",
        "window_size",
        "window_start_index",
        "window_end_index",
        "window_start_time",
        "window_end_time",
        "is_latest_window",
        "channel_role",
        "rolling_quality_score",
        "recency_score",
        "width_score",
        "entry_efficiency_score",
        "channel_width_ratio",
        "width_to_range_ratio",
        "median_candle_range",
    ]

    remaining_columns = [
        column for column in rolling_channels_df.columns
        if column not in ordered_columns
    ]

    rolling_channels_df = rolling_channels_df[ordered_columns + remaining_columns]

    # Optionally keep only the strongest candidates.
    if top_n is not None and top_n > 0:
        rolling_channels_df = rolling_channels_df.head(top_n).reset_index(drop=True)

    return rolling_channels_df


def summarize_rolling_channels(rolling_channels_df: pd.DataFrame) -> dict:
    """
    Builds a compact summary of rolling channel detection results.
    """
    # Return a safe summary if no channels were detected.
    if rolling_channels_df.empty:
        return {
            "total_channels": 0,
            "active_channels": 0,
            "historical_channels": 0,
            "best_channel_type": None,
            "best_window_size": None,
            "best_current_state": None,
            "best_channel_role": None,
            "best_rolling_quality_score": None,
        }

    # Use the first row because the table is already ranked.
    best_channel = rolling_channels_df.iloc[0]
    active_count = int(rolling_channels_df["is_latest_window"].sum())
    historical_count = int(len(rolling_channels_df) - active_count)

    return {
        "total_channels": int(len(rolling_channels_df)),
        "active_channels": active_count,
        "historical_channels": historical_count,
        "best_channel_type": best_channel["channel_type"],
        "best_window_size": int(best_channel["window_size"]),
        "best_current_state": best_channel["current_state"],
        "best_channel_role": best_channel["channel_role"],
        "best_rolling_quality_score": float(best_channel["rolling_quality_score"]),
    }
