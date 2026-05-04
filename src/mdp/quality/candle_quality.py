import pandas as pd

from mdp.timeframes.registry import (
    build_time_window,
    get_timeframe_delta,
    is_calendar_timeframe,
    validate_timeframe,
)


def get_expected_time_delta(timeframe: str):
    """
    Backward-compatible helper that returns the expected timeframe duration.

    Fixed timeframes return pd.Timedelta.
    Calendar timeframes such as MN1 return pd.DateOffset.
    """
    return get_timeframe_delta(timeframe)


def count_time_gaps(
    timestamps: pd.Series,
    timeframe: str,
) -> tuple[int, str | None]:
    """
    Counts gaps between consecutive candles for both fixed and calendar timeframes.
    """
    normalized_timeframe = validate_timeframe(timeframe)

    sorted_timestamps = (
        pd.to_datetime(timestamps)
        .sort_values()
        .reset_index(drop=True)
    )

    if len(sorted_timestamps) <= 1:
        return 0, None

    gap_count = 0
    largest_gap = None

    for index in range(len(sorted_timestamps) - 1):
        current_time = sorted_timestamps.iloc[index]
        next_time = sorted_timestamps.iloc[index + 1]

        _, expected_next_time = build_time_window(
            start_time=current_time,
            timeframe=normalized_timeframe,
        )

        if next_time != expected_next_time:
            gap_count += 1

            actual_gap = next_time - current_time

            if largest_gap is None or actual_gap > largest_gap:
                largest_gap = actual_gap

    largest_gap_text = None if largest_gap is None else str(largest_gap)

    return gap_count, largest_gap_text

def validate_candle_quality(
    candles_df: pd.DataFrame,
    symbol: str,
    timeframe: str,
) -> dict:
    """
    Runs basic data quality checks on market candles.
    """
    if candles_df.empty:
        return {
            "symbol": symbol,
            "timeframe": timeframe.upper(),
            "row_count": 0,
            "is_empty": True,
            "is_time_sorted": False,
            "duplicate_timestamps": 0,
            "missing_values": {},
            "invalid_ohlc_rows": 0,
            "gap_count": 0,
            "largest_gap": None,
        }

    required_columns = [
        "symbol",
        "timeframe",
        "timestamp_utc",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
    ]

    missing_columns = [
        column for column in required_columns if column not in candles_df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns for quality check: {missing_columns}")

    quality_df = candles_df[required_columns].copy()

    quality_df["timestamp_utc"] = pd.to_datetime(quality_df["timestamp_utc"])

    price_columns = [
        "open_price",
        "high_price",
        "low_price",
        "close_price",
    ]

    for column in price_columns:
        quality_df[column] = quality_df[column].astype(float)

    is_time_sorted = quality_df["timestamp_utc"].is_monotonic_increasing

    duplicate_timestamps = quality_df.duplicated(
        subset=["symbol", "timeframe", "timestamp_utc"]
    ).sum()

    missing_values = quality_df.isna().sum().to_dict()

    invalid_ohlc_rows = (
        (quality_df["high_price"] < quality_df["low_price"])
        | (quality_df["open_price"] > quality_df["high_price"])
        | (quality_df["open_price"] < quality_df["low_price"])
        | (quality_df["close_price"] > quality_df["high_price"])
        | (quality_df["close_price"] < quality_df["low_price"])
    ).sum()

    gap_count, largest_gap = count_time_gaps(
        timestamps=quality_df["timestamp_utc"],
        timeframe=timeframe,
    )

    return {
        "symbol": symbol,
        "timeframe": timeframe.upper(),
        "row_count": len(quality_df),
        "is_empty": False,
        "is_time_sorted": bool(is_time_sorted),
        "duplicate_timestamps": int(duplicate_timestamps),
        "missing_values": missing_values,
        "invalid_ohlc_rows": int(invalid_ohlc_rows),
        "gap_count": int(gap_count),
        "largest_gap": largest_gap,
    }