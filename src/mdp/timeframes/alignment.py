import pandas as pd

from mdp.quality.candle_quality import get_expected_time_delta


def build_parent_time_window(
    parent_start_time,
    parent_timeframe: str,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Builds the start and end time window for a parent candle.

    Example:
    H1 candle at 20:00 covers:
    20:00 <= child candles < 21:00
    """
    start_time = pd.to_datetime(parent_start_time)
    parent_delta = get_expected_time_delta(parent_timeframe)
    end_time = start_time + parent_delta

    return start_time, end_time


def get_expected_child_candle_count(
    parent_timeframe: str,
    child_timeframe: str,
) -> int | None:
    """
    Calculates how many child candles should fit inside one parent candle.

    Example:
    H1 / M5 = 12
    H1 / M1 = 60
    H1 / M15 = 4
    """
    parent_delta = get_expected_time_delta(parent_timeframe)
    child_delta = get_expected_time_delta(child_timeframe)

    if parent_delta.value % child_delta.value != 0:
        return None

    return parent_delta.value // child_delta.value


def validate_child_timeframe_alignment(
    child_candles_df: pd.DataFrame,
    parent_start_time,
    parent_timeframe: str,
    child_timeframe: str,
) -> dict:
    """
    Validates whether child timeframe candles align correctly inside a parent candle window.
    """
    parent_start, parent_end = build_parent_time_window(
        parent_start_time=parent_start_time,
        parent_timeframe=parent_timeframe,
    )

    expected_count = get_expected_child_candle_count(
        parent_timeframe=parent_timeframe,
        child_timeframe=child_timeframe,
    )

    if child_candles_df.empty:
        return {
            "parent_timeframe": parent_timeframe.upper(),
            "child_timeframe": child_timeframe.upper(),
            "parent_start": parent_start,
            "parent_end": parent_end,
            "expected_child_candles": expected_count,
            "actual_child_candles": 0,
            "is_aligned": False,
            "is_time_sorted": False,
            "first_child_timestamp": None,
            "last_child_timestamp": None,
        }

    alignment_df = child_candles_df.copy()
    alignment_df["timestamp_utc"] = pd.to_datetime(alignment_df["timestamp_utc"])
    alignment_df = alignment_df.sort_values("timestamp_utc").reset_index(drop=True)

    actual_count = len(alignment_df)

    is_time_sorted = alignment_df["timestamp_utc"].is_monotonic_increasing

    timestamps_inside_window = (
        (alignment_df["timestamp_utc"] >= parent_start)
        & (alignment_df["timestamp_utc"] < parent_end)
    ).all()

    count_matches = (
        expected_count is not None
        and actual_count == expected_count
    )

    is_aligned = bool(
        is_time_sorted
        and timestamps_inside_window
        and count_matches
    )

    return {
        "parent_timeframe": parent_timeframe.upper(),
        "child_timeframe": child_timeframe.upper(),
        "parent_start": parent_start,
        "parent_end": parent_end,
        "expected_child_candles": expected_count,
        "actual_child_candles": actual_count,
        "is_aligned": is_aligned,
        "is_time_sorted": bool(is_time_sorted),
        "first_child_timestamp": alignment_df["timestamp_utc"].min(),
        "last_child_timestamp": alignment_df["timestamp_utc"].max(),
    }

def aggregate_child_candles(child_candles_df: pd.DataFrame) -> dict:
    """
    Aggregates lower-timeframe candles into one higher-timeframe OHLC summary.

    Example:
    12 M5 candles can be aggregated into one H1 candle.
    """
    if child_candles_df.empty:
        return {
            "open_price": None,
            "high_price": None,
            "low_price": None,
            "close_price": None,
            "tick_volume": None,
            "child_count": 0,
        }

    aggregation_df = child_candles_df.copy()

    aggregation_df["timestamp_utc"] = pd.to_datetime(
        aggregation_df["timestamp_utc"]
    )

    price_columns = [
        "open_price",
        "high_price",
        "low_price",
        "close_price",
    ]

    for column in price_columns:
        aggregation_df[column] = aggregation_df[column].astype(float)

    if "tick_volume" in aggregation_df.columns:
        aggregation_df["tick_volume"] = aggregation_df["tick_volume"].astype(int)

    aggregation_df = aggregation_df.sort_values("timestamp_utc").reset_index(drop=True)

    aggregated = {
        "open_price": aggregation_df.iloc[0]["open_price"],
        "high_price": aggregation_df["high_price"].max(),
        "low_price": aggregation_df["low_price"].min(),
        "close_price": aggregation_df.iloc[-1]["close_price"],
        "tick_volume": (
            int(aggregation_df["tick_volume"].sum())
            if "tick_volume" in aggregation_df.columns
            else None
        ),
        "child_count": len(aggregation_df),
    }

    return aggregated


def compare_parent_to_child_aggregation(
    parent_candle: dict,
    child_candles_df: pd.DataFrame,
    tolerance: float = 0.000001,
) -> dict:
    """
    Compares a parent candle OHLC with the OHLC reconstructed from child candles.
    """
    aggregated_child = aggregate_child_candles(child_candles_df)

    if aggregated_child["child_count"] == 0:
        return {
            "child_count": 0,
            "is_ohlc_match": False,
            "parent": parent_candle,
            "aggregated_child": aggregated_child,
            "differences": {},
        }

    parent_values = {
        "open_price": float(parent_candle["open_price"]),
        "high_price": float(parent_candle["high_price"]),
        "low_price": float(parent_candle["low_price"]),
        "close_price": float(parent_candle["close_price"]),
    }

    differences = {}

    for key, parent_value in parent_values.items():
        child_value = float(aggregated_child[key])
        differences[key] = abs(parent_value - child_value)

    is_ohlc_match = all(
        difference <= tolerance for difference in differences.values()
    )

    return {
        "child_count": aggregated_child["child_count"],
        "is_ohlc_match": bool(is_ohlc_match),
        "parent": parent_values,
        "aggregated_child": aggregated_child,
        "differences": differences,
    }