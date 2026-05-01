import pandas as pd


TIMEFRAME_TO_TIMEDELTA = {
    "M1": pd.Timedelta(minutes=1),
    "M2": pd.Timedelta(minutes=2),
    "M3": pd.Timedelta(minutes=3),
    "M4": pd.Timedelta(minutes=4),
    "M5": pd.Timedelta(minutes=5),
    "M6": pd.Timedelta(minutes=6),
    "M10": pd.Timedelta(minutes=10),
    "M12": pd.Timedelta(minutes=12),
    "M15": pd.Timedelta(minutes=15),
    "M20": pd.Timedelta(minutes=20),
    "M30": pd.Timedelta(minutes=30),
    "H1": pd.Timedelta(hours=1),
    "H2": pd.Timedelta(hours=2),
    "H3": pd.Timedelta(hours=3),
    "H4": pd.Timedelta(hours=4),
    "H6": pd.Timedelta(hours=6),
    "H8": pd.Timedelta(hours=8),
    "H12": pd.Timedelta(hours=12),
    "D1": pd.Timedelta(days=1),
    "W1": pd.Timedelta(weeks=1),
}


def get_expected_time_delta(timeframe: str) -> pd.Timedelta:
    """
    Returns the expected time difference between consecutive candles.
    """
    normalized_timeframe = timeframe.upper()

    if normalized_timeframe not in TIMEFRAME_TO_TIMEDELTA:
        supported_timeframes = ", ".join(TIMEFRAME_TO_TIMEDELTA.keys())
        raise ValueError(
            f"Unsupported timeframe for quality checks: {timeframe}. "
            f"Supported timeframes are: {supported_timeframes}"
        )

    return TIMEFRAME_TO_TIMEDELTA[normalized_timeframe]


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

    expected_delta = get_expected_time_delta(timeframe)

    time_differences = quality_df["timestamp_utc"].diff().dropna()

    gaps = time_differences[time_differences > expected_delta]

    largest_gap = None

    if not gaps.empty:
        largest_gap = str(gaps.max())

    return {
        "symbol": symbol,
        "timeframe": timeframe.upper(),
        "row_count": len(quality_df),
        "is_empty": False,
        "is_time_sorted": bool(is_time_sorted),
        "duplicate_timestamps": int(duplicate_timestamps),
        "missing_values": missing_values,
        "invalid_ohlc_rows": int(invalid_ohlc_rows),
        "gap_count": int(len(gaps)),
        "largest_gap": largest_gap,
    }