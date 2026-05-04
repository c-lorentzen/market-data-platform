import pandas as pd


TIMEFRAME_TO_FIXED_TIMEDELTA = {
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


CALENDAR_TIMEFRAMES = {
    "MN1": pd.DateOffset(months=1),
}


SUPPORTED_TIMEFRAMES = list(TIMEFRAME_TO_FIXED_TIMEDELTA.keys()) + list(
    CALENDAR_TIMEFRAMES.keys()
)


def normalize_timeframe(timeframe: str) -> str:
    """
    Normalizes a timeframe string.

    Example:
    h1 -> H1
    m15 -> M15
    mn1 -> MN1
    """
    return timeframe.upper()


def is_supported_timeframe(timeframe: str) -> bool:
    """
    Returns True if the timeframe is supported by the project.
    """
    normalized_timeframe = normalize_timeframe(timeframe)

    return normalized_timeframe in SUPPORTED_TIMEFRAMES


def validate_timeframe(timeframe: str) -> str:
    """
    Validates and returns a normalized timeframe.
    """
    normalized_timeframe = normalize_timeframe(timeframe)

    if normalized_timeframe not in SUPPORTED_TIMEFRAMES:
        supported = ", ".join(SUPPORTED_TIMEFRAMES)
        raise ValueError(
            f"Unsupported timeframe: {timeframe}. "
            f"Supported timeframes are: {supported}"
        )

    return normalized_timeframe


def is_calendar_timeframe(timeframe: str) -> bool:
    """
    Returns True for calendar-based timeframes like MN1.
    """
    normalized_timeframe = validate_timeframe(timeframe)

    return normalized_timeframe in CALENDAR_TIMEFRAMES


def get_timeframe_delta(timeframe: str):
    """
    Returns the expected duration for a timeframe.

    Fixed timeframes return pd.Timedelta.
    Calendar timeframes return pd.DateOffset.
    """
    normalized_timeframe = validate_timeframe(timeframe)

    if normalized_timeframe in TIMEFRAME_TO_FIXED_TIMEDELTA:
        return TIMEFRAME_TO_FIXED_TIMEDELTA[normalized_timeframe]

    return CALENDAR_TIMEFRAMES[normalized_timeframe]


def build_time_window(
    start_time,
    timeframe: str,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Builds a time window for any supported timeframe.

    Example:
    H1 at 20:00 -> 20:00 to 21:00
    MN1 at 2026-02-01 -> 2026-02-01 to 2026-03-01
    """
    normalized_timeframe = validate_timeframe(timeframe)

    start_timestamp = pd.to_datetime(start_time)
    delta = get_timeframe_delta(normalized_timeframe)

    end_timestamp = start_timestamp + delta

    return start_timestamp, end_timestamp


def get_expected_child_candle_count(
    parent_timeframe: str,
    child_timeframe: str,
    parent_start_time=None,
) -> int | None:
    """
    Calculates how many child candles fit inside one parent candle.

    For fixed timeframes:
    H1 / M5 = 12

    For calendar timeframes like MN1, parent_start_time is required because
    months have variable lengths.
    """
    parent_timeframe = validate_timeframe(parent_timeframe)
    child_timeframe = validate_timeframe(child_timeframe)

    parent_delta = get_timeframe_delta(parent_timeframe)
    child_delta = get_timeframe_delta(child_timeframe)

    if isinstance(parent_delta, pd.DateOffset):
        if parent_start_time is None:
            return None

        if isinstance(child_delta, pd.DateOffset):
            return None

        parent_start, parent_end = build_time_window(
            start_time=parent_start_time,
            timeframe=parent_timeframe,
        )

        parent_duration = parent_end - parent_start

        if parent_duration.value % child_delta.value != 0:
            return None

        return parent_duration.value // child_delta.value

    if isinstance(child_delta, pd.DateOffset):
        return None

    if parent_delta.value % child_delta.value != 0:
        return None

    return parent_delta.value // child_delta.value