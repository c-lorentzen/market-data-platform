import pandas as pd


def parse_datetime(value: str | None) -> pd.Timestamp | None:
    """
    Parses a datetime string into a pandas Timestamp.
    """
    if value is None:
        return None

    return pd.to_datetime(value)


def resolve_end_time(end_time: str | None = None) -> pd.Timestamp:
    """
    Resolves the end time.

    If no end time is provided, it uses the current UTC time.
    """
    parsed_end = parse_datetime(end_time)

    if parsed_end is not None:
        return parsed_end

    return pd.Timestamp.utcnow().tz_localize(None).floor("s")


def resolve_start_time_from_calendar_lookback(
    end_timestamp: pd.Timestamp,
    lookback_days: int | None = None,
    lookback_weeks: int | None = None,
    lookback_months: int | None = None,
) -> pd.Timestamp | None:
    """
    Calculates a start time from a calendar-aware lookback.

    Months are calendar months, not fixed 30-day periods.
    """
    provided_lookbacks = [
        lookback_days is not None,
        lookback_weeks is not None,
        lookback_months is not None,
    ]

    if sum(provided_lookbacks) == 0:
        return None

    if sum(provided_lookbacks) > 1:
        raise ValueError(
            "Use only one lookback option at a time: days, weeks, or months."
        )

    if lookback_days is not None:
        if lookback_days <= 0:
            raise ValueError("lookback_days must be greater than 0.")

        return end_timestamp - pd.DateOffset(days=lookback_days)

    if lookback_weeks is not None:
        if lookback_weeks <= 0:
            raise ValueError("lookback_weeks must be greater than 0.")

        return end_timestamp - pd.DateOffset(weeks=lookback_weeks)

    if lookback_months is not None:
        if lookback_months <= 0:
            raise ValueError("lookback_months must be greater than 0.")

        return end_timestamp - pd.DateOffset(months=lookback_months)

    return None


def resolve_date_range(
    start_time: str | None = None,
    end_time: str | None = None,
    lookback_days: int | None = None,
    lookback_weeks: int | None = None,
    lookback_months: int | None = None,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """
    Resolves a flexible date range.

    Supported modes:
    1. Explicit start and end.
    2. Calendar lookback.
    3. No date range, so the caller can fall back to limit.
    """
    parsed_start = parse_datetime(start_time)
    parsed_end = parse_datetime(end_time)

    has_any_lookback = any(
        value is not None
        for value in [lookback_days, lookback_weeks, lookback_months]
    )

    if parsed_start is not None and has_any_lookback:
        raise ValueError("Use either explicit start_time or lookback, not both.")

    if parsed_start is None and parsed_end is not None and not has_any_lookback:
        raise ValueError("end_time requires start_time or a lookback option.")

    if parsed_start is not None:
        if parsed_end is None:
            raise ValueError("start_time requires end_time.")

        if parsed_start >= parsed_end:
            raise ValueError("start_time must be before end_time.")

        return parsed_start, parsed_end

    if has_any_lookback:
        resolved_end = resolve_end_time(end_time)

        resolved_start = resolve_start_time_from_calendar_lookback(
            end_timestamp=resolved_end,
            lookback_days=lookback_days,
            lookback_weeks=lookback_weeks,
            lookback_months=lookback_months,
        )

        if resolved_start is None:
            return None, None

        if resolved_start >= resolved_end:
            raise ValueError("Resolved start time must be before end time.")

        return resolved_start, resolved_end

    return None, None