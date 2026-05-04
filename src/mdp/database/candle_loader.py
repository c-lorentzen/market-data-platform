import pandas as pd

from mdp.database.candle_repository import (
    fetch_market_candles,
    fetch_market_candles_between,
)
from mdp.utils.date_ranges import resolve_date_range


def load_market_candles(
    symbol: str,
    timeframe: str,
    limit: int = 300,
    start_time: str | None = None,
    end_time: str | None = None,
    lookback_days: int | None = None,
    lookback_weeks: int | None = None,
    lookback_months: int | None = None,
) -> pd.DataFrame:
    """
    Loads market candles either by recent limit or by flexible date range.
    """
    resolved_start, resolved_end = resolve_date_range(
        start_time=start_time,
        end_time=end_time,
        lookback_days=lookback_days,
        lookback_weeks=lookback_weeks,
        lookback_months=lookback_months,
    )

    if resolved_start is not None and resolved_end is not None:
        return fetch_market_candles_between(
            symbol=symbol,
            timeframe=timeframe,
            start_time=resolved_start,
            end_time=resolved_end,
        )

    return fetch_market_candles(
        symbol=symbol,
        timeframe=timeframe,
        limit=limit,
    )