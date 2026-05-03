import pandas as pd


REQUIRED_SR_COLUMNS = [
    "timestamp_utc",
    "high_price",
    "low_price",
    "close_price",
]

def calculate_adaptive_zone_half_width(
    candles_df: pd.DataFrame,
    multiplier: float = 0.75,
    min_zone_width_points: float | None = None,
    max_zone_width_points: float | None = None,
) -> float:
    """
    Calculates an adaptive support/resistance zone half-width based on
    the median candle range of the selected market and timeframe.
    """
    if candles_df.empty:
        raise ValueError("Cannot calculate adaptive zone width from an empty DataFrame.")

    required_columns = ["high_price", "low_price"]

    missing_columns = [
        column for column in required_columns if column not in candles_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns for adaptive zone width: {missing_columns}"
        )

    if multiplier <= 0:
        raise ValueError("multiplier must be greater than 0.")

    width_df = candles_df.copy()
    width_df["high_price"] = width_df["high_price"].astype(float)
    width_df["low_price"] = width_df["low_price"].astype(float)

    candle_ranges = width_df["high_price"] - width_df["low_price"]
    median_range = float(candle_ranges.median())

    adaptive_half_width = median_range * multiplier

    if min_zone_width_points is not None:
        adaptive_half_width = max(adaptive_half_width, min_zone_width_points)

    if max_zone_width_points is not None:
        adaptive_half_width = min(adaptive_half_width, max_zone_width_points)

    return adaptive_half_width

def find_pivot_points(
    candles_df: pd.DataFrame,
    left_window: int = 3,
    right_window: int = 3,
) -> pd.DataFrame:
    """
    Finds pivot highs and pivot lows from OHLC candle data.

    A pivot high is a candle high that is higher than nearby highs.
    A pivot low is a candle low that is lower than nearby lows.
    """
    if candles_df.empty:
        return pd.DataFrame(
            columns=["timestamp_utc", "pivot_type", "price", "source_index"]
        )

    missing_columns = [
        column for column in REQUIRED_SR_COLUMNS if column not in candles_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns for support/resistance detection: {missing_columns}"
        )

    if left_window <= 0 or right_window <= 0:
        raise ValueError("left_window and right_window must be greater than 0.")

    pivot_df = candles_df.copy()

    pivot_df["timestamp_utc"] = pd.to_datetime(pivot_df["timestamp_utc"])
    pivot_df["high_price"] = pivot_df["high_price"].astype(float)
    pivot_df["low_price"] = pivot_df["low_price"].astype(float)
    pivot_df["close_price"] = pivot_df["close_price"].astype(float)

    pivot_df = pivot_df.sort_values("timestamp_utc").reset_index(drop=True)

    pivots = []

    start_index = left_window
    end_index = len(pivot_df) - right_window

    for index in range(start_index, end_index):
        current_high = pivot_df.loc[index, "high_price"]
        current_low = pivot_df.loc[index, "low_price"]

        left_highs = pivot_df.loc[index - left_window : index - 1, "high_price"]
        right_highs = pivot_df.loc[index + 1 : index + right_window, "high_price"]

        left_lows = pivot_df.loc[index - left_window : index - 1, "low_price"]
        right_lows = pivot_df.loc[index + 1 : index + right_window, "low_price"]

        is_pivot_high = (
            current_high > left_highs.max()
            and current_high >= right_highs.max()
        )

        is_pivot_low = (
            current_low < left_lows.min()
            and current_low <= right_lows.min()
        )

        if is_pivot_high:
            pivots.append(
                {
                    "timestamp_utc": pivot_df.loc[index, "timestamp_utc"],
                    "pivot_type": "resistance",
                    "price": current_high,
                    "source_index": index,
                }
            )

        if is_pivot_low:
            pivots.append(
                {
                    "timestamp_utc": pivot_df.loc[index, "timestamp_utc"],
                    "pivot_type": "support",
                    "price": current_low,
                    "source_index": index,
                }
            )

    return pd.DataFrame(pivots)


def cluster_price_levels(
    pivots_df: pd.DataFrame,
    tolerance_percent: float = 0.001,
    max_zone_width_points: float | None = None,
    fixed_zone_half_width_points: float | None = None,
) -> pd.DataFrame:
    """
    Groups nearby pivot prices into support/resistance zones.

    Example:
    If several pivot lows happen near 7970, they become one support zone.
    """
    if pivots_df.empty:
        return pd.DataFrame(
            columns=[
                "level_type",
                "level_price",
                "touch_count",
                "first_touch",
                "last_touch",
                "validation_score",
                "tolerance_percent",
            ]
        )

    if tolerance_percent <= 0:
        raise ValueError("tolerance_percent must be greater than 0.")

    clusters = []

    for level_type in ["support", "resistance"]:
        type_pivots = (
            pivots_df[pivots_df["pivot_type"] == level_type]
            .copy()
            .sort_values("price")
            .reset_index(drop=True)
        )

        for _, pivot in type_pivots.iterrows():
            pivot_price = float(pivot["price"])
            pivot_time = pd.to_datetime(pivot["timestamp_utc"])

            matched_cluster = None

            for cluster in clusters:
                if cluster["level_type"] != level_type:
                    continue

                distance_percent = abs(
                    pivot_price - cluster["level_price"]
                ) / cluster["level_price"]

                if distance_percent <= tolerance_percent:
                    matched_cluster = cluster
                    break

            if matched_cluster is None:
                clusters.append(
                    {
                        "level_type": level_type,
                        "level_price": pivot_price,
                        "touch_count": 1,
                        "first_touch": pivot_time,
                        "last_touch": pivot_time,
                        "tolerance_percent": tolerance_percent,
                    }
                )
            else:
                old_count = matched_cluster["touch_count"]
                new_count = old_count + 1

                matched_cluster["level_price"] = (
                    matched_cluster["level_price"] * old_count + pivot_price
                ) / new_count

                matched_cluster["touch_count"] = new_count
                matched_cluster["first_touch"] = min(
                    matched_cluster["first_touch"],
                    pivot_time,
                )
                matched_cluster["last_touch"] = max(
                    matched_cluster["last_touch"],
                    pivot_time,
                )

    levels_df = pd.DataFrame(clusters)

    if levels_df.empty:
        return levels_df

    total_touches = levels_df["touch_count"].sum()

    levels_df["validation_score"] = (
        levels_df["touch_count"] / total_touches
    )

    percent_half_width = (
        levels_df["level_price"] * levels_df["tolerance_percent"]
    )

    if fixed_zone_half_width_points is not None:
        if fixed_zone_half_width_points <= 0:
            raise ValueError("fixed_zone_half_width_points must be greater than 0.")

        half_width = fixed_zone_half_width_points

    elif max_zone_width_points is not None:
        if max_zone_width_points <= 0:
            raise ValueError("max_zone_width_points must be greater than 0.")

        half_width = percent_half_width.clip(upper=max_zone_width_points)

    else:
        half_width = percent_half_width

    levels_df["zone_low"] = levels_df["level_price"] - half_width
    levels_df["zone_high"] = levels_df["level_price"] + half_width
    levels_df["zone_half_width"] = half_width

    levels_df = levels_df.sort_values(
        ["touch_count", "validation_score"],
        ascending=False,
    ).reset_index(drop=True)

    return levels_df


def detect_support_resistance_zones(
    candles_df: pd.DataFrame,
    left_window: int = 3,
    right_window: int = 3,
    tolerance_percent: float = 0.001,
    min_touches: int = 2,
    max_zone_width_points: float | None = None,
    auto_zone_width: bool = False,
    zone_width_multiplier: float = 0.75,
    min_zone_width_points: float | None = None,
) -> pd.DataFrame:
    """
    Detects support and resistance zones from historical candle data.
    """
    pivots_df = find_pivot_points(
        candles_df=candles_df,
        left_window=left_window,
        right_window=right_window,
    )

    fixed_zone_half_width_points = None

    if auto_zone_width:
        fixed_zone_half_width_points = calculate_adaptive_zone_half_width(
            candles_df=candles_df,
            multiplier=zone_width_multiplier,
            min_zone_width_points=min_zone_width_points,
            max_zone_width_points=max_zone_width_points,
        )

    levels_df = cluster_price_levels(
        pivots_df=pivots_df,
        tolerance_percent=tolerance_percent,
        max_zone_width_points=max_zone_width_points,
        fixed_zone_half_width_points=fixed_zone_half_width_points,
    )

    if levels_df.empty:
        return levels_df

    levels_df = levels_df[levels_df["touch_count"] >= min_touches].copy()

    return levels_df.reset_index(drop=True)


def find_nearest_levels(
    levels_df: pd.DataFrame,
    current_price: float,
) -> dict:
    """
    Finds the nearest support below price and resistance above price.
    """
    if levels_df.empty:
        return {
            "nearest_support": None,
            "nearest_resistance": None,
        }

    levels = levels_df.copy()
    levels["level_price"] = levels["level_price"].astype(float)

    supports_below = levels[
        (levels["level_type"] == "support")
        & (levels["level_price"] <= current_price)
    ].copy()

    resistances_above = levels[
        (levels["level_type"] == "resistance")
        & (levels["level_price"] >= current_price)
    ].copy()

    nearest_support = None
    nearest_resistance = None

    if not supports_below.empty:
        supports_below["distance"] = current_price - supports_below["level_price"]
        nearest_support = (
            supports_below.sort_values("distance").iloc[0].to_dict()
        )

    if not resistances_above.empty:
        resistances_above["distance"] = (
            resistances_above["level_price"] - current_price
        )
        nearest_resistance = (
            resistances_above.sort_values("distance").iloc[0].to_dict()
        )

    return {
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
    }