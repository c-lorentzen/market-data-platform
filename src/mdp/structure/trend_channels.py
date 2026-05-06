import numpy as np
import pandas as pd


def prepare_channel_candles(candles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares candle data for trend channel detection.
    """
    # Stop early if the input data is empty.
    if candles_df.empty:
        return pd.DataFrame()

    # Copy the DataFrame so the original data is not modified.
    market_df = candles_df.copy()

    # Convert timestamps into datetime values.
    market_df["timestamp_utc"] = pd.to_datetime(market_df["timestamp_utc"])

    # Sort candles in chronological order.
    market_df = market_df.sort_values("timestamp_utc").reset_index(drop=True)

    # Create a numeric candle index for regression.
    market_df["candle_index"] = market_df.index.astype(float)

    # Convert OHLC columns into float values for mathematical calculations.
    for column in ["open_price", "high_price", "low_price", "close_price"]:
        market_df[column] = market_df[column].astype(float)

    return market_df


def detect_swing_pivots(
    candles_df: pd.DataFrame,
    left_window: int = 3,
    right_window: int = 3,
) -> pd.DataFrame:
    """
    Detects swing highs and swing lows using fixed left/right windows.
    """
    # Prepare candles with numeric index and clean types.
    market_df = prepare_channel_candles(candles_df)

    # Return an empty pivot table if there is not enough data.
    if market_df.empty:
        return pd.DataFrame()

    # Validate window sizes.
    if left_window <= 0 or right_window <= 0:
        raise ValueError("left_window and right_window must be greater than 0.")

    pivot_rows = []

    # Loop only through candles that have enough candles on both sides.
    for index in range(left_window, len(market_df) - right_window):
        # Select the local window around the current candle.
        window_df = market_df.iloc[index - left_window:index + right_window + 1]

        # Read the current candle.
        current_candle = market_df.iloc[index]

        # A pivot high is the highest high in the local window.
        is_pivot_high = (
            current_candle["high_price"] == window_df["high_price"].max()
        )

        # A pivot low is the lowest low in the local window.
        is_pivot_low = (
            current_candle["low_price"] == window_df["low_price"].min()
        )

        # Store pivot high if detected.
        if is_pivot_high:
            pivot_rows.append(
                {
                    "timestamp_utc": current_candle["timestamp_utc"],
                    "candle_index": current_candle["candle_index"],
                    "pivot_type": "high",
                    "pivot_price": current_candle["high_price"],
                }
            )

        # Store pivot low if detected.
        if is_pivot_low:
            pivot_rows.append(
                {
                    "timestamp_utc": current_candle["timestamp_utc"],
                    "candle_index": current_candle["candle_index"],
                    "pivot_type": "low",
                    "pivot_price": current_candle["low_price"],
                }
            )

    return pd.DataFrame(pivot_rows)


def fit_regression_line(points_df: pd.DataFrame) -> dict:
    """
    Fits a linear regression line to pivot points.
    """
    # A valid regression line needs at least two points.
    if len(points_df) < 2:
        raise ValueError("At least two pivot points are required to fit a line.")

    # Read x and y values for regression.
    x_values = points_df["candle_index"].astype(float).to_numpy()
    y_values = points_df["pivot_price"].astype(float).to_numpy()

    # Fit y = slope * x + intercept.
    slope, intercept = np.polyfit(x_values, y_values, deg=1)

    # Calculate predicted y values.
    predicted_y = slope * x_values + intercept

    # Calculate R-squared as a line-fit quality metric.
    residual_sum_squares = np.sum((y_values - predicted_y) ** 2)
    total_sum_squares = np.sum((y_values - y_values.mean()) ** 2)

    if total_sum_squares == 0:
        r_squared = 0.0
    else:
        r_squared = 1 - (residual_sum_squares / total_sum_squares)

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
    }


def calculate_line_values(
    candle_indexes: pd.Series,
    slope: float,
    intercept: float,
) -> pd.Series:
    """
    Calculates line values for each candle index.
    """
    # Apply the regression formula y = slope*x + intercept.
    return slope * candle_indexes.astype(float) + intercept


def calculate_channel_tolerance(
    candles_df: pd.DataFrame,
    tolerance_range_multiplier: float = 0.50,
) -> float:
    """
    Calculates price tolerance based on the median candle range.
    """
    # Calculate candle range for each row.
    candle_ranges = candles_df["high_price"] - candles_df["low_price"]

    # Use median range so one extreme candle does not dominate tolerance.
    median_range = float(candle_ranges.median())

    # Return a tolerance based on market volatility.
    return median_range * tolerance_range_multiplier


def calculate_channel_metrics(
    market_df: pd.DataFrame,
    lower_line: pd.Series,
    upper_line: pd.Series,
    tolerance: float,
) -> dict:
    """
    Calculates measurable channel quality metrics.
    """
    # Count closes inside the channel boundaries.
    closes_inside = (
        (market_df["close_price"] >= lower_line - tolerance)
        & (market_df["close_price"] <= upper_line + tolerance)
    )

    # Count lower boundary touches.
    lower_touches = (
        abs(market_df["low_price"] - lower_line) <= tolerance
    )

    # Count upper boundary touches.
    upper_touches = (
        abs(market_df["high_price"] - upper_line) <= tolerance
    )

    # Read the latest close and latest channel boundaries.
    current_close = float(market_df.iloc[-1]["close_price"])
    current_lower = float(lower_line.iloc[-1])
    current_upper = float(upper_line.iloc[-1])

    # Calculate the current relative position inside the channel.
    channel_width = current_upper - current_lower

    if channel_width <= 0:
        current_position_ratio = None
    else:
        current_position_ratio = (current_close - current_lower) / channel_width

    # Classify the latest close relative to the channel.
    if current_close > current_upper + tolerance:
        current_state = "above_channel"
    elif current_close < current_lower - tolerance:
        current_state = "below_channel"
    elif abs(current_close - current_upper) <= tolerance:
        current_state = "near_upper_boundary"
    elif abs(current_close - current_lower) <= tolerance:
        current_state = "near_lower_boundary"
    else:
        current_state = "inside_channel"

    return {
        "containment_rate": float(closes_inside.mean()),
        "lower_touch_count": int(lower_touches.sum()),
        "upper_touch_count": int(upper_touches.sum()),
        "total_touch_count": int(lower_touches.sum() + upper_touches.sum()),
        "current_close": current_close,
        "current_lower": current_lower,
        "current_upper": current_upper,
        "current_position_ratio": current_position_ratio,
        "current_state": current_state,
    }


def calculate_channel_quality_score(
    containment_rate: float,
    total_touch_count: int,
    r_squared: float,
    normalized_slope_strength: float,
) -> float:
    """
    Calculates a deterministic channel quality score from 0 to 1.
    """
    # Convert touch count into a capped score.
    touch_score = min(total_touch_count / 6, 1.0)

    # Convert slope strength into a capped score.
    slope_score = min(abs(normalized_slope_strength), 1.0)

    # Combine metrics using fixed weights.
    quality_score = (
        0.35 * containment_rate
        + 0.25 * touch_score
        + 0.20 * max(r_squared, 0)
        + 0.20 * slope_score
    )

    return float(quality_score)


def build_channel_candidate(
    market_df: pd.DataFrame,
    pivots_df: pd.DataFrame,
    channel_type: str,
    envelope_quantile: float = 0.90,
    tolerance_range_multiplier: float = 0.50,
    min_anchor_pivots: int = 3,
) -> dict | None:
    """
    Builds one ascending or descending channel candidate.
    """
    # Validate channel type.
    if channel_type not in {"ascending", "descending"}:
        raise ValueError("channel_type must be 'ascending' or 'descending'.")

    # Separate pivot highs and pivot lows.
    pivot_highs = pivots_df[pivots_df["pivot_type"] == "high"].copy()
    pivot_lows = pivots_df[pivots_df["pivot_type"] == "low"].copy()

    # Choose anchors based on channel direction.
    if channel_type == "ascending":
        anchor_points = pivot_lows
        opposite_points = pivot_highs
        required_slope_sign = 1
    else:
        anchor_points = pivot_highs
        opposite_points = pivot_lows
        required_slope_sign = -1

    # Stop if there are not enough anchor pivots.
    if len(anchor_points) < min_anchor_pivots or len(opposite_points) < 2:
        return None

    # Fit the main regression line using anchor pivots.
    line_model = fit_regression_line(anchor_points)

    slope = line_model["slope"]
    intercept = line_model["intercept"]
    r_squared = line_model["r_squared"]

    # Enforce slope direction for the channel type.
    if required_slope_sign == 1 and slope <= 0:
        return None

    if required_slope_sign == -1 and slope >= 0:
        return None

    # Calculate base line values for all candles.
    base_line = calculate_line_values(
        candle_indexes=market_df["candle_index"],
        slope=slope,
        intercept=intercept,
    )

    # Calculate base line values at opposite pivots.
    opposite_base_values = calculate_line_values(
        candle_indexes=opposite_points["candle_index"],
        slope=slope,
        intercept=intercept,
    )

    if channel_type == "ascending":
        # Measure distances from pivot highs to the lower support line.
        distances = opposite_points["pivot_price"].to_numpy() - opposite_base_values.to_numpy()

        # Keep only positive distances.
        distances = distances[distances > 0]

        if len(distances) == 0:
            return None

        # Use quantile width to avoid overreacting to one extreme wick.
        channel_width = float(np.quantile(distances, envelope_quantile))

        lower_line = base_line
        upper_line = base_line + channel_width

    else:
        # Measure distances from upper resistance line to pivot lows.
        distances = opposite_base_values.to_numpy() - opposite_points["pivot_price"].to_numpy()

        # Keep only positive distances.
        distances = distances[distances > 0]

        if len(distances) == 0:
            return None

        # Use quantile width to avoid overreacting to one extreme wick.
        channel_width = float(np.quantile(distances, envelope_quantile))

        upper_line = base_line
        lower_line = base_line - channel_width

    # Stop if the channel width is invalid.
    if channel_width <= 0:
        return None

    # Calculate dynamic tolerance from median candle range.
    tolerance = calculate_channel_tolerance(
        candles_df=market_df,
        tolerance_range_multiplier=tolerance_range_multiplier,
    )

    # Calculate channel metrics.
    metrics = calculate_channel_metrics(
        market_df=market_df,
        lower_line=lower_line,
        upper_line=upper_line,
        tolerance=tolerance,
    )

    # Normalize slope by median close price so different markets are comparable.
    median_close = float(market_df["close_price"].median())

    if median_close == 0:
        normalized_slope_strength = 0.0
    else:
        normalized_slope_strength = slope / median_close * 1000

    # Calculate deterministic quality score.
    quality_score = calculate_channel_quality_score(
        containment_rate=metrics["containment_rate"],
        total_touch_count=metrics["total_touch_count"],
        r_squared=r_squared,
        normalized_slope_strength=normalized_slope_strength,
    )

    return {
        "channel_type": channel_type,
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "channel_width": channel_width,
        "tolerance": tolerance,
        "normalized_slope_strength": normalized_slope_strength,
        "quality_score": quality_score,
        **metrics,
    }


def detect_trend_channels(
    candles_df: pd.DataFrame,
    left_window: int = 3,
    right_window: int = 3,
    envelope_quantile: float = 0.90,
    tolerance_range_multiplier: float = 0.50,
    min_anchor_pivots: int = 3,
) -> pd.DataFrame:
    """
    Detects ascending and descending trend channel candidates.
    """
    # Prepare candles for channel detection.
    market_df = prepare_channel_candles(candles_df)

    # Return empty output if there is not enough data.
    if market_df.empty:
        return pd.DataFrame()

    # Detect swing pivots from OHLC data.
    pivots_df = detect_swing_pivots(
        candles_df=market_df,
        left_window=left_window,
        right_window=right_window,
    )

    # Return empty output if no pivots exist.
    if pivots_df.empty:
        return pd.DataFrame()

    candidates = []

    # Try to build an ascending channel candidate.
    ascending_channel = build_channel_candidate(
        market_df=market_df,
        pivots_df=pivots_df,
        channel_type="ascending",
        envelope_quantile=envelope_quantile,
        tolerance_range_multiplier=tolerance_range_multiplier,
        min_anchor_pivots=min_anchor_pivots,
    )

    if ascending_channel is not None:
        candidates.append(ascending_channel)

    # Try to build a descending channel candidate.
    descending_channel = build_channel_candidate(
        market_df=market_df,
        pivots_df=pivots_df,
        channel_type="descending",
        envelope_quantile=envelope_quantile,
        tolerance_range_multiplier=tolerance_range_multiplier,
        min_anchor_pivots=min_anchor_pivots,
    )

    if descending_channel is not None:
        candidates.append(descending_channel)

    # Return empty output if no candidate was valid.
    if not candidates:
        return pd.DataFrame()

    channels_df = pd.DataFrame(candidates)

    # Add channel IDs for easier reading.
    channels_df = channels_df.sort_values(
        "quality_score",
        ascending=False,
    ).reset_index(drop=True)

    channels_df["channel_id"] = channels_df.index + 1

    # Move channel_id to the first column.
    ordered_columns = ["channel_id"] + [
        column for column in channels_df.columns
        if column != "channel_id"
    ]

    return channels_df[ordered_columns]