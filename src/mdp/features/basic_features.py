import numpy as np
import pandas as pd


REQUIRED_CANDLE_COLUMNS = [
    "timestamp_utc",
    "open_price",
    "high_price",
    "low_price",
    "close_price",
]


def add_basic_candle_features(candles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds basic reusable candle features to a market candles DataFrame.

    These features are strategy-neutral and can be reused later for:
    - market structure detection
    - baseline strategies
    - backtesting
    - machine learning models
    - visualization
    """
    if candles_df.empty:
        return candles_df.copy()

    missing_columns = [
        column for column in REQUIRED_CANDLE_COLUMNS if column not in candles_df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required candle columns: {missing_columns}")

    features_df = candles_df.copy()

    features_df["timestamp_utc"] = pd.to_datetime(features_df["timestamp_utc"])

    price_columns = [
        "open_price",
        "high_price",
        "low_price",
        "close_price",
    ]

    for column in price_columns:
        features_df[column] = features_df[column].astype(float)

    features_df = features_df.sort_values("timestamp_utc").reset_index(drop=True)

    features_df["close_return"] = features_df["close_price"].pct_change()

    features_df["log_return"] = np.log(
        features_df["close_price"] / features_df["close_price"].shift(1)
    )

    features_df["candle_range"] = (
        features_df["high_price"] - features_df["low_price"]
    )

    features_df["candle_body"] = (
        features_df["close_price"] - features_df["open_price"]
    )

    features_df["candle_body_abs"] = features_df["candle_body"].abs()

    features_df["upper_wick"] = (
        features_df["high_price"]
        - features_df[["open_price", "close_price"]].max(axis=1)
    )

    features_df["lower_wick"] = (
        features_df[["open_price", "close_price"]].min(axis=1)
        - features_df["low_price"]
    )

    safe_range = features_df["candle_range"].replace(0, np.nan)

    features_df["body_to_range_ratio"] = (
        features_df["candle_body_abs"] / safe_range
    )

    features_df["close_position_in_range"] = (
        (features_df["close_price"] - features_df["low_price"]) / safe_range
    )

    features_df["is_bullish_candle"] = (
        features_df["close_price"] > features_df["open_price"]
    )

    features_df["is_bearish_candle"] = (
        features_df["close_price"] < features_df["open_price"]
    )

    features_df["rolling_return_mean_5"] = (
        features_df["close_return"].rolling(window=5, min_periods=5).mean()
    )

    features_df["rolling_volatility_5"] = (
        features_df["close_return"].rolling(window=5, min_periods=5).std()
    )

    features_df["rolling_volatility_20"] = (
        features_df["close_return"].rolling(window=20, min_periods=20).std()
    )

    features_df["rolling_high_20"] = (
        features_df["high_price"].rolling(window=20, min_periods=20).max()
    )

    features_df["rolling_low_20"] = (
        features_df["low_price"].rolling(window=20, min_periods=20).min()
    )

    features_df["distance_to_rolling_high_20"] = (
        features_df["close_price"] - features_df["rolling_high_20"]
    )

    features_df["distance_to_rolling_low_20"] = (
        features_df["close_price"] - features_df["rolling_low_20"]
    )

    return features_df