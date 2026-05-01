import numpy as np
import pandas as pd


REQUIRED_STRUCTURE_COLUMNS = [
    "timestamp_utc",
    "close_price",
    "rolling_high_20",
    "rolling_low_20",
    "rolling_volatility_20",
]


def add_market_structure_features(
    features_df: pd.DataFrame,
    lookback: int = 20,
    trend_threshold: float = 0.0015,
    range_position_threshold: float = 0.20,
) -> pd.DataFrame:
    """
    Adds basic market structure labels to a features DataFrame.

    This first version classifies market condition as:
    - bullish
    - bearish
    - sideways
    - unknown

    The logic is intentionally simple and interpretable.
    """
    if features_df.empty:
        return features_df.copy()

    missing_columns = [
        column for column in REQUIRED_STRUCTURE_COLUMNS
        if column not in features_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns for market structure detection: {missing_columns}"
        )

    structure_df = features_df.copy()

    structure_df["timestamp_utc"] = pd.to_datetime(structure_df["timestamp_utc"])
    structure_df["close_price"] = structure_df["close_price"].astype(float)
    structure_df["rolling_high_20"] = structure_df["rolling_high_20"].astype(float)
    structure_df["rolling_low_20"] = structure_df["rolling_low_20"].astype(float)
    structure_df["rolling_volatility_20"] = structure_df["rolling_volatility_20"].astype(float)

    structure_df = structure_df.sort_values("timestamp_utc").reset_index(drop=True)

    structure_df["close_change_lookback"] = (
        structure_df["close_price"] / structure_df["close_price"].shift(lookback) - 1
    )

    rolling_range = (
        structure_df["rolling_high_20"] - structure_df["rolling_low_20"]
    ).replace(0, np.nan)

    structure_df["close_position_in_rolling_range_20"] = (
        (structure_df["close_price"] - structure_df["rolling_low_20"]) / rolling_range
    )

    bullish_condition = (
        (structure_df["close_change_lookback"] > trend_threshold)
        & (structure_df["close_position_in_rolling_range_20"] > 1 - range_position_threshold)
    )

    bearish_condition = (
        (structure_df["close_change_lookback"] < -trend_threshold)
        & (structure_df["close_position_in_rolling_range_20"] < range_position_threshold)
    )

    sideways_condition = (
        structure_df["close_change_lookback"].abs() <= trend_threshold
    )

    structure_df["market_structure"] = "unknown"

    structure_df.loc[sideways_condition, "market_structure"] = "sideways"
    structure_df.loc[bullish_condition, "market_structure"] = "bullish"
    structure_df.loc[bearish_condition, "market_structure"] = "bearish"

    structure_df["market_structure_score"] = structure_df["close_change_lookback"].abs()

    return structure_df


def summarize_market_structure(structure_df: pd.DataFrame) -> dict:
    """
    Summarizes market structure labels.
    """
    if structure_df.empty or "market_structure" not in structure_df.columns:
        return {
            "row_count": 0,
            "latest_structure": None,
            "structure_counts": {},
            "structure_percentages": {},
        }

    structure_counts = structure_df["market_structure"].value_counts(dropna=False)
    structure_percentages = (
        structure_df["market_structure"].value_counts(normalize=True, dropna=False) * 100
    )

    latest_row = structure_df.iloc[-1]

    return {
        "row_count": len(structure_df),
        "latest_timestamp": latest_row["timestamp_utc"],
        "latest_close": latest_row["close_price"],
        "latest_structure": latest_row["market_structure"],
        "latest_structure_score": latest_row["market_structure_score"],
        "structure_counts": structure_counts.to_dict(),
        "structure_percentages": structure_percentages.round(2).to_dict(),
    }