import pandas as pd


REQUIRED_EVALUATED_ZONE_COLUMNS = [
    "level_type",
    "level_price",
    "zone_low",
    "zone_high",
    "tested_touches",
    "respected_touches",
    "broken_touches",
    "respect_rate",
    "break_rate",
]


def classify_price_vs_zone(
    current_price: float,
    zone_low: float,
    zone_high: float,
) -> str:
    """
    Classifies the current price relative to a zone.
    """
    if current_price < zone_low:
        return "below_zone"
    if current_price > zone_high:
        return "above_zone"
    return "inside_zone"


def add_zone_context(
    evaluated_levels_df: pd.DataFrame,
    current_price: float,
) -> pd.DataFrame:
    """
    Adds context columns describing where the current price is relative to each zone.
    """
    if evaluated_levels_df.empty:
        return pd.DataFrame()

    missing_columns = [
        column
        for column in REQUIRED_EVALUATED_ZONE_COLUMNS
        if column not in evaluated_levels_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns for zone context: {missing_columns}"
        )

    context_df = evaluated_levels_df.copy()

    numeric_columns = [
        "level_price",
        "zone_low",
        "zone_high",
        "tested_touches",
        "respected_touches",
        "broken_touches",
        "respect_rate",
        "break_rate",
    ]

    for column in numeric_columns:
        context_df[column] = pd.to_numeric(context_df[column], errors="coerce")

    context_df["current_price"] = float(current_price)

    context_df["price_vs_zone"] = context_df.apply(
        lambda row: classify_price_vs_zone(
            current_price=float(current_price),
            zone_low=float(row["zone_low"]),
            zone_high=float(row["zone_high"]),
        ),
        axis=1,
    )

    context_df["distance_to_level"] = (
        context_df["level_price"] - float(current_price)
    )

    context_df["absolute_distance"] = context_df["distance_to_level"].abs()

    context_df["zone_strength_score"] = (
        context_df["respect_rate"].fillna(0) * 0.6
        + context_df["tested_touches"].fillna(0) * 0.02
        + context_df["validation_score"].fillna(0) * 0.2
    )

    return context_df


def find_nearest_context_zones(
    context_df: pd.DataFrame,
    current_price: float,
) -> dict:
    """
    Finds nearest support and resistance zones relative to current price.
    """
    if context_df.empty:
        return {
            "nearest_support": None,
            "nearest_resistance": None,
            "zones_containing_price": [],
        }

    working_df = context_df.copy()

    supports = working_df[
        (working_df["level_type"] == "support")
        & (working_df["zone_low"] <= current_price)
    ].copy()

    resistances = working_df[
        (working_df["level_type"] == "resistance")
        & (working_df["zone_high"] >= current_price)
    ].copy()

    containing_zones = working_df[
        working_df["price_vs_zone"] == "inside_zone"
    ].copy()

    nearest_support = None
    nearest_resistance = None

    if not supports.empty:
        supports = supports.sort_values(
            ["absolute_distance", "zone_strength_score"],
            ascending=[True, False],
        )
        nearest_support = supports.iloc[0].to_dict()

    if not resistances.empty:
        resistances = resistances.sort_values(
            ["absolute_distance", "zone_strength_score"],
            ascending=[True, False],
        )
        nearest_resistance = resistances.iloc[0].to_dict()

    zones_containing_price = containing_zones.to_dict(orient="records")

    return {
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "zones_containing_price": zones_containing_price,
    }


def build_zone_decision_snapshot(
    context_df: pd.DataFrame,
    current_price: float,
) -> dict:
    """
    Builds a high-level decision snapshot from evaluated support/resistance zones.
    """
    nearest = find_nearest_context_zones(
        context_df=context_df,
        current_price=current_price,
    )

    nearest_support = nearest["nearest_support"]
    nearest_resistance = nearest["nearest_resistance"]
    zones_containing_price = nearest["zones_containing_price"]

    support_strength = None
    resistance_strength = None
    stronger_side = "unknown"

    if nearest_support is not None:
        support_strength = nearest_support.get("zone_strength_score")

    if nearest_resistance is not None:
        resistance_strength = nearest_resistance.get("zone_strength_score")

    if support_strength is not None and resistance_strength is not None:
        if support_strength > resistance_strength:
            stronger_side = "support"
        elif resistance_strength > support_strength:
            stronger_side = "resistance"
        else:
            stronger_side = "balanced"

    elif support_strength is not None:
        stronger_side = "support"
    elif resistance_strength is not None:
        stronger_side = "resistance"

    return {
        "current_price": current_price,
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "zones_containing_price": zones_containing_price,
        "stronger_side": stronger_side,
    }