import pandas as pd


REQUIRED_CANDLE_COLUMNS = [
    "timestamp_utc",
    "open_price",
    "high_price",
    "low_price",
    "close_price",
]

REQUIRED_LEVEL_COLUMNS = [
    "level_type",
    "level_price",
    "touch_count",
    "validation_score",
    "zone_low",
    "zone_high",
]


def candle_touches_zone(candle: pd.Series, zone_low: float, zone_high: float) -> bool:
    """
    Returns True if the candle's price range intersects the zone.
    """
    candle_high = float(candle["high_price"])
    candle_low = float(candle["low_price"])

    return candle_low <= zone_high and candle_high >= zone_low


def evaluate_zone_outcome(
    future_candles_df: pd.DataFrame,
    level_type: str,
    zone_low: float,
    zone_high: float,
) -> str:
    """
    Evaluates whether a touched zone was respected, broken, or unresolved.

    Support:
    - respected if price moves above zone_high before breaking below zone_low
    - broken if price breaks below zone_low before moving above zone_high

    Resistance:
    - respected if price moves below zone_low before breaking above zone_high
    - broken if price breaks above zone_high before moving below zone_low
    """
    if future_candles_df.empty:
        return "unresolved"

    for _, candle in future_candles_df.iterrows():
        candle_high = float(candle["high_price"])
        candle_low = float(candle["low_price"])

        if level_type == "support":
            broke = candle_low < zone_low
            respected = candle_high > zone_high

        elif level_type == "resistance":
            broke = candle_high > zone_high
            respected = candle_low < zone_low

        else:
            raise ValueError(f"Unsupported level_type: {level_type}")

        if broke and respected:
            return "unresolved"

        if broke:
            return "broken"

        if respected:
            return "respected"

    return "unresolved"


def evaluate_single_zone(
    candles_df: pd.DataFrame,
    level_row: pd.Series,
    reaction_window: int = 5,
    cooldown: int = 3,
) -> dict:
    """
    Evaluates how a single support/resistance zone behaved historically.

    reaction_window:
        Number of future candles to inspect after a touch.

    cooldown:
        Number of candles to skip after detecting a touch, to avoid counting
        repeated nearby touches as multiple independent events.
    """
    level_type = level_row["level_type"]
    level_price = float(level_row["level_price"])
    zone_low = float(level_row["zone_low"])
    zone_high = float(level_row["zone_high"])
    pivot_touch_count = int(level_row["touch_count"])
    validation_score = float(level_row["validation_score"])

    respected_touches = 0
    broken_touches = 0
    unresolved_touches = 0
    tested_touches = 0

    last_touch_index = -999999

    max_index = len(candles_df) - 1

    for index in range(len(candles_df)):
        if index <= last_touch_index + cooldown:
            continue

        current_candle = candles_df.iloc[index]

        if not candle_touches_zone(current_candle, zone_low, zone_high):
            continue

        future_start = index + 1
        future_end = min(index + 1 + reaction_window, max_index + 1)

        future_candles_df = candles_df.iloc[future_start:future_end].copy()

        outcome = evaluate_zone_outcome(
            future_candles_df=future_candles_df,
            level_type=level_type,
            zone_low=zone_low,
            zone_high=zone_high,
        )

        tested_touches += 1

        if outcome == "respected":
            respected_touches += 1
        elif outcome == "broken":
            broken_touches += 1
        else:
            unresolved_touches += 1

        last_touch_index = index

    resolved_touches = respected_touches + broken_touches

    respect_rate = None
    break_rate = None

    if resolved_touches > 0:
        respect_rate = respected_touches / resolved_touches
        break_rate = broken_touches / resolved_touches

    return {
        "level_type": level_type,
        "level_price": level_price,
        "zone_low": zone_low,
        "zone_high": zone_high,
        "pivot_touch_count": pivot_touch_count,
        "validation_score": validation_score,
        "tested_touches": tested_touches,
        "respected_touches": respected_touches,
        "broken_touches": broken_touches,
        "unresolved_touches": unresolved_touches,
        "respect_rate": respect_rate,
        "break_rate": break_rate,
    }


def evaluate_support_resistance_zones(
    candles_df: pd.DataFrame,
    levels_df: pd.DataFrame,
    reaction_window: int = 5,
    cooldown: int = 3,
) -> pd.DataFrame:
    """
    Evaluates all detected support/resistance zones.
    """
    if candles_df.empty:
        raise ValueError("candles_df is empty.")

    if levels_df.empty:
        return pd.DataFrame()

    missing_candle_columns = [
        column for column in REQUIRED_CANDLE_COLUMNS if column not in candles_df.columns
    ]

    if missing_candle_columns:
        raise ValueError(
            f"Missing required candle columns: {missing_candle_columns}"
        )

    missing_level_columns = [
        column for column in REQUIRED_LEVEL_COLUMNS if column not in levels_df.columns
    ]

    if missing_level_columns:
        raise ValueError(
            f"Missing required level columns: {missing_level_columns}"
        )

    evaluation_df = candles_df.copy()

    evaluation_df["timestamp_utc"] = pd.to_datetime(evaluation_df["timestamp_utc"])

    price_columns = [
        "open_price",
        "high_price",
        "low_price",
        "close_price",
    ]

    for column in price_columns:
        evaluation_df[column] = evaluation_df[column].astype(float)

    evaluation_df = evaluation_df.sort_values("timestamp_utc").reset_index(drop=True)

    results = []

    for _, level_row in levels_df.iterrows():
        result = evaluate_single_zone(
            candles_df=evaluation_df,
            level_row=level_row,
            reaction_window=reaction_window,
            cooldown=cooldown,
        )
        results.append(result)

    results_df = pd.DataFrame(results)

    if results_df.empty:
        return results_df

    results_df = results_df.sort_values(
        ["respect_rate", "tested_touches", "pivot_touch_count"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    return results_df