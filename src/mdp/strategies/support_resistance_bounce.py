import pandas as pd


def build_support_resistance_bounce_signals(
    candles_df: pd.DataFrame,
    context_df: pd.DataFrame,
    min_respect_rate: float = 0.55,
    reward_risk_ratio: float = 1.5,
    min_risk_points: float = 0.0,
    confirmation_mode: str = "rejection_close",
) -> pd.DataFrame:
    """
    Builds simple support/resistance bounce signals.

    This is a strategy plugin, not part of the core backtesting engine.

    Long idea:
    - price touches support zone
    - support has sufficient historical respect_rate
    - take profit is nearest resistance

    Short idea:
    - price touches resistance zone
    - resistance has sufficient historical respect_rate
    - take profit is nearest support
    """
    if candles_df.empty or context_df.empty:
        return pd.DataFrame()
    
    # Make sure the minimum risk filter is not negative.
    if min_risk_points < 0:
        raise ValueError("min_risk_points cannot be negative.")
    
    # Make sure the confirmation mode is one of the allowed options.
    valid_confirmation_modes = {"touch", "rejection_close"}

    if confirmation_mode not in valid_confirmation_modes:
        raise ValueError(
            f"Invalid confirmation_mode: {confirmation_mode}. "
            f"Valid options are: {valid_confirmation_modes}"
        )

    market_df = candles_df.copy()
    market_df["timestamp_utc"] = pd.to_datetime(market_df["timestamp_utc"])

    for column in ["open_price", "high_price", "low_price", "close_price"]:
        market_df[column] = market_df[column].astype(float)

    zones_df = context_df.copy()

    numeric_columns = [
        "level_price",
        "zone_low",
        "zone_high",
        "respect_rate",
        "break_rate",
    ]

    for column in numeric_columns:
        zones_df[column] = pd.to_numeric(zones_df[column], errors="coerce")

    support_zones = zones_df[
        (zones_df["level_type"] == "support")
        & (zones_df["respect_rate"] >= min_respect_rate)
    ].copy()

    resistance_zones = zones_df[
        (zones_df["level_type"] == "resistance")
        & (zones_df["respect_rate"] >= min_respect_rate)
    ].copy()

    signals = []

    for _, candle in market_df.iterrows():
        # Read the current candle values needed for setup detection.
        candle_time = candle["timestamp_utc"]
        candle_open = float(candle["open_price"])
        candle_high = float(candle["high_price"])
        candle_low = float(candle["low_price"])
        candle_close = float(candle["close_price"])

        for _, support in support_zones.iterrows():
            touched_support = (
                candle_low <= support["zone_high"]
                and candle_high >= support["zone_low"]
            )

            if not touched_support:
                continue

            # Optional confirmation: require a bullish rejection close from support.
            if confirmation_mode == "rejection_close":
                confirmed_support_rejection = (
                    candle_close > candle_open
                    and candle_close >= float(support["level_price"])
                )

                if not confirmed_support_rejection:
                    continue

            resistances_above = resistance_zones[
                resistance_zones["level_price"] > candle_close
            ].copy()

            if resistances_above.empty:
                continue

            nearest_resistance = (
                resistances_above.assign(
                    distance=resistances_above["level_price"] - candle_close
                )
                .sort_values("distance")
                .iloc[0]
            )

            entry_price = candle_close
            stop_loss = float(support["zone_low"])
            take_profit = float(nearest_resistance["level_price"])

            # Calculate how much price can move against the trade before stop loss.
            risk = entry_price - stop_loss

            # Calculate how much price can move in favor of the trade before take profit.
            reward = take_profit - entry_price

            # Skip invalid long setups where the stop is not below the entry.
            if risk <= 0:
                continue

            # Skip setups with unrealistically small risk distance.
            if risk < min_risk_points:
                continue

            rr = reward / risk

            if rr < reward_risk_ratio:
                continue

            signals.append(
                {
                    "timestamp_utc": candle_time,
                    "direction": "long",
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "expected_rr": rr,
                    "source_zone_type": "support",
                    "target_zone_type": "resistance",
                }
            )

        for _, resistance in resistance_zones.iterrows():
            touched_resistance = (
                candle_low <= resistance["zone_high"]
                and candle_high >= resistance["zone_low"]
            )

            if not touched_resistance:
                continue

            # Optional confirmation: require a bearish rejection close from resistance.
            if confirmation_mode == "rejection_close":
                confirmed_resistance_rejection = (
                    candle_close < candle_open
                    and candle_close <= float(resistance["level_price"])
                )

                if not confirmed_resistance_rejection:
                    continue

            supports_below = support_zones[
                support_zones["level_price"] < candle_close
            ].copy()

            if supports_below.empty:
                continue

            nearest_support = (
                supports_below.assign(
                    distance=candle_close - supports_below["level_price"]
                )
                .sort_values("distance")
                .iloc[0]
            )

            entry_price = candle_close
            stop_loss = float(resistance["zone_high"])
            take_profit = float(nearest_support["level_price"])

            # Calculate how much price can move against the trade before stop loss.
            risk = stop_loss - entry_price

            # Calculate how much price can move in favor of the trade before take profit.
            reward = entry_price - take_profit

            # Skip invalid short setups where the stop is not above the entry.
            if risk <= 0:
                continue

            # Skip setups with unrealistically small risk distance.
            if risk < min_risk_points:
                continue

            rr = reward / risk

            if rr < reward_risk_ratio:
                continue

            signals.append(
                {
                    "timestamp_utc": candle_time,
                    "direction": "short",
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "expected_rr": rr,
                    "source_zone_type": "resistance",
                    "target_zone_type": "support",
                }
            )

    return pd.DataFrame(signals)