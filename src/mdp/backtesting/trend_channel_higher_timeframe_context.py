"""Higher-timeframe context enrichment for Step Index trend-channel ML datasets.

This module enriches an already exported ML-ready candidate dataset with D1/H4/H1/M15/M5
context features using only candles that were closed before each analysis cutoff.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mdp.database.candle_repository import fetch_market_candles_between
from mdp.timeframes.registry import get_timeframe_delta, validate_timeframe


@dataclass(frozen=True)
class HigherTimeframeContextConfig:
    ml_dataset_path: str
    symbol: str = "Step Index"
    context_timeframes: tuple[str, ...] = ("D1", "H4", "H1", "M15", "M5")
    context_lookback_candles: int = 60
    tag: str = "step_candidate_htf_context"
    output_dir: str = "outputs/trend_channel_htf_context"


def _safe_float_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def _load_ml_dataset(path: str | Path) -> pd.DataFrame:
    dataset = pd.read_csv(path)
    if "analysis_cutoff" not in dataset.columns:
        raise ValueError("ML dataset must include analysis_cutoff.")
    if "label_managed_r" not in dataset.columns:
        raise ValueError("ML dataset must include label_managed_r.")

    # Keep the rows chronological so later validation remains walk-forward friendly.
    dataset = dataset.copy()
    dataset["analysis_cutoff"] = pd.to_datetime(dataset["analysis_cutoff"], errors="coerce")
    dataset = dataset.dropna(subset=["analysis_cutoff"]).sort_values("analysis_cutoff").reset_index(drop=True)
    if "trade_date" not in dataset.columns:
        dataset["trade_date"] = dataset["analysis_cutoff"].dt.date.astype(str)
    if "hour_utc" not in dataset.columns:
        dataset["hour_utc"] = dataset["analysis_cutoff"].dt.hour
    return dataset


def _fetch_context_candles(
    symbol: str,
    timeframe: str,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> pd.DataFrame:
    tf = validate_timeframe(timeframe)
    df = fetch_market_candles_between(
        symbol=symbol,
        timeframe=tf,
        start_time=start_time.to_pydatetime(),
        end_time=end_time.to_pydatetime(),
    )
    if df.empty:
        return df

    df = df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")
    price_cols = ["open_price", "high_price", "low_price", "close_price"]
    for col in price_cols:
        df[col] = _safe_float_series(df[col])
    if "tick_volume" in df.columns:
        df["tick_volume"] = pd.to_numeric(df["tick_volume"], errors="coerce")
    return df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc").reset_index(drop=True)


def _prepare_context_features(candles: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Create scale-aware context features from closed higher-timeframe candles."""
    tf = validate_timeframe(timeframe)
    if candles.empty:
        return pd.DataFrame()

    delta = get_timeframe_delta(tf)
    if not isinstance(delta, pd.Timedelta):
        raise ValueError(f"Only fixed timeframes are supported for this exporter: {tf}")

    df = candles.copy().sort_values("timestamp_utc").reset_index(drop=True)
    prefix = f"htf_{tf.lower()}"

    candle_range = (df["high_price"] - df["low_price"]).abs()
    candle_body = (df["close_price"] - df["open_price"]).abs()
    median_range_20 = candle_range.rolling(window=20, min_periods=5).median()
    median_range_20 = median_range_20.replace(0, pd.NA)
    close_shift_5 = df["close_price"].shift(5)

    # available_at is the exact time when this candle is closed and safe to use.
    features = pd.DataFrame(
        {
            "available_at": df["timestamp_utc"] + delta,
            f"{prefix}_candle_start": df["timestamp_utc"],
            f"{prefix}_close": df["close_price"],
            f"{prefix}_range": candle_range,
            f"{prefix}_body": candle_body,
            f"{prefix}_body_to_range": candle_body / candle_range.replace(0, pd.NA),
            f"{prefix}_close_position_ratio": (df["close_price"] - df["low_price"]) / candle_range.replace(0, pd.NA),
            f"{prefix}_range_to_median_20": candle_range / median_range_20,
            f"{prefix}_slope_5_range_units": (df["close_price"] - close_shift_5) / median_range_20,
        }
    )

    direction = pd.Series("neutral", index=df.index, dtype="object")
    direction[df["close_price"] > df["open_price"]] = "bullish"
    direction[df["close_price"] < df["open_price"]] = "bearish"
    features[f"{prefix}_candle_direction"] = direction

    slope = pd.to_numeric(features[f"{prefix}_slope_5_range_units"], errors="coerce")
    close_pos = pd.to_numeric(features[f"{prefix}_close_position_ratio"], errors="coerce")
    bias = pd.Series("neutral", index=df.index, dtype="object")
    bias[(slope > 0.50) & (close_pos >= 0.55)] = "bullish"
    bias[(slope < -0.50) & (close_pos <= 0.45)] = "bearish"
    features[f"{prefix}_bias"] = bias

    return features.sort_values("available_at").reset_index(drop=True)


def _merge_context(dataset: pd.DataFrame, context_features: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    tf = validate_timeframe(timeframe)
    if context_features.empty:
        prefix = f"htf_{tf.lower()}"
        out = dataset.copy()
        out[f"{prefix}_context_available"] = False
        return out

    prefix = f"htf_{tf.lower()}"
    left = dataset.sort_values("analysis_cutoff").copy()
    right = context_features.sort_values("available_at").copy()

    # merge_asof with direction='backward' guarantees no future HTF candle is used.
    merged = pd.merge_asof(
        left,
        right,
        left_on="analysis_cutoff",
        right_on="available_at",
        direction="backward",
        allow_exact_matches=True,
    )

    merged[f"{prefix}_context_available"] = merged["available_at"].notna()
    merged[f"{prefix}_age_minutes"] = (
        (merged["analysis_cutoff"] - merged["available_at"]).dt.total_seconds() / 60.0
    )

    if "plan_direction" in merged.columns and f"{prefix}_bias" in merged.columns:
        plan = merged["plan_direction"].fillna("unknown").astype(str).str.lower().str.strip()
        bias = merged[f"{prefix}_bias"].fillna("unknown").astype(str).str.lower().str.strip()
        merged[f"{prefix}_bias_aligned_with_plan"] = (bias == plan) & bias.isin(["bullish", "bearish"])
        merged[f"{prefix}_bias_against_plan"] = bias.isin(["bullish", "bearish"]) & plan.isin(["bullish", "bearish"]) & (bias != plan)
        merged[f"{prefix}_bias_neutral_or_aligned"] = merged[f"{prefix}_bias_aligned_with_plan"] | bias.eq("neutral")

    # Rename this generic merge column before the next timeframe merge.
    merged = merged.rename(columns={"available_at": f"{prefix}_available_at"})
    return merged.sort_values("analysis_cutoff").reset_index(drop=True)


def _coverage_summary(enriched: pd.DataFrame, timeframe: str) -> dict:
    tf = validate_timeframe(timeframe)
    prefix = f"htf_{tf.lower()}"
    available_col = f"{prefix}_context_available"
    bias_col = f"{prefix}_bias"
    age_col = f"{prefix}_age_minutes"
    against_col = f"{prefix}_bias_against_plan"

    rows = len(enriched)
    coverage = float(enriched[available_col].mean()) if available_col in enriched.columns and rows else 0.0
    bias_counts = {}
    if bias_col in enriched.columns:
        bias_counts = enriched[bias_col].fillna("missing").astype(str).value_counts().to_dict()

    avg_age = float(pd.to_numeric(enriched.get(age_col, pd.Series(dtype=float)), errors="coerce").mean()) if rows else 0.0
    against_rate = 0.0
    if against_col in enriched.columns:
        against_rate = float(enriched[against_col].fillna(False).mean())

    return {
        "timeframe": tf,
        "rows": rows,
        "context_coverage_rate": coverage,
        "average_context_age_minutes": avg_age,
        "bias_against_plan_rate": against_rate,
        "bias_distribution": bias_counts,
    }


def export_higher_timeframe_context(config: HigherTimeframeContextConfig) -> dict[str, pd.DataFrame]:
    dataset = _load_ml_dataset(config.ml_dataset_path)
    if dataset.empty:
        raise ValueError("ML dataset is empty; cannot enrich with HTF context.")

    min_cutoff = dataset["analysis_cutoff"].min()
    max_cutoff = dataset["analysis_cutoff"].max()
    largest_delta = max(get_timeframe_delta(tf) for tf in config.context_timeframes)
    if not isinstance(largest_delta, pd.Timedelta):
        raise ValueError("Only fixed timeframes are supported.")

    # Pull enough candle history before the first setup for rolling context features.
    start_time = min_cutoff - (largest_delta * max(5, config.context_lookback_candles))
    end_time = max_cutoff + pd.Timedelta(days=1)

    enriched = dataset.copy()
    coverage_rows = []

    for timeframe in config.context_timeframes:
        tf = validate_timeframe(timeframe)
        candles = _fetch_context_candles(config.symbol, tf, start_time, end_time)
        features = _prepare_context_features(candles, tf)
        enriched = _merge_context(enriched, features, tf)
        coverage = _coverage_summary(enriched, tf)
        coverage["loaded_candles"] = int(len(candles))
        coverage_rows.append(coverage)

    summary = pd.DataFrame(
        [
            {
                "ml_dataset_path": config.ml_dataset_path,
                "symbol": config.symbol,
                "input_rows": int(len(dataset)),
                "output_rows": int(len(enriched)),
                "first_analysis_cutoff": dataset["analysis_cutoff"].min(),
                "last_analysis_cutoff": dataset["analysis_cutoff"].max(),
                "context_timeframes": ",".join(config.context_timeframes),
                "context_lookback_candles": config.context_lookback_candles,
                "design": "asof_merge_closed_htf_candles_only_no_lookahead",
            }
        ]
    )
    coverage_df = pd.DataFrame(coverage_rows)

    return {
        "summary": summary,
        "coverage": coverage_df,
        "enriched_dataset": enriched,
    }


def save_higher_timeframe_context_outputs(
    config: HigherTimeframeContextConfig,
    outputs: dict[str, pd.DataFrame],
) -> dict[str, str]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, str] = {}
    for name, df in outputs.items():
        path = output_dir / f"{config.tag}_{name}.csv"
        df.to_csv(path, index=False)
        saved[name] = str(path)
    return saved
