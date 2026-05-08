from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
import json

import pandas as pd

from mdp.backtesting.trend_channel_entry_refinement_validation import (
    EntryRefinementValidationConfig,
    build_entry_refinement_dataset,
)
from mdp.backtesting.trend_channel_candidate_refinement_validation import (
    CandidateRefinementRule,
    build_default_refinement_rules,
    build_refinement_preview,
    summarize_refinement_rules,
    summarize_refinement_walk_forward,
)


@dataclass(frozen=True)
class EntryResearchCacheConfig:
    """
    Stores the settings used to build a reusable trend-channel entry research cache.
    """

    symbol: str
    timeframes: str
    analysis_limit: int = 220
    dataset_limit: int = 3200
    warmup_candles: int = 220
    horizon_candles: int = 100
    cutoff_step_candles: int = 20
    max_cutoffs: int | None = 70
    management_policy_name: str = "targetless_scalp_manage_only"
    minimum_rr: float = 1.50
    min_trades_for_candidate: int = 8
    active_only: bool = True
    quick: bool = False


@dataclass(frozen=True)
class EntryResearchCachePaths:
    """
    Stores file paths created by the cache builder.
    """

    dataset_path: Path
    workload_path: Path
    metadata_path: Path


def _safe_symbol_name(symbol: str) -> str:
    """
    Converts a market symbol into a safe file-name fragment.
    """
    return symbol.lower().replace(" ", "_").replace("/", "_").replace("\\", "_")


def _quick_adjusted_config(config: EntryResearchCacheConfig) -> EntryResearchCacheConfig:
    """
    Shrinks the workload while keeping the same code path for smoke testing.
    """
    if not config.quick:
        return config

    return EntryResearchCacheConfig(
        symbol=config.symbol,
        timeframes=config.timeframes,
        analysis_limit=min(config.analysis_limit, 200),
        dataset_limit=min(config.dataset_limit, 1400),
        warmup_candles=min(config.warmup_candles, 200),
        horizon_candles=min(config.horizon_candles, 80),
        cutoff_step_candles=config.cutoff_step_candles,
        max_cutoffs=min(config.max_cutoffs or 35, 35),
        management_policy_name=config.management_policy_name,
        minimum_rr=config.minimum_rr,
        min_trades_for_candidate=config.min_trades_for_candidate,
        active_only=config.active_only,
        quick=True,
    )


def build_entry_research_cache(config: EntryResearchCacheConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Builds the expensive entry-variant research dataset once so later validators can reuse it.
    """
    started_at = perf_counter()
    adjusted = _quick_adjusted_config(config)

    # Reuse the existing entry-refinement dataset builder instead of duplicating strategy logic.
    entry_config = EntryRefinementValidationConfig(
        symbol=adjusted.symbol,
        timeframes=adjusted.timeframes,
        analysis_limit=adjusted.analysis_limit,
        dataset_limit=adjusted.dataset_limit,
        warmup_candles=adjusted.warmup_candles,
        horizon_candles=adjusted.horizon_candles,
        cutoff_step_candles=adjusted.cutoff_step_candles,
        max_cutoffs=adjusted.max_cutoffs,
        management_policy_name=adjusted.management_policy_name,
        minimum_rr=adjusted.minimum_rr,
        min_trades_for_candidate=adjusted.min_trades_for_candidate,
        active_only=adjusted.active_only,
    )
    dataset_df, workload_df = build_entry_refinement_dataset(entry_config)

    elapsed_seconds = round(perf_counter() - started_at, 3)
    metadata: dict[str, Any] = {
        "cache_type": "trend_channel_entry_research_cache_v1",
        "symbol": adjusted.symbol,
        "timeframes": adjusted.timeframes,
        "analysis_limit": adjusted.analysis_limit,
        "dataset_limit": adjusted.dataset_limit,
        "warmup_candles": adjusted.warmup_candles,
        "horizon_candles": adjusted.horizon_candles,
        "cutoff_step_candles": adjusted.cutoff_step_candles,
        "max_cutoffs": adjusted.max_cutoffs,
        "management_policy_name": adjusted.management_policy_name,
        "minimum_rr": adjusted.minimum_rr,
        "min_trades_for_candidate": adjusted.min_trades_for_candidate,
        "active_only": adjusted.active_only,
        "quick": adjusted.quick,
        "dataset_rows": int(len(dataset_df)),
        "unique_analysis_cutoffs": int(dataset_df["analysis_cutoff"].nunique()) if "analysis_cutoff" in dataset_df.columns else 0,
        "unique_baseline_trade_plans": int(
            dataset_df.loc[dataset_df.get("entry_variant_name") == "baseline_current_close", "analysis_cutoff"].nunique()
        ) if "entry_variant_name" in dataset_df.columns and "analysis_cutoff" in dataset_df.columns else 0,
        "elapsed_seconds_total_cache_build": elapsed_seconds,
        "cache_design": "entry_dataset_built_once_for_reusable_refinement_validation",
    }

    workload_df = workload_df.copy()
    workload_df["elapsed_seconds_total_cache_build"] = elapsed_seconds
    workload_df["cache_design"] = metadata["cache_design"]
    return dataset_df, workload_df, metadata


def save_entry_research_cache(
    dataset_df: pd.DataFrame,
    workload_df: pd.DataFrame,
    metadata: dict[str, Any],
    output_dir: str | Path = "outputs/trend_channel_research_cache",
    tag: str | None = None,
) -> EntryResearchCachePaths:
    """
    Saves the cache dataset, workload summary, and metadata to disk.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Use a readable file stem so multiple cache runs can coexist safely.
    symbol_part = _safe_symbol_name(str(metadata.get("symbol", "market")))
    timeframe_part = str(metadata.get("timeframes", "tf")).lower().replace(",", "_")
    tag_part = f"_{tag}" if tag else ""
    file_stem = f"{symbol_part}_{timeframe_part}_entry_research_cache{tag_part}"

    dataset_path = output_path / f"{file_stem}.csv"
    workload_path = output_path / f"{file_stem}_workload.csv"
    metadata_path = output_path / f"{file_stem}_metadata.json"

    dataset_df.to_csv(dataset_path, index=False)
    workload_df.to_csv(workload_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    return EntryResearchCachePaths(dataset_path=dataset_path, workload_path=workload_path, metadata_path=metadata_path)


def load_entry_research_cache(cache_path: str | Path) -> pd.DataFrame:
    """
    Loads a previously saved entry research cache from CSV.
    """
    dataset_df = pd.read_csv(cache_path)
    if "analysis_cutoff" in dataset_df.columns:
        dataset_df["analysis_cutoff"] = pd.to_datetime(dataset_df["analysis_cutoff"], errors="coerce")
    return dataset_df


def _select_rules(rule_names: str | None = None) -> list[CandidateRefinementRule]:
    """
    Selects the requested refinement rules or returns all default rules.
    """
    rules = build_default_refinement_rules()
    if not rule_names:
        return rules

    requested = {name.strip() for name in rule_names.split(",") if name.strip()}
    selected = [rule for rule in rules if rule.rule_name in requested]
    missing = sorted(requested - {rule.rule_name for rule in selected})
    if missing:
        available = ", ".join(rule.rule_name for rule in rules)
        raise ValueError(f"Unknown refinement rules: {missing}. Available rules: {available}")
    return selected


def validate_refinements_from_entry_cache(
    dataset_df: pd.DataFrame,
    fold_count: int = 3,
    min_trades_for_candidate: int = 8,
    rule_names: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Validates refinement rules using an already-built entry research cache.
    """
    rules = _select_rules(rule_names)

    # These summaries are cheap because the full strategy pipeline has already been cached.
    summary_df = summarize_refinement_rules(
        dataset_df=dataset_df,
        rules=rules,
        min_trades_for_candidate=min_trades_for_candidate,
    )
    robust_df, fold_df = summarize_refinement_walk_forward(
        dataset_df=dataset_df,
        fold_count=fold_count,
        rules=rules,
        min_trades_for_candidate=min_trades_for_candidate,
    )

    return summary_df, robust_df, fold_df


def build_best_refinement_preview_from_cache(dataset_df: pd.DataFrame, robust_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a compact preview for the top refinement rule in a cached validation run.
    """
    if robust_df.empty:
        return pd.DataFrame()
    best_rule_name = str(robust_df.iloc[0].get("rule_name"))
    return build_refinement_preview(dataset_df, best_rule_name)
