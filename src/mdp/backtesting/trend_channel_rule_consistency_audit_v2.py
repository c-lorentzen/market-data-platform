"""Rule-consistency audit v2 for Step Index trend-channel candidate datasets.

This module audits whether refinement rules really change the final
one-trade-per-cutoff selection. It also derives missing numeric buckets from
cache columns so rules like "high RR" do not silently return zero rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RuleAuditV2Config:
    """Small config object to keep script arguments clean."""

    cache_path: str
    rule_names: tuple[str, ...]
    tag: str = "step_rule_audit_v2"
    output_dir: str = "outputs/trend_channel_rule_audits"
    min_target_state_required: bool = True


def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Return the first available column name from a candidate list."""

    for column_name in candidates:
        if column_name in df.columns:
            return column_name
    return None


def _to_bool_series(series: pd.Series) -> pd.Series:
    """Convert messy True/False values from CSV into a clean boolean series."""

    if series.dtype == bool:
        return series.fillna(False)

    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y"})


def _safe_numeric(df: pd.DataFrame, column_name: str | None) -> pd.Series:
    """Return a numeric series; if the column is missing, return NaNs."""

    if column_name is None:
        return pd.Series(np.nan, index=df.index)

    return pd.to_numeric(df[column_name], errors="coerce")


def _derive_quantile_bucket(values: pd.Series, prefix: str) -> pd.Series:
    """Create low/mid/high buckets from numeric values.

    This avoids relying on columns that may not exist in older cache files.
    """

    numeric_values = pd.to_numeric(values, errors="coerce")
    bucket = pd.Series(f"{prefix}_unknown", index=values.index, dtype="object")

    valid = numeric_values.dropna()
    if valid.empty or valid.nunique() < 3:
        return bucket

    q33 = valid.quantile(1 / 3)
    q66 = valid.quantile(2 / 3)

    bucket.loc[numeric_values <= q33] = f"{prefix}_low"
    bucket.loc[(numeric_values > q33) & (numeric_values <= q66)] = f"{prefix}_mid"
    bucket.loc[numeric_values > q66] = f"{prefix}_high"
    return bucket


def prepare_audit_dataset(cache_path: str) -> pd.DataFrame:
    """Load a cache CSV and add robust helper columns used by audit rules."""

    df = pd.read_csv(cache_path)

    # This keeps chronological fields usable for grouping and sorting.
    if "analysis_cutoff" in df.columns:
        df["analysis_cutoff"] = pd.to_datetime(df["analysis_cutoff"], errors="coerce")
    if "trade_date" not in df.columns and "analysis_cutoff" in df.columns:
        df["trade_date"] = df["analysis_cutoff"].dt.date.astype(str)

    entered_col = _first_existing_column(df, ["entered_trade", "entered", "is_entered"])
    if entered_col is not None:
        df["_entered_trade"] = _to_bool_series(df[entered_col])
    else:
        # If the cache does not include an entered flag, keep all rows and let
        # the rule filters decide. This keeps the audit backward compatible.
        df["_entered_trade"] = True

    rr_col = _first_existing_column(
        df,
        [
            "_best_available_rr",
            "best_available_rr",
            "best_available_rr_after_entry",
            "best_rr",
        ],
    )
    risk_col = _first_existing_column(df, ["risk_points", "risk", "sl_distance_points"])
    score_col = _first_existing_column(
        df,
        [
            "pre_entry_selection_score",
            "rolling_quality_score",
            "event_quality_score",
            "quality_score",
        ],
    )

    df["_best_available_rr_numeric"] = _safe_numeric(df, rr_col)
    df["_risk_points_numeric"] = _safe_numeric(df, risk_col)
    df["_selection_score_numeric"] = _safe_numeric(df, score_col)

    # Derive missing buckets directly from numeric values. This fixes the issue
    # where high-RR rules can return zero rows simply because bucket columns are absent.
    if "best_available_rr_bucket" not in df.columns:
        df["best_available_rr_bucket"] = _derive_quantile_bucket(
            df["_best_available_rr_numeric"], "best_available_rr"
        )

    if "risk_points_bucket" not in df.columns:
        df["risk_points_bucket"] = _derive_quantile_bucket(
            df["_risk_points_numeric"], "risk_points"
        )

    # Use clear normalized strings for robust comparisons.
    for column_name in [
        "target_plan_state",
        "available_targets",
        "entry_variant_name",
        "entry_mode",
        "scenario_state",
        "event_state",
        "plan_direction",
    ]:
        if column_name not in df.columns:
            df[column_name] = ""
        df[column_name] = df[column_name].fillna("").astype(str)

    outcome_col = _first_existing_column(df, ["managed_realized_r", "label_managed_r"])
    df["_managed_realized_r_numeric"] = _safe_numeric(df, outcome_col)

    return df


def _base_mask(df: pd.DataFrame) -> pd.Series:
    """Candidate v1 base: entered trades with a usable target plan."""

    target_state = df["target_plan_state"].str.lower()
    available_targets = df["available_targets"].str.upper()

    has_target_state = ~target_state.isin(
        {"", "nan", "none", "no_available_target_after_entry_refinement"}
    )
    has_available_targets = ~available_targets.isin({"", "NAN", "NONE"})

    return df["_entered_trade"] & has_target_state & has_available_targets


def _not_tp3_only(df: pd.DataFrame) -> pd.Series:
    """Reject TP3-only rows using both target_plan_state and available_targets."""

    target_state = df["target_plan_state"].str.lower()
    available_targets = df["available_targets"].str.upper().str.replace(" ", "", regex=False)
    return (target_state != "tp3_available") & (available_targets != "TP3")


def apply_audit_rule_v2(df: pd.DataFrame, rule_name: str) -> pd.DataFrame:
    """Apply a named rule to the raw cache rows before deduplication."""

    base = _base_mask(df)
    rr_high = df["best_available_rr_bucket"].eq("best_available_rr_high")
    risk_high = df["risk_points_bucket"].eq("risk_points_high")

    if rule_name == "candidate_v1_base":
        mask = base
    elif rule_name == "avoid_tp3_only":
        mask = base & _not_tp3_only(df)
    elif rule_name == "require_best_available_rr_high":
        mask = base & rr_high
    elif rule_name == "require_risk_points_high":
        mask = base & risk_high
    elif rule_name == "keep_high_rr_or_risk_high":
        mask = base & (rr_high | risk_high)
    elif rule_name == "keep_high_rr_or_bullish_pullback":
        bullish_pullback = df["scenario_state"].eq("bullish_structure_bearish_entry_pullback")
        mask = base & (rr_high | bullish_pullback)
    elif rule_name == "avoid_entry_price_low":
        # Optional compatibility rule. If the column is missing, this becomes base.
        if "entry_price_bucket" in df.columns:
            mask = base & ~df["entry_price_bucket"].eq("entry_price_low")
        else:
            mask = base
    else:
        raise ValueError(f"Unknown rule_name for audit v2: {rule_name}")

    return df.loc[mask].copy()


def dedupe_one_trade_per_cutoff(selected_df: pd.DataFrame) -> pd.DataFrame:
    """Keep one selected row per analysis_cutoff using pre-entry score only."""

    if selected_df.empty:
        return selected_df.copy()

    sort_columns = []
    ascending = []

    if "analysis_cutoff" in selected_df.columns:
        sort_columns.append("analysis_cutoff")
        ascending.append(True)

    # Higher score is better. This is intentionally pre-entry only.
    sort_columns.append("_selection_score_numeric")
    ascending.append(False)

    # Prefer rows with higher RR as a deterministic tie-breaker.
    sort_columns.append("_best_available_rr_numeric")
    ascending.append(False)

    selected_df = selected_df.sort_values(sort_columns, ascending=ascending, kind="mergesort")

    if "analysis_cutoff" not in selected_df.columns:
        return selected_df

    return selected_df.drop_duplicates(subset=["analysis_cutoff"], keep="first").copy()


def summarize_rule(rule_name: str, raw_df: pd.DataFrame, final_df: pd.DataFrame) -> dict:
    """Build one summary row for a rule."""

    managed_r = final_df["_managed_realized_r_numeric"] if not final_df.empty else pd.Series(dtype=float)
    positive = managed_r > 0
    full_loss = managed_r <= -0.95

    return {
        "rule_name": rule_name,
        "raw_selected_rows": int(len(raw_df)),
        "deduped_selected_rows": int(len(final_df)),
        "unique_analysis_cutoffs": int(final_df["analysis_cutoff"].nunique()) if "analysis_cutoff" in final_df else int(len(final_df)),
        "positive_exit_rate": float(positive.mean()) if len(final_df) else 0.0,
        "full_loss_rate": float(full_loss.mean()) if len(final_df) else 0.0,
        "average_managed_r": float(managed_r.mean()) if len(final_df) else 0.0,
        "median_managed_r": float(managed_r.median()) if len(final_df) else 0.0,
        "total_managed_r": float(managed_r.sum()) if len(final_df) else 0.0,
        "target_plan_distribution": final_df["target_plan_state"].value_counts().to_dict() if len(final_df) else {},
        "entry_variant_distribution": final_df["entry_variant_name"].value_counts().to_dict() if len(final_df) else {},
        "rr_bucket_distribution": final_df["best_available_rr_bucket"].value_counts().to_dict() if len(final_df) else {},
        "risk_bucket_distribution": final_df["risk_points_bucket"].value_counts().to_dict() if len(final_df) else {},
    }


def build_pairwise_comparison(final_by_rule: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compare final selected cutoffs across rules."""

    rows: list[dict] = []
    rule_names = list(final_by_rule)

    for left_index, left_rule in enumerate(rule_names):
        for right_rule in rule_names[left_index + 1 :]:
            left_df = final_by_rule[left_rule]
            right_df = final_by_rule[right_rule]

            left_cutoffs = set(left_df["analysis_cutoff"].dropna().astype(str)) if "analysis_cutoff" in left_df else set()
            right_cutoffs = set(right_df["analysis_cutoff"].dropna().astype(str)) if "analysis_cutoff" in right_df else set()

            rows.append(
                {
                    "left_rule": left_rule,
                    "right_rule": right_rule,
                    "same_cutoff_count": len(left_cutoffs & right_cutoffs),
                    "left_only_cutoffs": len(left_cutoffs - right_cutoffs),
                    "right_only_cutoffs": len(right_cutoffs - left_cutoffs),
                    "identical_final_selection": left_cutoffs == right_cutoffs,
                }
            )

    return pd.DataFrame(rows)


def build_violation_check(final_by_rule: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Check whether avoid_tp3_only-type rules still include TP3-only rows."""

    rows: list[dict] = []

    for rule_name, df in final_by_rule.items():
        if df.empty:
            tp3_only_count = 0
        else:
            target_state = df["target_plan_state"].str.lower()
            available_targets = df["available_targets"].str.upper().str.replace(" ", "", regex=False)
            tp3_only_count = int(((target_state == "tp3_available") | (available_targets == "TP3")).sum())

        rows.append(
            {
                "rule_name": rule_name,
                "final_selected_rows": int(len(df)),
                "tp3_only_final_rows": tp3_only_count,
                "tp3_only_violation_state": "violation" if "avoid_tp3" in rule_name and tp3_only_count else "ok",
            }
        )

    return pd.DataFrame(rows)


def run_rule_consistency_audit_v2(config: RuleAuditV2Config) -> dict[str, pd.DataFrame | str]:
    """Run the full audit and save CSV outputs."""

    df = prepare_audit_dataset(config.cache_path)

    raw_by_rule: dict[str, pd.DataFrame] = {}
    final_by_rule: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict] = []

    for rule_name in config.rule_names:
        raw_df = apply_audit_rule_v2(df, rule_name)
        final_df = dedupe_one_trade_per_cutoff(raw_df)

        raw_by_rule[rule_name] = raw_df
        final_by_rule[rule_name] = final_df
        summary_rows.append(summarize_rule(rule_name, raw_df, final_df))

    summary_df = pd.DataFrame(summary_rows)
    pairwise_df = build_pairwise_comparison(final_by_rule)
    violation_df = build_violation_check(final_by_rule)

    workload_df = pd.DataFrame(
        [
            {
                "cache_path": config.cache_path,
                "cache_rows": len(df),
                "rule_count": len(config.rule_names),
                "cache_design": "load_saved_cache_once_then_audit_rule_consistency_v2_with_derived_buckets",
            }
        ]
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "workload": str(output_dir / f"{config.tag}_workload.csv"),
        "summary": str(output_dir / f"{config.tag}_summary.csv"),
        "pairwise_comparison": str(output_dir / f"{config.tag}_pairwise_comparison.csv"),
        "rule_violation_check": str(output_dir / f"{config.tag}_rule_violation_check.csv"),
    }

    workload_df.to_csv(paths["workload"], index=False)
    summary_df.to_csv(paths["summary"], index=False)
    pairwise_df.to_csv(paths["pairwise_comparison"], index=False)
    violation_df.to_csv(paths["rule_violation_check"], index=False)

    return {
        "workload_df": workload_df,
        "summary_df": summary_df,
        "pairwise_df": pairwise_df,
        "violation_df": violation_df,
        "paths": paths,
    }
