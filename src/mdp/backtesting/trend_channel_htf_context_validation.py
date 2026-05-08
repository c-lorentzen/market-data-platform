"""Static validation for higher-timeframe context filters on enriched ML datasets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class HTFContextValidationConfig:
    enriched_dataset_path: str
    rule_names: tuple[str, ...] = (
        "candidate_v1_base",
        "require_baseline_current_close",
        "baseline_no_d1_h4_opposition",
        "baseline_no_h4_opposition",
        "baseline_d1_h4_m15_m5_not_against",
        "baseline_h4_or_m15_aligned",
        "premium_baseline_no_htf_opposition",
    )
    fold_count: int = 10
    min_total_trades: int = 30
    min_trades_per_fold: int = 3
    tag: str = "step_candidate_htf_context_validation"
    output_dir: str = "outputs/trend_channel_htf_context_validation"


def _normalize(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


def load_enriched_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "analysis_cutoff" not in df.columns:
        raise ValueError("Enriched dataset must include analysis_cutoff.")
    if "label_managed_r" not in df.columns:
        raise ValueError("Enriched dataset must include label_managed_r.")

    df = df.copy()
    df["analysis_cutoff"] = pd.to_datetime(df["analysis_cutoff"], errors="coerce")
    df = df.dropna(subset=["analysis_cutoff"]).sort_values("analysis_cutoff").reset_index(drop=True)
    if "trade_date" not in df.columns:
        df["trade_date"] = df["analysis_cutoff"].dt.date.astype(str)
    return df


def _bool_col(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    raw = df[column]
    if raw.dtype == bool:
        return raw.fillna(False)
    return raw.fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])


def _not_against(df: pd.DataFrame, timeframe: str) -> pd.Series:
    return ~_bool_col(df, f"htf_{timeframe.lower()}_bias_against_plan")


def _aligned(df: pd.DataFrame, timeframe: str) -> pd.Series:
    return _bool_col(df, f"htf_{timeframe.lower()}_bias_aligned_with_plan")


def _available(df: pd.DataFrame, timeframe: str) -> pd.Series:
    return _bool_col(df, f"htf_{timeframe.lower()}_context_available")


def _rule_mask(df: pd.DataFrame, rule_name: str) -> pd.Series:
    entry_variant = _normalize(df.get("entry_variant_name", pd.Series(index=df.index, dtype=str)))
    base = pd.Series(True, index=df.index)
    baseline_current_close = entry_variant.eq("baseline_current_close")

    # Require HTF rows to actually be available before using HTF filters.
    d1_ok = _available(df, "D1") & _not_against(df, "D1")
    h4_ok = _available(df, "H4") & _not_against(df, "H4")
    m15_ok = _available(df, "M15") & _not_against(df, "M15")
    m5_ok = _available(df, "M5") & _not_against(df, "M5")

    if rule_name == "candidate_v1_base":
        return base
    if rule_name == "require_baseline_current_close":
        return baseline_current_close
    if rule_name == "baseline_no_d1_h4_opposition":
        return baseline_current_close & d1_ok & h4_ok
    if rule_name == "baseline_no_h4_opposition":
        return baseline_current_close & h4_ok
    if rule_name == "baseline_d1_h4_m15_m5_not_against":
        return baseline_current_close & d1_ok & h4_ok & m15_ok & m5_ok
    if rule_name == "baseline_h4_or_m15_aligned":
        return baseline_current_close & (_aligned(df, "H4") | _aligned(df, "M15"))
    if rule_name == "premium_baseline_no_htf_opposition":
        return baseline_current_close & d1_ok & h4_ok & m15_ok & m5_ok & (_aligned(df, "H4") | _aligned(df, "M15") | _aligned(df, "M5"))
    if rule_name == "diagnostic_h4_aligned_only":
        return _aligned(df, "H4")
    if rule_name == "diagnostic_m15_m5_aligned":
        return _aligned(df, "M15") & _aligned(df, "M5")

    raise ValueError(f"Unknown HTF context rule: {rule_name}")


def _summarize(df: pd.DataFrame, rule_name: str) -> dict:
    if df.empty:
        return {
            "rule_name": rule_name,
            "trades": 0,
            "unique_days": 0,
            "avg_trades_per_day": 0.0,
            "positive_exit_rate": 0.0,
            "full_loss_rate": 0.0,
            "average_managed_r": 0.0,
            "median_managed_r": 0.0,
            "total_managed_r": 0.0,
        }

    r = pd.to_numeric(df["label_managed_r"], errors="coerce")
    pos = pd.to_numeric(df.get("label_positive_r", (r > 0).astype(int)), errors="coerce")
    full_loss = pd.to_numeric(df.get("label_full_or_near_full_loss", (r <= -0.95).astype(int)), errors="coerce")
    unique_days = df["trade_date"].nunique()
    return {
        "rule_name": rule_name,
        "trades": int(len(df)),
        "unique_days": int(unique_days),
        "avg_trades_per_day": float(len(df) / unique_days) if unique_days else 0.0,
        "positive_exit_rate": float(pos.mean()) if len(df) else 0.0,
        "full_loss_rate": float(full_loss.mean()) if len(df) else 0.0,
        "average_managed_r": float(r.mean()) if len(df) else 0.0,
        "median_managed_r": float(r.median()) if len(df) else 0.0,
        "total_managed_r": float(r.sum()) if len(df) else 0.0,
    }


def _fold_summary(df: pd.DataFrame, rule_name: str, fold_count: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    ordered = df.sort_values("analysis_cutoff").reset_index(drop=True).copy()
    fold_count = max(1, min(fold_count, len(ordered)))
    ordered["fold_id"] = (ordered.index * fold_count // len(ordered)) + 1

    rows = []
    for fold_id, fold_df in ordered.groupby("fold_id", sort=True):
        summary = _summarize(fold_df, rule_name)
        summary.update(
            {
                "fold_id": int(fold_id),
                "fold_start_cutoff": fold_df["analysis_cutoff"].min(),
                "fold_end_cutoff": fold_df["analysis_cutoff"].max(),
            }
        )
        rows.append(summary)
    return pd.DataFrame(rows)


def _context_edge_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in [
        "htf_d1_bias",
        "htf_h4_bias",
        "htf_h1_bias",
        "htf_m15_bias",
        "htf_m5_bias",
        "entry_variant_name",
        "target_plan_state",
        "scenario_state",
        "event_state",
    ]:
        if col not in df.columns:
            continue
        for value, group in df.groupby(col, dropna=False):
            if len(group) < 3:
                continue
            row = _summarize(group, f"{col}={value}")
            row["feature_name"] = col
            row["feature_value"] = value
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("average_managed_r", ascending=False).reset_index(drop=True)


def validate_htf_context_rules(config: HTFContextValidationConfig) -> dict[str, pd.DataFrame]:
    dataset = load_enriched_dataset(config.enriched_dataset_path)

    summary_rows = []
    fold_frames = []
    preview_frames = []

    for rule_name in config.rule_names:
        selected = dataset[_rule_mask(dataset, rule_name)].copy()
        summary = _summarize(selected, rule_name)
        folds = _fold_summary(selected, rule_name, config.fold_count)

        if folds.empty:
            min_fold_average_r = 0.0
            positive_fold_rate = 0.0
            folds_with_min_trades_rate = 0.0
        else:
            min_fold_average_r = float(folds["average_managed_r"].min())
            positive_fold_rate = float((folds["average_managed_r"] > 0).mean())
            folds_with_min_trades_rate = float((folds["trades"] >= config.min_trades_per_fold).mean())

        if summary["trades"] < config.min_total_trades:
            validation_state = "insufficient_total_trades_do_not_lock"
        elif summary["average_managed_r"] <= 0 or summary["median_managed_r"] <= 0:
            validation_state = "not_profitable_do_not_lock"
        elif positive_fold_rate < 0.75 or min_fold_average_r < -0.25:
            validation_state = "positive_but_fold_instability_validate_harder"
        else:
            validation_state = "promising_htf_context_validate_harder"

        summary.update(
            {
                "min_fold_average_r": min_fold_average_r,
                "positive_average_r_fold_rate": positive_fold_rate,
                "folds_with_min_trades_rate": folds_with_min_trades_rate,
                "validation_state": validation_state,
            }
        )
        summary_rows.append(summary)

        if not folds.empty:
            fold_frames.append(folds)

        preview_cols = [
            c
            for c in [
                "analysis_cutoff",
                "trade_date",
                "plan_direction",
                "entry_variant_name",
                "target_plan_state",
                "htf_d1_bias",
                "htf_h4_bias",
                "htf_h1_bias",
                "htf_m15_bias",
                "htf_m5_bias",
                "htf_d1_bias_against_plan",
                "htf_h4_bias_against_plan",
                "htf_m15_bias_against_plan",
                "htf_m5_bias_against_plan",
                "label_managed_r",
            ]
            if c in selected.columns
        ]
        preview = selected[preview_cols].copy().head(40)
        preview.insert(0, "rule_name", rule_name)
        preview_frames.append(preview)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["validation_state", "average_managed_r"], ascending=[True, False]
    )

    return {
        "summary": summary_df,
        "fold_detail": pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame(),
        "context_edges": _context_edge_summary(dataset),
        "selected_preview": pd.concat(preview_frames, ignore_index=True) if preview_frames else pd.DataFrame(),
    }


def save_htf_context_validation_outputs(
    config: HTFContextValidationConfig,
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
