"""CLI for exporting Step Index ML datasets enriched with higher-timeframe context."""
from __future__ import annotations

import argparse
import time

from mdp.backtesting.trend_channel_higher_timeframe_context import (
    HigherTimeframeContextConfig,
    export_higher_timeframe_context,
    save_higher_timeframe_context_outputs,
)


def _parse_timeframes(raw: str) -> tuple[str, ...]:
    return tuple(part.strip().upper() for part in raw.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml-dataset-path", required=True)
    parser.add_argument("--symbol", default="Step Index")
    parser.add_argument("--context-timeframes", default="D1,H4,H1,M15,M5")
    parser.add_argument("--context-lookback-candles", type=int, default=60)
    parser.add_argument("--tag", default="step_candidate_htf_context")
    args = parser.parse_args()

    start = time.perf_counter()
    config = HigherTimeframeContextConfig(
        ml_dataset_path=args.ml_dataset_path,
        symbol=args.symbol,
        context_timeframes=_parse_timeframes(args.context_timeframes),
        context_lookback_candles=args.context_lookback_candles,
        tag=args.tag,
    )
    outputs = export_higher_timeframe_context(config)
    saved = save_higher_timeframe_context_outputs(config, outputs)
    elapsed = time.perf_counter() - start

    print("Step Index Higher-Timeframe Context Export Report")
    print("=" * 90)
    print(f"ML dataset path: {args.ml_dataset_path}")
    print(f"Symbol: {args.symbol}")
    print(f"Context timeframes: {', '.join(config.context_timeframes)}")
    print(f"Elapsed seconds: {elapsed:.3f}")
    print("Design: uses only closed higher-timeframe candles available before each analysis cutoff.")

    print("\nSummary")
    print("-" * 90)
    print(outputs["summary"].to_string(index=False))

    print("\nContext Coverage")
    print("-" * 90)
    print(outputs["coverage"].to_string(index=False))

    print("\nEnriched Dataset Preview")
    print("-" * 90)
    preview_cols = [
        c
        for c in [
            "analysis_cutoff",
            "trade_date",
            "plan_direction",
            "entry_variant_name",
            "label_managed_r",
            "htf_d1_bias",
            "htf_h4_bias",
            "htf_h1_bias",
            "htf_m15_bias",
            "htf_m5_bias",
            "htf_d1_bias_against_plan",
            "htf_h4_bias_against_plan",
            "htf_m15_bias_against_plan",
            "htf_m5_bias_against_plan",
        ]
        if c in outputs["enriched_dataset"].columns
    ]
    print(outputs["enriched_dataset"][preview_cols].head(30).to_string(index=False))

    print("\nSaved CSV Outputs")
    print("-" * 90)
    for name, path in saved.items():
        print(f"{name}: {path}")

    print("\nInterpretation Hint")
    print("-" * 90)
    print(
        "If coverage is close to 100%, validate whether D1/H4/M15/M5 context explains the weak folds. "
        "Do not train ML until the context-enriched dataset is validated chronologically."
    )


if __name__ == "__main__":
    main()
