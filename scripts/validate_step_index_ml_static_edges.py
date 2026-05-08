"""CLI for validating static pre-entry edges from Step Index ML-ready datasets."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from mdp.backtesting.trend_channel_ml_static_edge_validation import (
    StaticEdgeValidationConfig,
    save_static_edge_outputs,
    validate_static_edges,
)


def _parse_rule_names(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return StaticEdgeValidationConfig.rule_names
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml-dataset-path", required=True)
    parser.add_argument("--fold-count", type=int, default=7)
    parser.add_argument("--rule-names", default=None)
    parser.add_argument("--tag", default="step_ml_static_edges")
    args = parser.parse_args()

    start = time.perf_counter()
    config = StaticEdgeValidationConfig(
        ml_dataset_path=args.ml_dataset_path,
        rule_names=_parse_rule_names(args.rule_names),
        fold_count=args.fold_count,
        tag=args.tag,
    )
    outputs = validate_static_edges(config)
    saved = save_static_edge_outputs(config, outputs)
    elapsed = time.perf_counter() - start

    print("Step Index ML Static Edge Validation Report")
    print("=" * 90)
    print(f"ML dataset path: {args.ml_dataset_path}")
    print(f"Fold count: {args.fold_count}")
    print(f"Rules: {', '.join(config.rule_names)}")
    print(f"Elapsed seconds: {elapsed:.3f}")
    print("Design: loads exported ML dataset once; validates static pre-entry edges only.")

    print("\nStatic Edge Ranking")
    print("-" * 90)
    print(outputs["summary"].to_string(index=False))

    print("\nFold Detail Preview")
    print("-" * 90)
    fold_df = outputs["fold_detail"]
    if fold_df.empty:
        print("No fold detail available.")
    else:
        print(fold_df.head(50).to_string(index=False))

    print("\nSelected Trade Preview")
    print("-" * 90)
    print(outputs["selected_preview"].head(50).to_string(index=False))

    print("\nSaved CSV Outputs")
    print("-" * 90)
    for name, path in saved.items():
        print(f"{name}: {path}")

    print("\nInterpretation Hint")
    print("-" * 90)
    print(
        "Use this to decide whether a simple static pre-entry edge is stronger than the "
        "online filter. Do not lock time-of-day filters from small samples; validate on "
        "larger data and higher-timeframe context first."
    )


if __name__ == "__main__":
    main()
