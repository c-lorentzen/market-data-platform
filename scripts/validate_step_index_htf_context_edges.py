"""CLI for validating higher-timeframe context filters on enriched Step Index ML datasets."""
from __future__ import annotations

import argparse
import time

from mdp.backtesting.trend_channel_htf_context_validation import (
    HTFContextValidationConfig,
    save_htf_context_validation_outputs,
    validate_htf_context_rules,
)


def _parse_rule_names(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return HTFContextValidationConfig.rule_names
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enriched-dataset-path", required=True)
    parser.add_argument("--fold-count", type=int, default=10)
    parser.add_argument("--rule-names", default=None)
    parser.add_argument("--tag", default="step_candidate_htf_context_validation")
    args = parser.parse_args()

    start = time.perf_counter()
    config = HTFContextValidationConfig(
        enriched_dataset_path=args.enriched_dataset_path,
        rule_names=_parse_rule_names(args.rule_names),
        fold_count=args.fold_count,
        tag=args.tag,
    )
    outputs = validate_htf_context_rules(config)
    saved = save_htf_context_validation_outputs(config, outputs)
    elapsed = time.perf_counter() - start

    print("Step Index Higher-Timeframe Context Edge Validation Report")
    print("=" * 90)
    print(f"Enriched dataset path: {args.enriched_dataset_path}")
    print(f"Fold count: {args.fold_count}")
    print(f"Rules: {', '.join(config.rule_names)}")
    print(f"Elapsed seconds: {elapsed:.3f}")
    print("Design: validates only pre-entry HTF context columns from the enriched dataset.")

    print("\nHTF Context Rule Ranking")
    print("-" * 90)
    print(outputs["summary"].to_string(index=False))

    print("\nFold Detail Preview")
    print("-" * 90)
    if outputs["fold_detail"].empty:
        print("No fold detail available.")
    else:
        print(outputs["fold_detail"].head(60).to_string(index=False))

    print("\nContext Edge Preview")
    print("-" * 90)
    if outputs["context_edges"].empty:
        print("No context edge data available.")
    else:
        print(outputs["context_edges"].head(40).to_string(index=False))

    print("\nSelected Trade Preview")
    print("-" * 90)
    if outputs["selected_preview"].empty:
        print("No selected trades available.")
    else:
        print(outputs["selected_preview"].head(60).to_string(index=False))

    print("\nSaved CSV Outputs")
    print("-" * 90)
    for name, path in saved.items():
        print(f"{name}: {path}")

    print("\nInterpretation Hint")
    print("-" * 90)
    print(
        "Use this to see whether D1/H4/M15/M5 context improves Candidate v1.1. "
        "If a rule improves average R and fold stability without killing trades, validate on more data before ML training."
    )


if __name__ == "__main__":
    main()
