"""CLI for Step Index cache-based scalping-distance feasibility diagnostics."""

from __future__ import annotations

import argparse

from mdp.backtesting.trend_channel_scalping_distance_feasibility import (
    ScalpingFeasibilityConfig,
    build_scalping_feasibility_report,
    save_scalping_feasibility_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze whether cached trend-channel trades fit smaller scalping SL/TP distances."
    )
    parser.add_argument("--cache-path", required=True, help="Path to the saved entry research cache CSV.")
    parser.add_argument("--rule-name", default="retest_requires_tp2_tp3", help="Candidate rule to evaluate.")
    parser.add_argument("--output-dir", default="outputs/trend_channel_scalping_profiles")
    parser.add_argument("--tag", default="step_scalping_feasibility")
    parser.add_argument("--small-sl-points", type=float, default=3.0)
    parser.add_argument("--small-tp-points", type=float, default=6.0)
    parser.add_argument("--medium-sl-points", type=float, default=4.0)
    parser.add_argument("--medium-tp-points", type=float, default=8.0)
    parser.add_argument("--wide-sl-points", type=float, default=5.0)
    parser.add_argument("--wide-tp-points", type=float, default=10.0)
    parser.add_argument("--preview-rows", type=int, default=30)
    args = parser.parse_args()

    config = ScalpingFeasibilityConfig(
        cache_path=args.cache_path,
        rule_name=args.rule_name,
        output_dir=args.output_dir,
        tag=args.tag,
        small_sl_points=args.small_sl_points,
        small_tp_points=args.small_tp_points,
        medium_sl_points=args.medium_sl_points,
        medium_tp_points=args.medium_tp_points,
        wide_sl_points=args.wide_sl_points,
        wide_tp_points=args.wide_tp_points,
    )

    report = build_scalping_feasibility_report(config)
    saved_paths = save_scalping_feasibility_outputs(report, config)

    print("Trend Channel Scalping Distance Feasibility Report")
    print("=" * 90)
    print(f"Cache path: {args.cache_path}")
    print(f"Rule name: {args.rule_name}")
    print("Cached design: loads saved cache once; no strategy pipeline rebuild.")
    print("Fixed SL/TP profiles are conservative proxies based on stored max favorable/adverse movement.")

    print("\nWorkload Summary")
    print("-" * 90)
    print(report["workload"])

    print("\nDistance Distribution")
    print("-" * 90)
    print(report["distance_distribution"])

    print("\nCurrent-Logic Scalping Profile Summary")
    print("-" * 90)
    print(report["scalping_profile_summary"])

    print("\nFixed SL/TP Proxy Summary")
    print("-" * 90)
    print(report["fixed_sl_tp_proxy_summary"])

    print("\nSelected Trade Preview")
    print("-" * 90)
    print(report["selected_trade_preview"].head(args.preview_rows))

    print("\nSaved CSV Outputs")
    print("-" * 90)
    for name, path in saved_paths.items():
        print(f"{name}: {path}")

    print("\nInterpretation Hint")
    print("-" * 90)
    print(
        "Use this as a fast feasibility test. If fixed 3SL/6TP looks promising, "
        "build an exact candle-path scalping replay next. If it fails here, do not force smaller SL/TP blindly."
    )


if __name__ == "__main__":
    main()
