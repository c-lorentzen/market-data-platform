import argparse
import json
from pathlib import Path

import pandas as pd

from mdp.reporting.pdf_report import build_walk_forward_pdf_report
from mdp.reporting.performance_charts import (
    save_drawdown_chart,
    save_equity_curve_chart,
    save_trade_distribution_chart,
    save_window_activity_chart,
)


def parse_arguments():
    """
    Parses command-line arguments for PDF report generation.
    """
    parser = argparse.ArgumentParser(
        description="Generate a client-friendly PDF report from walk-forward output files."
    )

    parser.add_argument(
        "--report-dir",
        required=True,
        help="Folder containing walk-forward CSV and JSON output files.",
    )

    parser.add_argument(
        "--output-pdf",
        default=None,
        help="Optional custom PDF output path.",
    )

    return parser.parse_args()


def load_json_file(file_path: Path) -> dict:
    """
    Loads a JSON file into a dictionary.
    """
    # Open the JSON file using UTF-8 encoding.
    with open(file_path, "r", encoding="utf-8") as json_file:
        return json.load(json_file)


def main():
    args = parse_arguments()

    # Convert the report folder into a Path object.
    report_dir = Path(args.report_dir)

    # Check that the report folder exists before reading files.
    if not report_dir.exists():
        raise FileNotFoundError(f"Report directory not found: {report_dir}")

    # Load required CSV and JSON files from the walk-forward report folder.
    windows_df = pd.read_csv(report_dir / "windows.csv")
    trades_df = pd.read_csv(report_dir / "trades.csv")
    summary_df = pd.read_csv(report_dir / "summary.csv")
    performance_summary_df = pd.read_csv(report_dir / "performance_summary.csv")
    equity_curve_df = pd.read_csv(report_dir / "equity_curve.csv")
    trade_distribution_df = pd.read_csv(report_dir / "trade_distribution.csv")
    config = load_json_file(report_dir / "config.json")

    # Convert one-row CSV files into dictionaries.
    summary = summary_df.iloc[0].to_dict()
    performance_summary = performance_summary_df.iloc[0].to_dict()

    # Create a charts folder inside the report folder.
    charts_dir = report_dir / "charts"

    # Build chart file paths.
    equity_curve_path = charts_dir / "equity_curve.png"
    drawdown_path = charts_dir / "drawdown.png"
    trade_distribution_path = charts_dir / "trade_distribution.png"
    window_activity_path = charts_dir / "window_activity.png"

    # Generate visual charts from saved report data.
    save_equity_curve_chart(
        equity_curve_df=equity_curve_df,
        output_path=equity_curve_path,
    )

    save_drawdown_chart(
        equity_curve_df=equity_curve_df,
        output_path=drawdown_path,
    )

    save_trade_distribution_chart(
        trade_distribution_df=trade_distribution_df,
        output_path=trade_distribution_path,
    )

    save_window_activity_chart(
        windows_df=windows_df,
        output_path=window_activity_path,
    )

    # Use a default PDF path if the user did not provide one.
    if args.output_pdf is None:
        output_pdf_path = report_dir / "walk_forward_report.pdf"

    # Use the custom PDF path provided by the user.
    else:
        output_pdf_path = Path(args.output_pdf)

    # Build a readable report title.
    report_title = (
        f"Walk-Forward Backtest Report - "
        f"{config.get('symbol', 'Unknown Symbol')} "
        f"{config.get('timeframe', 'Unknown Timeframe')}"
    )

    # Assemble the PDF report.
    build_walk_forward_pdf_report(
        output_pdf_path=output_pdf_path,
        report_title=report_title,
        config=config,
        summary=summary,
        performance_summary=performance_summary,
        windows_df=windows_df,
        trades_df=trades_df,
        chart_paths={
            "equity_curve": equity_curve_path,
            "drawdown": drawdown_path,
            "trade_distribution": trade_distribution_path,
            "window_activity": window_activity_path,
        },
    )

    print("PDF report generated successfully.")
    print(f"Report directory: {report_dir}")
    print(f"PDF file: {output_pdf_path}")


if __name__ == "__main__":
    main()