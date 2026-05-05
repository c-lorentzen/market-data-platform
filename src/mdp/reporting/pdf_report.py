from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _format_value(value) -> str:
    """
    Formats values for readable PDF tables.
    """
    # Display missing values in a clean way.
    if pd.isna(value):
        return "N/A"

    # Format floats with four decimals for readability.
    if isinstance(value, float):
        return f"{value:.4f}"

    # Convert all other values to text.
    return str(value)


def _dict_to_table_data(data: dict, max_rows: int | None = None) -> list[list[str]]:
    """
    Converts a dictionary into two-column table data.
    """
    # Convert dictionary items into table rows.
    rows = [["Metric", "Value"]]

    # Optionally limit the number of rows shown.
    items = list(data.items())

    if max_rows is not None:
        items = items[:max_rows]

    # Format each key/value pair.
    for key, value in items:
        rows.append(
            [
                str(key),
                _format_value(value),
            ]
        )

    return rows


def _dataframe_to_table_data(
    df: pd.DataFrame,
    max_rows: int = 12,
    max_columns: int = 8,
) -> list[list[str]]:
    """
    Converts a DataFrame into table data for ReportLab.
    """
    # Return a simple message if the DataFrame is empty.
    if df.empty:
        return [["No data available"]]

    # Keep the table compact enough for a PDF page.
    compact_df = df.head(max_rows).iloc[:, :max_columns].copy()

    # Build the header row.
    rows = [list(compact_df.columns)]

    # Add formatted data rows.
    for _, row in compact_df.iterrows():
        rows.append([_format_value(value) for value in row.tolist()])

    return rows


def _build_table(table_data: list[list[str]]) -> Table:
    """
    Builds a styled ReportLab table.
    """
    # Create the table object.
    table = Table(table_data, repeatRows=1)

    # Apply a clean table style.
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )

    return table


def build_walk_forward_pdf_report(
    output_pdf_path: str | Path,
    report_title: str,
    config: dict,
    summary: dict,
    performance_summary: dict,
    windows_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    chart_paths: dict[str, Path],
) -> None:
    """
    Builds a client-friendly PDF report for a walk-forward backtest.
    """
    # Convert output path into a Path object.
    output_pdf_path = Path(output_pdf_path)

    # Create the output folder if it does not exist.
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    # Create the PDF document.
    document = SimpleDocTemplate(
        str(output_pdf_path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    # Load default report styles.
    styles = getSampleStyleSheet()

    # Store all PDF elements in this list.
    story = []

    # Add report title.
    story.append(Paragraph(report_title, styles["Title"]))
    story.append(Spacer(1, 12))

    # Add a short explanation for the client.
    story.append(
        Paragraph(
            "This report summarizes a walk-forward backtest. "
            "The strategy detects conditions using historical training windows "
            "and evaluates signals on future testing windows to reduce look-ahead bias.",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 14))

    # Add configuration table.
    story.append(Paragraph("Run Configuration", styles["Heading2"]))
    story.append(_build_table(_dict_to_table_data(config)))
    story.append(Spacer(1, 14))

    # Add walk-forward summary table.
    story.append(Paragraph("Walk-Forward Summary", styles["Heading2"]))
    story.append(_build_table(_dict_to_table_data(summary)))
    story.append(Spacer(1, 14))

    # Add performance summary table.
    story.append(Paragraph("Performance Summary", styles["Heading2"]))
    story.append(_build_table(_dict_to_table_data(performance_summary)))
    story.append(PageBreak())

    # Add equity curve chart.
    if chart_paths.get("equity_curve") is not None:
        story.append(Paragraph("Equity Curve", styles["Heading2"]))
        story.append(Image(str(chart_paths["equity_curve"]), width=500, height=250))
        story.append(Spacer(1, 14))

    # Add drawdown chart.
    if chart_paths.get("drawdown") is not None:
        story.append(Paragraph("Drawdown Curve", styles["Heading2"]))
        story.append(Image(str(chart_paths["drawdown"]), width=500, height=250))
        story.append(PageBreak())

    # Add trade distribution chart.
    if chart_paths.get("trade_distribution") is not None:
        story.append(Paragraph("Trade Distribution", styles["Heading2"]))
        story.append(Image(str(chart_paths["trade_distribution"]), width=500, height=250))
        story.append(Spacer(1, 14))

    # Add window activity chart.
    if chart_paths.get("window_activity") is not None:
        story.append(Paragraph("Walk-Forward Window Activity", styles["Heading2"]))
        story.append(Image(str(chart_paths["window_activity"]), width=500, height=250))
        story.append(PageBreak())

    # Add window diagnostics table.
    story.append(Paragraph("Window Diagnostics", styles["Heading2"]))
    story.append(
        _build_table(
            _dataframe_to_table_data(
                windows_df,
                max_rows=14,
                max_columns=10,
            )
        )
    )
    story.append(Spacer(1, 14))

    # Add trades table.
    story.append(Paragraph("Trades", styles["Heading2"]))
    story.append(
        _build_table(
            _dataframe_to_table_data(
                trades_df,
                max_rows=20,
                max_columns=9,
            )
        )
    )

    # Build the final PDF file.
    document.build(story)