import argparse

from mdp.database.candle_repository import fetch_market_candles
from mdp.quality.candle_quality import validate_candle_quality


def parse_arguments():
    """
    Parses command-line arguments for candle data quality checks.
    """
    parser = argparse.ArgumentParser(
        description="Run basic data quality checks on market candles stored in MySQL."
    )

    parser.add_argument(
        "--symbol",
        required=True,
        help='Market symbol, for example: "Step Index" or "Volatility 75 Index".',
    )

    parser.add_argument(
        "--timeframe",
        required=True,
        help="Timeframe, for example: M1, M5, M15, H1, H4, D1.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of recent candles to check.",
    )

    return parser.parse_args()


def print_quality_report(report: dict) -> None:
    """
    Prints a readable data quality report.
    """
    print("Candle Data Quality Report")
    print("=" * 35)

    print(f"Symbol: {report['symbol']}")
    print(f"Timeframe: {report['timeframe']}")
    print(f"Rows checked: {report['row_count']}")
    print(f"Is empty: {report['is_empty']}")
    print(f"Time sorted: {report['is_time_sorted']}")
    print(f"Duplicate timestamps: {report['duplicate_timestamps']}")
    print(f"Invalid OHLC rows: {report['invalid_ohlc_rows']}")
    print(f"Gap count: {report['gap_count']}")
    print(f"Largest gap: {report['largest_gap']}")

    print("\nMissing values by column:")
    for column, count in report["missing_values"].items():
        print(f"- {column}: {count}")


def main():
    args = parse_arguments()

    symbol = args.symbol
    timeframe = args.timeframe.upper()
    limit = args.limit

    candles_df = fetch_market_candles(
        symbol=symbol,
        timeframe=timeframe,
        limit=limit,
    )

    report = validate_candle_quality(
        candles_df=candles_df,
        symbol=symbol,
        timeframe=timeframe,
    )

    print_quality_report(report)


if __name__ == "__main__":
    main()