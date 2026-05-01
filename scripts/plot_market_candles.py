import argparse
from pathlib import Path

from mdp.database.candle_repository import fetch_market_candles
from mdp.visualization.price_charts import plot_candlestick_chart


def parse_arguments():
    """
    Parses command-line arguments for candlestick chart generation.
    """
    parser = argparse.ArgumentParser(
        description="Create a candlestick chart from market candles stored in MySQL."
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
        help="Number of candles to fetch from MySQL.",
    )

    return parser.parse_args()


def build_output_path(symbol: str, timeframe: str) -> Path:
    """
    Builds a clean output path for the generated candlestick chart image.
    """
    clean_symbol = symbol.lower().replace(" ", "_")
    clean_timeframe = timeframe.lower()

    return Path("outputs") / "charts" / f"{clean_symbol}_{clean_timeframe}_candles.png"


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

    output_path = build_output_path(symbol, timeframe)

    plot_candlestick_chart(
        candles_df=candles_df,
        symbol=symbol,
        timeframe=timeframe,
        output_path=output_path,
    )

    print("Candlestick chart created successfully.")
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Candles plotted: {len(candles_df)}")
    print(f"Output file: {output_path}")


if __name__ == "__main__":
    main()