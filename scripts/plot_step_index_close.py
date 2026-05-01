from mdp.database.candle_repository import fetch_market_candles
from mdp.visualization.price_charts import plot_close_price

def main():
    symbol = "Step Index"
    timeframe = "H1"
    limit = 100

    candles_df = fetch_market_candles(
        symbol=symbol,
        timeframe=timeframe,
        limit=limit,
    )

    output_path = "outputs/charts/step_index_h1_close.png"

    plot_close_price(
        candles_df=candles_df,
        symbol=symbol,
        timeframe=timeframe,
        output_path=output_path,
    )

    print("Close price chart created successfully.")
    print(f"Output file: {output_path}")


if __name__ == "__main__":
    main()