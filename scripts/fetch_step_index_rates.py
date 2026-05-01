from mdp.ingestion.mt5_connector import (
    fetch_rates,
    initialize_mt5,
    shutdown_mt5,
)


def main():
    symbol = "Step Index"
    timeframe = "H1"
    count = 100

    initialize_mt5()

    try:
        rates_df = fetch_rates(
            symbol=symbol,
            timeframe=timeframe,
            count=count,
        )

        print(f"Fetched {len(rates_df)} candles from MT5.")
        print(f"Symbol: {symbol}")
        print(f"Timeframe: {timeframe}")

        print("\nFirst 5 candles:")
        print(rates_df.head())

        print("\nLast 5 candles:")
        print(rates_df.tail())

        print("\nDataFrame info:")
        print(rates_df.info())

    finally:
        shutdown_mt5()
        print("\nMetaTrader 5 connection closed.")


if __name__ == "__main__":
    main()