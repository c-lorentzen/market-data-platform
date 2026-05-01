from mdp.ingestion.mt5_connector import (
    get_account_info,
    get_terminal_info,
    get_visible_symbols,
    initialize_mt5,
    shutdown_mt5,
)


def main():
    initialize_mt5()

    try:
        terminal_info = get_terminal_info()
        account_info = get_account_info()
        symbols_df = get_visible_symbols()

        print("MetaTrader 5 connection successful.")
        print(f"Terminal connected: {terminal_info.get('connected')}")
        print(f"Account login: {account_info.get('login')}")
        print(f"Account server: {account_info.get('server')}")
        print(f"Visible symbols found: {len(symbols_df)}")

        print("\nFirst 20 visible symbols:")
        print(symbols_df.head(20))

        print("\nSymbols containing 'Volatility' or 'Step':")
        filtered_symbols = symbols_df[
            symbols_df["name"].str.contains("Volatility|Step", case=False, na=False)
            | symbols_df["description"].str.contains("Volatility|Step", case=False, na=False)
        ]

        print(filtered_symbols)

    finally:
        shutdown_mt5()
        print("\nMetaTrader 5 connection closed.")


if __name__ == "__main__":
    main()