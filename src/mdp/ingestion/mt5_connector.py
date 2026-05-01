from pathlib import Path
import os

from dotenv import load_dotenv
import MetaTrader5 as mt5
import pandas as pd

MT5_TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
}

def resolve_timeframe(timeframe: str) -> int:
    """
    Converts a human-readable timeframe string into a MetaTrader 5 timeframe constant.
    """
    normalized_timeframe = timeframe.upper()

    if normalized_timeframe not in MT5_TIMEFRAMES:
        supported_timeframes = ", ".join(MT5_TIMEFRAMES.keys())
        raise ValueError(
            f"Unsupported timeframe: {timeframe}. "
            f"Supported timeframes are: {supported_timeframes}"
        )

    return MT5_TIMEFRAMES[normalized_timeframe]

def ensure_symbol_visible(symbol: str) -> None:
    """
    Ensures that a symbol is available and visible in the MT5 Market Watch.
    """
    symbol_info = mt5.symbol_info(symbol)

    if symbol_info is None:
        error_code, error_message = mt5.last_error()
        raise RuntimeError(
            f"Symbol not found in MT5: {symbol}. "
            f"Error code: {error_code}. Message: {error_message}"
        )

    if not symbol_info.visible:
        selected = mt5.symbol_select(symbol, True)

        if not selected:
            error_code, error_message = mt5.last_error()
            raise RuntimeError(
                f"Could not make symbol visible in Market Watch: {symbol}. "
                f"Error code: {error_code}. Message: {error_message}"
            )
        
def fetch_rates(
    symbol: str,
    timeframe: str,
    count: int = 100,
    start_pos: int = 0,
) -> pd.DataFrame:
    """
    Fetches OHLCV candle data from MetaTrader 5 and returns a standardized DataFrame.

    The returned DataFrame is aligned with the market_candles database schema.
    """
    if count <= 0:
        raise ValueError("count must be greater than 0.")

    ensure_symbol_visible(symbol)

    mt5_timeframe = resolve_timeframe(timeframe)

    rates = mt5.copy_rates_from_pos(
        symbol,
        mt5_timeframe,
        start_pos,
        count,
    )

    if rates is None:
        error_code, error_message = mt5.last_error()
        raise RuntimeError(
            f"Could not fetch rates for {symbol} on {timeframe}. "
            f"Error code: {error_code}. Message: {error_message}"
        )

    rates_df = pd.DataFrame(rates)

    if rates_df.empty:
        return rates_df

    rates_df["timestamp_utc"] = (
        pd.to_datetime(rates_df["time"], unit="s", utc=True)
        .dt.tz_localize(None)
    )

    rates_df["symbol"] = symbol
    rates_df["timeframe"] = timeframe.upper()
    rates_df["source"] = "MT5"

    rates_df = rates_df.rename(
        columns={
            "open": "open_price",
            "high": "high_price",
            "low": "low_price",
            "close": "close_price",
        }
    )

    ordered_columns = [
        "symbol",
        "timeframe",
        "timestamp_utc",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "tick_volume",
        "spread",
        "real_volume",
        "source",
    ]

    return rates_df[ordered_columns]

def load_environment() -> None:
    """
    Loads environment variables from the .env file located at the project root.
    """
    project_root = Path(__file__).resolve().parents[3]
    env_path = project_root / ".env"
    load_dotenv(env_path)


def initialize_mt5() -> None:
    """
    Initializes the connection between Python and the local MetaTrader 5 desktop terminal.
    """
    load_environment()

    terminal_path = os.getenv("MT5_TERMINAL_PATH")

    if terminal_path:
        initialized = mt5.initialize(path=terminal_path)
    else:
        initialized = mt5.initialize()

    if not initialized:
        error_code, error_message = mt5.last_error()
        raise RuntimeError(
            f"MetaTrader 5 initialization failed. "
            f"Error code: {error_code}. Message: {error_message}"
        )

def shutdown_mt5() -> None:
    """
    Closes the connection between Python and MetaTrader 5.
    """
    mt5.shutdown()


def get_terminal_info() -> dict:
    """
    Returns basic information about the connected MetaTrader 5 terminal.
    """
    terminal_info = mt5.terminal_info()

    if terminal_info is None:
        error_code, error_message = mt5.last_error()
        raise RuntimeError(
            f"Could not retrieve MT5 terminal info. "
            f"Error code: {error_code}. Message: {error_message}"
        )

    return terminal_info._asdict()


def get_account_info() -> dict:
    """
    Returns basic information about the currently logged-in MT5 account.
    """
    account_info = mt5.account_info()

    if account_info is None:
        error_code, error_message = mt5.last_error()
        raise RuntimeError(
            f"Could not retrieve MT5 account info. "
            f"Error code: {error_code}. Message: {error_message}"
        )

    return account_info._asdict()


def get_visible_symbols() -> pd.DataFrame:
    """
    Returns all symbols currently visible in the MT5 Market Watch as a DataFrame.
    """
    symbols = mt5.symbols_get()

    if symbols is None:
        error_code, error_message = mt5.last_error()
        raise RuntimeError(
            f"Could not retrieve MT5 symbols. "
            f"Error code: {error_code}. Message: {error_message}"
        )

    symbol_rows = [
        {
            "name": symbol.name,
            "description": symbol.description,
            "path": symbol.path,
            "visible": symbol.visible,
        }
        for symbol in symbols
        if symbol.visible
    ]

    return pd.DataFrame(symbol_rows)