from pathlib import Path
import os

from dotenv import load_dotenv
import MetaTrader5 as mt5
import pandas as pd


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