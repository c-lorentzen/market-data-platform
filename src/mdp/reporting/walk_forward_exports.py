import json
from pathlib import Path

import pandas as pd


def save_walk_forward_outputs(
    output_dir: str | Path,
    windows_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    summary: dict,
    config: dict,
) -> None:
    """
    Saves walk-forward backtest outputs into CSV and JSON files.
    """
    # Convert the output path into a Path object.
    output_dir = Path(output_dir)

    # Create the report folder if it does not exist.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the window diagnostics so we can review each train/test split later.
    windows_df.to_csv(
        output_dir / "windows.csv",
        index=False,
    )

    # Save all generated signals, even if the strategy did not take every signal.
    signals_df.to_csv(
        output_dir / "signals.csv",
        index=False,
    )

    # Save all simulated trades with entry, exit, outcome, and R multiple.
    trades_df.to_csv(
        output_dir / "trades.csv",
        index=False,
    )

    # Save the summary metrics as a one-row CSV for easy spreadsheet review.
    pd.DataFrame([summary]).to_csv(
        output_dir / "summary.csv",
        index=False,
    )

    # Save the run configuration so the result can be reproduced later.
    with open(output_dir / "config.json", "w", encoding="utf-8") as config_file:
        json.dump(
            config,
            config_file,
            indent=4,
            default=str,
        )