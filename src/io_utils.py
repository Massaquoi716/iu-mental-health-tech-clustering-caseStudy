from pathlib import Path

import pandas as pd

from src.paths import DATA_PROCESSED


def save_csv(
    df: pd.DataFrame,
    filename: str,
    index: bool = False,
    **to_csv_kwargs,
) -> Path:
    """Save a DataFrame to data/processed/ as CSV.

    Path resolution is anchored to the project root (via src.paths),
    so it works regardless of the kernel's current working directory.

    Parameters
    ----------
    df : pd.DataFrame
        The frame to write.
    filename : str
        Name of the CSV file, e.g. "02_clean.csv".
    index : bool, default False
        Whether to write the DataFrame index. Off by default because
        notebook-facing CSVs almost never need it.
    **to_csv_kwargs
        Forwarded to pandas.DataFrame.to_csv (e.g. sep=";", encoding="utf-8").
    """
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = DATA_PROCESSED / filename
    df.to_csv(out, index=index, **to_csv_kwargs)
    return out
