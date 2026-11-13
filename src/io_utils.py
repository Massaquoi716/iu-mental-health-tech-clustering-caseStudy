from pathlib import Path
from typing import Any, Callable, Optional, Union

import pandas as pd

from src.paths import DATA_PROCESSED


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# EDA helpers
# ---------------------------------------------------------------------------

def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-column missingness summary.

    Parameters
    ----------
    df : pd.DataFrame
        The frame to inspect.

    Returns
    -------
    pd.DataFrame
        Indexed by column name, with two columns:
        - ``missing_count`` : number of null values per column
        - ``missing_pct``   : percentage of null values per column
    """
    return pd.DataFrame(
        {
            "missing_count": df.isnull().sum(),
            "missing_pct": df.isnull().mean() * 100,
        }
    )


# ---------------------------------------------------------------------------
# Population filtering
# ---------------------------------------------------------------------------

def filter_population(
    df: pd.DataFrame,
    column: Optional[str] = None,
    target_value: Any = None,
    *,
    mask: Optional[Union[pd.Series, Callable[[pd.DataFrame], pd.Series]]] = None,
    label: Optional[str] = None,
) -> pd.DataFrame:
    """Filter a DataFrame and print population triage metrics.

    Two calling styles are supported:

    1. Single-column equality filter::

           filter_population(df, "Are you self-employed?", 0)

    2. Arbitrary boolean mask (or a callable that returns one)::

           filter_population(
               df,
               mask=(df["A"] == 1) | (df["B"] == 1),
               label="tech-oriented employees",
           )

    Exactly one of ``column``/``target_value`` or ``mask`` must be provided.

    Parameters
    ----------
    df : pd.DataFrame
        The source DataFrame.
    column : str, optional
        Column name for the simple equality filter.
    target_value : Any, optional
        Value to keep when using the simple equality filter.
    mask : pd.Series or callable, optional
        Boolean Series aligned to ``df``, or a callable that takes ``df``
        and returns such a Series.
    label : str, optional
        Human-readable description used in the printed report.
        Defaults to a description derived from the call style.

    Returns
    -------
    pd.DataFrame
        A deep copy of the filtered subset.
    """
    has_column_filter = column is not None and target_value is not None
    has_mask_filter = mask is not None

    if has_column_filter and has_mask_filter:
        raise ValueError(
            "Provide either (column, target_value) or mask, not both."
        )
    if not has_column_filter and not has_mask_filter:
        raise ValueError(
            "Provide either (column, target_value) or mask."
        )

    if has_mask_filter:
        resolved_mask = mask(df) if callable(mask) else mask
        df_filtered = df.loc[resolved_mask].copy()
        if label is None:
            label = "boolean mask"
    else:
        df_filtered = df[df[column] == target_value].copy()
        if label is None:
            label = f"{column} == {target_value!r}"

    total_raw = len(df)
    total_kept = len(df_filtered)
    total_dropped = total_raw - total_kept

    print(f"Population Gate Applied [{label}]:")
    print(f"  • Total respondents: {total_raw:,}")
    print(f"  • Retained cohort:   {total_kept:,}")
    print(f"  • Excluded subset:   {total_dropped:,}")

    return df_filtered

# ---------------------------------------------------------------------------
# Display unique values
# ---------------------------------------------------------------------------

def unique_values_table(df, max_display=10):
    """
    Returns a summary DataFrame detailing each column's data type, 
    number of unique values, and a list of unique values.
    """
    summary = []
    for col in df.columns:
        n_unique = df[col].nunique()
        unique_vals = df[col].dropna().unique()
        
        # Truncate if there are too many unique values for clean display
        if n_unique > max_display:
            vals_preview = list(unique_vals[:max_display]) + [f"... ({n_unique} total)"]
        else:
            vals_preview = list(unique_vals)
            
        summary.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "n_unique": n_unique,
            "missing_count": df[col].isna().sum(),
            "unique_values": vals_preview
        })
        
    return pd.DataFrame(summary)

