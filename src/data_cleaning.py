"""
Section 3.2 Data Cleaning — shared functions.
Import into any notebook with:
    from src.data_cleaning import load_and_clean
so every teammate's notebook produces an identical cleaned dataset.
"""
import pandas as pd
from src import config


def load_raw(csv_path: str) -> pd.DataFrame:
    """Load the raw apartment rent CSV (semicolon-delimited, latin-1 encoded)."""
    return pd.read_csv(csv_path, sep=";", encoding="latin-1", low_memory=False)


def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Apply all 3.2 cleaning steps and return the cleaned dataframe."""
    df = df.copy()
    n0 = len(df)

    # 3.2.1 — remove true duplicate listings (matched on id, before dropping it)
    df = df.drop_duplicates(subset=["id"], keep="first")
    if verbose:
        print(f"[3.2.1] Removed {n0 - len(df)} duplicate listings (matched on id)")

    # Drop columns excluded in 3.1
    df = df.drop(columns=config.COLS_TO_DROP)

    # 3.2.2 — remove non-standard / invalid listings
    n1 = len(df)
    df = df[df["category"] == config.VALID_CATEGORY]
    df = df[df["price_type"] == config.VALID_PRICE_TYPE]
    df = df.drop(columns=config.FILTER_ONLY_COLS)
    if verbose:
        print(f"[3.2.2] Removed {n1 - len(df)} non-standard-apartment / non-monthly rows")

    # 3.2.3 — drop rows missing critical fields
    n2 = len(df)
    df = df.dropna(subset=config.CRITICAL_COLS_DROP_IF_MISSING)
    if verbose:
        print(f"[3.2.3] Removed {n2 - len(df)} rows with missing critical values")

    # 3.2.4 — recode meaningful missingness as its own category
    for col, fill_value in config.RECODE_MISSING.items():
        df[col] = df[col].fillna(fill_value)

    # 3.2.5 — defensive type/whitespace cleanup
    for col in ["cityname", "state", "fee", "has_photo"]:
        df[col] = df[col].astype(str).str.strip()

    if verbose:
        print(f"\nFinal shape: {df.shape}  (kept {len(df)/n0:.1%} of raw rows)")
        print("Missing values remaining:\n", df.isnull().sum())

    return df


def load_and_clean(csv_path: str, verbose: bool = True) -> pd.DataFrame:
    """Convenience wrapper: load raw CSV and return cleaned dataframe in one call."""
    return clean(load_raw(csv_path), verbose=verbose)
