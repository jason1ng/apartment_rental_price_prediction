"""
Section 3.5 Prepared Dataset.

Runs Sections 3.2 (Cleaning) and 3.3 (Outlier Treatment) only.
Section 3.4 (Data Transformation) is intentionally applied AFTER the
train-test split so cityname's target encoding is learned from the training
data only, preserving the methodological integrity of the modelling pipeline.
"""
import pandas as pd

from src.data_cleaning import load_raw, clean
from src.outlier_treatment import treat_outliers

RAW_CSV_PATH = "data/apartments_for_rent_classified_100K/apartments_for_rent_classified_100K.csv"
OUTPUT_PATH = "data/apartments_prepared.csv"


def build_prepared_dataset(raw_csv_path: str = RAW_CSV_PATH, verbose: bool = True) -> pd.DataFrame:
    """Return the cleaned, outlier-treated dataset for Section 3.5."""
    df = load_raw(raw_csv_path)
    if verbose:
        print(f"Raw shape: {df.shape}")

    df = clean(df, verbose=verbose)
    df = treat_outliers(df, verbose=verbose)

    if verbose:
        print(f"\nPrepared dataset: {df.shape}")
        print(f"Missing values after cleaning/outlier treatment: {df.isna().sum().sum()}")

    return df

if __name__ == "__main__":
    prepared_df = build_prepared_dataset()
    prepared_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")
