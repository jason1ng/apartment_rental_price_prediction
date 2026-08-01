"""
Section 3.5 Final Prepared Dataset.

Runs the full 3.2 -> 3.4 -> 3.3 pipeline in order and writes the result
to data/apartments_prepared.csv, ready for the 80:20 train-test split
in Modelling (Section 4.0).

Order matters: outlier treatment (3.4) is applied to price/square_feet
BEFORE transformation (3.3) engineers amenity_count/city_frequency,
so those engineered features are computed from the same capped values
the models will actually train on.
"""
import pandas as pd

from src.data_cleaning import load_raw, clean
from src.outlier_treatment import treat_outliers
from src.data_transformation import transform, get_engineered_feature_lists

RAW_CSV_PATH = "data/apartments_for_rent_classified_100K/apartments_for_rent_classified_100K.csv"
OUTPUT_PATH = "data/apartments_prepared.csv"


def build_final_dataset(raw_csv_path: str = RAW_CSV_PATH, verbose: bool = True) -> pd.DataFrame:
    df = load_raw(raw_csv_path)
    if verbose:
        print(f"Raw shape: {df.shape}")

    df = clean(df, verbose=verbose)
    df = treat_outliers(df, verbose=verbose)
    df = transform(df)

    numeric, categorical = get_engineered_feature_lists(df)
    final_cols = numeric + categorical + ["price"]
    df = df[final_cols]

    if verbose:
        print(f"\nFinal prepared dataset: {df.shape}")
        print(f"Features: {len(numeric) + len(categorical)} "
              f"({len(numeric)} numeric, {len(categorical)} categorical)")

    return df


if __name__ == "__main__":
    final_df = build_final_dataset()
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")
