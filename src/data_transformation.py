"""
Section 3.3 Data Transformation.

Addresses the high-cardinality risk flagged in Section 1.6: cityname
(~2,975 unique) and amenities (~9,786 unique combinations) cannot be
one-hot encoded naively without exploding dimensionality and hurting
KNN's distance calculations. Each variable gets an encoding strategy
sized to its own cardinality and to how much city-specific signal is
worth preserving, rather than one blanket rule applied everywhere.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TOP_AMENITIES = [
    "Parking", "Pool", "Gym", "Patio/Deck", "Washer Dryer", "Storage",
    "Clubhouse", "Dishwasher", "AC", "Fireplace", "Refrigerator",
]

# Cities with at least this many listings keep their own category;
# everything below is bucketed as "Other". 300 was chosen empirically:
# it keeps the one-hot expansion to ~60 categories (comparable to
# state's 51) while covering ~40% of listings under their real city
# identity — the trade-off between preserving city-specific price
# signal and controlling dimensionality for KNN.
CITY_FREQUENCY_THRESHOLD = 300

LOW_CARD_CATEGORICAL = ["state", "pets_allowed", "fee", "has_photo", "city_category"]
NUMERIC_FEATURES_BASE = ["bathrooms", "bedrooms", "square_feet", "latitude", "longitude"]


def engineer_amenity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replace raw comma-separated amenities with amenity_count + top-amenity binary flags."""
    df = df.copy()
    amenities_str = df["amenities"].fillna("None").astype(str)

    df["amenity_count"] = amenities_str.apply(
        lambda s: 0 if s == "None" else len(s.split(","))
    )
    for amenity in TOP_AMENITIES:
        col_name = "amenity_has_" + amenity.lower().replace(" ", "_").replace("/", "_")
        df[col_name] = amenities_str.str.contains(amenity, regex=False).astype(int)

    return df.drop(columns=["amenities"])


def engineer_room_density(df: pd.DataFrame) -> pd.DataFrame:
    """square_feet per room (bedrooms + bathrooms) — apartment 'density', often more
    informative than raw square footage alone (e.g. distinguishing a studio from
    a large family unit of similar total size).
    bedrooms can be 0 (studio) but bathrooms is always >= 1 after cleaning, so
    the denominator is never zero.
    """
    df = df.copy()
    df["squarefeet_per_room"] = df["square_feet"] / (df["bedrooms"] + df["bathrooms"])
    return df


def encode_city(df: pd.DataFrame, train_df: pd.DataFrame = None) -> pd.DataFrame:
    """Hybrid city encoding:
    - city_category: the city name itself for cities with >= CITY_FREQUENCY_THRESHOLD
      listings, else "Other" — one-hot encoded downstream, preserving specific-city
      price signal for major markets.
    - city_frequency: listing count for the city, retained as a continuous feature
      for ALL rows (including "Other"-bucketed ones), so market-size information
      isn't fully lost for smaller cities.
    Pass train_df when transforming validation/test data so both the kept-city
    list and frequency counts are learned from training data only (no leakage).
    """
    df = df.copy()
    reference = train_df if train_df is not None else df
    freq_map = reference["cityname"].value_counts()
    kept_cities = freq_map[freq_map >= CITY_FREQUENCY_THRESHOLD].index

    df["city_frequency"] = df["cityname"].map(freq_map).fillna(0).astype(int)
    df["city_category"] = df["cityname"].where(df["cityname"].isin(kept_cities), "Other")

    return df.drop(columns=["cityname"])


def transform(df: pd.DataFrame, train_df: pd.DataFrame = None) -> pd.DataFrame:
    """Apply full 3.3 transformation. Low-cardinality categoricals are left
    as-is here and one-hot encoded inside the modelling pipeline, so encoders
    are fit on training data only (leakage prevention).
    """
    df = engineer_amenity_features(df)
    df = engineer_room_density(df)
    df = encode_city(df, train_df=train_df)
    return df


def get_engineered_feature_lists(df: pd.DataFrame) -> tuple:
    """Return (numeric_features, categorical_features) after transform() has been applied."""
    amenity_flag_cols = [c for c in df.columns if c.startswith("amenity_has_")]
    numeric = (NUMERIC_FEATURES_BASE + ["amenity_count", "city_frequency", "squarefeet_per_room"]
               + amenity_flag_cols)
    categorical = LOW_CARD_CATEGORICAL
    return numeric, categorical


def build_preprocessor(numeric_features: list, categorical_features: list,
                        scale_numeric: bool = True) -> ColumnTransformer:
    """scale_numeric=True for Linear Regression / KNN (scale-sensitive).
    scale_numeric=False for Random Forest / Gradient Boosting (scale-invariant).
    """
    numeric_steps = [("scaler", StandardScaler())] if scale_numeric else []
    numeric_pipeline = Pipeline(numeric_steps) if numeric_steps else "passthrough"

    return ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ])
