"""
Section 3.3 Data Transformation.

Addresses the high-cardinality risk flagged in Section 1.6: cityname
(~2,975 unique) and amenities (~9,786 unique combinations) cannot be
one-hot encoded naively without exploding dimensionality and hurting
KNN's distance calculations.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TOP_AMENITIES = [
    "Parking", "Pool", "Gym", "Patio/Deck", "Washer Dryer", "Storage",
    "Clubhouse", "Dishwasher", "AC", "Fireplace", "Refrigerator",
]

# Cities with enough observations keep their own category;
# less frequent cities are grouped into "Other" to control
# one-hot encoding dimensionality.
CITY_FREQUENCY_THRESHOLD = 300

LOW_CARD_CATEGORICAL = ["state", "fee", "has_photo", "city_category"]
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


def engineer_pets_features(df: pd.DataFrame) -> pd.DataFrame:
    """Split pets_allowed the same way as amenities: 'Cats' and 'Cats,Dogs' both
    indicate cats are allowed, but naive one-hot treats them as unrelated
    categories. allows_cats/allows_dogs share that signal explicitly.
    """
    df = df.copy()
    pets_str = df["pets_allowed"].fillna("Not Specified").astype(str)

    df["allows_cats"] = pets_str.str.contains("Cats", regex=False).astype(int)
    df["allows_dogs"] = pets_str.str.contains("Dogs", regex=False).astype(int)
    df["pets_policy_specified"] = (pets_str != "Not Specified").astype(int)

    return df.drop(columns=["pets_allowed"])


def engineer_room_density(df: pd.DataFrame) -> pd.DataFrame:
    """square_feet per room (bedrooms + bathrooms) — kept ALONGSIDE square_feet,
    not as a replacement, so models can use either or both.
    bathrooms is always >= 1 after cleaning, so the denominator is never zero.
    """
    df = df.copy()
    df["squarefeet_per_room"] = df["square_feet"] / (df["bedrooms"] + df["bathrooms"])
    return df


def encode_city(df: pd.DataFrame, train_df: pd.DataFrame = None) -> pd.DataFrame:
    """city_category: city name for cities with >= CITY_FREQUENCY_THRESHOLD listings,
    else "Other" — one-hot encoded downstream.

    Pass the untransformed training dataframe as train_df when transforming
    validation/test data so the kept-city list is learned from training data
    only (no leakage).
    """
    df = df.copy()
    reference = train_df if train_df is not None else df
    freq_map = reference["cityname"].value_counts()
    kept_cities = freq_map[freq_map >= CITY_FREQUENCY_THRESHOLD].index

    df["city_category"] = df["cityname"].where(df["cityname"].isin(kept_cities), "Other")

    return df.drop(columns=["cityname"])


def transform(df: pd.DataFrame, train_df: pd.DataFrame = None) -> pd.DataFrame:
    """Apply full 3.3 transformation. Low-cardinality categoricals are one-hot
    encoded inside the modelling pipeline (see build_preprocessor), so
    encoders are fit on training data only.
    """
    df = engineer_amenity_features(df)
    df = engineer_pets_features(df)
    df = engineer_room_density(df)
    df = encode_city(df, train_df=train_df)
    return df


def get_engineered_feature_lists(df: pd.DataFrame) -> tuple:
    """Return (numeric_features, categorical_features) after transform() has been applied."""
    amenity_flag_cols = [c for c in df.columns if c.startswith("amenity_has_")]
    numeric = (NUMERIC_FEATURES_BASE
               + ["amenity_count", "squarefeet_per_room",
                  "allows_cats", "allows_dogs", "pets_policy_specified"]
               + amenity_flag_cols)
    categorical = LOW_CARD_CATEGORICAL
    return numeric, categorical


def build_preprocessor(numeric_features: list, categorical_features: list,
                        scale_numeric: bool = True) -> ColumnTransformer:
    """Build a preprocessing transformer that converts engineered features into
    a numeric matrix for modelling.

    scale_numeric=True for Linear Regression / KNN (scale-sensitive).
    scale_numeric=False for Random Forest / Gradient Boosting (scale-invariant).
    """
    numeric_steps = [("scaler", StandardScaler())] if scale_numeric else []
    numeric_pipeline = Pipeline(numeric_steps) if numeric_steps else "passthrough"

    return ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ])
