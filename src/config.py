"""
Central column decisions from Section 3.1 (Data Selection).
Import these constants anywhere instead of hardcoding column names,
so every notebook and the Streamlit app stay in sync.
"""

KAGGLE_DATASET = "shashanks1202/apartment-rent-data"  # confirm this matches your repo's load_csv.py

# Columns dropped entirely per 3.1 justification (id, text, zero-variance, leaking, >90% missing)
COLS_TO_DROP = [
    "id", "title", "body", "currency", "price_display",
    "time", "source", "address", "fee"
]

# Columns used once to filter rows, then dropped (non-standard listings)
FILTER_ONLY_COLS = ["category", "price_type"]
VALID_CATEGORY = "housing/rent/apartment"
VALID_PRICE_TYPE = "Monthly"

# Rows with missing values here are DROPPED (trivial % missing, no safe imputation)
CRITICAL_COLS_DROP_IF_MISSING = [
    "price", "bathrooms", "bedrooms", "latitude", "longitude", "cityname", "state",
]

# Rows with missing values here are RECODED as an explicit category (meaningful missingness)
RECODE_MISSING = {
    "pets_allowed": "Not Specified",
    "amenities": "None",
}

# Final feature set after cleaning
NUMERIC_FEATURES = ["bathrooms", "bedrooms", "square_feet", "latitude", "longitude"]
CATEGORICAL_FEATURES = ["cityname", "state", "amenities", "pets_allowed", "has_photo"]
TARGET = "price"
