# DRAFT ONLY(gen by AI) — this is a prototype for the compulsory deliverable of BMDS2003 Data Science, not a production-ready application.

"""
Apartment Rental Price Prediction — deployment prototype (compulsory deliverable).

Design choice: the model is trained on app startup and cached with
st.cache_resource, rather than loading a pickled .pkl file. This avoids
the common deploy-breaking problem where a model pickled with one
scikit-learn version fails to load with a different version on
Streamlit Cloud. Training on ~99k rows with these features takes a
few seconds, and st.cache_resource means it only happens once per
container lifetime, not on every user interaction.
"""
import sys
from pathlib import Path

import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# allow "from src.xxx import yyy" when running from repo root
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import config
from src.data_cleaning import clean

st.set_page_config(page_title="Apartment Rent Predictor", page_icon="🏠")


@st.cache_resource(show_spinner="Loading data and training model (first load only)...")
def get_trained_pipeline():
    raw = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        config.KAGGLE_DATASET,
        "",  # update to the exact file_path used in load_csv.py if the dataset has multiple files
    )
    df = clean(raw, verbose=False)

    X = df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES]
    y = df[config.TARGET]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), config.NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES),
    ])

    # NOTE: swap in whichever model your evaluation (Section 4.0) selects as best.
    # RandomForest is used here as a reasonable placeholder default.
    model = Pipeline([
        ("preprocess", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)),
    ])
    model.fit(X, y)
    return model, df


pipeline, df = get_trained_pipeline()

st.title("🏠 Apartment Rental Price Predictor")
st.caption(
    "Prototype for BMDS2003 Data Science — predictions are estimates based on "
    "listed features only and do not account for negotiation, seasonality, "
    "or individual landlord pricing strategy."
)

col1, col2 = st.columns(2)
with col1:
    bathrooms = st.number_input("Bathrooms", min_value=1.0, max_value=9.0, value=1.0, step=0.5)
    bedrooms = st.number_input("Bedrooms", min_value=0.0, max_value=9.0, value=2.0, step=1.0)
    square_feet = st.number_input("Square feet", min_value=100, max_value=5000, value=850)
with col2:
    state = st.selectbox("State", sorted(df["state"].unique()))
    cities_in_state = sorted(df.loc[df["state"] == state, "cityname"].unique())
    cityname = st.selectbox("City", cities_in_state)
    fee = st.selectbox("Listing fee required?", sorted(df["fee"].unique()))

pets_allowed = st.selectbox("Pets allowed", sorted(df["pets_allowed"].unique()))
has_photo = st.selectbox("Has photo", sorted(df["has_photo"].unique()))
amenities = st.selectbox(
    "Amenities profile (pick a common combination from the data, or 'None')",
    sorted(df["amenities"].value_counts().head(20).index.tolist()) + ["None"],
)

# lat/long: use the median for the chosen city as a stand-in, since a real user
# won't type coordinates directly
city_rows = df[(df["state"] == state) & (df["cityname"] == cityname)]
latitude = city_rows["latitude"].median()
longitude = city_rows["longitude"].median()

if st.button("Predict rental price", type="primary"):
    input_row = pd.DataFrame([{
        "bathrooms": bathrooms, "bedrooms": bedrooms, "square_feet": square_feet,
        "latitude": latitude, "longitude": longitude,
        "cityname": cityname, "state": state, "amenities": amenities,
        "pets_allowed": pets_allowed, "fee": fee, "has_photo": has_photo,
    }])
    prediction = pipeline.predict(input_row)[0]
    st.metric("Estimated monthly rent", f"${prediction:,.0f}")
    st.caption(
        "This is a model estimate, not a valuation. It reflects patterns in "
        "listed features from the training data only."
    )
