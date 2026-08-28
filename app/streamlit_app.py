"""
Apartment rental price prediction — Streamlit prototype.

Nothing is trained here: every model is loaded from ``models/`` exactly as
``notebooks/03_modelling.ipynb`` saved it. The app only rebuilds the same
train/test split, so it can score the saved models on the held-out test set and
fill the input widgets with real categories from the prepared dataset.

Artifacts expected in ``models/`` (see notebooks/03_modelling.ipynb):
    preprocessor_scaled.pkl  fitted ColumnTransformer (scaled — linear + KNN)
    linear_regression.pkl    fitted LinearRegression
    knn.pkl                  fitted GridSearchCV over KNeighborsRegressor
    random_forest.pkl        fitted GridSearchCV over a preprocessing pipeline
    xgboost.pkl              fitted GridSearchCV over a preprocessing pipeline

This file is the entry point only: it checks the artifacts exist, then hands over
to the pages in ``app_pages/``. Shared loading and charting live in ``shared.py``.
"""

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
# Pages are executed by st.navigation as scripts, so both directories have to be
# importable before any of them run.
for path in (str(REPO_ROOT), str(APP_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

st.set_page_config(
    page_title="Apartment rent predictor",
    page_icon=":material/home_work:",
    layout="wide",
)

from shared import (  # noqa: E402  — must follow the sys.path bootstrap above
    RANDOM_FOREST_PATH,
    RAW_CSV,
    config,
    missing_files,
)
from src.model_parts import join_model, parts_available  # noqa: E402


# ---------------------------------------------------------------------------
# Preflight — everything the pages assume is on disk
# ---------------------------------------------------------------------------
# random_forest.pkl is too large for GitHub, so it is committed as parts under
# models/parts/ — a fresh clone has to reassemble it once before anything works.
if not RANDOM_FOREST_PATH.exists() and parts_available():
    with st.spinner("Rebuilding random_forest.pkl from its committed parts…"):
        join_model(verbose=False)

absent = missing_files()
if absent:
    st.title("Apartment rental price predictor")
    st.error(
        "Missing model artifacts in `models/`: "
        + ", ".join(f"`{name}`" for name in absent)
        + ". Run `notebooks/03_modelling.ipynb` to train and save them.",
        icon=":material/folder_off:",
    )
    st.stop()

if not RAW_CSV.exists():
    st.title("Apartment rental price predictor")
    st.error(
        f"Raw dataset not found at `{RAW_CSV}`. Run `python load_csv.py` to download it "
        "from Kaggle.",
        icon=":material/database:",
    )
    st.stop()


# ---------------------------------------------------------------------------
# Navigation — prediction first, because it is what the app is for
# ---------------------------------------------------------------------------
page = st.navigation(
    [
        st.Page(
            "app_pages/predict.py",
            title="Predict",
            icon=":material/query_stats:",
            default=True,
        ),
        st.Page("app_pages/eda.py", title="Explore the data", icon=":material/insights:"),
        st.Page("app_pages/models.py", title="Models", icon=":material/leaderboard:"),
        st.Page(
            "app_pages/diagnostics.py", title="Diagnostics", icon=":material/troubleshoot:"
        ),
        st.Page("app_pages/dataset.py", title="Dataset", icon=":material/database:"),
    ],
    position="sidebar",
)

st.title("Apartment rental price predictor")

page.run()

st.caption(
    f"Data: `{config.KAGGLE_DATASET}` (Kaggle) · models loaded from `models/`"
)
