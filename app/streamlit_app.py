"""Apartment rental price prediction — Streamlit prototype."""

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
for path in (str(REPO_ROOT), str(APP_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

st.set_page_config(page_title="Apartment rent predictor", page_icon=":material/home_work:", layout="wide")

# Keep the existing Streamlit look, but give each section enough room to scan.
st.markdown(
    """
    <style>
    :root {
        --glass-surface: rgba(255, 255, 255, .10);
        --glass-border: rgba(255, 255, 255, .20);
        --glass-shadow: rgba(0, 0, 0, .20);
    }
    .stApp {
        background: radial-gradient(circle at 12% 8%, #2d4261 0, transparent 34%),
                    radial-gradient(circle at 88% 18%, #44375c 0, transparent 30%),
                    #10151f;
    }
    @keyframes page-enter {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes chart-swap {
        from { opacity: 0; transform: translateX(14px) scale(.985); }
        to { opacity: 1; transform: translateX(0) scale(1); }
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 1220px;
        padding-top: 2.25rem;
        padding-bottom: 3rem;
        animation: page-enter 280ms cubic-bezier(.2, .75, .25, 1) both;
        will-change: opacity, transform;
    }
    [data-testid="stMainBlockContainer"] > div > div { gap: 1.35rem; }
    [data-testid="stHeader"],
    [data-testid="stNavigation"],
    div[data-testid="stVerticalBlockBorderWrapper"],
    div[data-testid="stMetric"] {
        background: var(--glass-surface);
        backdrop-filter: blur(16px) saturate(125%);
        -webkit-backdrop-filter: blur(16px) saturate(125%);
        border-radius: 12px;
        border-color: var(--glass-border);
        box-shadow: 0 10px 28px var(--glass-shadow);
        transition: background 180ms ease, border-color 180ms ease, box-shadow 180ms ease, transform 180ms ease;
    }
    [data-testid="stHeader"] { background: rgba(16, 21, 31, .56); border-bottom: 1px solid var(--glass-border); }
    [data-testid="stNavigation"] { padding: .3rem .45rem; }
    [data-testid="stNavigation"] a { border-radius: 8px; transition: background 180ms ease, color 180ms ease; }
    [data-testid="stNavigation"] a:hover, [data-testid="stNavigation"] a[aria-current="page"] { background: rgba(255, 255, 255, .13); }
    div[data-testid="stVerticalBlockBorderWrapper"] { margin: .35rem 0; padding: .25rem; }
    div[data-testid="stMetric"] { padding: .9rem 1rem; }
    [data-testid="stHorizontalBlock"] { gap: 1.25rem; }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover,
    div[data-testid="stMetric"]:hover { background: rgba(255, 255, 255, .15); border-color: rgba(255, 255, 255, .34); transform: translateY(-2px); }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
        background: rgba(15, 20, 30, .34) !important;
        border-color: var(--glass-border) !important;
        transition: background 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
    }
    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="input"] > div:focus-within { border-color: rgba(171, 205, 255, .70) !important; box-shadow: 0 0 0 3px rgba(124, 166, 224, .18); }
    button { transition: background 180ms ease, border-color 180ms ease, box-shadow 180ms ease, transform 180ms ease !important; }
    button:hover { transform: translateY(-1px); }
    [data-testid="stAltairChart"], [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.importance-transition) [data-testid="stAltairChart"] {
        animation: chart-swap 260ms cubic-bezier(.2, .75, .25, 1) both;
    }
    @media (max-width: 760px) {
        [data-testid="stMainBlockContainer"] { padding: 1.4rem 1rem 2.25rem; }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { transition-duration: .01ms !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

from shared import RANDOM_FOREST_PATH, RAW_CSV, config, missing_files  # noqa: E402
from src.model_parts import join_model, parts_available  # noqa: E402

if not RANDOM_FOREST_PATH.exists() and parts_available():
    with st.spinner("Rebuilding random_forest.pkl from its committed parts…"):
        join_model(verbose=False)

absent = missing_files()
if absent:
    st.title("Apartment rental price predictor")
    st.error("Missing model artifacts in `models/`: " + ", ".join(f"`{name}`" for name in absent) + ". Run `notebooks/03_modelling.ipynb` to train and save them.", icon=":material/folder_off:")
    st.stop()
if not RAW_CSV.exists():
    st.title("Apartment rental price predictor")
    st.error(f"Raw dataset not found at `{RAW_CSV}`. Run `python load_csv.py` to download it from Kaggle.", icon=":material/database:")
    st.stop()

page = st.navigation([
    st.Page("app_pages/predict.py", title="Predict", icon=":material/query_stats:", default=True),
    st.Page("app_pages/eda.py", title="Explore the data", icon=":material/insights:"),
    st.Page("app_pages/models.py", title="Models", icon=":material/leaderboard:"),
    st.Page("app_pages/diagnostics.py", title="Diagnostics", icon=":material/troubleshoot:"),
    st.Page("app_pages/dataset.py", title="Dataset", icon=":material/database:"),
], position="top")

st.title("Apartment rental price predictor")
page.run()
st.caption(f"Data: `{config.KAGGLE_DATASET}` (Kaggle) · models loaded from `models`")
