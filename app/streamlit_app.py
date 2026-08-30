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


_app_bg = (
    "radial-gradient(circle at 12% 8%, #f3e2ce 0, transparent 34%),"
    "radial-gradient(circle at 88% 18%, #f0e6da 0, transparent 30%),"
    "#faf9f5"
)
_glass_surface = "rgba(255, 252, 245, .55)"
_glass_border = "rgba(61, 53, 40, .12)"
_glass_shadow = "rgba(61, 53, 40, .10)"
_glass_hover_surface = "rgba(255, 252, 245, .82)"
_glass_hover_border = "rgba(61, 53, 40, .20)"
_header_bg = "rgba(250, 246, 237, .70)"
_nav_hover = "rgba(61, 53, 40, .07)"
_input_bg = "#f0eee6"
_focus_border = "rgba(193, 95, 60, .55)"
_focus_shadow = "rgba(193, 95, 60, .18)"

st.markdown(
    f"""
    <style>
    :root {{
        --glass-surface: {_glass_surface};
        --glass-border: {_glass_border};
        --glass-shadow: {_glass_shadow};
    }}
    .stApp {{
        background: {_app_bg};
    }}
    @keyframes page-enter {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes chart-swap {{
        from {{ opacity: 0; transform: translateX(14px) scale(.985); }}
        to {{ opacity: 1; transform: translateX(0) scale(1); }}
    }}
    [data-testid="stMainBlockContainer"] {{
        max-width: 1220px;
        padding-top: 2.25rem;
        padding-bottom: 3rem;
        animation: page-enter 280ms cubic-bezier(.2, .75, .25, 1) both;
        will-change: opacity, transform;
    }}
    [data-testid="stMainBlockContainer"] > div > div {{ gap: 1.35rem; }}
    [data-testid="stHeader"],
    [data-testid="stNavigation"],
    div[data-testid="stVerticalBlockBorderWrapper"],
    div[data-testid="stMetric"] {{
        background: var(--glass-surface);
        backdrop-filter: blur(16px) saturate(125%);
        -webkit-backdrop-filter: blur(16px) saturate(125%);
        border-radius: 12px;
        border-color: var(--glass-border);
        box-shadow: 0 10px 28px var(--glass-shadow);
        transition: background 180ms ease, border-color 180ms ease, box-shadow 180ms ease, transform 180ms ease;
    }}
    [data-testid="stHeader"] {{ background: {_header_bg}; border-bottom: 1px solid var(--glass-border); }}
    [data-testid="stNavigation"] {{ padding: .3rem .45rem; }}
    [data-testid="stNavigation"] a {{ border-radius: 8px; transition: background 180ms ease, color 180ms ease; }}
    [data-testid="stNavigation"] a:hover, [data-testid="stNavigation"] a[aria-current="page"] {{ background: {_nav_hover}; }}
    div[data-testid="stVerticalBlockBorderWrapper"] {{ margin: .35rem 0; padding: .25rem; }}
    div[data-testid="stMetric"] {{ padding: .9rem 1rem; }}
    /* Figures that sit in a sentence rather than a metric card — the predictor's
       "Likely range" pair — lifted above body text so the numbers read first. */
    .range-value {{ font-size: 1.25rem; font-weight: 700; letter-spacing: -.01em; }}
    [data-testid="stHorizontalBlock"] {{ gap: 1.25rem; }}
    div[data-testid="stVerticalBlockBorderWrapper"]:hover,
    div[data-testid="stMetric"]:hover {{ background: {_glass_hover_surface}; border-color: {_glass_hover_border}; transform: translateY(-2px); }}
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {{
        background: {_input_bg} !important;
        border-color: var(--glass-border) !important;
        transition: background 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
    }}
    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="input"] > div:focus-within {{ border-color: {_focus_border} !important; box-shadow: 0 0 0 3px {_focus_shadow}; }}
    button {{ transition: background 180ms ease, border-color 180ms ease, box-shadow 180ms ease, transform 180ms ease !important; }}
    button:hover {{ transform: translateY(-1px); }}
    [data-testid="stAltairChart"], [data-testid="stDataFrame"] {{ border-radius: 12px; overflow: hidden; }}
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.importance-transition) [data-testid="stAltairChart"] {{
        animation: chart-swap 260ms cubic-bezier(.2, .75, .25, 1) both;
    }}
    @media (max-width: 760px) {{
        [data-testid="stMainBlockContainer"] {{ padding: 1.4rem 1rem 2.25rem; }}
    }}
    @media (prefers-reduced-motion: reduce) {{
        *, *::before, *::after {{ transition-duration: .01ms !important; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

from shared import RANDOM_FOREST_PATH, RAW_CSV, missing_files  # noqa: E402
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
