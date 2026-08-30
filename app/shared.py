"""
Shared loading, caching and charting helpers for the Streamlit prototype.

Every page imports from here, so the dataset is prepared once, the models are
unpickled once, and the same formatting and chart conventions apply everywhere.
Nothing is trained in the app: the estimators are loaded from ``models/``
exactly as ``notebooks/03_modelling.ipynb`` saved them, and the app only
rebuilds the identical train/test split so it can score them honestly on rows
no model has seen.
"""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import config
from src.data_cleaning import load_raw
from src.data_transformation import TOP_AMENITIES, transform
from src.final_dataset_output import RAW_CSV_PATH, build_prepared_dataset
from src.model_parts import MODEL_PATH as RANDOM_FOREST_PATH
from src.modelling import evaluate_model, to_dense

MODELS_DIR = REPO_ROOT / "models"
FIGURES_DIR = REPO_ROOT / "outputs" / "figures"
RAW_CSV = REPO_ROOT / RAW_CSV_PATH

# Must match notebooks/03_modelling.ipynb — the saved models were fitted on this
# exact split, so changing either value here would score them on rows they saw.
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Vega sends every plotted row to the browser, so scatter plots are sampled.
SCATTER_SAMPLE = 4_000

PREPROCESSOR_FILE = "preprocessor_scaled.pkl"

# "scaled" models consume the shared preprocessor_scaled.pkl matrix; "pipeline"
# models carry their own preprocessing step and take the transformed frame as is.
MODEL_SPECS = {
    "linear": {
        "label": "Linear regression",
        "file": "linear_regression.pkl",
        "features": "scaled",
        "icon": ":material/show_chart:",
    },
    "knn": {
        "label": "K-nearest neighbours",
        "file": "knn.pkl",
        "features": "scaled",
        "icon": ":material/scatter_plot:",
    },
    "rf": {
        "label": "Random forest",
        "file": "random_forest.pkl",
        "features": "pipeline",
        "icon": ":material/park:",
    },
    "xgb": {
        "label": "XGBoost",
        "file": "xgboost.pkl",
        "features": "pipeline",
        "icon": ":material/bolt:",
    },
}

PETS_OPTIONS = {
    "Cats and dogs": "Cats,Dogs",
    "Cats only": "Cats",
    "Dogs only": "Dogs",
    "Not specified": "Not Specified",
}
PHOTO_OPTIONS = {"Full photo": "Yes", "Thumbnail only": "Thumbnail", "No photo": "No"}

# The dataset stores states as two-letter codes; these are only for display, so
# a selectbox reads "AK (Alaska)" instead of making the user decode "AK".
STATE_NAMES = {
    "AK": "Alaska", "AL": "Alabama", "AR": "Arkansas", "AZ": "Arizona",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut",
    "DC": "District of Columbia", "DE": "Delaware", "FL": "Florida",
    "GA": "Georgia", "HI": "Hawaii", "IA": "Iowa", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "KS": "Kansas", "KY": "Kentucky",
    "LA": "Louisiana", "MA": "Massachusetts", "MD": "Maryland", "ME": "Maine",
    "MI": "Michigan", "MN": "Minnesota", "MO": "Missouri", "MS": "Mississippi",
    "MT": "Montana", "NC": "North Carolina", "ND": "North Dakota",
    "NE": "Nebraska", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NV": "Nevada", "NY": "New York", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VA": "Virginia",
    "VT": "Vermont", "WA": "Washington", "WI": "Wisconsin",
    "WV": "West Virginia", "WY": "Wyoming",
}

NUMERIC_EDA_COLS = ["price", "square_feet", "bathrooms", "bedrooms", "latitude", "longitude"]


# ============================================================================
# Loading — models from disk, data rebuilt to match the saved split
# ============================================================================
@st.cache_resource(show_spinner="Loading saved models from models/…")
def load_artifacts() -> dict:
    """Load the fitted preprocessor and all four models from ``models/``.

    Cached as a resource: the unpickled estimators are shared across sessions
    and reruns, so the large random forest is read from disk only once.
    """
    artifacts = {"preprocessor": joblib.load(MODELS_DIR / PREPROCESSOR_FILE)}

    for key, spec in MODEL_SPECS.items():
        loaded = joblib.load(MODELS_DIR / spec["file"])
        entry = {**spec}

        # The notebook dumps the whole GridSearchCV (not just best_estimator_),
        # which keeps best_params_/cv_results_ available for the tuning charts.
        if hasattr(loaded, "best_estimator_"):
            entry["model"] = loaded.best_estimator_
            entry["best_params"] = loaded.best_params_
            entry["cv_rmse"] = -loaded.best_score_
            entry["cv_results"] = loaded.cv_results_
        else:
            entry["model"] = loaded
            entry["best_params"] = None
            entry["cv_rmse"] = None
            entry["cv_results"] = None

        artifacts[key] = entry

    return artifacts


@st.cache_data(show_spinner="Preparing the dataset…")
def load_dataset() -> dict:
    """Rebuild the prepared dataset and the modelling split used by the notebook."""
    prepared_df = build_prepared_dataset(raw_csv_path=str(RAW_CSV), verbose=False)

    X = prepared_df.drop(columns=[config.TARGET])
    y = prepared_df[config.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    return {
        "prepared_df": prepared_df,
        "X_test_transformed": transform(X_test),
        "y_test": y_test,
        "n_train": len(X_train),
    }


@st.cache_data(show_spinner="Reading the raw Kaggle CSV…")
def raw_row_count() -> int:
    """Rows in the raw file — the starting point of the cleaning funnel."""
    return len(load_raw(str(RAW_CSV)))


@st.cache_data(show_spinner="Scoring the saved models on the held-out test set…")
def score_models(_artifacts: dict) -> dict:
    """Score every loaded model on the test set the notebook held out.

    ``_artifacts`` is underscore-prefixed so Streamlit skips hashing the
    estimators; they are immutable for the life of the resource cache.
    """
    data = load_dataset()
    X_test_transformed = data["X_test_transformed"]
    y_test = data["y_test"]

    # Linear regression and KNN share this matrix, so transform it once.
    X_test_scaled = to_dense(_artifacts["preprocessor"].transform(X_test_transformed))

    scores = {}
    for key, spec in MODEL_SPECS.items():
        features = X_test_scaled if spec["features"] == "scaled" else X_test_transformed
        scores[key] = evaluate_model(_artifacts[key]["model"], features, y_test)

    return scores


def missing_files() -> list[str]:
    """Return the artifacts the app needs but cannot find on disk."""
    required = [PREPROCESSOR_FILE] + [spec["file"] for spec in MODEL_SPECS.values()]
    return [name for name in required if not (MODELS_DIR / name).exists()]


# ============================================================================
# Formatting helpers
# ============================================================================
def money(value: float) -> str:
    return f"${value:,.0f}"


def escape_dollars(text: str) -> str:
    """Escape ``$`` so Streamlit's Markdown does not read a pair as LaTeX math.

    Streamlit renders ``$...$`` in Markdown as an equation, so a sentence like
    "$373 test RMSE and explains 80.2%" silently turns into gibberish. Anything
    carrying a literal dollar sign into st.markdown, st.caption, an alert, or a
    widget/metric label has to go through this first.
    """
    return text.replace("$", r"\$")


def money_md(value: float) -> str:
    """Dollar amount for text Streamlit renders as Markdown.

    Use ``money()`` for st.metric *values*, which are plain text, and this for
    labels, deltas, captions, alerts and markdown, which are not.
    """
    return escape_dollars(money(value))


def note_metric(label: str, value: str, note: str, **kwargs) -> None:
    """A metric card whose badge is a neutral descriptor rather than a change.

    ``st.metric`` colours a delta green or red by sign, which is wrong for a
    label like "right-skewed" or "+36 vs best" — those are descriptive, not
    good or bad — so the colour and arrow are both turned off.

    Use it for *every* card in a horizontal row, not just one: the badge adds a
    line, and a horizontal container sizes each card to the tallest, so a lone
    badge makes the whole row grow and squeezes its neighbours.
    """
    st.metric(label, value, delta=note, delta_arrow="off", delta_color="off", **kwargs)


def state_label(code: str) -> str:
    """Render a state code with its full name, e.g. ``AK (Alaska)``."""
    name = STATE_NAMES.get(code)
    return f"{code} ({name})" if name else code


def metrics_frame(scores: dict, artifacts: dict) -> pd.DataFrame:
    """One row per model, ordered best (lowest test RMSE) first."""
    rows = [
        {
            "Model": MODEL_SPECS[key]["label"],
            "Test RMSE": scores[key]["rmse"],
            "Test MAE": scores[key]["mae"],
            "Test R²": scores[key]["r2"],
            "CV RMSE": artifacts[key]["cv_rmse"],
        }
        for key in MODEL_SPECS
    ]
    return pd.DataFrame(rows).sort_values("Test RMSE").reset_index(drop=True)


def search_frame(cv_results: dict) -> pd.DataFrame:
    """Flatten ``cv_results_`` into a tidy table of tuning evidence."""
    results = pd.DataFrame(cv_results)
    param_cols = [column for column in results.columns if column.startswith("param_")]

    frame = results[param_cols].astype(str)
    frame.columns = [
        column.replace("param_", "").replace("regressor__", "") for column in param_cols
    ]
    frame["CV RMSE"] = -results["mean_test_score"]
    frame["Mean fit time (s)"] = results["mean_fit_time"]
    return frame.sort_values("CV RMSE").reset_index(drop=True)


def pipeline_importances(model) -> pd.DataFrame:
    """Feature importances from a fitted preprocessing + regressor pipeline."""
    names = model.named_steps["preprocess"].get_feature_names_out()
    return pd.DataFrame(
        {
            "Feature": names,
            "Importance": model.named_steps["regressor"].feature_importances_,
        }
    )


def model_key_for(label: str) -> str:
    """Map a display label back to its MODEL_SPECS key."""
    return next(key for key, spec in MODEL_SPECS.items() if spec["label"] == label)


# ============================================================================
# Aggregation helpers — every chart is aggregated in pandas before it reaches
# Vega, so the browser never receives all 98k rows.
# ============================================================================
def histogram_frame(values: pd.Series, bins: int = 50, label: str = "Value") -> pd.DataFrame:
    """Bin a column in pandas so only ``bins`` rows are sent to the browser."""
    counts, edges = np.histogram(values.dropna(), bins=bins)
    return pd.DataFrame({label: (edges[:-1] + edges[1:]) / 2, "Listings": counts})


def spread_frame(df: pd.DataFrame, group: str, value: str = "price") -> pd.DataFrame:
    """Median with quartile range per group — a boxplot without shipping every row."""
    grouped = df.groupby(group)[value]
    return pd.DataFrame(
        {
            "Median": grouped.median(),
            "Q1": grouped.quantile(0.25),
            "Q3": grouped.quantile(0.75),
            "Listings": grouped.size(),
        }
    ).reset_index()


def amenity_frame(df: pd.DataFrame) -> pd.DataFrame:
    """How often each amenity is advertised, and the median rent with and without it."""
    amenities = df["amenities"].fillna("None").astype(str)
    rows = []
    for amenity in TOP_AMENITIES:
        has_it = amenities.str.contains(amenity, regex=False)
        rows.append(
            {
                "Amenity": amenity,
                "Listings": int(has_it.sum()),
                "Share of listings": float(has_it.mean()),
                "Median rent with": float(df.loc[has_it, "price"].median()),
                "Median rent without": float(df.loc[~has_it, "price"].median()),
            }
        )
    frame = pd.DataFrame(rows)
    frame["Rent premium"] = frame["Median rent with"] - frame["Median rent without"]
    return frame.sort_values("Listings", ascending=False).reset_index(drop=True)


def correlation_long(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Correlation matrix reshaped to long form for an Altair heatmap."""
    matrix = df[columns].corr(numeric_only=True).round(3)
    long = matrix.reset_index().melt(id_vars="index", var_name="Feature 2", value_name="r")
    return long.rename(columns={"index": "Feature 1"})


# ============================================================================
# Chart helpers
# ============================================================================
def spread_chart(frame: pd.DataFrame, group: str, title: str, sort=None) -> alt.LayerChart:
    """Median point with a Q1–Q3 rule per category — a lightweight boxplot."""
    order = sort if sort is not None else alt.EncodingSortField("Median", order="descending")
    base = alt.Chart(frame).encode(
        y=alt.Y(f"{group}:N", sort=order, title=None),
        tooltip=[
            group,
            alt.Tooltip("Median:Q", format="$,.0f"),
            alt.Tooltip("Q1:Q", format="$,.0f"),
            alt.Tooltip("Q3:Q", format="$,.0f"),
            alt.Tooltip("Listings:Q", format=","),
        ],
    )
    rule = base.mark_rule(size=3, opacity=0.35).encode(
        x=alt.X("Q1:Q", title=title, scale=alt.Scale(zero=False)), x2="Q3:Q"
    )
    point = base.mark_circle(size=110).encode(
        x="Median:Q", color=alt.Color("Median:Q", scale=alt.Scale(scheme="teals"), legend=None)
    )
    return (rule + point).properties(height=max(180, 30 * len(frame)))


def ranked_bar_chart(
    frame: pd.DataFrame, value: str, label: str, title: str, scheme: str = "blueorange"
) -> alt.Chart:
    """Horizontal bar chart sorted by ``value`` — coefficients, importances, rankings."""
    return (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X(f"{value}:Q", title=title),
            y=alt.Y(f"{label}:N", sort="-x", title=None),
            color=alt.Color(f"{value}:Q", scale=alt.Scale(scheme=scheme), legend=None),
            tooltip=[label, value],
        )
        .properties(height=28 * len(frame) + 40)
    )


def heatmap_chart(
    long: pd.DataFrame, x: str, y: str, value: str, scheme: str, number_format: str = ".2f"
) -> alt.LayerChart:
    """
    Rect heatmap with the value printed in each cell.
    """
    base = alt.Chart(long).encode(x=alt.X(f"{x}:N", title=None), y=alt.Y(f"{y}:N", title=None))
    cells = base.mark_rect().encode(
        color=alt.Color(f"{value}:Q", scale=alt.Scale(scheme=scheme), title=value),
        tooltip=[x, y, value],
    )
    labels = base.mark_text(fontSize=11).encode(
        text=alt.Text(f"{value}:Q", format=number_format),
        color=alt.condition(
            f"luminance(scale('color', datum['{value}'])) > 0.5",
            alt.value("#1f2328"),
            alt.value("white"),
        ),
    )
    return (cells + labels).properties(height=320)


def predicted_vs_actual_chart(actual: np.ndarray, predicted: np.ndarray) -> alt.LayerChart:
    """Scatter of predictions against truth, with the ideal y = x reference line."""
    frame = pd.DataFrame({"Actual rent": actual, "Predicted rent": predicted})
    if len(frame) > SCATTER_SAMPLE:
        frame = frame.sample(SCATTER_SAMPLE, random_state=RANDOM_STATE)

    upper = float(max(frame["Actual rent"].max(), frame["Predicted rent"].max()))
    scale = alt.Scale(domain=[0, upper])

    points = (
        alt.Chart(frame)
        .mark_circle(size=18, opacity=0.25)
        .encode(
            x=alt.X("Actual rent:Q", scale=scale, title="Actual rent (USD)"),
            y=alt.Y("Predicted rent:Q", scale=scale, title="Predicted rent (USD)"),
            tooltip=["Actual rent", "Predicted rent"],
        )
    )
    reference = (
        alt.Chart(pd.DataFrame({"value": [0, upper]}))
        .mark_line(strokeDash=[5, 5], color="grey")
        .encode(x="value:Q", y="value:Q")
    )
    return (points + reference).properties(height=380)


def residual_chart(actual: np.ndarray, predicted: np.ndarray) -> alt.LayerChart:
    """Residuals against predictions, with a zero line to read bias against."""
    frame = pd.DataFrame(
        {
            "Predicted rent": predicted,
            "Residual": np.asarray(actual) - np.asarray(predicted),
        }
    )
    if len(frame) > SCATTER_SAMPLE:
        frame = frame.sample(SCATTER_SAMPLE, random_state=RANDOM_STATE)

    points = (
        alt.Chart(frame)
        .mark_circle(size=18, opacity=0.25)
        .encode(
            x=alt.X("Predicted rent:Q", title="Predicted rent (USD)"),
            y=alt.Y("Residual:Q", title="Actual − predicted (USD)"),
            tooltip=["Predicted rent", "Residual"],
        )
    )
    zero_line = (
        alt.Chart(pd.DataFrame({"zero": [0]}))
        .mark_rule(strokeDash=[5, 5], color="grey")
        .encode(y="zero:Q")
    )
    return (points + zero_line).properties(height=380)


# ============================================================================
# Page-level shared state — resolved through the caches above, so the pages
# themselves stay free of loading logic.
# ============================================================================
def app_state() -> dict:
    """Everything the pages need: models, data, test scores and the leaderboard."""
    artifacts = load_artifacts()
    dataset = load_dataset()
    scores = score_models(artifacts)
    leaderboard = metrics_frame(scores, artifacts)

    return {
        "artifacts": artifacts,
        "dataset": dataset,
        "prepared_df": dataset["prepared_df"],
        "y_test": np.asarray(dataset["y_test"]),
        "scores": scores,
        "leaderboard": leaderboard,
        "best": leaderboard.iloc[0],
    }
