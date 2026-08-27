"""
Apartment rental price prediction — Streamlit prototype (BMDS2003 Data Science).

This is the Deployment step of the CRISP-DM write-up. Nothing is trained here:
every model is loaded from ``models/`` exactly as ``notebooks/03_modelling.ipynb``
saved it. The app only rebuilds the same train/test split, so it can score the
saved models on the held-out test set and fill the input widgets with real
categories from the prepared dataset.

Artifacts expected in ``models/`` (see notebooks/03_modelling.ipynb):
    preprocessor_scaled.pkl  fitted ColumnTransformer (scaled — linear + KNN)
    linear_regression.pkl    fitted LinearRegression
    knn.pkl                  fitted GridSearchCV over KNeighborsRegressor
    random_forest.pkl        fitted GridSearchCV over a preprocessing pipeline
    xgboost.pkl              fitted GridSearchCV over a preprocessing pipeline
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
from src.data_transformation import TOP_AMENITIES, transform
from src.final_dataset_output import RAW_CSV_PATH, build_prepared_dataset
from src.modelling import evaluate_model, to_dense

MODELS_DIR = REPO_ROOT / "models"
FIGURES_DIR = REPO_ROOT / "outputs" / "figures"
RAW_CSV = REPO_ROOT / RAW_CSV_PATH

# Must match notebooks/03_modelling.ipynb — the saved models were fitted on this
# exact split, so changing either value here would score them on rows they saw.
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Vega sends every plotted row to the browser, so the scatter plots are sampled.
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
# the selectbox reads "AK (Alaska)" instead of making the user decode "AK".
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


def state_label(code: str) -> str:
    """Render a state code with its full name, e.g. ``AK (Alaska)``."""
    name = STATE_NAMES.get(code)
    return f"{code} ({name})" if name else code

st.set_page_config(
    page_title="Apartment rent predictor",
    page_icon=":material/home_work:",
    layout="wide",
)


# ============================================================================
# Loading — models from disk, data rebuilt to match the saved split
# ============================================================================
@st.cache_resource(show_spinner="Loading saved models from models/…")
def load_artifacts() -> dict:
    """Load the fitted preprocessor and all four models from ``models/``.

    Cached as a resource: the unpickled estimators are shared across sessions
    and reruns, so the ~600 MB random forest is read from disk only once.
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
# Helpers
# ============================================================================
def money(value: float) -> str:
    return f"${value:,.0f}"


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


def ranked_bar_chart(frame: pd.DataFrame, value: str, label: str, title: str) -> alt.Chart:
    """Horizontal bar chart sorted by ``value`` — used for coefficients and importances."""
    return (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X(f"{value}:Q", title=title),
            y=alt.Y(f"{label}:N", sort="-x", title=None),
            color=alt.Color(f"{value}:Q", scale=alt.Scale(scheme="blueorange"), legend=None),
            tooltip=[label, value],
        )
        .properties(height=28 * len(frame) + 40)
    )


# ============================================================================
# Page header — rendered before the slow loads so the app never looks blank
# ============================================================================
st.title("Apartment rental price predictor")
st.caption(
    "Prototype for BMDS2003 Data Science. Predictions are estimates from listed "
    "features only — they do not account for negotiation, seasonality, or an "
    "individual landlord's pricing strategy."
)

absent = missing_files()
if absent:
    st.error(
        "Missing model artifacts in `models/`: "
        + ", ".join(f"`{name}`" for name in absent)
        + ". Run `notebooks/03_modelling.ipynb` to train and save them.",
        icon=":material/folder_off:",
    )
    st.stop()

if not RAW_CSV.exists():
    st.error(
        f"Raw dataset not found at `{RAW_CSV_PATH}`. Run `python load_csv.py` to "
        "download it from Kaggle.",
        icon=":material/database:",
    )
    st.stop()

artifacts = load_artifacts()
dataset = load_dataset()
scores = score_models(artifacts)

prepared_df = dataset["prepared_df"]
y_test = np.asarray(dataset["y_test"])
leaderboard = metrics_frame(scores, artifacts)
best_model = leaderboard.iloc[0]


# ============================================================================
# Sidebar — provenance of everything the app is showing
# ============================================================================
with st.sidebar:
    st.subheader("How this app runs", divider="gray")
    st.caption("Models are read from `models/` — this app scores them, it never retrains.")

    st.subheader("Split", divider="gray")
    st.caption(
        f"{dataset['n_train']:,} training rows and {len(y_test):,} test rows — an "
        f"{int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)} split at "
        f"random_state={RANDOM_STATE}, identical to the modelling notebook, so the "
        "metrics here are measured on listings no model has seen."
    )


# ============================================================================
# Headline metrics
# ============================================================================
with st.container(horizontal=True):
    st.metric(
        "Best model",
        best_model["Model"],
        delta=f"RMSE {money(best_model['Test RMSE'])}",
        delta_arrow="off",
        border=True,
    )
    st.metric("Prepared listings", f"{len(prepared_df):,}", border=True)
    st.metric("Median monthly rent", money(prepared_df[config.TARGET].median()), border=True)
    st.metric(
        "Typical error of best model",
        money(best_model["Test MAE"]),
        delta=f"R² {best_model['Test R²']:.3f}",
        delta_arrow="off",
        border=True,
    )

comparison_tab, diagnostics_tab, drivers_tab, tuning_tab, data_tab, figures_tab = st.tabs(
    [
        "Model comparison",
        "Predicted vs actual",
        "What drives rent",
        "Hyperparameter search",
        "Data overview",
        "Saved figures",
    ]
)


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------
with comparison_tab:
    with st.container(horizontal=True):
        for key in MODEL_SPECS:
            entry = artifacts[key]
            with st.container(border=True, width=320):
                st.markdown(f"{entry['icon']} **{entry['label']}**")
                st.metric("Test RMSE", money(scores[key]["rmse"]))
                with st.container(horizontal=True):
                    st.metric("MAE", money(scores[key]["mae"]))
                    st.metric("R²", f"{scores[key]['r2']:.3f}")
                if entry["cv_rmse"] is not None:
                    st.caption(f"Best 5-fold CV RMSE: {money(entry['cv_rmse'])}")
                else:
                    st.caption("No hyperparameters to tune.")

    metric_choice = st.segmented_control(
        "Metric",
        ["Test RMSE", "Test MAE", "Test R²"],
        default="Test RMSE",
        key="comparison_metric",
    )
    if metric_choice:
        lower_is_better = metric_choice != "Test R²"
        st.altair_chart(
            alt.Chart(leaderboard)
            .mark_bar()
            .encode(
                x=alt.X(f"{metric_choice}:Q", title=metric_choice),
                y=alt.Y("Model:N", sort="x" if lower_is_better else "-x", title=None),
                tooltip=["Model", metric_choice],
            )
            .properties(height=200)
        )
        st.caption(
            "Lower is better for RMSE and MAE."
            if lower_is_better
            else "R² is the share of test-set price variance the model explains — higher is better."
        )

    st.dataframe(
        leaderboard,
        hide_index=True,
        column_config={
            "Test RMSE": st.column_config.NumberColumn(format="$%.0f"),
            "Test MAE": st.column_config.NumberColumn(format="$%.0f"),
            "Test R²": st.column_config.NumberColumn(format="%.4f"),
            "CV RMSE": st.column_config.NumberColumn(
                format="$%.0f", help="Best cross-validated RMSE reached during tuning"
            ),
        },
    )
    st.caption(
        f"**{best_model['Model']}** gives the lowest test RMSE "
        f"({money(best_model['Test RMSE'])}), so it is the model this project "
        "recommends for deployment."
    )


# ---------------------------------------------------------------------------
# Predicted vs actual
# ---------------------------------------------------------------------------
with diagnostics_tab:
    diagnostic_label = st.segmented_control(
        "Model",
        [spec["label"] for spec in MODEL_SPECS.values()],
        default=MODEL_SPECS["linear"]["label"],
        key="diagnostic_model",
    )
    if diagnostic_label:
        diagnostic_key = next(
            key for key, spec in MODEL_SPECS.items() if spec["label"] == diagnostic_label
        )
        predictions = np.asarray(scores[diagnostic_key]["predictions"])

        left, right = st.columns(2)
        with left:
            with st.container(border=True):
                st.markdown("**Predicted against actual rent**")
                st.altair_chart(predicted_vs_actual_chart(y_test, predictions))
        with right:
            with st.container(border=True):
                st.markdown("**Residuals against predicted rent**")
                st.altair_chart(residual_chart(y_test, predictions))

        over_predicted = int((predictions > y_test).sum())
        st.caption(
            f"Points on the dashed diagonal are exact predictions. {diagnostic_label} "
            f"over-predicts {over_predicted / len(y_test):.0%} of test listings; a "
            "residual cloud centred on zero means the model is not systematically too "
            f"high or too low. Both charts show a random sample of "
            f"{min(SCATTER_SAMPLE, len(y_test)):,} test listings."
        )


# ---------------------------------------------------------------------------
# What drives rent
# ---------------------------------------------------------------------------
with drivers_tab:
    coefficient_column, importance_column = st.columns(2)

    with coefficient_column:
        with st.container(border=True):
            st.markdown("**Linear regression coefficients**")
            coefficients = pd.DataFrame(
                {
                    "Feature": artifacts["preprocessor"].get_feature_names_out(),
                    "Coefficient": artifacts["linear"]["model"].coef_,
                }
            )
            top_coefficients = coefficients.reindex(
                coefficients["Coefficient"].abs().sort_values(ascending=False).index
            ).head(12)
            st.altair_chart(
                ranked_bar_chart(
                    top_coefficients, "Coefficient", "Feature", "Coefficient (USD per unit)"
                )
            )
            st.caption(
                "Numeric inputs are standardised, so magnitudes are comparable. A "
                "positive coefficient raises predicted rent, holding everything else "
                "constant."
            )

    with importance_column:
        with st.container(border=True):
            st.markdown("**Tree-model feature importance**")
            tree_label = st.segmented_control(
                "Tree model",
                [MODEL_SPECS["rf"]["label"], MODEL_SPECS["xgb"]["label"]],
                default=MODEL_SPECS["rf"]["label"],
                key="importance_model",
            )
            tree_key = "xgb" if tree_label == MODEL_SPECS["xgb"]["label"] else "rf"
            importances = pipeline_importances(artifacts[tree_key]["model"])
            st.altair_chart(
                ranked_bar_chart(
                    importances.nlargest(12, "Importance"), "Importance", "Feature", "Importance"
                )
            )
            st.caption(
                "Importance is how much each feature reduced prediction error across the "
                "trees. Unlike a coefficient it has no direction — only strength."
            )


# ---------------------------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------------------------
with tuning_tab:
    tuned = {
        key: spec
        for key, spec in MODEL_SPECS.items()
        if artifacts[key]["cv_results"] is not None
    }
    tuned_label = st.segmented_control(
        "Tuned model",
        [spec["label"] for spec in tuned.values()],
        default=MODEL_SPECS["rf"]["label"],
        key="tuning_model",
    )
    if tuned_label:
        tuned_key = next(key for key, spec in tuned.items() if spec["label"] == tuned_label)
        search = search_frame(artifacts[tuned_key]["cv_results"])
        parameters = [
            column
            for column in search.columns
            if column not in {"CV RMSE", "Mean fit time (s)"}
        ]

        st.caption(
            f"{len(search)} parameter combinations, each scored with 5-fold "
            f"cross-validation. Best: `{artifacts[tuned_key]['best_params']}` at "
            f"{money(artifacts[tuned_key]['cv_rmse'])} CV RMSE."
        )

        focus = st.segmented_control(
            "Compare by", parameters, default=parameters[0], key=f"tuning_focus_{tuned_key}"
        )
        if focus:
            # Best score reached per value, so one parameter stays readable without
            # averaging away the combinations that actually won.
            best_per_value = (
                search.groupby(focus, as_index=False)["CV RMSE"].min().sort_values("CV RMSE")
            )
            st.altair_chart(
                alt.Chart(best_per_value)
                .mark_bar()
                .encode(
                    x=alt.X("CV RMSE:Q", title="Best cross-validated RMSE (USD)"),
                    y=alt.Y(f"{focus}:N", sort="x", title=focus),
                    tooltip=[focus, "CV RMSE"],
                )
                .properties(height=max(160, 32 * len(best_per_value)))
            )

        with st.expander("Full search results", icon=":material/table_chart:"):
            st.dataframe(
                search,
                hide_index=True,
                column_config={
                    "CV RMSE": st.column_config.NumberColumn(format="$%.0f"),
                    "Mean fit time (s)": st.column_config.NumberColumn(format="%.1f"),
                },
            )


# ---------------------------------------------------------------------------
# Data overview
# ---------------------------------------------------------------------------
with data_tab:
    with st.container(horizontal=True):
        st.metric("Listings", f"{len(prepared_df):,}", border=True)
        st.metric("Features", prepared_df.shape[1] - 1, border=True)
        st.metric("Missing values", f"{int(prepared_df.isna().sum().sum()):,}", border=True)
        st.metric("Cities covered", f"{prepared_df['cityname'].nunique():,}", border=True)

    distribution_column, size_column = st.columns(2)

    with distribution_column:
        with st.container(border=True):
            st.markdown("**Distribution of monthly rent**")
            # Binned in pandas rather than Vega so only 50 rows reach the browser.
            counts, edges = np.histogram(prepared_df[config.TARGET], bins=50)
            st.bar_chart(
                pd.DataFrame(
                    {"Monthly rent (USD)": (edges[:-1] + edges[1:]) / 2, "Listings": counts}
                ),
                x="Monthly rent (USD)",
                y="Listings",
                height=340,
            )

    with size_column:
        with st.container(border=True):
            st.markdown("**Rent against apartment size**")
            st.scatter_chart(
                prepared_df.sample(
                    min(SCATTER_SAMPLE, len(prepared_df)), random_state=RANDOM_STATE
                ),
                x="square_feet",
                y="price",
                x_label="Square feet",
                y_label="Monthly rent (USD)",
                height=340,
            )

    st.caption(
        "A wide spread of rents at the same apartment size shows that size alone cannot "
        "explain price — location and listing characteristics matter too, which is why "
        "cityname is target-encoded rather than dropped."
    )

    with st.expander("Sample of the prepared dataset", icon=":material/table_rows:"):
        st.dataframe(prepared_df.head(200), hide_index=True)


# ---------------------------------------------------------------------------
# Saved figures
# ---------------------------------------------------------------------------
with figures_tab:
    saved_figures = sorted(FIGURES_DIR.glob("*.png")) if FIGURES_DIR.exists() else []
    if not saved_figures:
        st.caption("No figures found in `outputs/figures/`. Run the notebooks to export them.")
    else:
        chosen_figure = st.selectbox(
            "Figure",
            saved_figures,
            format_func=lambda path: path.stem.replace("_", " ").capitalize(),
        )
        st.image(str(chosen_figure), caption=chosen_figure.name, width="stretch")
        st.caption(
            f"{len(saved_figures)} figures exported by the notebooks into "
            "`outputs/figures/`. A model missing from this list simply has not been "
            "re-exported since it was last trained."
        )


# ============================================================================
# Prediction
# ============================================================================
st.header("Estimate a rent", divider="gray")
st.caption("Describe a listing once to see what all four saved models predict for it.")

location_column, property_column = st.columns(2)

with location_column:
    with st.container(border=True):
        st.markdown("**Location**")
        state = st.selectbox(
            "State", sorted(prepared_df["state"].unique()), format_func=state_label
        )
        cities = sorted(prepared_df.loc[prepared_df["state"] == state, "cityname"].unique())
        cityname = st.selectbox("City", cities)

        # Latitude and longitude are model inputs, so they are filled from the
        # median listing in the chosen city rather than asked for.
        city_rows = prepared_df[
            (prepared_df["state"] == state) & (prepared_df["cityname"] == cityname)
        ]
        latitude = float(city_rows["latitude"].median())
        longitude = float(city_rows["longitude"].median())
        st.caption(
            f"{len(city_rows):,} listings in {cityname}, {state_label(state)}. Coordinates auto-filled "
            f"from the city median: {latitude:.4f}, {longitude:.4f}."
        )

with property_column:
    with st.container(border=True):
        st.markdown("**Property**")
        left_input, right_input = st.columns(2)
        square_feet = left_input.number_input(
            "Square feet", min_value=100, max_value=5_000, value=850, step=25
        )
        bedrooms = right_input.number_input(
            "Bedrooms", min_value=0.0, max_value=9.0, value=2.0, step=1.0
        )
        bathrooms = left_input.number_input(
            "Bathrooms", min_value=1.0, max_value=9.0, value=1.0, step=0.5
        )
        photo_label = right_input.selectbox("Listing photo", list(PHOTO_OPTIONS))

with st.container(border=True):
    st.markdown("**Amenities and pets**")
    selected_amenities = st.pills(
        "Amenities", TOP_AMENITIES, selection_mode="multi", key="amenities"
    )
    pets_label = st.segmented_control(
        "Pets allowed", list(PETS_OPTIONS), default="Not specified", key="pets"
    )

predict = st.button(
    "Predict rental price", type="primary", icon=":material/query_stats:", width="stretch"
)

if predict:
    listing = pd.DataFrame(
        [
            {
                "bathrooms": bathrooms,
                "bedrooms": bedrooms,
                "square_feet": square_feet,
                "latitude": latitude,
                "longitude": longitude,
                "cityname": cityname,
                "state": state,
                "amenities": ",".join(selected_amenities) if selected_amenities else "None",
                "pets_allowed": PETS_OPTIONS.get(pets_label, "Not Specified"),
                "has_photo": PHOTO_OPTIONS[photo_label],
            }
        ]
    )

    # Exactly the transformation the models were trained on.
    listing_transformed = transform(listing)
    listing_scaled = to_dense(artifacts["preprocessor"].transform(listing_transformed))

    predictions = {}
    for key, spec in MODEL_SPECS.items():
        features = listing_scaled if spec["features"] == "scaled" else listing_transformed
        predictions[spec["label"]] = float(artifacts[key]["model"].predict(features)[0])

    st.subheader("Predictions")
    with st.container(horizontal=True):
        for key, spec in MODEL_SPECS.items():
            with st.container(border=True, width=320):
                st.markdown(f"{spec['icon']} **{spec['label']}**")
                st.metric("Estimated monthly rent", money(predictions[spec["label"]]))
                st.caption(f"Test RMSE {money(scores[key]['rmse'])}")

    values = list(predictions.values())
    spread = max(values) - min(values)
    relative_spread = spread / (sum(values) / len(values))

    with st.container(horizontal=True):
        st.metric("Lowest estimate", money(min(values)), border=True)
        st.metric("Highest estimate", money(max(values)), border=True)
        st.metric(
            "Spread across models",
            money(spread),
            delta=f"{relative_spread:.1%} of the average",
            delta_arrow="off",
            border=True,
        )
        st.metric(
            f"{best_model['Model']} estimate",
            money(predictions[best_model["Model"]]),
            border=True,
        )

    if relative_spread < 0.10:
        st.success("The models agree closely on this listing.", icon=":material/check_circle:")
    elif relative_spread < 0.20:
        st.warning(
            "The models disagree moderately — treat the estimate as a range.",
            icon=":material/warning:",
        )
    else:
        st.error(
            "The models disagree strongly, which usually means this listing is unlike "
            "anything in the training data.",
            icon=":material/priority_high:",
        )

    st.caption(
        f"This is a model estimate, not a valuation. **{best_model['Model']}** is the most "
        f"accurate model on the held-out test set ({money(best_model['Test RMSE'])} RMSE), "
        "so its figure is the one to quote if you need a single number."
    )

st.caption(
    "BMDS2003 Data Science — deployment prototype · data: "
    f"`{config.KAGGLE_DATASET}` (Kaggle) · models loaded from `models/`"
)
