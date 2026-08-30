"""
Dataset and method — provenance for everything the other pages show.

Where the data came from, what cleaning and transformation did to it, how the
split was made, and which figures the notebooks exported.
"""

import pandas as pd
import streamlit as st

from shared import (
    FIGURES_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    app_state,
    config,
    money,
    note_metric,
    raw_row_count,
)

state = app_state()
df = state["prepared_df"]
dataset = state["dataset"]

st.header("Dataset and method", anchor=False)
st.caption(
    "Where the numbers on the other pages come from — the trail from the raw Kaggle file "
    "to the models this app scores."
)

pipeline_tab, data_tab, figures_tab = st.tabs(
    ["Pipeline", "Prepared dataset", "Figures"]
)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
with pipeline_tab:
    raw_rows = raw_row_count()

    with st.container(horizontal=True):
        note_metric("Raw rows", f"{raw_rows:,}", "as downloaded", border=True)
        note_metric(
            "Prepared rows", f"{len(df):,}", f"{len(df) / raw_rows:.1%} retained", border=True
        )
        note_metric(
            "Training rows",
            f"{dataset['n_train']:,}",
            f"{int((1 - TEST_SIZE) * 100)}% of prepared",
            border=True,
        )
        note_metric(
            "Test rows",
            f"{len(state['y_test']):,}",
            f"{int(TEST_SIZE * 100)}% held out",
            border=True,
        )

    st.markdown(
        f"""
**1 · Data selection** (`src/config.py`) — {len(config.COLS_TO_DROP)} columns dropped as
identifiers, free text, zero-variance, leaking or over 90% missing; `category` and
`price_type` used to keep only monthly apartment rentals, then dropped themselves.

**2 · Cleaning** (`src/data_cleaning.py`) — duplicate listing ids removed, rows missing any
of {', '.join(f'`{column}`' for column in config.CRITICAL_COLS_DROP_IF_MISSING)} dropped, and
missing `pets_allowed` / `amenities` recoded as explicit categories rather than deleted,
because that missingness is meaningful.

**3 · Outlier treatment** (`src/outlier_treatment.py`) — implausible prices removed, then
square footage winsorized at the IQR upper bound. Capping rather than deleting keeps the
otherwise-valid rest of those rows.

**4 · Transformation** (`src/data_transformation.py`) — applied *after* the split so
nothing leaks: `cityname` is target-encoded (~2,975 unique values, too many to one-hot),
amenities become a count plus binary flags, pets become `allows_cats` / `allows_dogs`, and
`squarefeet_per_room` is engineered alongside the raw size.

**5 · Modelling** (`notebooks/03_modelling.ipynb`) — four regressors, each tuned by
5-fold cross-validated grid search, saved to `models/`.
"""
    )

    st.caption(
        f"The app rebuilds the same {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)} split at "
        f"`random_state={RANDOM_STATE}` used by the notebook, so every metric shown is measured "
        "on listings no model has seen. It never retrains — it only loads and scores."
    )


# ---------------------------------------------------------------------------
# Prepared dataset
# ---------------------------------------------------------------------------
with data_tab:
    with st.container(horizontal=True):
        st.metric("Listings", f"{len(df):,}", border=True)
        st.metric("Columns", df.shape[1], border=True)
        st.metric("Missing values", f"{int(df.isna().sum().sum()):,}", border=True)
        st.metric("Median rent", money(df["price"].median()), border=True)

    summary = pd.DataFrame(
        {
            "Column": df.columns,
            "Type": [str(dtype) for dtype in df.dtypes],
            "Unique values": [df[column].nunique() for column in df.columns],
            "Missing": [int(df[column].isna().sum()) for column in df.columns],
            "Role": [
                "Target" if column == config.TARGET
                else "Numeric feature" if column in config.NUMERIC_FEATURES
                else "Categorical feature"
                for column in df.columns
            ],
        }
    )
    st.dataframe(summary, hide_index=True)
    st.caption(
        "Zero missing values across every column is the intended outcome of cleaning — "
        "critical fields were dropped row-wise and meaningful missingness was recoded, so no "
        "model has to impute at prediction time."
    )

    with st.expander("First 200 prepared listings", icon=":material/table_rows:"):
        st.dataframe(df.head(200), hide_index=True)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
with figures_tab:
    # outputs/figures/ also holds per-model diagnostic exports (knn_*, rf_*,
    # xgb_*, ...) from the modelling notebook — those belong with the Models
    # and Diagnostics pages, which already render live equivalents of them.
    # This tab sticks to the numbered cleaning/transformation figures that
    # actually match what the rest of this page (Pipeline, Prepared dataset)
    # covers.
    saved_figures = (
        sorted(path for path in FIGURES_DIR.glob("*.png") if path.stem[:1].isdigit())
        if FIGURES_DIR.exists()
        else []
    )

    if not saved_figures:
        st.caption(
            "No figures found in `outputs/figures/`. Run the notebooks to export them."
        )
    else:
        chosen_figure = st.selectbox(
            "Figure",
            saved_figures,
            format_func=lambda path: path.stem.replace("_", " ").capitalize(),
        )
        st.image(str(chosen_figure), caption=chosen_figure.name, width="stretch")
        st.caption(
            f"{len(saved_figures)} cleaning/transformation figures exported by the notebooks "
            "into `outputs/figures/`."
        )
