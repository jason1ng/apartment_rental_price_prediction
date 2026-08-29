"""
Diagnostics — where the models are wrong, and which features drive their predictions.

Two questions the leaderboard cannot answer on its own: is the error evenly spread
or concentrated in one part of the market, and which features is each model
actually using? Together they answer which property features have the greatest
impact on rental price.
"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from shared import (
    MODEL_SPECS,
    app_state,
    escape_dollars,
    model_key_for,
    money,
    money_md,
    note_metric,
    pipeline_importances,
    predicted_vs_actual_chart,
    ranked_bar_chart,
    residual_chart,
)

state = app_state()
artifacts = state["artifacts"]
scores = state["scores"]
y_test = state["y_test"]
best = state["best"]

st.header("Diagnostics and price drivers", anchor=False)
st.caption(
    "How the errors are distributed across the market, and what each model has learned "
    "matters most."
)

accuracy_tab, drivers_tab = st.tabs(["Prediction accuracy", "What drives rent"])


# ---------------------------------------------------------------------------
# Prediction accuracy
# ---------------------------------------------------------------------------
with accuracy_tab:
    diagnostic_label = st.segmented_control(
        "Model",
        [spec["label"] for spec in MODEL_SPECS.values()],
        default=best["Model"],
        key="diagnostic_model",
    )

    if diagnostic_label:
        diagnostic_key = model_key_for(diagnostic_label)
        predictions = np.asarray(scores[diagnostic_key]["predictions"])
        residuals = y_test - predictions

        with st.container(horizontal=True):
            note_metric(
                "Test RMSE", money(scores[diagnostic_key]["rmse"]), "squares each error",
                border=True,
            )
            note_metric(
                "Test MAE", money(scores[diagnostic_key]["mae"]), "typical listing", border=True
            )
            note_metric(
                "Test R²",
                f"{scores[diagnostic_key]['r2']:.3f}",
                "variance explained",
                border=True,
            )
            note_metric(
                "Median error",
                money(np.median(np.abs(residuals))),
                "half are closer",
                border=True,
            )
            note_metric(
                r"Within ±\$200",
                f"{(np.abs(residuals) <= 200).mean():.0%}",
                "of test listings",
                border=True,
            )

        st.caption(
            "Half of all test listings are predicted closer than the median error. RMSE sits "
            "above MAE because it squares each error first, so a handful of large misses on "
            "expensive listings weigh more than many small ones."
        )

        left, right = st.columns(2)
        with left:
            with st.container(border=True):
                st.markdown("**Predicted against actual rent**")
                st.altair_chart(predicted_vs_actual_chart(y_test, predictions))
                st.caption(
                    "Points on the dashed diagonal are exact predictions. Points below it are "
                    "under-predictions, above it over-predictions."
                )
        with right:
            with st.container(border=True):
                st.markdown("**Residuals against predicted rent**")
                st.altair_chart(residual_chart(y_test, predictions))
                st.caption(
                    "A residual cloud centred on zero with even width means the model is not "
                    "systematically too high or too low at any price level."
                )

        over_predicted = float((predictions > y_test).mean())
        st.caption(
            f"{diagnostic_label} over-predicts {over_predicted:.0%} of test listings — close to "
            "50% means no systematic bias in either direction. Both charts show a random sample "
            f"of {min(4_000, len(y_test)):,} of the {len(y_test):,} test listings, because Vega "
            "sends every plotted point to the browser."
        )

        # Error by price band: an average error hides that cheap and expensive
        # listings are not predicted equally well.
        bands = pd.DataFrame({"Actual": y_test, "Absolute error": np.abs(residuals)})
        edges = [0, 750, 1_000, 1_250, 1_500, 2_000, 3_000, np.inf]
        labels = [
            "under $750", "$750–1k", "$1k–1.25k", "$1.25k–1.5k",
            "$1.5k–2k", "$2k–3k", "over $3k",
        ]
        bands["Price band"] = pd.cut(bands["Actual"], bins=edges, labels=labels, right=False)
        band_summary = (
            bands.groupby("Price band", observed=True)
            .agg(**{
                "Mean absolute error": ("Absolute error", "mean"),
                "Listings": ("Absolute error", "size"),
            })
            .reset_index()
        )

        with st.container(border=True):
            st.markdown("**Where the error is concentrated**")
            st.altair_chart(
                alt.Chart(band_summary)
                .mark_bar()
                .encode(
                    x=alt.X("Price band:N", sort=labels, title="Actual rent"),
                    y=alt.Y("Mean absolute error:Q", title="Mean absolute error (USD)"),
                    color=alt.Color(
                        "Mean absolute error:Q", scale=alt.Scale(scheme="oranges"), legend=None
                    ),
                    tooltip=[
                        "Price band",
                        alt.Tooltip("Mean absolute error:Q", format="$,.0f"),
                        alt.Tooltip("Listings:Q", format=","),
                    ],
                )
                .properties(height=300)
            )
            worst_band = band_summary.loc[band_summary["Mean absolute error"].idxmax()]
            # The band labels carry their own literal "$", which the chart needs
            # raw but the caption below must escape.
            worst_label = escape_dollars(str(worst_band["Price band"]))
            st.caption(
                f"Error is not uniform across the market: it is largest for listings "
                f"{worst_label} "
                f"({money_md(worst_band['Mean absolute error'])} mean "
                f"absolute error across {int(worst_band['Listings']):,} listings). Expensive "
                "listings are both rarer in the training data and more variable in price, so the "
                "headline estimate is most trustworthy in the middle of the market."
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
                "Numeric inputs are standardised, so magnitudes are comparable across features. "
                "A positive coefficient raises predicted rent, holding everything else constant; "
                "a negative one lowers it."
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
            st.markdown('<div class="importance-transition"></div>', unsafe_allow_html=True)
            st.altair_chart(
                ranked_bar_chart(
                    importances.nlargest(12, "Importance"), "Importance", "Feature", "Importance"
                )
            )
            st.caption(
                "Importance is how much each feature reduced prediction error across the trees. "
                "Unlike a coefficient it has no direction — only strength."
            )

    top_tree_feature = importances.nlargest(1, "Importance").iloc[0]
    st.caption(
        f"**What drives rent:** `{top_tree_feature['Feature']}` carries "
        f"{top_tree_feature['Importance']:.0%} of the total importance in the tree model — "
        "location-derived features (the target-encoded city, latitude and longitude) dominate "
        "both views, ahead of size, and well ahead of bedrooms, bathrooms and every amenity. "
        "That matches the EDA, where rents varied more between states than across any property "
        "feature."
    )
