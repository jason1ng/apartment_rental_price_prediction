"""
Model comparison — how effective the four regressors are, and which one wins.

Every number here is measured on the same held-out test set, so the comparison is
like for like: no model is scored on a listing it was trained on.
"""

import altair as alt
import streamlit as st

from shared import (
    MODEL_SPECS,
    app_state,
    model_key_for,
    money,
    money_md,
    search_frame,
)

state = app_state()
artifacts = state["artifacts"]
scores = state["scores"]
leaderboard = state["leaderboard"]
best = state["best"]

st.header("Model comparison", anchor=False)
st.caption(
    f"Four regression models trained in `notebooks/03_modelling.ipynb`, all scored on the same "
    f"{len(state['y_test']):,} held-out listings. Lower RMSE and MAE are better; higher R² is better."
)


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
worst = leaderboard.iloc[-1]
improvement = (worst["Test RMSE"] - best["Test RMSE"]) / worst["Test RMSE"]

with st.container(border=True):
    st.markdown(
        f"### :material/trophy: {best['Model']} wins\n"
        f"It reaches **{money_md(best['Test RMSE'])} test RMSE** and explains "
        f"**{best['Test R²']:.1%}** of the variance in held-out rents, "
        f"{improvement:.0%} better than the weakest model ({worst['Model']}, "
        f"{money_md(worst['Test RMSE'])}). A typical prediction lands within "
        f"**{money_md(best['Test MAE'])}** of the true rent, so it is the model the "
        "prediction page uses for its headline estimate."
    )

model_columns = st.columns(4, gap="medium")
for column, (_, row) in zip(model_columns, leaderboard.iterrows()):
    with column:
        key = model_key_for(row["Model"])
        spec = MODEL_SPECS[key]
        with st.container(border=True):
            leader = row["Model"] == best["Model"]
            st.markdown(
                f"{spec['icon']} **{spec['label']}**"
                + ("  :green-badge[best]" if leader else "")
            )
            st.metric("Test RMSE", money(row["Test RMSE"]))
            with st.container(horizontal=True):
                st.metric("MAE", money(row["Test MAE"]))
                st.metric("R²", f"{row['Test R²']:.3f}")


# ---------------------------------------------------------------------------
# Metric comparison
# ---------------------------------------------------------------------------
st.subheader("Compare on one metric at a time", anchor=False, divider="gray")

metric_choice = st.segmented_control(
    "Metric", ["Test RMSE", "Test MAE", "Test R²"], default="Test RMSE", key="comparison_metric"
)

if metric_choice:
    lower_is_better = metric_choice != "Test R²"
    with st.container(border=True):
        st.altair_chart(
            alt.Chart(leaderboard)
            .mark_bar()
            .encode(
                x=alt.X(f"{metric_choice}:Q", title=metric_choice),
                y=alt.Y("Model:N", sort="x" if lower_is_better else "-x", title=None),
                color=alt.condition(
                    alt.datum.Model == best["Model"],
                    alt.value("#54a24b"),
                    alt.value("#a9b6c4"),
                ),
                tooltip=["Model", metric_choice],
            )
            .properties(height=220)
        )
        st.caption(
            "RMSE squares each error before averaging, so it punishes the large misses on "
            "expensive listings; MAE is the error on a typical listing. Both are in dollars per "
            "month and directly comparable to the rents themselves."
            if lower_is_better
            else "R² is the share of test-set price variance the model explains. 1.0 would be a "
            "perfect fit; 0.0 would be no better than always predicting the mean rent."
        )

st.dataframe(
    leaderboard,
    hide_index=True,
    column_config={
        "Test RMSE": st.column_config.NumberColumn(
            format="$%.0f", help="Root mean squared error on the held-out test set"
        ),
        "Test MAE": st.column_config.NumberColumn(
            format="$%.0f", help="Mean absolute error — the error on a typical listing"
        ),
        "Test R²": st.column_config.NumberColumn(
            format="%.4f", help="Share of test-set price variance explained"
        ),
        "CV RMSE": st.column_config.NumberColumn(
            format="$%.0f", help="Best cross-validated RMSE reached during tuning"
        ),
    },
)

gaps = leaderboard.dropna(subset=["CV RMSE"])
if not gaps.empty:
    overfit = gaps.assign(gap=gaps["Test RMSE"] - gaps["CV RMSE"])
    st.caption(
        "**CV RMSE against test RMSE** is the reliability check: cross-validated error is measured "
        "on training folds, test error on listings held out entirely. The two staying close means "
        "the tuning did not overfit the training data. The largest gap here is "
        f"{money_md(overfit['gap'].abs().max())}."
    )


# ---------------------------------------------------------------------------
# Hyperparameter tuning evidence
# ---------------------------------------------------------------------------
st.subheader("Hyperparameter tuning", anchor=False, divider="gray")

tuned = {key: spec for key, spec in MODEL_SPECS.items() if artifacts[key]["cv_results"] is not None}
tuned_label = st.segmented_control(
    "Tuned model",
    [spec["label"] for spec in tuned.values()],
    default=MODEL_SPECS["rf"]["label"],
    key="tuning_model",
)

if tuned_label:
    tuned_key = model_key_for(tuned_label)
    search = search_frame(artifacts[tuned_key]["cv_results"])
    parameters = [
        column for column in search.columns if column not in {"CV RMSE", "Mean fit time (s)"}
    ]

    st.caption(
        f"{len(search)} parameter combinations, each scored with 5-fold cross-validation "
        f"({len(search) * 5} model fits). Best combination: "
        f"`{artifacts[tuned_key]['best_params']}` at "
        f"{money_md(artifacts[tuned_key]['cv_rmse'])} CV RMSE."
    )

    focus = st.segmented_control(
        "Compare by", parameters, default=parameters[0], key=f"tuning_focus_{tuned_key}"
    )
    if focus:
        # Best score reached per value, not the mean: averaging would hide the
        # combinations that actually won behind the ones that were paired badly.
        best_per_value = (
            search.groupby(focus, as_index=False)["CV RMSE"].min().sort_values("CV RMSE")
        )
        with st.container(border=True):
            st.altair_chart(
                alt.Chart(best_per_value)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "CV RMSE:Q",
                        title="Best cross-validated RMSE (USD)",
                        scale=alt.Scale(zero=False),
                    ),
                    y=alt.Y(f"{focus}:N", sort="x", title=focus),
                    color=alt.Color("CV RMSE:Q", scale=alt.Scale(scheme="greens", reverse=True), legend=None),
                    tooltip=[focus, alt.Tooltip("CV RMSE:Q", format="$,.0f")],
                )
                .properties(height=max(160, 34 * len(best_per_value)))
            )
            st.caption(
                f"Each bar is the best CV RMSE any combination reached at that value of "
                f"`{focus}`, so a flat chart means the parameter barely matters and a sloped one "
                "means it does."
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
