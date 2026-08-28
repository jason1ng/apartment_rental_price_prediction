"""
Deployment page — describe a listing, get a rent estimate from every saved model.

This is the deliverable of the project's final objective: a functional prototype
that lets a user estimate a rental price interactively. It is the landing page,
so the prediction tool is the first thing anyone sees.

Predicting a single row costs milliseconds even for the random forest, so the
estimate is recomputed live on every input change rather than hidden behind a
submit button — no form, so the city list also refreshes the moment the state
changes.
"""

import altair as alt
import pandas as pd
import streamlit as st

from shared import (
    MODEL_SPECS,
    PETS_OPTIONS,
    PHOTO_OPTIONS,
    TOP_AMENITIES,
    app_state,
    histogram_frame,
    model_key_for,
    money,
    money_md,
    note_metric,
    state_label,
    to_dense,
    transform,
)

state = app_state()
prepared_df = state["prepared_df"]
artifacts = state["artifacts"]
scores = state["scores"]
best = state["best"]
best_key = model_key_for(best["Model"])

# Defaults land on the best-represented city rather than whatever sorts first
# alphabetically, so the page opens on an example the models actually agree on.
busiest = prepared_df["cityname"].value_counts().idxmax()
default_state = prepared_df.loc[prepared_df["cityname"] == busiest, "state"].mode()[0]

st.header("Estimate a rent", anchor=False)
st.caption(
    f"Describe a listing and all four trained models price it. **{best['Model']}** is the "
    f"most accurate on the held-out test set (±{money_md(best['Test MAE'])} typical error), so "
    "its figure is the headline. The estimate updates as you change any input."
)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
with st.container(border=True):
    location_column, property_column = st.columns(2)

    with location_column:
        st.markdown("**Location** &nbsp; :grey-badge[strongest price driver]")
        states = sorted(prepared_df["state"].unique())
        selected_state = st.selectbox(
            "State", states, index=states.index(default_state), format_func=state_label
        )
        cities = sorted(
            prepared_df.loc[prepared_df["state"] == selected_state, "cityname"].unique()
        )
        cityname = st.selectbox(
            "City", cities, index=cities.index(busiest) if busiest in cities else 0
        )

    with property_column:
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

    st.markdown("**Amenities and pet policy**")
    selected_amenities = st.pills(
        "Amenities advertised", TOP_AMENITIES, selection_mode="multi", key="amenities"
    )
    pets_label = st.segmented_control(
        "Pets allowed", list(PETS_OPTIONS), default="Not specified", key="pets"
    )

# Latitude and longitude are model inputs, so they are filled from the median
# listing in the chosen city rather than asked for — someone pricing a flat in
# Austin should not have to look its coordinates up.
city_rows = prepared_df[
    (prepared_df["state"] == selected_state) & (prepared_df["cityname"] == cityname)
]
latitude = float(city_rows["latitude"].median())
longitude = float(city_rows["longitude"].median())


# ---------------------------------------------------------------------------
# Prediction — exactly the transformation the models were trained on
# ---------------------------------------------------------------------------
listing = pd.DataFrame(
    [
        {
            "bathrooms": bathrooms,
            "bedrooms": bedrooms,
            "square_feet": square_feet,
            "latitude": latitude,
            "longitude": longitude,
            "cityname": cityname,
            "state": selected_state,
            "amenities": ",".join(selected_amenities) if selected_amenities else "None",
            "pets_allowed": PETS_OPTIONS.get(pets_label, "Not Specified"),
            "has_photo": PHOTO_OPTIONS[photo_label],
        }
    ]
)

listing_transformed = transform(listing)
listing_scaled = to_dense(artifacts["preprocessor"].transform(listing_transformed))

predictions = {}
for key, spec in MODEL_SPECS.items():
    features = listing_scaled if spec["features"] == "scaled" else listing_transformed
    predictions[key] = float(artifacts[key]["model"].predict(features)[0])

headline = predictions[best_key]
margin = float(best["Test MAE"])


# ---------------------------------------------------------------------------
# Headline answer
# ---------------------------------------------------------------------------
with st.container(border=True):
    answer_column, range_column = st.columns([2, 3], vertical_alignment="center")

    with answer_column:
        note_metric(
            f"Estimated monthly rent · {best['Model']}",
            money(headline),
            f"± {money_md(margin)} typical error",
        )
    with range_column:
        st.markdown(
            f"A realistic asking range for this listing is "
            f"**{money_md(headline - margin)} – {money_md(headline + margin)}** per month. "
            f"The margin is {best['Model']}'s mean absolute error across "
            f"{len(state['y_test']):,} listings it was never trained on."
        )
        st.caption(
            f"{len(city_rows):,} listings in {cityname}, {state_label(selected_state)} back "
            f"this estimate · coordinates auto-filled from the city median "
            f"({latitude:.4f}, {longitude:.4f})."
        )

st.markdown("**What each model predicts for this listing**")
with st.container(horizontal=True):
    for key, spec in MODEL_SPECS.items():
        with st.container(border=True, width=300):
            st.markdown(f"{spec['icon']} **{spec['label']}**")
            difference = predictions[key] - headline
            note_metric(
                "Monthly rent",
                money(predictions[key]),
                "best model" if key == best_key else f"{difference:+,.0f} vs best",
                label_visibility="collapsed",
            )
            st.caption(
                f"Test RMSE {money_md(scores[key]['rmse'])} · R² {scores[key]['r2']:.3f}"
            )

values = list(predictions.values())
spread = max(values) - min(values)
relative_spread = spread / (sum(values) / len(values))

if relative_spread < 0.10:
    st.success(
        f"The four models agree closely — {money_md(spread)} apart, {relative_spread:.0%} of the "
        "average. This listing is well covered by the training data.",
        icon=":material/check_circle:",
    )
elif relative_spread < 0.20:
    st.warning(
        f"The models disagree moderately — {money_md(spread)} apart, {relative_spread:.0%} of the "
        "average. Treat the estimate as a range rather than a single number.",
        icon=":material/warning:",
    )
else:
    st.error(
        f"The models disagree strongly — {money_md(spread)} apart, {relative_spread:.0%} of the "
        "average. That usually means this listing is unlike anything in the training data; "
        "check that the size and bedroom count are realistic together.",
        icon=":material/priority_high:",
    )


# ---------------------------------------------------------------------------
# Market context — an estimate only means something against the local market
# ---------------------------------------------------------------------------
st.subheader("How this compares to the local market", anchor=False, divider="gray")

comparable = city_rows[
    (city_rows["bedrooms"] == bedrooms)
    & (city_rows["square_feet"].between(square_feet * 0.75, square_feet * 1.25))
]
close_comparables = len(comparable) >= 20
reference = comparable if close_comparables else city_rows
reference_label = (
    f"{len(comparable):,} comparable listings ({bedrooms:.0f}-bed, within 25% of "
    f"{square_feet:,} sq ft) in {cityname}"
    if close_comparables
    else f"all {len(city_rows):,} listings in {cityname} — too few close comparables to isolate"
)

percentile = float((reference["price"] < headline).mean())

with st.container(horizontal=True):
    note_metric("Estimate", money(headline), best["Model"], border=True)
    note_metric(
        "Median comparable rent",
        money(reference["price"].median()),
        "local market",
        border=True,
    )
    note_metric(
        "Position in the local market",
        f"{percentile * 100:.0f}th percentile",
        f"{headline - reference['price'].median():+,.0f} vs median",
        border=True,
    )
    note_metric(
        "Listings compared",
        f"{len(reference):,}",
        "close comparables" if close_comparables else "citywide",
        border=True,
    )

with st.container(border=True):
    st.markdown("**Where the estimate falls in the local rent distribution**")
    distribution = histogram_frame(reference["price"], bins=40, label="Monthly rent (USD)")
    bars = (
        alt.Chart(distribution)
        .mark_bar(opacity=0.75)
        .encode(
            x=alt.X("Monthly rent (USD):Q", title="Monthly rent (USD)"),
            y=alt.Y("Listings:Q", title="Comparable listings"),
            tooltip=["Monthly rent (USD)", "Listings"],
        )
    )
    marker = (
        alt.Chart(pd.DataFrame({"Estimate": [headline]}))
        .mark_rule(color="#e45756", size=3)
        .encode(x="Estimate:Q", tooltip=[alt.Tooltip("Estimate:Q", format="$,.0f")])
    )
    st.altair_chart((bars + marker).properties(height=300))
    st.caption(
        f"Red line: this listing's estimate. Bars: {reference_label}. An estimate near the "
        "middle of the distribution is well supported by real listings; one far into either "
        "tail is extrapolation and should be treated with caution."
    )

st.caption(
    "This is a model estimate from listed features only. It does not account for negotiation, "
    "seasonality, building condition, or an individual landlord's pricing strategy, and it is "
    "not a valuation."
)
