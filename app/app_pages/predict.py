"""Deployment page — describe a listing, get a rent estimate from saved models."""

import altair as alt
import pandas as pd
import streamlit as st

from shared import MODEL_SPECS, PETS_OPTIONS, PHOTO_OPTIONS, TOP_AMENITIES, app_state, histogram_frame, model_key_for, money, money_md, note_metric, state_label, to_dense, transform
from components.property_viewer import render_property_viewer

state = app_state()
prepared_df, artifacts, scores, best = state["prepared_df"], state["artifacts"], state["scores"], state["best"]
best_key = model_key_for(best["Model"])
busiest = prepared_df["cityname"].value_counts().idxmax()
default_state = prepared_df.loc[prepared_df["cityname"] == busiest, "state"].mode()[0]

st.header("Estimate a rent", anchor=False)
form_column, viewer_column = st.columns([.92, 1.08], gap="large")
with form_column:
    with st.container(border=True):
        location_column, property_column = st.columns(2)
        with location_column:
            st.markdown("**Location**")
            states = sorted(prepared_df["state"].unique())
            selected_state = st.selectbox("State", states, index=states.index(default_state), format_func=state_label)
            cities = sorted(prepared_df.loc[prepared_df["state"] == selected_state, "cityname"].unique())
            cityname = st.selectbox("City", cities, index=cities.index(busiest) if busiest in cities else 0)
        with property_column:
            st.markdown("**Property**")
            square_feet = st.number_input("Square feet", min_value=100, max_value=5_000, value=850, step=25)
            bedrooms = st.number_input("Bedrooms", min_value=0.0, max_value=9.0, value=2.0, step=1.0)
            bathrooms = st.number_input("Bathrooms", min_value=1.0, max_value=9.0, value=1.0, step=0.5)
            photo_label = st.selectbox("Listing photo", list(PHOTO_OPTIONS))
        st.markdown("**Details**")
        selected_amenities = st.pills("Amenities", TOP_AMENITIES, selection_mode="multi", key="amenities")
        pets_label = st.segmented_control("Pets", list(PETS_OPTIONS), default="Not specified", key="pets")

city_rows = prepared_df[(prepared_df["state"] == selected_state) & (prepared_df["cityname"] == cityname)]
latitude, longitude = float(city_rows["latitude"].median()), float(city_rows["longitude"].median())
pets_allowed = PETS_OPTIONS.get(pets_label, "Not Specified")
listing = pd.DataFrame([{"bathrooms": bathrooms, "bedrooms": bedrooms, "square_feet": square_feet, "latitude": latitude, "longitude": longitude, "cityname": cityname, "state": selected_state, "amenities": ",".join(selected_amenities) if selected_amenities else "None", "pets_allowed": pets_allowed, "has_photo": PHOTO_OPTIONS[photo_label]}])
listing_transformed = transform(listing)
listing_scaled = to_dense(artifacts["preprocessor"].transform(listing_transformed))
predictions = {key: float(artifacts[key]["model"].predict(listing_scaled if spec["features"] == "scaled" else listing_transformed)[0]) for key, spec in MODEL_SPECS.items()}
headline, margin = predictions[best_key], float(best["Test MAE"])

with viewer_column:
    with st.container(border=True):
        st.markdown("**Property preview**")
        render_property_viewer(
            city=cityname,
            state=state_label(selected_state),
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            square_feet=square_feet,
            predicted_rent=headline,
            pets_allowed=pets_allowed,
        )

with st.container(border=True):
    answer_column, range_column = st.columns([2, 3], vertical_alignment="center")
    with answer_column:
        note_metric(f"Estimated monthly rent · {best['Model']}", money(headline), f"± {money_md(margin)} typical error")
    with range_column:
        st.markdown(
            f'Likely range: <span class="range-value">{money_md(headline - margin)} – {money_md(headline + margin)}</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"Based on {len(city_rows):,} listings in {cityname}.")

st.markdown("**What each model predicts for this listing**")
prediction_columns = st.columns(4, gap="medium")
for column, (key, spec) in zip(prediction_columns, MODEL_SPECS.items()):
    with column:
        with st.container(border=True):
            st.markdown(f"{spec['icon']} **{spec['label']}**")
            difference = predictions[key] - headline
            note_metric("Monthly rent", money(predictions[key]), "best model" if key == best_key else f"{difference:+,.0f} vs best", label_visibility="collapsed")
            st.caption(f"RMSE {money_md(scores[key]['rmse'])}")

values = list(predictions.values())
spread, relative_spread = max(values) - min(values), (max(values) - min(values)) / (sum(values) / len(values))
agreement = "High" if relative_spread < 0.10 else "Moderate" if relative_spread < 0.20 else "Low"
st.info(f"Model agreement: {agreement}", icon=":material/info:")

st.subheader("Local market", anchor=False, divider="gray")
comparable = city_rows[(city_rows["bedrooms"] == bedrooms) & city_rows["square_feet"].between(square_feet * 0.75, square_feet * 1.25)]
close_comparables = len(comparable) >= 20
reference = comparable if close_comparables else city_rows
reference_label = f"{len(comparable):,} comparable listings ({bedrooms:.0f}-bed, within 25% of {square_feet:,} sq ft) in {cityname}" if close_comparables else f"all {len(city_rows):,} listings in {cityname} — too few close comparables to isolate"
percentile = float((reference["price"] < headline).mean())
with st.container(horizontal=True):
    note_metric("Estimate", money(headline), best["Model"], border=True)
    note_metric("Median comparable rent", money(reference["price"].median()), "local market", border=True)
    note_metric("Position in the local market", f"{percentile * 100:.0f}th percentile", f"{headline - reference['price'].median():+,.0f} vs median", border=True)
    note_metric("Listings compared", f"{len(reference):,}", "close comparables" if close_comparables else "citywide", border=True)
with st.container(border=True):
    st.markdown("**Where the estimate falls in the local rent distribution**")
    distribution = histogram_frame(reference["price"], bins=40, label="Monthly rent (USD)")
    bars = alt.Chart(distribution).mark_bar(opacity=0.75).encode(x=alt.X("Monthly rent (USD):Q", title="Monthly rent (USD)"), y=alt.Y("Listings:Q", title="Comparable listings"), tooltip=["Monthly rent (USD)", "Listings"])
    marker = alt.Chart(pd.DataFrame({"Estimate": [headline]})).mark_rule(color="#e45756", size=3).encode(x="Estimate:Q", tooltip=[alt.Tooltip("Estimate:Q", format="$,.0f")])
    st.altair_chart((bars + marker).properties(height=300))
    st.caption(f"Red line: your estimate · {reference_label}.")
st.caption("Planning estimate only.")
