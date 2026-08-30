"""
Exploratory data analysis — what the listings say about rent, before any model.

Mirrors ``notebooks/01_exploratory_data_analysis.ipynb`` in the deployed app, so
the findings that justified the modelling decisions can be inspected
interactively instead of only as static figures. Every chart is aggregated in
pandas first: the browser receives tens of rows, never 98k.

Identifies which property features have the greatest impact on rental price from
the data alone, before any model is fitted.
"""

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from shared import (
    NUMERIC_EDA_COLS,
    RANDOM_STATE,
    SCATTER_SAMPLE,
    amenity_frame,
    app_state,
    correlation_long,
    heatmap_chart,
    histogram_frame,
    money,
    money_md,
    note_metric,
    ranked_bar_chart,
    raw_row_count,
    spread_chart,
    spread_frame,
    state_label,
)

state = app_state()
df = state["prepared_df"]
price = df["price"]

st.header("Exploratory data analysis", anchor=False)
st.caption(
    "What the listings themselves say about rental prices, before any model is involved."
)


# ---------------------------------------------------------------------------
# Dataset at a glance
# ---------------------------------------------------------------------------
with st.container(horizontal=True):
    st.metric("Prepared listings", f"{len(df):,}", border=True)
    st.metric("Median rent", money(price.median()), border=True)
    st.metric("Mean rent", money(price.mean()), border=True)
    st.metric("Cities", f"{df['cityname'].nunique():,}", border=True)
    st.metric("States", f"{df['state'].nunique():,}", border=True)
    st.metric("Missing values", f"{int(df.isna().sum().sum()):,}", border=True)

price_tab, location_tab, property_tab, listing_tab, correlation_tab = st.tabs(
    [
        "Rental price",
        "Location",
        "Property features",
        "Amenities and listing quality",
        "Correlations",
    ]
)


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------
with price_tab:
    st.subheader("How rents are distributed", anchor=False)

    skew = price.skew()
    with st.container(horizontal=True):
        note_metric("25th percentile", money(price.quantile(0.25)), "cheapest quarter", border=True)
        note_metric("Median", money(price.median()), "typical listing", border=True)
        note_metric("75th percentile", money(price.quantile(0.75)), "priciest quarter", border=True)
        note_metric(
            "Skewness",
            f"{skew:.2f}",
            "right-skewed" if skew > 0 else "left-skewed",
            border=True,
        )

    raw_column, log_column = st.columns(2)

    with raw_column:
        with st.container(border=True):
            st.markdown("**Monthly rent (USD)**")
            st.altair_chart(
                alt.Chart(histogram_frame(price, bins=60, label="Monthly rent (USD)"))
                .mark_bar()
                .encode(
                    x=alt.X("Monthly rent (USD):Q", title="Monthly rent (USD)"),
                    y=alt.Y("Listings:Q", title="Listings"),
                    tooltip=["Monthly rent (USD)", "Listings"],
                )
                .properties(height=320)
            )

    with log_column:
        with st.container(border=True):
            st.markdown("**Monthly rent, log scale**")
            st.altair_chart(
                alt.Chart(
                    histogram_frame(np.log10(price), bins=60, label="log10 monthly rent")
                )
                .mark_bar(color="#54a24b")
                .encode(
                    x=alt.X("log10 monthly rent:Q", title="log10(monthly rent)"),
                    y=alt.Y("Listings:Q", title="Listings"),
                    tooltip=["log10 monthly rent", "Listings"],
                )
                .properties(height=320)
            )

    st.caption(
        f"**Reading it:** rent is right-skewed (skewness {price.skew():.2f}) — most listings "
        f"cluster between {money_md(price.quantile(0.25))} and {money_md(price.quantile(0.75))}, "
        "with a long tail of expensive apartments pulling the mean above the median. On a log "
        "scale the distribution is close to symmetric, which is why RMSE (which squares errors) "
        "is dominated by the expensive tail while MAE stays representative of a typical listing."
    )

    with st.container(border=True):
        st.markdown("**Apartment size (square feet)**")
        st.altair_chart(
            alt.Chart(histogram_frame(df["square_feet"], bins=60, label="Square feet"))
            .mark_bar(color="#4c78a8")
            .encode(
                x=alt.X("Square feet:Q", title="Square feet"),
                y=alt.Y("Listings:Q", title="Listings"),
                tooltip=["Square feet", "Listings"],
            )
            .properties(height=300)
        )
        st.caption(
            "The spike at the right edge is the effect of outlier treatment: square footage was "
            "winsorized at the IQR upper bound, so mansion-sized entries and "
            "data-entry errors are capped rather than deleted, keeping the rest of those rows usable."
        )

    with st.expander("Cleaning funnel — raw file to prepared dataset", icon=":material/filter_alt:"):
        raw_rows = raw_row_count()
        with st.container(horizontal=True):
            note_metric(
                "Raw rows in the Kaggle CSV", f"{raw_rows:,}", "as downloaded", border=True
            )
            note_metric(
                "After cleaning and outlier treatment",
                f"{len(df):,}",
                f"{len(df) - raw_rows:,} rows",
                border=True,
            )
            note_metric(
                "Retained", f"{len(df) / raw_rows:.1%}", "of the raw file", border=True
            )
        st.caption(
            "Rows are lost to duplicate listing ids, non-apartment or non-monthly categories, "
            "missing critical fields (price, bedrooms, coordinates, city) and implausible prices. "
            "Retaining most of the file means the cleaning rules are targeted rather than blunt."
        )


# ---------------------------------------------------------------------------
# Geography
# ---------------------------------------------------------------------------
with location_tab:
    st.subheader("Location is the strongest single price driver", anchor=False)

    state_stats = spread_frame(df, "state")
    state_stats = state_stats[state_stats["Listings"] >= 100]
    state_stats["State"] = state_stats["state"].map(state_label)

    top_states = state_stats.nlargest(12, "Median")
    bottom_states = state_stats.nsmallest(12, "Median")

    expensive_column, cheap_column = st.columns(2)
    with expensive_column:
        with st.container(border=True):
            st.markdown("**Most expensive states**")
            st.altair_chart(spread_chart(top_states, "State", "Monthly rent (USD)"))
    with cheap_column:
        with st.container(border=True):
            st.markdown("**Least expensive states**")
            st.altair_chart(spread_chart(bottom_states, "State", "Monthly rent (USD)"))

    ratio = top_states["Median"].max() / bottom_states["Median"].min()
    st.caption(
        f"**Reading it:** the dot is the median rent, the bar the interquartile range. The most "
        f"expensive state's median rent is {ratio:.1f}× the cheapest — a wider gap than any "
        "property feature produces on its own. States with fewer than 100 listings are excluded "
        "so a handful of listings cannot dominate a median."
    )

    city_counts = df["cityname"].value_counts().head(20).index
    city_stats = spread_frame(df[df["cityname"].isin(city_counts)], "cityname")
    city_stats = city_stats.rename(columns={"cityname": "City"}).nlargest(15, "Median")

    with st.container(border=True):
        st.markdown("**Median rent in the 15 most expensive well-represented cities**")
        st.altair_chart(spread_chart(city_stats, "City", "Monthly rent (USD)"))
        st.caption(
            "Drawn from the 20 cities with the most listings, so every median rests on a large "
            "sample."
        )

    with st.container(border=True):
        st.markdown("**Where the listings are**")
        map_sample = df.sample(min(SCATTER_SAMPLE, len(df)), random_state=RANDOM_STATE)[
            ["latitude", "longitude", "price", "cityname", "state", "bedrooms", "bathrooms", "square_feet"]
        ].copy()
        # Bucketed rather than continuous: a colour ramp over a right-skewed
        # price column is unreadable — nearly every point would land at one end.
        quartiles = map_sample["price"].quantile([0.25, 0.5, 0.75]).tolist()
        colours = [[166, 206, 227, 190], [91, 155, 213, 190], [245, 133, 24, 190], [228, 87, 86, 190]]
        map_sample["colour"] = [
            colours[int(np.searchsorted(quartiles, value))] for value in map_sample["price"]
        ]
        map_sample["Rent"] = map_sample["price"].map(money)

        # show listing details on dot-hover
        listings_layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_sample,
            id="listings",
            get_position=["longitude", "latitude"],
            get_fill_color="colour",
            get_radius=18_000,
            radius_min_pixels=3,
            radius_max_pixels=9,
            pickable=True,
            auto_highlight=True,
        )
        view_state = pdk.ViewState(
            latitude=float(map_sample["latitude"].mean()),
            longitude=float(map_sample["longitude"].mean()),
            zoom=3,
        )
        event = st.pydeck_chart(
            pdk.Deck(
                layers=[listings_layer],
                initial_view_state=view_state,
                map_style=None,  # follow the app's (light-locked) Streamlit theme
                tooltip={
                    "html": "<b>{Rent}</b><br/>{cityname}, {state}<br/>{bedrooms} bed · {bathrooms} bath",
                },
            ),
            on_select="rerun",
            selection_mode="single-object",
            height=440,
        )

        selected = event.selection.objects.get("listings", [])
        if selected:
            listing = selected[0]
            detail_columns = st.columns(4)
            with detail_columns[0]:
                note_metric("Rent", money(listing["price"]), f"{listing['cityname']}, {listing['state']}", border=True)
            with detail_columns[1]:
                note_metric("Bedrooms", f"{listing['bedrooms']:.0f}", "beds", border=True)
            with detail_columns[2]:
                note_metric("Bathrooms", f"{listing['bathrooms']:.1f}", "baths", border=True)
            with detail_columns[3]:
                note_metric("Square feet", f"{listing['square_feet']:,.0f}", "sq ft", border=True)
        else:
            st.caption("Click a dot to see that listing's rent and details.")

        st.caption(
            f"A random sample of {len(map_sample):,} listings, coloured by rent quartile — blue "
            f"is the cheapest quarter (under {money_md(quartiles[0])}), red the most expensive "
            f"(over {money_md(quartiles[2])}). Expensive listings concentrate on the coasts and in "
            "major metros."
        )


# ---------------------------------------------------------------------------
# Property features
# ---------------------------------------------------------------------------
with property_tab:
    st.subheader("What the property itself contributes", anchor=False)

    size_column, bedroom_column = st.columns(2)

    with size_column:
        with st.container(border=True):
            st.markdown("**Rent against apartment size**")
            scatter_sample = df.sample(min(SCATTER_SAMPLE, len(df)), random_state=RANDOM_STATE)
            points = (
                alt.Chart(scatter_sample)
                .mark_circle(size=16, opacity=0.2)
                .encode(
                    x=alt.X("square_feet:Q", title="Square feet"),
                    y=alt.Y("price:Q", title="Monthly rent (USD)"),
                    tooltip=["square_feet", "price", "cityname", "state"],
                )
            )
            trend = points.transform_regression("square_feet", "price").mark_line(
                color="#e45756", size=2
            )
            st.altair_chart((points + trend).properties(height=340))
            correlation = df["square_feet"].corr(price)
            st.caption(
                f"Correlation r = {correlation:.2f}. Size clearly pushes rent up, but the vertical "
                "spread at any given size is enormous — a 900 sq ft flat rents anywhere from a few "
                "hundred to several thousand dollars. Size alone cannot explain price; location has "
                "to be in the model."
            )

    with bedroom_column:
        with st.container(border=True):
            st.markdown("**Rent by bedroom count**")
            bedroom_stats = spread_frame(df[df["bedrooms"] <= 5], "bedrooms")
            bedroom_stats["Bedrooms"] = bedroom_stats["bedrooms"].map(
                lambda n: "Studio" if n == 0 else f"{n:.0f} bed"
            )
            st.altair_chart(
                spread_chart(
                    bedroom_stats,
                    "Bedrooms",
                    "Monthly rent (USD)",
                    sort=alt.EncodingSortField("bedrooms", order="ascending"),
                ).properties(height=340)  # match "Rent against apartment size" in this row
            )
            st.caption(
                "Median rent rises with every bedroom added, but the interquartile ranges overlap "
                "heavily between adjacent counts — bedroom count is a real signal, not a decisive one."
            )

    configuration_column, size_by_bed_column = st.columns(2)

    with configuration_column:
        with st.container(border=True):
            st.markdown("**Which bed/bath configurations exist**")
            configurations = df[(df["bedrooms"] <= 4) & (df["bathrooms"] <= 3)]
            counts = (
                configurations.groupby(["bedrooms", "bathrooms"])
                .size()
                .reset_index(name="Listings")
            )
            counts["Bedrooms"] = counts["bedrooms"].map(lambda n: f"{n:.0f}")
            counts["Bathrooms"] = counts["bathrooms"].map(lambda n: f"{n:.1f}")
            st.altair_chart(
                heatmap_chart(counts, "Bathrooms", "Bedrooms", "Listings", "blues", ",.0f")
            )
            st.caption(
                "The market is concentrated in a handful of configurations — 1-bed/1-bath and "
                "2-bed/2-bath dominate. Rare configurations are exactly where the models have the "
                "least evidence and disagree most."
            )

    with size_by_bed_column:
        with st.container(border=True):
            st.markdown("**Apartment size by bedroom count**")
            size_stats = spread_frame(df[df["bedrooms"] <= 5], "bedrooms", value="square_feet")
            size_stats["Bedrooms"] = size_stats["bedrooms"].map(
                lambda n: "Studio" if n == 0 else f"{n:.0f} bed"
            )
            st.altair_chart(
                alt.Chart(size_stats)
                .mark_bar()
                .encode(
                    x=alt.X("Median:Q", title="Median square feet"),
                    y=alt.Y(
                        "Bedrooms:N",
                        sort=alt.EncodingSortField("bedrooms", order="ascending"),
                        title=None,
                    ),
                    tooltip=["Bedrooms", "Median", "Q1", "Q3", "Listings"],
                    color=alt.Color("Median:Q", scale=alt.Scale(scheme="blues"), legend=None),
                )
                .properties(height=320)  # match "Which bed/bath configurations exist" in this row
            )
            st.caption(
                "Size and bedroom count move together, which is why `squarefeet_per_room` was "
                "engineered during transformation — it separates a spacious 2-bed from a cramped "
                "one, "
                "something neither raw column expresses on its own."
            )


# ---------------------------------------------------------------------------
# Listing characteristics
# ---------------------------------------------------------------------------
with listing_tab:
    st.subheader("Amenities, pets and listing quality", anchor=False)

    amenities = amenity_frame(df)

    frequency_column, premium_column = st.columns(2)
    with frequency_column:
        with st.container(border=True):
            st.markdown("**How often each amenity is advertised**")
            st.altair_chart(
                ranked_bar_chart(
                    amenities, "Listings", "Amenity", "Listings advertising it", "blues"
                )
            )
    with premium_column:
        with st.container(border=True):
            st.markdown("**Median rent premium when advertised**")
            st.altair_chart(
                ranked_bar_chart(
                    amenities, "Rent premium", "Amenity", "Median rent difference (USD)"
                )
            )

    st.caption(
        "**Reading it:** the two rankings do not match. The most *common* amenities (parking, "
        "dishwasher) carry little premium because almost everyone advertises them, while scarcer "
        "amenities line up with much higher rents. That premium is association, not causation — a "
        "gym signals a newer, better-located building as much as it adds value itself."
    )

    with st.expander("Amenity table", icon=":material/table_chart:"):
        st.dataframe(
            amenities,
            hide_index=True,
            column_config={
                "Share of listings": st.column_config.ProgressColumn(
                    format="percent", min_value=0, max_value=1
                ),
                "Median rent with": st.column_config.NumberColumn(format="$%.0f"),
                "Median rent without": st.column_config.NumberColumn(format="$%.0f"),
                "Rent premium": st.column_config.NumberColumn(format="$%.0f"),
                "Listings": st.column_config.NumberColumn(format="%d"),
            },
        )

    count_column, pets_column, photo_column = st.columns(3)

    with count_column:
        with st.container(border=True, height="stretch"):
            st.markdown("**Rent by amenity count**")
            amenity_string = df["amenities"].fillna("None").astype(str)
            counted = df.assign(
                amenity_count=amenity_string.apply(
                    lambda value: 0 if value == "None" else len(value.split(","))
                )
            )
            counted = counted[counted["amenity_count"] <= 8]
            count_stats = spread_frame(counted, "amenity_count")
            count_stats["Amenities"] = count_stats["amenity_count"].map(lambda n: f"{n:.0f}")
            st.altair_chart(
                spread_chart(
                    count_stats,
                    "Amenities",
                    "Monthly rent (USD)",
                    sort=alt.EncodingSortField("amenity_count", order="ascending"),
                ).properties(height=270)
            )
            st.caption("Rent rises steadily with the number of amenities advertised.")

    with pets_column:
        with st.container(border=True, height="stretch"):
            st.markdown("**Rent by pet policy**")
            pet_stats = spread_frame(df, "pets_allowed").rename(
                columns={"pets_allowed": "Pet policy"}
            )
            st.altair_chart(
                spread_chart(pet_stats, "Pet policy", "Monthly rent (USD)").properties(height=270)
            )
            st.caption(
                "Pet policy barely moves rent — medians sit within a few hundred dollars of each "
                "other, aside from the small 126-listing 'dogs only' group."
            )

    with photo_column:
        with st.container(border=True, height="stretch"):
            st.markdown("**Rent by photo status**")
            photo_stats = spread_frame(df, "has_photo").rename(columns={"has_photo": "Photo"})
            st.altair_chart(
                spread_chart(photo_stats, "Photo", "Monthly rent (USD)").properties(height=270)
            )
            st.caption(
                "No-photo listings have the highest median rent, full-photo sits in the middle, "
                "and thumbnail-only is cheapest."
            )


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------
with correlation_tab:
    st.subheader("Correlation between numeric features", anchor=False)

    correlations = correlation_long(df, NUMERIC_EDA_COLS)

    matrix_column, ranking_column = st.columns([3, 2])

    with matrix_column:
        with st.container(border=True):
            st.markdown("**Pearson correlation matrix**")
            st.altair_chart(
                heatmap_chart(correlations, "Feature 2", "Feature 1", "r", "redblue")
            )

    with ranking_column:
        with st.container(border=True):
            st.markdown("**Correlation with price**")
            with_price = (
                correlations[
                    (correlations["Feature 1"] == "price")
                    & (correlations["Feature 2"] != "price")
                ]
                .rename(columns={"Feature 2": "Feature", "r": "Correlation with price"})
                .reset_index(drop=True)
            )
            st.altair_chart(
                ranked_bar_chart(
                    with_price, "Correlation with price", "Feature", "Pearson r", "redblue"
                ).properties(height=320)  # match the correlation matrix's fixed height
            )

    with st.expander("Descriptive statistics", icon=":material/functions:"):
        st.dataframe(
            df[NUMERIC_EDA_COLS].describe().T.rename(
                columns={"50%": "median", "25%": "Q1", "75%": "Q3"}
            ),
            column_config={
                column: st.column_config.NumberColumn(format="%.2f")
                for column in ["mean", "std", "min", "Q1", "median", "Q3", "max"]
            },
        )
