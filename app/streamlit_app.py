# DRAFT ONLY (gen by AI) — this is a prototype for the compulsory deliverable of BMDS2003 Data Science, not a production-ready application.

"""
Apartment Rental Price Prediction — Streamlit App

Features:
- Model performance comparison (Linear Regression, KNN, and Random Forest)
- Visualizations from model evaluation
- Single predictor input form showing predictions from all models
"""

import sys
from pathlib import Path

import kagglehub
from kagglehub import KaggleDatasetAdapter
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# allow "from src.xxx import yyy" when running from repo root
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import config
from src.data_cleaning import clean
from src.data_transformation import build_preprocessor, transform, get_engineered_feature_lists
from src.final_dataset_output import build_prepared_dataset
from src.modelling import evaluate_model, to_dense, train_knn, train_linear_regression

st.set_page_config(page_title="Apartment Rent Predictor", page_icon="🏠", layout="wide")


@st.cache_resource(show_spinner="Loading data and training models (first load only)...")
def get_trained_models():
    """Load data, build features, and train Linear Regression, KNN, and RF."""
    # Load and prepare data
    prepared_df = build_prepared_dataset(verbose=False)

    # Split BEFORE any transformation (same as notebook)
    X = prepared_df.drop(columns=["price"])
    y = prepared_df["price"]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_transformed = transform(X_train)
    X_test_transformed = transform(X_test)

    numeric, categorical, city = get_engineered_feature_lists(X_train_transformed)

    # --- KNN (uses scaled features) ---
    preprocessor_knn = build_preprocessor(numeric, categorical, city, scale_numeric=True)
    XX_train = preprocessor_knn.fit_transform(X_train_transformed, y_train)
    XX_test = preprocessor_knn.transform(X_test_transformed)
    XX_train_knn = to_dense(XX_train)
    XX_test_knn = to_dense(XX_test)

    # --- Linear Regression baseline (uses the same scaled features as KNN) ---
    linear_estimator = train_linear_regression(XX_train_knn, y_train)
    linear_metrics = evaluate_model(linear_estimator, XX_test_knn, y_test)

    knn_grid = train_knn(XX_train_knn, y_train, cv=5)
    knn_metrics = evaluate_model(knn_grid.best_estimator_, XX_test_knn, y_test)

    # --- Random Forest (uses unscaled features) ---
    rf_pipeline = Pipeline([
        ("preprocess", build_preprocessor(numeric, categorical, city, scale_numeric=False)),
        ("regressor", RandomForestRegressor(random_state=42, n_jobs=-1)),
    ])
    rf_param_grid = {
        "regressor__n_estimators": [100, 200, 300],
        "regressor__max_depth": [10, 15, 20, None],
        "regressor__min_samples_leaf": [1, 2, 4],
    }
    from sklearn.model_selection import GridSearchCV
    rf_grid = GridSearchCV(
        rf_pipeline, param_grid=rf_param_grid,
        scoring="neg_root_mean_squared_error", cv=5, n_jobs=-1, verbose=0,
    )
    rf_grid.fit(X_train_transformed, y_train)
    rf_metrics = evaluate_model(rf_grid.best_estimator_, X_test_transformed, y_test)

    # Also load raw data for UI dropdowns
    raw = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        config.KAGGLE_DATASET,
        "",
    )
    df_clean = clean(raw, verbose=False)

    return {
        "linear": {
            "model": linear_estimator,
            "preprocessor": preprocessor_knn,
            "metrics": linear_metrics,
        },
        "knn": {
            "model": knn_grid.best_estimator_,
            "preprocessor": preprocessor_knn,
            "metrics": knn_metrics,
            "best_params": knn_grid.best_params_,
            "cv_rmse": -knn_grid.best_score_,
        },
        "rf": {
            "model": rf_grid.best_estimator_,
            "preprocessor": None,  # RF pipeline includes its own
            "metrics": rf_metrics,
            "best_params": rf_grid.best_params_,
            "cv_rmse": -rf_grid.best_score_,
        },
        "df_clean": df_clean,
        "prepared_df": prepared_df,
        "X_train_transformed": X_train_transformed,
        "X_test_transformed": X_test_transformed,
        "y_test": y_test,
    }


# Load models and data
models = get_trained_models()
linear_model = models["linear"]
knn_model = models["knn"]
rf_model = models["rf"]
df_clean = models["df_clean"]
prepared_df = models["prepared_df"]

# Page title
st.title("🏠 Apartment Rental Price Predictor")
st.caption(
    "Prototype for BMDS2003 Data Science — predictions are estimates based on "
    "listed features only and do not account for negotiation, seasonality, "
    "or individual landlord pricing strategy."
)

# ============================================================================
# SIDEBAR: Model Performance Comparison
# ============================================================================
with st.sidebar:
    st.header("📊 Model Performance")

    # Linear Regression metrics
    st.subheader("Linear Regression")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("RMSE", f"${linear_model['metrics']['rmse']:,.0f}")
        st.metric("MAE", f"${linear_model['metrics']['mae']:,.0f}")
    with col2:
        st.metric("R²", f"{linear_model['metrics']['r2']:.4f}")

    st.divider()

    # KNN Metrics
    st.subheader("K-Nearest Neighbors")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("RMSE", f"${knn_model['metrics']['rmse']:,.0f}")
        st.metric("MAE", f"${knn_model['metrics']['mae']:,.0f}")
    with col2:
        st.metric("R²", f"{knn_model['metrics']['r2']:.4f}")
        st.metric("CV RMSE", f"${knn_model['cv_rmse']:,.0f}")

    st.caption(f"Best params: {knn_model['best_params']}")

    st.divider()

    # RF Metrics
    st.subheader("Random Forest")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("RMSE", f"${rf_model['metrics']['rmse']:,.0f}")
        st.metric("MAE", f"${rf_model['metrics']['mae']:,.0f}")
    with col2:
        st.metric("R²", f"{rf_model['metrics']['r2']:.4f}")
        st.metric("CV RMSE", f"${rf_model['cv_rmse']:,.0f}")

    st.caption(f"Best params: {rf_model['best_params']}")

    st.divider()

    # Quick comparison
    st.subheader("🏆 Comparison")
    if rf_model['metrics']['rmse'] < knn_model['metrics']['rmse']:
        st.success("**Random Forest** outperforms KNN on test RMSE")
    else:
        st.success("**KNN** outperforms Random Forest on test RMSE")

    st.caption("Lower RMSE/MAE = better. Higher R² = better.")


# ============================================================================
# MAIN: EDA AND LINEAR REGRESSION INSIGHTS
# ============================================================================
st.header("Data exploration and Linear Regression insights")
st.caption(
    "These charts are generated from the prepared dataset and the held-out test set, "
    "so they help explain both the data and the baseline model."
)

eda_tab, diagnostics_tab, coefficients_tab = st.tabs([
    "Explore the data", "Linear Regression diagnostics", "Linear Regression coefficients"
])

with eda_tab:
    total_missing = int(prepared_df.isna().sum().sum())
    median_price = prepared_df["price"].median()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Prepared listings", f"{len(prepared_df):,}")
    col2.metric("Features used", prepared_df.shape[1] - 1)
    col3.metric("Missing values", f"{total_missing:,}")
    col4.metric("Median monthly rent", f"${median_price:,.0f}")

    price_col, size_col = st.columns(2)
    with price_col:
        price_chart = px.histogram(
            prepared_df, x="price", nbins=50,
            labels={"price": "Monthly rent (USD)"},
            title="Distribution of monthly rent",
            color_discrete_sequence=["#4C78A8"],
        )
        price_chart.update_layout(showlegend=False, margin=dict(t=50, l=0, r=0, b=0))
        st.plotly_chart(price_chart, use_container_width=True)

    with size_col:
        scatter_sample = prepared_df.sample(min(5_000, len(prepared_df)), random_state=42)
        size_chart = px.scatter(
            scatter_sample, x="square_feet", y="price", opacity=0.35,
            labels={"square_feet": "Square feet", "price": "Monthly rent (USD)"},
            title="Rent versus apartment size (5,000-listing sample)",
            color_discrete_sequence=["#F58518"],
        )
        size_chart.update_layout(showlegend=False, margin=dict(t=50, l=0, r=0, b=0))
        st.plotly_chart(size_chart, use_container_width=True)

    st.caption(
        "A wide spread at the same apartment size indicates that size alone cannot "
        "explain rent; location and listing characteristics also matter."
    )

with diagnostics_tab:
    actual = np.asarray(models["y_test"])
    predicted = np.asarray(linear_model["metrics"]["predictions"])
    diagnostics_df = pd.DataFrame({
        "Actual rent": actual,
        "Predicted rent": predicted,
        "Residual": actual - predicted,
    })
    lower_bound = float(min(actual.min(), predicted.min()))
    upper_bound = float(max(actual.max(), predicted.max()))

    actual_col, residual_col = st.columns(2)
    with actual_col:
        actual_chart = px.scatter(
            diagnostics_df, x="Actual rent", y="Predicted rent", opacity=0.45,
            title="Actual versus predicted rent",
            color_discrete_sequence=["#54A24B"],
        )
        actual_chart.add_shape(
            type="line", x0=lower_bound, y0=lower_bound, x1=upper_bound, y1=upper_bound,
            line=dict(color="#444", dash="dash"),
        )
        actual_chart.update_layout(showlegend=False, margin=dict(t=50, l=0, r=0, b=0))
        st.plotly_chart(actual_chart, use_container_width=True)

    with residual_col:
        residual_chart = px.scatter(
            diagnostics_df, x="Predicted rent", y="Residual", opacity=0.45,
            title="Residuals versus predicted rent",
            color_discrete_sequence=["#E45756"],
        )
        residual_chart.add_hline(y=0, line_dash="dash", line_color="#444")
        residual_chart.update_layout(showlegend=False, margin=dict(t=50, l=0, r=0, b=0))
        st.plotly_chart(residual_chart, use_container_width=True)

    st.info(
        "Points nearest the dashed line are the most accurate predictions. In the residual plot, "
        "a random cloud around zero suggests the model is not consistently over- or under-predicting."
    )

with coefficients_tab:
    feature_names = linear_model["preprocessor"].get_feature_names_out()
    coefficient_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": linear_model["model"].coef_,
    })
    coefficient_df["Absolute coefficient"] = coefficient_df["Coefficient"].abs()
    top_coefficients = coefficient_df.nlargest(12, "Absolute coefficient").sort_values("Coefficient")
    coefficient_chart = px.bar(
        top_coefficients, x="Coefficient", y="Feature", orientation="h",
        color="Coefficient", color_continuous_scale="RdBu",
        title="12 strongest standardized Linear Regression coefficients",
    )
    coefficient_chart.update_layout(coloraxis_showscale=False, margin=dict(t=50, l=0, r=0, b=0))
    st.plotly_chart(coefficient_chart, use_container_width=True)
    st.caption(
        "Positive coefficients raise the predicted rent and negative coefficients lower it, "
        "holding the other model inputs constant. Numeric inputs are standardized, making their "
        "coefficient magnitudes easier to compare."
    )


# ============================================================================
# MAIN: MODEL EVALUATION VISUALIZATIONS
# ============================================================================
st.header("📈 Model Evaluation Visualizations")

viz_tabs = st.tabs([
    "Linear Regression: Predicted vs Actual",
    "KNN: k vs CV RMSE",
    "KNN: Predicted vs Actual",
    "RF: Hyperparameter Search",
    "RF: Predicted vs Actual",
])

with viz_tabs[0]:
    st.image("outputs/figures/linear_regression_pred_vs_actual.png", caption="Linear Regression predicted vs actual price on the test set")

with viz_tabs[1]:
    st.image("outputs/figures/knn_k_search.png", caption="KNN Hyperparameter Search — 5-fold CV RMSE by k and weighting")

with viz_tabs[2]:
    st.image("outputs/figures/knn_pred_vs_actual.png", caption="KNN — Predicted vs Actual Price on Test Set")

with viz_tabs[3]:
    st.image("outputs/figures/rf_hyperparameter_search.png", caption="Random Forest Hyperparameter Search — 5-fold CV RMSE")

with viz_tabs[4]:
    st.image("outputs/figures/rf_pred_vs_actual.png", caption="Random Forest — Predicted vs Actual Price on Test Set")


# ============================================================================
# MAIN: Predictor Input (Single input for all models)
# ============================================================================
st.header("🔮 Rent Price Prediction")
st.markdown("Enter apartment details **once** to get predictions from all models.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Numeric Features")
    bathrooms = st.number_input("Bathrooms", min_value=1.0, max_value=9.0, value=1.0, step=0.5)
    bedrooms = st.number_input("Bedrooms", min_value=0.0, max_value=9.0, value=2.0, step=1.0)
    square_feet = st.number_input("Square feet", min_value=100, max_value=5000, value=850)

with col2:
    st.subheader("Categorical Features")
    state = st.selectbox("State", sorted(df_clean["state"].unique()))
    cities_in_state = sorted(df_clean.loc[df_clean["state"] == state, "cityname"].unique())
    cityname = st.selectbox("City", cities_in_state)

pets_allowed = st.selectbox("Pets allowed", sorted(df_clean["pets_allowed"].unique()))
has_photo = st.selectbox("Has photo", sorted(df_clean["has_photo"].unique()))
amenities = st.selectbox(
    "Amenities profile (pick a common combination from the data, or 'None')",
    sorted(df_clean["amenities"].value_counts().head(20).index.tolist()) + ["None"],
)

# lat/long: use the median for the chosen city as a stand-in
city_rows = df_clean[(df_clean["state"] == state) & (df_clean["cityname"] == cityname)]
latitude = city_rows["latitude"].median()
longitude = city_rows["longitude"].median()

# Show lat/long being used
with st.expander("📍 Coordinates used (auto-filled from city)"):
    st.write(f"Latitude: {latitude:.4f}")
    st.write(f"Longitude: {longitude:.4f}")


# ============================================================================
# PREDICTION BUTTON - Single input, both models
# ============================================================================
if st.button("Predict Rental Price", type="primary", use_container_width=True):
    # Build input dataframe
    input_row = pd.DataFrame([{
        "bathrooms": bathrooms,
        "bedrooms": bedrooms,
        "square_feet": square_feet,
        "latitude": latitude,
        "longitude": longitude,
        "cityname": cityname,
        "state": state,
        "amenities": amenities,
        "pets_allowed": pets_allowed,
        "has_photo": has_photo,
    }])

    # Apply same transformation as training
    input_transformed = transform(input_row)

    # --- Linear Regression and KNN predictions ---
    # Both use the same scaled preprocessor.
    XX_input = knn_model["preprocessor"].transform(input_transformed)
    XX_input_dense = to_dense(XX_input)
    linear_pred = linear_model["model"].predict(XX_input_dense)[0]
    knn_pred = knn_model["model"].predict(XX_input_dense)[0]

    # --- RF Prediction ---
    # RF pipeline includes its own preprocessor
    rf_pred = rf_model["model"].predict(input_transformed)[0]

    # Display results side by side
    st.subheader("Predictions")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Linear Regression")
        st.metric("Estimated Monthly Rent", f"${linear_pred:,.0f}")
        st.caption(f"Baseline model | Test RMSE: ${linear_model['metrics']['rmse']:,.0f}")

    with col2:
        st.markdown("### 🤖 K-Nearest Neighbors")
        st.metric("Estimated Monthly Rent", f"${knn_pred:,.0f}")
        st.caption(
            f"Model: k={knn_model['best_params']['n_neighbors']}, "
            f"weights={knn_model['best_params']['weights']} | "
            f"Test RMSE: ${knn_model['metrics']['rmse']:,.0f}"
        )

    with col3:
        st.markdown("### 🌳 Random Forest")
        st.metric("Estimated Monthly Rent", f"${rf_pred:,.0f}")
        st.caption(
            f"Model: n_estimators={rf_model['best_params']['regressor__n_estimators']}, "
            f"max_depth={rf_model['best_params']['regressor__max_depth']}, "
            f"min_samples_leaf={rf_model['best_params']['regressor__min_samples_leaf']} | "
            f"Test RMSE: ${rf_model['metrics']['rmse']:,.0f}"
        )

    # Comparison
    st.divider()
    st.subheader("📊 Comparison")
    predictions = {
        "Linear Regression": linear_pred,
        "KNN": knn_pred,
        "Random Forest": rf_pred,
    }
    prediction_range = max(predictions.values()) - min(predictions.values())
    average_prediction = sum(predictions.values()) / len(predictions)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Lowest prediction", f"${min(predictions.values()):,.0f}")
    with col2:
        st.metric("Highest prediction", f"${max(predictions.values()):,.0f}")
    with col3:
        st.metric("Model range", f"${prediction_range:,.0f}", delta=f"{prediction_range/average_prediction*100:.1f}% of avg")

    if prediction_range / average_prediction < 0.1:
        st.success("✅ Models agree closely (difference < 10%)")
    elif prediction_range / average_prediction < 0.2:
        st.warning("⚠️ Models moderately disagree (difference 10-20%)")
    else:
        st.error("❌ Models significantly disagree (difference > 20%)")

    st.caption(
        "This is a model estimate, not a valuation. It reflects patterns in "
        "listed features from the training data only. Random Forest typically "
        "performs better on this dataset (lower RMSE, higher R²)."
    )


# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.caption(
    "BMDS2003 Data Science — Compulsory Deliverable | "
    "Data source: shashanks1202/apartment-rent-data (Kaggle) | "
    "Models trained on ~99k listings, 80/20 train/test split"
)
