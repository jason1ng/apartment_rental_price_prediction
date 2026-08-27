"""
Section 4 Modelling — shared training and evaluation helpers.

Every model in Section 4 is scored the same way (RMSE, MAE, R2 on the held-out
test set) so the four models can be compared like for like in Section 5. Keeping
that logic here rather than in each notebook section means a change to the metric
set applies to everyone's model at once.

Import into 03_modelling.ipynb with:
    from src.modelling import train_linear_regression, train_knn, train_random_forest, train_xgboost, evaluate_model
"""
import numpy as np
import pandas as pd
import plotly as plt
from scipy import sparse
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# Searched on a log-ish spacing rather than every integer: CV RMSE changes slowly
# with k, so a fine grid costs many refits for differences well inside fold noise.
KNN_PARAM_GRID = {
    "n_neighbors": [3, 5, 7, 10, 15, 20, 30],
    "weights": ["uniform", "distance"],
}

# n_estimators/max_depth/learning_rate are tuned together: more/deeper trees add
# capacity, a lower learning rate trades training speed for generalisation.
XGB_PARAM_GRID = {
    "regressor__n_estimators": [100, 300, 500],
    "regressor__max_depth": [3, 5, 7],
    "regressor__learning_rate": [0.01, 0.05, 0.1],
}

RF_PARAM_GRID = {
    "regressor__n_estimators": [100, 200, 300],
    "regressor__max_depth": [10, 15, 20],
    "regressor__min_samples_leaf": [2, 5, 10],
}

def to_dense(X) -> np.ndarray:
    """Convert a sparse design matrix to a dense array for distance-based models."""
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


def train_linear_regression(X_train, y_train) -> LinearRegression:
    """Fit and return the untuned Linear Regression baseline.

    ``X_train`` must be the scaled, preprocessed feature matrix. Linear
    Regression has no hyperparameters to tune in this project, so fitting it
    once is the complete baseline training step.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_knn(X_train, y_train, param_grid: dict = None, cv: int = 5) -> GridSearchCV:
    """Tune a K-Nearest Neighbours regressor by grid search and return the fitted search.

    Scored with neg_root_mean_squared_error so the selected model is the one that
    minimises RMSE, the headline metric used to compare all four models.

    Returns the GridSearchCV object (not just the estimator) so the notebook can
    report best_params_/best_score_ and plot the full cv_results_ search curve as
    evidence of the tuning process.
    """
    param_grid = param_grid if param_grid is not None else KNN_PARAM_GRID

    grid = GridSearchCV(
        KNeighborsRegressor(n_jobs=-1),
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
    )
    grid.fit(X_train, y_train)

    return grid

def train_xgboost(
    X_train, y_train, preprocessor, param_grid: dict = None, cv: int = 5, use_gpu: bool = True
) -> GridSearchCV:
    """Tune an XGBoost regressor by grid search and return the fitted search.
 
    Scored with neg_root_mean_squared_error, same convention as train_knn, so the
    selected model minimises RMSE.
 
    use_gpu=True sets tree_method="hist" + device="cuda" so training runs on the
    GPU. Because there is only one GPU, GridSearchCV is run with n_jobs=1: firing
    folds in parallel (n_jobs=-1) would just contend for the same device instead
    of speeding anything up. CPU-only training can still use n_jobs=-1, since
    sklearn is then free to spread folds across CPU cores as usual.
 
    Returns the GridSearchCV object (not just the estimator) so the notebook can
    report best_params_/best_score_ and plot the full cv_results_ search curve as
    evidence of the tuning process.
    """
    param_grid = param_grid if param_grid is not None else XGB_PARAM_GRID

    regressor_kwargs = {"random_state": 42, "objective": "reg:squarederror"}
    if use_gpu:
        regressor_kwargs.update(tree_method="hist", device="cuda")

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("regressor", XGBRegressor(**regressor_kwargs)),
    ])

    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=1 if use_gpu else -1,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    return grid

def train_random_forest(
    X_train, y_train, preprocessor, param_grid: dict = None, cv: int = 5
) -> GridSearchCV:
    """Tune a preprocessed Random Forest pipeline by grid search.

    ``preprocessor`` is fitted inside each cross-validation fold, preventing data
    leakage from the target encoder and keeping the returned estimator ready to
    score raw transformed features.

    Scored with neg_root_mean_squared_error so the selected model is the one that
    minimises RMSE, the headline metric used to compare all four models.

    Returns the GridSearchCV object (not just the estimator) so the notebook can
    report best_params_/best_score_ and plot the full cv_results_ search results
    as evidence of the tuning process.
    """
    param_grid = param_grid if param_grid is not None else RF_PARAM_GRID

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42, n_jobs=-1)),
    ])

    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    grid.fit(X_train, y_train)

    return grid

def evaluate_model(model, X_test, y_test) -> dict:
    """Score a fitted model on the held-out test set.

    Returns rmse/mae/r2 plus the raw predictions, so the caller can reuse the
    same predictions for diagnostic plots without predicting a second time.
    """
    y_pred = model.predict(X_test)

    return {
        "rmse": root_mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "predictions": y_pred,
    }

def build_comparison_table(results: dict) -> pd.DataFrame:
    """Combine each model's evaluate_model() output into one comparison table.

    ``results`` maps model name -> the dict returned by evaluate_model() (must
    contain rmse/mae/r2). Sorted by RMSE ascending so the best model appears
    first. This is the single source of truth for the report's comparison
    table and the Streamlit app's Model Comparison page - both call this
    instead of duplicating the aggregation logic, so the numbers can't drift
    apart between the two.
    """
    rows = [
        {"model": name, "rmse": m["rmse"], "mae": m["mae"], "r2": m["r2"]}
        for name, m in results.items()
    ]
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def plot_metric_comparison(
    comparison_df: pd.DataFrame,
    metric: str = "rmse",
    ax=None
):
    """
    Plot a bar chart for one regression metric.

    RMSE and MAE: lower is better.
    R²: higher is better.
    """

    if metric not in ["rmse", "mae", "r2"]:
        raise ValueError("metric must be 'rmse', 'mae', or 'r2'")

    ascending = metric != "r2"
    ordered = comparison_df.sort_values(
        metric,
        ascending=ascending
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(
        ordered["model"],
        ordered[metric]
    )

    label = "R²" if metric == "r2" else metric.upper()

    ax.set_ylabel(label, fontsize=12)
    ax.set_title(
        f"{label} Comparison",
        fontsize=14,
        fontweight="bold"
    )

    ax.tick_params(
        axis="x",
        labelsize=10,
        rotation=20
    )

    # Add values above bars
    for bar, value in zip(bars, ordered[metric]):
        if metric == "r2":
            text = f"{value:.3f}"
        else:
            text = f"{value:.2f}"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            text,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    # Give labels some space above bars
    ymax = ordered[metric].max()

    if ymax > 0:
        ax.set_ylim(0, ymax * 1.15)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    return ax