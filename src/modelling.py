"""
Section 4 Modelling — shared training and evaluation helpers.

Every model in Section 4 is scored the same way (RMSE, MAE, R2 on the held-out
test set) so the four models can be compared like for like in Section 5. Keeping
that logic here rather than in each notebook section means a change to the metric
set applies to everyone's model at once.

Import into 03_modelling.ipynb with:
    from src.modelling import train_knn, evaluate_model
"""
import numpy as np
from scipy import sparse
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

# Searched on a log-ish spacing rather than every integer: CV RMSE changes slowly
# with k, so a fine grid costs many refits for differences well inside fold noise.
KNN_PARAM_GRID = {
    "n_neighbors": [3, 5, 7, 10, 15, 20, 30],
    "weights": ["uniform", "distance"],
}


def to_dense(X) -> np.ndarray:
    """Convert a sparse design matrix to a dense array for distance-based models."""
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


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
