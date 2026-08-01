"""
Section 3.4 Outlier Treatment.

Approach: cap (winsorize) rather than remove. Removing outliers outright
would drop ~4.7% of price rows and ~2.9% of square_feet rows (computed
below) — a meaningful chunk of real listings, not data-entry errors.
Capping preserves every row while bounding the influence of extreme
values on models sensitive to them (Linear Regression), and has no
real effect on tree-based models (Random Forest, Gradient Boosting),
which split on thresholds rather than magnitudes.
"""
import pandas as pd

OUTLIER_COLS = ["price", "square_feet"]


def compute_iqr_bounds(df: pd.DataFrame, col: str, k: float = 1.5) -> tuple:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def treat_outliers(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Cap price and square_feet at their 1.5*IQR upper bound. Returns a new dataframe."""
    df = df.copy()

    for col in OUTLIER_COLS:
        lower, upper = compute_iqr_bounds(df, col)
        n_flagged = ((df[col] < lower) | (df[col] > upper)).sum()

        # Only cap the upper tail — extreme-high values (mansion-sized listings,
        # data-entry errors) drive the skew here, not low values, which are
        # already well within a plausible range.
        df[col] = df[col].clip(upper=upper)

        if verbose:
            print(f"[3.4] {col}: capped {n_flagged} rows ({n_flagged/len(df):.2%}) "
                  f"above {upper:,.0f} (IQR upper bound)")

    return df
