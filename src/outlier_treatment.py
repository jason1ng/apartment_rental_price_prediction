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

OUTLIER_COLS = ["price", "square_feet"]  # all numeric features
PRICE_MAX_VALID = 20000

def audit_iqr(df: pd.DataFrame, columns: list, k: float = 1.5):
    header = (
        f"{'Variable':<15}"
        f"{'Count':>8}"
        f"{'Mean':>10}"
        f"{'SD':>10}"
        f"{'Min':>10}"
        f"{'25%':>10}"
        f"{'Median':>10}"
        f"{'75%':>10}"
        f"{'Max':>10}"
        f"{'Lower':>12}"
        f"{'Upper':>12}"
        f"{'Flagged':>10}"
    )

    print(header)
    print("-" * len(header))

    for col in columns:
        s = df[col].dropna()

        q1 = s.quantile(0.25)
        q2 = s.median()
        q3 = s.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - k * iqr
        upper = q3 + k * iqr

        flagged = ((s < lower) | (s > upper)).sum()

        print(
            f"{col:<15}"
            f"{len(s):>8}"
            f"{s.mean():>10.2f}"
            f"{s.std():>10.2f}"
            f"{s.min():>10.2f}"
            f"{q1:>10.2f}"
            f"{q2:>10.2f}"
            f"{q3:>10.2f}"
            f"{s.max():>10.2f}"
            f"{lower:>12.2f}"
            f"{upper:>12.2f}"
            f"{flagged:>10}"
        )

def compute_iqr_bounds(df: pd.DataFrame, col: str, k: float = 1.5) -> tuple:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

def remove_extreme_price_errors(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Remove implausible rental listings with extremely high monthly prices.
    These are treated as probable data-entry errors or listings outside the
    intended residential apartment market.
    """
    df = df.copy()

    n_before = len(df)
    df = df[df["price"] <= PRICE_MAX_VALID]

    if verbose:
        print(
            f"[3.3] Removed {n_before - len(df)} listings with "
            f"price > ${PRICE_MAX_VALID:,}"
        )

    return df

def winsorize(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
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

def treat_outliers(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Remove extreme price errors, then winsorize price and square_feet."""
    df = df.copy()
    df = remove_extreme_price_errors(df, verbose=verbose)
    df = winsorize(df, verbose=verbose)
    return df