"""
Section 3.3 Outlier Treatment.

Approach: remove only clearly implausible rental prices, then cap the
upper tail of square_feet. High rents are retained because they may be
valid luxury listings and price is the prediction target; capping it
would teach the model that every rent above the cap has the same value.

The square_feet cap preserves rows while bounding the influence of very
large units on scale-sensitive models such as Linear Regression and KNN.
Tree-based models are largely invariant to feature scale.
"""
import pandas as pd

# Only square_feet is winsorized. Other numeric fields are audited but retained.
OUTLIER_COLS = ["square_feet"]
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
    """Cap only the upper tail of square_feet at its 1.5*IQR bound.

    Price is intentionally not capped: it is the target variable and high
    prices can represent valid premium listings.
    """
    df = df.copy()

    for col in OUTLIER_COLS:
        lower, upper = compute_iqr_bounds(df, col)
        n_upper = (df[col] > upper).sum()

        # Only cap the upper tail — extreme-high values (mansion-sized listings,
        # data-entry errors) drive the skew here, not low values, which are
        # already well within a plausible range.
        df[col] = df[col].clip(upper=upper)

        if verbose:
            print(f"[3.3] {col}: capped {n_upper} rows ({n_upper/len(df):.2%}) "
                  f"above {upper:,.0f} (IQR upper bound)")

    return df

def treat_outliers(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Remove implausible prices, then winsorize extreme square-footage values."""
    df = df.copy()
    df = remove_extreme_price_errors(df, verbose=verbose)
    df = winsorize(df, verbose=verbose)
    return df
