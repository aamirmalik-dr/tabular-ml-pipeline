"""Data sources for the tabular pipeline.

* :func:`load_adult` fetches the public UCI Adult census-income dataset (mixed
  numeric and categorical features, binary income target) from OpenML.
* :func:`synthetic_tabular` builds a small mixed-type dataframe with missing
  values, used by the tests so continuous integration needs no download.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class TabularData:
    """A feature dataframe with a binary target and typed column lists.

    Attributes:
        X: Feature dataframe (may contain missing values and categoricals).
        y: Binary integer target aligned with ``X``.
        numeric: Names of numeric columns.
        categorical: Names of categorical columns.
    """

    X: pd.DataFrame
    y: np.ndarray
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)


def infer_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split dataframe columns into numeric and categorical name lists."""
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


def synthetic_tabular(n_rows: int = 400, seed: int = 0, missing_rate: float = 0.05) -> TabularData:
    """Generate a small mixed-type classification dataframe with missing values.

    The target depends on a mix of numeric and categorical signals so a working
    pipeline can beat chance.

    Args:
        n_rows: Number of rows.
        seed: Random seed.
        missing_rate: Fraction of numeric cells set to NaN.

    Returns:
        A :class:`TabularData` with two numeric and two categorical columns.
    """
    rng = np.random.default_rng(seed)
    age = rng.normal(40, 12, n_rows)
    hours = rng.normal(40, 10, n_rows)
    education = rng.choice(["hs", "college", "grad"], size=n_rows, p=[0.5, 0.35, 0.15])
    sector = rng.choice(["public", "private", "self"], size=n_rows)

    logit = (
        0.04 * (age - 40)
        + 0.05 * (hours - 40)
        + np.where(education == "grad", 1.2, np.where(education == "college", 0.4, -0.3))
        + np.where(sector == "self", 0.3, 0.0)
    )
    prob = 1 / (1 + np.exp(-logit))
    y = (rng.random(n_rows) < prob).astype(int)

    X = pd.DataFrame({"age": age, "hours": hours, "education": education, "sector": sector})
    # Inject missing values into numeric columns.
    for col in ["age", "hours"]:
        mask = rng.random(n_rows) < missing_rate
        X.loc[mask, col] = np.nan

    numeric, categorical = infer_column_types(X)
    return TabularData(X=X, y=y, numeric=numeric, categorical=categorical)


def load_adult(n_sample: int | None = 5000, seed: int = 0) -> TabularData:
    """Fetch the UCI Adult income dataset from OpenML.

    Args:
        n_sample: If set, randomly subsample this many rows (keeps the demo fast
            on a CPU). Pass ``None`` for the full dataset.
        seed: Random seed for subsampling.

    Returns:
        The dataset with the income target encoded as 1 for ">50K".

    Raises:
        RuntimeError: If the dataset cannot be fetched.
    """
    from sklearn.datasets import fetch_openml

    try:
        bunch = fetch_openml("adult", version=2, as_frame=True)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"could not fetch Adult from OpenML: {exc}") from exc

    X = bunch.data.copy()
    target = bunch.target.astype(str)
    y = target.str.contains(">50K").astype(int).to_numpy()

    if n_sample is not None and n_sample < len(X):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=n_sample, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        y = y[idx]

    numeric, categorical = infer_column_types(X)
    return TabularData(X=X, y=y, numeric=numeric, categorical=categorical)


def train_test_split_frame(
    data: TabularData, test_fraction: float = 0.25, seed: int = 0
) -> tuple[TabularData, TabularData]:
    """Split rows into train and test, stratified by the target."""
    from sklearn.model_selection import train_test_split

    X_tr, X_te, y_tr, y_te = train_test_split(
        data.X, data.y, test_size=test_fraction, random_state=seed, stratify=data.y
    )
    train = TabularData(X_tr.reset_index(drop=True), y_tr, data.numeric, data.categorical)
    test = TabularData(X_te.reset_index(drop=True), y_te, data.numeric, data.categorical)
    return train, test
