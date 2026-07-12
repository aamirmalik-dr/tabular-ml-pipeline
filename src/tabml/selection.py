"""LASSO-based feature selection for the tabular pipeline."""

from __future__ import annotations

import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso


def lasso_selected_features(X: np.ndarray, y: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Select features with a LASSO (L1-penalized) linear model.

    The L1 penalty drives many coefficients to exactly zero; features with a
    non-zero coefficient are kept. The binary target is treated as a numeric
    response. Operates on an already-encoded numeric matrix (for example the
    output of the preprocessor).

    Args:
        X: Encoded feature matrix.
        y: Binary target (used as a 0/1 numeric response).
        alpha: L1 regularization strength; larger keeps fewer features.

    Returns:
        Integer indices of the selected columns. If the penalty zeros out every
        coefficient, an empty array is returned.
    """
    model = Lasso(alpha=alpha, max_iter=5000)
    selector = SelectFromModel(model, threshold=1e-10)
    selector.fit(X, y.astype(float))
    return np.where(selector.get_support())[0]
