"""Feature-importance extraction from a fitted pipeline.

Handles the two-stage naming problem: the preprocessor expands categoricals into
one-hot columns (renamed by ``get_feature_names_out``), an optional LASSO step
drops some of them, and the final classifier exposes either linear ``coef_`` or
tree ``feature_importances_``. :func:`extract_importances` threads through all
three and returns a name-to-importance mapping on the surviving features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.pipeline import Pipeline


@dataclass
class FeatureImportance:
    """One feature and its importance.

    Attributes:
        feature: Encoded feature name.
        importance: Non-negative importance (absolute coefficient or tree gain).
    """

    feature: str
    importance: float


def _feature_names(pipe: Pipeline) -> np.ndarray:
    """Return the feature names entering the classifier, after any selection."""
    names = np.asarray(pipe.named_steps["prep"].get_feature_names_out())
    if "select" in pipe.named_steps:
        support = pipe.named_steps["select"].get_support()
        names = names[support]
    return names


def _raw_importances(clf, n_features: int) -> np.ndarray:
    """Return non-negative per-feature importances from a fitted classifier.

    Raises:
        ValueError: If the classifier exposes neither coefficients nor importances.
    """
    if hasattr(clf, "feature_importances_"):
        return np.asarray(clf.feature_importances_, dtype=float)
    if hasattr(clf, "coef_"):
        return np.abs(np.asarray(clf.coef_, dtype=float)).ravel()[:n_features]
    raise ValueError(
        f"{type(clf).__name__} exposes no feature_importances_ or coef_; "
        "importance is undefined (for example for KNN or an RBF SVM)"
    )


def extract_importances(pipe: Pipeline, top_k: int | None = None) -> list[FeatureImportance]:
    """Extract feature importances from a fitted pipeline, sorted descending.

    Args:
        pipe: A fitted pipeline with a ``prep`` step and a ``clf`` step, and an
            optional ``select`` step.
        top_k: If set, return only the ``top_k`` most important features.

    Returns:
        A list of :class:`FeatureImportance`, most important first.

    Raises:
        ValueError: If the final classifier does not expose importances.
    """
    names = _feature_names(pipe)
    values = _raw_importances(pipe.named_steps["clf"], len(names))
    order = np.argsort(values)[::-1]
    ranked = [FeatureImportance(str(names[i]), float(values[i])) for i in order]
    return ranked[:top_k] if top_k else ranked
