"""Tuned multi-model comparison for tabular classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GridSearchCV

from tabml.data import TabularData
from tabml.pipeline import PARAM_GRIDS, build_model_pipeline


def _positive_scores(estimator, X) -> np.ndarray:
    """Return positive-class scores, using probabilities or decision function."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    return estimator.decision_function(X)


@dataclass
class ModelResult:
    """Tuning and test results for one classifier.

    Attributes:
        name: Classifier name.
        best_params: Best hyperparameters from the grid search.
        cv_score: Best cross-validated ROC-AUC on the training set.
        test_accuracy: Accuracy on the held-out test set.
        test_roc_auc: ROC-AUC on the held-out test set.
    """

    name: str
    best_params: dict
    cv_score: float
    test_accuracy: float
    test_roc_auc: float


def compare_models(
    train: TabularData,
    test: TabularData,
    models: list[str] | None = None,
    cv: int = 3,
    scoring: str = "roc_auc",
) -> list[ModelResult]:
    """Tune each model with grid search and evaluate on the test set.

    Args:
        train: Training data.
        test: Held-out test data.
        models: Model names to compare. Defaults to all five.
        cv: Number of cross-validation folds for the grid search.
        scoring: Scoring metric for model selection.

    Returns:
        One :class:`ModelResult` per model, sorted by test ROC-AUC descending.
    """
    from sklearn.metrics import accuracy_score, roc_auc_score

    models = models or ["logreg", "svm", "knn", "random_forest", "gradient_boosting"]
    results: list[ModelResult] = []
    for name in models:
        pipe = build_model_pipeline(train.numeric, train.categorical, name)
        search = GridSearchCV(pipe, PARAM_GRIDS[name], cv=cv, scoring=scoring, n_jobs=-1)
        search.fit(train.X, train.y)
        best = search.best_estimator_
        proba = _positive_scores(best, test.X)
        pred = best.predict(test.X)
        results.append(
            ModelResult(
                name=name,
                best_params=search.best_params_,
                cv_score=float(search.best_score_),
                test_accuracy=float(accuracy_score(test.y, pred)),
                test_roc_auc=float(roc_auc_score(test.y, proba)),
            )
        )
    results.sort(key=lambda r: r.test_roc_auc, reverse=True)
    return results


def results_table(results: list[ModelResult]) -> str:
    """Format model results as a plain-text table."""
    header = f"{'model':<20}{'CV AUC':>10}{'test AUC':>10}{'test acc':>10}"
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r.name:<20}{r.cv_score:>10.4f}{r.test_roc_auc:>10.4f}{r.test_accuracy:>10.4f}"
        )
    return "\n".join(lines)
