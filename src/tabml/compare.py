"""Tuned multi-model comparison for tabular classification.

Each model in the config is grid-searched with cross-validation on the training
set, then scored on a held-out test set. Results are returned sorted by test
ROC-AUC and carry enough state (the fitted estimator and test scores) to draw
the ROC panel without refitting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from sklearn.model_selection import GridSearchCV

from tabml.config import PipelineConfig, as_config
from tabml.data import TabularData
from tabml.pipeline import build_pipeline


def positive_scores(estimator, X) -> np.ndarray:
    """Return positive-class scores, using probabilities or a decision function."""
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
        estimator: The fitted best pipeline (excluded from serialization).
        test_scores: Positive-class scores on the test set (excluded from serialization).
    """

    name: str
    best_params: dict
    cv_score: float
    test_accuracy: float
    test_roc_auc: float
    estimator: Any = field(default=None, repr=False)
    test_scores: np.ndarray | None = field(default=None, repr=False)

    def summary(self) -> dict:
        """Return the JSON-serializable metrics (no fitted estimator or arrays)."""
        d = asdict(self)
        d.pop("estimator", None)
        d.pop("test_scores", None)
        return d


def compare_models(
    train: TabularData,
    test: TabularData,
    models: list[str] | None = None,
    cv: int | None = None,
    scoring: str | None = None,
    config: PipelineConfig | dict | None = None,
) -> list[ModelResult]:
    """Tune each model with grid search and evaluate on the test set.

    Args:
        train: Training data.
        test: Held-out test data.
        models: Model names to compare. Overrides ``config.models`` when given.
        cv: Grid-search folds. Overrides ``config.cv.folds`` when given.
        scoring: Selection metric. Overrides ``config.cv.scoring`` when given.
        config: Pipeline config supplying preprocessing, selection, grids and CV.

    Returns:
        One :class:`ModelResult` per model, sorted by test ROC-AUC descending.
    """
    from sklearn.metrics import accuracy_score, roc_auc_score

    cfg = as_config(config)
    names = models or cfg.models
    folds = cv or cfg.cv.folds
    metric = scoring or cfg.cv.scoring

    results: list[ModelResult] = []
    for name in names:
        pipe = build_pipeline(cfg, model=name, numeric=train.numeric, categorical=train.categorical)
        grid = cfg.grids.get(name, {})
        search = GridSearchCV(pipe, grid, cv=folds, scoring=metric, n_jobs=-1)
        search.fit(train.X, train.y)
        best = search.best_estimator_
        scores = positive_scores(best, test.X)
        pred = best.predict(test.X)
        results.append(
            ModelResult(
                name=name,
                best_params=search.best_params_,
                cv_score=float(search.best_score_),
                test_accuracy=float(accuracy_score(test.y, pred)),
                test_roc_auc=float(roc_auc_score(test.y, scores)),
                estimator=best,
                test_scores=np.asarray(scores),
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
