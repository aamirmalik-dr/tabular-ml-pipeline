"""Typed configuration for the tabular pipeline builder.

A :class:`PipelineConfig` fully describes a pipeline: which imputers, scaler and
encoder the preprocessor uses, whether LASSO feature selection is inserted, which
classifier the single-model builder assembles, and the model list, grids and
cross-validation settings the comparison uses. Configs load from a plain dict or
a YAML file so the same run is reproducible from ``configs/pipeline.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_MODELS = ["logreg", "svm", "knn", "random_forest", "gradient_boosting"]

DEFAULT_GRIDS: dict[str, dict[str, list]] = {
    "logreg": {"clf__C": [0.1, 1.0, 10.0]},
    "svm": {"clf__C": [0.5, 1.0], "clf__kernel": ["rbf"]},
    "knn": {"clf__n_neighbors": [5, 11, 21]},
    "random_forest": {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 10]},
    "gradient_boosting": {"clf__n_estimators": [100], "clf__learning_rate": [0.05, 0.1]},
}


@dataclass
class ImputerConfig:
    """Imputation strategies passed to :class:`sklearn.impute.SimpleImputer`.

    Attributes:
        numeric: Strategy for numeric columns (``median``, ``mean``).
        categorical: Strategy for categorical columns (``most_frequent``, ``constant``).
    """

    numeric: str = "median"
    categorical: str = "most_frequent"


@dataclass
class SelectionConfig:
    """LASSO feature-selection settings.

    Attributes:
        method: ``lasso`` to insert an L1 ``SelectFromModel`` step, ``none`` to skip.
        alpha: L1 penalty strength; larger keeps fewer features.
    """

    method: str = "lasso"
    alpha: float = 0.01


@dataclass
class CVConfig:
    """Cross-validation settings for the tuned comparison.

    Attributes:
        folds: Number of grid-search folds.
        scoring: Metric used to select hyperparameters.
    """

    folds: int = 3
    scoring: str = "roc_auc"


@dataclass
class PipelineConfig:
    """Complete, typed description of a tabular pipeline.

    Attributes:
        imputer: Numeric and categorical imputation strategies.
        scaler: Numeric scaler (``standard``, ``minmax`` or ``none``).
        encoder: Categorical encoder (``onehot``).
        selection: LASSO feature-selection settings.
        model: Classifier assembled by the single-model builder.
        models: Classifiers compared by the tuned comparison.
        cv: Cross-validation settings.
        grids: Per-model hyperparameter grids keyed by classifier name.
    """

    imputer: ImputerConfig = field(default_factory=ImputerConfig)
    scaler: str = "standard"
    encoder: str = "onehot"
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    model: str = "gradient_boosting"
    models: list[str] = field(default_factory=lambda: list(DEFAULT_MODELS))
    cv: CVConfig = field(default_factory=CVConfig)
    grids: dict[str, dict[str, list]] = field(
        default_factory=lambda: {k: dict(v) for k, v in DEFAULT_GRIDS.items()}
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        """Build a config from a nested dict, filling defaults for missing keys."""
        data = dict(data or {})
        imputer = ImputerConfig(**(data.get("imputer") or {}))
        selection = SelectionConfig(**(data.get("selection") or {}))
        cv = CVConfig(**(data.get("cv") or {}))
        return cls(
            imputer=imputer,
            scaler=data.get("scaler", "standard"),
            encoder=data.get("encoder", "onehot"),
            selection=selection,
            model=data.get("model", "gradient_boosting"),
            models=list(data.get("models", DEFAULT_MODELS)),
            cv=cv,
            grids=dict(data.get("grids", DEFAULT_GRIDS)),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        """Load a config from a YAML file.

        Raises:
            ImportError: If PyYAML is not installed.
        """
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise ImportError("install pyyaml to load YAML configs") from exc
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls.from_dict(data)


def as_config(config: PipelineConfig | dict[str, Any] | None) -> PipelineConfig:
    """Coerce a dict, ``PipelineConfig`` or ``None`` into a ``PipelineConfig``."""
    if config is None:
        return PipelineConfig()
    if isinstance(config, PipelineConfig):
        return config
    if isinstance(config, dict):
        return PipelineConfig.from_dict(config)
    raise TypeError(f"unsupported config type: {type(config)!r}")
