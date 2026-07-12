"""Config-driven preprocessing and model pipeline builders.

The public entry point is :func:`build_pipeline`, which consumes a
:class:`~tabml.config.PipelineConfig` (or a plain dict) and returns a ready to
fit scikit-learn :class:`~sklearn.pipeline.Pipeline`. The pipeline auto-detects
numeric and categorical columns with :func:`sklearn.compose.make_column_selector`
so the same config works on any mixed-type dataframe.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from tabml.config import PipelineConfig, as_config

_SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "none": None,
}


def make_classifier(name: str):
    """Return an unfitted classifier by name.

    Raises:
        ValueError: If the name is not recognized.
    """
    classifiers = {
        "logreg": LogisticRegression(max_iter=2000),
        "svm": SVC(random_state=0),
        "knn": KNeighborsClassifier(),
        "random_forest": RandomForestClassifier(random_state=0),
        "gradient_boosting": GradientBoostingClassifier(random_state=0),
    }
    if name not in classifiers:
        raise ValueError(f"unknown classifier {name!r}; choose from {list(classifiers)}")
    return classifiers[name]


def build_preprocessor(
    numeric: list[str] | None = None,
    categorical: list[str] | None = None,
    config: PipelineConfig | dict | None = None,
) -> ColumnTransformer:
    """Build a ColumnTransformer for mixed tabular data from a config.

    Numeric columns are imputed and optionally scaled; categorical columns are
    imputed and one-hot encoded with unknown categories ignored at transform
    time. When ``numeric`` and ``categorical`` are omitted, column types are
    detected at fit time by dtype.

    Args:
        numeric: Explicit numeric column names, or ``None`` to auto-detect.
        categorical: Explicit categorical column names, or ``None`` to auto-detect.
        config: Pipeline config controlling imputers, scaler and encoder.

    Returns:
        An unfitted :class:`~sklearn.compose.ColumnTransformer`.
    """
    cfg = as_config(config)

    num_steps: list = [("impute", SimpleImputer(strategy=cfg.imputer.numeric))]
    scaler_cls = _SCALERS.get(cfg.scaler, StandardScaler)
    if scaler_cls is not None:
        num_steps.append(("scale", scaler_cls()))
    numeric_pipe = Pipeline(num_steps)

    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy=cfg.imputer.categorical)),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    num_sel = numeric if numeric is not None else make_column_selector(dtype_include="number")
    cat_sel = (
        categorical if categorical is not None else make_column_selector(dtype_exclude="number")
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, num_sel),
            ("cat", categorical_pipe, cat_sel),
        ]
    )


def build_pipeline(
    config: PipelineConfig | dict | None = None,
    model: str | None = None,
    numeric: list[str] | None = None,
    categorical: list[str] | None = None,
) -> Pipeline:
    """Assemble a full fit/predict pipeline from a config.

    The pipeline is preprocessor, optional LASSO ``SelectFromModel`` step, then
    the chosen classifier. Because selection lives inside the pipeline, it is
    refit on each cross-validation fold and never leaks test information.

    Args:
        config: A :class:`~tabml.config.PipelineConfig` or dict. Defaults are used
            when ``None``.
        model: Classifier name overriding ``config.model``.
        numeric: Explicit numeric columns, or ``None`` to auto-detect by dtype.
        categorical: Explicit categorical columns, or ``None`` to auto-detect.

    Returns:
        An unfitted :class:`~sklearn.pipeline.Pipeline`.
    """
    cfg = as_config(config)
    clf_name = model or cfg.model
    steps: list = [("prep", build_preprocessor(numeric, categorical, cfg))]
    if cfg.selection.method == "lasso":
        selector = SelectFromModel(Lasso(alpha=cfg.selection.alpha, max_iter=5000), threshold=1e-10)
        steps.append(("select", selector))
    steps.append(("clf", make_classifier(clf_name)))
    return Pipeline(steps)


def build_model_pipeline(
    numeric: list[str], categorical: list[str], classifier: str = "logreg"
) -> Pipeline:
    """Compose preprocessor and a classifier with explicit column lists.

    Thin backward-compatible wrapper over :func:`build_pipeline` that skips LASSO
    selection and takes explicit column names.
    """
    cfg = PipelineConfig(model=classifier)
    cfg.selection.method = "none"
    return build_pipeline(cfg, model=classifier, numeric=numeric, categorical=categorical)


# Small hyperparameter grids per model, keyed by the pipeline's ``clf`` step.
PARAM_GRIDS: dict[str, dict[str, list]] = {
    "logreg": {"clf__C": [0.1, 1.0, 10.0]},
    "svm": {"clf__C": [0.5, 1.0], "clf__kernel": ["rbf"]},
    "knn": {"clf__n_neighbors": [5, 11, 21]},
    "random_forest": {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 10]},
    "gradient_boosting": {"clf__n_estimators": [100], "clf__learning_rate": [0.05, 0.1]},
}
