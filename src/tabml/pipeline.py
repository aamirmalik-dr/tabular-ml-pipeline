"""Preprocessing and model pipeline builders."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Build a ColumnTransformer for mixed tabular data.

    Numeric columns are median-imputed and standardized; categorical columns are
    most-frequent-imputed and one-hot encoded with unknown categories ignored at
    transform time.
    """
    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ]
    )


def make_classifier(name: str):
    """Return an unfitted classifier by name.

    Raises:
        ValueError: If the name is not recognized.
    """
    classifiers = {
        "logreg": LogisticRegression(max_iter=2000),
        "svm": SVC(),
        "knn": KNeighborsClassifier(),
        "random_forest": RandomForestClassifier(random_state=0),
        "gradient_boosting": GradientBoostingClassifier(random_state=0),
    }
    if name not in classifiers:
        raise ValueError(f"unknown classifier {name!r}; choose from {list(classifiers)}")
    return classifiers[name]


def build_model_pipeline(
    numeric: list[str], categorical: list[str], classifier: str = "logreg"
) -> Pipeline:
    """Compose the preprocessor and a classifier into one fit/predict pipeline."""
    return Pipeline(
        [
            ("prep", build_preprocessor(numeric, categorical)),
            ("clf", make_classifier(classifier)),
        ]
    )


# Small hyperparameter grids per model, keyed by the pipeline's ``clf`` step.
PARAM_GRIDS: dict[str, dict[str, list]] = {
    "logreg": {"clf__C": [0.1, 1.0, 10.0]},
    "svm": {"clf__C": [0.5, 1.0], "clf__kernel": ["rbf"]},
    "knn": {"clf__n_neighbors": [5, 11, 21]},
    "random_forest": {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 10]},
    "gradient_boosting": {"clf__n_estimators": [100], "clf__learning_rate": [0.05, 0.1]},
}
