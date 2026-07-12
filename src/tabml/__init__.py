"""A configurable scikit-learn pipeline for messy tabular data.

The library assembles one workflow from a typed config: a ColumnTransformer that
imputes and scales numeric columns and imputes and one-hot encodes categorical
columns, an optional LASSO selection step, and a tuned comparison across several
classifiers (logistic regression, SVM, KNN, random forest, gradient boosting).

Typical use::

    from tabml import build_pipeline, load_sample, PipelineConfig

    data = load_sample()
    pipe = build_pipeline(PipelineConfig(model="gradient_boosting"))
    pipe.fit(data.X, data.y)
"""

from tabml.compare import ModelResult, compare_models, results_table
from tabml.config import (
    CVConfig,
    ImputerConfig,
    PipelineConfig,
    SelectionConfig,
)
from tabml.data import (
    TabularData,
    load_adult,
    load_sample,
    synthetic_tabular,
    train_test_split_frame,
)
from tabml.importance import FeatureImportance, extract_importances
from tabml.pipeline import (
    build_model_pipeline,
    build_pipeline,
    build_preprocessor,
    make_classifier,
)
from tabml.selection import lasso_selected_features

__all__ = [
    "TabularData",
    "load_adult",
    "load_sample",
    "synthetic_tabular",
    "train_test_split_frame",
    "PipelineConfig",
    "ImputerConfig",
    "SelectionConfig",
    "CVConfig",
    "build_pipeline",
    "build_preprocessor",
    "build_model_pipeline",
    "make_classifier",
    "lasso_selected_features",
    "extract_importances",
    "FeatureImportance",
    "compare_models",
    "results_table",
    "ModelResult",
]

__version__ = "0.2.0"
