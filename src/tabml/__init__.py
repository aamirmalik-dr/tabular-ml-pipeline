"""A reusable scikit-learn pipeline for messy tabular data.

The pieces fit together into one configurable workflow: a ColumnTransformer that
imputes and scales numeric columns and imputes and one-hot encodes categorical
columns, optional LASSO-based feature selection, and a tuned comparison across
several classifiers (logistic regression, SVM, KNN, random forest, gradient
boosting).
"""

from tabml.compare import compare_models
from tabml.data import (
    TabularData,
    load_adult,
    synthetic_tabular,
    train_test_split_frame,
)
from tabml.pipeline import build_model_pipeline, build_preprocessor
from tabml.selection import lasso_selected_features

__all__ = [
    "TabularData",
    "load_adult",
    "synthetic_tabular",
    "train_test_split_frame",
    "build_preprocessor",
    "build_model_pipeline",
    "lasso_selected_features",
    "compare_models",
]

__version__ = "0.1.0"
