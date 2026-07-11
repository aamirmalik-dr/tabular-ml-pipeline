import numpy as np
import pandas as pd

from tabml.compare import compare_models, results_table
from tabml.data import (
    infer_column_types,
    synthetic_tabular,
    train_test_split_frame,
)
from tabml.pipeline import build_model_pipeline, build_preprocessor
from tabml.selection import lasso_selected_features


def test_synthetic_has_missing_and_types():
    data = synthetic_tabular(n_rows=200, seed=0)
    assert data.X.isna().any().any()  # missing values injected
    assert set(data.numeric) == {"age", "hours"}
    assert set(data.categorical) == {"education", "sector"}
    assert set(np.unique(data.y)) <= {0, 1}


def test_infer_column_types():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"], "c": [3, 4]})
    numeric, categorical = infer_column_types(df)
    assert set(numeric) == {"a", "c"}
    assert categorical == ["b"]


def test_preprocessor_handles_missing_and_categoricals():
    data = synthetic_tabular(n_rows=100)
    prep = build_preprocessor(data.numeric, data.categorical)
    Xt = prep.fit_transform(data.X)
    assert Xt.shape[0] == len(data.X)
    assert np.isfinite(np.asarray(Xt)).all()  # no NaNs after imputation


def test_pipeline_fits_and_predicts():
    data = synthetic_tabular(n_rows=200)
    train, test = train_test_split_frame(data, seed=0)
    pipe = build_model_pipeline(train.numeric, train.categorical, "logreg")
    pipe.fit(train.X, train.y)
    preds = pipe.predict(test.X)
    assert preds.shape[0] == len(test.X)


def test_lasso_selection_returns_indices():
    data = synthetic_tabular(n_rows=300)
    prep = build_preprocessor(data.numeric, data.categorical)
    X = prep.fit_transform(data.X)
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    idx = lasso_selected_features(X, data.y, alpha=0.01)
    assert idx.ndim == 1
    assert idx.size > 0  # some features survive the penalty on this signal
    assert idx.max() < X.shape[1]


def test_compare_models_runs_and_beats_chance():
    data = synthetic_tabular(n_rows=400)
    train, test = train_test_split_frame(data, seed=0)
    results = compare_models(train, test, models=["logreg", "random_forest"], cv=3)
    assert len(results) == 2
    # Sorted by test ROC-AUC descending, and the best should beat chance.
    assert results[0].test_roc_auc >= results[1].test_roc_auc
    assert results[0].test_roc_auc > 0.6
    assert isinstance(results_table(results), str)
