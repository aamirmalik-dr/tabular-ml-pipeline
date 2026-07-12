import numpy as np
import pandas as pd
import pytest

from tabml.compare import ModelResult, compare_models, positive_scores, results_table
from tabml.config import PipelineConfig
from tabml.data import (
    infer_column_types,
    load_sample,
    synthetic_tabular,
    train_test_split_frame,
)
from tabml.importance import extract_importances
from tabml.pipeline import build_model_pipeline, build_pipeline, build_preprocessor
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


def test_preprocessor_autodetects_columns():
    # With no explicit column lists, the ColumnTransformer must detect by dtype.
    data = synthetic_tabular(n_rows=120)
    prep = build_preprocessor()
    Xt = prep.fit_transform(data.X)
    assert Xt.shape[0] == len(data.X)
    names = list(prep.get_feature_names_out())
    assert any(n.startswith("num__") for n in names)
    assert any(n.startswith("cat__") for n in names)


def test_config_from_dict_and_yaml_roundtrip(tmp_path):
    cfg = PipelineConfig.from_dict(
        {"scaler": "minmax", "selection": {"method": "none"}, "model": "logreg"}
    )
    assert cfg.scaler == "minmax"
    assert cfg.selection.method == "none"
    assert cfg.model == "logreg"

    import yaml

    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"scaler": "minmax", "model": "knn"}), encoding="utf-8")
    loaded = PipelineConfig.from_yaml(path)
    assert loaded.scaler == "minmax"
    assert loaded.model == "knn"
    # Missing keys fall back to defaults.
    assert loaded.imputer.numeric == "median"


def test_build_pipeline_assembles_expected_steps():
    cfg = PipelineConfig(model="random_forest")
    pipe = build_pipeline(cfg)
    step_names = [name for name, _ in pipe.steps]
    assert step_names == ["prep", "select", "clf"]  # LASSO select present by default
    assert type(pipe.named_steps["clf"]).__name__ == "RandomForestClassifier"


def test_build_pipeline_selection_none_drops_select_step():
    cfg = PipelineConfig(model="logreg")
    cfg.selection.method = "none"
    pipe = build_pipeline(cfg)
    assert [name for name, _ in pipe.steps] == ["prep", "clf"]


def test_build_pipeline_scaler_none_omits_scaler():
    cfg = PipelineConfig(scaler="none")
    pipe = build_pipeline(cfg)
    num_pipe = pipe.named_steps["prep"].transformers[0][1]
    assert [s for s, _ in num_pipe.steps] == ["impute"]


def test_pipeline_fits_and_predicts_from_config():
    data = synthetic_tabular(n_rows=200)
    train, test = train_test_split_frame(data, seed=0)
    pipe = build_pipeline(PipelineConfig(model="logreg"))
    pipe.fit(train.X, train.y)
    preds = pipe.predict(test.X)
    assert preds.shape[0] == len(test.X)


def test_build_model_pipeline_backward_compatible():
    data = synthetic_tabular(n_rows=150)
    pipe = build_model_pipeline(data.numeric, data.categorical, "logreg")
    assert [name for name, _ in pipe.steps] == ["prep", "clf"]
    pipe.fit(data.X, data.y)
    assert pipe.predict(data.X).shape[0] == len(data.X)


def test_lasso_selection_returns_indices():
    data = synthetic_tabular(n_rows=300)
    prep = build_preprocessor(data.numeric, data.categorical)
    X = prep.fit_transform(data.X)
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    idx = lasso_selected_features(X, data.y, alpha=0.01)
    assert idx.ndim == 1
    assert idx.size > 0  # some features survive the penalty on this signal
    assert idx.max() < X.shape[1]


def test_feature_importance_extraction_tree_and_linear():
    data = synthetic_tabular(n_rows=300)
    for model in ("random_forest", "logreg"):
        pipe = build_pipeline(PipelineConfig(model=model))
        pipe.fit(data.X, data.y)
        ranked = extract_importances(pipe, top_k=3)
        assert len(ranked) == 3
        # Non-negative and sorted descending.
        vals = [f.importance for f in ranked]
        assert all(v >= 0 for v in vals)
        assert vals == sorted(vals, reverse=True)


def test_feature_importance_raises_for_knn():
    data = synthetic_tabular(n_rows=150)
    pipe = build_pipeline(PipelineConfig(model="knn"))
    pipe.fit(data.X, data.y)
    with pytest.raises(ValueError):
        extract_importances(pipe)


def test_positive_scores_shape():
    data = synthetic_tabular(n_rows=200)
    train, test = train_test_split_frame(data, seed=0)
    pipe = build_pipeline(PipelineConfig(model="logreg"))
    pipe.fit(train.X, train.y)
    scores = positive_scores(pipe, test.X)
    assert scores.shape[0] == len(test.X)


def test_compare_models_runs_and_beats_chance():
    data = synthetic_tabular(n_rows=400)
    train, test = train_test_split_frame(data, seed=0)
    cfg = PipelineConfig(models=["logreg", "random_forest"])
    results = compare_models(train, test, config=cfg, cv=3)
    assert len(results) == 2
    assert all(isinstance(r, ModelResult) for r in results)
    # Sorted by test ROC-AUC descending, and the best should beat chance.
    assert results[0].test_roc_auc >= results[1].test_roc_auc
    assert results[0].test_roc_auc > 0.6
    assert isinstance(results_table(results), str)
    # Each result carries a fitted estimator and test scores for the ROC panel.
    assert results[0].estimator is not None
    assert results[0].test_scores.shape[0] == len(test.y)
    # summary() drops the non-serializable fields.
    summary = results[0].summary()
    assert "estimator" not in summary and "test_scores" not in summary


def test_load_sample_offline():
    # The committed sample must load with no network and carry mixed types.
    data = load_sample()
    assert len(data.X) > 100
    assert len(data.numeric) > 0 and len(data.categorical) > 0
    assert set(np.unique(data.y)) == {0, 1}
