# API reference

Everything below is importable directly from `tabml`.

## Data

| Object | Purpose |
| --- | --- |
| `TabularData` | Dataclass holding `X`, `y`, `numeric`, `categorical`. |
| `load_sample(path=None, target="income_gt_50k")` | Load the committed offline sample as `TabularData`. |
| `load_adult(n_sample=5000, seed=0)` | Fetch the full UCI Adult set from OpenML. Needs network. |
| `synthetic_tabular(n_rows=400, seed=0, missing_rate=0.05)` | Generate a mixed-type frame with injected missing values. |
| `train_test_split_frame(data, test_fraction=0.25, seed=0)` | Stratified train/test split returning two `TabularData`. |

## Config

| Object | Purpose |
| --- | --- |
| `PipelineConfig` | Complete typed description of a pipeline. `from_dict`, `from_yaml`. |
| `ImputerConfig` | Numeric and categorical imputation strategies. |
| `SelectionConfig` | LASSO `method` (`lasso` / `none`) and `alpha`. |
| `CVConfig` | Grid-search `folds` and `scoring`. |

## Pipeline builders

| Function | Returns |
| --- | --- |
| `build_pipeline(config=None, model=None, numeric=None, categorical=None)` | Unfitted `Pipeline`: prep, optional LASSO select, classifier. |
| `build_preprocessor(numeric=None, categorical=None, config=None)` | Unfitted `ColumnTransformer`. |
| `build_model_pipeline(numeric, categorical, classifier="logreg")` | Backward-compatible builder with explicit column lists, no selection. |
| `make_classifier(name)` | Unfitted classifier by name. |

Classifier names: `logreg`, `svm`, `knn`, `random_forest`, `gradient_boosting`.

## Selection and importance

| Function | Returns |
| --- | --- |
| `lasso_selected_features(X, y, alpha=0.01)` | Integer indices of columns a LASSO keeps on an encoded matrix. |
| `extract_importances(pipe, top_k=None)` | List of `FeatureImportance` sorted descending. Raises when undefined. |

## Comparison

| Function | Returns |
| --- | --- |
| `compare_models(train, test, models=None, cv=None, scoring=None, config=None)` | List of `ModelResult` sorted by test ROC-AUC. |
| `results_table(results)` | Plain-text results table. |
| `ModelResult` | Per-model tuning and test metrics plus the fitted estimator and test scores. |

## Figures (`tabml.plots`)

| Function | Writes |
| --- | --- |
| `roc_panel(results, test, out_path)` | The hero panel: multi-model ROC overlay plus best-model confusion matrix. |
| `confusion_figure(result, test, out_path)` | A standalone confusion matrix. |
| `importance_figure(result, out_path, top_k=15)` | A feature-importance bar chart. |
