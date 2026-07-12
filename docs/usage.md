# Usage

`tabml` assembles a scikit-learn pipeline for mixed-type tabular data from a
typed config. This page walks through the common tasks. Every snippet runs on
the committed sample with no network.

## Load data

```python
from tabml import load_sample, synthetic_tabular, load_adult

data = load_sample()             # committed carved UCI Adult subset (offline)
data = synthetic_tabular(600)    # generated mixed-type frame with missing cells
data = load_adult(n_sample=5000) # full UCI Adult from OpenML (needs network)
```

Each returns a `TabularData` with `X` (a dataframe that may hold missing values
and categoricals), `y` (a binary integer array), and `numeric` / `categorical`
column name lists inferred by dtype.

## Configure

A `PipelineConfig` fully describes one pipeline. Build it in code or load it from
`configs/pipeline.yaml`.

```python
from tabml import PipelineConfig

cfg = PipelineConfig(
    scaler="standard",           # standard | minmax | none
    model="gradient_boosting",   # single-model builder default
)
cfg.selection.method = "lasso"   # lasso | none
cfg.selection.alpha = 0.001      # larger keeps fewer features

# Or from YAML:
cfg = PipelineConfig.from_yaml("configs/pipeline.yaml")
```

## Build and fit

```python
from tabml import build_pipeline

pipe = build_pipeline(cfg)       # prep -> optional LASSO select -> classifier
pipe.fit(data.X, data.y)
pipe.predict(data.X)
```

Column types are detected at fit time with `make_column_selector`, so the same
config transfers to any mixed-type dataframe. Selection lives inside the pipeline
and is refit per cross-validation fold, so it never leaks test information.

## Compare models

```python
from tabml import compare_models, results_table, train_test_split_frame

train, test = train_test_split_frame(data, seed=0)
results = compare_models(train, test, config=cfg)   # sorted by test ROC-AUC
print(results_table(results))
best = results[0]                # carries the fitted estimator and test scores
```

## Feature importance

```python
from tabml import extract_importances

ranked = extract_importances(best.estimator, top_k=10)
```

Threads through the one-hot expansion and the LASSO step, then reads either
linear coefficients or tree importances. It raises for models that expose neither
(for example KNN or an RBF SVM).

## Figures

```python
from tabml.plots import roc_panel, confusion_figure, importance_figure

roc_panel(results, test, "results/roc_panel.png")   # the hero panel
```

See [api.md](api.md) for the full public reference.
