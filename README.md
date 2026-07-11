# Tabular ML pipeline

A reusable scikit-learn pipeline for messy tabular data, assembled from parts
that are each configurable and testable:

- A **ColumnTransformer** that median-imputes and standardizes numeric columns
  and most-frequent-imputes and one-hot encodes categorical columns, with
  unknown categories handled safely at transform time.
- **LASSO feature selection** on the encoded matrix.
- A **tuned multi-model comparison** across logistic regression, SVM, KNN,
  random forest, and gradient boosting, each grid-searched and scored by
  cross-validated ROC-AUC, then evaluated on a held-out test set.

## What it does

- Infers numeric vs categorical columns and builds the matching preprocessor.
- Fits everything inside cross-validation folds so imputation and scaling never
  leak test information into training.
- Produces a ranked model table plus ROC and confusion-matrix figures.
- Works on the public UCI Adult dataset or on a built-in synthetic dataframe
  with injected missing values.

## What it does not do

- It does not do automated feature engineering beyond encoding and LASSO
  selection.
- The hyperparameter grids are intentionally small so the demo runs on a CPU in
  a minute or two; widen them in `pipeline.py` for a serious search.
- Only binary classification is wired up end to end.

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Run

Fully offline (synthetic mixed-type data with missing values):

```bash
python scripts/compare_models.py
```

On the public UCI Adult census-income dataset:

```bash
python scripts/download_data.py --out data/adult.csv --n-sample 5000
python scripts/compare_models.py --dataset adult --n-sample 5000
```

## Results

Measured on the public UCI Adult dataset from OpenML, a 5000-row subsample
(positive rate 0.234), 75/25 stratified train/test split, 3-fold grid search,
seed 0. Produced by `scripts/compare_models.py` in this repository.

| Model             | CV AUC | Test AUC | Test accuracy |
|-------------------|-------:|---------:|--------------:|
| gradient_boosting | 0.9117 |   0.9129 |        0.8608 |
| random_forest     | 0.9064 |   0.9075 |        0.8560 |
| logreg            | 0.8992 |   0.8992 |        0.8488 |
| svm               | 0.8929 |   0.8846 |        0.8496 |
| knn               | 0.8790 |   0.8722 |        0.8384 |

Gradient boosting is the strongest model on this dataset and split, with the
tree ensembles ahead of the linear and instance-based baselines. Absolute
numbers shift a little with the subsample size and seed; the ranking is stable.

## Layout

```
src/tabml/       data, pipeline, selection, compare
scripts/         download_data.py, compare_models.py
notebooks/       demo.ipynb (executed)
tests/           pytest suite for preprocessing, selection, and comparison
data/            gitignored; see data/README.md
```

## Tests

```bash
pytest -q
ruff check src tests scripts
```

## License

MIT, see [LICENSE](LICENSE).

## Author

Aamir Malik. [GitHub](https://github.com/aamirmalik-dr) ·
[LinkedIn](https://linkedin.com/in/dr-aamirmalik)
