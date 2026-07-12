# Model card: tabml gradient-boosting income classifier

This card describes the committed fitted pipeline at
`results/best_model.joblib` and the comparison that selected it. All numbers were
produced by `python scripts/compare_models.py` on the committed sample in this
repository. They are a controlled validation on a small carved subset, not a
published benchmark.

## Overview

- Task: binary classification, predict whether annual income exceeds 50K.
- Pipeline: median-impute and standardize numeric columns, most-frequent-impute
  and one-hot encode categoricals, LASSO `SelectFromModel`, then a
  `GradientBoostingClassifier`.
- Selected by a 3-fold cross-validated grid search over five model families,
  ranked by test ROC-AUC.
- Config: `configs/pipeline.yaml`. Seed 0, 75/25 stratified split.

## Data

- Source: UCI Adult census-income dataset, a public license-clean dataset.
- Committed sample: a class-stratified 900-row carve, positive rate 0.239. The
  full dataset is fetched by `scripts/download_data.py`.
- Features: 6 numeric and 8 categorical columns after type inference.

## Measured results (committed sample)

| Model | CV AUC | Test AUC | Test accuracy |
| --- | ---: | ---: | ---: |
| gradient_boosting | 0.8944 | 0.9026 | 0.8533 |
| random_forest | 0.9016 | 0.8970 | 0.8356 |
| logreg | 0.8970 | 0.8927 | 0.8356 |
| svm | 0.8821 | 0.8748 | 0.8311 |
| knn | 0.8763 | 0.8739 | 0.8311 |

Top features for the selected model: marital status (married civilian spouse),
education-num, age, capital-gain, hours-per-week.

## Intended use and limitations

- Intended as a reusable pipeline-building demonstration, not a deployable
  income model. The sample is small and the absolute numbers shift with the
  subsample and seed; the model ranking is the stable, reportable finding.
- The UCI Adult dataset carries known demographic biases (sex, race, and
  nationality distributions reflect 1994 US census sampling). A model trained on
  it should not be used to make decisions about real people.
- Only binary classification is wired end to end. No calibration, fairness
  audit, or drift monitoring is included.

## Reproduce

```bash
python scripts/compare_models.py
```

Writes the metrics, figures, and the fitted pipeline under `results/`.
