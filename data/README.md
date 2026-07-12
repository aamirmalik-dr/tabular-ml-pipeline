# Data

## Committed sample (offline quickstart)

`adult_sample.csv` is committed so the quickstart and tests run with no network.
It is a **carved public subset**, not synthetic: a class-stratified 900-row
sample of the UCI Adult census-income dataset (positive rate 0.239), with a
binary `income_gt_50k` target. UCI Adult is a public, license-clean dataset. The
sample is produced by:

```bash
python scripts/download_data.py --out data/adult_sample.csv --stratified-rows 900
```

## Full dataset

`scripts/download_data.py` fetches the full UCI Adult dataset from OpenML (mixed
numeric and categorical features, binary >50K income target):

```bash
python scripts/download_data.py --out data/adult.csv --n-sample 5000
```

Pass `--n-sample 0` for every row. The full CSV is gitignored.

## Synthetic (tests)

The unit tests build a small mixed-type dataframe with injected missing values
via `synthetic_tabular`, which exercises imputation, encoding, and selection with
no files on disk. Everything under `data/` except this README and the committed
sample stays gitignored.
