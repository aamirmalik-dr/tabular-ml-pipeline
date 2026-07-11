# Data

This directory is gitignored. No datasets are committed.

## UCI Adult (real)

`scripts/download_data.py` fetches the public UCI Adult census-income dataset
from OpenML (mixed numeric and categorical features, binary >50K income target):

```bash
python scripts/download_data.py --out data/adult.csv --n-sample 5000
```

The demo subsamples rows by default so it runs quickly on a CPU; pass
`--n-sample 0` for the full dataset.

## Synthetic (offline)

The unit tests and the default `compare_models.py` invocation need no download.
They build a small mixed-type dataframe with injected missing values
(`synthetic_tabular`), which exercises imputation, encoding, and selection. This
keeps continuous integration fully self-contained.
