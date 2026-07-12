"""Fetch the public UCI Adult census-income dataset from OpenML.

Writes a CSV with the features plus an ``income_gt_50k`` target column. With
``--stratified-rows N`` it instead writes a small class-stratified subset, which
is how the committed ``data/adult_sample.csv`` quickstart file is produced. If
OpenML is unreachable the script exits with a message; the tests and the offline
quickstart use the committed sample or the synthetic generator instead.

Usage:
    python scripts/download_data.py --out data/adult.csv --n-sample 5000
    python scripts/download_data.py --out data/adult_sample.csv --stratified-rows 900
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tabml.data import load_adult


def stratified_subset(frame: pd.DataFrame, target: str, rows: int, seed: int) -> pd.DataFrame:
    """Return a class-stratified row subset preserving the target balance."""
    rng = np.random.default_rng(seed)
    parts = []
    for _, group in frame.groupby(target):
        take = max(1, round(rows * len(group) / len(frame)))
        take = min(take, len(group))
        idx = rng.choice(len(group), size=take, replace=False)
        parts.append(group.iloc[idx])
    out = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/adult.csv")
    parser.add_argument("--n-sample", type=int, default=5000, help="rows to keep (0 = all)")
    parser.add_argument(
        "--stratified-rows",
        type=int,
        default=0,
        help="if > 0, write a class-stratified subset of this many rows",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_sample = None if args.n_sample == 0 else args.n_sample
    try:
        data = load_adult(n_sample=n_sample)
    except RuntimeError as exc:
        print(f"Download failed: {exc}")
        print("The tests and offline quickstart use the committed sample or synthetic data.")
        return 1

    frame = data.X.copy()
    frame["income_gt_50k"] = data.y

    if args.stratified_rows > 0:
        frame = stratified_subset(frame, "income_gt_50k", args.stratified_rows, args.seed)

    frame.to_csv(out_path, index=False)
    print(
        f"Wrote {frame.shape[0]} rows, {frame.shape[1] - 1} features "
        f"(positive rate {frame['income_gt_50k'].mean():.3f}) -> {out_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
