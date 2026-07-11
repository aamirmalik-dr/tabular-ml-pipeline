"""Fetch the public UCI Adult census-income dataset from OpenML.

Writes a CSV with the features plus an ``income_gt_50k`` target column. If
OpenML is unreachable, the script exits with a message; the tests and the
default benchmark do not need this file (they use the synthetic generator).

Usage:
    python scripts/download_data.py --out data/adult.csv --n-sample 5000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tabml.data import load_adult


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/adult.csv")
    parser.add_argument("--n-sample", type=int, default=5000, help="rows to keep (0 = all)")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_sample = None if args.n_sample == 0 else args.n_sample
    try:
        data = load_adult(n_sample=n_sample)
    except RuntimeError as exc:
        print(f"Download failed: {exc}")
        print("The tests and default benchmark use the offline synthetic generator instead.")
        return 1

    frame = data.X.copy()
    frame["income_gt_50k"] = data.y
    frame.to_csv(out_path, index=False)
    print(f"Wrote Adult dataset ({frame.shape[0]} rows, {frame.shape[1] - 1} features) -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
