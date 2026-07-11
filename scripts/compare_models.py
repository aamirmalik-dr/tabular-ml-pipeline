"""Run the tuned multi-model comparison and write figures.

Usage:
    # Offline synthetic data:
    python scripts/compare_models.py

    # On the downloaded Adult dataset:
    python scripts/compare_models.py --dataset adult
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from tabml.compare import compare_models, results_table
from tabml.data import load_adult, synthetic_tabular, train_test_split_frame
from tabml.pipeline import build_model_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["synthetic", "adult"], default="synthetic")
    parser.add_argument("--n-sample", type=int, default=5000)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "adult":
        data = load_adult(n_sample=args.n_sample)
    else:
        data = synthetic_tabular(n_rows=600)
    print(
        f"Dataset: {len(data.X)} rows, {len(data.numeric)} numeric + "
        f"{len(data.categorical)} categorical features, positive rate {data.y.mean():.3f}"
    )

    train, test = train_test_split_frame(data, seed=0)
    results = compare_models(train, test)
    print("\n" + results_table(results))

    best = results[0]
    print(f"\nBest model: {best.name} (test ROC-AUC {best.test_roc_auc:.4f})")

    # Refit the best pipeline for the figures.
    pipe = build_model_pipeline(train.numeric, train.categorical, best.name)
    pipe.set_params(**best.best_params)
    pipe.fit(train.X, train.y)

    RocCurveDisplay.from_estimator(pipe, test.X, test.y)
    plt.title(f"ROC ({best.name})")
    plt.tight_layout()
    plt.savefig(out_dir / "roc.png", dpi=120)
    plt.close()

    ConfusionMatrixDisplay.from_estimator(pipe, test.X, test.y)
    plt.title(f"Confusion matrix ({best.name})")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion.png", dpi=120)
    plt.close()

    print(f"Wrote figures to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
