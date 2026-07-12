"""Run the tuned multi-model comparison, write figures and metrics.

Loads a pipeline config, compares all models with cross-validated grid search on
a train/test split, and writes the ROC panel, a confusion matrix, a
feature-importance chart and ``metrics.json``.

Usage:
    # Offline, on the committed sample (no network):
    python scripts/compare_models.py

    # On the full Adult dataset from OpenML:
    python scripts/compare_models.py --dataset adult --n-sample 5000

    # On the built-in synthetic generator:
    python scripts/compare_models.py --dataset synthetic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tabml.compare import compare_models, results_table
from tabml.config import PipelineConfig
from tabml.data import (
    load_adult,
    load_sample,
    synthetic_tabular,
    train_test_split_frame,
)
from tabml.importance import extract_importances
from tabml.plots import confusion_figure, importance_figure, roc_panel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["sample", "adult", "synthetic"], default="sample")
    parser.add_argument("--n-sample", type=int, default=5000)
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = PipelineConfig.from_yaml(args.config) if Path(args.config).exists() else PipelineConfig()

    if args.dataset == "adult":
        data = load_adult(n_sample=args.n_sample)
    elif args.dataset == "synthetic":
        data = synthetic_tabular(n_rows=600)
    else:
        data = load_sample()
    print(
        f"Dataset: {len(data.X)} rows, {len(data.numeric)} numeric + "
        f"{len(data.categorical)} categorical features, positive rate {data.y.mean():.3f}"
    )

    train, test = train_test_split_frame(data, seed=0)
    results = compare_models(train, test, config=cfg)
    print("\n" + results_table(results))

    best = results[0]
    print(f"\nBest model: {best.name} (test ROC-AUC {best.test_roc_auc:.4f})")

    roc_path = roc_panel(results, test, out_dir / "roc_panel.png")
    conf_path = confusion_figure(best, test, out_dir / "confusion.png")
    print(f"Wrote {roc_path} and {conf_path}")

    # Feature importance for the best model that exposes it, else the best tree model.
    imp_source = next(
        (r for r in results if r.name in {"random_forest", "gradient_boosting", "logreg"}),
        best,
    )
    imp_path = importance_figure(imp_source, out_dir / "feature_importance.png")
    print(f"Wrote {imp_path} (importances from {imp_source.name})")

    metrics = {
        "dataset": args.dataset,
        "n_rows": int(len(data.X)),
        "positive_rate": float(data.y.mean()),
        "cv_folds": cfg.cv.folds,
        "best_model": best.name,
        "models": [r.summary() for r in results],
        "top_features": [
            {"feature": f.feature, "importance": f.importance}
            for f in extract_importances(imp_source.estimator, top_k=15)
        ],
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote {metrics_path}")

    # Persist the best fitted pipeline so inference runs with no retraining.
    import joblib

    model_path = out_dir / "best_model.joblib"
    joblib.dump(best.estimator, model_path)
    print(f"Wrote {model_path} (fitted {best.name} pipeline)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
