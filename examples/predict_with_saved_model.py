"""Load the committed fitted pipeline and predict without retraining.

The comparison script writes ``results/best_model.joblib``, a fully fitted
scikit-learn pipeline (preprocessing plus the best classifier). This example
loads it and scores a handful of rows from the committed sample, so inference
runs immediately with no training and no network.

    python examples/predict_with_saved_model.py
"""

from __future__ import annotations

from pathlib import Path

import joblib

from tabml import load_sample

MODEL_PATH = Path(__file__).resolve().parents[1] / "results" / "best_model.joblib"


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"no saved model at {MODEL_PATH}; run 'python scripts/compare_models.py' first"
        )

    pipe = joblib.load(MODEL_PATH)
    data = load_sample()
    rows = data.X.head(10)

    preds = pipe.predict(rows)
    proba = pipe.predict_proba(rows)[:, 1]
    for i, (p, q, truth) in enumerate(zip(preds, proba, data.y[:10], strict=False)):
        print(f"row {i:>2}  pred={p}  p(>50K)={q:0.3f}  actual={truth}")


if __name__ == "__main__":
    main()
