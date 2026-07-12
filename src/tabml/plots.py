"""Figure generation for the model comparison.

Produces the signature ROC panel (all model ROC curves overlaid, with the best
model's confusion matrix alongside), a standalone confusion matrix, and a
feature-importance bar chart. All figures render headless through the Agg
backend so they generate the same way in CI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    roc_curve,
)

from tabml.compare import ModelResult  # noqa: E402
from tabml.data import TabularData  # noqa: E402
from tabml.importance import extract_importances  # noqa: E402


def roc_panel(results: list[ModelResult], test: TabularData, out_path: str | Path) -> Path:
    """Draw the multi-model ROC overlay next to the best model's confusion matrix.

    Args:
        results: Model results sorted best-first, each carrying ``test_scores``.
        test: The held-out test data (for the true labels).
        out_path: Destination PNG path.

    Returns:
        The path written.
    """
    fig, (ax_roc, ax_cm) = plt.subplots(1, 2, figsize=(11, 4.8), width_ratios=[1.35, 1])

    for r in results:
        if r.test_scores is None:
            continue
        fpr, tpr, _ = roc_curve(test.y, r.test_scores)
        ax_roc.plot(fpr, tpr, lw=2, label=f"{r.name} (AUC {auc(fpr, tpr):.3f})")
    ax_roc.plot([0, 1], [0, 1], ls="--", color="0.6", lw=1)
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("ROC curves, all models")
    ax_roc.legend(loc="lower right", fontsize=9)
    ax_roc.set_xlim(-0.01, 1.01)
    ax_roc.set_ylim(-0.01, 1.01)

    best = results[0]
    preds = best.estimator.predict(test.X)
    cm = confusion_matrix(test.y, preds)
    ConfusionMatrixDisplay(cm).plot(ax=ax_cm, colorbar=False, cmap="Blues")
    ax_cm.set_title(f"Confusion matrix, best ({best.name})")

    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def confusion_figure(result: ModelResult, test: TabularData, out_path: str | Path) -> Path:
    """Draw a standalone confusion matrix for one fitted model."""
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(
        result.estimator, test.X, test.y, ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(f"Confusion matrix ({result.name})")
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def importance_figure(result: ModelResult, out_path: str | Path, top_k: int = 15) -> Path:
    """Draw a horizontal bar chart of the top feature importances.

    Args:
        result: A model result whose fitted estimator exposes importances.
        out_path: Destination PNG path.
        top_k: Number of top features to show.

    Returns:
        The path written.
    """
    ranked = extract_importances(result.estimator, top_k=top_k)
    names = [f.feature for f in ranked][::-1]
    values = [f.importance for f in ranked][::-1]

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.barh(np.arange(len(names)), values, color="#3b6ea5")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {len(names)} features ({result.name})")
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
