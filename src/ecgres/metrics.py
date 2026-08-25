"""Metrics for multi-label ECG classification.

PTB-XL tasks are multi-label with severe class imbalance, especially the
43-class diagnostic setting. Under bootstrap resampling some label columns can
end up with a single class present, where AUROC is undefined. We drop those
columns for that replicate rather than imputing a value, and we record how
often it happens, because a metric that is silently undefined on rare classes
is exactly the kind of thing that makes benchmark tables misleading.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def macro_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Macro-averaged AUROC over label columns that are defined."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
        y_score = y_score[:, None]

    scores = []
    for j in range(y_true.shape[1]):
        col = y_true[:, j]
        if col.min() == col.max():
            continue  # undefined for this replicate
        scores.append(roc_auc_score(col, y_score[:, j]))
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def defined_label_fraction(y_true: np.ndarray) -> float:
    """Share of label columns with both classes present."""
    y_true = np.asarray(y_true)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    ok = [y_true[:, j].min() != y_true[:, j].max() for j in range(y_true.shape[1])]
    return float(np.mean(ok))
