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
from sklearn.metrics import average_precision_score, roc_auc_score


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


def per_label_scores(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, np.ndarray]:
    """AUROC and AUPRC for every label column, with NaN where undefined.

    Section 7 of the protocol requires per-label figures for every run, not only
    the macro summaries: a macro average hides which labels carry it, and the
    labels that carry it are the ones with the most support. ``n_positive`` is
    returned alongside so a reader can weigh each figure without recomputing it.

    AUROC is undefined for a column with a single class present; AUPRC is
    undefined without positives. Both are left as NaN rather than imputed, for
    the reason given in the module docstring.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
        y_score = y_score[:, None]

    n_labels = y_true.shape[1]
    auroc = np.full(n_labels, np.nan)
    auprc = np.full(n_labels, np.nan)
    positives = np.zeros(n_labels, dtype=int)

    for j in range(n_labels):
        col = y_true[:, j]
        positives[j] = int(col.sum())
        if col.min() == col.max():
            continue
        auroc[j] = roc_auc_score(col, y_score[:, j])
        auprc[j] = average_precision_score(col, y_score[:, j])

    return {"auroc": auroc, "auprc": auprc, "n_positive": positives}


def defined_label_fraction(y_true: np.ndarray) -> float:
    """Share of label columns with both classes present."""
    y_true = np.asarray(y_true)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    ok = [y_true[:, j].min() != y_true[:, j].max() for j in range(y_true.shape[1])]
    return float(np.mean(ok))
