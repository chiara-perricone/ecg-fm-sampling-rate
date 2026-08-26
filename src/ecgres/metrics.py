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
from scipy.stats import rankdata
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


def macro_auroc_fast(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Same quantity as :func:`macro_auroc`, computed in one pass over columns.

    AUROC equals the Mann-Whitney statistic, so it can be read off the midranks
    of the scores instead of by sweeping a threshold. ``scipy.stats.rankdata``
    ranks every label column at once, which turns 71 calls into one.

    This exists for bootstrapping. Section 3 resamples 10,000 times over three
    seeds, i.e. 2.1 million per-label evaluations; measured on a 2198 x 71 test
    set that is about 40 minutes through scikit-learn against 5 minutes here.
    The slow implementation is kept as the definition and this one is pinned to
    it by ``test_metrics.py`` — including on tied scores, where the midrank
    convention is exactly what makes the two agree.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
        y_score = y_score[:, None]

    n = y_true.shape[0]
    n_pos = y_true.sum(axis=0).astype(np.float64)
    n_neg = n - n_pos
    defined = (n_pos > 0) & (n_neg > 0)
    if not defined.any():
        return float("nan")

    ranks = rankdata(y_score, axis=0)
    rank_sum = (ranks * (y_true > 0)).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc[defined].mean())


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
