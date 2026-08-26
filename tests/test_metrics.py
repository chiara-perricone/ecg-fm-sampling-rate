"""Metriche per etichetta e macro, incluso il comportamento sulle indefinite.

Il punto delicato non e' il calcolo — quello lo fa sklearn — ma cosa succede
alle colonne senza entrambe le classi. PTB-XL ne ha 22 sotto i dieci positivi
(§5.1), e una macro che le imputasse a 0,5 sarebbe una macro diversa da quella
dichiarata in §7.
"""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from ecgres.metrics import defined_label_fraction, macro_auroc, per_label_scores


def _case():
    y = np.array(
        [
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [1, 1, 0, 1],
            [0, 0, 0, 1],
            [1, 0, 0, 1],
        ]
    )
    p = np.array(
        [
            [0.9, 0.1, 0.5, 0.2],
            [0.2, 0.8, 0.5, 0.4],
            [0.7, 0.6, 0.5, 0.6],
            [0.1, 0.3, 0.5, 0.8],
            [0.8, 0.2, 0.5, 0.1],
        ]
    )
    return y, p  # colonna 2 tutta zeri, colonna 3 tutta uni


def test_per_label_scores_leaves_undefined_columns_as_nan():
    """Colonne senza entrambe le classi: NaN, non 0,5 (§7)."""
    y, p = _case()
    got = per_label_scores(y, p)
    assert np.isfinite(got["auroc"][:2]).all()
    assert np.isnan(got["auroc"][2]) and np.isnan(got["auroc"][3])
    assert np.isnan(got["auprc"][2]) and np.isnan(got["auprc"][3])


def test_per_label_scores_counts_positives_even_when_undefined():
    """Il supporto va riportato comunque: dice al lettore quanto pesare il resto."""
    y, p = _case()
    got = per_label_scores(y, p)
    assert got["n_positive"].tolist() == [3, 2, 0, 5]


def test_per_label_scores_agrees_with_sklearn():
    y, p = _case()
    got = per_label_scores(y, p)
    for j in (0, 1):
        assert got["auroc"][j] == pytest.approx(roc_auc_score(y[:, j], p[:, j]))
        assert got["auprc"][j] == pytest.approx(average_precision_score(y[:, j], p[:, j]))


def test_macro_auroc_is_the_mean_of_the_defined_columns():
    """La macro non deve diluirsi sulle colonne che ha appena escluso."""
    y, p = _case()
    got = per_label_scores(y, p)
    assert macro_auroc(y, p) == pytest.approx(np.nanmean(got["auroc"]))
    assert defined_label_fraction(y) == pytest.approx(0.5)


def test_macro_auroc_is_nan_when_nothing_is_defined():
    y = np.ones((4, 3), dtype=int)
    assert np.isnan(macro_auroc(y, np.random.default_rng(0).random((4, 3))))


def test_perfect_and_inverted_scores():
    y = np.array([[0], [0], [1], [1]])
    assert macro_auroc(y, np.array([[0.1], [0.2], [0.8], [0.9]])) == pytest.approx(1.0)
    assert macro_auroc(y, np.array([[0.9], [0.8], [0.2], [0.1]])) == pytest.approx(0.0)
