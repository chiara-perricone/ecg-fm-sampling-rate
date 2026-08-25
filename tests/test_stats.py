"""Tests for the uncertainty machinery.

These are behavioural tests on synthetic data with a known answer: if the
statistics cannot tell a genuinely better model from an identical one, nothing
downstream is trustworthy.
"""

from __future__ import annotations

import numpy as np
import pytest

from ecgres.metrics import macro_auroc
from ecgres.stats import bootstrap_ci, holm, paired_bootstrap, summarise


def make_data(n=800, n_labels=5, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.3, size=(n, n_labels))
    signal = y + rng.normal(0, 1.0, size=y.shape)
    return y, signal


def test_bootstrap_ci_contains_point_estimate():
    y, s = make_data()
    est = bootstrap_ci(y, s, macro_auroc, n_boot=300, seed=1)
    assert est.lo < est.point < est.hi
    assert 0.5 < est.point < 1.0


def test_identical_models_are_not_significant():
    y, s = make_data()
    cmp = paired_bootstrap(y, s, s.copy(), macro_auroc, n_boot=300, seed=1)
    assert cmp.diff == pytest.approx(0.0, abs=1e-12)
    assert cmp.crosses_zero
    assert cmp.p_value > 0.5


def test_clearly_better_model_is_detected():
    y, weak = make_data()
    rng = np.random.default_rng(7)
    strong = y + rng.normal(0, 0.35, size=y.shape)  # much cleaner signal
    cmp = paired_bootstrap(y, strong, weak, macro_auroc, n_boot=300, seed=1)
    assert cmp.diff > 0
    assert not cmp.crosses_zero
    assert cmp.p_value < 0.05


def test_paired_bootstrap_is_tighter_than_naive_difference():
    """The whole point of pairing: shared resampling shrinks the CI."""
    y, weak = make_data()
    rng = np.random.default_rng(7)
    strong = y + rng.normal(0, 0.5, size=y.shape)

    paired = paired_bootstrap(y, strong, weak, macro_auroc, n_boot=400, seed=1)
    a = bootstrap_ci(y, strong, macro_auroc, n_boot=400, seed=1)
    b = bootstrap_ci(y, weak, macro_auroc, n_boot=400, seed=2)
    unpaired_width = (a.hi - a.lo) + (b.hi - b.lo)

    assert (paired.hi - paired.lo) < unpaired_width


def test_holm_is_monotone_and_conservative():
    p = [0.001, 0.02, 0.04, 0.6]
    adj = holm(p)
    assert np.all(adj >= np.asarray(p))
    assert np.all(np.diff(adj[np.argsort(p)]) >= -1e-12)
    assert adj[0] == pytest.approx(0.004)


def test_holm_caps_at_one():
    adj = holm([0.5, 0.6, 0.9])
    assert np.all(adj <= 1.0)


def test_summarise_preserves_order():
    y, s = make_data()
    rng = np.random.default_rng(3)
    comparisons = [
        paired_bootstrap(y, y + rng.normal(0, sd, y.shape), s, macro_auroc,
                         name_a=f"m{k}", name_b="base", n_boot=200, seed=k)
        for k, sd in enumerate([0.3, 1.0, 2.0])
    ]
    out = summarise(comparisons)
    assert [c.name_a for c, _ in out] == ["m0", "m1", "m2"]
    assert all(adj >= c.p_value - 1e-12 for c, adj in out)
