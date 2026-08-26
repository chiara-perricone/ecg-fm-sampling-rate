"""Uncertainty quantification for model comparison on a fixed test set.

A benchmark score is a point estimate. A comparison between two models is a
hypothesis test. This module treats them as such.

Design notes
------------
* Comparisons use a *paired* bootstrap: both models are evaluated on the same
  test patients, so the case resampling must be shared. Using two independent
  bootstraps would inflate the variance of the difference and hide real gaps.
* Resampling is at the level of the independent unit (one row = one patient
  record). PTB-XL ships patient-stratified folds, so rows in the held-out fold
  are independent across patients.
* p-values from a bootstrap of the difference are two-sided and computed by
  inverting the CI, which avoids assuming a null distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

MetricFn = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class Estimate:
    """A metric with a bootstrap confidence interval."""

    name: str
    point: float
    lo: float
    hi: float
    alpha: float

    def __str__(self) -> str:
        return f"{self.name}: {self.point:.4f} [{self.lo:.4f}, {self.hi:.4f}]"


@dataclass(frozen=True)
class Comparison:
    """A paired difference between two models, with CI and p-value."""

    name_a: str
    name_b: str
    diff: float
    lo: float
    hi: float
    p_value: float
    alpha: float

    @property
    def crosses_zero(self) -> bool:
        return self.lo <= 0.0 <= self.hi

    def __str__(self) -> str:
        verdict = "n.s." if self.crosses_zero else "excludes 0"
        return (
            f"{self.name_a} - {self.name_b}: {self.diff:+.4f} "
            f"[{self.lo:+.4f}, {self.hi:+.4f}] p={self.p_value:.4f} ({verdict})"
        )


def bootstrap_indices(n: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Shared case-resampling index matrix, shape (n_boot, n).

    Public because more than one comparison needs the *same* resamples. Section
    3 evaluates three seeds on one test set: giving each seed its own resamples
    would treat as independent what is in fact the same set of patients, and
    would widen the interval by variation that does not exist.
    """
    return rng.integers(0, n, size=(n_boot, n))


def percentile_ci(draws: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile interval of a bootstrap distribution, dropping undefined draws.

    Separated out because section 8 needs intervals at two different alphas over
    the same draws: the per-comparison 95% interval and the Bonferroni
    simultaneous one at alpha/m. Recomputing the draws for the second would cost
    an hour and would not change them.
    """
    draws = np.asarray(draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        raise RuntimeError("all bootstrap replicates were undefined")
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def p_value_from_draws(draws: np.ndarray) -> float:
    """Two-sided p-value by interval inversion, with add-one smoothing.

    The smallest alpha at which the percentile interval would still exclude
    zero. Add-one smoothing keeps the value strictly positive, so that a
    difference no replicate crossed does not report p = 0, which no finite
    bootstrap can support.
    """
    draws = np.asarray(draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        raise RuntimeError("all bootstrap replicates were undefined")
    n = draws.size
    below = (np.sum(draws <= 0.0) + 1) / (n + 1)
    above = (np.sum(draws >= 0.0) + 1) / (n + 1)
    return float(min(1.0, 2.0 * min(below, above)))


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: MetricFn,
    *,
    name: str = "metric",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Estimate:
    """Percentile bootstrap CI for a single model's metric."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same number of rows")

    rng = np.random.default_rng(seed)
    idx = bootstrap_indices(len(y_true), n_boot, rng)

    point = metric_fn(y_true, y_score)
    draws = np.array([metric_fn(y_true[i], y_score[i]) for i in idx], dtype=float)
    lo, hi = percentile_ci(draws, alpha)
    return Estimate(name=name, point=float(point), lo=lo, hi=hi, alpha=alpha)


def paired_bootstrap(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    metric_fn: MetricFn,
    *,
    name_a: str = "A",
    name_b: str = "B",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Comparison:
    """Paired bootstrap of metric(A) - metric(B) on a shared test set."""
    y_true = np.asarray(y_true)
    score_a = np.asarray(score_a)
    score_b = np.asarray(score_b)
    if not (len(y_true) == len(score_a) == len(score_b)):
        raise ValueError("y_true, score_a and score_b must have the same number of rows")

    rng = np.random.default_rng(seed)
    idx = bootstrap_indices(len(y_true), n_boot, rng)

    diff = metric_fn(y_true, score_a) - metric_fn(y_true, score_b)
    draws = np.array(
        [metric_fn(y_true[i], score_a[i]) - metric_fn(y_true[i], score_b[i]) for i in idx],
        dtype=float,
    )
    lo, hi = percentile_ci(draws, alpha)
    p = p_value_from_draws(draws)

    return Comparison(
        name_a=name_a,
        name_b=name_b,
        diff=float(diff),
        lo=float(lo),
        hi=float(hi),
        p_value=p,
        alpha=alpha,
    )


def holm(p_values: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values.

    Controls the family-wise error rate without assuming independence, which
    matters here because comparisons on the same test set are correlated.
    """
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)

    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def summarise(comparisons: Sequence[Comparison]) -> list[tuple[Comparison, float]]:
    """Attach Holm-adjusted p-values to a family of comparisons."""
    adjusted = holm([c.p_value for c in comparisons])
    return list(zip(comparisons, adjusted))
