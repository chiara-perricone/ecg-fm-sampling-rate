"""Aggregazione fra seed e verdetto di §3 (§10 voce 15).

Il test che vale la pena leggere e' ``test_the_same_resample_goes_to_every_run``:
dare a ciascun seed ricampionamenti propri restringerebbe l'intervallo
trattando come indipendente cio' che e' lo stesso insieme di pazienti, e nessun
altro controllo se ne accorgerebbe.
"""

import numpy as np
import pytest

from ecgres.aggregate import (
    STAGE0_TARGET,
    load_predictions,
    seed_mean_bootstrap,
    stage0_verdict,
)
from ecgres.metrics import macro_auroc_fast

N = 240
N_LABELS = 8
CLEAN = np.array([0, 1, 2, 3])


def _write(tmp_path, name, y_true, y_prob, ecg_id=None, clean=CLEAN):
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "predictions-macro_all.npz"
    np.savez_compressed(
        path,
        record_idx=np.arange(len(y_true)),
        ecg_id=np.arange(len(y_true)) if ecg_id is None else ecg_id,
        y_true=y_true.astype(np.uint8),
        y_prob=y_prob.astype(np.float32),
        clean_columns=np.asarray(clean, dtype=np.int64),
    )
    return path


def _case(seed=0, signal=1.0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=(N, N_LABELS))
    y[0] = 1
    y[1] = 0
    noise = rng.normal(size=(N, N_LABELS))
    return y, 1 / (1 + np.exp(-(signal * y + noise)))


# --------------------------------------------------------------------------- #
# Caricamento
# --------------------------------------------------------------------------- #


def test_load_predictions_reads_several_runs(tmp_path):
    y, p = _case()
    paths = [_write(tmp_path, f"run{i}", y, p + i * 1e-3) for i in range(3)]
    runs, columns = load_predictions(paths)
    assert [r.run_id for r in runs] == ["run0", "run1", "run2"]
    assert columns.tolist() == CLEAN.tolist()


def test_load_predictions_rejects_a_different_population(tmp_path):
    """Aggregare su record diversi darebbe numeri plausibili e senza senso."""
    y, p = _case()
    a = _write(tmp_path, "a", y, p)
    b = _write(tmp_path, "b", y, p, ecg_id=np.arange(N) + 5)
    with pytest.raises(ValueError, match="record diversi"):
        load_predictions([a, b])


def test_load_predictions_rejects_different_labels(tmp_path):
    y, p = _case()
    other = y.copy()
    other[10, 0] ^= 1
    a = _write(tmp_path, "a", y, p)
    b = _write(tmp_path, "b", other, p)
    with pytest.raises(ValueError, match="etichette diverse"):
        load_predictions([a, b])


def test_load_predictions_rejects_a_different_frozen_set(tmp_path):
    y, p = _case()
    a = _write(tmp_path, "a", y, p)
    b = _write(tmp_path, "b", y, p, clean=np.array([0, 1]))
    with pytest.raises(ValueError, match="insieme congelato"):
        load_predictions([a, b])


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #


def _runs(tmp_path, n=3):
    """Piu' seed sullo **stesso** test set: cambiano le predizioni, non le etichette."""
    y, _ = _case(seed=0)
    rng = np.random.default_rng(100)
    paths = [
        _write(tmp_path, f"run{i}", y, 1 / (1 + np.exp(-(y + rng.normal(size=y.shape)))))
        for i in range(n)
    ]
    return load_predictions(paths)[0]


def test_point_estimate_is_the_mean_over_runs(tmp_path):
    runs = _runs(tmp_path)
    got = seed_mean_bootstrap(runs, n_boot=100)
    expected = np.mean([macro_auroc_fast(r.y_true, r.y_prob) for r in runs])
    assert got.point == pytest.approx(expected, abs=1e-12)
    assert set(got.per_run) == {"run0", "run1", "run2"}


def test_interval_brackets_the_point_estimate(tmp_path):
    got = seed_mean_bootstrap(_runs(tmp_path), n_boot=400, seed=1)
    assert got.lo < got.point < got.hi


def test_the_same_resample_goes_to_every_run(tmp_path):
    """Duplicare un run non deve stringere l'intervallo.

    Con lo stesso ricampionamento per tutti, la media di due copie identiche e'
    la copia stessa, quindi l'intervallo coincide **esattamente** con quello di
    un run solo. Se ciascun run avesse ricampionamenti propri, mediare due
    rumori indipendenti restringerebbe l'intervallo e questo test fallirebbe.
    """
    y, p = _case()
    one = load_predictions([_write(tmp_path, "a", y, p)])[0]
    two = load_predictions(
        [_write(tmp_path, "a", y, p), _write(tmp_path, "b", y, p)]
    )[0]
    single = seed_mean_bootstrap(one, n_boot=300, seed=7)
    doubled = seed_mean_bootstrap(two, n_boot=300, seed=7)
    assert doubled.lo == pytest.approx(single.lo, abs=1e-12)
    assert doubled.hi == pytest.approx(single.hi, abs=1e-12)


def test_clean_columns_restrict_the_metric(tmp_path):
    runs = _runs(tmp_path)
    everything = seed_mean_bootstrap(runs, n_boot=100)
    subset = seed_mean_bootstrap(runs, columns=CLEAN, n_boot=100)
    assert everything.point != pytest.approx(subset.point, abs=1e-9)


def test_seed_spread_is_reported_and_is_not_the_interval(tmp_path):
    """Voce 15: lo scarto fra seed sta **accanto** al CI, non dentro."""
    got = seed_mean_bootstrap(_runs(tmp_path), n_boot=200)
    values = list(got.per_run.values())
    assert got.seed_spread == pytest.approx(max(values) - min(values))
    assert got.seed_std > 0


def test_bootstrap_is_reproducible(tmp_path):
    runs = _runs(tmp_path)
    a = seed_mean_bootstrap(runs, n_boot=200, seed=5)
    b = seed_mean_bootstrap(runs, n_boot=200, seed=5)
    assert (a.lo, a.hi) == (b.lo, b.hi)


# --------------------------------------------------------------------------- #
# Verdetto
# --------------------------------------------------------------------------- #


def _estimate(point, lo, hi):
    from ecgres.aggregate import SeedMean

    return SeedMean(point=point, lo=lo, hi=hi, per_run={"a": point}, n_boot=10)


def test_verdict_accepts_only_when_both_conditions_hold():
    assert _verdict(0.941, 0.935, 0.947)["accepted"]
    # target fuori dal CI, stima vicinissima: intervallo stretto, molta potenza
    near = _verdict(0.9385, 0.9370, 0.9400)
    assert near["point_within_tolerance"] and not near["ci_contains_target"]
    assert not near["accepted"]
    # stima lontana ma CI largo: il contrario
    wide = _verdict(0.920, 0.900, 0.950)
    assert wide["ci_contains_target"] and not wide["point_within_tolerance"]
    assert not wide["accepted"]


def _verdict(point, lo, hi):
    return stage0_verdict(_estimate(point, lo, hi))


def test_the_analysis_path_does_not_import_torch():
    """``stage0.py`` deve girare dove non c'e' torch.

    L'analisi legge file gia' scritti: non serve GPU, non servono i 12,5 GB di
    cache, e non deve servire nemmeno un ambiente di training. E' facilissimo
    romperlo importando un aiutante dal modulo sbagliato, e nulla se ne
    accorgerebbe finche' non serve davvero.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    code = (
        "import sys, ecgres.aggregate, ecgres.report, ecgres.runs, ecgres.stats; "
        "sys.exit('torch' in sys.modules)"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
    )
    assert done.returncode == 0, done.stderr or "torch importato dal percorso di analisi"


def test_verdict_reports_the_signed_difference():
    got = _verdict(0.930, 0.925, 0.945)
    assert got["target"] == STAGE0_TARGET
    assert got["difference"] == pytest.approx(0.930 - STAGE0_TARGET)
