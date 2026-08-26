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


# --------------------------------------------------------------------------- #
# §8.1 — confronto fra rate
# --------------------------------------------------------------------------- #


def _group(tmp_path, name, y, n_seeds=3, signal=1.0, spread=1.0, seed=0):
    """``n_seeds`` run sullo stesso test set, con predizioni piu' o meno buone."""
    rng = np.random.default_rng(seed)
    paths = [
        _write(
            tmp_path,
            f"{name}-s{i}",
            y,
            1 / (1 + np.exp(-(signal * y + spread * rng.normal(size=y.shape)))),
        )
        for i in range(n_seeds)
    ]
    return load_predictions(paths)[0]


def test_ensemble_averages_probabilities_not_metrics(tmp_path):
    """§8.1 media le predizioni; §3 media le metriche (§10 voce 19)."""
    from ecgres.aggregate import ensemble

    y, _ = _case()
    runs = _group(tmp_path, "r", y, n_seeds=3)
    got = ensemble(runs)
    assert np.allclose(got, np.mean([r.y_prob for r in runs], axis=0))
    # L'ensemble non e' la media delle metriche: se lo fosse, la voce 19 non
    # avrebbe motivo di esistere.
    mean_of_metrics = np.mean([macro_auroc_fast(y, r.y_prob) for r in runs])
    assert macro_auroc_fast(y, got) != pytest.approx(mean_of_metrics, abs=1e-6)


def test_compare_groups_produces_every_pair(tmp_path):
    from ecgres.aggregate import compare_groups

    y, _ = _case()
    groups = {
        str(fs): _group(tmp_path, f"fs{fs}", y, seed=k)
        for k, fs in enumerate((100, 240, 250, 500))
    }
    got = compare_groups(y, groups, n_boot=200, seed=1)
    assert [g.name for g in got.groups] == ["100", "240", "250", "500"]
    assert len(got.pairs) == 6  # le sei coppie di §8.1
    assert {(p.a, p.b) for p in got.pairs} == {
        ("100", "240"), ("100", "250"), ("100", "500"),
        ("240", "250"), ("240", "500"), ("250", "500"),
    }


def test_simultaneous_interval_is_wider_than_the_unadjusted_one(tmp_path):
    """Voce 20: l'intervallo simultaneo e' conservativo, e deve vedersi."""
    from ecgres.aggregate import compare_groups

    y, _ = _case()
    groups = {str(k): _group(tmp_path, f"g{k}", y, seed=k) for k in range(4)}
    got = compare_groups(y, groups, n_boot=2000, seed=2)
    for pair in got.pairs:
        assert pair.lo_simultaneous <= pair.lo
        assert pair.hi_simultaneous >= pair.hi


def test_holm_adjusted_p_is_never_below_the_raw_one(tmp_path):
    from ecgres.aggregate import compare_groups

    y, _ = _case()
    groups = {str(k): _group(tmp_path, f"h{k}", y, seed=k) for k in range(4)}
    got = compare_groups(y, groups, n_boot=500, seed=3)
    for pair in got.pairs:
        assert pair.p_holm >= pair.p_raw - 1e-12


def test_pairs_are_differences_of_shared_draws(tmp_path):
    """L'accoppiamento: la differenza fra due gruppi identici e' esattamente zero.

    Con bootstrap indipendenti per coppia questa differenza sarebbe rumore, e
    l'intervallo non collasserebbe su zero.
    """
    from ecgres.aggregate import compare_groups

    y, _ = _case()
    runs = _group(tmp_path, "same", y)
    got = compare_groups(y, {"a": runs, "b": runs}, n_boot=200, seed=4)
    pair = got.pairs[0]
    assert pair.diff == pytest.approx(0.0, abs=1e-12)
    assert (pair.lo, pair.hi) == pytest.approx((0.0, 0.0), abs=1e-12)


def test_delta_is_the_largest_absolute_difference(tmp_path):
    from ecgres.aggregate import compare_groups

    y, _ = _case()
    groups = {str(k): _group(tmp_path, f"d{k}", y, signal=k, seed=k) for k in range(3)}
    got = compare_groups(y, groups, n_boot=200, seed=5)
    assert abs(got.max_abs_diff.diff) == max(abs(p.diff) for p in got.pairs)


def _fake_group(name, null_contrasts=(0.001, -0.001), per_run=None):
    from ecgres.aggregate import GroupResult

    return GroupResult(
        name, 0.9, 0.88, 0.92, per_run or {"s0": 0.90, "s1": 0.91},
        null_contrasts=tuple(null_contrasts),
    )


def _fake_pair(diff, lo, hi, p_holm=0.01):
    from ecgres.aggregate import PairResult

    return PairResult(
        a="a", b="b", diff=diff, lo=lo, hi=hi, p_raw=p_holm / 3, p_holm=p_holm,
        lo_simultaneous=lo - 0.005, hi_simultaneous=hi + 0.005,
    )


def test_interpret_delta_checks_detectability_before_size():
    """§8.2: prima se l'effetto esiste, poi quanto e' grande.

    Un Δ enorme il cui intervallo contiene zero resta "non rilevabile": chiamarlo
    grande sarebbe leggere la dimensione di qualcosa che non c'e'.
    """
    from ecgres.aggregate import RateComparison, interpret_delta

    groups = [_fake_group("a")]
    huge_but_null = _fake_pair(0.05, -0.01, 0.11, p_holm=0.9)
    got = interpret_delta(RateComparison(groups, [huge_but_null], 100, 0.05))
    assert got["verdict"].startswith("non rilevabile")

    clear = _fake_pair(0.025, 0.02, 0.03)
    got = interpret_delta(RateComparison(groups, [clear], 100, 0.05))
    assert "scavalcare" in got["verdict"]
    assert got["gap_typical"] == 0.006 and got["gap_largest"] == 0.019


def test_the_comparator_is_the_null_contrast_not_the_seed_spread():
    """Voce 21, ed e' la differenza che cambia la conclusione.

    Δ = 0,010 con un contrasto nullo di 0,002 e' rilevabile; la dispersione fra
    run singoli e' 0,05, e col vecchio metro lo stesso Δ sarebbe stato
    dichiarato invisibile. I due numeri non misurano la stessa cosa.
    """
    from ecgres.aggregate import RateComparison, interpret_delta

    group = _fake_group(
        "a",
        null_contrasts=(0.002, -0.002, 0.001),
        per_run={"s0": 0.88, "s1": 0.90, "s2": 0.93},
    )
    got = interpret_delta(RateComparison([group], [_fake_pair(0.010, 0.006, 0.014)], 100, 0.05))

    assert got["comparator"] == "null_contrast"
    assert got["comparator_value"] == pytest.approx(0.002)
    assert got["max_seed_spread"] == pytest.approx(0.05)  # riportato, non usato
    assert not got["verdict"].startswith("non rilevabile")


def test_delta_below_the_null_contrast_is_not_detectable():
    """Il confronto centrale del lavoro: rate contro riseed, alla pari."""
    from ecgres.aggregate import RateComparison, interpret_delta

    group = _fake_group("a", null_contrasts=(0.008, -0.009, 0.007))
    got = interpret_delta(RateComparison([group], [_fake_pair(0.004, 0.001, 0.007)], 100, 0.05))
    assert got["comparator_value"] == pytest.approx(0.008)
    assert got["verdict"].startswith("non rilevabile")


def test_comparator_falls_back_and_says_so():
    """Un gruppo con un solo seed non ha contrasto nullo: va dichiarato."""
    from ecgres.aggregate import RateComparison, interpret_delta

    group = _fake_group("a", null_contrasts=())
    got = interpret_delta(RateComparison([group], [_fake_pair(0.03, 0.02, 0.04)], 100, 0.05))
    assert got["comparator"].startswith("seed_spread")
    assert not np.isfinite(got["null_contrast"])


def test_null_contrast_enumerates_equal_sized_disjoint_splits(tmp_path):
    """Cinque seed danno le quindici coppie 2 contro 2 di §8.2 (voce 22)."""
    from ecgres.aggregate import seed_null_contrast

    y, _ = _case()
    runs = _group(tmp_path, "n", y, n_seeds=5)
    values = seed_null_contrast(runs)
    assert len(values) == 15
    assert all(np.isfinite(values))


def test_null_contrast_does_not_measure_ensemble_size(tmp_path):
    """Il test che avrebbe intercettato la voce 21.

    Con sottoinsiemi di dimensione diversa il "contrasto nullo" era in realta'
    il vantaggio di mediare un modello in piu': effetto vero, di segno fisso, e
    grande abbastanza da rendere non rilevabile qualunque Delta.
    """
    from ecgres.aggregate import ensemble, seed_null_contrast
    from ecgres.metrics import macro_auroc_fast

    y, _ = _case()
    runs = _group(tmp_path, "sz", y, n_seeds=5, spread=2.5)
    size_effect = abs(
        macro_auroc_fast(y, ensemble(runs[:3])) - macro_auroc_fast(y, ensemble(runs[:2]))
    )
    contrast = float(np.median(np.abs(seed_null_contrast(runs))))
    assert size_effect > 0.01, "dati di prova troppo puliti per essere informativi"
    assert contrast < size_effect / 2


def test_null_contrast_of_identical_seeds_is_zero(tmp_path):
    """Se i seed non si muovessero, il metro sarebbe zero — ed e' giusto cosi'."""
    from ecgres.aggregate import load_predictions, seed_null_contrast

    y, p = _case()
    runs = load_predictions(
        [_write(tmp_path, f"same{i}", y, p) for i in range(4)]
    )[0]
    assert seed_null_contrast(runs) == pytest.approx([0.0] * 3, abs=1e-12)


def test_matched_factor_is_reported_but_not_used(tmp_path):
    """Voce 22: il verdetto usa il metro grezzo, il riscalato sta accanto."""
    from ecgres.aggregate import RateComparison, interpret_delta, matching_factor

    assert matching_factor(5) == pytest.approx(np.sqrt(2 / 5))
    group = _fake_group("a", null_contrasts=(0.01, -0.01),
                        per_run={f"s{i}": 0.9 for i in range(5)})
    got = interpret_delta(RateComparison([group], [_fake_pair(0.008, 0.004, 0.012)], 100, 0.05))
    assert got["comparator_value"] == pytest.approx(0.01)  # grezzo
    assert got["null_contrast_matched"] == pytest.approx(0.01 * np.sqrt(2 / 5))
    assert got["verdict"].startswith("non rilevabile")  # 0.008 < 0.010


def test_compare_groups_computes_the_null_contrast(tmp_path):
    from ecgres.aggregate import compare_groups

    y, _ = _case()
    groups = {str(k): _group(tmp_path, f"nc{k}", y, n_seeds=5, seed=k) for k in range(2)}
    got = compare_groups(y, groups, n_boot=100, seed=6)
    assert all(len(g.null_contrasts) == 15 for g in got.groups)
    assert np.isfinite(got.null_contrast)
    assert got.null_contrast > 0


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
