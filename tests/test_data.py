"""Test di ``ecgres.data`` contro un mini-PTB-XL sintetico.

Il fixture scrive record WFDB veri (12 derivazioni, 10 s, 500 Hz) piu'
``ptbxl_database.csv`` e ``scp_statements.csv`` nella stessa forma del dataset
reale, cosi' il percorso testato e' quello di produzione e non un mock.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import wfdb

from ecgres import data as D

N_RECORDS = 12
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
STATEMENTS = ["NORM", "MI", "STTC", "CD", "HYP", "AFIB", "PVC"]


def _synthetic_signal(rng: np.random.Generator) -> np.ndarray:
    """Segnale a banda stretta piu' interferenza di rete a 50 Hz."""
    t = np.arange(D.SOURCE_FS * int(D.RECORD_SECONDS)) / D.SOURCE_FS
    base = np.zeros((D.N_LEADS, t.size))
    for lead in range(D.N_LEADS):
        for freq in (1.2, 4.0, 9.0):
            base[lead] += rng.uniform(0.3, 1.0) * np.sin(
                2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi)
            )
        base[lead] += 0.4 * np.sin(2 * np.pi * 50.0 * t + rng.uniform(0, 2 * np.pi))
    return base * 0.5


@pytest.fixture(scope="module")
def mini_ptbxl(tmp_path_factory) -> tuple:
    root = tmp_path_factory.mktemp("mini-ptb-xl")
    rec_dir = root / "records500" / "00000"
    rec_dir.mkdir(parents=True)

    rng = np.random.default_rng(20260826)
    rows = []
    for i in range(N_RECORDS):
        name = f"{i + 1:05d}_hr"
        signal = _synthetic_signal(rng)
        wfdb.wrsamp(
            record_name=name,
            fs=D.SOURCE_FS,
            units=["mV"] * D.N_LEADS,
            sig_name=LEADS,
            p_signal=signal.T,
            fmt=["16"] * D.N_LEADS,
            write_dir=str(rec_dir),
        )
        fold = (i % 10) + 1
        codes = {"NORM": 100.0} if i % 3 == 0 else {"MI": 0.0, "STTC": 50.0}
        rows.append(
            {
                "ecg_id": i + 1,
                "patient_id": 1000 + i,  # un paziente per record: nessun leakage
                "strat_fold": fold,
                "scp_codes": repr(codes),
                "filename_hr": f"records500/00000/{name}",
                "filename_lr": f"records100/00000/{i + 1:05d}_lr",
            }
        )

    pd.DataFrame(rows).to_csv(root / "ptbxl_database.csv", index=False)
    pd.DataFrame(
        {"diagnostic": [1] * len(STATEMENTS)}, index=pd.Index(STATEMENTS, name="")
    ).to_csv(root / "scp_statements.csv")

    meta = D.load_metadata(root)
    return root, meta


@pytest.fixture(scope="module")
def cache_100(mini_ptbxl, tmp_path_factory) -> D.SignalCache:
    root, meta = mini_ptbxl
    cache = D.SignalCache(root, tmp_path_factory.mktemp("cache100"), meta, fs=100)
    cache.build()
    return cache


@pytest.fixture(scope="module")
def cache_500(mini_ptbxl, tmp_path_factory) -> D.SignalCache:
    root, meta = mini_ptbxl
    cache = D.SignalCache(root, tmp_path_factory.mktemp("cache500"), meta, fs=500)
    cache.build()
    return cache


# -- costanti e griglia temporale ------------------------------------------


def test_resample_factors_are_exact():
    for fs, (up, down) in D.RESAMPLE_FACTORS.items():
        assert D.SOURCE_FS * up == fs * down


def test_window_length_per_rate():
    got = {fs: D.n_samples_for(D.WINDOW_SECONDS, fs) for fs in D.RATES}
    assert got == {100: 250, 240: 600, 250: 625, 500: 1250}


def test_n_samples_for_rejects_non_integer():
    with pytest.raises(ValueError):
        D.n_samples_for(2.5, 111)


def test_crop_grid_is_integer_at_every_rate():
    grid = D.crop_start_grid()
    assert grid[0] == 0.0 and grid[-1] == pytest.approx(7.5)
    assert grid.size == 76
    for fs in D.RATES:
        exact = grid * fs
        assert np.allclose(exact, np.round(exact), atol=1e-6)


def test_inference_crops_tile_the_record():
    starts = D.inference_crop_starts()
    assert np.allclose(starts, [0.0, 2.5, 5.0, 7.5])
    assert starts[-1] + D.WINDOW_SECONDS == pytest.approx(D.RECORD_SECONDS)


# -- metadati ed etichette -------------------------------------------------


def test_metadata_parses_scp_codes(mini_ptbxl):
    _, meta = mini_ptbxl
    assert len(meta) == N_RECORDS
    assert isinstance(meta["scp_codes"].iloc[0], dict)


def test_labels_use_presence_not_likelihood(mini_ptbxl):
    root, meta = mini_ptbxl
    y, labels = D.load_label_matrix(root, meta)
    assert y.shape == (N_RECORDS, len(STATEMENTS))
    mi = labels.index("MI")
    # il record 2 ha MI con likelihood 0.0: conta comunque come positivo
    assert y[1, mi] == 1


def test_labels_respect_frozen_set(mini_ptbxl):
    root, meta = mini_ptbxl
    frozen = ["NORM", "STTC"]
    y, labels = D.load_label_matrix(root, meta, label_set=frozen)
    assert labels == frozen
    assert y.shape == (N_RECORDS, 2)


# -- split -----------------------------------------------------------------


def test_splits_follow_official_folds(mini_ptbxl):
    _, meta = mini_ptbxl
    splits = D.get_splits(meta)
    folds = meta["strat_fold"].to_numpy()
    assert set(folds[splits.train]) <= set(range(1, 9))
    assert set(folds[splits.val]) == {9}
    assert set(folds[splits.test]) == {10}
    total = splits.train.size + splits.val.size + splits.test.size
    assert total == N_RECORDS


def test_splits_detect_patient_leakage(mini_ptbxl):
    _, meta = mini_ptbxl
    leaky = meta.copy()
    col = leaky.columns.get_loc("patient_id")
    train_row = int(np.flatnonzero(leaky["strat_fold"].to_numpy() == 1)[0])
    test_row = int(np.flatnonzero(leaky["strat_fold"].to_numpy() == 10)[0])
    leaky.iloc[train_row, col] = leaky["patient_id"].iloc[test_row]
    with pytest.raises(ValueError, match="pazienti condivisi"):
        D.get_splits(leaky)


# -- filtro e ricampionamento ----------------------------------------------


def _power_at(x: np.ndarray, freq: float, fs: int) -> float:
    spec = np.fft.rfft(x, axis=-1)
    bins = np.fft.rfftfreq(x.shape[-1], d=1 / fs)
    k = int(np.argmin(np.abs(bins - freq)))
    return float(np.abs(spec[..., k]).mean())


def test_notch_attenuates_mains(mini_ptbxl):
    rng = np.random.default_rng(1)
    x = _synthetic_signal(rng)
    before = _power_at(x, 50.0, D.SOURCE_FS)
    after = _power_at(D.notch_50hz(x, D.SOURCE_FS), 50.0, D.SOURCE_FS)
    assert after < 0.1 * before


def test_notch_preserves_signal_band(mini_ptbxl):
    rng = np.random.default_rng(2)
    x = _synthetic_signal(rng)
    y = D.notch_50hz(x, D.SOURCE_FS)
    for freq in (1.2, 4.0, 9.0):
        assert _power_at(y, freq, D.SOURCE_FS) > 0.9 * _power_at(x, freq, D.SOURCE_FS)


def test_notch_refuses_wrong_rate():
    with pytest.raises(ValueError, match="prima del ricampionamento"):
        D.notch_50hz(np.zeros((D.N_LEADS, 1000)), fs=100)


def test_resample_lengths(mini_ptbxl):
    rng = np.random.default_rng(3)
    x = _synthetic_signal(rng)
    expected = {100: 1000, 240: 2400, 250: 2500, 500: 5000}
    for fs, n in expected.items():
        assert D.resample_to(x, fs).shape[-1] == n


def test_preprocess_applies_notch_before_resampling():
    rng = np.random.default_rng(4)
    x = _synthetic_signal(rng)
    manual = D.resample_to(D.notch_50hz(x, D.SOURCE_FS), 100).astype(np.float32)
    assert np.allclose(D.preprocess_record(x, 100), manual, atol=0)


def test_preprocess_notch_flag_changes_output():
    rng = np.random.default_rng(5)
    x = _synthetic_signal(rng)
    assert not np.allclose(
        D.preprocess_record(x, 100, notch=True),
        D.preprocess_record(x, 100, notch=False),
    )


# -- cache -----------------------------------------------------------------


def test_cache_shape_and_dtype(cache_100):
    assert cache_100.array.shape == (N_RECORDS, D.N_LEADS, 1000)
    assert cache_100.array.dtype == np.float32


def test_cache_build_is_idempotent(cache_100):
    stamp = cache_100.array_path.stat().st_mtime_ns
    cache_100.build()
    assert cache_100.array_path.stat().st_mtime_ns == stamp


def test_cache_rejects_changed_record_order(cache_100, mini_ptbxl):
    root, meta = mini_ptbxl
    shuffled = meta.iloc[::-1]
    other = D.SignalCache(root, cache_100.cache_dir, shuffled, fs=100)
    with pytest.raises(FileNotFoundError, match="non coerente"):
        _ = other.array


def test_cache_filename_encodes_notch(mini_ptbxl, tmp_path):
    root, meta = mini_ptbxl
    a = D.SignalCache(root, tmp_path, meta, fs=100, notch=True)
    b = D.SignalCache(root, tmp_path, meta, fs=100, notch=False)
    assert a.array_path != b.array_path


def test_cache_rejects_records100_above_100hz(mini_ptbxl, tmp_path):
    root, meta = mini_ptbxl
    with pytest.raises(ValueError, match="8.3.2"):
        D.SignalCache(root, tmp_path, meta, fs=250, source="lr")


def test_cache_metadata_is_written(cache_100):
    info = json.loads(cache_100.meta_path.read_text())
    assert info["fs"] == 100 and info["notch"] is True


# -- finestratura ----------------------------------------------------------


def test_crop_off_grid_raises(cache_100):
    with pytest.raises(ValueError, match="griglia"):
        cache_100.crop([0], [0.123])


def test_crop_past_record_end_raises(cache_100):
    with pytest.raises(ValueError, match="fuori dal record"):
        cache_100.crop([0], [8.0])


def test_grid_index_rejects_on_sample_but_off_grid():
    """0,05 s da' 5 a 100 Hz e 12 a 240 Hz, ma 12,5 a 250 Hz.

    E' il caso che distingue le due condizioni: essere un campione intero a
    *questo* rate non basta, serve esserlo a tutti e quattro insieme.
    """
    assert 0.05 * 100 == 5 and 0.05 * 240 == 12  # integro a due rate su quattro
    for fs in D.RATES:
        with pytest.raises(ValueError, match="griglia"):
            D.grid_index(0.05, fs)


def test_grid_index_is_exact_on_the_grid():
    for t in D.crop_start_grid():
        for fs in D.RATES:
            assert D.grid_index(t, fs) / fs == pytest.approx(t, abs=1e-12)


def test_window_matches_crop(cache_100):
    """``crop`` e' un ciclo su ``window``: i due percorsi non devono divergere."""
    for start in (0.0, 2.5, 7.5):
        one = cache_100.window(1, start)
        many = cache_100.crop([1], [start]).signals[0]
        assert one.dtype == np.float32
        assert np.array_equal(one, many)


def test_window_returns_a_copy_not_a_view(cache_100):
    """Chi consuma la finestra la normalizza: una vista scriverebbe sulla cache."""
    x = cache_100.window(0, 0.0)
    x[:] = -999.0
    assert not np.array_equal(cache_100.window(0, 0.0), x)


def test_window_rejects_off_grid_and_past_end(cache_100):
    with pytest.raises(ValueError, match="griglia"):
        cache_100.window(0, 0.123)
    with pytest.raises(ValueError, match="fuori dal record"):
        cache_100.window(0, 8.0)


def test_inference_batch_layout(cache_100):
    batch = cache_100.inference_batch([0, 1, 2])
    assert batch.signals.shape == (12, D.N_LEADS, 250)
    assert batch.record_idx[:4].tolist() == [0, 0, 0, 0]
    assert np.allclose(batch.start_s[:4], [0.0, 2.5, 5.0, 7.5])


def test_random_batch_starts_on_grid(cache_100):
    rng = np.random.default_rng(6)
    batch = cache_100.random_batch(np.arange(N_RECORDS), rng)
    assert np.isin(batch.start_s, D.crop_start_grid()).all()
    assert batch.signals.shape == (N_RECORDS, D.N_LEADS, 250)


# -- allineamento ----------------------------------------------------------


def test_assert_aligned_across_rates(cache_100, cache_500):
    idx = np.arange(N_RECORDS)
    a = cache_100.inference_batch(idx)
    b = cache_500.inference_batch(idx)
    D.assert_aligned(a, b, check_signal=True)


def test_assert_aligned_detects_offset_mismatch(cache_100, cache_500):
    a = cache_100.crop([0, 1], [0.0, 2.5])
    b = cache_500.crop([0, 1], [0.0, 3.0])
    with pytest.raises(AssertionError, match="istanti di partenza"):
        D.assert_aligned(a, b)


def test_assert_aligned_detects_record_mismatch(cache_100, cache_500):
    a = cache_100.crop([0, 1], [0.0, 0.0])
    b = cache_500.crop([1, 0], [0.0, 0.0])
    with pytest.raises(AssertionError, match="stessi record"):
        D.assert_aligned(a, b)


def test_assert_aligned_detects_duration_mismatch(cache_100, cache_500):
    a = cache_100.crop([0], [0.0], window_seconds=2.5)
    b = cache_500.crop([0], [0.0], window_seconds=5.0)
    with pytest.raises(AssertionError, match="durata diversa"):
        D.assert_aligned(a, b)


def test_assert_aligned_rejects_same_rate(cache_100):
    a = cache_100.crop([0], [0.0])
    with pytest.raises(ValueError, match="stessa frequenza"):
        D.assert_aligned(a, a)
