"""Invarianti di ``ecgres.model``.

Questi test non verificano che il modello impari: verificano le proprieta' su
cui poggia l'intero disegno sperimentale. Se uno di questi cade, il confronto
fra rate misura qualcosa di diverso da quello che dichiara.

Girano su modelli piccoli (``d_model=16``, ``n_layers=1``) e senza dati veri.
"""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ecgres.model import (  # noqa: E402
    DT_MAX_DEFAULT,
    DT_MIN_DEFAULT,
    FIT_FS,
    GRID_SECONDS,
    INFERENCE_CROP_STARTS_S,
    RATES,
    CropSampler,
    GlobalScaler,
    ModelConfig,
    S4Backbone,
    build_model,
    count_parameters,
)

SEED = 0


def tiny(fs: int, arm: str = "A") -> ModelConfig:
    return ModelConfig(
        fs=fs, arm=arm, n_classes=5, d_model=16, d_state=4, n_layers=1, dropout=0.0
    )


# --------------------------------------------------------------------------- #
# Lunghezze
# --------------------------------------------------------------------------- #


def test_window_lengths():
    assert [tiny(fs).n_samples for fs in RATES] == [250, 600, 625, 1250]


def test_inference_crops_tile_the_record():
    starts = INFERENCE_CROP_STARTS_S
    assert len(starts) == 4
    assert starts[0] == 0.0
    diffs = np.diff(starts)
    assert np.allclose(diffs, 2.5)
    assert starts[-1] + 2.5 == 10.0


def test_config_rejects_non_integer_window():
    # 2,51 s a 240 Hz da' 602,4 campioni. Attenzione a scegliere l'esempio:
    # 0,05 s a 240 Hz da' 12 campioni esatti e non solleverebbe niente.
    with pytest.raises(ValueError):
        ModelConfig(fs=240, arm="A", n_classes=5, window_seconds=2.51)


# --------------------------------------------------------------------------- #
# Invarianza rispetto al rate
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("arm", ["A", "B"])
def test_param_count_identical_across_rates(arm):
    """Il conteggio parametri non deve muoversi al variare di fs.

    E' l'ipotesi che giustifica la scelta di S4. Se un giorno entra uno stem
    convoluzionale o un pooling che appiattisce, questo test cade.
    """
    counts = {fs: count_parameters(build_model(tiny(fs, arm), SEED)) for fs in RATES}
    assert len(set(counts.values())) == 1, counts


def _kernel_C(model) -> list[torch.Tensor]:
    return [p for n, p in model.named_parameters() if n.endswith("kernel.C")]


def _non_kernel_params(model) -> dict[str, torch.Tensor]:
    return {n: p for n, p in model.named_parameters() if not n.endswith("kernel.C")}


@pytest.mark.parametrize("arm", ["A", "B"])
def test_non_kernel_params_identical_across_rates(arm):
    """Encoder, norm e testa non dipendono da fs: init identica bit per bit.

    Verifica anche, indirettamente, che il consumo di RNG durante la
    costruzione non dipenda dalla lunghezza: se dipendesse, la testa (creata
    per ultima) divergerebbe.
    """
    ref = _non_kernel_params(build_model(tiny(RATES[0], arm), SEED))
    for fs in RATES[1:]:
        other = _non_kernel_params(build_model(tiny(fs, arm), SEED))
        assert ref.keys() == other.keys()
        for name in ref:
            assert torch.equal(ref[name], other[name]), f"{name} differisce a {fs} Hz"


def test_dt_times_length_is_constant_in_arm_b():
    """La chiave di tutto: in Arm B ``dt * L`` non dipende dal rate.

    ``length_correction=True`` moltiplica C per ``I - dA^L`` all'init, e
    ``dA^L ~ exp(A * dt * L)``. Poiche' in Arm B ``dt`` scala di ``100/fs`` e
    ``L`` scala di ``fs``, il prodotto e' costante e la correzione e' la stessa a
    ogni rate. In Arm A no.
    """
    prod_b = {tiny(fs, "B").dt_max * tiny(fs, "B").n_samples for fs in RATES}
    assert len(prod_b) == 1 or max(prod_b) - min(prod_b) < 1e-9, prod_b

    prod_a = [tiny(fs, "A").dt_max * tiny(fs, "A").n_samples for fs in RATES]
    assert max(prod_a) / min(prod_a) > 4  # scala col rate: 250 -> 1250


def test_arm_b_preserves_kernel_init_across_rates():
    """Conseguenza del test precedente: in Arm B C e' (quasi) invariante.

    Non bit per bit: ``power(L, dA)`` accumula errore float32 su una potenza di
    matrice di ordine fino a 1250.
    """
    ref = _kernel_C(build_model(tiny(RATES[0], "B"), SEED))
    assert ref, "nessun parametro kernel.C trovato: la vendorizzazione e' cambiata"
    for fs in RATES[1:]:
        other = _kernel_C(build_model(tiny(fs, "B"), SEED))
        for a, b in zip(ref, other):
            assert torch.allclose(a, b, atol=1e-3, rtol=1e-3), f"a {fs} Hz"


def test_arm_a_kernel_init_drifts_with_rate():
    """In Arm A no, e va messo a verbale.

    Non e' un difetto da correggere: ``dt * L`` e ``dt * fs`` differiscono solo
    per la costante ``window_seconds``, quindi sono la stessa manopola e A-B
    resta un confronto a un solo grado di liberta'. Questo test esiste perche'
    il meccanismo non cambi in silenzio con un'altra versione del file vendorato.
    """
    ref = _kernel_C(build_model(tiny(100, "A"), SEED))
    drift_a = max(
        (a - b).abs().max().item()
        for fs in RATES[1:]
        for a, b in zip(ref, _kernel_C(build_model(tiny(fs, "A"), SEED)))
    )
    ref_b = _kernel_C(build_model(tiny(100, "B"), SEED))
    drift_b = max(
        (a - b).abs().max().item()
        for fs in RATES[1:]
        for a, b in zip(ref_b, _kernel_C(build_model(tiny(fs, "B"), SEED)))
    )
    assert drift_a > 1e-2, drift_a
    assert drift_a > 50 * drift_b, (drift_a, drift_b)


def test_encoder_is_pointwise():
    enc = build_model(tiny(500), SEED).encoder
    assert enc.kernel_size == (1,)
    assert enc.stride == (1,)
    assert enc.dilation == (1,)


def test_pooling_is_mean_not_sum():
    """Input costante, due lunghezze: l'uscita non deve scalare con L."""
    model = build_model(tiny(100), SEED).eval()
    x_short = torch.ones(1, 12, 250)
    x_long = torch.ones(1, 12, 500)
    with torch.no_grad():
        y_short = model(x_short)
        y_long = model(x_long)
    # Non chiediamo uguaglianza esatta (il kernel S4 ha memoria finita ai bordi),
    # ma il rapporto deve stare lontano dal 2x che darebbe una somma.
    ratio = (y_long.abs().mean() / y_short.abs().mean()).item()
    assert 0.5 < ratio < 1.5, ratio


# --------------------------------------------------------------------------- #
# Arm A vs Arm B
# --------------------------------------------------------------------------- #


def test_arms_coincide_at_100hz():
    """A 100 Hz la compensazione e' l'identita': dt_scale == 1."""
    a, b = tiny(100, "A"), tiny(100, "B")
    assert b.dt_scale == 1.0
    assert (a.dt_min, a.dt_max) == (b.dt_min, b.dt_max)


@pytest.mark.parametrize("fs", RATES)
def test_arm_b_preserves_horizons_in_seconds(fs):
    """L'orizzonte 1/(dt * fs) di Arm B e' quello di Arm A a 100 Hz."""
    cfg = tiny(fs, "B")
    assert math.isclose(1.0 / (cfg.dt_max * fs), 1.0 / (DT_MAX_DEFAULT * 100))
    assert math.isclose(1.0 / (cfg.dt_min * fs), 1.0 / (DT_MIN_DEFAULT * 100))


@pytest.mark.parametrize("fs", RATES)
def test_arm_b_shifts_log_dt_by_a_constant(fs):
    """``log_dt`` di B = ``log_dt`` di A + log(100/fs), draw per draw.

    Segue dal fatto che ``log_dt = rand * (log dt_max - log dt_min) + log dt_min``
    e che scalare entrambi gli estremi aggiunge una costante. Se questo test
    fallisce, il RNG non e' allineato fra i due bracci e A-B non isola piu' il
    solo prior sulle scale temporali.
    """
    a = build_model(tiny(fs, "A"), SEED)
    b = build_model(tiny(fs, "B"), SEED)
    log_a = dict(a.named_buffers())
    log_b = dict(b.named_buffers())
    names = [n for n in log_a if n.endswith("log_dt")]
    assert names, "nessun buffer log_dt trovato: la vendorizzazione e' cambiata"
    shift = math.log(100 / fs)
    for n in names:
        assert torch.allclose(log_b[n], log_a[n] + shift, atol=1e-6)


def test_log_dt_is_a_buffer_not_a_parameter():
    """Coi default upstream ``log_dt`` non e' addestrabile.

    E' cio' che rende il prior sulle scale temporali un vincolo e non un punto
    di partenza. Se una futura versione del file vendorato lo rendesse un
    Parameter, l'interpretazione di A-B cambierebbe e va ridiscussa.
    """
    model = build_model(tiny(500), SEED)
    param_names = {n for n, _ in model.named_parameters()}
    buffer_names = {n for n, _ in model.named_buffers()}
    assert any(n.endswith("log_dt") for n in buffer_names)
    assert not any(n.endswith("log_dt") for n in param_names)


def test_forward_does_not_expose_rate():
    """Arm B vive in dt_min/dt_max, non nell'argomento ``rate`` di S4."""
    import inspect

    sig = inspect.signature(S4Backbone.forward)
    assert "rate" not in sig.parameters


# --------------------------------------------------------------------------- #
# Crop
# --------------------------------------------------------------------------- #


def test_crop_start_is_on_the_grid():
    s = CropSampler(seed=SEED)
    for rec in range(200):
        t = s.start_seconds(epoch=0, record_id=rec)
        assert 0.0 <= t <= 10.0 - 2.5
        k = t / GRID_SECONDS
        assert abs(k - round(k)) < 1e-9


def test_crop_start_is_shared_across_rates():
    """Lo stesso istante fisico da' un indice intero a tutti e quattro i rate."""
    s = CropSampler(seed=SEED)
    for rec in range(200):
        t = s.start_seconds(epoch=0, record_id=rec)
        idx = {fs: CropSampler.to_index(t, fs) for fs in RATES}
        for fs, i in idx.items():
            assert math.isclose(i / fs, t, abs_tol=1e-9)


def test_crop_start_is_deterministic_and_varies_with_epoch():
    s = CropSampler(seed=SEED)
    assert s.start_seconds(0, 42) == s.start_seconds(0, 42)
    starts = {s.start_seconds(e, 42) for e in range(50)}
    assert len(starts) > 1


def _fake_signals(n=20, leads=12, length=64, seed=7):
    rng = np.random.default_rng(seed)
    # Scale diverse per derivazione: serve a distinguere lo scaler globale da
    # uno per derivazione.
    scales = np.linspace(0.1, 3.0, leads)[None, :, None]
    return (rng.normal(size=(n, leads, length)) * scales).astype(np.float32)


class _FakeCache:
    """Minimo indispensabile per ``GlobalScaler.fit_from_cache``."""

    def __init__(self, array, fs=FIT_FS, notch=True, source="hr"):
        self.array = array
        self.fs = fs
        self.notch = notch
        self.source = source
        self.stem = f"ptbxl_{source}_fs{fs}_notch{int(notch)}"


# --------------------------------------------------------------------------- #
# GlobalScaler (§10 voci 7 e 8)
# --------------------------------------------------------------------------- #


def test_scaler_is_global_not_per_lead():
    """Voce 7: una sola coppia (mean, std) su tutte le derivazioni insieme.

    Replica ``ss.transform(x.flatten()[:, np.newaxis])`` del riferimento. Con
    uno scaler per derivazione la varianza relativa fra derivazioni sparirebbe,
    che e' una scelta legittima ma **non** quella del run da riprodurre.
    """
    x = _fake_signals()
    scaler = GlobalScaler.fit(x)
    flat = np.asarray(x, dtype=np.float64)
    assert scaler.mean == pytest.approx(flat.mean(), abs=1e-12)
    assert scaler.std == pytest.approx(flat.std(), rel=1e-12)

    per_lead = flat.std(axis=(0, 2))
    assert per_lead.max() / per_lead.min() > 5  # le derivazioni sono diverse
    assert not np.allclose(per_lead, scaler.std, rtol=0.1)


def test_scaler_standardizes_what_it_was_fitted_on():
    x = _fake_signals()
    z = GlobalScaler.fit(x).transform(np.asarray(x, dtype=np.float64))
    assert z.mean() == pytest.approx(0.0, abs=1e-12)
    assert z.std() == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("chunk", [1, 3, 7, 10_000])
def test_scaler_fit_is_independent_of_chunking(chunk):
    """Il blocco esiste per non materializzare 4,2 GB, non per cambiare il fit."""
    x = _fake_signals()
    ref = GlobalScaler.fit(x, chunk=10**9)
    got = GlobalScaler.fit(x, chunk=chunk)
    assert got.mean == pytest.approx(ref.mean, abs=1e-12)
    assert got.std == pytest.approx(ref.std, rel=1e-12)


def test_scaler_fit_uses_only_the_selected_records():
    """Fold 1-8, non tutto: il test set non deve entrare nella costante."""
    x = _fake_signals()
    idx = [0, 1, 2, 5, 8, 13]
    got = GlobalScaler.fit(x, idx)
    ref = GlobalScaler.fit(x[idx])
    assert got.mean == pytest.approx(ref.mean, abs=1e-12)
    assert got.std == pytest.approx(ref.std, rel=1e-12)
    assert got.provenance["n_records"] == len(idx)
    assert got.provenance["n_values"] == x[idx].size
    # Un sottoinsieme diverso deve dare un risultato diverso, o il test sopra
    # passerebbe anche con un fit su tutto.
    assert GlobalScaler.fit(x, [3, 4]).std != pytest.approx(ref.std, rel=1e-6)


def test_scaler_survives_a_large_offset():
    """La seconda passata serve a questo.

    Con ``E[x^2] - E[x]^2`` su un segnale traslato di 1e6 la varianza
    verrebbe da una differenza fra numeri quasi uguali e perderebbe cifre.
    """
    x = _fake_signals().astype(np.float64) + 1e6
    scaler = GlobalScaler.fit(x)
    assert scaler.std == pytest.approx(x.std(), rel=1e-9)


def test_scaler_rejects_degenerate_fit():
    """Segnale costante: std nulla, lo scaler non deve nascere."""
    with pytest.raises(ValueError):
        GlobalScaler.fit(np.zeros((4, 12, 32), dtype=np.float32))


def test_scaler_fingerprint_depends_on_the_record_order():
    x = _fake_signals()
    a = GlobalScaler.fit(x, [0, 1, 2])
    b = GlobalScaler.fit(x, [2, 1, 0])
    assert a.mean == pytest.approx(b.mean, abs=1e-12)
    assert a.provenance["record_fingerprint"] != b.provenance["record_fingerprint"]


def test_fit_from_cache_records_provenance():
    x = _fake_signals()
    scaler = GlobalScaler.fit_from_cache(_FakeCache(x), np.arange(10))
    p = scaler.provenance
    assert p["fs"] == FIT_FS
    assert p["folds"] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert p["notch"] is True
    assert p["source"] == "hr"
    assert p["window"] == "full_record"


@pytest.mark.parametrize("fs", [fs for fs in RATES if fs != FIT_FS])
def test_fit_from_cache_refuses_other_rates(fs):
    """Voce 8: rifittare per braccio assorbirebbe l'effetto nella costante."""
    with pytest.raises(ValueError):
        GlobalScaler.fit_from_cache(_FakeCache(_fake_signals(), fs=fs), [0, 1])


def test_scaler_json_roundtrip(tmp_path):
    scaler = GlobalScaler.fit_from_cache(_FakeCache(_fake_signals()), np.arange(10))
    path = tmp_path / "scaler.json"
    scaler.to_json(path)
    back = GlobalScaler.from_json(path)
    assert back.mean == scaler.mean
    assert back.std == scaler.std
    for key, value in scaler.provenance.items():
        assert back.provenance[key] == value


def test_scaler_from_json_rejects_another_format(tmp_path):
    path = tmp_path / "scaler.json"
    path.write_text('{"version": 99, "mean": 0.0, "std": 1.0}')
    with pytest.raises(ValueError):
        GlobalScaler.from_json(path)


def test_to_index_rejects_off_grid():
    """0,05 s e' fuori griglia anche se a 100 Hz darebbe un indice intero (5).

    E' il caso che distingue "sulla griglia a 0,1 s" da "intero a questo rate":
    il primo e' l'invariante che serve, perche' garantisce tutti e quattro i
    rate insieme.
    """
    with pytest.raises(ValueError):
        CropSampler.to_index(0.05, 100)
    with pytest.raises(ValueError):
        CropSampler.to_index(0.33, 500)
