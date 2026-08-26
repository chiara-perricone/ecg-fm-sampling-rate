"""Meccanica del training loop, su CPU e senza PTB-XL.

Il test che conta e' ``test_resume_equals_uninterrupted``: §6.7 alleva su
istanze interrompibili, quindi un run ripreso deve valere un run mai
interrotto. Tutto il resto del disegno lo da' per scontato; qui lo si verifica.
"""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ecgres.model import ModelConfig, GlobalScaler, CropSampler, WindowDataset  # noqa: E402
from ecgres.runs import RunSpec  # noqa: E402
from ecgres.train import (  # noqa: E402
    SELECTION_METRICS,
    TrainConfig,
    aggregate_crops,
    best_epochs,
    epoch_metrics,
    epoch_seed,
    evaluate,
    read_history,
    run_training,
    write_history,
)

N_RECORDS = 24
N_CLASSES = 3
CLEAN = (0, 2)


# --------------------------------------------------------------------------- #
# Aggregazione dei crop
# --------------------------------------------------------------------------- #


def test_aggregate_crops_averages_within_records():
    probs = np.arange(8 * N_CLASSES, dtype=float).reshape(8, N_CLASSES)
    ids = np.repeat([10, 11], 4)
    per_record, order = aggregate_crops(probs, ids)
    assert per_record.shape == (2, N_CLASSES)
    assert order.tolist() == [10, 11]
    assert np.allclose(per_record[0], probs[:4].mean(axis=0))
    assert np.allclose(per_record[1], probs[4:].mean(axis=0))


def test_aggregate_crops_rejects_interleaved_order():
    """Ordine crop-esterno: mediarlo mescolerebbe record diversi in silenzio."""
    probs = np.zeros((8, N_CLASSES))
    ids = np.tile([10, 11], 4)
    with pytest.raises(ValueError, match="record-esterno"):
        aggregate_crops(probs, ids)


def test_aggregate_crops_rejects_ragged_input():
    with pytest.raises(ValueError, match="multiplo"):
        aggregate_crops(np.zeros((6, N_CLASSES)), np.repeat([10, 11], 3))


# --------------------------------------------------------------------------- #
# Metriche per epoca
# --------------------------------------------------------------------------- #


def _labels_and_scores(seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=(40, N_CLASSES))
    y[:, :] = np.where(y.sum(axis=0) == 0, 1, y)  # nessuna colonna degenere
    return y, rng.random((40, N_CLASSES))


def test_epoch_metrics_returns_both_criteria():
    y, p = _labels_and_scores()
    got = epoch_metrics(y, p, CLEAN)
    assert set(got) == set(SELECTION_METRICS)
    assert all(0.0 <= v <= 1.0 for v in got.values())


def test_macro_clean_uses_only_the_frozen_columns():
    """Se ignorasse ``clean_columns``, le due metriche coinciderebbero sempre."""
    y, p = _labels_and_scores()
    p[:, 1] = y[:, 1]  # colonna 1 resa perfetta, ed e' fuori da CLEAN
    got = epoch_metrics(y, p, CLEAN)
    assert got["macro_all"] > got["macro_clean"]


def test_epoch_metrics_rejects_columns_outside_the_matrix():
    y, p = _labels_and_scores()
    with pytest.raises(ValueError, match="esce dalla matrice"):
        epoch_metrics(y, p, (0, 99))


# --------------------------------------------------------------------------- #
# Selezione del checkpoint
# --------------------------------------------------------------------------- #


def _history(all_values, clean_values):
    return [
        {"epoch": i, "train_loss": 1.0 / (i + 1), "macro_all": a, "macro_clean": c}
        for i, (a, c) in enumerate(zip(all_values, clean_values))
    ]


def test_best_epochs_is_per_criterion():
    history = _history([0.80, 0.91, 0.85], [0.70, 0.72, 0.88])
    assert best_epochs(history) == {"macro_all": 1, "macro_clean": 2}


def test_best_epochs_breaks_ties_on_the_first_maximum():
    history = _history([0.9, 0.9, 0.5], [0.5, 0.5, 0.5])
    assert best_epochs(history)["macro_all"] == 0


def test_best_epochs_ignores_nan():
    """Una macro AUROC indefinita non deve vincere per via di un confronto falso."""
    history = _history([float("nan"), 0.6], [0.5, 0.5])
    assert best_epochs(history)["macro_all"] == 1


# --------------------------------------------------------------------------- #
# Storia e seed
# --------------------------------------------------------------------------- #


def test_history_roundtrip(tmp_path):
    history = _history([0.8, 0.9], [0.7, 0.75])
    path = write_history(tmp_path / "history.csv", history)
    assert read_history(path) == history
    assert "\r\n" not in path.read_bytes().decode()


def test_epoch_seed_is_deterministic_and_epoch_dependent():
    """Non ``hash()``: dev'essere stabile fra processi, o il resume cambia run."""
    assert epoch_seed(3, 7) == epoch_seed(3, 7)
    assert epoch_seed(3, 7) != epoch_seed(3, 8)
    assert epoch_seed(3, 7) != epoch_seed(4, 7)
    assert 0 < epoch_seed(0, 0) < 2**31


# --------------------------------------------------------------------------- #
# Ciclo completo
# --------------------------------------------------------------------------- #


class _FakeCache:
    def __init__(self, fs=100, n_records=N_RECORDS, seed=0):
        self.fs = fs
        rng = np.random.default_rng(seed)
        self._data = rng.normal(size=(n_records, 12, int(10.0 * fs))).astype(np.float32)

    def __len__(self):
        return self._data.shape[0]

    def window(self, record_idx, start_s, *, window_seconds=2.5):
        begin = int(round(start_s * self.fs))
        width = int(round(window_seconds * self.fs))
        return np.array(self._data[record_idx, :, begin : begin + width])


def _labels():
    rng = np.random.default_rng(11)
    y = rng.integers(0, 2, size=(N_RECORDS, N_CLASSES)).astype(np.uint8)
    y[0, :] = 0  # garantisce entrambe le classi in ogni colonna
    y[1, :] = 1
    return y


def _fixture(train: bool, cfg: ModelConfig, cache: _FakeCache):
    return WindowDataset(
        cache=cache,
        record_ids=list(range(N_RECORDS)),
        labels=_labels(),
        scaler=GlobalScaler(0.0, 1.0),
        cfg=cfg,
        train=train,
        sampler=CropSampler(seed=0) if train else None,
    )


def _setup():
    run = RunSpec(
        run_id="test-run",
        block=1,
        fs=100,
        arm="A",
        notch=True,
        source="hr",
        seed=0,
        selection="macro_clean",
    )
    cfg = ModelConfig(
        fs=100, arm="A", n_classes=N_CLASSES, d_model=8, d_state=2, n_layers=1, dropout=0.1
    )
    cache = _FakeCache()
    return run, cfg, _fixture(True, cfg, cache), _fixture(False, cfg, cache)


def test_evaluate_returns_one_row_per_record():
    from ecgres.model import build_model

    run, cfg, _, val = _setup()
    model = build_model(cfg, run.seed)
    y_true, y_prob = evaluate(model, val, TrainConfig(batch_size=8), torch.device("cpu"))
    assert y_true.shape == (N_RECORDS, N_CLASSES)
    assert y_prob.shape == (N_RECORDS, N_CLASSES)
    assert ((y_prob >= 0) & (y_prob <= 1)).all()
    assert np.array_equal(y_true, _labels())


def test_evaluate_refuses_a_training_dataset():
    from ecgres.model import build_model

    run, cfg, train_ds, _ = _setup()
    with pytest.raises(ValueError, match="inferenza"):
        evaluate(build_model(cfg, run.seed), train_ds, TrainConfig(), torch.device("cpu"))


def test_resume_equals_uninterrupted(tmp_path):
    """La proprieta' su cui poggia l'uso di istanze interrompibili (§6.7).

    Non e' ottenuta salvando lo stato dell'RNG, ma rendendo ogni epoca funzione
    di ``(seed, epoch)``: crop, ordine di shuffle e dropout. Se un giorno una di
    queste tornasse a dipendere dallo stato globale, questo test cade.
    """
    run, cfg, train_ds, val_ds = _setup()
    kwargs = dict(
        run=run, model_cfg=cfg, train_dataset=train_ds, val_dataset=val_ds,
        clean_columns=CLEAN, device="cpu",
    )

    whole = run_training(out_dir=tmp_path / "whole", cfg=TrainConfig(epochs=2, batch_size=8), **kwargs)

    part = tmp_path / "part"
    run_training(out_dir=part, cfg=TrainConfig(epochs=1, batch_size=8), **kwargs)
    resumed = run_training(out_dir=part, cfg=TrainConfig(epochs=2, batch_size=8), **kwargs)

    assert len(whole) == len(resumed) == 2
    for a, b in zip(whole, resumed):
        for key in ("train_loss", *SELECTION_METRICS):
            assert a[key] == pytest.approx(b[key], rel=1e-9, abs=1e-9), key

    a = torch.load(tmp_path / "whole" / "last.pt", map_location="cpu", weights_only=False)
    b = torch.load(part / "last.pt", map_location="cpu", weights_only=False)
    assert a["model"].keys() == b["model"].keys()
    for name in a["model"]:
        assert torch.equal(a["model"][name], b["model"][name]), name


def test_run_writes_history_manifest_and_selection(tmp_path):
    run, cfg, train_ds, val_ds = _setup()
    out = tmp_path / run.run_id
    history = run_training(
        run=run, model_cfg=cfg, train_dataset=train_ds, val_dataset=val_ds,
        clean_columns=CLEAN, out_dir=out, cfg=TrainConfig(epochs=2, batch_size=8),
        device="cpu",
    )

    assert read_history(out / "history.csv") == history
    assert json.loads((out / "manifest.json").read_text())["run"]["run_id"] == run.run_id

    selection = json.loads((out / "selection.json").read_text())
    assert selection["selection"] == "macro_clean"
    assert selection["selected_epoch"] == best_epochs(history)["macro_clean"]
    # Il controfattuale della voce 13 e' registrato anche quando non serve.
    assert set(selection["best_epoch_per_metric"]) == set(SELECTION_METRICS)

    # Un checkpoint per criterio, non solo per quello del run.
    for metric in SELECTION_METRICS:
        assert (out / f"best-{metric}.pt").exists()


def test_resume_refuses_a_reduced_run(tmp_path):
    """Il caso vero: una prova su pochi record lascia un ``last.pt`` valido.

    Ha l'id giusto, quindi il controllo sul ``run_id`` lo lascerebbe passare, e
    il run del protocollo ripartirebbe dall'epoca 1 con i dati completi. Le
    prime epoche avrebbero visto altro, e nulla lo direbbe.
    """
    run, cfg, _, val_ds = _setup()
    cache = _FakeCache()
    reduced = WindowDataset(
        cache=cache, record_ids=list(range(8)), labels=_labels(),
        scaler=GlobalScaler(0.0, 1.0), cfg=cfg, train=True, sampler=CropSampler(seed=0),
    )
    full = _fixture(True, cfg, cache)
    out = tmp_path / run.run_id

    common = dict(
        run=run, model_cfg=cfg, val_dataset=val_ds, clean_columns=CLEAN,
        out_dir=out, device="cpu",
    )
    run_training(train_dataset=reduced, cfg=TrainConfig(epochs=1, batch_size=8), **common)
    with pytest.raises(ValueError, match="configurazione diversa"):
        run_training(train_dataset=full, cfg=TrainConfig(epochs=2, batch_size=8), **common)


def test_fingerprint_ignores_the_epoch_count():
    """Allungare uno schedule interrotto e' l'uso previsto, non un cambio."""
    from ecgres.train import training_fingerprint

    run, cfg, train_ds, val_ds = _setup()
    a = training_fingerprint(run, cfg, TrainConfig(epochs=1), train_ds, val_ds, CLEAN)
    b = training_fingerprint(run, cfg, TrainConfig(epochs=100), train_ds, val_ds, CLEAN)
    c = training_fingerprint(run, cfg, TrainConfig(epochs=1, batch_size=8), train_ds, val_ds, CLEAN)
    assert a == b
    assert a != c  # il batch invece conta


def test_fingerprint_changes_with_the_scaler():
    """Due costanti di normalizzazione diverse sono due esperimenti diversi."""
    from ecgres.train import training_fingerprint

    run, cfg, train_ds, val_ds = _setup()
    other = _fixture(True, cfg, _FakeCache())
    other.scaler = GlobalScaler(1.0, 2.0)
    a = training_fingerprint(run, cfg, TrainConfig(), train_ds, val_ds, CLEAN)
    b = training_fingerprint(run, cfg, TrainConfig(), other, val_ds, CLEAN)
    assert a != b


def test_resume_refuses_a_checkpoint_from_another_run(tmp_path):
    run, cfg, train_ds, val_ds = _setup()
    out = tmp_path / "shared"
    run_training(
        run=run, model_cfg=cfg, train_dataset=train_ds, val_dataset=val_ds,
        clean_columns=CLEAN, out_dir=out, cfg=TrainConfig(epochs=1, batch_size=8),
        device="cpu",
    )
    other = RunSpec(**{**run.__dict__, "run_id": "another-run"})
    with pytest.raises(ValueError, match="appartiene a"):
        run_training(
            run=other, model_cfg=cfg, train_dataset=train_ds, val_dataset=val_ds,
            clean_columns=CLEAN, out_dir=out, cfg=TrainConfig(epochs=2, batch_size=8),
            device="cpu",
        )
