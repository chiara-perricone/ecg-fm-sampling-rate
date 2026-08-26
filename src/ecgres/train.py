"""Il ciclo di training di un singolo run, piu' la valutazione per epoca.

Separazione voluta rispetto a ``scripts/train.py``: qui non si legge la riga di
comando e non si toccano i dati veri. ``run_training`` riceve dataset gia'
costruiti, cosi' la meccanica — epoche, aggregazione dei crop, selezione del
checkpoint, resume — e' verificabile su CPU in un paio di secondi, senza PTB-XL
e senza GPU. Il collegamento ai dati veri — metadati, cache, scaler, insieme
congelato delle etichette — vive in ``scripts/train.py``, che e' l'unico punto
in cui questo modulo tocca ``ecgres.data``.

Tre proprieta' che il resto del disegno da' per scontate e che qui vanno
garantite:

* **Il resume e' esatto.** §6.7 usa istanze interrompibili, quindi un run ripreso
  deve valere un run mai interrotto. Non si ottiene salvando lo stato dell'RNG:
  si ottiene rendendo ogni epoca funzione di ``(seed, epoch)`` e nient'altro.
  I crop lo erano gia' per costruzione (``CropSampler``); qui si aggiungono
  l'ordine di shuffle e il flusso del dropout.
* **Le due macro AUROC si calcolano sempre entrambe** su fold 9, a ogni epoca
  (§10 voce 13), e si registra l'epoca che ciascun criterio avrebbe scelto.
* **Fold 9 si valuta come fold 10**: quattro crop non sovrapposti, probabilita'
  mediate per record. Criterio di selezione e metrica riportata sono la stessa
  quantita' su fold diversi.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import macro_auroc
from .model import INFERENCE_CROP_STARTS_S, ModelConfig, WindowDataset, build_model
from .runs import RunSpec

__all__ = [
    "TrainConfig",
    "SELECTION_METRICS",
    "epoch_seed",
    "aggregate_crops",
    "evaluate",
    "epoch_metrics",
    "best_epochs",
    "training_fingerprint",
    "write_history",
    "read_history",
    "run_training",
]

#: Le due metriche di fold 9 calcolate a ogni epoca. La prima e' l'endpoint
#: della condizione bloccante, la seconda quello del confronto (§7).
SELECTION_METRICS: tuple[str, ...] = ("macro_all", "macro_clean")

N_INFERENCE_CROPS = len(INFERENCE_CROP_STARTS_S)


@dataclass(frozen=True)
class TrainConfig:
    """§6.5, che segue arXiv:2509.25095v2 §3.3 e non ``ssm_ecg``.

    In particolare ``batch_size=64``, non 128: il numero viene dal Reality
    Check, che e' il lavoro di cui si riproduce il risultato.
    """

    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-3
    num_workers: int = 0


def epoch_seed(seed: int, epoch: int) -> int:
    """Seed derivato, stabile fra processi e fra piattaforme.

    Non ``hash()``: in Python e' randomizzato per processo, quindi un run
    ripreso in un altro processo vedrebbe un altro flusso. ``SeedSequence`` di
    NumPy e' invece deterministico e documentato.

    Riseedare a ogni epoca invece che una volta all'inizio e' cio' che rende il
    resume esatto: l'epoca 37 estrae le stesse maschere di dropout e lo stesso
    ordine di shuffle, che il run sia partito dall'epoca 0 o dal checkpoint 36.
    """
    return int(np.random.default_rng((seed, epoch)).integers(1, 2**31 - 1))


def aggregate_crops(
    probabilities: np.ndarray, record_ids: Sequence[int], n_crops: int = N_INFERENCE_CROPS
) -> tuple[np.ndarray, np.ndarray]:
    """Media le probabilita' dei crop di uno stesso record (§6.2).

    Si mediano le **probabilita'**, non i logit. La sigmoide e' monotona ma la
    media non commuta con lei, quindi le due scelte producono ordinamenti dei
    record diversi e AUROC diverse; la pipeline di riferimento media le
    probabilita'.

    Presuppone l'ordinamento record-esterno / crop-interno prodotto da
    ``WindowDataset`` in valutazione, e lo verifica invece di fidarsi.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    ids = np.asarray(record_ids)
    if probabilities.shape[0] != ids.size:
        raise ValueError("probabilita' e record_ids hanno lunghezze diverse")
    if ids.size % n_crops:
        raise ValueError(f"{ids.size} elementi non sono un multiplo di {n_crops} crop")

    grouped = ids.reshape(-1, n_crops)
    if not (grouped == grouped[:, :1]).all():
        raise ValueError(
            "l'ordine non e' record-esterno / crop-interno: aggregare qui "
            "mescolerebbe record diversi"
        )
    per_record = probabilities.reshape(-1, n_crops, probabilities.shape[1]).mean(axis=1)
    return per_record, grouped[:, 0]


@torch.no_grad()
def evaluate(
    model: nn.Module, dataset: WindowDataset, cfg: TrainConfig, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """``(y_true, y_prob)`` per **record**, non per crop.

    Il loader non mescola: l'ordinamento e' quello che ``aggregate_crops``
    pretende, e romperlo qui darebbe medie fra record diversi senza errori.
    """
    if dataset.train:
        raise ValueError("evaluate richiede un dataset in modalita' inferenza")
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    probs, labels, ids = [], [], []
    for x, y, record_id in loader:
        logits = model(x.to(device))
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
        labels.append(y.numpy())
        ids.append(np.asarray(record_id))
    prob_per_record, order = aggregate_crops(
        np.concatenate(probs), np.concatenate(ids)
    )
    # Le etichette sono costanti sui crop di un record: basta il primo di ogni
    # gruppo, e prenderlo cosi' verifica implicitamente l'allineamento.
    y_true = np.concatenate(labels).reshape(-1, N_INFERENCE_CROPS, prob_per_record.shape[1])
    if not (y_true == y_true[:, :1]).all():
        raise ValueError("le etichette variano fra i crop di uno stesso record")
    return y_true[:, 0], prob_per_record


def epoch_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, clean_columns: Sequence[int]
) -> dict[str, float]:
    """Le due macro AUROC, sempre entrambe (§10 voce 13).

    ``clean_columns`` sono le posizioni delle 49 etichette di
    ``results/frozen_label_set.json`` dentro la matrice a 71 colonne. Il
    sottoinsieme e' congelato prima di ogni run e non si ricalcola qui.
    """
    columns = np.asarray(clean_columns, dtype=int)
    if columns.size == 0:
        raise ValueError("insieme di etichette clean vuoto")
    if columns.max(initial=-1) >= y_true.shape[1]:
        raise ValueError("clean_columns esce dalla matrice delle etichette")
    return {
        "macro_all": macro_auroc(y_true, y_prob),
        "macro_clean": macro_auroc(y_true[:, columns], y_prob[:, columns]),
    }


def best_epochs(history: Sequence[dict]) -> dict[str, int]:
    """Epoca che massimizza ciascun criterio. Il primo massimo vince i pari.

    Si registrano entrambe anche quando il run ne usa una sola: quanto spesso i
    due criteri divergano e' una quantita' da misurare, non da assumere (§6.5).
    """
    best: dict[str, int] = {}
    for metric in SELECTION_METRICS:
        values = [row[metric] for row in history]
        finite = [v if np.isfinite(v) else -np.inf for v in values]
        best[metric] = int(np.argmax(finite))
    return best


HISTORY_FIELDS = ("epoch", "train_loss", *SELECTION_METRICS)


def write_history(path: str | Path, history: Sequence[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in history:
            writer.writerow({k: row[k] for k in HISTORY_FIELDS})
    return path


def read_history(path: str | Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [
            {
                "epoch": int(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                **{m: float(row[m]) for m in SELECTION_METRICS},
            }
            for row in csv.DictReader(handle)
        ]


def _train_one_epoch(
    model: nn.Module,
    dataset: WindowDataset,
    optimiser: torch.optim.Optimizer,
    loss_fn: nn.Module,
    cfg: TrainConfig,
    device: torch.device,
    epoch: int,
    seed: int,
) -> float:
    """Una passata. Tutto cio' che e' casuale qui dipende da ``(seed, epoch)``."""
    torch.manual_seed(epoch_seed(seed, epoch))  # dropout
    dataset.set_epoch(epoch)  # crop
    generator = torch.Generator().manual_seed(epoch_seed(seed, epoch))  # shuffle
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    model.train()
    total, n_batches = 0.0, 0
    for x, y, _ in loader:
        optimiser.zero_grad(set_to_none=True)
        loss = loss_fn(model(x.to(device)), y.to(device))
        loss.backward()
        optimiser.step()
        total += float(loss.detach())
        n_batches += 1
    return total / max(n_batches, 1)


def training_fingerprint(
    run: RunSpec,
    model_cfg: ModelConfig,
    cfg: TrainConfig,
    train_dataset: WindowDataset,
    val_dataset: WindowDataset,
    clean_columns: Sequence[int],
    numerics: dict | None = None,
) -> str:
    """Impronta di tutto cio' che, cambiando, rende un checkpoint inutilizzabile.

    Il solo ``run_id`` non basta. Una prova ridotta — poche epoche su un
    sottoinsieme di record — produce un ``last.pt`` con l'id giusto, e il run
    vero lo riprenderebbe continuando con i dati completi: un run del protocollo
    la cui prima epoca ha visto altri dati, senza un solo errore a dirlo.

    ``epochs`` e' escluso di proposito: allungare uno schedule interrotto e'
    esattamente l'uso previsto. ``num_workers`` pure, perche' non tocca i
    risultati.
    """
    payload = {
        "run": asdict(run),
        "model": asdict(model_cfg),
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "n_train_items": len(train_dataset),
        "n_val_items": len(val_dataset),
        "train_record_ids": _digest(train_dataset.record_ids),
        "val_record_ids": _digest(val_dataset.record_ids),
        "scaler": [train_dataset.scaler.mean, train_dataset.scaler.std],
        "clean_columns": _digest(clean_columns),
        # TF32, determinismo e backend del kernel di Cauchy cambiano i numeri
        # prodotti, non solo la velocita'. Riprendere un run misurando la
        # velocita' con un backend e proseguendo con un altro darebbe un run le
        # cui epoche non sono state calcolate allo stesso modo.
        "numerics": dict(sorted((numerics or {}).items())),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def _digest(values: Sequence[int]) -> str:
    return hashlib.sha256(
        ",".join(str(int(v)) for v in values).encode()
    ).hexdigest()[:16]


def _checkpoint(
    model, optimiser, epoch: int, history: list[dict], run_id: str, fingerprint: str
) -> dict:
    return {
        "run_id": run_id,
        "fingerprint": fingerprint,
        "epoch": epoch,
        "model": model.state_dict(),
        "optimiser": optimiser.state_dict(),
        "history": history,
    }


def run_training(
    run: RunSpec,
    model_cfg: ModelConfig,
    train_dataset: WindowDataset,
    val_dataset: WindowDataset,
    clean_columns: Sequence[int],
    out_dir: str | Path,
    cfg: TrainConfig | None = None,
    device: str | torch.device = "cpu",
    numerics: dict | None = None,
    manifest_extra: dict | None = None,
) -> list[dict]:
    """Allena un run e restituisce la storia per epoca.

    Riprende da ``last.pt`` se esiste. Salva a ogni epoca:

    * ``last.pt`` — per riprendere dopo un'interruzione;
    * ``best-<metrica>.pt`` per **entrambi** i criteri, non solo per quello del
      run. Costa 9 MB e permette di rispondere a "la scelta della voce 13
      avrebbe cambiato il risultato?" senza riallenare nulla;
    * ``history.csv`` — leggibile anche mentre il run e' in corso.
    """
    cfg = cfg or TrainConfig()
    device = torch.device(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(model_cfg, run.seed).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    loss_fn = nn.BCEWithLogitsLoss()

    fingerprint = training_fingerprint(
        run, model_cfg, cfg, train_dataset, val_dataset, clean_columns, numerics
    )

    history: list[dict] = []
    start_epoch = 0
    last_path = out_dir / "last.pt"
    if last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state["run_id"] != run.run_id:
            raise ValueError(
                f"{last_path} appartiene a {state['run_id']}, non a {run.run_id}"
            )
        if state.get("fingerprint") != fingerprint:
            raise ValueError(
                f"{last_path} viene da una configurazione diversa "
                f"({state.get('fingerprint')} contro {fingerprint}): record, "
                "batch, scaler o modello non coincidono. Riprenderlo darebbe un "
                "run le cui prime epoche hanno visto altro. Cancellare la "
                "cartella e ripartire."
            )
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        history = list(state["history"])
        start_epoch = state["epoch"] + 1

    _write_manifest(
        out_dir, run, model_cfg, cfg, device,
        {"fingerprint": fingerprint, "numerics": numerics or {}, **(manifest_extra or {})},
    )

    for epoch in range(start_epoch, cfg.epochs):
        loss = _train_one_epoch(
            model, train_dataset, optimiser, loss_fn, cfg, device, epoch, run.seed
        )
        y_true, y_prob = evaluate(model, val_dataset, cfg, device)
        row = {"epoch": epoch, "train_loss": loss, **epoch_metrics(y_true, y_prob, clean_columns)}
        history.append(row)

        state = _checkpoint(model, optimiser, epoch, history, run.run_id, fingerprint)
        torch.save(state, last_path)
        for metric, best_epoch in best_epochs(history).items():
            if best_epoch == epoch:
                torch.save(state, out_dir / f"best-{metric}.pt")
        write_history(out_dir / "history.csv", history)

    _write_selection(out_dir, run, history)
    return history


def _write_selection(out_dir: Path, run: RunSpec, history: Sequence[dict]) -> None:
    """L'esito della selezione, incluso il controfattuale (§6.5)."""
    if not history:
        return
    best = best_epochs(history)
    (out_dir / "selection.json").write_text(
        json.dumps(
            {
                "run_id": run.run_id,
                "selection": run.selection,
                "selected_epoch": best[run.selection],
                "best_epoch_per_metric": best,
                "criteria_agree": len(set(best.values())) == 1,
                "value_at_selected_epoch": {
                    m: history[best[run.selection]][m] for m in SELECTION_METRICS
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


def _write_manifest(
    out_dir: Path,
    run: RunSpec,
    model_cfg: ModelConfig,
    cfg: TrainConfig,
    device: torch.device,
    extra: dict | None,
) -> None:
    """Come e' nato questo run. Con 43 run, uno di ignota provenienza e' perso."""
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run": asdict(run),
                "model": asdict(model_cfg),
                "train": asdict(cfg),
                "device": str(device),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "python": platform.python_version(),
                "cuda": torch.version.cuda,
                "gpu": (
                    torch.cuda.get_device_name(0)
                    if device.type == "cuda" and torch.cuda.is_available()
                    else None
                ),
                "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                **(extra or {}),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
