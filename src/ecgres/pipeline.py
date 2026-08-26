"""Il collegamento fra la riga di ``configs/runs.csv`` e i dati veri.

Sta in ``src`` e non dentro uno script perche' lo usano in due — ``train.py`` e
``evaluate.py`` — e devono usarne **la stessa copia**: se il training e la
valutazione risolvessero lo scaler o le colonne delle etichette in modo anche
solo leggermente diverso, il numero riportato non sarebbe quello del modello
allenato, e niente lo segnalerebbe.

Importa ``ecgres.data``, quindi pandas e scipy. La meccanica del training, che
non ne ha bisogno, resta in ``ecgres.train``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from . import data as D
from .model import FIT_FS, GlobalScaler
from .runs import RunSpec

__all__ = [
    "BLOCKING_BLOCK",
    "clean_columns",
    "scaler_spec",
    "resolve_scaler",
    "configure_numerics",
    "git_sha",
    "section",
    "info",
]

#: Blocco della condizione bloccante (§3). Vedi ``scaler_spec``.
BLOCKING_BLOCK = 0


# --------------------------------------------------------------------------
# Stampa
# --------------------------------------------------------------------------

def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}", flush=True)


def info(label: str, detail: str = "") -> None:
    print(f"  [    ] {label}{(': ' + detail) if detail else ''}", flush=True)


# --------------------------------------------------------------------------
# Etichette
# --------------------------------------------------------------------------

def clean_columns(label_names: Sequence[str], frozen_path: str | Path) -> list[int]:
    """Posizioni delle 49 etichette congelate dentro la matrice a 71 colonne.

    Il set viene **letto, non ricalcolato**: e' stato congelato prima di ogni run
    (§7), e ricalcolarlo qui lo renderebbe funzione dei dati presenti sul disco
    di turno invece che una scelta pre-registrata. Si verifica pero' che il file
    sia coerente con quello che dichiara di essere, perche' un set giusto per
    caso sarebbe peggio di uno dichiaratamente sbagliato.
    """
    frozen = json.loads(Path(frozen_path).read_text(encoding="utf-8"))
    names = list(frozen["labels_clean"])
    if len(names) != frozen["n_labels_clean"]:
        raise ValueError(
            f"il set congelato dichiara {frozen['n_labels_clean']} etichette "
            f"ma ne elenca {len(names)}"
        )
    if len(label_names) != frozen["n_labels_all"]:
        raise ValueError(
            f"la matrice ha {len(label_names)} colonne, il set congelato ne "
            f"presuppone {frozen['n_labels_all']}"
        )
    position = {name: i for i, name in enumerate(label_names)}
    missing = [n for n in names if n not in position]
    if missing:
        raise ValueError(f"etichette congelate assenti dalla matrice: {missing}")
    return [position[n] for n in names]


# --------------------------------------------------------------------------
# Scaler
# --------------------------------------------------------------------------

def scaler_spec(run: RunSpec) -> tuple[int, bool]:
    """``(fs, notch)`` della cache su cui va fittato lo scaler del run.

    Due scaler, non uno, ed e' il protocollo a volerlo. La voce 8 dichiara
    esplicitamente che *"Stage 0 uses the reference pipeline's own scaler"*:

    * **blocco 0** — si fitta sulla cache del run stesso, 100 Hz senza notch,
      come la pipeline di riferimento. Deve riprodurre 0,941, e normalizzare con
      costanti che il riferimento non ha usato aggiungerebbe un candidato in piu'
      a spiegare un fallimento stretto;
    * **blocchi 1-3** — un unico scaler a 500 Hz **con notch**, sui fold 1-8,
      applicato invariato a tutti i bracci (voce 8). Anche il blocco 3, che gira
      a 100 Hz senza notch: il suo scopo e' isolare l'effetto del notch contro il
      blocco 1, e cambiare insieme anche la normalizzazione lo confonderebbe.
    """
    if run.block == BLOCKING_BLOCK:
        return run.fs, run.notch
    return FIT_FS, True


def resolve_scaler(
    run: RunSpec,
    root: str | Path,
    cache_dir: str | Path,
    meta,
    train_idx: np.ndarray,
    log: Callable[[str, str], None] = info,
) -> tuple[GlobalScaler, Path]:
    """Lo scaler del run: riletto se esiste, fittato una volta sola altrimenti.

    Un file gia' presente viene accettato solo dopo aver verificato la
    provenienza registrata dentro di esso. Uno scaler e' due numeri: senza
    provenienza, due numeri sbagliati sono indistinguibili da due giusti.
    """
    fit_fs, fit_notch = scaler_spec(run)
    cache_dir = Path(cache_dir)
    path = cache_dir / f"scaler_fs{fit_fs}_notch{int(fit_notch)}.json"
    expected = {"fs": fit_fs, "notch": fit_notch, "source": "hr", "window": "full_record"}

    if path.exists():
        scaler = GlobalScaler.from_json(path)
        wrong = {
            k: (scaler.provenance.get(k), v)
            for k, v in expected.items()
            if scaler.provenance.get(k) != v
        }
        if wrong:
            raise ValueError(f"{path.name} ha provenienza inattesa: {wrong}")
        return scaler, path

    cache = D.SignalCache(root, cache_dir, meta, fs=fit_fs, notch=fit_notch)
    log("fit dello scaler", f"{fit_fs} Hz, notch={fit_notch}, {train_idx.size} record")
    provenance = {
        "fs": fit_fs,
        "notch": fit_notch,
        "source": "hr",
        "folds": list(D.SPLIT_FOLDS["train"]),
        "cache_stem": cache.stem,
        "fitted_for": "stage0" if run.block == BLOCKING_BLOCK else "comparison",
    }
    scaler = GlobalScaler.fit(cache.array, train_idx, provenance=provenance)
    scaler.to_json(path)
    return scaler, path


# --------------------------------------------------------------------------
# Impostazioni numeriche e provenienza
# --------------------------------------------------------------------------

def configure_numerics(tf32: bool, deterministic: bool) -> None:
    """TF32 e determinismo, entrambi espliciti e registrati nel manifest.

    TF32 ha 10 bit di mantissa: lasciarlo acceso significa che il "fp32" di §6.7
    non e' vero. Va spento per default e acceso solo di proposito, per esempio
    per misurare quanto costa spegnerlo.

    Con ``deterministic`` torch **solleva** invece di rallentare in silenzio, se
    un'operazione non ha implementazione deterministica: e' un fallimento
    leggibile nei primi secondi invece di un costo nascosto per tutto il run.
    """
    import torch

    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=False)


def git_sha(repo_root: str | Path) -> str | None:
    """Commit corrente, o ``None`` fuori da un repo. Non solleva mai."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    return out.stdout.strip() or None
