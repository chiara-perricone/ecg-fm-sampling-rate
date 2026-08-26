"""Aggregazione fra run: la condizione bloccante di §3.

``evaluate.py`` scrive le probabilita' per record di **un** run. Qui si leggono
piu' file e si combinano. La separazione non e' estetica: i 43 run girano in
momenti diversi su una macchina noleggiata, e questo modulo deve poter girare
dopo, su CPU, senza rifare inferenza.

La forma dell'aggregazione di §3 (voce 15):

* la stima puntuale e' la **media dei tre seed** di ``macro_all`` su fold 10;
* i record vengono ricampionati 10.000 volte e **lo stesso ricampionamento va a
  tutti i seed**, perche' i seed condividono fold 10 e quello e' l'unico
  accoppiamento vero disponibile;
* il CI e' sui percentili della media ricampionata;
* si accetta se 0,941 cade nel CI **e** la media dista da 0,941 meno di 0,010.

L'intervallo e' **sui record soltanto**. Dentro un ricampionamento i modelli
sono fissi, quindi lo scarto fra seed non viene ricampionato: va riportato
accanto, non dentro. Con tre seed non c'e' modo onesto di farcelo stare.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .metrics import macro_auroc_fast
from .stats import bootstrap_indices

__all__ = [
    "STAGE0_TARGET",
    "STAGE0_TOLERANCE",
    "RunPredictions",
    "load_predictions",
    "seed_mean_bootstrap",
    "stage0_verdict",
]

#: arXiv:2509.25095v2 §4.1, la macro AUROC supervisionata di riferimento.
STAGE0_TARGET = 0.941

#: §3: scarto massimo ammesso fra la stima puntuale e il target.
STAGE0_TOLERANCE = 0.010


@dataclass(frozen=True)
class RunPredictions:
    """Il contenuto di un ``predictions-*.npz``."""

    run_id: str
    ecg_id: np.ndarray
    y_true: np.ndarray
    y_prob: np.ndarray
    clean_columns: np.ndarray


def load_predictions(paths: Sequence[str | Path]) -> tuple[list[RunPredictions], np.ndarray]:
    """Carica piu' file e verifica che descrivano lo **stesso** test set.

    Restituisce anche le colonne del set congelato, dopo aver verificato che
    coincidano fra i run. Aggregare predizioni allineate a popolazioni diverse
    produrrebbe numeri plausibili e privi di significato, quindi il controllo
    viene prima e non e' disattivabile.
    """
    loaded: list[RunPredictions] = []
    for path in paths:
        path = Path(path)
        with np.load(path) as npz:
            loaded.append(
                RunPredictions(
                    run_id=path.parent.name,
                    ecg_id=npz["ecg_id"],
                    y_true=npz["y_true"],
                    y_prob=npz["y_prob"],
                    clean_columns=npz["clean_columns"],
                )
            )
    if not loaded:
        raise ValueError("nessun file di predizioni")

    first = loaded[0]
    for other in loaded[1:]:
        if not np.array_equal(first.ecg_id, other.ecg_id):
            raise ValueError(
                f"{other.run_id} copre record diversi da {first.run_id}: "
                "non sono aggregabili"
            )
        if not np.array_equal(first.y_true, other.y_true):
            raise ValueError(
                f"{other.run_id} ha etichette diverse da {first.run_id} sugli "
                "stessi record"
            )
        if not np.array_equal(first.clean_columns, other.clean_columns):
            raise ValueError(
                f"{other.run_id} usa un insieme congelato diverso da {first.run_id}"
            )
    return loaded, first.clean_columns


@dataclass(frozen=True)
class SeedMean:
    """Media fra seed di una macro AUROC, con CI sui record."""

    point: float
    lo: float
    hi: float
    per_run: dict[str, float]
    n_boot: int

    @property
    def seed_spread(self) -> float:
        """Escursione fra i seed. **Fuori** dal CI, di proposito (voce 15)."""
        values = list(self.per_run.values())
        return float(max(values) - min(values)) if len(values) > 1 else 0.0

    @property
    def seed_std(self) -> float:
        values = list(self.per_run.values())
        return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def seed_mean_bootstrap(
    runs: Sequence[RunPredictions],
    columns: Sequence[int] | None = None,
    *,
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> SeedMean:
    """Media fra run di ``macro_auroc``, con CI percentile sui record.

    ``columns`` restringe alle etichette del set congelato; ``None`` usa tutte.

    Un solo ``bootstrap_indices``, riusato per ogni run: e' il punto in cui
    l'accoppiamento fra seed diventa operativo.
    """
    if not runs:
        raise ValueError("nessun run da aggregare")
    y_true = runs[0].y_true
    if columns is not None:
        columns = np.asarray(columns, dtype=int)
        y_true = y_true[:, columns]

    probs = [r.y_prob if columns is None else r.y_prob[:, columns] for r in runs]
    per_run = {r.run_id: macro_auroc_fast(y_true, p) for r, p in zip(runs, probs)}
    point = float(np.mean(list(per_run.values())))

    rng = np.random.default_rng(seed)
    idx = bootstrap_indices(len(y_true), n_boot, rng)
    means = np.empty(n_boot)
    for b in range(n_boot):
        take = idx[b]
        yt = y_true[take]
        means[b] = float(np.mean([macro_auroc_fast(yt, p[take]) for p in probs]))

    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return SeedMean(point=point, lo=float(lo), hi=float(hi), per_run=per_run, n_boot=n_boot)


def stage0_verdict(
    estimate: SeedMean,
    target: float = STAGE0_TARGET,
    tolerance: float = STAGE0_TOLERANCE,
) -> dict:
    """Le due condizioni di §3, valutate e riportate separatamente.

    Separate di proposito: se la riproduzione fallisse, quale delle due sia
    caduta dice cose diverse. Un target fuori dal CI con stima vicina indica un
    intervallo stretto, cioe' molta potenza; una stima lontana con CI ampio
    indica il contrario.
    """
    contains = bool(estimate.lo <= target <= estimate.hi)
    within = bool(abs(estimate.point - target) <= tolerance)
    return {
        "target": target,
        "tolerance": tolerance,
        "point": estimate.point,
        "ci": [estimate.lo, estimate.hi],
        "difference": estimate.point - target,
        "ci_contains_target": contains,
        "point_within_tolerance": within,
        "accepted": contains and within,
        "per_run": estimate.per_run,
        "seed_spread": estimate.seed_spread,
        "seed_std": estimate.seed_std,
        "n_boot": estimate.n_boot,
    }
