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

from itertools import combinations

from .metrics import macro_auroc_fast
from .stats import bootstrap_indices, holm, p_value_from_draws, percentile_ci

__all__ = [
    "STAGE0_TARGET",
    "STAGE0_TOLERANCE",
    "GAP_TYPICAL",
    "GAP_LARGEST",
    "RunPredictions",
    "load_predictions",
    "seed_mean_bootstrap",
    "stage0_verdict",
    "ensemble",
    "GroupResult",
    "PairResult",
    "RateComparison",
    "compare_groups",
    "seed_null_contrast",
    "interpret_delta",
]

#: §8.2, dalla Table 3 del benchmark: mediana del gap fra posizioni adiacenti.
GAP_TYPICAL = 0.006

#: §8.2: il piu' grande gap fra posizioni adiacenti nella stessa classifica.
GAP_LARGEST = 0.019

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


# --------------------------------------------------------------------------- #
# §8.1 — confronto fra rate
# --------------------------------------------------------------------------- #


def ensemble(runs: Sequence[RunPredictions]) -> np.ndarray:
    """Media delle **probabilita'** sui seed di uno stesso rate (§8.1).

    Diverso da §3, che media le metriche (§10 voce 19). Qui costruire un
    ensemble e' voluto: il confronto e' interno, la stessa operazione va a tutti
    i rate, e mediare le predizioni sopprime il rumore di inizializzazione
    lasciando l'effetto sistematico del rate, che e' l'unica cosa in esame.
    """
    if not runs:
        raise ValueError("nessun run da mediare")
    return np.mean([r.y_prob for r in runs], axis=0)


def seed_null_contrast(
    runs: Sequence[RunPredictions], columns: Sequence[int] | None = None
) -> list[float]:
    """Differenze fra ensemble **dentro** un gruppo (§10 voce 21).

    I seed vengono divisi in due sottoinsiemi disgiunti **della stessa
    dimensione** e si calcola la stessa differenza fra ensemble che §8.1 calcola
    fra rate. Con cinque seed sono 2 contro 2, quindici coppie, un seed escluso
    ogni volta.

    L'uguaglianza delle dimensioni non e' un dettaglio: sottoinsiemi diversi
    misurano quanto e' migliore un ensemble piu' grande, che e' un effetto vero
    di segno fisso e non ha niente a che vedere coi seed. E' l'errore della voce
    21, corretto dalla 22.

    E' il metro di §8.2, e la ragione per cui non basta la deviazione standard
    fra seed: Δ e' una differenza fra ensemble di cinque, mentre la dispersione
    dei run singoli e' un oggetto con circa cinque volte la varianza. Il metro
    resta comunque piu' largo del rumore vero di Δ di circa ``sqrt(5/2)``,
    perche' confronta ensemble di due; con cinque seed non si puo' fare di
    meglio, e la distorsione e' dichiarata invece che corretta.
    """
    n = len(runs)
    if n < 2:
        return []
    columns = None if columns is None else np.asarray(columns, dtype=int)
    y_true = runs[0].y_true if columns is None else runs[0].y_true[:, columns]
    probs = [r.y_prob if columns is None else r.y_prob[:, columns] for r in runs]

    def score(idx) -> float:
        return macro_auroc_fast(y_true, np.mean([probs[i] for i in idx], axis=0))

    k = n // 2
    values, seen = [], set()
    for left in combinations(range(n), k):
        rest = [i for i in range(n) if i not in left]
        for right in combinations(rest, k):
            key = frozenset((left, right))
            if key in seen:
                continue
            seen.add(key)
            values.append(score(left) - score(right))
    return values


#: Fattore che porta un contrasto fra ensemble di ``k`` alla scala di uno fra
#: ensemble di ``n``: sqrt(k / n). Riportato, non applicato al verdetto (voce 22).
def matching_factor(n_seeds: int) -> float:
    k = n_seeds // 2
    return float(np.sqrt(k / n_seeds)) if n_seeds >= 2 else float("nan")


@dataclass(frozen=True)
class GroupResult:
    """Un rate (o un braccio): ensemble, seed individuali, dispersione."""

    name: str
    point: float
    lo: float
    hi: float
    per_run: dict[str, float]
    null_contrasts: tuple[float, ...] = ()

    @property
    def null_median(self) -> float:
        """Mediana dei valori assoluti del contrasto nullo interno."""
        if not self.null_contrasts:
            return float("nan")
        return float(np.median(np.abs(self.null_contrasts)))

    @property
    def seed_spread(self) -> float:
        values = list(self.per_run.values())
        return float(max(values) - min(values)) if len(values) > 1 else 0.0

    @property
    def seed_std(self) -> float:
        values = list(self.per_run.values())
        return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


@dataclass(frozen=True)
class PairResult:
    """Un confronto accoppiato, con i tre numeri della voce 20."""

    a: str
    b: str
    diff: float
    lo: float
    hi: float
    p_raw: float
    p_holm: float
    lo_simultaneous: float
    hi_simultaneous: float

    @property
    def crosses_zero(self) -> bool:
        return self.lo <= 0.0 <= self.hi

    @property
    def significant(self) -> bool:
        """Holm sui valori p e' il criterio di §8.1, non l'intervallo grezzo."""
        return self.p_holm < 0.05


@dataclass(frozen=True)
class RateComparison:
    groups: list[GroupResult]
    pairs: list[PairResult]
    n_boot: int
    alpha: float

    @property
    def max_abs_diff(self) -> PairResult:
        """Δ di §8.2: il confronto con la differenza assoluta maggiore."""
        return max(self.pairs, key=lambda p: abs(p.diff))

    @property
    def null_contrast(self) -> float:
        """Il metro di §8.2, messo in comune fra i gruppi.

        Si mettono insieme tutte le partizioni di tutti i gruppi e se ne prende
        la mediana dei valori assoluti: con cinque seed per gruppo sono dieci
        partizioni ciascuno, e metterle insieme rende il metro piu' stabile
        senza privilegiare il gruppo che ha avuto piu' fortuna.
        """
        pooled = [v for g in self.groups for v in g.null_contrasts]
        if not pooled:
            return float("nan")
        return float(np.median(np.abs(pooled)))


def compare_groups(
    y_true: np.ndarray,
    groups: dict[str, Sequence[RunPredictions]],
    columns: Sequence[int] | None = None,
    *,
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> RateComparison:
    """Confronto accoppiato fra gruppi di run sullo stesso test set (§8.1).

    Un solo ``bootstrap_indices`` per tutti i gruppi, e le distribuzioni per
    gruppo vengono calcolate **una volta**: ogni differenza accoppiata e' poi la
    differenza di due colonne di estrazioni gia' pronte. Farne invece un
    bootstrap indipendente per ciascuna delle sei coppie costerebbe sei volte
    tanto e romperebbe l'accoppiamento fra confronti.

    Restituisce, per ogni coppia, i tre numeri della voce 20: intervallo
    percentile non corretto, valore p corretto con Holm, e intervallo
    simultaneo alla Bonferroni su ``alpha / n_coppie``.
    """
    if len(groups) < 2:
        raise ValueError("servono almeno due gruppi da confrontare")
    columns = None if columns is None else np.asarray(columns, dtype=int)
    if columns is not None:
        y_true = y_true[:, columns]

    def scores(runs: Sequence[RunPredictions]) -> np.ndarray:
        probs = ensemble(runs)
        return probs if columns is None else probs[:, columns]

    rng = np.random.default_rng(seed)
    idx = bootstrap_indices(len(y_true), n_boot, rng)

    names = list(groups)
    ens = {name: scores(groups[name]) for name in names}
    draws = {
        name: np.array([macro_auroc_fast(y_true[i], ens[name][i]) for i in idx])
        for name in names
    }

    results = []
    for name in names:
        lo, hi = percentile_ci(draws[name], alpha)
        per_run = {
            r.run_id: macro_auroc_fast(
                y_true, r.y_prob if columns is None else r.y_prob[:, columns]
            )
            for r in groups[name]
        }
        results.append(
            GroupResult(
                name=name,
                point=macro_auroc_fast(y_true, ens[name]),
                lo=lo,
                hi=hi,
                per_run=per_run,
                null_contrasts=tuple(seed_null_contrast(groups[name], columns)),
            )
        )

    pairs_raw = []
    for a, b in combinations(names, 2):
        d = draws[a] - draws[b]
        pairs_raw.append((a, b, d))

    m = len(pairs_raw)
    p_values = [p_value_from_draws(d) for _, _, d in pairs_raw]
    adjusted = holm(p_values)

    pairs = []
    for (a, b, d), p_raw, p_adj in zip(pairs_raw, p_values, adjusted):
        lo, hi = percentile_ci(d, alpha)
        lo_sim, hi_sim = percentile_ci(d, alpha / m)
        point_a = next(g.point for g in results if g.name == a)
        point_b = next(g.point for g in results if g.name == b)
        pairs.append(
            PairResult(
                a=a,
                b=b,
                diff=point_a - point_b,
                lo=lo,
                hi=hi,
                p_raw=float(p_raw),
                p_holm=float(p_adj),
                lo_simultaneous=lo_sim,
                hi_simultaneous=hi_sim,
            )
        )

    return RateComparison(groups=results, pairs=pairs, n_boot=n_boot, alpha=alpha)


def interpret_delta(comparison: RateComparison) -> dict:
    """Le soglie di §8.2, applicate a Δ e fissate prima di vedere i numeri.

    L'ordine dei controlli e' quello del protocollo: prima si guarda se
    l'effetto e' distinguibile da zero e dalla dispersione fra seed, e solo poi
    quanto e' grande. Invertirlo permetterebbe di chiamare "grande" un effetto
    che non e' nemmeno rilevabile.
    """
    delta = comparison.max_abs_diff
    magnitude = abs(delta.diff)
    max_spread = max((g.seed_spread for g in comparison.groups), default=0.0)

    # Il metro e' il contrasto nullo interno (voce 21). Se un gruppo ha un solo
    # seed non esiste, e si ripiega sulla dispersione fra run dichiarandolo:
    # un metro diverso da quello pre-registrato non deve passare in silenzio.
    comparator = comparison.null_contrast
    comparator_name = "null_contrast"
    if not np.isfinite(comparator):
        comparator = max_spread
        comparator_name = "seed_spread (ripiego: contrasto nullo non calcolabile)"

    if delta.crosses_zero or magnitude <= comparator:
        verdict = "non rilevabile a questa scala"
    elif magnitude < GAP_TYPICAL:
        verdict = "rilevabile ma sotto il gap tipico fra modelli adiacenti"
    elif magnitude < GAP_LARGEST:
        verdict = "confrontabile con i gap che la classifica tratta come differenze"
    else:
        verdict = "capace di scavalcare qualsiasi gap della classifica"

    return {
        "delta_pair": [delta.a, delta.b],
        "delta": delta.diff,
        "delta_abs": magnitude,
        "ci": [delta.lo, delta.hi],
        "ci_simultaneous": [delta.lo_simultaneous, delta.hi_simultaneous],
        "p_holm": delta.p_holm,
        "crosses_zero": delta.crosses_zero,
        "comparator": comparator_name,
        "comparator_value": comparator,
        "null_contrast": comparison.null_contrast,
        # Riportato accanto, non usato dal verdetto: il metro grezzo confronta
        # ensemble di due, Δ ensemble di cinque (voce 22).
        "null_contrast_matched": comparison.null_contrast * matching_factor(
            max((len(g.per_run) for g in comparison.groups), default=0)
        ),
        "max_seed_spread": max_spread,
        "max_seed_std": max((g.seed_std for g in comparison.groups), default=0.0),
        "gap_typical": GAP_TYPICAL,
        "gap_largest": GAP_LARGEST,
        "verdict": verdict,
    }


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
