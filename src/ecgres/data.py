"""Caricamento, filtraggio, ricampionamento e finestratura di PTB-XL.

Questo modulo e' il livello dati dell'esperimento: tenendo fissi architettura,
task, durata della finestra in secondi, schedule e seed, produce viste dello
stesso segnale che differiscono *solo* per la frequenza di campionamento.

Scelte di progetto (vedi PROTOCOL.md):

* **NumPy puro, nessuna dipendenza da torch.** Il wrapper ``Dataset`` vive in
  ``model.py``. Cosi' i test girano senza GPU e senza torch installato.
* **Sorgente unica ``records500``.** Ogni rate, compreso quello a 100 Hz, nasce
  da un ricampionamento di ``records500``. ``records100`` e' ammesso solo per
  l'analisi secondaria 8.3.2 e va richiesto esplicitamente.
* **Notch 50 Hz prima del ricampionamento**, a 500 Hz, uguale per tutti i
  bracci (emendamento 2 del 26/08/2026). Disattivabile con ``notch=False``, che
  serve allo Stage 0 e all'analisi 8.3.4.
* **Cache su disco, una per coppia ``(fs, notch)``**, costruita on-demand come
  memmap ``float32``. Float32 e non int16: l'errore di quantizzazione a 16 bit
  non sarebbe identico fra i quattro rate, e le soglie di interpretazione del
  protocollo partono da 0,006 di AUROC.
* **Griglia dei crop a 0,1 s.** Un istante ``t`` produce un indice campione
  intero a tutti e quattro i rate solo se ``t * fs`` e' intero per ogni
  ``fs in {100, 240, 250, 500}``, cioe' solo se ``t`` e' multiplo di
  ``1 / gcd(100, 240, 250, 500) = 1/10`` s. Fuori da questa griglia due viste
  differirebbero anche per l'offset temporale, non solo per la frequenza.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.signal import filtfilt, iirnotch, resample_poly

__all__ = [
    "SOURCE_FS",
    "RATES",
    "RESAMPLE_FACTORS",
    "WINDOW_SECONDS",
    "RECORD_SECONDS",
    "CROP_GRID_SECONDS",
    "NOTCH_FREQ",
    "NOTCH_Q",
    "N_LEADS",
    "SPLIT_FOLDS",
    "WindowBatch",
    "Splits",
    "load_metadata",
    "load_label_matrix",
    "get_splits",
    "notch_50hz",
    "resample_to",
    "preprocess_record",
    "n_samples_for",
    "grid_index",
    "crop_start_grid",
    "inference_crop_starts",
    "SignalCache",
    "assert_aligned",
]

# --------------------------------------------------------------------------
# Costanti dell'esperimento
# --------------------------------------------------------------------------

SOURCE_FS = 500
"""Frequenza della sorgente. Tutto nasce da qui, sempre."""

RATES: tuple[int, ...] = (100, 240, 250, 500)
"""Le quattro frequenze osservate sul campo (RELATED_WORK.md, tabella run.sh)."""

RESAMPLE_FACTORS: dict[int, tuple[int, int]] = {
    100: (1, 5),
    240: (12, 25),
    250: (1, 2),
    500: (1, 1),
}
"""``(up, down)`` per ``resample_poly`` a partire da 500 Hz. Tutti interi."""

WINDOW_SECONDS = 2.5
RECORD_SECONDS = 10.0
N_INFERENCE_CROPS = 4
N_LEADS = 12

CROP_GRID_SECONDS = 0.1
"""1 / gcd(100, 240, 250, 500). Vedi il docstring del modulo."""

NOTCH_FREQ = 50.0
NOTCH_Q = 30.0

SPLIT_FOLDS: dict[str, tuple[int, ...]] = {
    "train": (1, 2, 3, 4, 5, 6, 7, 8),
    "val": (9,),
    "test": (10,),
}

_CACHE_FORMAT_VERSION = 1


# --------------------------------------------------------------------------
# Verifiche di coerenza sulle costanti (falliscono all'import, non a meta' run)
# --------------------------------------------------------------------------

def _check_constants() -> None:
    for fs in RATES:
        up, down = RESAMPLE_FACTORS[fs]
        if SOURCE_FS * up != fs * down:
            raise ValueError(f"fattore di ricampionamento errato per {fs} Hz")
        for seconds in (WINDOW_SECONDS, RECORD_SECONDS, CROP_GRID_SECONDS):
            exact = seconds * fs
            if abs(exact - round(exact)) > 1e-9:
                raise ValueError(
                    f"{seconds} s non e' un numero intero di campioni a {fs} Hz"
                )
    if abs(RECORD_SECONDS / WINDOW_SECONDS - N_INFERENCE_CROPS) > 1e-9:
        raise ValueError("i crop di inferenza non tassellano esattamente il record")


_check_constants()


# --------------------------------------------------------------------------
# Metadati ed etichette
# --------------------------------------------------------------------------

def load_metadata(root: str | Path) -> pd.DataFrame:
    """Legge ``ptbxl_database.csv``, indicizzato per ``ecg_id``.

    ``scp_codes`` viene deserializzato in un dizionario Python. L'ordine delle
    righe e' quello del file: e' l'ordine su cui si basa la cache, quindi non
    va riordinato a valle.
    """
    root = Path(root)
    meta = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    meta["scp_codes"] = meta["scp_codes"].apply(ast.literal_eval)
    missing = {"filename_hr", "filename_lr", "strat_fold", "patient_id"} - set(meta.columns)
    if missing:
        raise ValueError(f"colonne assenti in ptbxl_database.csv: {sorted(missing)}")
    return meta


def load_label_matrix(
    root: str | Path,
    meta: pd.DataFrame,
    *,
    label_set: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Matrice binaria ``(n_record, n_etichette)``.

    Presenza, non likelihood: un codice conta come positivo se compare in
    ``scp_codes``, anche con likelihood 0.0. Le etichette sono tutte quelle di
    ``scp_statements.csv`` salvo che non venga passato ``label_set``, nel qual
    caso viene usato quello (serve per ``frozen_label_set.json``, il set
    pre-registrato a 49 etichette dell'endpoint ``macro_clean``).
    """
    root = Path(root)
    statements = pd.read_csv(root / "scp_statements.csv", index_col=0)
    labels = list(statements.index) if label_set is None else list(label_set)
    position = {name: i for i, name in enumerate(labels)}

    y = np.zeros((len(meta), len(labels)), dtype=np.uint8)
    for row, codes in enumerate(meta["scp_codes"]):
        for code in codes:
            col = position.get(code)
            if col is not None:
                y[row, col] = 1
    return y, labels


@dataclass(frozen=True)
class Splits:
    """Indici posizionali (non ``ecg_id``) nelle righe di ``meta``."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"train": self.train, "val": self.val, "test": self.test}


def get_splits(meta: pd.DataFrame) -> Splits:
    """Split per ``strat_fold``: 1-8 train, 9 validation, 10 test.

    Verifica che nessun paziente compaia in piu' di uno split. I fold ufficiali
    sono gia' stratificati per paziente, ma il controllo costa poco e un
    leakage qui invaliderebbe l'intero confronto.
    """
    fold = meta["strat_fold"].to_numpy()
    parts = {
        name: np.flatnonzero(np.isin(fold, folds))
        for name, folds in SPLIT_FOLDS.items()
    }

    patients = meta["patient_id"].to_numpy()
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = np.intersect1d(patients[parts[a]], patients[parts[b]])
            if shared.size:
                raise ValueError(
                    f"{shared.size} pazienti condivisi fra {a} e {b}: "
                    f"primi id {shared[:5].tolist()}"
                )
    return Splits(**parts)


# --------------------------------------------------------------------------
# Filtro e ricampionamento
# --------------------------------------------------------------------------

def notch_50hz(x: np.ndarray, fs: int = SOURCE_FS, *, q: float = NOTCH_Q) -> np.ndarray:
    """Notch IIR a 50 Hz, applicato a fase zero con ``filtfilt``.

    Fase zero perche' uno sfasamento dipendente dalla frequenza si tradurrebbe
    in un disallineamento temporale fra i bracci, cioe' esattamente la
    confusione che il disegno vuole escludere.

    Va applicato **a 500 Hz, prima del ricampionamento**: e' il senso
    dell'emendamento 2. La fondamentale di rete sta esattamente sul Nyquist del
    braccio a 100 Hz, e senza notch quel braccio riceverebbe dal filtro
    anti-aliasing di ``resample_poly`` una soppressione parziale del disturbo
    che gli altri non hanno.
    """
    if fs != SOURCE_FS:
        raise ValueError(
            f"il notch va applicato a {SOURCE_FS} Hz prima del ricampionamento, "
            f"non a {fs} Hz"
        )
    b, a = iirnotch(w0=NOTCH_FREQ, Q=q, fs=fs)
    return filtfilt(b, a, x, axis=-1)


def resample_to(x: np.ndarray, target_fs: int) -> np.ndarray:
    """Ricampiona da 500 Hz a ``target_fs`` con ``resample_poly``, fattori interi."""
    if target_fs not in RESAMPLE_FACTORS:
        raise ValueError(f"rate non previsto dal protocollo: {target_fs}")
    up, down = RESAMPLE_FACTORS[target_fs]
    if (up, down) == (1, 1):
        return np.asarray(x, dtype=np.float64)
    return resample_poly(x, up, down, axis=-1)


def preprocess_record(x500: np.ndarray, target_fs: int, *, notch: bool = True) -> np.ndarray:
    """Notch (opzionale) a 500 Hz, poi ricampionamento. In quest'ordine."""
    x = np.asarray(x500, dtype=np.float64)
    if notch:
        x = notch_50hz(x, SOURCE_FS)
    return resample_to(x, target_fs).astype(np.float32, copy=False)


# --------------------------------------------------------------------------
# Griglia temporale
# --------------------------------------------------------------------------

def n_samples_for(seconds: float, fs: int) -> int:
    """Campioni in ``seconds`` a ``fs``, con errore se non e' un intero esatto."""
    exact = seconds * fs
    n = round(exact)
    if abs(exact - n) > 1e-9:
        raise ValueError(f"{seconds} s non e' un numero intero di campioni a {fs} Hz")
    return n


def grid_index(start_s: float, fs: int) -> int:
    """Indice campione di ``start_s``, con due controlli **in quest'ordine**.

    Prima la griglia a ``CROP_GRID_SECONDS``, poi l'integralita' dell'indice a
    ``fs``. Sono condizioni diverse e la prima e' piu' forte: 0,05 s da' un
    indice intero a 100 Hz (5) e a 240 Hz (12), ma non a 250 Hz (12,5). Solo la
    griglia garantisce tutti e quattro i rate **insieme**, che e' l'invariante
    su cui poggia il confronto accoppiato.

    E' l'unico punto in cui i secondi diventano campioni: la conversione non va
    replicata a valle.
    """
    k = start_s / CROP_GRID_SECONDS
    if abs(k - round(k)) > 1e-9:
        raise ValueError(
            f"start {start_s} s non cade sulla griglia da {CROP_GRID_SECONDS} s"
        )
    exact = start_s * fs
    idx = round(exact)
    if abs(exact - idx) > 1e-9:
        raise ValueError(
            f"start {start_s} s non cade su un campione intero a {fs} Hz"
        )
    return int(idx)


def crop_start_grid(
    *,
    window_seconds: float = WINDOW_SECONDS,
    record_seconds: float = RECORD_SECONDS,
    grid: float = CROP_GRID_SECONDS,
) -> np.ndarray:
    """Istanti di partenza ammessi per un crop, in secondi.

    Multipli di ``grid`` da 0 fino a ``record_seconds - window_seconds``
    inclusi. Con i valori di default: 76 posizioni da 0,0 a 7,5 s.
    """
    last = record_seconds - window_seconds
    n = int(round(last / grid)) + 1
    return np.round(np.arange(n) * grid, 6)


def inference_crop_starts() -> np.ndarray:
    """I 4 crop non sovrapposti dell'inferenza: 0,0 / 2,5 / 5,0 / 7,5 s.

    Tassellano esattamente i 10 s del record, quindi non c'e' nulla da
    scegliere e nulla da mediare in modo dipendente dal seed.
    """
    return np.round(np.arange(N_INFERENCE_CROPS) * WINDOW_SECONDS, 6)


# --------------------------------------------------------------------------
# Vista finestrata
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class WindowBatch:
    """Un insieme di finestre estratte a una singola frequenza.

    ``signals`` ha forma ``(n_finestre, N_LEADS, n_campioni)``; ``record_idx``
    e ``start_s`` dicono da dove viene ciascuna finestra.
    """

    fs: int
    signals: np.ndarray
    record_idx: np.ndarray
    start_s: np.ndarray
    notch: bool

    def __post_init__(self) -> None:
        n = self.signals.shape[0]
        if self.signals.ndim != 3 or self.signals.shape[1] != N_LEADS:
            raise ValueError(f"forma inattesa: {self.signals.shape}")
        if self.record_idx.shape != (n,) or self.start_s.shape != (n,):
            raise ValueError("record_idx e start_s devono avere lunghezza n_finestre")

    @property
    def n_samples(self) -> int:
        return int(self.signals.shape[2])

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.fs


# --------------------------------------------------------------------------
# Cache su disco
# --------------------------------------------------------------------------

class SignalCache:
    """Memmap ``float32`` di forma ``(n_record, N_LEADS, n_campioni)``.

    Una cache per coppia ``(fs, notch)``, costruita on-demand: un singolo
    braccio materializza solo cio' che gli serve invece dei ~11,4 GB di tutti i
    rate insieme. Il file di metadati registra un hash dell'ordine dei record,
    cosi' una cache costruita su una lista diversa viene rifiutata invece di
    essere usata silenziosamente.
    """

    def __init__(
        self,
        root: str | Path,
        cache_dir: str | Path,
        meta: pd.DataFrame,
        *,
        fs: int,
        notch: bool = True,
        source: str = "hr",
    ) -> None:
        if fs not in RATES:
            raise ValueError(f"rate non previsto dal protocollo: {fs}")
        if source not in ("hr", "lr"):
            raise ValueError("source deve essere 'hr' (records500) o 'lr' (records100)")
        if source == "lr":
            # Ammesso solo per l'analisi secondaria 8.3.2, e solo a 100 Hz.
            if fs != 100:
                raise ValueError("records100 e' utilizzabile solo a 100 Hz (analisi 8.3.2)")

        self.root = Path(root)
        self.cache_dir = Path(cache_dir)
        self.meta = meta
        self.fs = fs
        self.notch = notch
        self.source = source
        self.n_record_samples = n_samples_for(RECORD_SECONDS, fs)
        self._array: np.ndarray | None = None

    # -- percorsi -----------------------------------------------------------

    @property
    def stem(self) -> str:
        return f"ptbxl_{self.source}_fs{self.fs}_notch{int(self.notch)}"

    @property
    def array_path(self) -> Path:
        return self.cache_dir / f"{self.stem}.npy"

    @property
    def meta_path(self) -> Path:
        return self.cache_dir / f"{self.stem}.json"

    def _fingerprint(self) -> str:
        payload = ",".join(str(i) for i in self.meta.index.tolist())
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    # -- costruzione --------------------------------------------------------

    def build(self, *, overwrite: bool = False, progress: bool = False) -> Path:
        """Costruisce la cache. Idempotente se i metadati coincidono."""
        if self.array_path.exists() and not overwrite and self._meta_matches():
            return self.array_path

        import wfdb  # import locale: serve solo qui

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        column = "filename_hr" if self.source == "hr" else "filename_lr"
        paths = self.meta[column].tolist()

        out = np.lib.format.open_memmap(
            self.array_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(paths), N_LEADS, self.n_record_samples),
        )
        try:
            for row, rel in enumerate(paths):
                signal, _ = wfdb.rdsamp(str(self.root / rel))
                x = np.asarray(signal, dtype=np.float64).T  # (leads, campioni)
                if x.shape[0] != N_LEADS:
                    raise ValueError(f"{rel}: {x.shape[0]} derivazioni invece di {N_LEADS}")
                if self.source == "hr":
                    x = preprocess_record(x, self.fs, notch=self.notch)
                elif self.notch:
                    raise ValueError(
                        "il notch richiede records500: a 100 Hz la fondamentale "
                        "e' gia' sul Nyquist e il filtro non e' piu' applicabile"
                    )
                else:
                    x = x.astype(np.float32, copy=False)
                if x.shape[1] != self.n_record_samples:
                    raise ValueError(
                        f"{rel}: {x.shape[1]} campioni invece di {self.n_record_samples}"
                    )
                out[row] = x
                if progress and row % 1000 == 0:
                    print(f"  {row}/{len(paths)}", flush=True)
            out.flush()
        finally:
            del out

        self.meta_path.write_text(
            json.dumps(
                {
                    "version": _CACHE_FORMAT_VERSION,
                    "fs": self.fs,
                    "notch": self.notch,
                    "source": self.source,
                    "n_records": len(paths),
                    "n_samples": self.n_record_samples,
                    "record_fingerprint": self._fingerprint(),
                },
                indent=2,
            )
        )
        return self.array_path

    def _meta_matches(self) -> bool:
        if not self.meta_path.exists():
            return False
        info = json.loads(self.meta_path.read_text())
        return (
            info.get("version") == _CACHE_FORMAT_VERSION
            and info.get("fs") == self.fs
            and info.get("notch") is self.notch
            and info.get("source") == self.source
            and info.get("n_records") == len(self.meta)
            and info.get("n_samples") == self.n_record_samples
            and info.get("record_fingerprint") == self._fingerprint()
        )

    # -- lettura ------------------------------------------------------------

    @property
    def array(self) -> np.ndarray:
        if self._array is None:
            if not self._meta_matches():
                raise FileNotFoundError(
                    f"cache assente o non coerente: {self.array_path}. "
                    "Chiamare build() prima."
                )
            self._array = np.load(self.array_path, mmap_mode="r")
        return self._array

    def window(
        self,
        record_idx: int,
        start_s: float,
        *,
        window_seconds: float = WINDOW_SECONDS,
    ) -> np.ndarray:
        """Una singola finestra ``(N_LEADS, n_campioni)``, in ``float32``.

        E' l'accesso elementare alla cache: ``crop`` e' un ciclo su questo.
        Prende l'istante in **secondi**, non in campioni, perche' i secondi sono
        la coordinata condivisa fra i bracci; la conversione avviene qui, una
        volta sola, in ``grid_index``.

        Restituisce una **copia**, non una vista sul memmap: chi consuma la
        finestra la normalizza subito dopo, e una vista scriverebbe sulla cache.
        """
        width = n_samples_for(window_seconds, self.fs)
        begin = grid_index(start_s, self.fs)
        if begin < 0 or begin + width > self.n_record_samples:
            raise ValueError(
                f"finestra fuori dal record: start {start_s} s a {self.fs} Hz"
            )
        return np.array(
            self.array[record_idx, :, begin : begin + width], dtype=np.float32
        )

    def crop(self, record_idx: Sequence[int] | np.ndarray, start_s: Sequence[float] | np.ndarray,
             *, window_seconds: float = WINDOW_SECONDS) -> WindowBatch:
        """Estrae una finestra per ciascuna coppia ``(record_idx, start_s)``."""
        record_idx = np.asarray(record_idx, dtype=np.int64)
        start_s = np.asarray(start_s, dtype=np.float64)
        if record_idx.shape != start_s.shape:
            raise ValueError("record_idx e start_s devono avere la stessa forma")

        width = n_samples_for(window_seconds, self.fs)
        out = np.empty((record_idx.size, N_LEADS, width), dtype=np.float32)
        for i, (r, t) in enumerate(zip(record_idx, start_s)):
            out[i] = self.window(int(r), float(t), window_seconds=window_seconds)
        return WindowBatch(
            fs=self.fs,
            signals=out,
            record_idx=record_idx,
            start_s=start_s,
            notch=self.notch,
        )

    def inference_batch(self, record_idx: Sequence[int] | np.ndarray) -> WindowBatch:
        """I 4 crop non sovrapposti per ciascun record, in ordine di record."""
        record_idx = np.asarray(record_idx, dtype=np.int64)
        starts = inference_crop_starts()
        return self.crop(
            np.repeat(record_idx, starts.size),
            np.tile(starts, record_idx.size),
        )

    def random_batch(
        self,
        record_idx: Sequence[int] | np.ndarray,
        rng: np.random.Generator,
    ) -> WindowBatch:
        """Un crop random per record, con partenza sulla griglia a 0,1 s."""
        record_idx = np.asarray(record_idx, dtype=np.int64)
        grid = crop_start_grid()
        starts = rng.choice(grid, size=record_idx.size, replace=True)
        return self.crop(record_idx, starts)


# --------------------------------------------------------------------------
# Allineamento fra viste
# --------------------------------------------------------------------------

def assert_aligned(
    a: WindowBatch,
    b: WindowBatch,
    *,
    check_signal: bool = False,
    min_corr: float = 0.95,
) -> None:
    """Verifica che due viste differiscano **solo** per la frequenza.

    Controlla, nell'ordine: stessi record, stessi istanti di partenza in
    secondi, stessa durata, conteggio campioni coerente con la frequenza,
    stesso stato del notch. Con ``check_signal=True`` ricampiona la vista a
    frequenza maggiore su quella minore e richiede una correlazione media
    almeno ``min_corr``: e' un controllo piu' lento, utile nei test e prima di
    un run lungo, non a ogni batch.
    """
    if a.fs == b.fs:
        raise ValueError("le due viste hanno la stessa frequenza: non c'e' nulla da allineare")
    if a.notch != b.notch:
        raise AssertionError(f"notch diverso: {a.notch} vs {b.notch}")
    if not np.array_equal(a.record_idx, b.record_idx):
        raise AssertionError("le due viste non coprono gli stessi record nello stesso ordine")
    if not np.allclose(a.start_s, b.start_s, atol=1e-9):
        raise AssertionError("istanti di partenza diversi fra le due viste")
    if abs(a.duration_s - b.duration_s) > 1e-9:
        raise AssertionError(
            f"durata diversa: {a.duration_s:.6f} s a {a.fs} Hz contro "
            f"{b.duration_s:.6f} s a {b.fs} Hz"
        )
    for view in (a, b):
        expected = n_samples_for(view.duration_s, view.fs)
        if view.n_samples != expected:
            raise AssertionError(
                f"a {view.fs} Hz: {view.n_samples} campioni invece di {expected}"
            )

    if not check_signal:
        return

    hi, lo = (a, b) if a.fs > b.fs else (b, a)
    ratio = hi.fs / lo.fs
    up, down = _integer_ratio(lo.fs, hi.fs)
    down_sampled = resample_poly(hi.signals.astype(np.float64), up, down, axis=-1)
    if down_sampled.shape != lo.signals.shape:
        raise AssertionError(
            f"ricampionando {hi.fs} -> {lo.fs} Hz (rapporto {ratio:g}) la forma "
            f"non torna: {down_sampled.shape} contro {lo.signals.shape}"
        )
    corr = _mean_corr(down_sampled, lo.signals.astype(np.float64))
    if corr < min_corr:
        raise AssertionError(
            f"correlazione media {corr:.4f} sotto la soglia {min_corr}: "
            "le due viste non sono lo stesso segnale"
        )


def _integer_ratio(target_fs: int, source_fs: int) -> tuple[int, int]:
    g = math.gcd(target_fs, source_fs)
    return target_fs // g, source_fs // g


def _mean_corr(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean(axis=-1, keepdims=True)
    yc = y - y.mean(axis=-1, keepdims=True)
    num = (xc * yc).sum(axis=-1)
    den = np.sqrt((xc**2).sum(axis=-1) * (yc**2).sum(axis=-1))
    valid = den > 0
    if not valid.any():
        return 0.0
    return float((num[valid] / den[valid]).mean())


# --------------------------------------------------------------------------
# Convenienza
# --------------------------------------------------------------------------

def open_caches(
    root: str | Path,
    cache_dir: str | Path,
    meta: pd.DataFrame,
    rates: Iterable[int] = RATES,
    *,
    notch: bool = True,
    build: bool = False,
) -> dict[int, SignalCache]:
    """Una ``SignalCache`` per rate, opzionalmente costruendole."""
    caches: dict[int, SignalCache] = {}
    for fs in rates:
        cache = SignalCache(root, cache_dir, meta, fs=fs, notch=notch)
        if build:
            cache.build()
        caches[fs] = cache
    return caches
