"""Modello e wrapper torch per il confronto fra frequenze di campionamento.

Questo modulo contiene tutto ciò che dipende da torch. Il core NumPy
(caricamento, notch, ricampionamento, finestratura, ``SignalCache``) resta in
``ecgres.data``.

Principio guida
---------------
Il conteggio dei parametri deve essere identico ai quattro rate. Cambiando
``fs`` cambia solo la lunghezza dell'input. Ogni componente introdotto qui deve
rispettare questo invariante, che e' verificato in ``tests/test_model.py``:

* encoder pointwise (``Conv1d`` con ``kernel_size=1``), mai un kernel > 1;
* pooling temporale a **media**, mai somma ne' flatten;
* testa lineare su ``d_model``, mai su ``d_model * L``.

L'unico iperparametro che dipende da ``fs`` e' la coppia ``dt_min``/``dt_max``
del kernel S4, ed e' esattamente cio' che distingue i due bracci.

Provenienza
-----------
Il layer S4 e' vendorato in ``ecgres.vendor.s4`` da ``HazyResearch/state-spaces``
(Apache-2.0). Vedere l'header di quel file per commit e modifiche.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Sequence

import numpy as np
import torch
import torch.nn as nn

from .vendor.s4 import S4  # Apache-2.0, vedere header del file

__all__ = [
    "RATES",
    "ModelConfig",
    "S4Backbone",
    "build_model",
    "decouple_aliased_state",
    "GlobalScaler",
    "FIT_FS",
    "CropSampler",
    "INFERENCE_CROP_STARTS_S",
    "WindowDataset",
    "count_parameters",
]

# --------------------------------------------------------------------------- #
# Costanti
# --------------------------------------------------------------------------- #

RATES: tuple[int, ...] = (100, 240, 250, 500)
N_LEADS = 12

WINDOW_SECONDS = 2.5
GRID_SECONDS = 0.1  # 1 / gcd(100, 240, 250, 500); vedi ecgres.data.assert_aligned
RECORD_SECONDS = 10.0

#: I 4 crop di inferenza tassellano esattamente i 10 s del record (§ inferenza).
INFERENCE_CROP_STARTS_S: tuple[float, ...] = (0.0, 2.5, 5.0, 7.5)

#: Default upstream di S4 (``s4.py``, ``dt_min``/``dt_max``). Arm A li usa tali
#: e quali; Arm B li scala di ``100 / fs``.
DT_MIN_DEFAULT = 1e-3
DT_MAX_DEFAULT = 1e-1

#: Rate a cui i default upstream sono "nativi": a 100 Hz coprono orizzonti
#: 0,1-10 s. La compensazione di Arm B riporta ogni rate su questi orizzonti.
DT_REFERENCE_FS = 100

#: Frequenza a cui lo scaler globale viene fittato, una volta sola (voce 8).
FIT_FS = 500

_SCALER_FORMAT_VERSION = 1


# --------------------------------------------------------------------------- #
# Configurazione
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelConfig:
    """Configurazione di una cella dell'esperimento.

    Iperparametri di default presi dal run S4 supervisionato di riferimento su
    PTB-XL ``label_all`` (batch 128, 100 epoche, lr 1e-3, AdamW). Quelli che
    seguono sono i soli che toccano l'architettura.
    """

    fs: int
    arm: Literal["A", "B"]
    n_classes: int

    d_model: int = 512
    d_state: int = 8
    n_layers: int = 4
    dropout: float = 0.2
    bidirectional: bool = True
    prenorm: bool = False
    norm: Literal["layer", "batch"] = "layer"

    window_seconds: float = WINDOW_SECONDS

    def __post_init__(self) -> None:
        if self.fs not in RATES:
            raise ValueError(f"fs deve stare in {RATES}, ricevuto {self.fs}")
        if self.arm not in ("A", "B"):
            raise ValueError(f"arm deve essere 'A' o 'B', ricevuto {self.arm!r}")
        n = self.window_seconds * self.fs
        if abs(n - round(n)) > 1e-9:
            raise ValueError(
                f"finestra di {self.window_seconds} s non da' un numero intero "
                f"di campioni a {self.fs} Hz"
            )

    @property
    def n_samples(self) -> int:
        """Lunghezza della finestra in campioni: 250 / 600 / 625 / 1250."""
        return int(round(self.window_seconds * self.fs))

    @property
    def dt_scale(self) -> float:
        """Fattore applicato a ``dt_min``/``dt_max``.

        Arm A (naive): 1.0, i default upstream a ogni rate.
        Arm B (compensato): ``100 / fs``, che riporta gli orizzonti temporali
        in secondi a quelli che i default coprono a 100 Hz.

        A ``fs == 100`` i due bracci coincidono per costruzione.
        """
        return 1.0 if self.arm == "A" else DT_REFERENCE_FS / self.fs

    @property
    def dt_min(self) -> float:
        return DT_MIN_DEFAULT * self.dt_scale

    @property
    def dt_max(self) -> float:
        return DT_MAX_DEFAULT * self.dt_scale


# --------------------------------------------------------------------------- #
# Modello
# --------------------------------------------------------------------------- #


class S4Backbone(nn.Module):
    """Stack S4 senza encoder convoluzionale.

    Input  ``(B, 12, L)`` -> output ``(B, n_classes)`` (logit).

    Il forward **non** accetta ne' propaga l'argomento ``rate`` di S4. La
    compensazione di Arm B vive interamente in ``dt_min``/``dt_max`` alla
    costruzione, dove e' interpretabile come prior sulle scale temporali; il
    percorso ``rate`` in piu' scarta i nodi FFT cachati e puo' raddoppiare la
    lunghezza interna del kernel.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Pointwise: nessun kernel, nessuno stride, nessun parametro legato a L.
        self.encoder = nn.Conv1d(N_LEADS, cfg.d_model, kernel_size=1)

        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(cfg.n_layers):
            self.blocks.append(
                S4(
                    d_model=cfg.d_model,
                    d_state=cfg.d_state,
                    l_max=cfg.n_samples,
                    bidirectional=cfg.bidirectional,
                    postact="glu",
                    dropout=cfg.dropout,
                    transposed=True,
                    dt_min=cfg.dt_min,
                    dt_max=cfg.dt_max,
                )
            )
            self.norms.append(
                nn.BatchNorm1d(cfg.d_model)
                if cfg.norm == "batch"
                else nn.LayerNorm(cfg.d_model)
            )
            # Dropout1d, non Dropout2d: su input 3D fanno oggi la stessa cosa
            # (dropout per canale), ma torch avverte che la semantica di
            # Dropout2d cambiera'. Dropout1d la fissa esplicitamente.
            self.dropouts.append(nn.Dropout1d(cfg.dropout))

        self.head = nn.Linear(cfg.d_model, cfg.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] != N_LEADS:
            raise ValueError(f"attese 12 derivazioni, ricevute {x.shape[-2]}")

        x = self.encoder(x)  # (B, 12, L) -> (B, d_model, L)

        for block, norm, drop in zip(self.blocks, self.norms, self.dropouts):
            z = x
            if self.cfg.prenorm:
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            z, _ = block(z)
            z = drop(z)
            x = z + x
            if not self.cfg.prenorm:
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        # Media, non somma: unica riduzione temporale invariante rispetto a L.
        x = x.mean(dim=-1)  # (B, d_model)
        return self.head(x)


def _is_aliased(t: torch.Tensor) -> bool:
    """Vero se piu' indici del tensore indirizzano la stessa cella di memoria."""
    return any(stride == 0 for stride in t.stride())


def decouple_aliased_state(model: nn.Module) -> list[str]:
    """Materializza i tensori del kernel S4 che condividono memoria.

    ``s4.py`` inizializza A, B e P con ``einops.repeat``, che non copia ma
    espande: tutti gli head puntano alla stessa cella. E' innocuo dal punto di
    vista numerico — sono buffer, non parametri, quindi nessuno ci scrive, e i
    valori letti sono gli stessi — ma rende il modello **non ricaricabile**:
    ``load_state_dict`` copia in-place e rifiuta una destinazione aliasata con
    "more than one element of the written-to tensor refers to a single memory
    location".

    Senza questa normalizzazione il resume di §6.7 non funziona, e con esso
    cade l'uso di istanze interrompibili.

    Sostituire le viste con copie contigue **non cambia un solo valore**: cambia
    il layout. Va fatto alla costruzione, prima che l'ottimizzatore prenda i
    riferimenti ai tensori. Restituisce i nomi toccati.
    """
    touched: list[str] = []
    for module_name, module in model.named_modules():
        prefix = f"{module_name}." if module_name else ""

        for name, buf in list(module._buffers.items()):
            if buf is None or not _is_aliased(buf):
                continue
            persistent = name not in module._non_persistent_buffers_set
            module.register_buffer(name, buf.contiguous(), persistent=persistent)
            touched.append(prefix + name)

        for name, param in list(module._parameters.items()):
            if param is None or not _is_aliased(param.data):
                continue
            module._parameters[name] = nn.Parameter(
                param.data.contiguous(), requires_grad=param.requires_grad
            )
            touched.append(prefix + name)

    return touched


def build_model(cfg: ModelConfig, seed: int) -> S4Backbone:
    """Costruisce il modello con RNG seedato.

    A parita' di ``seed`` e di ``arm``, i pesi iniziali sono **identici ai
    quattro rate**: nessuna shape dipende da ``fs``. Verificato in
    ``test_init_identical_across_rates``.
    """
    torch.manual_seed(seed)
    model = S4Backbone(cfg)
    decouple_aliased_state(model)
    return model


def count_parameters(model: nn.Module) -> int:
    """Somma di ``numel`` sui soli ``Parameter``.

    Non usare ``state_dict``: i buffer del kernel S4 (``omega``, ``z``)
    dipendono dalla lunghezza e differiscono legittimamente fra rate.
    """
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- #
# Normalizzazione (§10 voci 7 e 8)
# --------------------------------------------------------------------------- #


class GlobalScaler:
    """Media e deviazione standard **scalari**, non per derivazione (voce 7).

    Replica ``apply_standardizer`` del riferimento, che fitta su
    ``x.flatten()[:, np.newaxis]``.

    Il fit avviene **una volta sola**, sui fold 1-8 a 500 Hz, e lo scaler viene
    applicato invariato a tutti i bracci (voce 8): rifittarlo per braccio
    assorbirebbe nelle costanti la differenza di varianza fra rate, cancellando
    parte dell'effetto in esame.
    """

    def __init__(
        self, mean: float, std: float, provenance: dict | None = None
    ) -> None:
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
            raise ValueError(f"scaler non valido: mean={mean}, std={std}")
        self.mean = float(mean)
        self.std = float(std)
        #: Da dove viene il fit. Serializzata insieme a mean/std: uno scaler
        #: senza provenienza non e' verificabile a posteriori.
        self.provenance = dict(provenance or {})

    @classmethod
    def fit(
        cls,
        signals,
        record_idx: Sequence[int] | np.ndarray | None = None,
        *,
        chunk: int = 256,
        provenance: dict | None = None,
    ) -> "GlobalScaler":
        """Media e std scalari su ``signals``: fold 1-8, 500 Hz, record interi.

        ``signals`` ha forma ``(n_record, 12, n_campioni)`` e puo' essere un
        memmap: viene letto **a blocchi di ``chunk`` record**, mai
        materializzato. A 500 Hz i fold 1-8 sono ~4,2 GB.

        ``record_idx`` seleziona il sottoinsieme di training. Indicizzare un
        memmap con un array copierebbe tutto in RAM, quindi la selezione avviene
        blocco per blocco.

        Due passate, non una: la prima calcola la media, la seconda la somma
        degli scarti **dalla media gia' nota**. Con una sola passata la varianza
        verrebbe da ``E[x^2] - E[x]^2``, una differenza fra numeri quasi uguali
        su 1e9 addendi. Il costo e' una seconda lettura del memmap; la posta e'
        la costante che normalizza tutti e quattro i bracci.

        Il fit usa i **record interi da 10 s**, non i crop: i crop dipendono da
        seed ed epoca, e lo scaler deve essere lo stesso per ogni cella
        dell'esperimento (voce 8).

        ``ddof=0``, come ``sklearn.preprocessing.StandardScaler``.
        """
        rows = (
            np.arange(len(signals))
            if record_idx is None
            else np.asarray(record_idx, dtype=np.int64)
        )
        if rows.size == 0:
            raise ValueError("nessun record su cui fittare lo scaler")
        if chunk < 1:
            raise ValueError(f"chunk deve essere >= 1, ricevuto {chunk}")

        def blocks():
            for begin in range(0, rows.size, chunk):
                sel = rows[begin : begin + chunk]
                yield np.asarray(signals[sel], dtype=np.float64)

        total = 0.0
        n_values = 0
        for block in blocks():
            total += float(block.sum())
            n_values += block.size
        mean = total / n_values

        sq = 0.0
        for block in blocks():
            sq += float(((block - mean) ** 2).sum())
        std = math.sqrt(sq / n_values)

        info = {
            "n_records": int(rows.size),
            "n_values": int(n_values),
            "record_fingerprint": _fingerprint(rows),
            "window": "full_record",
            "ddof": 0,
        }
        info.update(provenance or {})
        return cls(mean, std, info)

    @classmethod
    def fit_from_cache(
        cls,
        cache,
        record_idx: Sequence[int] | np.ndarray,
        *,
        folds: Sequence[int] = (1, 2, 3, 4, 5, 6, 7, 8),
        chunk: int = 256,
    ) -> "GlobalScaler":
        """``fit`` su una ``SignalCache``, con la provenienza compilata da sola.

        Richiede una cache a ``SOURCE_FS`` (voce 8): fittare a un rate diverso
        cambierebbe la costante e con essa il confronto.
        """
        fs = int(getattr(cache, "fs"))
        if fs != FIT_FS:
            raise ValueError(
                f"lo scaler va fittato a {FIT_FS} Hz (voce 8), non a {fs} Hz"
            )
        provenance = {
            "fs": fs,
            "notch": bool(getattr(cache, "notch", True)),
            "source": str(getattr(cache, "source", "hr")),
            "folds": [int(f) for f in folds],
            "cache_stem": str(getattr(cache, "stem", "")),
        }
        return cls.fit(
            cache.array, record_idx, chunk=chunk, provenance=provenance
        )

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def to_json(self, path) -> None:
        payload = {
            "version": _SCALER_FORMAT_VERSION,
            "mean": self.mean,
            "std": self.std,
            "numpy_version": np.__version__,
            **self.provenance,
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))

    @classmethod
    def from_json(cls, path) -> "GlobalScaler":
        info = json.loads(Path(path).read_text())
        version = info.pop("version", None)
        if version != _SCALER_FORMAT_VERSION:
            raise ValueError(
                f"scaler in formato {version}, atteso {_SCALER_FORMAT_VERSION}"
            )
        mean = info.pop("mean")
        std = info.pop("std")
        return cls(mean, std, info)


def _fingerprint(values: np.ndarray) -> str:
    """Hash corto dell'ordine degli indici usati, come in ``SignalCache``."""
    payload = ",".join(str(int(v)) for v in values)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Crop
# --------------------------------------------------------------------------- #


class CropSampler:
    """Estrae l'istante di partenza in **secondi**, non in campioni.

    Il draw e' funzione di ``(seed, epoch, record_id)`` e non di ``fs``: i
    bracci vedono le stesse finestre fisiche. Estrarre l'indice campione per
    rate aggiungerebbe varianza non dovuta alla frequenza, proprio nella
    quantita' che il bootstrap accoppiato confronta con la varianza-tra-seed.

    L'istante cade sempre sulla griglia a ``GRID_SECONDS``, l'unica su cui tutti
    e quattro i rate danno un indice campione intero.
    """

    def __init__(self, seed: int, window_seconds: float = WINDOW_SECONDS) -> None:
        self.seed = seed
        self.window_seconds = window_seconds
        self.n_positions = int(
            round((RECORD_SECONDS - window_seconds) / GRID_SECONDS)
        ) + 1

    def start_seconds(self, epoch: int, record_id: int) -> float:
        """Deterministico: stesso (epoch, record_id) -> stesso istante."""
        rng = np.random.default_rng((self.seed, epoch, record_id))
        return float(rng.integers(self.n_positions)) * GRID_SECONDS

    @staticmethod
    def to_index(start_seconds: float, fs: int) -> int:
        """Il controllo sulla griglia viene **prima** di quello sull'indice.

        Sono cose diverse: 0,05 s e' fuori dalla griglia a 0,1 s ma darebbe
        indice 5 a 100 Hz. La griglia e' l'invariante che serve, perche'
        garantisce un indice intero a tutti e quattro i rate insieme.

        Duplica ``ecgres.data.grid_index`` di proposito: ``data`` importa pandas
        e scipy, e ``model`` deve restare importabile senza. La duplicazione e'
        vincolata da ``test_to_index_agrees_with_data_grid_index``. Nel percorso
        di training non viene usata: ``WindowDataset`` passa i secondi a
        ``SignalCache.window`` e la conversione avviene una volta sola, in
        ``data``.
        """
        k = start_seconds / GRID_SECONDS
        if abs(k - round(k)) > 1e-9:
            raise ValueError(
                f"{start_seconds} s non e' sulla griglia a {GRID_SECONDS} s"
            )
        idx = start_seconds * fs
        if abs(idx - round(idx)) > 1e-9:
            raise ValueError(f"{start_seconds} s non da' un indice intero a fs={fs}")
        return int(round(idx))


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


class CacheLike(Protocol):
    """Interfaccia minima richiesta a ``ecgres.data.SignalCache``.

    Dichiarata come Protocol perche' i test girino su una cache finta, senza
    dipendere dai dati veri e senza importare ``data`` (e quindi pandas e
    scipy) dentro ``model``.

    ``window`` prende l'istante in **secondi**: e' la coordinata condivisa fra i
    bracci, ed e' ``data`` a convertirla in campioni.
    """

    fs: int

    def __len__(self) -> int: ...

    def window(
        self, record_idx: int, start_s: float, *, window_seconds: float = ...
    ) -> np.ndarray: ...


class WindowDataset(torch.utils.data.Dataset):
    """Finestre da una ``SignalCache``, etichette per **presenza**.

    ``train=True``: un crop random sulla griglia, via ``CropSampler``.
    ``train=False``: i 4 crop di ``INFERENCE_CROP_STARTS_S``, restituiti come
    elementi distinti; l'aggregazione (media delle predizioni) avviene a valle,
    per ``record_id``.
    """

    def __init__(
        self,
        cache: CacheLike,
        record_ids: Sequence[int],
        labels: np.ndarray,
        scaler: GlobalScaler,
        cfg: ModelConfig,
        train: bool,
        sampler: CropSampler | None = None,
    ) -> None:
        if cache.fs != cfg.fs:
            raise ValueError(f"cache a {cache.fs} Hz, config a {cfg.fs} Hz")
        if train and sampler is None:
            raise ValueError("in training serve un CropSampler")
        labels = np.asarray(labels)
        if labels.ndim != 2:
            raise ValueError(f"labels deve essere 2D, ricevuto {labels.shape}")
        # ``record_ids`` e ``labels`` vivono nello stesso spazio di indici della
        # cache (posizioni in ``meta``), non in quello dello split: cosi' il
        # medesimo array di etichette serve train, val e test senza rimappature,
        # che sono il punto in cui un leakage entra senza farsi notare.
        if labels.shape[0] != len(cache):
            raise ValueError(
                f"labels ha {labels.shape[0]} righe, la cache {len(cache)}: "
                "le etichette devono essere allineate ai record della cache"
            )
        if labels.shape[1] != cfg.n_classes:
            raise ValueError(
                f"labels ha {labels.shape[1]} colonne, la config {cfg.n_classes}"
            )
        self.cache = cache
        self.record_ids = list(record_ids)
        self.labels = labels
        self.scaler = scaler
        self.cfg = cfg
        self.train = train
        self.sampler = sampler
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Da chiamare a ogni epoca: i crop dipendono da ``epoch``."""
        self.epoch = epoch

    def __len__(self) -> int:
        n = len(self.record_ids)
        return n if self.train else n * len(INFERENCE_CROP_STARTS_S)

    def start_seconds(self, i: int) -> tuple[int, float]:
        """``i`` -> ``(record_id, start_s)``. In secondi, mai in campioni.

        In inferenza l'ordine e' ``record`` esterno e ``crop`` interno, lo
        stesso di ``SignalCache.inference_batch``: chi aggrega le 4 predizioni a
        valle puo' fare reshape ``(-1, 4)`` senza consultare i ``record_id``.
        """
        if self.train:
            record_id = self.record_ids[i]
            return record_id, self.sampler.start_seconds(self.epoch, record_id)
        n_crops = len(INFERENCE_CROP_STARTS_S)
        record_id = self.record_ids[i // n_crops]
        return record_id, INFERENCE_CROP_STARTS_S[i % n_crops]

    def __getitem__(self, i: int):
        record_id, start_s = self.start_seconds(i)
        x = self.cache.window(
            record_id, start_s, window_seconds=self.cfg.window_seconds
        )
        # ``transform`` con scalari Python non promuove: float32 in, float32 out.
        x = np.ascontiguousarray(self.scaler.transform(x), dtype=np.float32)
        y = np.ascontiguousarray(self.labels[record_id], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y), record_id
