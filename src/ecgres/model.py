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

from dataclasses import dataclass
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
    "GlobalScaler",
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


def build_model(cfg: ModelConfig, seed: int) -> S4Backbone:
    """Costruisce il modello con RNG seedato.

    A parita' di ``seed`` e di ``arm``, i pesi iniziali sono **identici ai
    quattro rate**: nessuna shape dipende da ``fs``. Verificato in
    ``test_init_identical_across_rates``.
    """
    torch.manual_seed(seed)
    return S4Backbone(cfg)


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

    def __init__(self, mean: float, std: float) -> None:
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
            raise ValueError(f"scaler non valido: mean={mean}, std={std}")
        self.mean = float(mean)
        self.std = float(std)

    @classmethod
    def fit(cls, signals: np.ndarray) -> "GlobalScaler":
        """``signals``: ``(n_records, 12, n_samples)``, fold 1-8 a 500 Hz."""
        raise NotImplementedError  # TODO: media/std su tutti gli assi, float64

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def to_json(self, path) -> None:
        raise NotImplementedError  # TODO: mean, std, fs del fit, fold, fingerprint

    @classmethod
    def from_json(cls, path) -> "GlobalScaler":
        raise NotImplementedError


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
    dipendere dai dati veri.
    """

    fs: int

    def __len__(self) -> int: ...

    def window(self, record_idx: int, start: int, length: int) -> np.ndarray: ...


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

    def __getitem__(self, i: int):
        raise NotImplementedError
        # TODO
        #  1. mappa i -> (record_id, start_seconds)
        #  2. start = CropSampler.to_index(start_seconds, cfg.fs)
        #  3. x = cache.window(record_idx, start, cfg.n_samples)
        #  4. x = scaler.transform(x).astype(np.float32)
        #  5. ritorna (torch.from_numpy(x), torch.from_numpy(y), record_id)
