"""La matrice sperimentale: quali run esistono e quali duplicano una configurazione.

Questo modulo e' la forma eseguibile della tabella di PROTOCOL.md §6.7. Vive in
``src`` e non in ``scripts`` perche' il training loop la legge e i test la
verificano; ``scripts/make_runs.py`` si limita a materializzarla in
``configs/runs.csv``, che e' l'artefatto versionato.

Perche' un artefatto versionato e non solo argomenti da riga di comando: con un
protocollo pre-registrato il repo deve dichiarare **quali run intendeva fare**,
non soltanto quali ha fatto. La matrice e' committata prima di eseguirla.

Le duplicazioni sono volute
---------------------------
Tre run del blocco 3 hanno configurazione identica a tre del blocco 0 — stesso
rate, stesso braccio, stessa assenza di notch, stessi seed — e differiscono solo
per la metrica su cui si seleziona il checkpoint (§10 voce 13). Vengono allenati
**separatamente**: a parita' di seed devono coincidere, quindi le tre coppie
sono l'unica verifica end-to-end che il pipeline sia riproducibile, che e'
l'assunzione su cui poggia l'uso di spot instance in §6.7. Vedi §10 voce 14.

Il blocco 4 **non** e' un duplicato del 3, per quanto gli somigli: stesso rate,
stesso braccio, stessa assenza di notch, ma segnale da ``records100`` invece che
da ``records500`` ricampionato. E' per questo che ``source`` entra in
``config_key``.

Nessuna dipendenza da pandas, scipy o torch: la matrice deve essere ispezionabile
ovunque, compreso un pod GPU appena creato.
"""

from __future__ import annotations

import csv
from dataclasses import astuple, dataclass, fields
from pathlib import Path
from typing import Sequence

__all__ = [
    "RATES",
    "ARM_B_RATES",
    "SEEDS",
    "BLOCKING_SEEDS",
    "RunSpec",
    "enumerate_runs",
    "load_runs",
    "write_runs",
    "duplicate_configs",
]

#: Deve coincidere con ``ecgres.data.RATES``. Duplicata per tenere questo modulo
#: privo di dipendenze; il vincolo e' verificato in ``test_runs.py``.
RATES: tuple[int, ...] = (100, 240, 250, 500)

#: Arm B a 100 Hz coincide con Arm A per costruzione (§6.3), quindi non si corre.
ARM_B_RATES: tuple[int, ...] = (240, 250, 500)

SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

#: §6.7 assegna tre seed alla condizione bloccante, non cinque.
BLOCKING_SEEDS: tuple[int, ...] = (0, 1, 2)


@dataclass(frozen=True)
class RunSpec:
    """Una riga della matrice.

    ``selection`` e' la metrica di fold 9 su cui si tiene il checkpoint
    (§10 voce 13), non la metrica riportata: entrambe le macro AUROC sono
    calcolate e registrate a ogni epoca di ogni run.
    """

    run_id: str
    block: int
    fs: int
    arm: str
    notch: bool
    source: str
    seed: int
    selection: str

    @property
    def config_key(self) -> tuple[int, str, bool, str, int]:
        """Cio' che determina i pesi. ``selection`` non ne fa parte.

        ``source`` **si**: i blocchi 3 e 4 girano entrambi a 100 Hz, Arm A,
        senza notch, e differiscono solo per la provenienza del segnale. Senza
        questo campo verrebbero scambiati per configurazioni duplicate.
        """
        return (self.fs, self.arm, self.notch, self.source, self.seed)


FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in fields(RunSpec))


def enumerate_runs() -> tuple[RunSpec, ...]:
    """Le 43 righe di §6.7, nell'ordine in cui vanno eseguite.

    Il blocco 0 viene per primo perche' e' una condizione bloccante: se non
    passa, i blocchi 1-3 non si eseguono affatto (§3).
    """
    runs: list[RunSpec] = []

    def add(block: int, fs: int, arm: str, notch: bool, seed: int, selection: str,
            source: str = "hr") -> None:
        runs.append(
            RunSpec(
                run_id=f"b{block}-fs{fs}-{arm}-n{int(notch)}-s{seed}",
                block=block,
                fs=fs,
                arm=arm,
                notch=notch,
                source=source,
                seed=seed,
                selection=selection,
            )
        )

    # Blocco 0 — condizione bloccante (§3). Senza notch, come il riferimento.
    for seed in BLOCKING_SEEDS:
        add(0, 100, "A", False, seed, "macro_all")

    # Blocco 1 — Arm A, i quattro rate.
    for fs in RATES:
        for seed in SEEDS:
            add(1, fs, "A", True, seed, "macro_clean")

    # Blocco 2 — Arm B, senza 100 Hz.
    for fs in ARM_B_RATES:
        for seed in SEEDS:
            add(2, fs, "B", True, seed, "macro_clean")

    # Blocco 3 — 100 Hz senza notch (§8.3.4). Duplica il blocco 0 sui primi
    # tre seed: e' voluto, vedi il docstring del modulo.
    for seed in SEEDS:
        add(3, 100, "A", False, seed, "macro_clean")

    # Blocco 4 — 100 Hz da ``records100`` (§8.3.2, §10 voce 23). Si confronta
    # col blocco 3, da cui differisce solo per la provenienza del segnale.
    # Senza notch per forza: a 100 Hz la fondamentale di rete e' sul Nyquist.
    for seed in SEEDS:
        add(4, 100, "A", False, seed, "macro_clean", source="lr")

    return tuple(runs)


def duplicate_configs(
    runs: Sequence[RunSpec],
) -> dict[tuple[int, str, bool, int], tuple[RunSpec, ...]]:
    """Configurazioni presenti in piu' di un run.

    Sono le coppie che devono produrre gli stessi pesi. Il training loop non le
    tratta in modo speciale — le allena tutte — ma l'analisi le confronta.
    """
    groups: dict[tuple[int, str, bool, int], list[RunSpec]] = {}
    for run in runs:
        groups.setdefault(run.config_key, []).append(run)
    return {k: tuple(v) for k, v in groups.items() if len(v) > 1}


def write_runs(path: str | Path, runs: Sequence[RunSpec] | None = None) -> Path:
    """Scrive il CSV con terminatore ``\\n`` esplicito.

    Il default di ``csv.writer`` e' ``\\r\\n``, che su Windows produrrebbe un file
    a terminazione mista dopo il primo passaggio di git.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = enumerate_runs() if runs is None else tuple(runs)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(FIELD_NAMES)
        for run in rows:
            writer.writerow(
                [int(v) if isinstance(v, bool) else v for v in astuple(run)]
            )
    return path


def load_runs(path: str | Path) -> tuple[RunSpec, ...]:
    """Rilegge il CSV con i tipi giusti, e rifiuta un'intestazione diversa."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELD_NAMES:
            raise ValueError(
                f"intestazione inattesa in {path}: {reader.fieldnames}"
            )
        return tuple(
            RunSpec(
                run_id=row["run_id"],
                block=int(row["block"]),
                fs=int(row["fs"]),
                arm=row["arm"],
                notch=bool(int(row["notch"])),
                source=row["source"],
                seed=int(row["seed"]),
                selection=row["selection"],
            )
            for row in reader
        )
