"""Materializza la matrice di PROTOCOL.md §6.7 in ``configs/runs.csv``.

Il CSV e' un artefatto versionato: va rigenerato con questo script e
ricommittato, mai modificato a mano. ``tests/test_runs.py`` verifica che il file
committato coincida con cio' che ``ecgres.runs.enumerate_runs`` produce, quindi
una modifica manuale viene rifiutata dalla suite.

    python scripts/make_runs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecgres.runs import duplicate_configs, enumerate_runs, write_runs  # noqa: E402


def main() -> None:
    runs = enumerate_runs()
    path = write_runs(ROOT / "configs" / "runs.csv", runs)
    duplicates = duplicate_configs(runs)

    print(f"scritto {path.relative_to(ROOT)}")
    print(f"  {len(runs)} run")
    for block in sorted({r.block for r in runs}):
        n = sum(1 for r in runs if r.block == block)
        print(f"  blocco {block}: {n} run")
    if duplicates:
        print(f"  {len(duplicates)} configurazioni duplicate (§10 voce 14):")
        for key, rows in sorted(duplicates.items()):
            fs, arm, notch, source, seed = key
            label = f"fs{fs} {arm} notch{int(notch)} {source} seed{seed}"
            print(f"    {label}: {', '.join(r.run_id for r in rows)}")


if __name__ == "__main__":
    main()
