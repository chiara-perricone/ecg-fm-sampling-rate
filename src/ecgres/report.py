"""Stampa e provenienza, senza alcuna dipendenza.

Vive da solo perche' lo usano anche gli script di sola analisi — ``stage0.py`` e
quelli di §8 — che devono poter girare su una macchina senza torch, senza GPU e
senza i 12,5 GB di cache. Se questi tre aiutanti stessero in ``pipeline``, il
solo importarli tirerebbe dentro ``ecgres.model`` e quindi torch.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

__all__ = ["section", "info", "git_sha"]


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}", flush=True)


def info(label: str, detail: str = "") -> None:
    print(f"  [    ] {label}{(': ' + detail) if detail else ''}", flush=True)


def git_sha(repo_root: str | Path) -> str | None:
    """Commit corrente, o ``None`` fuori da un repo. Non solleva mai.

    Serve a marcare ogni artefatto con il codice che lo ha prodotto: con 43 run
    scritti in momenti diversi, un file di cui non si sa da quale commit venga
    non e' verificabile.
    """
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
