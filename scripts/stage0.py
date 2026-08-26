"""Valuta la condizione bloccante di §3 sui tre run del blocco 0.

Legge i ``predictions-*.npz`` gia' scritti da ``evaluate.py``, quindi non serve
GPU e non si rifa' inferenza. Se anche un solo run manca, si ferma: §3 e' un
cancello, e un cancello valutato su due seed su tre non e' quel cancello.

    python scripts/stage0.py
    python scripts/stage0.py --n-boot 1000     # prova rapida, non il verdetto

Il verdetto e' in ``results/stage0.json``. Le due condizioni — 0,941 dentro il
CI, e stima puntuale entro 0,010 — sono riportate separatamente, perche' se la
riproduzione fallisse, sapere **quale** delle due e' caduta dice cose diverse.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ecgres.aggregate import (  # noqa: E402
    STAGE0_TARGET,
    load_predictions,
    seed_mean_bootstrap,
    stage0_verdict,
)
from ecgres.report import git_sha, info, section  # noqa: E402
from ecgres.runs import load_runs  # noqa: E402

RUNS_CSV = REPO_ROOT / "configs" / "runs.csv"
BLOCKING_BLOCK = 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-csv", type=Path, default=RUNS_CSV)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "runs")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "stage0.json")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    block0 = [r for r in load_runs(args.runs_csv) if r.block == BLOCKING_BLOCK]
    if not block0:
        print("nessun run nel blocco 0", file=sys.stderr)
        return 2

    section("Condizione bloccante §3")
    info("run del blocco 0", ", ".join(r.run_id for r in block0))
    info("metrica", "macro_all su fold 10, media dei seed")
    info("target", f"{STAGE0_TARGET} +/- 0.010, CI 95% su {args.n_boot} ricampionamenti")

    paths, missing = [], []
    for run in block0:
        path = args.out_dir / run.run_id / f"predictions-{run.selection}.npz"
        if path.exists():
            paths.append(path)
        else:
            missing.append(run.run_id)
    if missing:
        print(f"\npredizioni assenti per: {', '.join(missing)}", file=sys.stderr)
        print("§3 va valutata su tutti e tre i seed. Eseguire prima "
              "scripts/train.py e scripts/evaluate.py per ciascuno.", file=sys.stderr)
        return 2

    runs, _ = load_predictions(paths)
    info("record di test", str(len(runs[0].y_true)))

    section("Bootstrap")
    info("in corso", f"{args.n_boot} ricampionamenti x {len(runs)} seed, "
                     "qualche minuto")
    estimate = seed_mean_bootstrap(runs, n_boot=args.n_boot, seed=args.seed)
    verdict = stage0_verdict(estimate)

    section("Esito")
    for run_id, value in verdict["per_run"].items():
        info(run_id, f"{value:.4f}")
    info("media dei seed", f"{verdict['point']:.4f}")
    info("CI 95% sui record",
         f"[{verdict['ci'][0]:.4f}, {verdict['ci'][1]:.4f}]")
    info("scarto fra seed",
         f"escursione {verdict['seed_spread']:.4f}, "
         f"dev.st. {verdict['seed_std']:.4f}  (fuori dal CI, §10 voce 15)")
    info("differenza dal target", f"{verdict['difference']:+.4f}")
    info("0.941 dentro il CI", "si" if verdict["ci_contains_target"] else "NO")
    info("stima entro 0.010", "si" if verdict["point_within_tolerance"] else "NO")

    verdict["git_sha"] = git_sha(REPO_ROOT)
    verdict["run_ids"] = [r.run_id for r in block0]
    verdict["bootstrap_seed"] = args.seed
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(verdict, indent=2, sort_keys=True, default=str))
    info("scritto", str(args.out.relative_to(REPO_ROOT)))

    print()
    if verdict["accepted"]:
        print("  RIPRODUZIONE ACCETTATA — si procede ai blocchi 1-3.")
        return 0
    print("  RIPRODUZIONE NON ACCETTATA.")
    print("  §3: il lavoro si ferma e il fallimento va scritto.")
    print("  Se il fallimento e' stretto, restano due soli candidati: la voce")
    print("  §10 numero 11 (stride dei crop di training) e la 18 (i fold non")
    print("  sono dichiarati dal paper). Le voci 4 e 5 sono state ritirate")
    print("  dalle 16 e 17: versione del dataset e provenienza del segnale")
    print("  coincidono con quelle del riferimento.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
