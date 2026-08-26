"""Confronti di §8: rate contro rate, braccio contro braccio, notch contro notch.

Legge i ``predictions-*.npz`` scritti da ``evaluate.py``. Nessuna GPU, nessuna
cache, nessun torch: e' analisi di file che esistono gia'.

    python scripts/analyse.py                       # §8.1, il confronto primario
    python scripts/analyse.py --analysis arms       # §8.3.1, Arm A contro Arm B
    python scripts/analyse.py --analysis notch      # §8.3.4, effetto del notch
    python scripts/analyse.py --n-boot 500          # prova rapida, non il risultato

Si rifiuta di procedere se manca anche un solo run del confronto richiesto. Una
famiglia di sei confronti valutata su tre rate su quattro non e' la famiglia che
§8.1 dichiara, e Holm su un numero di confronti diverso da quello dichiarato non
e' la correzione dichiarata.

Entrambe le macro AUROC vengono riportate sempre (§7). L'endpoint primario del
confronto fra rate e' ``macro_clean`` (§10 voce 1); ``macro_all`` compare
accanto, non al suo posto.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ecgres.aggregate import (  # noqa: E402
    compare_groups,
    interpret_delta,
    load_predictions,
)
from ecgres.report import git_sha, info, section  # noqa: E402
from ecgres.runs import load_runs  # noqa: E402

RUNS_CSV = REPO_ROOT / "configs" / "runs.csv"
ENDPOINTS = ("macro_clean", "macro_all")

#: Coppia del controllo negativo interno (§8.1). Nessun effetto e' atteso.
NEGATIVE_CONTROL = ("240", "250")


def _grouping(analysis: str, runs) -> list[tuple[str, dict[str, list]]]:
    """``[(titolo, {nome_gruppo: [RunSpec]})]`` per il confronto richiesto."""
    by_block: dict[int, list] = {}
    for run in runs:
        by_block.setdefault(run.block, []).append(run)

    if analysis == "rates":
        groups: dict[str, list] = {}
        for run in sorted(by_block.get(1, []), key=lambda r: (r.fs, r.seed)):
            groups.setdefault(str(run.fs), []).append(run)
        return [("§8.1 — rate contro rate, Arm A", groups)]

    if analysis == "arms":
        out = []
        for fs in (240, 250, 500):
            groups = {
                "A": [r for r in by_block.get(1, []) if r.fs == fs],
                "B": [r for r in by_block.get(2, []) if r.fs == fs],
            }
            out.append((f"§8.3.1 — Arm A contro Arm B a {fs} Hz", groups))
        return out

    if analysis == "notch":
        groups = {
            "con notch": [r for r in by_block.get(1, []) if r.fs == 100],
            "senza notch": list(by_block.get(3, [])),
        }
        return [("§8.3.4 — effetto del notch a 100 Hz", groups)]

    if analysis == "source":
        # Entrambi senza notch: records100 non puo' portarlo, la fondamentale
        # di rete sta sul suo Nyquist. Differiscono solo per la provenienza.
        groups = {
            "records500 ricampionato": list(by_block.get(3, [])),
            "records100": list(by_block.get(4, [])),
        }
        return [("§8.3.2 — provenienza del segnale a 100 Hz", groups)]

    raise ValueError(f"analisi non prevista: {analysis}")


def _paths(specs, out_dir: Path) -> tuple[list[Path], list[str]]:
    paths, missing = [], []
    for spec in specs:
        path = out_dir / spec.run_id / f"predictions-{spec.selection}.npz"
        if path.exists():
            paths.append(path)
        else:
            missing.append(spec.run_id)
    return paths, missing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis", choices=("rates", "arms", "notch", "source"),
                    default="rates")
    ap.add_argument("--runs-csv", type=Path, default=RUNS_CSV)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "runs")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_path = args.out or REPO_ROOT / "results" / f"analysis_{args.analysis}.json"
    families = _grouping(args.analysis, load_runs(args.runs_csv))

    report: dict = {
        "analysis": args.analysis,
        "n_boot": args.n_boot,
        "bootstrap_seed": args.seed,
        "git_sha": git_sha(REPO_ROOT),
        "primary_endpoint": "macro_clean",
        "families": [],
    }
    rows = []

    for title, groups in families:
        section(title)
        empty = [name for name, specs in groups.items() if not specs]
        if empty:
            print(f"gruppi vuoti nella matrice: {empty}", file=sys.stderr)
            return 2

        loaded, missing_all = {}, {}
        for name, specs in groups.items():
            paths, missing = _paths(specs, args.out_dir)
            if missing:
                missing_all[name] = missing
                continue
            loaded[name] = paths
        if missing_all:
            print(f"\npredizioni assenti: {missing_all}", file=sys.stderr)
            print("Il confronto richiede tutti i run che dichiara. Eseguire prima "
                  "scripts/train.py e scripts/evaluate.py per quelli mancanti.",
                  file=sys.stderr)
            return 2

        # Un solo caricamento per famiglia: ``load_predictions`` verifica insieme
        # che tutti i run coprano gli stessi record con le stesse etichette.
        flat = [p for name in loaded for p in loaded[name]]
        all_runs, clean_columns = load_predictions(flat)
        by_id = {r.run_id: r for r in all_runs}
        grouped = {
            name: [by_id[p.parent.name] for p in paths] for name, paths in loaded.items()
        }
        y_true = all_runs[0].y_true
        info("gruppi", ", ".join(f"{k} ({len(v)} seed)" for k, v in grouped.items()))
        info("record di test", str(len(y_true)))

        family = {"title": title, "endpoints": {}}
        for endpoint in ENDPOINTS:
            columns = clean_columns if endpoint == "macro_clean" else None
            comparison = compare_groups(
                y_true, grouped, columns, n_boot=args.n_boot, seed=args.seed
            )
            delta = interpret_delta(comparison)

            print()
            info(f"[{endpoint}]", "ensemble per gruppo, dispersione fra seed accanto")
            for group in comparison.groups:
                info(f"  {group.name}",
                     f"{group.point:.4f} [{group.lo:.4f}, {group.hi:.4f}]  "
                     f"seed: escursione {group.seed_spread:.4f}, "
                     f"dev.st. {group.seed_std:.4f}, "
                     f"contrasto nullo {group.null_median:.4f}")
            for pair in comparison.pairs:
                mark = ""
                if (pair.a, pair.b) == NEGATIVE_CONTROL:
                    mark = "   <- controllo negativo, nessun effetto atteso"
                info(f"  {pair.a} - {pair.b}",
                     f"{pair.diff:+.4f} [{pair.lo:+.4f}, {pair.hi:+.4f}] "
                     f"p_holm={pair.p_holm:.4f} "
                     f"sim [{pair.lo_simultaneous:+.4f}, {pair.hi_simultaneous:+.4f}]"
                     f"{mark}")
                rows.append({
                    "analysis": args.analysis, "family": title, "endpoint": endpoint,
                    "a": pair.a, "b": pair.b, "diff": f"{pair.diff:.6f}",
                    "lo": f"{pair.lo:.6f}", "hi": f"{pair.hi:.6f}",
                    "p_raw": f"{pair.p_raw:.6f}", "p_holm": f"{pair.p_holm:.6f}",
                    "lo_simultaneous": f"{pair.lo_simultaneous:.6f}",
                    "hi_simultaneous": f"{pair.hi_simultaneous:.6f}",
                })
            info("  Delta (§8.2)",
                 f"{delta['delta']:+.4f} fra {delta['delta_pair'][0]} e "
                 f"{delta['delta_pair'][1]}, metro {delta['comparator_value']:.4f} "
                 f"({delta['comparator']}) — {delta['verdict']}")

            family["endpoints"][endpoint] = {
                "groups": [
                    {"name": g.name, "point": g.point, "ci": [g.lo, g.hi],
                     "per_run": g.per_run, "seed_spread": g.seed_spread,
                     "seed_std": g.seed_std}
                    for g in comparison.groups
                ],
                "pairs": [
                    {**vars(p), "crosses_zero": p.crosses_zero,
                     "significant_holm": p.significant}
                    for p in comparison.pairs
                ],
                "delta": delta,
            }
        report["families"].append(family)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    csv_path = out_path.with_suffix(".csv")
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    def rel(path: Path) -> str:
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return str(path)  # --out fuori dal repo, tipico in una prova

    section("Scritti")
    info("json", rel(out_path))
    info("csv", rel(csv_path))
    print("\n  §9: un risultato nullo va riportato con lo stesso rilievo di uno positivo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
