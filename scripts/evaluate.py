"""Valuta **un** run su fold 10 e salva predizioni e metriche per etichetta.

Un run per volta, di proposito. I 43 run girano in momenti diversi su una
macchina noleggiata, a volte interrotti, e l'aggregazione fra seed (§3) o fra
rate (§8.1) e' un passo separato che legge i file scritti qui. Metterla dentro
significherebbe riscriverla due volte e non poterla eseguire finche' non e'
finito l'ultimo run.

Il checkpoint caricato e' quello del criterio del run (§10 voce 13). Con
``--checkpoint`` si puo' chiedere l'altro, che ``train.py`` ha salvato apposta:
serve a rispondere a "la scelta della voce 13 avrebbe cambiato il risultato?"
senza riallenare nulla.

    python scripts/evaluate.py --run-id b0-fs100-A-n0-s0
    python scripts/evaluate.py --run-id b0-fs100-A-n0-s0 --checkpoint macro_clean
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from ecgres import data as D  # noqa: E402
from ecgres.metrics import macro_auroc, per_label_scores  # noqa: E402
from ecgres.model import ModelConfig, WindowDataset, build_model  # noqa: E402
from ecgres.pipeline import (  # noqa: E402
    clean_columns,
    configure_numerics,
    git_sha,
    info,
    resolve_scaler,
    section,
)
from ecgres.runs import load_runs  # noqa: E402
from ecgres.train import SELECTION_METRICS, TrainConfig, evaluate  # noqa: E402

FROZEN_LABELS = REPO_ROOT / "results" / "frozen_label_set.json"
RUNS_CSV = REPO_ROOT / "configs" / "runs.csv"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--runs-csv", type=Path, default=RUNS_CSV)
    ap.add_argument("--root", type=Path, default=REPO_ROOT / "data" / "ptb-xl")
    ap.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data" / "cache")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "runs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--checkpoint", choices=SELECTION_METRICS, default=None,
                    help="criterio del checkpoint; default: quello del run")
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()

    runs = {r.run_id: r for r in load_runs(args.runs_csv)}
    if args.run_id not in runs:
        print(f"run-id sconosciuto: {args.run_id}", file=sys.stderr)
        return 2
    run = runs[args.run_id]
    criterion = args.checkpoint or run.selection
    out_dir = args.out_dir / run.run_id

    section(f"Valutazione {run.run_id} su fold 10")
    info("configurazione", f"{run.fs} Hz, Arm {run.arm}, notch={run.notch}, seed={run.seed}")
    info("checkpoint", f"best-{criterion}.pt"
                       f"{'' if criterion == run.selection else '  (controfattuale)'}")
    info("device", args.device)
    configure_numerics(tf32=args.tf32, deterministic=args.deterministic)

    ckpt_path = out_dir / f"best-{criterion}.pt"
    if not ckpt_path.exists():
        print(f"checkpoint assente: {ckpt_path}\n"
              f"allenare prima con: python scripts/train.py --run-id {run.run_id}",
              file=sys.stderr)
        return 2

    section("Dati")
    meta = D.load_metadata(args.root)
    splits = D.get_splits(meta)
    labels, label_names = D.load_label_matrix(args.root, meta)
    columns = clean_columns(label_names, FROZEN_LABELS)
    cache = D.SignalCache(args.root, args.cache_dir, meta, fs=run.fs, notch=run.notch)
    _ = cache.array
    scaler, scaler_path = resolve_scaler(
        run, args.root, args.cache_dir, meta, splits.train
    )
    info("test", f"{splits.test.size} record (fold 10)")
    info("scaler", f"mean {scaler.mean:.6g}, std {scaler.std:.6g} — {scaler_path.name}")

    model_cfg = ModelConfig(fs=run.fs, arm=run.arm, n_classes=labels.shape[1])
    test_ds = WindowDataset(
        cache=cache, record_ids=splits.test, labels=labels, scaler=scaler,
        cfg=model_cfg, train=False,
    )

    state = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    if state["run_id"] != run.run_id:
        print(f"{ckpt_path} appartiene a {state['run_id']}", file=sys.stderr)
        return 2
    model = build_model(model_cfg, run.seed).to(args.device)
    model.load_state_dict(state["model"])
    info("epoca del checkpoint", str(state["epoch"]))

    section("Inferenza")
    y_true, y_prob = evaluate(
        model, test_ds,
        TrainConfig(batch_size=args.batch_size, num_workers=args.num_workers),
        torch.device(args.device),
    )

    macro = {
        "macro_all": macro_auroc(y_true, y_prob),
        "macro_clean": macro_auroc(y_true[:, columns], y_prob[:, columns]),
    }
    scores = per_label_scores(y_true, y_prob)
    in_clean = np.zeros(len(label_names), dtype=bool)
    in_clean[columns] = True

    # Predizioni per record: sono l'input di ogni analisi a valle — bootstrap di
    # §3, confronti accoppiati di §8.1 — che deve poter girare senza rifare
    # l'inferenza e senza la GPU.
    np.savez_compressed(
        out_dir / f"predictions-{criterion}.npz",
        record_idx=np.asarray(splits.test, dtype=np.int64),
        ecg_id=np.asarray(meta.index[splits.test], dtype=np.int64),
        y_true=y_true.astype(np.uint8),
        y_prob=y_prob.astype(np.float32),
        clean_columns=np.asarray(columns, dtype=np.int64),
    )

    import csv

    metrics_path = out_dir / f"test_per_label-{criterion}.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["label", "in_clean_set", "n_positive", "auroc", "auprc"])
        for j, name in enumerate(label_names):
            writer.writerow([
                name, int(in_clean[j]), int(scores["n_positive"][j]),
                "" if np.isnan(scores["auroc"][j]) else f"{scores['auroc'][j]:.6f}",
                "" if np.isnan(scores["auprc"][j]) else f"{scores['auprc'][j]:.6f}",
            ])

    summary = {
        "run_id": run.run_id,
        "block": run.block,
        "fs": run.fs,
        "arm": run.arm,
        "notch": run.notch,
        "seed": run.seed,
        "checkpoint_criterion": criterion,
        "is_counterfactual": criterion != run.selection,
        "checkpoint_epoch": int(state["epoch"]),
        "fold": 10,
        "n_records": int(splits.test.size),
        "n_labels_defined": int(np.isfinite(scores["auroc"]).sum()),
        "n_labels_defined_clean": int(np.isfinite(scores["auroc"][columns]).sum()),
        "git_sha": git_sha(REPO_ROOT),
        "scaler": {"path": scaler_path.name, "mean": scaler.mean, "std": scaler.std},
        **macro,
    }
    summary_path = out_dir / f"test_summary-{criterion}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))

    section("Esito")
    info("macro_all", f"{macro['macro_all']:.4f} "
                      f"su {summary['n_labels_defined']}/{len(label_names)} etichette")
    info("macro_clean", f"{macro['macro_clean']:.4f} "
                        f"su {summary['n_labels_defined_clean']}/{len(columns)} etichette")
    if summary["n_labels_defined"] < len(label_names):
        info("nota", "alcune etichette non hanno entrambe le classi in fold 10 "
                     "e restano indefinite (§7)")
    info("scritti", f"{metrics_path.name}, {summary_path.name}, "
                    f"predictions-{criterion}.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
