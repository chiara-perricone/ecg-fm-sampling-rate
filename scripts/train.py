"""Esegue un run della matrice di PROTOCOL.md §6.7.

E' l'unico file che mette insieme i dati veri e il ciclo di training: legge la
riga di ``configs/runs.csv``, apre la cache giusta, procura lo scaler, costruisce
i due dataset e chiama ``ecgres.train.run_training``. La meccanica non e' qui —
sta in ``src/ecgres/train.py``, dove i test la raggiungono senza PTB-XL.

Uso tipico:

    python scripts/train.py --run-id b0-fs100-A-n0-s0            # condizione bloccante
    python scripts/train.py --run-id b1-fs500-A-n1-s3 --device cuda
    python scripts/train.py --run-id b0-fs100-A-n0-s0 --epochs 1 --max-records 256

Riprende da solo: se ``last.pt`` esiste nella cartella del run, riparte
dall'epoca successiva. E' la proprieta' che rende utilizzabili le istanze
interrompibili di §6.7, ed e' verificata in ``tests/test_train.py``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from ecgres import data as D  # noqa: E402
from ecgres.model import FIT_FS, CropSampler, GlobalScaler, ModelConfig, WindowDataset  # noqa: E402
from ecgres.runs import RunSpec, load_runs  # noqa: E402
from ecgres.train import TrainConfig, run_training  # noqa: E402

FROZEN_LABELS = REPO_ROOT / "results" / "frozen_label_set.json"
RUNS_CSV = REPO_ROOT / "configs" / "runs.csv"

#: Blocco della condizione bloccante (§3). Vedi ``resolve_scaler``.
BLOCKING_BLOCK = 0


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}", flush=True)


def info(label: str, detail: str = "") -> None:
    print(f"  [    ] {label}{(': ' + detail) if detail else ''}", flush=True)


# --------------------------------------------------------------------------
# Etichette
# --------------------------------------------------------------------------

def clean_columns(label_names: list[str]) -> list[int]:
    """Posizioni delle 49 etichette congelate dentro la matrice a 71 colonne.

    Il set e' letto, non ricalcolato: e' stato congelato prima di ogni run
    (§7) e ricalcolarlo qui lo renderebbe funzione dei dati presenti sul disco
    di turno. Si verifica pero' che sia coerente con quanto dichiara di essere.
    """
    frozen = json.loads(FROZEN_LABELS.read_text(encoding="utf-8"))
    names = list(frozen["labels_clean"])
    if len(names) != frozen["n_labels_clean"]:
        raise ValueError(
            f"{FROZEN_LABELS.name} dichiara {frozen['n_labels_clean']} etichette "
            f"ma ne elenca {len(names)}"
        )
    if len(label_names) != frozen["n_labels_all"]:
        raise ValueError(
            f"la matrice ha {len(label_names)} colonne, il set congelato ne "
            f"presuppone {frozen['n_labels_all']}"
        )
    position = {name: i for i, name in enumerate(label_names)}
    missing = [n for n in names if n not in position]
    if missing:
        raise ValueError(f"etichette congelate assenti da scp_statements.csv: {missing}")
    return [position[n] for n in names]


# --------------------------------------------------------------------------
# Scaler
# --------------------------------------------------------------------------

def resolve_scaler(
    run: RunSpec, root: Path, cache_dir: Path, meta, train_idx: np.ndarray
) -> tuple[GlobalScaler, Path]:
    """Lo scaler del run, fittato una volta sola e riusato (§10 voci 7 e 8).

    Due scaler, non uno, ed e' il protocollo a volerlo. La voce 8 dichiara
    esplicitamente che *"Stage 0 uses the reference pipeline's own scaler"*:

    * **blocco 0** — scaler fittato sulla cache del run stesso, cioe' 100 Hz
      senza notch, come fa la pipeline di riferimento. Deve riprodurre 0,941, e
      normalizzare con costanti che il riferimento non ha usato sarebbe un
      ulteriore candidato a spiegare un fallimento stretto;
    * **blocchi 1-3** — un unico scaler fittato a 500 Hz **con notch**, sui
      fold 1-8, applicato invariato a tutti i bracci del confronto. E' il
      500 Hz da cui ogni braccio del confronto deriva.

    Il file viene riusato se esiste, dopo aver verificato che la provenienza sia
    quella attesa: uno scaler giusto per caso sarebbe peggio di uno sbagliato.
    """
    if run.block == BLOCKING_BLOCK:
        fit_fs, fit_notch = run.fs, run.notch
    else:
        fit_fs, fit_notch = FIT_FS, True

    path = cache_dir / f"scaler_fs{fit_fs}_notch{int(fit_notch)}.json"
    expected = {"fs": fit_fs, "notch": fit_notch, "source": "hr", "window": "full_record"}

    if path.exists():
        scaler = GlobalScaler.from_json(path)
        wrong = {k: (scaler.provenance.get(k), v) for k, v in expected.items()
                 if scaler.provenance.get(k) != v}
        if wrong:
            raise ValueError(f"{path.name} ha provenienza inattesa: {wrong}")
        return scaler, path

    cache = D.SignalCache(root, cache_dir, meta, fs=fit_fs, notch=fit_notch)
    info("fit dello scaler", f"{fit_fs} Hz, notch={fit_notch}, {train_idx.size} record")
    provenance = {
        "fs": fit_fs,
        "notch": fit_notch,
        "source": "hr",
        "folds": list(D.SPLIT_FOLDS["train"]),
        "cache_stem": cache.stem,
        "fitted_for": "stage0" if run.block == BLOCKING_BLOCK else "comparison",
    }
    scaler = GlobalScaler.fit(cache.array, train_idx, provenance=provenance)
    scaler.to_json(path)
    return scaler, path


# --------------------------------------------------------------------------
# Impostazioni numeriche
# --------------------------------------------------------------------------

def configure_numerics(tf32: bool, deterministic: bool) -> None:
    """TF32 e determinismo, entrambi espliciti e registrati nel manifest.

    TF32 ha 10 bit di mantissa: lasciarlo acceso significa che "fp32" di §6.7
    non e' vero. Va spento per default e acceso solo di proposito, ad esempio
    per misurare quanto costa spegnerlo.
    """
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        # Solleva invece di rallentare in silenzio, se un'operazione non ha
        # implementazione deterministica: e' un fallimento leggibile e immediato.
        torch.use_deterministic_algorithms(True, warn_only=False)


def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        )
    except Exception:
        return None
    return out.stdout.strip()


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--runs-csv", type=Path, default=RUNS_CSV)
    ap.add_argument("--root", type=Path, default=REPO_ROOT / "data" / "ptb-xl")
    ap.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data" / "cache")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "runs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=None, help="override di §6.5, per prove")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--tf32", action="store_true",
                    help="riattiva TF32; per default e' spento perche' §6.7 dice fp32")
    ap.add_argument("--deterministic", action="store_true",
                    help="forza gli algoritmi deterministici; solleva se non esistono")
    ap.add_argument("--max-records", type=int, default=None,
                    help="usa solo i primi N record di train e val; solo per prove")
    args = ap.parse_args()

    runs = {r.run_id: r for r in load_runs(args.runs_csv)}
    if args.run_id not in runs:
        print(f"run-id sconosciuto: {args.run_id}", file=sys.stderr)
        print(f"disponibili: {', '.join(sorted(runs)[:5])} ... ({len(runs)} in tutto)",
              file=sys.stderr)
        return 2
    run = runs[args.run_id]

    smoke = args.epochs is not None or args.max_records is not None
    train_cfg = TrainConfig(
        **{
            k: v
            for k, v in {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
            }.items()
            if v is not None
        }
    )

    section(f"Run {run.run_id}")
    info("blocco", str(run.block))
    info("configurazione", f"{run.fs} Hz, Arm {run.arm}, notch={run.notch}, seed={run.seed}")
    info("selezione del checkpoint", f"{run.selection} su fold 9")
    info("device", args.device)
    if smoke:
        info("ATTENZIONE", "run ridotto: non e' un run del protocollo")

    configure_numerics(tf32=args.tf32, deterministic=args.deterministic)
    info("tf32", str(args.tf32))
    info("algoritmi deterministici", str(args.deterministic))

    section("Dati")
    if not (args.root / "ptbxl_database.csv").exists():
        print(f"ptbxl_database.csv non trovato in {args.root}", file=sys.stderr)
        return 2
    meta = D.load_metadata(args.root)
    splits = D.get_splits(meta)
    labels, label_names = D.load_label_matrix(args.root, meta)
    columns = clean_columns(label_names)
    info("record", f"{len(meta)} — train {splits.train.size}, val {splits.val.size}, "
                   f"test {splits.test.size}")
    info("etichette", f"{labels.shape[1]} in tutto, {len(columns)} nel set congelato")

    cache = D.SignalCache(args.root, args.cache_dir, meta, fs=run.fs, notch=run.notch)
    if not cache.array_path.exists():
        print(f"cache assente: {cache.array_path}\n"
              f"costruirla con: python scripts/build_cache.py --fs {run.fs}"
              f"{'' if run.notch else ' --no-notch'}", file=sys.stderr)
        return 2
    _ = cache.array  # forza la verifica di coerenza dei metadati
    info("cache", f"{cache.array_path.name}")

    scaler, scaler_path = resolve_scaler(run, args.root, args.cache_dir, meta, splits.train)
    info("scaler", f"mean {scaler.mean:.6g}, std {scaler.std:.6g} — {scaler_path.name}")

    train_idx, val_idx = splits.train, splits.val
    if args.max_records:
        train_idx = train_idx[: args.max_records]
        val_idx = val_idx[: args.max_records]

    model_cfg = ModelConfig(fs=run.fs, arm=run.arm, n_classes=labels.shape[1])
    common = dict(cache=cache, labels=labels, scaler=scaler, cfg=model_cfg)
    train_ds = WindowDataset(
        record_ids=train_idx, train=True, sampler=CropSampler(seed=run.seed), **common
    )
    val_ds = WindowDataset(record_ids=val_idx, train=False, **common)
    info("finestre", f"{model_cfg.n_samples} campioni ({model_cfg.window_seconds} s)")
    info("dataset", f"train {len(train_ds)} crop, val {len(val_ds)} crop")

    section("Training")
    out_dir = args.out_dir / run.run_id
    t0 = time.perf_counter()
    history = run_training(
        run=run,
        model_cfg=model_cfg,
        train_dataset=train_ds,
        val_dataset=val_ds,
        clean_columns=columns,
        out_dir=out_dir,
        cfg=train_cfg,
        device=args.device,
        manifest_extra={
            "git_sha": git_sha(),
            "scaler": {"path": scaler_path.name, **scaler.provenance,
                       "mean": scaler.mean, "std": scaler.std},
            "cache_stem": cache.stem,
            "n_train_records": int(train_idx.size),
            "n_val_records": int(val_idx.size),
            "reduced_run": smoke,
            "argv": sys.argv[1:],
        },
    )
    elapsed = time.perf_counter() - t0

    section("Esito")
    if not history:
        info("nessuna epoca eseguita", "il run era gia' completo")
        return 0
    selection = json.loads((out_dir / "selection.json").read_text())
    last = history[-1]
    info("epoche", f"{len(history)} — {elapsed / 60:.1f} min "
                   f"({elapsed / len(history):.1f} s/epoca)")
    info("ultima epoca", f"loss {last['train_loss']:.4f}, "
                         f"macro_all {last['macro_all']:.4f}, "
                         f"macro_clean {last['macro_clean']:.4f}")
    info("epoca selezionata", f"{selection['selected_epoch']} su {run.selection}")
    info("l'altro criterio avrebbe scelto",
         f"{selection['best_epoch_per_metric']}"
         f"{' (coincidono)' if selection['criteria_agree'] else ''}")
    info("artefatti", str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
