"""Costruisce la cache dei segnali per un rate e valida le assunzioni sui dati veri.

I test unitari girano su un mini-PTB-XL sintetico: verificano il codice, non i
dati. Questo script fa la cosa complementare, cioe' mette ``data.py`` di fronte
ai 21.837 record veri e controlla che le assunzioni hardcoded reggano.

Uso tipico:

    python scripts/build_cache.py --fs 100 --limit 200     # smoke test veloce
    python scripts/build_cache.py --fs 100                 # cache completa
    python scripts/build_cache.py --fs 100 --no-notch      # braccio Stage 0

Con ``--limit`` la cache finisce in una sottocartella ``_smoke`` e non puo'
sovrascrivere quella buona.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ecgres import data as D  # noqa: E402


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

class Report:
    """Raccoglie esiti e decide il codice di uscita."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        mark = "OK  " if ok else "FAIL"
        print(f"  [{mark}] {label}{(': ' + detail) if detail else ''}", flush=True)
        if not ok:
            self.failures.append(label)
        return ok

    def warn(self, label: str, detail: str = "") -> None:
        print(f"  [WARN] {label}{(': ' + detail) if detail else ''}", flush=True)
        self.warnings.append(label)

    def info(self, label: str, detail: str = "") -> None:
        print(f"  [    ] {label}{(': ' + detail) if detail else ''}", flush=True)


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}", flush=True)


# --------------------------------------------------------------------------
# Controlli
# --------------------------------------------------------------------------

def check_raw_sample(root: Path, meta: pd.DataFrame, rep: Report, n_probe: int = 40) -> None:
    """Legge un campione di record grezzi prima della passata lunga.

    Serve a far fallire in dieci secondi un percorso sbagliato o un file
    mancante, invece che a meta' di una costruzione da mezz'ora.
    """
    import wfdb

    rng = np.random.default_rng(0)
    positions = np.unique(
        np.concatenate([
            [0, len(meta) - 1],
            rng.choice(len(meta), size=min(n_probe, len(meta)), replace=False),
        ])
    )

    bad_shape, bad_nan, missing = [], [], []
    for pos in positions:
        rel = meta["filename_hr"].iloc[pos]
        try:
            signal, fields = wfdb.rdsamp(str(root / rel))
        except Exception as exc:  # file mancante o header corrotto
            missing.append(f"{rel} ({type(exc).__name__})")
            continue
        x = np.asarray(signal, dtype=np.float64).T
        if x.shape != (D.N_LEADS, D.SOURCE_FS * int(D.RECORD_SECONDS)):
            bad_shape.append(f"{rel} {x.shape}")
        if not np.isfinite(x).all():
            bad_nan.append(rel)
        if fields.get("fs") != D.SOURCE_FS:
            bad_shape.append(f"{rel} fs={fields.get('fs')}")

    rep.check(not missing, f"{positions.size} record grezzi leggibili",
              "" if not missing else "; ".join(missing[:3]))
    rep.check(not bad_shape, f"forma (12, 5000) e fs=500 sul campione",
              "" if not bad_shape else "; ".join(bad_shape[:3]))
    rep.check(not bad_nan, "nessun NaN/Inf nel campione grezzo",
              "" if not bad_nan else "; ".join(bad_nan[:3]))


N_PTBXL_RECORDS = 21799
"""Numero di record di PTB-XL v1.0.3. Alcuni controlli hanno senso solo sul totale."""


def check_labels(root: Path, meta: pd.DataFrame, rep: Report) -> pd.DataFrame:
    """Conteggi etichette, confrontati con results/ se disponibile."""
    y, labels = D.load_label_matrix(root, meta)
    fold = meta["strat_fold"].to_numpy()

    counts = pd.DataFrame(
        {
            "label": labels,
            "n_total": y.sum(axis=0),
            "n_fold10": y[fold == 10].sum(axis=0),
        }
    ).set_index("label")

    rep.check(len(labels) == 71, "71 etichette in scp_statements.csv",
              f"trovate {len(labels)}")
    if len(meta) == N_PTBXL_RECORDS:
        rep.check(True, f"{N_PTBXL_RECORDS} record, come atteso per v1.0.3")
    else:
        rep.warn(f"il database non ha {N_PTBXL_RECORDS} record",
                 f"trovati {len(meta)}: alcuni controlli verranno saltati")
    rep.check(bool((y.sum(axis=1) > 0).all()), "ogni record ha almeno un'etichetta",
              f"{int((y.sum(axis=1) == 0).sum())} record senza etichette")

    n_clean = int((counts["n_fold10"] >= 10).sum())
    full = len(meta) == N_PTBXL_RECORDS
    if full:
        rep.check(n_clean == 49, "49 etichette con >=10 positivi nel fold 10",
                  f"trovate {n_clean}")
    else:
        rep.info("etichette con >=10 positivi nel fold 10",
                 f"{n_clean} (atteso 49 solo sul dataset completo)")

    frozen_path = REPO_ROOT / "results" / "frozen_label_set.json"
    if frozen_path.exists() and full:
        frozen_raw = json.loads(frozen_path.read_text(encoding="utf-8"))
        if isinstance(frozen_raw, dict):
            frozen = frozen_raw.get("labels_clean") or frozen_raw.get("labels") or []
            # i parametri con cui il set e' stato congelato devono essere quelli
            # che lo script usa oggi: un accordo casuale sarebbe peggio di un
            # disaccordo dichiarato
            rep.check(frozen_raw.get("min_test_positives") == 10,
                      "soglia del set pre-registrato",
                      f"file: {frozen_raw.get('min_test_positives')}, script: 10")
            rep.check(frozen_raw.get("test_fold") == 10,
                      "fold di test del set pre-registrato",
                      f"file: {frozen_raw.get('test_fold')}, script: 10")
        else:
            frozen = list(frozen_raw)
        frozen = [str(x) for x in frozen]
        if not frozen:
            rep.warn("frozen_label_set.json non interpretabile", str(frozen_path))
        derived = set(counts.index[counts["n_fold10"] >= 10])
        rep.check(
            set(frozen) == derived,
            "il set pre-registrato coincide con quello ricalcolato",
            f"solo in frozen: {sorted(set(frozen) - derived)[:5]}; "
            f"solo qui: {sorted(derived - set(frozen))[:5]}",
        )
    elif not frozen_path.exists():
        rep.warn("frozen_label_set.json non trovato", str(frozen_path))
    else:
        rep.info("confronto con frozen_label_set.json", "saltato su dataset parziale")

    return counts


def check_splits(meta: pd.DataFrame, rep: Report) -> None:
    try:
        splits = D.get_splits(meta)
    except ValueError as exc:
        rep.check(False, "split senza leakage paziente", str(exc))
        return
    sizes = {k: v.size for k, v in splits.as_dict().items()}
    rep.check(sum(sizes.values()) == len(meta), "gli split coprono tutti i record",
              str(sizes))
    rep.check(True, "nessun paziente condiviso fra split")


def check_cache(cache: D.SignalCache, meta: pd.DataFrame, rep: Report) -> None:
    """Validazione della cache costruita, sul memmap."""
    arr = cache.array
    expected = (len(meta), D.N_LEADS, cache.n_record_samples)
    rep.check(arr.shape == expected, "forma della cache", f"{arr.shape}")
    rep.check(arr.dtype == np.float32, "dtype float32", str(arr.dtype))

    n_nan = n_flat = 0
    amp_min, amp_max = np.inf, -np.inf
    peak = np.empty(arr.shape[0], dtype=np.float32)
    block = 512
    for start in range(0, arr.shape[0], block):
        chunk = np.asarray(arr[start : start + block])
        n_nan += int((~np.isfinite(chunk)).sum())
        spread = chunk.max(axis=-1) - chunk.min(axis=-1)
        n_flat += int((spread == 0).all(axis=-1).sum())
        amp_min = min(amp_min, float(np.nanmin(chunk)))
        amp_max = max(amp_max, float(np.nanmax(chunk)))
        peak[start : start + chunk.shape[0]] = np.abs(chunk).max(axis=(1, 2))

    rep.check(n_nan == 0, "nessun NaN/Inf nella cache", f"{n_nan} valori")
    rep.check(n_flat == 0, "nessun record piatto su tutte le derivazioni",
              f"{n_flat} record")
    rep.info("ampiezza (mV)", f"min {amp_min:.3f}, max {amp_max:.3f}")

    # Conteggi per soglia: PROTOCOL.md §10 voce 7 cita queste cifre, quindi
    # devono essere riproducibili da chiunque cloni il repo.
    over = " | ".join(
        f">{t} mV: {int((peak > t).sum())} ({(peak > t).mean() * 100:.2f}%)"
        for t in (5, 10, 20, 30)
    )
    rep.info("record oltre soglia", over)
    rep.info("ampiezza di picco per record (mV)",
             f"mediana {float(np.median(peak)):.2f}, "
             f"p99 {float(np.percentile(peak, 99)):.2f}")
    top = np.argsort(peak)[-6:][::-1]
    rep.info("sei record con picco maggiore",
             ", ".join(f"{meta.index[i]}: {peak[i]:.1f}" for i in top))
    if amp_max > 20 or amp_min < -20:
        rep.warn("ampiezze fuori dal range fisiologico tipico",
                 "possibile problema di unita' o di gain")


def check_recompute(root: Path, cache: D.SignalCache, meta: pd.DataFrame,
                    rep: Report, n_probe: int = 5) -> None:
    """Ricalcola qualche record da zero e lo confronta con la cache.

    Verifica insieme l'ordine delle righe e l'integrita' della scrittura: se la
    cache fosse disallineata di una riga, qui si vede.
    """
    import wfdb

    rng = np.random.default_rng(1)
    positions = rng.choice(len(meta), size=min(n_probe, len(meta)), replace=False)
    worst = 0.0
    for pos in positions:
        rel = meta["filename_hr"].iloc[pos]
        signal, _ = wfdb.rdsamp(str(root / rel))
        fresh = D.preprocess_record(
            np.asarray(signal, dtype=np.float64).T, cache.fs, notch=cache.notch
        )
        worst = max(worst, float(np.abs(fresh - np.asarray(cache.array[pos])).max()))
    rep.check(worst == 0.0, f"{positions.size} record ricalcolati identici alla cache",
              f"scarto massimo {worst:.3e}")


def check_alignment(root: Path, cache_dir: Path, meta: pd.DataFrame,
                    cache: D.SignalCache, rep: Report, n_probe: int = 8) -> None:
    """Allineamento contro un altro rate, sui dati veri.

    Costruisce al volo le due viste per pochi record, senza materializzare una
    seconda cache completa.
    """
    import wfdb

    other_fs = 500 if cache.fs != 500 else 100
    rng = np.random.default_rng(2)
    positions = rng.choice(len(meta), size=min(n_probe, len(meta)), replace=False)
    starts = D.inference_crop_starts()

    def view(fs: int) -> D.WindowBatch:
        width = D.n_samples_for(D.WINDOW_SECONDS, fs)
        out = np.empty((positions.size * starts.size, D.N_LEADS, width), dtype=np.float32)
        rec_idx, start_s, k = [], [], 0
        for pos in positions:
            signal, _ = wfdb.rdsamp(str(root / meta["filename_hr"].iloc[pos]))
            x = D.preprocess_record(
                np.asarray(signal, dtype=np.float64).T, fs, notch=cache.notch
            )
            for t in starts:
                begin = int(round(t * fs))
                out[k] = x[:, begin : begin + width]
                rec_idx.append(pos)
                start_s.append(t)
                k += 1
        return D.WindowBatch(
            fs=fs, signals=out, record_idx=np.asarray(rec_idx),
            start_s=np.asarray(start_s), notch=cache.notch,
        )

    try:
        D.assert_aligned(view(cache.fs), view(other_fs), check_signal=True)
    except AssertionError as exc:
        rep.check(False, f"allineamento {cache.fs} Hz vs {other_fs} Hz", str(exc))
        return
    rep.check(True, f"allineamento {cache.fs} Hz vs {other_fs} Hz su {positions.size} record")


def write_environment(cache: D.SignalCache) -> Path:
    """Sidecar con le versioni usate per costruire la cache.

    ``resample_poly`` costruisce il filtro anti-aliasing con l'algoritmo della
    versione di SciPy installata. Se la cache viene costruita con una versione
    e il training gira con un'altra, i segnali non sono piu' bit-per-bit gli
    stessi: qui resta scritto, cosi' un mismatch si vede invece di passare
    inosservato.
    """
    path = cache.cache_dir / f"{cache.stem}.env.json"
    path.write_text(json.dumps({
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
    }, indent=2), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fs", type=int, required=True, choices=sorted(D.RATES))
    ap.add_argument("--root", type=Path, default=REPO_ROOT / "data" / "ptb-xl")
    ap.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data" / "cache")
    ap.add_argument("--no-notch", action="store_true",
                    help="disattiva il notch 50 Hz (Stage 0, analisi 8.3.4)")
    ap.add_argument("--limit", type=int, default=None,
                    help="usa solo i primi N record; scrive in una cartella _smoke")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    notch = not args.no_notch
    rep = Report()

    section("Ambiente")
    rep.info("python", sys.version.split()[0])
    rep.info("numpy / scipy / pandas",
             f"{np.__version__} / {scipy.__version__} / {pd.__version__}")
    rep.info("root dati", str(args.root))

    if not (args.root / "ptbxl_database.csv").exists():
        print(f"\nptbxl_database.csv non trovato in {args.root}", file=sys.stderr)
        return 2

    section("Metadati")
    meta = D.load_metadata(args.root)
    rep.info("record nel database", str(len(meta)))
    cache_dir = args.cache_dir
    if args.limit:
        meta = meta.iloc[: args.limit]
        cache_dir = cache_dir / "_smoke"
        rep.warn("modalita' smoke test", f"solo {len(meta)} record, cache in {cache_dir}")

    check_splits(meta, rep)
    counts = check_labels(args.root, meta, rep)

    section("Campione di record grezzi")
    check_raw_sample(args.root, meta, rep)
    blocking = [f for f in rep.failures
                if "record grezzi" in f or "forma" in f or "NaN" in f]
    if blocking:
        print(f"\nControlli bloccanti falliti: {', '.join(blocking)}", file=sys.stderr)
        return 1
    if rep.failures:
        print(f"\nAvanti con la cache, ma {len(rep.failures)} controlli sui metadati "
              f"sono falliti: {', '.join(rep.failures)}", file=sys.stderr)

    section(f"Costruzione cache — {args.fs} Hz, notch={notch}")
    cache = D.SignalCache(args.root, cache_dir, meta, fs=args.fs, notch=notch)
    t0 = time.perf_counter()
    cache.build(overwrite=args.overwrite, progress=True)
    elapsed = time.perf_counter() - t0
    size_gb = cache.array_path.stat().st_size / 1e9
    rep.info("tempo", f"{elapsed:.1f} s ({elapsed / max(len(meta), 1) * 1000:.1f} ms/record)")
    rep.info("dimensione", f"{size_gb:.2f} GB — {cache.array_path}")

    section("Validazione cache")
    check_cache(cache, meta, rep)
    check_recompute(args.root, cache, meta, rep)
    check_alignment(args.root, cache_dir, meta, cache, rep)
    env_path = write_environment(cache)
    rep.info("ambiente registrato", str(env_path.name))

    out = REPO_ROOT / "results" / f"label_support_from_cache_fs{args.fs}.csv"
    if not args.limit:
        out.parent.mkdir(exist_ok=True)
        counts.to_csv(out)
        rep.info("conteggi etichette", str(out.name))

    section("Esito")
    if rep.failures:
        print(f"  {len(rep.failures)} controlli falliti: {', '.join(rep.failures)}")
        return 1
    print(f"  tutti i controlli superati"
          + (f", {len(rep.warnings)} avvisi" if rep.warnings else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
