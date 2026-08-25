"""
Descriptive analysis of PTB-XL, scoped to what this repository's protocol needs.

This is not general-purpose EDA. It answers four questions that PROTOCOL.md
depends on, and it must be run before any training run because one of its
outputs is pre-registered:

  1. Integrity      -- is the local copy complete and internally consistent?
  2. Folds          -- do the official folds have the expected sizes, and are
                       patients disjoint across them?
  3. Label support  -- how many positive examples does each of the 71 labels
                       have in fold 10? This freezes the `macro_clean` label set
                       (PROTOCOL.md §7) before any result can influence it.
  4. Spectrum       -- how much signal power lies above the Nyquist frequency of
                       each rate under test? This is the physical quantity the
                       whole experiment is about: downsampling to fs discards
                       everything above fs/2, so the power above 50 Hz (for
                       100 Hz), 120 Hz (240 Hz) and 125 Hz (250 Hz) bounds what
                       *can* be lost.

Outputs go to results/ and are small enough to commit. The data itself is not.

Usage:
    python scripts/describe.py --data data/ptb-xl [--n-spectrum 500]
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy import signal

FS_NATIVE = 500.0
RATES_UNDER_TEST = [100, 240, 250, 500]
MIN_TEST_POSITIVES = 10  # PROTOCOL.md §7
TRAIN_FOLDS, VAL_FOLD, TEST_FOLD = range(1, 9), 9, 10


# --------------------------------------------------------------------------
# 1. Load
# --------------------------------------------------------------------------

def load_metadata(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    db = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    db.scp_codes = db.scp_codes.apply(ast.literal_eval)
    scp = pd.read_csv(root / "scp_statements.csv", index_col=0)
    return db, scp


def build_label_matrix(db: pd.DataFrame, scp: pd.DataFrame) -> pd.DataFrame:
    """PTB-XL (all): the full set of SCP statements, as a binary matrix.

    A statement counts as present when it appears in scp_codes at all. PTB-XL
    attaches a likelihood to some statements; 0 is used in the dataset to mean
    'unknown likelihood', not 'absent', so thresholding on it would silently
    drop labels. Presence is the convention used by the PTB-XL benchmark.
    """
    labels = sorted(scp.index)
    mat = pd.DataFrame(0, index=db.index, columns=labels, dtype=np.int8)
    for ecg_id, codes in db.scp_codes.items():
        for code in codes:
            if code in mat.columns:
                mat.at[ecg_id, code] = 1
    return mat


# --------------------------------------------------------------------------
# 2. Integrity and folds
# --------------------------------------------------------------------------

def check_integrity(root: Path, db: pd.DataFrame) -> list[str]:
    problems = []

    if len(db) != 21799:
        problems.append(f"expected 21799 records in metadata, found {len(db)}")

    n_patients = db.patient_id.nunique()
    if n_patients != 18869:
        problems.append(f"expected 18869 patients, found {n_patients}")

    missing = [
        fn for fn in db.filename_hr
        if not (root / f"{fn}.dat").exists() or not (root / f"{fn}.hea").exists()
    ]
    if missing:
        problems.append(
            f"{len(missing)} records referenced in metadata are absent from disk "
            f"(first: {missing[0]})"
        )

    # Patient disjointness across folds. If this fails the splits leak and every
    # downstream number is inflated.
    fold_of_patient = db.groupby("patient_id").strat_fold.nunique()
    leaking = fold_of_patient[fold_of_patient > 1]
    if len(leaking):
        problems.append(
            f"{len(leaking)} patients appear in more than one fold "
            f"(first: patient {leaking.index[0]})"
        )

    return problems


def fold_table(db: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold in range(1, 11):
        sub = db[db.strat_fold == fold]
        split = ("train" if fold in TRAIN_FOLDS
                 else "validation" if fold == VAL_FOLD else "test")
        rows.append({
            "fold": fold,
            "split": split,
            "records": len(sub),
            "patients": sub.patient_id.nunique(),
            "female_pct": round(100 * sub.sex.mean(), 1),  # sex: 0 male, 1 female
            "age_median": sub.age.median(),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3. Label support -> frozen macro_clean set
# --------------------------------------------------------------------------

def label_support(db: pd.DataFrame, y: pd.DataFrame, scp: pd.DataFrame) -> pd.DataFrame:
    test_ids = db.index[db.strat_fold == TEST_FOLD]
    train_ids = db.index[db.strat_fold.isin(TRAIN_FOLDS)]

    out = pd.DataFrame({
        "description": scp.reindex(y.columns).get(
            "description", pd.Series(index=y.columns, dtype=object)
        ),
        "n_total": y.sum(axis=0),
        "n_train": y.loc[train_ids].sum(axis=0),
        "n_test": y.loc[test_ids].sum(axis=0),
    })
    out["prevalence_pct"] = (100 * out.n_total / len(y)).round(3)
    out["in_macro_clean"] = out.n_test >= MIN_TEST_POSITIVES
    return out.sort_values("n_test", ascending=False)


# --------------------------------------------------------------------------
# 4. Spectrum
# --------------------------------------------------------------------------

def power_above_nyquist(root: Path, db: pd.DataFrame, n_records: int,
                        rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Average Welch PSD across a sample of training records, and the fraction
    of total power above each rate's Nyquist frequency.

    Sampled from the training folds only, so that nothing about the test set
    informs a design decision.
    """
    pool = db.index[db.strat_fold.isin(TRAIN_FOLDS)]
    chosen = rng.choice(pool, size=min(n_records, len(pool)), replace=False)

    psd_sum, freqs, n_used = None, None, 0
    for ecg_id in chosen:
        rec = wfdb.rdrecord(str(root / db.at[ecg_id, "filename_hr"]))
        x = rec.p_signal  # (samples, leads)
        if not np.isfinite(x).all():
            continue
        # Welch per lead, then average over leads. 2 s segments at 500 Hz give
        # 0.5 Hz resolution, ample for locating the 50-125 Hz region.
        f, p = signal.welch(x, fs=FS_NATIVE, nperseg=1000, axis=0)
        p = p.mean(axis=1)
        psd_sum = p if psd_sum is None else psd_sum + p
        freqs, n_used = f, n_used + 1

    if n_used == 0:
        raise RuntimeError("no usable records sampled for spectral analysis")

    psd = psd_sum / n_used
    total = np.trapezoid(psd, freqs)

    rows = []
    for fs in RATES_UNDER_TEST:
        nyq = fs / 2
        mask = freqs > nyq
        discarded = np.trapezoid(psd[mask], freqs[mask]) if mask.any() else 0.0
        rows.append({
            "rate_hz": fs,
            "nyquist_hz": nyq,
            "power_discarded_pct": round(100 * discarded / total, 4),
        })
    return pd.DataFrame(rows), freqs, psd


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="PTB-XL root")
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--n-spectrum", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("Loading metadata ...")
    db, scp = load_metadata(args.data)
    y = build_label_matrix(db, scp)

    print("\n=== 1. Integrity ===")
    problems = check_integrity(args.data, db)
    if problems:
        for p in problems:
            print(f"  FAIL  {p}")
        print("\nStop here. PROTOCOL.md §3 treats an unverified dataset as a")
        print("blocking failure, because it would fail the 0.941 reproduction")
        print("with no indication of why.")
        raise SystemExit(1)
    print(f"  OK    {len(db)} records, {db.patient_id.nunique()} patients")
    print("  OK    all referenced signal files present")
    print("  OK    no patient appears in more than one fold")

    print("\n=== 2. Folds ===")
    folds = fold_table(db)
    print(folds.to_string(index=False))
    folds.to_csv(args.out / "folds.csv", index=False)

    print("\n=== 3. Label support (fold 10) ===")
    support = label_support(db, y, scp)
    support.to_csv(args.out / "label_support.csv")
    n_clean = int(support.in_macro_clean.sum())
    print(f"  {len(support)} labels total")
    print(f"  {n_clean} with >= {MIN_TEST_POSITIVES} positives in fold 10 "
          f"-> macro_clean")
    print(f"  {len(support) - n_clean} below threshold -> reported separately")
    print("\n  Lowest-support labels retained under macro_all:")
    print(support.tail(8)[["n_test", "prevalence_pct", "in_macro_clean"]].to_string())

    frozen = {
        "min_test_positives": MIN_TEST_POSITIVES,
        "test_fold": TEST_FOLD,
        "n_labels_all": int(len(support)),
        "n_labels_clean": n_clean,
        "labels_clean": sorted(support.index[support.in_macro_clean].tolist()),
    }
    (args.out / "frozen_label_set.json").write_text(json.dumps(frozen, indent=2))
    print(f"\n  Frozen to {args.out / 'frozen_label_set.json'} — commit this "
          f"before the first training run.")

    print(f"\n=== 4. Spectrum ({args.n_spectrum} training records) ===")
    spec, freqs, psd = power_above_nyquist(args.data, db, args.n_spectrum, rng)
    print(spec.to_string(index=False))
    spec.to_csv(args.out / "power_above_nyquist.csv", index=False)
    np.savez_compressed(args.out / "mean_psd.npz", freqs=freqs, psd=psd)
    print("\n  Read this as an upper bound on what downsampling can destroy.")
    print("  A small percentage does not guarantee a small effect on AUROC —")
    print("  discarded power is not the same as discarded discriminative")
    print("  information — but a large one would make a large effect expected.")


if __name__ == "__main__":
    main()
