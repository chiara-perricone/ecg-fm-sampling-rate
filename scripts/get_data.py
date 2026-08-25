"""
Download PTB-XL v1.0.3 and verify the local copy is complete.

Cross-platform (Windows / macOS / Linux), standard library only.

Deliberately downloads the single published archive rather than crawling the
PhysioNet file tree recursively. Berger et al. (arXiv:2602.17531v2, App. B)
report that the recursive crawl can silently produce incomplete local copies,
and that two machines produced different partial copies of PTB-XL. An archive
plus a file count cannot fail that way, and PROTOCOL.md §3 treats an unverified
dataset as a blocking failure.

Usage:
    python scripts/get_data.py                 # downloads into ./data
    python scripts/get_data.py --out D:/data   # elsewhere
    python scripts/get_data.py --verify-only   # re-check an existing copy

Needs ~5 GB free: 1.7 GB archive plus ~3 GB extracted. The archive can be
deleted afterwards with --clean.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

VERSION = "1.0.3"
STEM = f"ptb-xl-a-large-publicly-available-electrocardiography-dataset-{VERSION}"
ZIP_URL = f"https://physionet.org/static/published-projects/ptb-xl/{STEM}.zip"
PROJECT_PAGE = f"https://physionet.org/content/ptb-xl/{VERSION}/"

EXPECTED_RECORDS = 21799
EXPECTED_PATIENTS = 18869


# --------------------------------------------------------------------------

def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def check_disk(path: Path, need_gb: float = 5.0) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free < need_gb * 1024**3:
        sys.exit(
            f"Only {human(free)} free at {path}. Need about {need_gb} GB "
            f"(1.7 GB archive + ~3 GB extracted)."
        )


def download(url: str, dest: Path) -> None:
    """Stream to disk with resume, so an interrupted 1.7 GB download is cheap."""
    existing = dest.stat().st_size if dest.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "ecg-fm-sampling-rate"})
    if existing:
        req.add_header("Range", f"bytes={existing}-")
        print(f"    resuming from {human(existing)}")

    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # range not satisfiable: already complete
            print("    already fully downloaded")
            return
        sys.exit(
            f"HTTP {e.code} fetching {url}\n"
            f"If the path has moved, get the zip manually from the Files "
            f"section of {PROJECT_PAGE} and re-run with --verify-only."
        )

    total = int(resp.headers.get("Content-Length", 0)) + existing
    mode = "ab" if existing and resp.status == 206 else "wb"
    got = existing if mode == "ab" else 0

    with open(dest, mode) as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            got += len(chunk)
            if total:
                pct = 100 * got / total
                print(f"\r    {human(got)} / {human(total)}  ({pct:5.1f}%)",
                      end="", flush=True)
    print()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def extract(zip_path: Path, out: Path) -> Path:
    print("    testing archive integrity ...")
    with zipfile.ZipFile(zip_path) as z:
        bad = z.testzip()
        if bad is not None:
            sys.exit(f"Archive is corrupt at {bad}. Delete it and re-run.")
        print("    OK")
        print("    extracting (a few minutes) ...")
        z.extractall(out)

    nested = out / STEM
    root = out / "ptb-xl"
    if nested.exists():
        if root.exists():
            shutil.rmtree(root)
        nested.rename(root)
    return root


# --------------------------------------------------------------------------

def verify(root: Path) -> None:
    print(f"\n=== Verifying {root} ===")
    problems = []

    for f in ("ptbxl_database.csv", "scp_statements.csv", "RECORDS"):
        if not (root / f).exists():
            problems.append(f"missing {f}")

    for d in ("records100", "records500"):
        if not (root / d).is_dir():
            problems.append(f"missing {d}/")

    for d in ("records100", "records500"):
        if (root / d).is_dir():
            hea = sum(1 for _ in (root / d).rglob("*.hea"))
            dat = sum(1 for _ in (root / d).rglob("*.dat"))
            status = "OK  " if hea == dat == EXPECTED_RECORDS else "FAIL"
            print(f"  {status} {d}: {hea} .hea / {dat} .dat "
                  f"(expected {EXPECTED_RECORDS} each)")
            if hea != EXPECTED_RECORDS or dat != EXPECTED_RECORDS:
                problems.append(f"{d} incomplete")

    db = root / "ptbxl_database.csv"
    if db.exists():
        with open(db, encoding="utf-8") as f:
            rows = sum(1 for _ in f) - 1
        status = "OK  " if rows == EXPECTED_RECORDS else "FAIL"
        print(f"  {status} ptbxl_database.csv: {rows} rows "
              f"(expected {EXPECTED_RECORDS})")
        if rows != EXPECTED_RECORDS:
            problems.append("metadata row count wrong")

    if problems:
        print("\nFAILED:")
        for p in problems:
            print(f"  - {p}")
        print("\nDelete the target directory and re-run rather than patching\n"
              "individual files. PROTOCOL.md §3 treats an incomplete dataset as\n"
              "a blocking failure: it would fail the 0.941 reproduction with no\n"
              "indication of why.")
        sys.exit(1)

    print("\n  All checks passed.")
    print(f"  Next: python scripts/describe.py --data {root}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--clean", action="store_true",
                    help="delete the archive after successful extraction")
    args = ap.parse_args()

    root = args.out / "ptb-xl"

    if args.verify_only:
        if not root.exists():
            sys.exit(f"{root} does not exist. Run without --verify-only first.")
        verify(root)
        return

    check_disk(args.out)
    zip_path = args.out / f"{STEM}.zip"

    print(f"=== Downloading PTB-XL {VERSION} (~1.7 GB) ===")
    print(f"    {ZIP_URL}")
    download(ZIP_URL, zip_path)

    print("\n=== Checksum ===")
    digest = sha256(zip_path)
    print(f"    SHA256 {digest}")
    print(f"    Compare against the value shown on {PROJECT_PAGE}")
    print("    (record it in the repo once confirmed, so future runs can assert it)")

    print("\n=== Extracting ===")
    root = extract(zip_path, args.out)

    verify(root)

    if args.clean:
        zip_path.unlink()
        print(f"\n  Removed {zip_path.name}")


if __name__ == "__main__":
    main()
