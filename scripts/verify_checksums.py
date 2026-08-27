"""
Verify every extracted PTB-XL file against PhysioNet's published SHA256SUMS.txt.

Stronger than an archive-level check: it catches a single silently corrupted
record among 43,000, not just a truncated download. PROTOCOL.md §3 treats an
unverified dataset as a blocking failure, because an incomplete or corrupted
copy would fail the 0.941 reproduction with no indication of why. Berger et al.
(arXiv:2602.17531v2, App. B) report exactly this: two machines ended up with
different partial copies of PTB-XL.

Run once after extraction. It takes a few minutes and prints a digest line to
paste into the repo as a record that the local copy was verified.

Usage:
    python scripts/verify_checksums.py --data data/ptb-xl --sums data/ptb-xl/SHA256SUMS.txt
    python scripts/verify_checksums.py --data data/ptb-xl --sums data/ptb-xl/SHA256SUMS.txt --quick

PhysioNet ships SHA256SUMS.txt inside the archive, so it lands next to
records100/ and records500/ rather than one level up.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

CHUNK = 1 << 20


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def parse_sums(path: Path) -> dict[str, str]:
    """PhysioNet's format is '<hex>  <relative/path>' or '<hex> *<path>'."""
    mapping: dict[str, str] = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            digest, name = parts
            name = name.lstrip("*").strip()
            # Normalise separators so Windows and POSIX agree.
            mapping[name.replace("\\", "/")] = digest.lower()
    return mapping


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="PTB-XL root")
    ap.add_argument("--sums", type=Path, required=True, help="SHA256SUMS.txt")
    ap.add_argument("--quick", action="store_true",
                    help="check metadata and records500 only, skipping records100")
    args = ap.parse_args()

    if not args.sums.exists():
        sys.exit(f"{args.sums} not found. Download it from "
                 f"https://physionet.org/files/ptb-xl/1.0.3/SHA256SUMS.txt")

    expected = parse_sums(args.sums)
    print(f"Loaded {len(expected)} checksums from {args.sums.name}")

    names = sorted(expected)
    if args.quick:
        names = [n for n in names if not n.startswith("records100/")]
        print(f"Quick mode: checking {len(names)} entries "
              f"(records100 skipped; it is not used by this study)")

    missing: list[str] = []
    mismatched: list[str] = []
    checked = 0
    t0 = time.time()

    for i, name in enumerate(names, 1):
        target = args.data / name
        if not target.exists():
            missing.append(name)
            continue
        if sha256(target) != expected[name]:
            mismatched.append(name)
        checked += 1

        if i % 2000 == 0 or i == len(names):
            rate = i / max(time.time() - t0, 1e-9)
            print(f"\r  {i}/{len(names)}  ({rate:.0f} files/s)", end="", flush=True)
    print()

    extra_note = ""
    if not args.quick:
        on_disk = {
            str(p.relative_to(args.data)).replace("\\", "/")
            for p in args.data.rglob("*") if p.is_file()
        }
        unexpected = sorted(on_disk - set(expected) - {"SHA256SUMS.txt"})
        if unexpected:
            extra_note = (f"\n  Note: {len(unexpected)} file(s) on disk are not "
                          f"listed in SHA256SUMS.txt (first: {unexpected[0]}). "
                          f"This is informational, not a failure.")

    print(f"\n=== Result ===")
    print(f"  verified   {checked}")
    print(f"  missing    {len(missing)}")
    print(f"  mismatched {len(mismatched)}")
    if extra_note:
        print(extra_note)

    if missing or mismatched:
        for label, items in (("MISSING", missing), ("MISMATCHED", mismatched)):
            for name in items[:10]:
                print(f"  {label}  {name}")
            if len(items) > 10:
                print(f"  ... and {len(items) - 10} more")
        print("\nFAILED. Delete the data directory and re-download rather than\n"
              "patching individual files: a partial repair leaves you unable to\n"
              "say what the local copy actually is.")
        sys.exit(1)

    print("\n  All files match the published checksums.")
    print(f"\n  Record this in the repo:")
    print(f"  PTB-XL v1.0.3 verified against SHA256SUMS.txt "
          f"({checked} files{' , quick mode' if args.quick else ''}) "
          f"on {time.strftime('%Y-%m-%d')}.")


if __name__ == "__main__":
    main()
