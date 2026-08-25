"""
Where does the power that downsampling discards actually sit?

describe.py reports that resampling PTB-XL from 500 Hz to 100 Hz discards about
0.54% of total signal power. That figure is only meaningful once we know what
kind of power it is.

PTB-XL was recorded in Germany, where the mains frequency is 50 Hz. Mains
interference appears at 50 Hz and at its harmonics: 100, 150, 200 Hz. The
Nyquist frequency of the 100 Hz arm is exactly 50 Hz, so the 100 and 150 Hz
harmonics fall inside the band that downsampling removes.

If the discarded power is concentrated at those harmonics, downsampling to
100 Hz is removing interference rather than signal, and the expected effect on
macro AUROC is nil or favourable. If it is spread smoothly, it is broadband
signal content and the effect could go either way.

The 60-90 Hz window is the control: it contains no mains harmonic, so it
represents the background level against which the harmonic bands are compared.

Usage:
    python scripts/psd_bands.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# NumPy renamed trapz to trapezoid in 2.0.
_integrate = getattr(np, "trapezoid", None) or np.trapz

PSD_FILE = Path("results/mean_psd.npz")

BANDS = [
    (45, 55, "50 Hz mains fundamental"),
    (95, 105, "100 Hz  2nd harmonic"),
    (145, 155, "150 Hz  3rd harmonic"),
    (195, 205, "200 Hz  4th harmonic"),
    (60, 90, "60-90 Hz control (no harmonic)"),
    (110, 140, "110-140 Hz control (no harmonic)"),
]


def main() -> None:
    if not PSD_FILE.exists():
        raise SystemExit(
            f"{PSD_FILE} not found. Run scripts/describe.py first."
        )

    d = np.load(PSD_FILE)
    f, psd = d["freqs"], d["psd"]
    total = _integrate(psd, f)

    print("Share of total signal power by band\n")
    print(f"  {'band':34s} {'% of total':>11s} {'% per Hz':>10s}")
    print(f"  {'-' * 34} {'-' * 11} {'-' * 10}")

    for lo, hi, label in BANDS:
        m = (f >= lo) & (f <= hi)
        if not m.any():
            continue
        share = 100 * _integrate(psd[m], f[m]) / total
        density = share / (hi - lo)
        print(f"  {label:34s} {share:10.4f}% {density:9.5f}%")

    # Density is the comparable quantity: the bands have different widths, so
    # raw share would favour the wider control windows by construction.
    def density(lo: float, hi: float) -> float:
        m = (f >= lo) & (f <= hi)
        return (100 * _integrate(psd[m], f[m]) / total) / (hi - lo)

    ctrl = (density(60, 90) + density(110, 140)) / 2
    print(f"\n  Control density (mean of the two control windows): "
          f"{ctrl:.5f}% per Hz")
    print("\n  Ratio of each harmonic band to the control density:")
    for lo, hi, label in BANDS[:4]:
        r = density(lo, hi) / ctrl if ctrl > 0 else float("nan")
        verdict = ("clear peak" if r > 3 else
                   "mild excess" if r > 1.5 else "no excess")
        print(f"    {label:34s} {r:7.2f}x   {verdict}")

    # What fraction of everything above 50 Hz sits in the harmonic bands?
    above = (f > 50)
    harm = ((f >= 95) & (f <= 105)) | ((f >= 145) & (f <= 155)) | \
           ((f >= 195) & (f <= 205))
    p_above = _integrate(psd[above], f[above])
    p_harm = _integrate(psd[harm], f[harm])
    print(f"\n  Of all power above 50 Hz (the band the 100 Hz arm discards),")
    print(f"  {100 * p_harm / p_above:.1f}% sits within +/-5 Hz of a mains harmonic.")


if __name__ == "__main__":
    main()
