# Runbook

How to reproduce this work from nothing, in the order the steps have to happen.
Written before the first rented hour, because on a metered machine a mistake in
sequencing costs money rather than time.

Every step states what to check before moving on. If a check fails, stop there:
none of the later steps is worth running on a broken earlier one.

---

## 0. Before renting anything

Everything up to and including the caches can be done on a laptop, and should
be, because none of it needs a GPU and all of it can fail.

- Python 3.11, the versions pinned in `requirements.txt`. `scipy` matters more
  than the others: `resample_poly` builds its anti-aliasing filter with the
  algorithm of the installed version, so a cache built under one version is not
  bit-identical to one built under another. Each cache carries a `.env.json`
  recording the versions that produced it.
- About 25 GB of disk: 3 GB for PTB-XL, 13.5 GB for the six caches, the rest for
  checkpoints.

```bash
pip install -r requirements-cpu.txt
pytest -q                      # expect: all green, no skips other than optional deps
```

`requirements.txt` holds everything except torch and is not installed directly.
The torch build is platform-specific — `requirements-cpu.txt` here,
`requirements-cuda.txt` on the pod — because a `+cpu` or `+cu130` pin resolves
only against `download.pytorch.org` and fails outright anywhere else. Both
wrapper files pin the same torch version; if the pod forces that version to
change, both change together.

---

## 1. Data

```bash
python scripts/get_data.py
python scripts/verify_checksums.py
```

**Check:** 21,799 records, and every file matching `SHA256SUMS.txt`. PhysioNet's
recursive `wget` is known to produce silently incomplete copies, and an
incomplete dataset fails §3 with no indication of why (PROTOCOL §3).

---

## 2. Caches

Six of them. Five derive from `records500`; the sixth is `records100` read as
distributed, for §8.3.2.

```bash
python scripts/build_cache.py --fs 500                      # comparison arms
python scripts/build_cache.py --fs 250
python scripts/build_cache.py --fs 240
python scripts/build_cache.py --fs 100
python scripts/build_cache.py --fs 100 --no-notch           # blocks 0 and 3
python scripts/build_cache.py --fs 100 --no-notch --source lr   # block 4
```

**Check:** every cache reports `5 record ricalcolati identici alla cache` with a
maximum deviation of exactly zero, and the `hr` caches pass the cross-rate
alignment check. The `lr` cache skips alignment by design: between `records100`
and resampled `records500` there is no alignment to verify, there is a
difference to measure, and measuring it is block 4.

Try `--limit 200` first on any cache you are unsure about. It writes to a
`_smoke` subdirectory and cannot overwrite the real one.

---

## 3. Rent the machine

An RTX 4090-class card with strong fp32 throughput, about €0.35/h on Vast.ai or
RunPod, 30 GB of volume. Persistent storage, so an interrupted session does not
lose the caches.

Repeat steps 0 to 2 on the pod, with `requirements-cuda.txt` in place of
`requirements-cpu.txt`. **The caches are rebuilt there rather than copied**, so
that every signal the models see was produced by one environment.

Two things to establish before installing anything, in this order, because both
can send you back to a different pod:

```bash
nvidia-smi          # "CUDA Version" here is the driver ceiling, not the install
nvcc --version      # the toolkit in the image; pykeops compiles against this
```

torch 2.13.0 is published for cu130 only, so the driver has to support CUDA 13.0.
The `runpod/pytorch:*-cu1281-*` image ships an nvcc from 12.8, one major version
below the runtime that wheel bundles, which is the kind of mismatch that makes
pykeops fail to build — and pykeops is the largest single speed lever in step 4.
Either find an image whose toolkit matches, or take the fallback recorded in
`requirements-cuda.txt`: torch 2.11.0 on cu128, in both wrapper files, verified
by `pytest -q`.

A stopped pod still bills for its volume disk, and a network volume bills while
nothing is running at all. Size the disk to about 40 GB rather than accepting the
default, and delete the volume after step 8 rather than merely stopping the pod.

---

## 4. Measure before committing to the budget

This is the step most likely to be skipped and the one that most repays being
done. §6.7 estimates 20–30 GPU-hours for all 48 runs. That estimate is not
measured, and the CPU timings gathered while writing the code suggest it may be
optimistic.

```bash
python -c "import ecgres.vendor.s4"      # note which Cauchy backend it reports
pip install pykeops                      # then check again
python scripts/train.py --run-id b0-fs100-A-n0-s0 --epochs 1 --deterministic
python scripts/train.py --run-id b1-fs500-A-n1-s0 --epochs 1 --deterministic
```

`--deterministic` and the absence of `--tf32` are §10 entry 25, and they are
settled here rather than later because the fingerprint will refuse to resume
across a change in either. If the first command raises instead of training,
`torch` is naming an operation S4 needs that has no deterministic CUDA
implementation: record the operation, drop the flag from **every** run including
these two, and note it as the branch entry 25 pre-specified. Do not keep the flag
for some blocks and not others.

The two cheapest and most expensive rates bracket the whole matrix: sequence
length is 250 samples at 100 Hz and 1250 at 500 Hz, and cost scales with it.

**Check:** seconds per epoch at both rates, then

    total ≈ 100 × (18·t₁₀₀ + 10·t₂₄₀ + 10·t₂₅₀ + 10·t₅₀₀)

against the €40 ceiling of §6.7. The counts are the ones in `configs/runs.csv`,
which is the only place they should ever be read from: 18 runs at 100 Hz — 5 in
the comparison arms plus all of blocks 0, 3 and 4 — and 10 at each of 240, 250
and 500 Hz.

`t₂₄₀` and `t₂₅₀` are interpolated between the two measured endpoints rather than
scaled from one of them. Cost grows with sequence length but not necessarily in
proportion to it: at 250 samples a 4090 is nowhere near saturated, so the short
end is dominated by per-step overhead and the linear rule overstates the cheap
rates and understates the expensive one. Measuring both ends is what removes the
assumption; that is why the probe is two epochs and not one.

Those two epochs are not throwaway. They are epoch 0 of two real runs, and the
rest resumes from them, provided the numerical settings do not change in
between. They are part of the fingerprint precisely so that they cannot: install
`pykeops` after the measurement and the resume will refuse rather than silently
mix two arithmetics.

**If the extrapolation exceeds the budget**, in this order:

1. implementation speed only — `pykeops`, DataLoader workers, batching. These
   change how fast the arithmetic runs, not what is computed, except in the last
   bits, which the fingerprint records.
2. raise the ceiling. A resource decision, recorded in §10, with no bearing on
   any conclusion.
3. only last, reduce the design. Fewer seeds changes the estimate of the null
   scale, which is the denominator of the whole comparison; fewer epochs breaks
   comparability with the 0.941 of §3. Either would be a §10 entry with its own
   justification, and neither should be reached for before the first two.

The chosen numerical settings are §10 entry 25, written before the runs rather
than after. The only thing this step can still discover is that the deterministic
branch of that entry has to be taken.

---

## 5. Stage 0 — the gate

```bash
for s in 0 1 2; do
  python scripts/train.py    --run-id b0-fs100-A-n0-s$s --deterministic
  python scripts/evaluate.py --run-id b0-fs100-A-n0-s$s --deterministic
done
python scripts/stage0.py
```

**Check:** `RIPRODUZIONE ACCETTATA`. §3 requires the 95% interval of the
three-seed mean to contain 0.941 and the mean to lie within 0.010 of it.

**If it fails, the work stops and the failure is written up** (§3, §9). Two
candidate explanations remain, in order: §10 entry 11, the training crop stride,
and entry 18, the split the paper never states. Entries 4 and 5 were withdrawn
by 16 and 17 — dataset version and signal provenance both match the reference.

---

## 6. The remaining 45 runs

Only if the gate passed. `configs/runs.csv` lists them in execution order.

```bash
python - <<'PY'
import csv, subprocess
for row in csv.DictReader(open("configs/runs.csv")):
    if row["block"] == "0":
        continue
    for step in ("train", "evaluate"):
        subprocess.run(["python", f"scripts/{step}.py",
                        "--run-id", row["run_id"], "--deterministic"], check=True)
PY
```

Interruptions are expected on spot instances and are not a problem: every run
writes a checkpoint each epoch and resumes from it exactly, which
`tests/test_train.py` verifies rather than assumes. Re-running the loop skips
what is already finished.

**Check:** the three pairs `b0-*` and `b3-*` at seeds 0–2 are identical
configurations trained separately (§10 entry 14). Compare their results. Under
forced determinism they must agree; without it, their disagreement is the
run-to-run floor at fixed seed, and either way it is a number to report.

---

## 7. Analysis

No GPU, no caches, no torch. It reads the per-record predictions written in
step 6.

```bash
python scripts/analyse.py                    # §8.1, the primary comparison
python scripts/analyse.py --analysis arms    # §8.3.1
python scripts/analyse.py --analysis notch   # §8.3.4
python scripts/analyse.py --analysis source  # §8.3.2
```

**Check:** the negative control. 240 Hz against 250 Hz discards 0.099% and
0.087% of signal power respectively, so no effect is expected there. A
significant difference in that pair means noise or a defect, not sampling rate,
and everything else has to be read against it (§8.1).

---

## 8. Bring back

`results/` only: per-epoch histories, manifests, per-label metrics, per-record
predictions, the analysis tables. A few hundred megabytes. Checkpoints stay on
the pod — they are reproducible from the manifests, and `*.pt` is in
`.gitignore` for that reason.

§9: a null result is reported with the same prominence as a positive one. If
the rate turns out not to matter at this scale, that is the answer, and it is
the one the leaderboard would have wanted to know either way.
