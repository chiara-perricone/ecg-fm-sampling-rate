# Protocol

Pre-registered experimental protocol. Written before any training run.

Everything below is fixed in advance: the architecture, the rates, the metric,
the statistical procedure, the stopping condition, and the criterion by which the
result will be called large or small. Any departure from it is recorded in §10
with a date and a reason, and nothing is removed from that section.

Companion documents: `RELATED_WORK.md` establishes why the question is open.
`src/ecgres/` implements what follows.

## 1. Question

The ECG foundation model benchmarks surveyed in `RELATED_WORK.md` evaluate each
model at its own sampling rate: 100, 240, 250 and 500 Hz appear on a single
leaderboard. Sampling rate therefore varies together with the architecture, and
no published work reports the marginal effect of the rate on a model that was
trained at that rate.

**Q1.** Holding architecture, task, window duration in seconds, optimisation
schedule and seed fixed, how far does macro AUROC on PTB-XL (all) move when only
the sampling rate changes?

**Q2.** Is that movement small or large relative to the gaps the leaderboard
treats as differences between models?

Q2 has a fixed reference. From the Reality Check's PTB-XL (all) finetuning column
(arXiv:2509.25095v2, Table 3), the ten evaluated models sort to 0.949, 0.941,
0.940, 0.934, 0.929, 0.927, 0.925, 0.915, 0.908, 0.889. Adjacent gaps are 0.008,
0.001, 0.006, 0.005, 0.002, 0.002, 0.010, 0.007, 0.019 — **median 0.006**, full
range 0.060.

## 2. What is not claimed

This work does not re-rank the foundation models and cannot: each public
checkpoint is bound to the rate it was pretrained at, and retraining eight models
at four rates is outside the compute budget. It does not assert the published
ranking is wrong. It measures the size of one uncontrolled factor and leaves the
interpretation to the reader.

## 3. Blocking condition

**Before any sampling rate is varied**, the supervised S4 baseline must be
reproduced on PTB-XL (all) at 100 Hz.

- Target: **0.941** macro AUROC, the Reality Check's own supervised result
  (arXiv:2509.25095v2, §4.1), reported there against 0.9417 from Mehari &
  Strodthoff (2023).
- Acceptance: the reproduction is accepted if the paired bootstrap 95% CI of the
  difference from 0.941 contains zero, and the point estimate is within 0.010.
- If the reproduction fails, **the work stops** and the failure is written up.
  The result of this repository is not worth reporting on top of a pipeline that
  cannot reproduce a published number.

A known failure mode to rule out first: PhysioNet's recursive `wget` can produce
silently incomplete local copies of PTB-XL, and this has been observed to differ
between machines (Berger et al., arXiv:2602.17531v2, App. B). `data.py` must
verify the download against PhysioNet's checksum files and refuse to proceed on
mismatch. An incomplete dataset would fail the blocking condition with no
indication of why.

## 4. Instrument: why S4

S4 is the supervised baseline of the benchmark under examination, its
configuration is stated in full (100 Hz, 2.5 s windows, 4 layers, state dimension
8, model dimension 512, no convolutional encoder, 2.2M parameters —
arXiv:2509.25095v2, §A.2.2), and it is small enough to train many times on free
GPU.

The decisive property is architectural. **S4 without a convolutional encoder has
a parameter count that does not depend on sequence length.** The A, B, C and D
matrices are parameterised by the state and model dimensions; the classification
head pools over time. Changing the sampling rate therefore changes the input
sequence length and nothing else — no kernel size, no stride, no patch size, no
parameter count.

This is not true of the alternatives. ECG-CPC's authors, moving that model from
100 Hz to 240 Hz, had to change the first convolutional layer to kernel size 3
and stride 2 and extend the CPC prediction horizon from 12 to 14 steps
(arXiv:2509.25095v2, §A.2.1). ST-MEM's window is quantised by a 75-sample patch
size. Any architecture with a fixed-size convolutional or patch front end
confounds the rate with the compensation required to accommodate it. S4 does not.

**One caveat, verified in the implementation.** S4's timescale parameter is in
steps, not seconds: `log_dt` is initialised log-uniform on [log 0.001, log 0.1]
(`AI4HealthUOL/ecg-fm-benchmarking@2384098`,
`code/clinical_ts/ts/s4_modules/s42.py:971–996`), and discretisation indexes
samples. At 100 Hz that range spans memory horizons of roughly 0.1 s to 10 s; at
500 Hz the same values span 0.02 s to 2 s. The model's prior over physical
timescales therefore shifts by a factor of five across the rates tested. §6
turns this into a designed contrast rather than leaving it as a confound.

## 5. Data

- PTB-XL v1.0.3 (DOI 10.13026/kfzx-aw45, CC BY 4.0), 21,799 records from 18,869
  patients, 10 s, 12 leads.
- Task: PTB-XL **(all)**, 71 SCP statements, multi-label.
- Splits: the official stratified folds — 1–8 train, 9 validation, 10 test. No
  patient appears in more than one fold.
- Source signal: **`records500` only.** Every rate, including 100 Hz, is produced
  by resampling from the 500 Hz source with one code path. The distributed
  `records100` is a downsample performed by the dataset authors with an unstated
  method; using it for the 100 Hz arm would confound rate with resampling
  implementation. This also matches the benchmark, whose `run.sh` reads
  `ptb-xl/records500` with `--fs-data 500` for every model.
- `records100` is used once, as a secondary check on resampling sensitivity
  (§8.3), never in the primary comparison.

### 5.1 What the data says before any run

Produced by `scripts/describe.py` and `scripts/psd_bands.py` on 2026-08-26,
against a copy verified file-by-file against PhysioNet's `SHA256SUMS.txt`
(87,203 files, zero mismatches). Three findings bear on the design, and the
amendments they motivated are logged in §10.

**Label support is thin.** Of the 71 labels, only **49 have ten or more positive
examples in fold 10**; 22 fall below, and two — `PRC(S)` and `2AVB` — have a
single positive. Macro-averaging weights all 71 equally, so 31% of the primary
metric would be carried by labels whose per-label AUROC is dominated by sampling
noise. Against differences of 0.005–0.010 this is not a rounding concern.

**The power that downsampling discards is small.** Mean Welch PSD over 500
training records gives the share of total signal power above each rate's Nyquist
frequency:

| Rate | Nyquist | Power discarded |
|---|---|---|
| 100 Hz | 50 Hz | 0.538% |
| 240 Hz | 120 Hz | 0.099% |
| 250 Hz | 125 Hz | 0.087% |
| 500 Hz | 250 Hz | 0% (identity) |

Consistent with the 0.01–0.03 prior from Berger et al. (§8.2). Discarded power
is not discarded discriminative information — pacemaker spikes are brief, hence
high-frequency and low-energy, so they contribute negligibly to 0.538% while
being the entire evidence for `PACE` — but a large figure would have made a
large effect expected, and this is not one.

**It is not mains interference.** PTB-XL was recorded in Germany, so mains is at
50 Hz and harmonics fall at 100, 150 and 200 Hz — inside the band the 100 Hz arm
removes. Measured against the mean density of two harmonic-free control windows
(60–90 and 110–140 Hz), the harmonics show no excess at all: 0.70×, 0.29× and
0.08× respectively. Only the 50 Hz fundamental stands out, at 5.46×. The
discarded content is therefore broadband, and the hypothesis that downsampling
to 100 Hz would remove interference rather than signal is rejected.

That last result creates a problem the design did not previously account for.
The 50 Hz fundamental is the single dominant peak in the spectrum and it sits
**exactly on the Nyquist frequency of the 100 Hz arm**, inside the transition
band of any anti-aliasing filter. Left alone, the 100 Hz arm receives partial
mains suppression that the 240, 250 and 500 Hz arms do not. The arms would then
differ in how much interference they carry as well as in temporal resolution —
the same species of confound this repository exists to quantify in others. See
§6.6 and §10.

## 6. Design

### 6.1 Rates

**100, 240, 250, 500 Hz** — the four values observed in the wild across the
Reality Check and BenchECG. Testing the rates actually in use makes Q2 directly
interpretable; testing round numbers would not.

Resampling: `scipy.signal.resample_poly` with integer up/down factors from 500 Hz
(1:5, 12:25, 1:2, 1:1). The 500 Hz arm is the identity and applies no filter, so
its anti-aliasing behaviour differs from the others by construction; this is
noted rather than corrected, since inserting a null resampling step would be its
own intervention.

### 6.2 Window

**2.5 seconds at every rate**, i.e. 250 / 600 / 625 / 1250 samples. Duration is
held constant in seconds, never in samples. This is the constraint that the
benchmark's own ST-MEM configuration violates in effect (2.4 s, being 8 whole
75-sample patches), and holding it is only possible because §4's architecture
imposes no quantisation.

Inference follows the benchmark: predictions averaged over four non-overlapping
2.5 s crops of each 10 s record (arXiv:2509.25095v2, §3.3).

### 6.3 Arms

**Arm A — naive.** `dt_min` and `dt_max` left at their defaults (0.001, 0.1) at
every rate. This is what a practitioner changing the sampling rate does, and it
is the arm that answers Q1 as asked.

**Arm B — compensated.** `dt_min` and `dt_max` scaled by 100/fs, holding the
prior over *physical* timescales constant across rates. At 100 Hz Arm B is
identical to Arm A by construction, so Arm B requires only the three non-
reference rates. It is run unconditionally, whatever Arm A shows (§6.7).

The contrast A − B isolates how much of any observed effect is the timescale
prior falling out of alignment, as opposed to information genuinely lost or
gained at the input. That is the same quantity ECG-CPC's authors compensated for
without measuring.

### 6.4 Seeds and the null scale

**Five seeds per cell**, fixed and recorded. The benchmark deliberately does not
fix a global seed (arXiv:2509.25095v2, §3.3); this work does, because separating
a rate effect from run-to-run variation is the entire point.

Within-rate seed spread is the natural null scale for Q2, and reporting it is a
primary output, not a diagnostic. The headline comparison is **between-rate
variation against within-rate variation**: if changing the sampling rate moves
macro AUROC no further than reseeding does, the factor is not worth controlling,
and that is a publishable answer.

### 6.5 Fixed across all runs

Optimiser AdamW, learning rate 1e-3, weight decay 1e-3, batch size 64, 100
epochs, model selection on fold 9 by macro AUROC, loss binary cross-entropy —
all following arXiv:2509.25095v2, §3.3. Layer-dependent learning rates are not
used, since the S4 baseline there is trained from scratch rather than finetuned.

### 6.6 Mains filtering

*Added 2026-08-26; see §5.1 and §10.*

**Comparison arms (Stage 1 and 2): a 50 Hz notch is applied at every rate**, as
fixed preprocessing, before resampling. Second-order IIR notch, Q = 30, applied
identically to all four arms.

The reason is in §5.1: the 50 Hz mains fundamental is the dominant spectral peak
and sits on the 100 Hz arm's Nyquist frequency. Without a notch, that arm gets
partial mains suppression for free while the others keep the interference, and
any measured difference would mix "less interference" with "less temporal
resolution". Removing the fundamental everywhere leaves sampling rate as the only
thing that varies, which is the entire point of the design.

**Blocking reproduction (Stage 0): no notch.** The 0.941 target was produced
under the benchmark's stated preprocessing, which applies no additional
filtering. Reproducing it requires matching that, not improving on it. The two
stages do not conflict because Stage 0 is a separate comparison against a
published number, not a member of the rate family.

This is a real cost, stated plainly: the Stage 1 arms are no longer directly
comparable to the published 0.941, because they carry one preprocessing step it
did not. That is the correct trade. Stage 0 establishes that the pipeline can
reproduce the literature; Stage 1 answers the question the repository asks, and
answering it cleanly requires controlling the confound that §5.1 uncovered.

An unnotched 100 Hz arm is retained as a secondary analysis (§8.3.4) to quantify
what the notch decision itself is worth.

### 6.7 Compute

All 43 runs below are committed in advance and executed on rented GPU. Neither
arm is conditional on the other's result.

| Block | Cells | Runs |
|---|---|---|
| 0 — blocking check (§3) | 100 Hz × 3 seeds | 3 |
| 1 — Arm A | 4 rates × 5 seeds | 20 |
| 2 — Arm B | 3 rates × 5 seeds | 15 |
| 3 — unnotched 100 Hz (§8.3.4) | 1 cell × 5 seeds | 5 |

**Why rented rather than free.** Kaggle's free tier would fit this workload only
by spreading it across several weeks, which in turn creates pressure to make
Arm B conditional on Arm A's outcome and to trim seeds at the expensive rates.
Both are analytic decisions taken after seeing data, and both are the kind of
thing this protocol exists to prevent. Paying for a few days of GPU removes the
pressure. The cost is not a constraint here — S4 is a 2.2M-parameter model.

**Sizing.** Cost scales with sequence length, which scales with rate: 250 / 600 /
625 / 1250 samples at 2.5 s. Training is fp32, following the benchmark's
`--precision 32` for S4. Folds 1–8 give roughly 17.4k records, batch 64, 100
epochs, so on the order of 27k optimisation steps per run. On a consumer card
with strong fp32 throughput (RTX 4090 class, ~€0.35/h on Vast.ai or RunPod), the
estimate is under an hour per run at 100 Hz and a few hours at 500 Hz, putting
all 43 runs at roughly 20–30 GPU-hours and **well inside the €40 budget**, with
several times the headroom needed if the estimate is off.

**These are estimates, so they get calibrated before anything is rented at
scale.** The first blocking-check run (§3) doubles as the pilot: it establishes
wall-clock per epoch at 100 Hz, from which the other three rates are extrapolated
by sequence length. If the extrapolated total exceeds €40, the shortfall is
resolved by choosing a cheaper card or a longer rental — **not** by cutting seeds
or dropping Arm B, and any such decision is recorded in §10.

**Operationally**, all runs happen in as few sessions as possible on persistent
storage, since PTB-XL `records500` is several GB and re-downloading it per
session wastes more time than it saves. Checkpoints are written every epoch so an
interrupted instance costs one epoch rather than one run. Spot pricing is
acceptable given checkpointing; if interruptions prove frequent, on-demand is
used instead, which changes the cost and nothing else.

**Data handling.** PTB-XL is CC BY 4.0 and is the only dataset used. The rented
instance holds nothing else.

## 7. Metrics

*Primary endpoint amended 2026-08-26; see §5.1 and §10.*

- **Primary endpoint for the rate comparison (§8.1): `macro_clean`**, the macro
  AUROC over the 49 labels with ten or more positive examples in fold 10.
- **Primary endpoint for the blocking reproduction (§3): `macro_all`**, over all
  71 labels. The published 0.941 is defined on the full label set and has to be
  compared as published.
- Both are reported at every rate, always, so the effect of the choice is visible
  rather than asserted.
- **Why they differ.** §5.1 measures 22 of 71 labels below the ten-positive
  threshold, two of them with a single positive. Macro-averaging weights all
  labels equally, so under `macro_all` roughly a third of the metric is carried
  by estimates whose sampling variance is very large. The differences being
  measured here are 0.005–0.010; a metric that noisy on a third of its terms
  cannot resolve them. Berger et al. (§5.4.2, Tables 4, 23–25) show that removing
  low-support labels shifts macro AUROC enough to reorder methods.
- **The label set is frozen.** Computed once from fold 10 by
  `scripts/describe.py` and committed as `results/frozen_label_set.json` before
  any training run, so it is identical across rates and cannot be influenced by
  a result.
- **Per-label AUROC and AUPRC** reported for every run in `results/`, not only
  the macro summaries (Berger et al., arXiv:2602.17531v2, §4).
- **Undefined labels.** A label with no positive or no negative example in a
  bootstrap resample has undefined AUROC. `src/ecgres/metrics.py` excludes it
  from that resample's macro average rather than imputing 0.5, and the exclusion
  count is reported.

## 8. Analysis

### 8.1 Primary comparison

All six pairwise rate comparisons within Arm A, on the shared fold 10 test set.

- **Paired** bootstrap over test records, 10,000 resamples, the same resample
  indices applied to every rate.
- 95% percentile confidence intervals on the paired difference.
- **Holm–Bonferroni** across the family of six comparisons. Holm is used rather
  than the Bonferroni correction of Berger et al. because it controls the same
  family-wise error rate with uniformly greater power.
- Implemented in `src/ecgres/stats.py`, already tested.

Seed variation is handled by averaging predictions across seeds within a rate
before the paired bootstrap, and separately by reporting the seed-level spread
per rate. Both are reported; neither is chosen after seeing the result.

**Negative control: 240 Hz versus 250 Hz.** *Added 2026-08-26; see §10.*

§5.1 measures the power discarded at these two rates as 0.099% and 0.087% — a
difference of about a hundredth of a percent of total signal power, and both
Nyquist frequencies sit in a region where the spectrum is already near the noise
floor. Physically, this pair should show no effect.

It therefore serves as a calibration of the null within the experiment itself.
If the paired bootstrap finds a significant difference between 240 and 250 Hz,
the finding is noise, a pipeline defect, or an artefact of the resampling
implementation — not sampling rate. Any effect claimed elsewhere in the family
has to be read against this pair.

This costs nothing: both rates are already in the design for other reasons. It
is declared here rather than noticed afterwards, which is the only way a control
of this kind is worth anything.

### 8.2 Pre-specified interpretation

Let Δ be the largest paired difference in macro AUROC between any two rates in
Arm A, with its Holm-adjusted 95% CI.

- **Δ CI contains zero**, or Δ is no larger than the within-rate seed spread →
  the factor is not detectable at this scale. Reported as such, without
  reaching for a subgroup or an alternative metric.
- **Δ below 0.006** (the median adjacent leaderboard gap, §1) → detectable but
  smaller than a typical gap between adjacent models.
- **Δ at or above 0.006** → comparable to the differences the leaderboard treats
  as differences between models.
- **Δ at or above 0.019** (the largest adjacent gap) → capable of spanning any
  gap in the ranking.

These thresholds are fixed now. They will not be adjusted after the numbers are
seen.

The prior from the nearest published measurement is that Δ will be small: Berger
et al. report a standard deviation of 0.01–0.03 across 18 randomly initialised
ResNet configurations crossing three sampling rates with filtering and
normalisation (§5.4.4, Table 6). That result is for untrained encoders under
linear probing with the rate not isolated, so it bounds rather than answers the
question — but a Δ far outside 0.01–0.03 would warrant checking the pipeline
before writing it up.

### 8.3 Secondary analyses

Declared now so they are not mistaken later for the primary result.

1. **Arm A versus Arm B** at 240, 250 and 500 Hz: how much of Δ is the timescale
   prior rather than the signal.
2. **`records100` versus resampled-to-100 Hz**: whether the dataset authors'
   downsample and `resample_poly` give different macro AUROC. A difference here
   is about resampling implementation, not about rate.
3. **Per-label breakdown** of where any effect concentrates — whether it falls on
   high-frequency morphology labels, as a Nyquist argument would predict.
   `PACE` is the label to watch: pacemaker spikes are brief, so they carry
   almost no energy but everything the label depends on.
4. **Notched versus unnotched at 100 Hz** (added 2026-08-26): one extra cell,
   5 seeds, Arm A at 100 Hz without the §6.6 notch. Quantifies what the notch
   decision is worth, and whether the mains fundamental sitting on that arm's
   Nyquist frequency mattered at all.

Analysis 3 is exploratory and will be labelled as such. It is the one most likely
to produce a compelling post-hoc story, which is exactly why it is fenced off
here.

## 9. Outputs

- `results/` — per-run predictions on fold 10, per-label metrics, run configs,
  seeds, and library versions. Predictions are committed so the analysis is
  re-runnable without a GPU.
- `notebooks/` — the analysis producing every figure and number in the write-up.
- A results section in `README.md` reporting Δ with its CI, the within-rate seed
  spread, and the §8.2 verdict, whichever way it falls.

A null result is reported with the same prominence as a positive one. The
question is worth answering in either direction, and a repository that only
publishes when the answer is interesting is not evidence of anything.

## 10. Deviations

Entries are appended with the date and the reason. **Nothing is removed from this
section.** If a decision in §1–§9 turns out to be wrong or unworkable, the change
is recorded here rather than edited into the protocol above.

Where an amendment does change the body text, the affected section carries a
dated marker pointing here, so the current text and its history are both
readable. All three entries below predate any training run: they follow from
descriptive analysis of the inputs (§5.1), not from any result.

| Date | Section | Change | Reason |
|---|---|---|---|
| 2026-08-26 | §7 | Primary endpoint for the rate comparison changed from `macro_all` (71 labels) to `macro_clean` (49 labels with ≥10 positives in fold 10). `macro_all` retained as the endpoint for the §3 blocking reproduction, and both reported everywhere. | `scripts/describe.py` measures 22 of 71 labels below the threshold, two with a single positive. Under `macro_all`, ~31% of the metric is carried by estimates whose sampling variance is very large, against target differences of 0.005–0.010. The 0.941 reproduction still uses `macro_all` because the published figure is defined on all 71 labels. |
| 2026-08-26 | §6.6 (new) | A 50 Hz notch (IIR, Q=30) is applied at every rate in the comparison arms, before resampling. Stage 0 keeps the benchmark's unfiltered preprocessing. Unnotched 100 Hz added as secondary analysis §8.3.4. | `scripts/psd_bands.py` shows the 50 Hz mains fundamental is the dominant spectral peak (5.46× the harmonic-free control density) and sits exactly on the 100 Hz arm's Nyquist frequency. Unnotched, that arm would receive partial mains suppression the other arms do not, mixing "less interference" with "less temporal resolution" — the confound this work exists to measure in others. Cost acknowledged: Stage 1 arms are no longer directly comparable to the published 0.941. |
| 2026-08-26 | §8.1 | 240 Hz vs 250 Hz declared as an internal negative control. | Power discarded at the two rates is 0.099% and 0.087%; physically no effect is expected. A significant difference there would indicate noise or a pipeline defect, calibrating the null scale for the rest of the family. Costs no additional runs. |
