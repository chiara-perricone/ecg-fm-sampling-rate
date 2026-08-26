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
- Acceptance *(made executable 2026-08-26; see §10)*: block 0 is run at three
  seeds (§6.7), and the point estimate is the **mean of the three per-seed
  `macro_all` values on fold 10**. Records are resampled with replacement 10,000
  times, the same resample being given to all three seeds, and the 95%
  percentile interval is taken over the resampled means. The reproduction is
  accepted if that interval contains 0.941 **and** the mean is within 0.010 of
  it. The three per-seed values and their spread are reported beside the
  interval, not inside it: the interval is over records only, and with three
  seeds the between-seed component cannot honestly be folded into it.
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

*Model selection disambiguated 2026-08-26; see §7 and §10.*

Optimiser AdamW, learning rate 1e-3, weight decay 1e-3, batch size 64, 100
epochs, loss binary cross-entropy — all following arXiv:2509.25095v2, §3.3.
Layer-dependent learning rates are not used, since the S4 baseline there is
trained from scratch rather than finetuned.

**Model selection** is on fold 9, by the macro AUROC that is the endpoint of the
stage: `macro_all` for the blocking reproduction of §3, `macro_clean` for the
rate comparison of §8.1–§8.2. The checkpoint kept is the epoch maximising that
criterion. Checkpoints are additionally written every epoch, for interruption
recovery, which is a separate concern from selection.

Fold 9 is evaluated exactly as fold 10 is: the four non-overlapping 2.5 s crops
tiling the record, predictions averaged per record. The selection criterion and
the reported metric are therefore the same quantity computed on different folds,
not two differently constructed estimates.

**Both** macro AUROCs are computed on fold 9 at every epoch of every run and
written to the run history, together with the epoch each criterion would have
selected. How often the two disagree is then a measured quantity rather than an
assumption.

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

*Duplicate configurations recorded, and block 4 added, 2026-08-26; see §10
entries 14 and 23. Calibration procedure changed and the stale run count
corrected the same day; see entry 26.*

All 48 runs below are committed in advance and executed on rented GPU. Neither
arm is conditional on the other's result.

| Block | Cells | Runs |
|---|---|---|
| 0 — blocking check (§3) | 100 Hz × 3 seeds | 3 |
| 1 — Arm A | 4 rates × 5 seeds | 20 |
| 2 — Arm B | 3 rates × 5 seeds | 15 |
| 3 — unnotched 100 Hz (§8.3.4) | 1 cell × 5 seeds | 5 |
| 4 — `records100` at 100 Hz (§8.3.2) | 1 cell × 5 seeds | 5 |

Three runs of block 3 duplicate three runs of block 0 exactly — 100 Hz, Arm A,
no notch, seeds 0–2 — differing only in the fold 9 metric on which the
checkpoint is selected (§6.5). **They are trained separately, not shared.** At
equal seed the two must agree, so the three pairs are an end-to-end check that
the pipeline is reproducible, which is the assumption the interruptible-instance
strategy below relies on and which nothing else in the design tests. Their
agreement is reported in `results/` whatever it turns out to be.

Block 4 is *not* a further duplicate of that cell: it shares rate, arm and the
absence of a notch with block 3, but reads `records100` instead of resampling
`records500`, which is precisely the difference §8.3.2 exists to measure.

Runs are enumerated in `configs/runs.csv`, generated by `scripts/make_runs.py`
and committed before any of them is executed. The block counts above, the
duplication, and the selection rule of §6.5 are pinned to that file by
`tests/test_runs.py`.

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
all 48 runs at roughly 20–30 GPU-hours and **well inside the €40 budget**, with
several times the headroom needed if the estimate is off. *The count read 43
until entry 23 added block 4; the hour range predates that entry and was not
recomputed, which is one more reason it is calibrated by measurement before
anything is rented at scale.*

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
per rate. Both are reported; neither is chosen after seeing the result. Note
that §3 aggregates seeds by averaging *metrics* rather than predictions, for the
reason given in §10 entry 19; the asymmetry is deliberate.

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

*Reporting of intervals made executable 2026-08-26; see §10 entry 20.*

Let Δ be the largest paired difference in macro AUROC between any two rates in
Arm A. Three quantities are reported for every pairwise comparison, never one
chosen after the fact: the unadjusted 95% percentile interval of the difference,
the Holm-adjusted p-value across the family of six, and a Bonferroni-simultaneous
percentile interval at α/12 and 1 − α/12, marked as conservative. The thresholds
below apply to Δ itself, not to any interval.

- **Δ CI contains zero**, or |Δ| is no larger than the **internal null
  contrast** defined below → the factor is not detectable at this scale.
  Reported as such, without reaching for a subgroup or an alternative metric.
- **Δ below 0.006** (the median adjacent leaderboard gap, §1) → detectable but
  smaller than a typical gap between adjacent models.
- **Δ at or above 0.006** → comparable to the differences the leaderboard treats
  as differences between models.
- **Δ at or above 0.019** (the largest adjacent gap) → capable of spanning any
  gap in the ranking.

These thresholds are fixed now. They will not be adjusted after the numbers are
seen.

**The internal null contrast** *(added 2026-08-26; see §10 entries 21 and 22,
the second of which corrects the first)*. Δ is a difference between two
five-seed ensembles, so the quantity it is measured against has to be one too.
Within each rate the seeds are split into two disjoint subsets **of equal size**
— 2 against 2, fifteen pairs, one seed left out each time — and the same
ensemble difference is computed as between rates; the comparator is the median
absolute value, pooled across rates. Equal sizes matter: unequal subsets measure
how much better a larger ensemble is, which is a real effect of fixed sign and
nothing to do with seeds.

The comparator is used **raw and declared as an upper bound**. A contrast
between two disjoint ensembles of k has standard deviation σ√(2/k), so a
2-against-2 contrast exceeds the seed-noise floor of a five-against-five Δ by
about √(5/2) ≈ 1.58; an exactly matched contrast would require ten seeds per
rate. The rescaled value is reported beside the raw one, and the rule of §8.2
uses the raw one, which makes the rule conservative by a known factor in a known
direction. The per-rate seed standard deviation and range are reported as well.

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
   is about resampling implementation, not about rate. *Answered by block 4
   against block 3 (§6.7), both unnotched at 100 Hz and differing only in signal
   provenance; added 2026-08-26, see §10 entry 23.*
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
readable. All entries below predate any training run. Entries 1–3 follow from
descriptive analysis of the inputs (§5.1); entries 4–8 from validation of the
data pipeline against the full dataset, from inspection of the reference
implementations, and from design decisions taken while implementing the loader;
entries 9–15 from reading the vendored S4 layer and the reference training
wrapper line by line, and from design decisions taken while implementing the
model, the normalisation, the run matrix and the evaluation. Entry 14 is a
property of the matrix that was noticed while writing it down, and is recorded
because it was almost removed; entry 15 makes §3 executable, an omission found
only when the code that had to execute it was about to be written. Entries
16–18 close the two verification tasks left open by entries 4 and 5, against the
sources rather than by inference, and record what those sources turned out not
to say. Two of them withdraw a cost this protocol had assumed against itself.
Entries 19 and 20 do for §8 what entry 15 did for §3: they were written while
the analysis code was being prepared, and they replace two phrases that could
not have been executed as written; entry 22 corrects entry 21, which was wrong,
and was caught by running the analysis on synthetic predictions before any real
one existed. Entry 23 closes a gap between §6.7 and §8.3 by adding runs rather
than by weakening the question. None follows from any result.

| # | Date | Section | Change | Reason |
|---|---|---|---|---|
| 1 | 2026-08-26 | §7 | Primary endpoint for the rate comparison changed from `macro_all` (71 labels) to `macro_clean` (49 labels with ≥10 positives in fold 10). `macro_all` retained as the endpoint for the §3 blocking reproduction, and both reported everywhere. | `scripts/describe.py` measures 22 of 71 labels below the threshold, two with a single positive. Under `macro_all`, ~31% of the metric is carried by estimates whose sampling variance is very large, against target differences of 0.005–0.010. The 0.941 reproduction still uses `macro_all` because the published figure is defined on all 71 labels. |
| 2 | 2026-08-26 | §6.6 (new) | A 50 Hz notch (IIR, Q=30) is applied at every rate in the comparison arms, before resampling. Stage 0 keeps the benchmark's unfiltered preprocessing. Unnotched 100 Hz added as secondary analysis §8.3.4. | `scripts/psd_bands.py` shows the 50 Hz mains fundamental is the dominant spectral peak (5.46× the harmonic-free control density) and sits exactly on the 100 Hz arm's Nyquist frequency. Unnotched, that arm would receive partial mains suppression the other arms do not, mixing "less interference" with "less temporal resolution" — the confound this work exists to measure in others. Cost acknowledged: Stage 1 arms are no longer directly comparable to the published 0.941. |
| 3 | 2026-08-26 | §8.1 | 240 Hz vs 250 Hz declared as an internal negative control. | Power discarded at the two rates is 0.099% and 0.087%; physically no effect is expected. A significant difference there would indicate noise or a pipeline defect, calibrating the null scale for the rest of the family. Costs no additional runs. |
| 4 | 2026-08-26 | §3 | Recorded: the blocking reproduction target (0.941 macro AUROC) is defined on a test set that differs from the one used here. PTB-XL v1.0.3, verified file-by-file against `SHA256SUMS.txt`, contains 21,799 records; fold 10 contains 2,198. | `get_datasets.sh` in `helme/ecg_ptbxl_benchmarking` renames the downloaded directory `ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1`, i.e. it fetches v1.0.1 (21,837 records); the repository's finetuning notebook links the same version. The 0.9417 of Mehari & Strodthoff 2023, against which the Reality Check authors compare their 0.941, is therefore defined on a population differing from ours by 38 records (0.17%). Not verified directly for the 0.941 itself — inferred from that comparison. The §3 criterion requires a point estimate within 0.010, so this is a systematic offset of unknown sign, plausibly well below threshold; it is the first thing to examine if Stage 0 fails narrowly. The rate comparison is internal and holds the population fixed, so it is unaffected. |
| 5 | 2026-08-26 | §4 | Recorded: `records500` is the sole signal source for every arm, including the 100 Hz arm and the §3 blocking reproduction. `records100` is not used except in secondary analysis §8.3.2. | PhysioNet's official example script, from which the `ecg_ptbxl_benchmarking` pipeline derives, selects `filename_lr` when `sampling_rate == 100` and `filename_hr` otherwise; `tmehari/ssm_ecg` operates on pre-processed directories named `ptb_xl_fs100`. The distributed `records100` files were produced with an anti-aliasing filter that is not `resample_poly`'s, so the two 100 Hz signals are not identical. Verified for the PhysioNet example; probable but not directly verified for `utils.load_dataset`. Adopting `records100` for the 100 Hz arm only would introduce a difference in provenance superimposed on the difference in rate — the exact confound this design exists to exclude — so fidelity to the external reference is subordinated to cleanliness of the internal comparison. Second systematic offset from 0.941; examine immediately after entry 4 if Stage 0 fails narrowly. §8.3.2 quantifies the effect. |
| 6 | 2026-08-26 | §6.6 | The 50 Hz notch of entry 2 is applied zero-phase, via `scipy.signal.filtfilt`, at 500 Hz before resampling. | A causal IIR filter (`lfilter`) introduces a frequency-dependent phase shift. Applied to arms subsequently resampled to different rates, it would produce a temporal misalignment between views: `assert_aligned` would still pass on structure, but the arms would no longer be the same signal observed at different resolutions. Cost acknowledged: `filtfilt` is non-causal, so the pipeline is not transferable to a streaming or real-time setting without revisiting this choice. Not an operational constraint here — the work is offline on complete 10 s records — but stated for anyone reusing `src/ecgres/data.py`. |
| 7 | 2026-08-26 | §6 | Normalisation replicates the reference pipeline: a single `StandardScaler` fitted on flattened values, i.e. one mean and one standard deviation shared across all 12 leads and all timesteps — not per-lead. | In `ecg_ptbxl_benchmarking`, `apply_standardizer` calls `ss.transform(x.flatten()[:, np.newaxis])`; the record is flattened to a single column before transform. The §3 criterion requires reproducing 0.941, and every departure from the reference pipeline both lowers the chance of success and makes a failure ambiguous to interpret, so the more natural per-lead choice is subordinated to this. Related observation, reported by `scripts/build_cache.py`: 14 of 21,799 records (0.06%) exceed 20 mV, six of them above 30 mV, ranging 30.4–31.2 mV at 100 Hz. The int16 full scale at gain 1000 ADU/mV is 32.767 mV, so these records saturate the converter; at 240, 250 and 500 Hz peaks exceed full scale (up to 34.4 mV) through filter overshoot around the saturation steps. Because the scaler is global, these records contribute to the standard deviation used to normalise every other record. They are not excluded: exclusion would itself be a departure from the reference protocol and would require justification, and 14 records in 21,799 are not expected to move macro AUROC measurably. |
| 8 | 2026-08-26 | §6 | The `StandardScaler` of entry 7 is fitted once, on the training folds (1–8) at 500 Hz, and the resulting (mean, sd) pair is applied unchanged to every arm of the comparison. Not refitted per arm. | Resampling alters signal variance, and no choice of scaler removes that: fitting per arm absorbs the difference into the normalisation constants, fitting once leaves it in the post-normalisation signal. The covariate cannot be eliminated, only placed. It is placed in the signal, for two reasons. First, per-arm fitting would partially cancel the very effect under study — if lower rates carry less variance and the scaler rescales each arm to unit variance, part of the rate effect is divided out before the model sees it. Second, per-arm constants would be a rate-dependent quantity chosen by the pipeline rather than by the data, and therefore harder to reason about than a variance difference that is a property of resampling. 500 Hz is chosen as the fitting rate because it is the source from which every arm is derived; fitting at 100 Hz would be defensible for fidelity to the reference, but 100 Hz is itself one of the competing arms and giving it reference status is a gratuitous asymmetry. Cost acknowledged: Stage 0 uses the reference pipeline's own scaler, so Stage 1 arms are not directly comparable to the published 0.941 — the same cost already accepted for the notch in entry 2. Alternative rejected: per-arm fitting, i.e. maximum fidelity to the reference pipeline at each rate. |
| 9 | 2026-08-26 | §6 Arms | Recorded: the A/B contrast does not isolate the prior over timescales alone. It also carries the length correction applied to C at initialisation. | The vendored S4 file (state-spaces, commit 137e49d) uses `length_correction=True`, which multiplies C by `I - dA^L` at init, with `dA` a function of `dt` and `L = window_seconds * fs`. The initialisation of C therefore depends on `dt * L`. In Arm B, `dt * L = dt_default * window_seconds * 100` is constant across rates; in Arm A it scales with fs. Because `dt * L` and `dt * fs` differ only by the constant `window_seconds`, the two are not separable: A−B remains a comparison with a single degree of freedom, but the correct description of it is "everything that depends on dt expressed in steps", not "the prior over timescales". Verified pre-run in `test_dt_times_length_is_constant_in_arm_b` and `test_arm_a_kernel_init_drifts_with_rate`. |
| 10 | 2026-08-26 | §6 | The S4 layer is vendored from `HazyResearch/state-spaces` commit 137e49d (Apache-2.0), not reimplemented from scratch. | Provenance verified rather than assumed: the copy in `tmehari/ssm_ecg` (MIT, same group as the Reality Check) differs from upstream commit 2bb0858 by 17 lines, all identifiable as a URL comment, a torch version-check fix, and their own `# MODIFIED` rate passthrough. Commit 137e49d was chosen over 2bb0858 because it adds a CPU fallback for the Cauchy kernel and fixes the `torch.__version__.startswith('1.10')` check, which is wrong under torch 2.x. Diffing 2bb0858 against 137e49d, excluding the Cauchy backend and version-check plumbing, leaves only a docstring and a reordering: `dt_min`, `dt_max`, `resample`, `self.rate` and `dt = exp(log_dt) * rate` are line-for-line identical, so the model mathematics is unchanged. `helme/ecg_ptbxl_benchmarking` is GPL-3.0 and therefore excluded from reuse in this MIT repository; `AI4HealthUOL/ecg-fm-benchmarking` carries no licence at all. Only modification to the vendored file: the `pytorch_lightning` import replaced by a no-op, verified line-by-line against upstream. |
| 11 | 2026-08-26 | §6 | Training crops are drawn at random on the 0.1 s grid, not enumerated deterministically at stride `input_size // 4` as in the reference. | `ecg_dataset_wrapper.py` sets `stride_length_train = chunk_length_train // 4`, giving ~13 overlapping deterministic crops per record. Integer division breaks cross-arm alignment: `250 // 4 = 62` is 0.620 s while `1250 // 4 = 312` is 0.624 s, so arms would see physically different windows — the confound this design exists to exclude. Random crops on the 0.1 s grid preserve the pairing on which the paired bootstrap of §8.1 depends. Note that inference crops do coincide with the reference: `stride_valid = input_size`, four non-overlapping windows, predictions averaged. Third candidate explanation if Stage 0 fails narrowly, after entries 4 and 5. |
| 12 | 2026-08-26 | §6 | The scaler of entries 7–8 is fitted on the **full 10 s records** of folds 1–8, not on the training crops the model actually sees. | Entries 7 and 8 fix what the scaler is (one global scalar pair) and where it is fitted (folds 1–8, 500 Hz, once), but not the unit of observation, and the two candidates are not equivalent under entry 11. Training crops are now drawn at random per epoch, so a scaler fitted on them would be a function of the seed and of the epoch schedule: each of the 43 runs would normalise by slightly different constants, which is precisely what entry 8 exists to prevent — the comparison across rates would then rest on constants chosen per cell rather than shared. The reference pipeline also standardises whole records, before chunking. Cost acknowledged: random crops on the 0.1 s grid do not sample record positions uniformly (a 2.5 s window starting anywhere in 0–7.5 s covers the middle of the record more often than the edges), so the training-window distribution is not exactly the record distribution and the fitted constants are, strictly, those of a slightly different population. Two reasons this is accepted: within a 10 s resting ECG the signal is close to stationary, so a global scalar mean and standard deviation are nearly position-independent; and at inference the four crops tile the record exactly (entry 11), so there the two populations coincide by construction. The fitted pair is serialised with the fitting rate, the folds, the record count and a fingerprint of the record order, so the population behind the constants is recoverable after the fact. Verified pre-run in `test_scaler_is_global_not_per_lead`, `test_scaler_fit_uses_only_the_selected_records` and `test_fit_from_cache_refuses_other_rates`. |
| 13 | 2026-08-26 | §6.5 | Model selection disambiguated: the checkpoint is chosen on fold 9 by the macro AUROC that is the endpoint of the stage — `macro_all` for the §3 reproduction, `macro_clean` for the §8.1–§8.2 comparison. Both are computed every epoch of every run and recorded, together with the epoch each criterion would have selected. Fold 9 is evaluated with the same four-crop aggregation used on fold 10. | §6.5 read "model selection on fold 9 by macro AUROC". That was unambiguous while there was one macro AUROC and stopped being so with entry 1. Left unresolved, the choice would have been made in effect by whichever metric the training code happened to compute first. `macro_all` is kept for Stage 0 because the reference selects on the macro over the full label set and the published 0.941 was produced that way; the blocking condition already carries three candidate explanations for a narrow failure (entries 4, 5, 11), and a fourth of our own making would leave a failure close to uninterpretable. `macro_clean` is used for the comparison because `macro_all` is a noisy *selection* criterion for exactly the reason it was rejected as an *endpoint*: 22 of 71 labels sit below ten positives and fold 9 is the size of fold 10, so roughly a third of the criterion carries very large sampling variance. Taking the maximum over 100 epochs of a noisy quantity both biases the selected value upward and makes the argmax close to arbitrary among near-tied epochs; that arbitrariness enters each run as an independent noise term, widening both the paired bootstrap intervals of §8.1 and the between-seed spread of §6.4 — which is the resolution this design exists to buy. Objection considered and rejected: selecting on `macro_clean` makes the stopping point depend on a label set defined by support in fold 10, the test fold. The information that crossed from fold 10 is the list of labels with at least ten positives, computed once, frozen and committed before any training run (§7); reusing that same list does not spend more of it, and no model output on fold 10 enters selection at any point. Cost acknowledged: two selection rules instead of one, so the comparison arms are selected differently from the Stage 0 run against which the pipeline is validated. Recorded here rather than smoothed over, and made auditable by logging both criteria for every epoch of every run. |
| 14 | 2026-08-26 | §6.7 | Recorded: three runs of block 3 duplicate three runs of block 0 exactly (100 Hz, Arm A, no notch, seeds 0–2), differing only in the selection metric of entry 13. The duplication is retained deliberately and the agreement of each pair is reported. The run matrix is committed in advance as `configs/runs.csv`. | Both blocks train at 100 Hz, Arm A, without the notch of entry 2, on the same folds with the same schedule of §6.5. The model has 71 outputs in every run and `macro_clean` is a subset evaluated afterwards, not a different task, so at equal seed the two blocks should produce identical weights at every epoch. Before entry 13 they were the same run written twice in the §6.7 table; after it they differ only in which checkpoint is kept from a training history, so three trainings would have sufficed. They are nevertheless run separately. §6.7 trains on interruptible instances and §6.5 writes a checkpoint every epoch, so the entire compute plan rests on the assumption that an interrupted-and-resumed run equals an uninterrupted one — and that assumption is untested unless two independent executions of an identical configuration are compared somewhere. The three pairs are that comparison, and the design already pays for them. Their outcome is informative under either numerical-determinism setting: with deterministic algorithms forced the pairs must agree bit-for-bit, and a disagreement points at seeding, at the crop sampler or at data ordering; without, the pairs measure the run-to-run variation remaining at fixed seed, which is the floor of the null scale of §6.4 and a quantity the analysis wants in its own right. Alternative rejected: sharing the training and selecting two checkpoints from one history. It saves roughly ninety minutes at the cheapest rate, introduces a distinction between runs and trainings into the training loop — the part of the code where an undetected defect is most expensive — and deletes the only end-to-end reproducibility check the plan contains. |
| 15 | 2026-08-26 | §3 | The acceptance criterion is made executable. "Paired bootstrap of the difference from 0.941" becomes a record-level bootstrap of the three-seed mean `macro_all`, accepted if the 95% percentile interval contains 0.941 and the mean lies within 0.010 of it. Per-seed values and their spread are reported separately. | A paired bootstrap needs two prediction vectors over the same records; of the reference result only a published scalar exists, so there is nothing to pair with. With 0.941 treated as a constant, "the interval of the difference contains zero" and "0.941 lies in the interval of our estimate" are arithmetically the same statement, so the intent of §3 survives intact; what does not survive is the word *paired*, and the interval reflects our sampling variability alone, not the reference's. §3 was also written as though one number were being compared, while §6.7 assigns three seeds to block 0. The mean of the three is used because a decision that can stop the work should not rest on a single draw, and because requiring all three to pass individually would close the gate roughly one time in seven on noise alone at a nominal 95% level — §3 says the work stops in that case, which is too consequential to leave to a multiplicity artefact. Records are resampled once per replicate and the same resample is given to all three seeds, since the seeds share fold 10 and that is the only genuine pairing available anywhere in this comparison. Stated against the temptation to claim more: the interval is over records only. Within a replicate the three trained models are fixed, so the between-seed component is never resampled and is not inside the interval; averaging three runs stabilises the estimate without representing that variability, which is therefore reported beside the interval rather than absorbed into it. Entry 4 remains the first thing to examine if the criterion fails narrowly, the population under test differing from the reference's by 38 records. |
| 16 | 2026-08-26 | §3 | Verified: the 0.941 target is defined on PTB-XL v1.0.3, the same 21,799 records used here. The systematic offset assumed in entry 4 does not exist, and that entry's inference is superseded. | Entry 4 inferred a v1.0.1 population of 21,837 records from `get_datasets.sh` in `helme/ecg_ptbxl_benchmarking`. The inference was drawn from the wrong codebase. The repository behind 0.941 is `AI4HealthUOL/ecg-fm-benchmarking`, whose README directs the reader to `physionet.org/content/ptb-xl/1.0.3/` and whose paper cites Wagner et al. (2022), "PTB-XL, a large publicly available electrocardiography dataset (version 1.0.3), doi:10.13026/kfzx-aw45". `helme/ecg_ptbxl_benchmarking` produced the 0.9417 of Mehari & Strodthoff — the figure the Reality Check compares itself *against*, not the target of §3. Table 2 of the paper reports 21,799 samples for PTB-XL (all/sub/super), which is exactly the population verified file-by-file here against `SHA256SUMS.txt`. One caveat is closed by our own measurement rather than by the paper: that table calls 21,799 an *effective* sample size, meaning records carrying at least one positive label rather than the dataset total. `scripts/build_cache.py` verifies that every record in v1.0.3 carries at least one label, so for PTB-XL (all) the effective and total counts coincide. Consequence for §3: the first of the three candidate explanations for a narrow failure is withdrawn, which makes the blocking condition stricter rather than easier to pass. |
| 17 | 2026-08-26 | §4 | Verified: the pipeline behind 0.941 reads `records500` and resamples to each model's rate, including for the 100 Hz S4 baseline. Entry 5's choice therefore agrees with the reference instead of departing from it, and the second assumed offset does not exist. Recorded separately: the paper states that no resampling is performed, while its own configuration requires it. | `run.sh` in `AI4HealthUOL/ecg-fm-benchmarking` sets `--data ${DATASET_DIR}/ptb-xl/records500` and `--fs-data 500` for `ptbxl_all`, and gives the S4 baseline `--fs-model 100`, `--input-size 2.5`, `--s4-n 8`, `--s4-h 512`, `--s4-layers 4`, `--precision 32` — matching §4 and §6.5 in every stated hyperparameter. Signals therefore travel from 500 Hz to 100 Hz inside that pipeline, which is what §4 of this protocol does. Entry 5 recorded the use of `records500` as a departure from the reference and as a second systematic offset from 0.941, on the strength of PhysioNet's example script and of `tmehari/ssm_ecg` operating on directories named `ptb_xl_fs100`; both describe the codebase behind 0.9417, not the one behind 0.941. Entry 5's decision stands and its justification is unchanged — only its cost is withdrawn. The second half of this entry bears on the question this repository asks rather than on its execution: §3.3 of the paper states that "all models use the standard 12 ECG leads without additional resampling or filtering", while the same work's configuration declares 500 Hz data together with per-model rates of 100 Hz for S4, 240 Hz for ECG-CPC and 250 Hz for ST-MEM and ECG-JEPA. Resampling necessarily occurs and is nowhere described. This is not an allegation of error; it is the observation that the step whose effect this repository measures is invisible in the methodology as reported, which is the premise of §1. |
| 18 | 2026-08-26 | §3 | Recorded: the paper does not state how PTB-XL is split. That the folds used here — 1–8 train, 9 validation, 10 test — are the ones behind 0.941 is an assumption, not a verified correspondence. | The word "fold" does not occur anywhere in the paper or its appendices, and no split ratio or split procedure is described for PTB-XL; §3.3 says only "model selection on the validation set" and "bootstrapping on the test set (n=1,000)". The official stratified folds are the near-universal convention for this dataset and the pipeline behind 0.941 is built on `clinical_ts`, which uses them, so the assumption is a reasonable one. It is nonetheless an assumption, and unlike entries 16 and 17 it cannot be closed from the published material: it is recorded rather than resolved. If Stage 0 fails narrowly this is now the second thing to examine, after entry 11, those two being the only candidates left. The rate comparison of §8.1 is internal and holds the split fixed, so it is unaffected either way. |
| 19 | 2026-08-26 | §3, §8.1 | Recorded: the two stages aggregate seeds differently on purpose. §8.1 averages *predictions* across seeds within a rate, as written; §3 averages *metrics* across seeds, per entry 15. The asymmetry is deliberate and the estimands differ accordingly. | Averaging predictions builds an ensemble, and the macro AUROC of a five-model ensemble is generally higher than the mean of the five individual values, so the two operations are not interchangeable and read as an inconsistency unless the reason is stated. For §3 averaging predictions would be wrong: the comparison is against 0.941, a single published model, and an ensemble of three would not be the same kind of object. For §8.1 averaging predictions is preferable: the comparison is internal, the same operation is applied to every rate, and ensembling suppresses initialisation noise while leaving the systematic rate effect, which is the only quantity of interest. The interpretive consequence is stated here rather than left implicit: the Δ of §8.2 is measured on five-seed ensembles and compared against 0.006, an adjacent gap between *single* models on the leaderboard. Ensembling removes variance rather than signal, so the magnitudes remain comparable, but a reader is entitled to know that the two sides of that comparison are not built the same way. Per-rate seed spread is reported in both stages regardless, as §8.1 already requires. |
| 20 | 2026-08-26 | §8.2 | The phrase "Holm-adjusted 95% CI" is replaced by three quantities that exist: the unadjusted 95% percentile interval of each pairwise difference, the Holm-adjusted p-value for the family of six, and a Bonferroni-simultaneous percentile interval at α/12 and 1 − α/12, labelled as conservative. | Holm–Bonferroni adjusts p-values; it does not define an interval, and there is no standard construction that turns a step-down p-value procedure into a confidence interval. Left as written, the phrase would have been implemented as whatever the analysis code happened to compute, and reported under a name that sounds authoritative and denotes nothing. The three replacements each answer a real question: the unadjusted interval is the uncertainty of that one comparison, the Holm-adjusted p-value is the family-wise significance §8.1 prescribes, and the Bonferroni-simultaneous interval is the interval a reader may use to view all six comparisons at once, at the cost of being wider than necessary. Reporting all three also prevents the choice among them from being made after the numbers are seen. The interpretation thresholds of §8.2 — 0.006 and 0.019 — are unchanged and continue to apply to Δ itself, not to any interval. |
| 21 | 2026-08-26 | §8.2 | "Within-rate seed spread" is made precise as an **internal null contrast built exactly like Δ**: within each rate the five seeds are split into two disjoint subsets, the same ensemble difference is computed as between rates, and the comparator is the median absolute value over all ten 2/3 partitions, pooled across rates. Per-rate seed standard deviation and range continue to be reported. | §8.1 forms Δ by averaging predictions across the five seeds of a rate (entry 19), so Δ is a difference between two five-seed ensembles. "Seed spread", read as the dispersion of individual runs, is an object of a different kind: an ensemble of five carries roughly a fifth of the variance of a single run, so measuring Δ against individual-run dispersion sets a systematically slack threshold and tilts the rule of §8.2 toward "not detectable". That tilt is toward the null, which is the safer direction for credibility and the wrong one for power; either way it is not the comparison §6.4 describes. The internal null contrast removes the mismatch by construction: numerator and denominator are both differences between ensembles over the same records, differing only in whether the two ensembles come from different rates or from the same one. It answers the question §6.4 actually asks — whether changing the rate moves macro AUROC further than reseeding does — without changing object halfway through. Two alternatives were considered and rejected. The pooled between-seed standard deviation is simpler to describe and does not depend on the number of seeds, but remains an individual-run quantity and so keeps the mismatch. The max-minus-min range is the most literal reading of the original wording, but its expectation grows with the number of seeds — roughly 2.3σ at five, 3σ at ten — which would make the verdict depend on a budget decision rather than on the data. The choice was made before any run and costs no additional compute; it was noticed only because the analysis code was exercised on synthetic predictions before the real ones existed. |
| 22 | 2026-08-26 | §8.2 | **Corrects entry 21, which was wrong.** The internal null contrast uses two disjoint subsets of **equal size** — 2 against 2 out of five seeds, fifteen pairs, one seed left out of each — not the 2/3 partitions of entry 21. The comparator is the median absolute value, used raw and declared as an upper bound. | Entry 21's 2/3 partition does not measure seed variation: it measures ensemble size. On synthetic predictions written to exercise the analysis code, ensembles of one, two, three, four and five seeds scored 0.738, 0.813, 0.861, 0.895 and 0.919, and the 2-against-3 contrast had median absolute value 0.048 with mean −0.048 — a fixed sign, therefore not noise but the systematic advantage of averaging one more model. A comparator built that way would have declared every plausible Δ undetectable, and would have done so for a reason having nothing to do with seeds. Equal-sized subsets remove the confound: the same synthetic data gives a median of 0.010 for 2 against 2, against a between-seed standard deviation of 0.014. One imperfection remains and is not removable at this budget. A contrast between two disjoint ensembles of k has standard deviation σ√(2/k); Δ compares ensembles of five, the contrast compares ensembles of two, so the empirical comparator exceeds the true seed-noise floor of Δ by about √(5/2) ≈ 1.58. An exactly matched contrast would need ten seeds per rate. The raw value is used rather than a √(2/5) rescaling because rescaling assumes the between-seed noise averages like a variance, which is reasonable but unverified for AUROC, a functional that is not linear in the predictions. The consequence is stated plainly: the rule of §8.2 is conservative by a known factor in a known direction, an effect between roughly 0.6 and 1.0 times the measured comparator will be called undetectable, and a Δ that clears this threshold cannot be dismissed as an artefact of the correction. Both the raw and the rescaled values are reported. |
| 23 | 2026-08-26 | §6.7, §8.3.2 | Block 4 added: five runs at 100 Hz, Arm A, unnotched, reading `records100` instead of resampling `records500`. The matrix becomes 48 runs. `source` joins the run specification and the configuration key. | §8.3.2 asks whether the dataset authors' downsample and `resample_poly` give different macro AUROC, and no block in §6.7 used `records100`, so the analysis was declared but not executable — a gap between §6.7 and §8.3 rather than a decision. It is closed by adding runs rather than by weakening the question. A signal-level comparison would have been free but would answer something else: that the two signals differ is known by construction, since the anti-aliasing filters differ, and the question is whether they differ enough to move the metric, which no spectrum can settle. Three further reasons to spend the compute. First, the 100 Hz arm of §8.1 is resampled by us, so any effect found there invites the objection that it is our filter rather than the rate; measuring that contribution separately answers it with a number instead of an argument. Second, entry 17 established that 0.941 and 0.9417 differ in exactly this provenance — `records500` resampled against `records100` distributed — so the size of the effect is also an estimate of how comparable those two published figures are, which neither paper reports. Third, the comparison is clean without new machinery: `records100` cannot carry the notch of entry 2, since at 100 Hz the mains fundamental sits on the Nyquist frequency, so block 4 is compared against block 3, which is already the unnotched 100 Hz cell on `records500` and differs in one respect only. Cost: five runs at the cheapest rate, about 12% more compute, well inside the €40 ceiling; a cache built from `records100`, a path `data.py` supports but has never exercised on real data; and a regenerated `configs/runs.csv`, committed before execution as before. Alternative rejected: measuring the signals first and training only if the difference looks large. §6.7 refuses conditional designs for the same reason it refuses making Arm B conditional on Arm A — a choice made after looking is hard to defend even when it is reasonable. |
| 24 | 2026-08-26 | §8.3.2, §5.1 | Measured before any run: the two 100 Hz provenances agree on the bulk of the amplitude distribution and differ in its tail. On the full dataset `records100` has 306 records above 5 mV, 25 above 10, one above 20 and none above 30, with a largest sample of 20.0 mV; the same records resampled from `records500` give 319, 38, 14 and six, with a largest sample of 31.1 mV. Registered as an expectation for §8.3.2. | The bulk is indistinguishable — 1.40% against 1.46% above 5 mV — while the extreme tail is not, which is the signature of the overshoot described in entry 7: only records saturating the int16 converter at 32.767 mV (gain 1000 ADU/mV) produce ringing, and only those are amplified by the anti-aliasing filter of `resample_poly`. PhysioNet's own downsample attenuates the same spikes instead of ringing around them. A further observation, not anticipated: the six records with the largest peaks under one provenance are disjoint from the six under the other, so which records count as extreme is itself provenance-dependent. Two consequences are registered now rather than discovered afterwards. First, if block 4 differs from block 3, the expectation is that the difference concentrates on a small number of saturated records rather than spreading across the population; the per-record predictions written by `evaluate.py` make this checkable without further runs, and a difference spread evenly would be evidence against the overshoot explanation rather than for it. Second, and stated to close a door before it opens: entry 7 declined to exclude the saturated records, and this measurement does not reopen that decision. Registering an expectation about which records drive an effect is not a licence to remove them once the effect has been seen. Reproducible by running `scripts/build_cache.py --fs 100 --no-notch` with and without `--source lr`. |
| 25 | 2026-08-26 | §6.7 | Numerical settings fixed before the first epoch of the first run, and identical across all 48: TF32 off, deterministic algorithms forced (`--deterministic`), and whichever Cauchy backend `s4.py` reports once `pykeops` has been installed or has failed to install. All three are already in the checkpoint fingerprint and in every run manifest, so none of them can change mid-matrix. Pre-specified branch: if `torch.use_deterministic_algorithms(True)` raises on an operation S4 requires, determinism is dropped for **all** 48 runs rather than for some, and the operation named in the exception is recorded. Determinism is never traded for speed; if forcing it puts the extrapolated cost over the ceiling, that is resolved by the hierarchy of §6.7 — faster implementation first, longer rental second. | TF32 is not a choice so much as a correction: it carries ten bits of mantissa, and §6.7 says training is fp32 following the benchmark's `--precision 32`, a sentence that is simply false with TF32 left at its default. Determinism is a real choice, and it is made in favour of forcing it because of what the design spends its resolution on. The effect sizes this comparison has to speak about are the ones §8.2 borrows from Table 3 of the reference — a median gap of 0.006 between adjacent leaderboard positions, 0.019 at the widest — while the null scale it measures against comes from reseeding. Any variance at fixed seed that is not the manipulation is subtracted from that resolution, and non-deterministic kernel selection is the one such term the design can remove outright at no analytic cost. The apparent argument for leaving it off is that entry 14 then measures the run-to-run floor at fixed seed; that argument is weaker than it looks, because entry 22 already fixed the metre of §8.2 as the internal two-against-two null contrast across seeds, so the fixed-seed floor is a quantity the analysis would report and not consume, bought by widening the very intervals it does consume. The branch exists because `configure_numerics` raises rather than degrading — deliberately, so that a missing deterministic implementation is a visible failure and not a silent slowdown — and because S4 leans on index, scatter and FFT-adjacent operations whose deterministic CUDA coverage cannot be established from a CPU-only build on a laptop. Writing a single unconditional setting here would mean either discovering on the pod that the protocol is unexecutable, or quietly relaxing it after the fact; naming the fallback in advance costs nothing and is the difference between the two. Uniformity across the matrix is the substantive part: a fingerprint that held determinism for blocks 0 and 3 and not for 1 and 2 would partition the runs into two arithmetics, leaving the reproducibility check of entry 14 to certify a path the comparison arms never used, which is worse than not running the check. Entry 14 anticipated both branches and is read according to whichever held. Settled before the probe epochs of the runbook rather than after, because those two epochs are epoch 0 of two real runs and the fingerprint will refuse to resume across a change — which is the intended behaviour, not an obstacle to work around. |
| 26 | 2026-08-26 | §6.7 | The calibration this section mandates is carried out differently from the way it describes: one epoch at 100 Hz and one at 500 Hz, with 240 and 250 Hz **interpolated between the two measured endpoints**, in place of one full blocking-check run at 100 Hz with the other three rates **extrapolated by sequence length**. Recorded separately: the sizing paragraph read "all 43 runs" after entry 23 had made the matrix 48, contradicting the table above it in the same section; the figure was corrected editorially to 48 and the 20–30 GPU-hour range, which predates block 4, was left as written and flagged as such rather than silently re-estimated. | Proportionality to sequence length is an assumption about how fully the card is occupied, not about the model. At 250 samples and batch 64 an RTX 4090 is far from saturated, so the cheap end is dominated by per-step overhead that does not shrink with the sequence, and the linear rule therefore overstates 100 Hz and understates 500 Hz — the direction that matters, since the ten 500 Hz runs are the largest single item in the budget and the one an optimistic estimate would hide. Measuring both ends removes the assumption instead of refining it. One epoch rather than one full run because the quantity wanted is seconds per epoch and the schedule of §6.5 is fixed at 100 epochs: nothing is learned from epochs 2–100 that epoch 1 does not already give, while completing a 500 Hz run before the total is known is precisely the expenditure the calibration exists to prevent. Neither probe epoch is additional spend — they are epoch 0 of `b0-fs100-A-n0-s0` and `b1-fs500-A-n1-s0` and both runs resume from them, which the fingerprint of entry 25 protects by refusing to resume across any change in the numerical settings. Bearing on any conclusion: none. This entry changes only how a cost estimate is formed, and it makes that estimate more accurate, which lowers rather than raises the chance of ever reaching the hierarchy of §6.7 — cheaper card or longer rental, never fewer seeds and never dropping Arm B. It is recorded all the same, because §6.7 states a procedure and this is a different one, and a protocol that quietly improves on itself where nobody is looking is not distinguishable afterwards from one that quietly relaxed. |