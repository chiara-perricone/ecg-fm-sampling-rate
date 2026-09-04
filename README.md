# How much of the ECG foundation model leaderboard is sampling rate?

A controlled ablation on PTB-XL, holding architecture fixed and varying only the
rate at which the signal is read.

## The gap this addresses

Al-Masud, Lopez Alcaraz and Strodthoff (ICLR 2026, [arXiv:2509.25095](https://arxiv.org/abs/2509.25095))
benchmark eight ECG foundation models across 12 datasets and 26 clinical tasks.
Their evaluation is statistically careful: bootstrap confidence
intervals on the test set, pairwise comparisons by bootstrapping differences,
rankings that assign ties where intervals overlap. Their main finding is
that architecture matters more than scale.

Their methods section states that all models use the standard 12 leads
"without additional resampling or filtering". Each pretrained encoder therefore
reads the signal at whatever rate it was pretrained on. Table 32 reports input
lengths that differ by an order of magnitude across models — 500 timesteps for
HuBERT-ECG, 5000 for ECGFM-KED, 600 for ST-MEM and ECG-CPC, 2500 for ECG-JEPA
and ECG-FM, 1250 for ECGFounder and MERL. Where the paper states rates directly,
they differ too: ECG-CPC runs at 240 Hz, the floor of its pretraining corpus,
while the supervised S4 baseline runs at 100 Hz.

This is a sensible choice — evaluating models as released is the honest thing to
do, and the authors say so. But it means **sampling rate varies together with
architecture across the leaderboard**, and the paper does not report how much
of the resulting spread that accounts for. The authors do acknowledge a related
confound, noting that models pretrained on different datasets make it hard to
isolate what drives performance differences. Rate is not on that list.

The margins involved are small. On PTB-XL (all) under fine-tuning, the reported
macro AUROC runs from 0.949 for ECG-CPC down to 0.889 for ECGFM-KED — a total
spread of 0.060, with adjacent positions separated by a median of 0.006 and by
0.019 at the widest. A preprocessing choice does not need a large effect to
reorder that.

## The question

**Q1.** Holding architecture, task, window duration in seconds, schedule and
seed fixed, how much does macro AUROC on PTB-XL move when only the sampling rate
changes?

**Q2.** Is that movement small or large relative to the gaps the leaderboard
treats as differences between models?

## The answer

**One architecture, not the leaderboard.** What follows measures S4, the supervised
state-space baseline of the benchmark, trained from scratch at each rate. That is what makes
the experiment possible at all: the eight foundation models are pretrained at a fixed rate
and cannot be moved off it without pretraining again, so only the two from-scratch baselines
are eligible. Of those two, S4 is the harder test, because it is the one that *claims* not to
care — parameterised in continuous time with a learned step size, it is the architecture
least exposed to one of the two ways a rate change can bite. A transformer or a CNN, whose
receptive field is fixed in samples, is not in the same position. What that means for how far
the numbers travel is set out below, and at length in
[about-this-repository.md](about-this-repository.md) §6, §7 and §13.

All 48 runs are complete: 3 for the blocking reproduction of §3, 45 for the
comparison. Every estimate below is a paired bootstrap over 10,000 resamples,
five seeds per cell, Holm-corrected across the family of comparisons.

The endpoint is `macro_clean`: the macro AUROC over the 49 diagnostic labels
with at least ten positive cases in fold 10, a set computed once, frozen and
committed before any model was trained. `macro_all` averages all 71, of which 22
fall below that threshold and two carry a single positive; it stays the endpoint
for the §3 reproduction, because the published 0.941 is defined on the full
label set. Both are computed every epoch of every run and reported side by side.
See §10 entry 1.

**Q1 — at most 0.0015.** Ensemble AUROC by rate: 100 → 0.9456, 240 → 0.9454,
250 → 0.9458, 500 → 0.9469. No pair separates by more than that, in either
direction.

| comparison | Δ | 95% CI | p_holm |
|---|---|---|---|
| 100 − 240 | +0.0001 | [−0.0009, +0.0011] | 1.000 |
| 100 − 250 | −0.0002 | [−0.0012, +0.0007] | 1.000 |
| 100 − 500 | −0.0014 | [−0.0026, −0.0001] | 0.147 |
| 240 − 250 | −0.0004 | [−0.0011, +0.0003] | 0.977 |
| 240 − 500 | −0.0015 | [−0.0026, −0.0004] | 0.049 |
| 250 − 500 | −0.0011 | [−0.0022, −0.0001] | 0.147 |

**Q2 — small.** The pre-registered relevance threshold is 0.006, the median gap
between adjacent leaderboard positions. The uncontrolled factor is worth about a
quarter of the smallest difference that ranking treats as real.

**The negative control holds.** 240 Hz and 250 Hz differ by 4% in rate and by
nothing physiological, so a real difference between them would mean the pipeline
is measuring something other than what it claims to. That contrast is null,
which is what licenses reading the rest of the table. It is looked at first, as
the protocol requires.

**How far this travels.** A change of rate acts through two channels. One is the
diagnostic information the signal still carries, which every model in the
benchmark faces alike. The other is the relation between an architecture's
receptive field and physical time, which S4 largely absorbs because its step size
is learned rather than fixed in samples. The table above bounds the first
channel, and that bound transfers to the whole leaderboard. The second is
measured here only inside S4, through a pre-registered contrast on the time-scale
prior — which came out with the sign opposite to the hypothesis. Architectures
whose temporal extent is defined in samples remain exposed to it, and nothing
here settles their case.

The result is null, and §9 of the protocol commits to reporting it with the same
prominence as a positive one.

The rest — the time-scale prior contrast, the notch and source controls, the
reproducibility check that did not pass, and why the verdict is read off effect
size rather than significance — is in
[about-this-repository.md](about-this-repository.md), with the specifications in
[PROTOCOL.md](PROTOCOL.md) §8 and §10 and the full output in
`results/analysis_{rates,arms,notch,source}.{json,csv}`.

## What this is not

This does not re-rank the foundation models. Doing so would require retraining
each of them at matched rates, which is not possible with public checkpoints and
would not be affordable here anyway. The claim is narrower and, I think, more
defensible: it puts a number on the size of one uncontrolled factor, so that
readers of a leaderboard can judge which gaps are large enough to survive it.

The protocol is registered in [PROTOCOL.md](PROTOCOL.md) before any model is
trained. The literature review that produced this design is in
[RELATED_WORK.md](RELATED_WORK.md), and it comes first this time.
[about-this-repository.md](about-this-repository.md) is the long form: the
reasoning behind each design choice, the full results, and what they do not
settle.

## Why this exists

My background is in time series, which is why one sentence in the paper's methods section caught my eye: the frequency at which a signal is read is not a neutral preprocessing step.

Since then I have worked in consulting — delivery, across a much wider range of domains and problems than I started in, which meant learning new techniques continuously while holding the same methodological standards across all of them.

That is what this repository is meant to show. Not that I build architectures: my hands-on level is fine-tuning and evaluation design, and I would rather state that than let anyone assume otherwise. It is how I take a problem, work out what is actually being asked, and build an approach that holds together — the protocol written before the runs, the controls that make the comparison mean something, the statistics, and the record of what changed and why.

I started this while preparing an application for a research position. The question is one I would want answered regardless, and of the domains I could have picked, cardiology is the one closest to home.

## Status

| Stage | State |
| --- | --- |
| Literature review | done — `RELATED_WORK.md`, findings verified against code and papers |
| Protocol registered | done — `PROTOCOL.md`, 27 amendments, 26 of them pre-run |
| Data pipeline | done — PTB-XL v1.0.3, 87,203 files verified against PhysioNet's checksums |
| Model, training, evaluation | done — reproducibility check run, and it failed: see §10 entry 27 |
| Run matrix | committed before execution — `configs/runs.csv`, 48 runs |
| Analysis code | written and tested against synthetic predictions, before any real one exists |
| Blocking reproduction (§3) | **accepted** — 0.9373 against a target of 0.941 |
| Comparison runs (blocks 1–4) | done — 45 runs, 26.9 GPU-hours |
| Pre-registered analysis (§8) | done — four contrasts at 10,000 resamples, reported above |

**The blocking condition of §3 is met.** Three seeds at 100 Hz, macro AUROC over
all 71 labels on fold 10: 0.9351, 0.9377, 0.9392, mean 0.9373. The 95% bootstrap
interval over the 2,198 test records is [0.9299, 0.9438] and contains the 0.941
the reference reports; the mean sits 0.0037 from it against a pre-registered
tolerance of 0.010. `results/stage0.json` holds the numbers and
`results/runs/b0-*` the per-record predictions behind them. The point estimate
falls below the target rather than above it, which is worth stating plainly: the
criterion is met because it was written before the number existed, not because
the number came out flattering.

One figure in that result matters beyond the gate. The standard deviation across
the three seeds is **0.0020**, where Berger et al. report 0.01–0.03 for reseeding
alone, and where Table 3 of the benchmark this work interrogates separates
adjacent leaderboard positions by a median of 0.006. Whether the question asked
here is answerable at all depends on that ratio, because reseeding noise is the
denominator against which any effect of sampling rate has to be read. At this
scale the design resolves differences the leaderboard treats as meaningful
instead of drowning them. Two caveats, since the number is favourable: three
seeds is a small sample, and the comparator of §8.2 is the five-seed equal-subset
contrast of §10 entry 22, so this is an early reading rather than the figure the
comparison will use.

**Cost, measured rather than assumed.** One epoch at each end of the matrix, on
an RTX 5090 at batch 64, fp32, deterministic algorithms forced, Cauchy kernel on
pykeops: 10.1 s at 100 Hz (250 samples) and 38.0 s at 500 Hz (1,250 samples).
Five times the sequence for 3.76 times the time, which separates a fixed 3.1 s
per epoch from 0.0279 s per sample and puts all 48 runs at 26.9 GPU-hours.
Scaling linearly from the cheap end would have claimed 32.8, because the fixed
cost gets charged five times over. At the Community-tier price of the card that
is roughly €17 for the whole matrix, which is the more interesting number: the
question of how much of this leaderboard is sampling rate was answerable for the
cost of a paperback.

**Environment, stated because a claim depends on it.** Every run writes a
`manifest.json` recording the GPU model, the torch and CUDA versions, Python,
numpy, and the three numerical settings — TF32, cuDNN determinism, cuDNN
benchmarking — so that no run in `results/` is of unknown provenance. That is
more than housekeeping. Forcing deterministic algorithms makes two executions of
one configuration agree on the *same* card, not across architectures, and §10
entry 14 rests on precisely that agreement, so the card has to be on the record
for the claim to be checkable. These runs are on one rented RTX 5090, driver
CUDA 13.1, torch 2.13.0+cu130, Python 3.11. Training is the only step needing a
GPU: `scripts/analyse.py` reads the per-record predictions and finishes on a
laptop CPU in minutes, which is why those predictions are committed rather than
left on a machine that no longer exists.

`RUNBOOK.md` gives the steps to reproduce all of it from nothing, in the order
they have to happen, with what to check before each one.

The ordering above is the point rather than an accident. The protocol, the run
matrix and the analysis are all committed before the data they will be applied
to exists, so that no choice among them can have been made by looking at an
answer. §10 of `PROTOCOL.md` records every amendment with its date and reason,
including the two that turned out to be wrong and the one that withdrew a cost
this work had assumed against itself.

## Design notes

**Architecture held fixed, trained from scratch.** The confound cannot be
isolated using pretrained checkpoints, since each is locked to its pretraining
rate. Training one architecture from scratch at several rates is the only way to
vary rate alone.

**Window length held constant in seconds, not in samples.** Otherwise a change
of rate silently changes how much signal the model sees, and the two effects
cannot be separated.

**Paired bootstrap, not two independent ones.** Runs are scored on the same test
patients, so the case-resampling indices are shared by construction — see
`src/ecgres/stats.py`.

**Holm–Bonferroni over the family of comparisons.** Every rate pair on every task
is a hypothesis test. Holm controls the family-wise error rate without assuming
independence, which these comparisons do not have.

## Install

```bash
git clone https://github.com/chiara-perricone/ecg-fm-sampling-rate
cd ecg-fm-sampling-rate
pip install -e ".[dev]"
pytest -q
```

## Data and licence

PTB-XL v1.0.3 (PhysioNet, [DOI 10.13026/kfzx-aw45](https://doi.org/10.13026/kfzx-aw45)),
CC BY 4.0, not redistributed here. Code is MIT licensed. No proprietary or client
data is used anywhere in this repository.

---

Chiara Perricone — [chiara.perricone@gmail.com](mailto:chiara.perricone@gmail.com)
