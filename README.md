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
spread of 0.060, with adjacent positions frequently separated by 0.005 to 0.010.
A preprocessing choice does not need a large effect to reorder that.

## The question

**Q1.** Holding architecture, task, window duration in seconds, schedule and
seed fixed, how much does macro AUROC on PTB-XL move when only the sampling rate
changes?

**Q2.** Is that movement small or large relative to the gaps the leaderboard
treats as differences between models?

## What this is not

This does not re-rank the foundation models. Doing so would require retraining
each of them at matched rates, which is not possible with public checkpoints and
would not be affordable here anyway. The claim is narrower and, I think, more
defensible: it puts a number on the size of one uncontrolled factor, so that
readers of a leaderboard can judge which gaps are large enough to survive it.

The protocol is registered in [PROTOCOL.md](PROTOCOL.md) before any model is
trained. The literature review that produced this design is in
[RELATED_WORK.md](RELATED_WORK.md), and it comes first this time.

## Why this exists

My background is in time series, which is why one sentence in the paper's methods section caught my eye: the frequency at which a signal is read is not a neutral preprocessing step.

Since then I have worked in consulting — delivery, across a much wider range of domains and problems than I started in, which meant learning new techniques continuously while holding the same methodological standards across all of them.

That is what this repository is meant to show. Not that I build architectures: my hands-on level is fine-tuning and evaluation design, and I would rather state that than let anyone assume otherwise. It is how I take a problem, work out what is actually being asked, and build an approach that holds together — the protocol written before the runs, the controls that make the comparison mean something, the statistics, and the record of what changed and why.

I started this while preparing an application for a research position. The question is one I would want answered regardless, and of the domains I could have picked, cardiology is the one closest to home.

## Status

| Stage | State |
| --- | --- |
| Literature review | in progress |
| Protocol registered | not yet |
| Statistical machinery + tests | done |
| Data pipeline | not started |
| Multi-rate training runs | not started |
| Analysis | not started |

**No results yet.** This table is updated as runs complete, including runs that
do not support the hypothesis.

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
