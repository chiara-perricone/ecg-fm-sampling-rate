# `ecg-fm-sampling-rate` — what it is and why it exists

*Chiara Perricone — `github.com/chiara-perricone/ecg-fm-sampling-rate` (MIT)*

---

## Why a repository built for this

I am a lead data scientist delivering AI projects for large enterprises. The code I write
there belongs to clients and to my employer: it cannot be shown, and it is not shaped to
answer an open research question. So rather than point at work I am not free to open, I built
something from scratch, in the open, on public data — chosen specifically so that the design
choices, not the deliverable, are the visible part.

The constraint was also a filter. I picked a question that a single rented consumer GPU could
settle in a weekend, and I let the honest answer be the deliverable — including the
possibility of a null result.

---

## 1. The signal, and how much of it you keep

Twelve electrodes measure a potential difference across the skin. Each lead is a time series
of voltages, and the **sampling rate** is how many measurements per second are recorded: at
100 Hz, one hundred numbers per second per lead.

The rate is not a storage detail, because it determines what the signal *can contain*. By the
Nyquist limit, sampling at `fs` represents frequency content only up to `fs/2`. At 100 Hz you
retain content up to 50 Hz; at 500 Hz up to 250 Hz. Everything above is not attenuated — it is
gone.

**What clinical cardiology requires.** The American Heart Association has specified a
**0.05–150 Hz** bandwidth for routine diagnostic 12-lead ECG since 1990, and the supporting
studies indicate that **500 samples per second** are needed to sustain that 150 Hz cutoff while
holding amplitude error near 1% in adults. Higher frequencies matter for measuring rapid
upstroke velocity, peak amplitude, and short-duration waves.

**And yet.** Published ECG models operate at 100, 240, 250 and 500 Hz, and the ones at 100 Hz
— a 50 Hz bandwidth, a third of what the clinical standard protects — are competitive. The
supervised S4 baseline runs at 100 Hz and reports 0.9417 macro-AUROC.

One of two things must be true. Either deep models recover the diagnosis from content below
50 Hz, in which case four fifths of the data can be discarded with consequences for storage,
wearables and telemedicine — or something *is* lost at 100 Hz and the published comparison is
concealing it. That is the question this repository exists to answer, for one architecture,
under control.

---

## 2. The task

The data is PTB-XL (public, PhysioNet): nearly 22,000 real 12-lead ECG recordings, annotated
by cardiologists with the 71 diagnostic statements of the standard `all` task.

The problem is **multi-label** — meaning the 71 statements are not alternatives to choose
between, but 71 independent yes/no questions asked of the same recording, any number of which
can be true together. One ECG may carry a single label, *atrial fibrillation*. Another may
carry three at once — *right bundle branch block* **and** *left ventricular hypertrophy*
**and** *ST/T changes* — because one heart can present all three.

Mechanically, this shows up in the output layer. The model emits 71 values, each passed
through its **own sigmoid** rather than a shared softmax: a softmax normalises to sum 1 and
would impose mutual exclusivity, which is exactly wrong here. Training uses binary
cross-entropy summed over the 71 labels. It is 71 binary problems sharing one backbone, not
one choice among 71.

---

## 3. The score

Each statement gets an **AUROC**: the probability that an ECG which does have that condition
receives a higher score than one which does not. 0.5 is a coin flip, 1.0 is perfect. It
requires no decision threshold, which is why it is the metric here.

The 71 values are then averaged **with equal weight** — the "macro" part. A rare condition
therefore counts as much as a very common one, so a model that only gets the frequent
diagnoses right cannot hide behind them.

That single number is measured on the **2,198 recordings of fold 10**, which no model ever
trains on. Same metric, same split as the published leaderboard, so the numbers sit directly
alongside it. This is the quantity called `macro_all` throughout the repository.

---

## 4. What the benchmark asked, and what it found

The reference benchmark is not a contest between architecture families — there are CNNs and
transformers on both sides of its divide:

| model | family | pretraining |
|---|---|---|
| HuBERT-ECG | transformer | self-supervised |
| ST-MEM | transformer, masked modelling | self-supervised |
| ECG-JEPA | transformer, JEPA | self-supervised |
| ECG-FM | transformer | self-supervised |
| MERL | CNN (ResNet) | self-supervised, with text |
| ECGFounder | CNN (Net-1D) | large-scale supervised |
| ECG-CPC | convolutional + autoregressive | self-supervised, contrastive |
| ECGFM-KED | hybrid, ECG + text | self-supervised |
| **S4** *(baseline)* | **state space** | **none** |
| **Net-1D** *(baseline)* | **CNN** | **none** |

The real axis is **pretraining**: eight foundation models, pretrained self-supervised on large
quantities of unlabelled ECG, against two baselines trained from scratch on the labelled task
alone. The question is whether large-scale pretraining actually buys anything over a good
supervised model — the "reality check" of the title — evaluated across 26 tasks and 12 public
datasets, both fine-tuned and with frozen backbones.

The answer is **mixed**. In adult ECG interpretation three foundation models consistently beat
the baselines, but across several other categories most foundation models fail to surpass
supervised learning. This is why S4's 0.9417 is a target worth reproducing rather than a floor
to clear: it is the baseline that much more expensive models do not reliably pass.

My `macro_all` on PTB-XL is **one cell** of that matrix.

---

## 5. Why sampling rate is confounded — and why that is not an error

Each model reads the signal at the rate it was pretrained for, over a window of its own
length. Both are declared explicitly, model by model, in the benchmark's launch script:

| model | native rate | window | timesteps |
|---|---|---|---|
| HuBERT-ECG | 100 Hz | 5 s | 500 |
| **S4** *(supervised baseline)* | **100 Hz** | **2.5 s** | **250** |
| ECG-CPC | 240 Hz | 2.5 s | 600 |
| ST-MEM | 250 Hz | 2.4 s | 600 |
| ECG-JEPA | 250 Hz | 10 s | 2,500 |
| ECGFounder | 500 Hz | 2.5 s | 1,250 |
| MERL | 500 Hz | 2.5 s | 1,250 |
| Net-1D *(supervised baseline)* | 500 Hz | 2.5 s | 1,250 |
| ECG-FM | 500 Hz | 5 s | 2,500 |
| ECGFM-KED | 500 Hz | 10 s | 5,000 |

Every cell was read from the benchmark's own code; every timestep count agrees with the
corresponding table in the paper.

**That is the correct choice for their question.** Handing a model an unfamiliar resolution
would penalise it for a preprocessing mismatch and then report the result as a verdict on
pretraining. What follows is therefore an unavoidable side effect of a sound design, not an
oversight — which is also why nobody went back to isolate it.

**What follows.** Reading down the rows, architecture, sampling rate and window duration all
change together. Sampling rate is **confounded with architecture in every row**. And — noticed
less often — so is **window duration**: 2.4 to 10 seconds, a fourfold range.

ST-MEM and ECG-CPC show how tangled this gets. Both feed their model **600 timesteps**, but
one is 2.4 s at 250 Hz and the other 2.5 s at 240 Hz. Identical tensor shape, different
stretch of heartbeat.

**Why no reanalysis can fix it.** This is structural, not a matter of statistical power. To
separate two factors they must vary *independently*: you need rows with the same architecture
at different rates. No such rows exist — each architecture appears at exactly one rate. The
confound is complete, so the published numbers cannot be corrected after the fact. Answering
the question requires **new runs**: hold the architecture fixed and move the rate alone. The
question is not answerable by reading, only by training.

---

## 6. Why S4 is the vehicle

S4 is a **Structured State Space sequence model**. Rather than attention or convolution, it
treats the input as a linear dynamical system in **continuous time**:

$$x'(t) = A\,x(t) + B\,u(t) \qquad y(t) = C\,x(t) + D\,u(t)$$

Here `u` is the input signal — one ECG lead — `x` a hidden state, `y` the output. Note what
the model is defined over: not samples, but *time*.

To apply it to sampled data the system is **discretised with a step size Δ**, and the result
can be computed in two mathematically equivalent forms:

| form | cost | used for |
|---|---|---|
| recurrence, like an RNN | O(1) per step | sequential inference |
| one global convolution, via FFT | O(L log L) | training, fully parallel |

So it trains like a CNN and can run like an RNN. Neither classical family offers both.

**What "structured" means.** Computing that long convolution kernel from a general `A` is
prohibitive. S4 imposes structure: `A` is initialised by **HiPPO** — a theory of how to
optimally compress the history of a signal — and parameterised as **diagonal plus a low-rank
correction**, which allows stable diagonalisation and reduces the kernel computation to a
**Cauchy matrix product**. This is not incidental infrastructure: it is why `pykeops` is a
dependency of this repository, since that is the library evaluating the Cauchy kernel.

**Why it suits ECG.** Ten seconds at 500 Hz is 5,000 timesteps. Transformer attention costs
O(L²) — 25 million pairs — while S4 attains a global receptive field at O(L log L). For long
physiological signals this is the regime where it wins, and in the benchmark it is one of the
two baselines that several foundation models do not beat.

**And the reason it is the right vehicle for this particular question.** Because S4 is
formulated in continuous time, **resolution invariance is a claimed property of the
architecture**. The original paper takes a model trained on 16 kHz speech, halves the rate to
8 kHz with no retraining — rescaling Δ and leaving every other parameter untouched — and
retains 96.3% accuracy.

**Why not one of the transformers.** Not cost, and not flattery. Eight of the ten rows in §5 are
*pretrained* models, and this experiment trains from scratch at four different rates: doing that
to a foundation model means re-pretraining it four times, which is orders of magnitude outside
this budget, while fine-tuning a checkpoint at a rate it was never pretrained on measures the
mismatch instead of the rate. Only the two supervised baselines, S4 and Net-1D, are trained from
scratch on the labelled task, so only those two are eligible at all.

That left S4 against Net-1D, a CNN — and S4 is the more demanding test. A convolution kernel's
width is defined in samples, so a rate effect on Net-1D would be close to expected and would say
little. S4 is the architecture that *claims* not to care, which is what makes an effect
meaningful and a null informative. Choosing the CNN would have made a positive result easier to
find and cheaper to believe.

It is also the architecture I understand best, which I would rather state than leave implied.
Designing a controlled experiment on a model whose inductive bias you cannot reason about is one
of the ways confounded experiments get made.

---

## 7. What this experiment asks, which is not the same question

S4's claim and this experiment's question are adjacent but distinct, and conflating them would
be the single easiest error to catch in this work:

| | what S4 demonstrates | what is measured here |
|---|---|---|
| model | trained **once**, at one rate | trained **from scratch** at each rate |
| question | does it transfer to another rate? | does the *achievable* score depend on the rate? |
| what is at stake | reparameterisation (Δ) | diagnostic information present in the signal |

S4's property makes the null hypothesis **principled rather than arbitrary**. It does not
establish it. Training from scratch at each rate — with Δ itself learned — asks how much
diagnostic information survives at each resolution, which is a harder question than transfer.

---

## 8. The design

48 runs. The four rates are not chosen by hand: they are **exactly the distinct native rates**
of the table in §5 — 100, 240, 250 and 500 Hz.

Held fixed: architecture, task, **window duration in seconds** (2.5 s, matching the S4 row of
the benchmark), optimisation schedule, seeds. Deliberately varied: the sampling rate. Only the
sample count changes — 250 at 100 Hz, 600 at 240 Hz, 625 at 250 Hz, 1,250 at 500 Hz — alongside
three pre-registered control factors. Five seeds per cell in the comparison; three in the
reproduction gate of §3, where the comparator is a single published number.

**Arms — where the prior over timescales lives.** S4's step size Δ is learned, but its
*initialisation* is specified in steps, so at a higher rate the same default covers less
physical time. Arm A leaves that default alone, which means the prior moves with the rate; Arm B
rescales it so the prior in seconds is the same at every rate. The contrast is pre-registered
and it is the only way to ask whether an effect is information in the signal or a
reparameterisation. One limit is recorded rather than discovered later: the initialisation of
the C matrix also depends on Δ·L through the vendored layer's length correction, and Δ·L and
Δ·fs differ only by the window length in seconds, so A−B is one degree of freedom that reads as
"everything depending on Δ expressed in steps" rather than the timescale prior alone
(`PROTOCOL.md` §10 entry 9).

**Notch — removing a confound the design would otherwise create.** The 50 Hz mains peak is the
dominant spectral feature of this dataset, at 5.46× the harmonic-free control density, and it
sits exactly on the Nyquist frequency of the 100 Hz arm. Left in place, that arm alone would
receive partial mains suppression for free, mixing "less interference" with "less temporal
resolution" — the confound this repository exists to measure in others. A zero-phase 50 Hz notch
is therefore applied to every arm before resampling, and unnotched runs are kept as a control.

**Source — whose downsampling.** The 100 Hz arm is produced by resampling the 500 Hz records, so
any effect at 100 Hz invites the objection that it is my anti-aliasing filter rather than the
rate. Five runs read the dataset authors' own distributed 100 Hz files instead, which separates
the two with a number instead of an argument. It also happens to measure how comparable the two
published baselines are to each other, since they differ in exactly this provenance.

---

## 9. The denominator: measuring the noise before measuring the effect

This is the step applied work skips most often. Retraining the **same** model with a different
random seed changes the score anyway. If that noise is larger than the effect you are looking
for, the experiment cannot answer the question — however clean everything else is.

So the noise has to be measured first, and it becomes the yardstick. Three seeds at an
identical configuration:

- 0.9351 / 0.9377 / 0.9392 → **between-seed standard deviation 0.0020**

For comparison: the published prior for reseeding noise alone is 0.01–0.03, and adjacent
positions in the benchmark's own ranking differ by a median of 0.006, with a maximum of 0.019.
At 0.0020 the design **resolves** differences that the leaderboard treats as meaningful,
instead of drowning them. None of the cited work reports this figure for a controlled setup.

Two cautions, since the number came out favourable. Three seeds is a small sample. And the
yardstick the analysis will actually use is the pre-registered internal 2-vs-2 null contrast
over five seeds, not this standard deviation — this is a preview, not the final denominator.

---

## 10. Safeguards written before the data

**Pre-registration, with timestamps to prove it.** The protocol, the full run matrix and the
analysis script were committed *before* the data they apply to existed, so no choice among them
can have been made after seeing an answer. `PROTOCOL.md` §10 is an **append-only** amendment
log: every deviation recorded with date and reason, and entries are never edited — when one
turns out to be wrong, a new entry corrects it. Two entries were corrected this way, and one
withdrew a cost the work had charged against itself.

**A real gate, with the consequence written first.** §3 required reproducing the published
baseline at 100 Hz within a pre-registered tolerance of 0.010 macro-AUROC. Failing meant
stopping and publishing the failure.

- three seeds: 0.9351 / 0.9377 / 0.9392, mean **0.9373**
- bootstrap 95% CI over the 2,198 test records: **[0.9299, 0.9438]**
- published target 0.941 — inside the interval, **0.0037** below the mean

The estimate points *below* the target. The criterion is met because it was written before the
number existed, not because the number came out flattering.

**A negative control, taken from the table rather than invented.** 240 Hz vs 250 Hz. Both rates
appear in the leaderboard — ECG-CPC at 240, ST-MEM and ECG-JEPA at 250 — and a 4% difference in
rate cannot plausibly carry extra diagnostic information. An effect there would indicate an
artefact in my pipeline, not a property of the signal. It is the first thing the analysis looks
at, before any headline result is read.

**Numerical settings frozen across all 48 runs.** TF32 disabled — it carries ten mantissa bits,
which would make the declared fp32 precision a false statement — and deterministic algorithms
forced. Both are part of the checkpoint fingerprint, so they cannot drift mid-matrix. A
pre-specified fallback existed in case determinism could not be enforced: disable it on *all*
48 runs, never on some. It was not needed — `torch.use_deterministic_algorithms(True)` raised
on nothing S4 requires.

**And an end-to-end check of that, which did not pass.** The matrix contains three pairs of
runs that are identical in every respect, placed there in advance for no other purpose than to
test whether one execution reproduces another. They do not agree bit-for-bit. Determinism holds
for the *configuration* — seeds, cuDNN flags, TF32 and the Cauchy backend are set, recorded and
verified identical — but not for the output: at epoch 0 the training loss differs by 1.1e-6 and
macro-AUROC by 3.2e-4, on all three seeds. The cause is not established, and I am not going to
assert one: `num_workers=8` and batch ordering are the first candidate, `pykeops` — which
compiles CUDA reductions outside the scope of `use_deterministic_algorithms` — the second, and
neither has been tested. What can be said is the scale. Between two *seeds* of the same cell
the same quantities differ by 7.2e-4 and 1.8e-3, so the residual drift is roughly **650× smaller
than seed-to-seed variation**, and it sits inside the internal null contrast that §8.2 of the
protocol uses as its yardstick. No result here rests on bit-exactness; they rest on five seeds
per cell and a paired bootstrap. Nor did anything rest on it in practice: no run was interrupted
and resumed, so the assumption those pairs existed to test was never relied upon. The check is
written up as `PROTOCOL.md` §10 entry 27 — the first post-run entry in that log, and the reason
the log is append-only.

**The architecture is vendored, not installed.** `src/ecgres/vendor/` holds a copy of the model
code, versioned alongside the experiment, with a header declaring "No other changes". A `pip
install` resolves a version at install time and may resolve a different one next year, which
would make these 48 runs incomparable to anyone else's. Vendoring also makes the attribution
checkable with a `diff`: the claim that I did not touch the architecture is verifiable rather
than asserted.

---

## 11. Cost, measured rather than estimated

One-epoch probes at both ends of the matrix separate fixed from variable cost:

| fs | samples | s/epoch |
|---|---|---|
| 100 | 250 | **10.1** (measured) |
| 240 | 600 | 19.9 (interpolated) |
| 250 | 625 | 20.6 (interpolated) |
| 500 | 1,250 | **38.0** (measured) |

Five times the sequence for 3.76 times the time. Two points separate **3.1 s per epoch** of
dataloader, validation and interpreter from **0.0279 s per window sample**.

Total: **26.9 GPU-hours for 48 runs, ≈ €17** of compute on a rented consumer card — about €20
including setup and the data download. A pizza for two.

Extrapolating linearly from the cheap end would have predicted 50.5 s at 500 Hz and 32.8 hours
overall: a **22% overestimate**, because the fixed cost gets charged five times. The error is in
the safe direction for a budget and the wrong direction for choosing hardware, since it makes
every rate look equally expensive and therefore makes a faster card look more useful than it is.

---

## 12. Status

- Stage 0 (reproduction gate): **passed** — 0.9373 against a target of 0.941, inside the
  pre-registered interval.
- Main campaign: **complete**. All 48 runs, 26.9 GPU-hours, none interrupted.
- Negative control (240 vs 250 Hz): **holds.** Δ = −0.0004, CI [−0.0011, +0.0007],
  p_holm = 0.977. Read first, before any headline number, as the protocol requires.
- Rate effect: **null, and bounded.** No pair of rates separates by more than **0.0015** in
  ensemble macro-AUROC — about a quarter of the 0.006 that separates adjacent leaderboard
  positions. The largest contrast, 240 vs 500 Hz, is −0.0015 with p_holm = 0.049: significant
  at the conventional threshold and far below the pre-registered relevance threshold, which is
  why the verdict is read off effect size and the rule for doing so was fixed in advance.
- Determinism check (block 0 vs block 3): **failed**, and written up as `PROTOCOL.md` §10
  entry 27. See §10 above: the drift is ~650× smaller than seed-to-seed variation and no
  result depends on it.
- One pre-registered contrast came out **against** the hypothesis: at 500 Hz the arms separate
  in favour of Arm A, the arm the design expected to be at a disadvantage. It is reported as
  found.

The write-up reports a null result with the same prominence as a positive one; that is fixed in
the protocol, not decided afterwards.

---

## 13. What the result can and cannot conclude

**It does not recreate the leaderboard, and could not.** Running the eight foundation models at
all four rates is not a budget problem but an impossibility at accessible cost: each is
*pretrained* at its native rate, so feeding one a different rate measures the mismatch, and
removing the mismatch means pretraining again — orders of magnitude beyond 26.9 GPU-hours. The
confound in the published table is not cheaply removable by anyone. That is a finding, not a
concession.

**What it delivers instead** is a magnitude for the nuisance variable: how much macro-AUROC
sampling rate moves on its own, architecture held fixed, read against the 0.006 that separates
adjacent leaderboard positions. If it is of that order, part of the published ranking reflects
preprocessing rather than modelling. If it is far smaller, the ranking is robust to it. Neither
number is recoverable from the published results.

**The effect has two components, and the design separates them on purpose.** Changing the rate
alters both the information present in the signal — shared by every model — and the relation
between an architecture's receptive field and physical time. The separation is not a property
of S4: it is the Arm A / Arm B contrast of §8, registered before any run, which moves the
initialisation of the timescale prior while holding the rate fixed. What that contrast isolates
is one degree of freedom carrying everything that depends on Δ expressed in steps, the length
correction of the vendored layer included (`PROTOCOL.md` §10 entry 9); it is not the timescale
prior in isolation, and the limit is recorded there rather than left for a reader to find. A
convolution kernel of 7 samples spans 70 ms at 100 Hz and 14 ms at 500 Hz; S4, parameterised in
continuous time with a learned step size Δ, is the architecture least exposed to that second
component.

**Hence an asymmetry, stated before the results.** A null on S4 is weak evidence for the rest of
the field: those models remain exposed to the receptive-field component that S4 largely
neutralises. An effect on S4 is strong evidence: if the architecture built to be
resolution-invariant is sensitive, the likeliest mechanism is information loss in the signal
itself — and that every model shares.

**Scope.** One architecture, one task, one dataset. No claim about the benchmark's other tasks,
and none about which model deserves which rank.

---

## Where to look in the repo

| If you want to see | Look at |
|---|---|
| The question, the gate, the analysis plan | `PROTOCOL.md` §3, §6.7, §8 |
| That nothing was decided after the fact | `PROTOCOL.md` §10, and commit dates |
| The reproduction result | `results/stage0.json` |
| Exact reproduction from zero | `RUNBOOK.md` |
| The analysis, runnable without a GPU | `scripts/analyse.py` |
| What I did *not* write | `src/ecgres/vendor/` — architecture vendored unmodified |

---

## What this does and does not demonstrate

My hands-on level is fine-tuning and evaluation design, not architecture design. The repository
matches that claim exactly: the model is vendored without changes, and the contribution is the
experimental design, the confound control, the cost model and the evaluation. I would rather be
judged on that than on a claim I cannot support.

---

## Why this domain

The same method could have been demonstrated on any benchmark with a confounded variable. I
chose cardiology because I came to it through a paediatric cardiology ward rather than through a
paper — congenital heart disease is close to home. I would rather say that than let the choice
look arbitrary.

It changes nothing about the method: the protocol was written to be checkable by someone who
does not care why I picked the question. It does bear on one limitation worth naming. PTB-XL is
an adult dataset, and the clinical bandwidth requirement for paediatric ECG is *higher* than for
adults — up to 250 Hz against 150 Hz, because smaller hearts put diagnostic content at higher
frequencies. So if sampling rate turns out to matter at all, it plausibly matters more where the
recordings are of children.
