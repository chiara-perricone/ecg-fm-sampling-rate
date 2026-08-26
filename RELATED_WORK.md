# Related work

This document establishes the premise of the repository: that in the current
generation of ECG foundation model benchmarks, **sampling rate varies together
with the architecture**, that the benchmarks' own authors treat sampling rate as
a design variable requiring compensation when they control it, and that nobody
has quantified how much of the reported spread between models it accounts for.

It is a review, not an experiment. The experiment is specified in
`PROTOCOL.md` and implemented in `src/`.

## Verification convention

Every factual claim below is tagged with how it was established:

- **[code]** — read directly from source code, with file and line.
- **[paper]** — read directly from the official publication, with section.

Nothing here is inferred from secondary sources or from arithmetic on reported
numbers. Where an inference is offered, it is labelled as an inference and the
observable fact it rests on is given separately.

Code references are to the following commits:

| Repository | Commit |
|---|---|
| `AI4HealthUOL/ecg-fm-benchmarking` | `2384098` (2026-06-29) |
| `dlaskalab/bench-xecg` | cloned 2026-08-25 |
| `control-spiderman/ECGFM-KED` | cloned 2026-08-25 |

## 1. The benchmark under examination

Al-Masud, Lopez Alcaraz and Strodthoff, *Benchmarking ECG FMs: A Reality Check
Across Clinical Tasks* (ICLR 2026, arXiv:2509.25095v2), evaluate eight ECG
foundation models and two supervised baselines across 26 clinical tasks on 12
public datasets. It is the most systematic comparison of ECG foundation models
currently available, and it is a careful piece of work. It states its own
principal limitation explicitly: because the models were pretrained on different
datasets, isolating which factors drive performance differences is difficult, and
retraining on a unified corpus would be computationally prohibitive
**[paper: §5, Limitations]**. It goes further and calls for like-for-like
pretraining comparisons on a common dataset with standardised architectures
**[paper: §5, Pretraining strategies]**.

This repository takes that call seriously and applies it to a variable the paper
does not name: the sampling rate.

The margins at stake are small. On PTB-XL (all) under finetuning, macro AUROC
runs from 0.949 (ECG-CPC) to 0.889 (ECGFM-KED), with adjacent positions
frequently separated by 0.005–0.010 **[paper: Table 3]**. The benchmark also
declines to fix a global random seed, citing robustness recommendations
**[paper: §3.3]**, so run-to-run variation is deliberately not held constant
either.

## 2. What varies across the ranking, and what is disclosed

The benchmark's launch script sets two independent quantities per model:
`--fs-model`, the rate the signal is delivered at, and `--input-size`, the
window length, documented as *"input size (in seconds)"* and parsed as a float
**[code: `code/main_lite.py:421,440`]**. Reading the per-model branches of
`run.sh` **[code: `run.sh:150–260`]** and cross-checking against the paper's
reported timestep counts **[paper: Table 32]**:

| Model | Rate | Window | Samples | Table 32 | Leads |
|---|---|---|---|---|---|
| ECGFounder | 500 Hz | 2.5 s | 1250 | 1250 | 12 |
| ECG-JEPA | 250 Hz | 10 s | 2500 | 2500 | **8** |
| ST-MEM | 250 Hz | 2.4 s | 600 | 600 | 12 |
| MERL (ResNet) | 500 Hz | 2.5 s | 1250 | 1250 | 12 |
| ECGFM-KED | 500 Hz | 10 s | 5000 | 5000 | 12 |
| ECG-FM | 500 Hz | 5 s | 2500 | 2500 | 12 |
| ECG-CPC | 240 Hz | 2.5 s | 600 | 600 | 12 |
| HuBERT-ECG | 100 Hz | 5 s | 500 | 500 | 12 |
| S4 (baseline) | 100 Hz | 2.5 s | 250 | — | 12 |
| Net1D (baseline) | 500 Hz | 2.5 s | 1250 | — | 12 |

Every derived sample count matches the published table.

**Window length is disclosed.** The paper states that where the architecture
supports it, 2.5-second segments are used instead of full 10-second recordings,
because longer windows cost compute for little gain, and that at inference
predictions are averaged over four non-overlapping 2.5-second segments
**[paper: §3.3]**. The Discussion returns to it: adjustable input size matters,
and 2.5-second crops with test-time averaging outperform full 10-second inputs
**[paper: §5, Details matter]**. This is properly reported, and any claim that
the benchmark conceals its window-length heterogeneity would be wrong.

**Sampling rate is not.** Table 1 characterises the ten models by backbone,
pretraining method, pretraining dataset and parameter count — not by sampling
rate **[paper: Table 1]**. Table 32 reports timesteps, which is the *product* of
rate and duration and cannot be decomposed into either **[paper: Table 32]**:
600 timesteps is 2.5 s at 240 Hz for ECG-CPC and 2.4 s at 250 Hz for ST-MEM;
2500 is 10 s at 250 Hz for ECG-JEPA and 5 s at 500 Hz for ECG-FM. The only two
models whose sampling rate appears anywhere in the paper are ECG-CPC (240 Hz)
and S4 (100 Hz) **[paper: §A.2.1, §A.2.2]** — the two the authors built
themselves. For the eight models taken from the literature, the rate is
recoverable only from the launch script in the repository.

So the spread — 100, 240, 250 and 500 Hz, a factor of five between HuBERT-ECG
and ECGFounder — is real, is not an artefact of one team's pipeline (§4), and is
not visible to a reader of the publication.

**The resampling is stated not to occur.** §3.3 says that "all models use the
standard 12 ECG leads without additional resampling or filtering"
**[paper: §3.3]**. The configuration says otherwise: `ptbxl_all` is loaded from
`.../ptb-xl/records500` with `--fs-data 500`, while nine of the ten models
declare an `--fs-model` other than 500 **[code: `run.sh`]**. Signals must
therefore be resampled from 500 Hz to 100, 240 or 250 Hz for nine of the ten
entries in the ranking, and the sentence a reader would rely on says they are
not. The charitable reading — that "additional" means "beyond what the pipeline
performs natively" — is probably the intended one, in which case the wording is
unfortunate rather than incorrect. Either way the operation whose effect this
repository measures is one that a careful reader of the paper would conclude had
not been performed.

**The split is not described either.** The word "fold" does not appear in the
paper or its appendices, and no split ratio or procedure is given for PTB-XL;
§3.3 states only that model selection uses the validation set and that
uncertainty comes from bootstrapping the test set with n = 1000
**[paper: §3.3]**. The official stratified folds are the standard convention for
this dataset and the underlying `clinical_ts` code uses them, so the reader can
infer the split with reasonable confidence — but only by inference. This
repository uses folds 1–8 / 9 / 10 and records that correspondence as an
assumption rather than a verified match (PROTOCOL §10, entry 18).

## 3. The authors know sampling rate is not neutral

This is the strongest evidence in this document, and it comes from the benchmark
itself.

ECG-CPC is the model the authors introduce, and it finishes first on PTB-XL
(all) and leads five of seven task categories. Its appendix reads: the
architecture largely follows Mehari & Strodthoff (2023), but the model operates
at 240 Hz, the minimum sampling frequency in the HEEDB pretraining corpus; and
**to account for the deviation from the sampling frequency in the original
publication**, the first convolutional layer uses kernel size 3 and stride 2,
and the CPC objective predicts 14 steps ahead instead of 12
**[paper: §A.2.1]**.

Read that as a statement about the variable. Moving a model from 100 Hz to
240 Hz required changing the receptive field of the input layer and the
prediction horizon of the training objective. The authors identified the
dependency, and engineered around it, and documented both.

The same paper then ranks that model against eight others whose sampling rates
were fixed by whoever released the checkpoint, were never re-tuned, and are not
reported. The benchmark is scrupulous about the factor exactly where it has
control over it, and silent about it everywhere else. That is not a criticism of
the authors' integrity — it is the structural consequence of evaluating released
checkpoints, which they defend on grounds of practical utility
**[paper: §5, Limitations]**. But it means the one place we can observe the
field's own judgement about whether sampling rate matters, the answer is that it
matters enough to redesign the input stage for.

Two model authors reached the same conclusion independently:

**HuBERT-ECG.** The authors state that sampling rate has no standard value and
regulates how far the information content is diluted, that they therefore
investigated its effect on both upstream and downstream performance, and that
they found 100 Hz to give the best trade-off between downstream performance and
training time **[paper: medRxiv 10.1101/2024.11.14.24317328, Methods; experiment
in Supplementary Information Sec. 1.2]**.

**ST-MEM.** Their appendix notes that CPC was pretrained at 100 Hz while their
own default is 250 Hz, and reports both configurations in an ablation
**[paper: arXiv:2402.09450, App. B.1, Table 13]**.

Sampling rate is thus a hyperparameter tuned per model, on that model's own
objective, during that model's development — and then carried into cross-model
comparison as though it were a fixed architectural property. What this repository
measures is what that costs.

## 4. Independent corroboration: BenchECG

BenchECG (Lunelli et al., arXiv:2509.10151) is a separate benchmark by a separate
group, posted two weeks before the Reality Check. Its downstream defaults set one
rate for every dataset — `sampling_freq: 100`
**[code: `config_defaults/train_ptb_xl_defaults.yaml:36`]** — which is then
overridden per model at config-merge time, under a comment saying it is more
convenient to hardcode the frequency and other important parameters there than in
the config files **[code: `bench_xecg/config.py:56–81`]**:

| Model | Rate | Also overridden |
|---|---|---|
| ECG-JEPA | 250 Hz | patch size 50; **8 leads** (I, II, V1–V6) |
| ST-MEM | 250 Hz | patch size 75; bandpass 0.67–40 Hz; standardisation |
| ECGFounder | 500 Hz | bandpass 0.5–50 Hz; no layerwise LR decay |
| ECG-CPC | 240 Hz | patch size 2; no layerwise LR decay |
| xECG (proposed) | 100 Hz | (inherits the default) |

Four rates, identical to the Reality Check's, in code that shares no lineage with
it. The per-model rate is a property of the released checkpoints, not of one
pipeline. Filtering varies per model here too, which is what the per-model
alternative looks like when written out explicitly.

BenchECG also inherits the asymmetry described in §3: the model its authors
propose runs at 100 Hz while its competitors run at 240–500 Hz. Nothing follows
from this about the validity of their results — each model at its pretraining
rate is the principled choice — but it is again the situation in which knowing
the size of the effect would matter.

## 5. Configuration findings

### 5.1 ECGFM-KED is run at five times its pretraining rate

The Reality Check's wrapper docstring declares *Model sampling frequency: 100 Hz*
**[code: `code/clinical_ts/models/fm_ecg.py:636`]**. The launch script passes
`--fs-model 500` **[code: `run.sh`, `ecgfm_ked` branch]**, and the 5000 timesteps
in Table 32 confirm 500 Hz × 10 s is what ran. No resampling occurs inside the
wrapper to reconcile them: the file contains no call to `Resample`, `interpolate`
or `decimate` in any model class.

The KED authors' own code settles it. Their PTB-XL preprocessing defaults to
`sampling_frequency=100`, producing arrays of shape (21799, 1000, 12) — 1000
samples for a 10-second recording **[code:
`dataset/ptb-xl/data_preprocess.py:20,26`]**. Their CPSC, Georgia and Shaoxing
preprocessors each resample by a factor of five from the native 500 Hz
**[code: `dataset/cpsc/data_preprocess.py:38`,
`dataset/georgia/data_preprocess.py:65`,
`dataset/shaoxing/data_preprocess.py:57`]**. ECGFM-KED is a 100 Hz model.

The docstring is right; the launch configuration is not. The model that finishes
last on PTB-XL (all) at 0.889 was evaluated at five times the rate it was
pretrained at.

A second configuration choice compounds this. The benchmark applies
layer-dependent learning rates during finetuning, and reports an ablation:
ECGFM-KED is the only model of the ten that is *harmed* by them, scoring 0.889
with and 0.918 without on PTB-XL (all) **[paper: Table 33]**. The headline table
reports 0.889. At 0.918 the model would sit mid-field rather than last.

Neither observation establishes that misconfiguration explains KED's last place,
and there is direct evidence against that reading. Berger et al. (§6) also feed
KED at 500 Hz — they standardise every model's input to 500 Hz, 10 s
**[paper: arXiv:2602.17531, §5.2]** — and under linear probing KED is at or near
the top of their tables, reaching 0.909 macro-AUROC on PTB-XL SUPER at full
label availability **[paper: same, Table 1]**. The same model, at the same
mismatched rate, finishes last in one benchmark and first-equal in another.

That comparison is worth more than the anomaly that produced it. It says the
rate mismatch is not by itself disabling, and that protocol differences between
the two evaluations — finetuning with layer-dependent learning rates versus a
frozen encoder with ℓ2-regularised logistic regression — move a model further
than a five-fold change in sampling rate does. Which is a reason to measure the
sampling-rate effect rather than assume it, and a reason not to assume it is
large.

### 5.2 ST-MEM's 2.4-second window is quantisation, not error

ST-MEM is the only model whose window is not 2.5, 5 or 10 seconds
**[code: `run.sh`, `st_mem` branch]**. The explanation is in the architecture:
ST-MEM tokenises in patches of 75 samples **[code:
`code/clinical_ts/models/fm_ecg.py:353`; corroborated in
`bench_xecg/models/st_mem/encoder/st_mem_vit.py:244`]**, which at 250 Hz is
0.3 seconds per patch. The benchmark's 2.5-second target is 8.33 patches; 2.4 s
is 8 patches exactly. The window is the largest whole number of patches that fits
under the intended crop.

This is a principled choice, not a mistake, and an earlier draft of this document
was wrong to suggest otherwise. It carries a consequence worth stating: **window
duration is quantised differently by each architecture**, so "hold the window
constant in seconds" is not fully achievable across a set of released
checkpoints. Duration and rate cannot both be controlled in a
multi-architecture comparison. They can be in a single-architecture one, which is
what `PROTOCOL.md` specifies.

### 5.3 A reproducibility note on the released resampling path

In `main_lite.py`, both branches of the resampling block are guarded by
`len(memmap_meta)==0`, and the first would then index `memmap_meta["fs"]` on an
empty dictionary **[code: `code/main_lite.py:178–182`]**.
`load_memmap_meta_dict` returns a plain dictionary, empty only when the `meta`
key is absent **[code:
`code/clinical_ts/data/time_series_dataset_utils.py:432–449`]**. With a populated
meta file — the ordinary case — no `Resample` transform is appended. The sibling
entry point `main_lite_base.py` contains the unguarded and evidently intended
form **[code: `code/main_lite_base.py:297–298`]**.

The released file most likely diverges from the one that produced the published
results: were resampling actually skipped, ECG-JEPA's input assertion
(`x.shape[2] == 2500` **[code: `code/clinical_ts/models/fm_ecg.py:266`]**) would
fail on 500 Hz PTB-XL data. Recorded for anyone re-running the benchmark, not as
a criticism of its results.

### 5.4 Lead count also varies

ECG-JEPA runs with eight input channels, every other model with twelve
**[code: `run.sh`]**; BenchECG hardcodes the same eight-lead subset
**[code: `bench_xecg/config.py:60`]**. A third factor that varies with the
architecture.

### 5.5 The 0.9417 lineage, and what the older pipeline actually does

The Reality Check places its own 0.941 beside 0.9417 from Mehari & Strodthoff
**[paper: §4.1]**. That second figure comes from
`helme/ecg_ptbxl_benchmarking`, which is GPL-3.0 and was therefore inspected and
not reused. Three properties of it are worth recording.

**Signal source at 100 Hz.** `load_raw_data_ptbxl` reads `df.filename_lr`, i.e.
the distributed `records100`, when `sampling_rate == 100`
**[code: `code/utils/utils.py`]**, and `get_datasets.sh` fetches PTB-XL v1.0.1
**[code: `get_datasets.sh`]**. The Reality Check's own pipeline does neither: it
reads `records500` at v1.0.3 (§2). The two numbers the paper sets side by side
therefore rest on different signal provenance and different dataset versions.
The gap between them is 0.0007.

**Normalisation.** `preprocess_signals` fits one `StandardScaler` on
`np.vstack(X_train).flatten()[:, np.newaxis]`, and `apply_standardizer`
transforms each record through `x.flatten()[:, np.newaxis]`
**[code: `code/utils/utils.py`]**. A single scalar mean and standard deviation,
shared across all twelve leads and all timesteps, fitted on the training folds
over whole records before any chunking. This is what PROTOCOL §10 entries 7 and
12 replicate; both were written from this file and are now verified against it
rather than asserted.

**Undefined labels.** `evaluate_experiment` calls
`roc_auc_score(y_true, y_pred, average='macro')` over all label columns, and
`get_appropriate_bootstrap_samples` draws resamples in a loop, discarding any in
which some label has no positive example **[code: `code/utils/utils.py`]**.
Undefined columns are thus avoided by restricting the resample space rather than
dropped from the average. The published intervals are correspondingly
conditional on resamples in which every label is represented, which is not the
unconditional sampling distribution. This repository takes the other route —
drop the undefined column from that replicate and report how often it happens
(PROTOCOL §7) — and the difference is recorded rather than treated as
equivalent.

## 6. Other work

**ECG-InterpBench** (arXiv:2607.27404) benchmarks interpretability rather than
accuracy across six ECG encoders, and states the heterogeneity problem directly:
the released channel selection, sample rate, duration, tokenisation and pooling
interfaces differ, and the authors preserve those interfaces and standardise only
their own comparison coordinates — relative depth and SAE capacity. They quantify
the resulting spread of the inputs: between eight and twelve leads, and between
2,250 and 5,000 samples **[paper: arXiv:2607.27404, §2]**. This is the closest
published work to the present question. It measures the spread of the inputs; it
does not measure what that spread does to the outputs.

**Strodthoff et al. (2021)**, the original PTB-XL benchmark, established the
protocol this repository uses: the ten official stratified folds with 1–8 for
training, 9 for validation and 10 for testing; macro-averaged AUC over class-wise
AUCs as the primary metric, chosen because it requires no thresholding; and 95%
confidence intervals from bootstrapping on the test set
**[paper: IEEE JBHI 25(5):1519–1528, §II, §III-A]**. `src/ecgres/stats.py`
follows this convention, extended to the paired case.

**Berger, Prakah-Asante, Guttag and Stultz**, *Position: Evaluation of ECG
Representations Must Be Fixed* (ICML 2026, arXiv:2602.17531v2), is the nearest
existing work to this one, and requires care to position correctly.

Its headline arguments are about task selection, metrics and baselines: that
evaluation has converged on three arrhythmia/morphology benchmarks and should
expand to structural disease, hemodynamics and patient forecasting; that
macro-AUROC point estimates without uncertainty are unreliable; and that a
randomly initialised encoder under linear probing matches state-of-the-art
pretraining on many tasks **[paper: §1, §5.4.1]**. Sampling rate is not among
its claims.

**But it contains the only published measurement adjacent to the present
question.** To test whether the random-encoder result is an artefact of one
configuration, the authors evaluate a grid of 36 randomly initialised encoders
crossing **sampling rate (100, 250 or 500 Hz)**, filtering (none or 5th-order
Butterworth 0.5–40 Hz), normalisation (none, dataset-level or per-sample
z-score) and backbone (ResNet18 or ViT-Medium), and report mean macro-AUROC with
standard deviation across the grid **[paper: §5.4.4, Table 6]**. Backbone
dominates: on PTB-XL SUPER the full grid gives 0.82 ± 0.07, while the ResNet
subset gives 0.87 ± 0.01 and the ViT subset 0.76 ± 0.07. Within the ResNet
subset — 18 configurations spanning all three sampling rates — the standard
deviation across every public dataset is 0.01 to 0.03.

That number bounds what this repository should expect to find, and it does not
supersede the measurement. Four reasons: the encoders are untrained, so the
result speaks to how much information random convolutions preserve, not to what
a model *learns* at each rate; sampling rate is not isolated, being crossed with
filtering and normalisation inside that standard deviation; a standard deviation
across a grid is not the marginal effect of one factor, and no per-rate
breakdown is reported; and evaluation is linear probing on frozen features
rather than training from scratch.

It is nonetheless the right order-of-magnitude prior, and it cuts toward the
effect being small — 0.01 to 0.03 across three crossed factors. Note, though,
that the gaps this repository compares against are 0.005–0.010. A factor whose
plausible range is 0.01 is not negligible relative to differences of 0.005.

Two further things are worth taking from this paper rather than merely citing.
Its preprocessing standardises every model to 500 Hz and 10 s regardless of
pretraining rate **[paper: §5.2]** — a third design point, distinct from the
per-model preprocessing of the Reality Check and BenchECG, and a demonstration
that uniform preprocessing is a live option. And its recommended protocol is
the one `src/ecgres/stats.py` already implements: bootstrapped confidence
intervals rather than point estimates, **paired** comparisons for claims of
improvement, and exclusion of labels with insufficient test support
**[paper: §4]**. The last of these is not yet implemented here; see §7.

**MERL** (arXiv:2403.06659) built the first cross-dataset benchmark of ECG
self-supervised methods, and is itself one of the eight models evaluated by the
Reality Check.

Three designs are therefore in use: per-model preprocessing that follows each
checkpoint's pretraining rate (Reality Check, BenchECG), uniform preprocessing
that ignores it (Berger et al.), and preservation of released interfaces with
standardisation elsewhere (ECG-InterpBench). Each is defended and each is
defensible. What none of them provides is the number that would let a reader
choose between them: how much macro AUROC moves when only the sampling rate
changes, for a model that was actually trained at that rate.

## 7. What this repository adds, and what it does not claim

**Adds.** The marginal effect of sampling rate on a *trained* model, isolated.
Holding architecture, task, window duration in seconds, schedule and seed fixed,
and training from scratch at several sampling rates, it measures how far macro
AUROC on PTB-XL moves when only the sampling rate changes — and compares that
movement against the gaps the Reality Check's leaderboard treats as differences
between models.

The distinction from Berger et al. §5.4.4 is the whole contribution and should be
stated plainly in any write-up: they vary sampling rate alongside two other
preprocessing factors, across untrained encoders, under linear probing, and
report a pooled standard deviation. This varies sampling rate alone, across
models trained at each rate, and reports a per-rate estimate with paired
bootstrap confidence intervals. Their 0.01–0.03 is the prior; this is the
measurement.

**Does not claim.** It does not re-rank the foundation models: impossible with
public checkpoints, each bound to its pretraining rate, and beyond the compute
budget of this work. It does not assert the published ranking is wrong, and §5.1
records evidence that at least one rate mismatch is not disabling. It answers the
question the Reality Check's own §A.2.1 raises and does not pursue — how much does
the sampling rate move the number — and leaves the reader to judge what follows.

**Adopted from Berger et al. §4.** Two protocol items, recorded here so they are
not lost: results should be reported per label as well as macro-averaged, and
labels with fewer than ten positive test examples should be excluded from
quantitative evaluation, with both the original and cleaned macro-AUROC shown.
PTB-XL (all) has thirteen such labels; removing them can reorder methods
**[paper: arXiv:2602.17531, §5.4.2, Tables 4, 23–25]**. `src/ecgres/metrics.py`
currently handles labels that become undefined under resampling but does not
implement the support threshold. `PROTOCOL.md` should specify it before any run.
One refinement: their pairwise testing uses Bonferroni correction, where
`stats.py` uses Holm, which is uniformly more powerful at the same family-wise
error rate.

The blocking condition is in `PROTOCOL.md`: before any rate is varied, the S4
baseline on PTB-XL (all) at 100 Hz must be reproduced. The target is 0.941, the
Reality Check's own supervised result **[paper: §4.1]**, which they report
against 0.9417 from Mehari & Strodthoff (2023). If it is not reproduced, the work
stops.

## 8. Reproducing the tables

```bash
# Reality Check (§2, §5.1–5.3)
git clone https://github.com/AI4HealthUOL/ecg-fm-benchmarking.git
cd ecg-fm-benchmarking && git checkout 2384098
grep -n -A6 'ecg_founder\|ecg_jepa_multiblock\|st_mem\|merl_resnet\|ecgfm_ked' run.sh
grep -n "Model sampling frequency" code/clinical_ts/models/fm_ecg.py
sed -n '176,184p' code/main_lite.py

# BenchECG (§4)
git clone https://github.com/dlaskalab/bench-xecg.git
sed -n '54,82p' bench-xecg/bench_xecg/config.py

# ECGFM-KED (§5.1)
git clone https://github.com/control-spiderman/ECGFM-KED.git
sed -n '20,26p' ECGFM-KED/dataset/ptb-xl/data_preprocess.py
```

## References

- Al-Masud, Lopez Alcaraz, Strodthoff. *Benchmarking ECG FMs: A Reality Check
  Across Clinical Tasks.* ICLR 2026. arXiv:2509.25095v2.
  Code: `AI4HealthUOL/ecg-fm-benchmarking`.
- Lunelli, Nicolson, Pröll, Reinstadler, Bauer, Dlaska. *BenchECG and xECG: a
  benchmark and baseline for ECG foundation models.* arXiv:2509.10151.
  Code: `dlaskalab/bench-xecg`.
- *ECG-InterpBench: Benchmarking the Interpretability of ECG Foundation Models
  with Matched-Scale Sparse Autoencoders.* arXiv:2607.27404.
- Berger, Prakah-Asante, Guttag, Stultz. *Position: Evaluation of ECG
  Representations Must Be Fixed.* ICML 2026, PMLR 306. arXiv:2602.17531v2.
  Code: `zackeberger/ecg-fix`.
- Strodthoff, Wagner, Schaeffter, Samek. *Deep Learning for ECG Analysis:
  Benchmarks and Insights from PTB-XL.* IEEE JBHI 25(5):1519–1528, 2021.
  DOI 10.1109/JBHI.2020.3022989. arXiv:2004.13701.
- Mehari, Strodthoff. *Towards Quantitative Precision for ECG Analysis:
  Leveraging State Space Models, Self-supervision and Patient Metadata.*
  IEEE JBHI 27(11):5326–5334, 2023.
- Wagner et al. *PTB-XL, a large publicly available electrocardiography dataset.*
  Scientific Data 7, 154 (2020). PhysioNet v1.0.3, DOI 10.13026/kfzx-aw45,
  CC BY 4.0.
- ST-MEM: Na et al. *Guiding Masked Representation Learning to Capture
  Spatio-Temporal Relationship of Electrocardiogram.* ICLR 2024.
  arXiv:2402.09450.
- ECG-JEPA: Kim. *Learning General Representation of 12-Lead Electrocardiogram
  with a Joint-Embedding Predictive Architecture.* arXiv:2410.08559.
- ECG-FM: McKeen et al. *ECG-FM: An Open Electrocardiogram Foundation Model.*
  JAMIA Open 8(5), ooaf122 (2025). arXiv:2408.05178.
- ECGFounder: Li et al. *An Electrocardiogram Foundation Model Built on over 10
  Million Recordings.* NEJM AI 2(7), 2025. arXiv:2410.04133.
- MERL: Liu et al. *Zero-Shot ECG Classification with Multimodal Learning and
  Test-time Clinical Knowledge Enhancement.* ICML 2024. arXiv:2403.06659.
- ECGFM-KED: Tian et al. *Foundation model of ECG diagnosis.* Cell Reports
  Medicine 5(12):101875, 2024. Code: `control-spiderman/ECGFM-KED`.
- HuBERT-ECG. medRxiv 10.1101/2024.11.14.24317328.
