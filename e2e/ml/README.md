# `e2e.ml` — FMCW MIMO radar ML dataset + perception models

## What it is

`e2e.ml` generates labeled FMCW MIMO radar training data from declarative
`e2e.scenario.Scenario` scenes — vehicles, pedestrians, and static clutter drawn at one
of four difficulty tiers (D0–D3) — and synthesizes the corresponding raw-ADC / range-Doppler
tensors analytically. No Sionna ray tracing and no GPU are required to generate a
dataset: the point-scatterer model (`scatterers.py`) and the FMCW beat-signal synthesizer
(`rd_synth.py`) are closed-form, so `python -m e2e.ml.dataset` runs on a plain CPU machine
in the same spirit as the scenario runner's `--dry-run` mode.

On top of the dataset layer sit two ported detection models — `FFTRadNet` (from
valeoai/RADIal) and `SSMRadNet` (a Mamba-style selective-state-space detector, from
AnuvabSen1/SSMRadNet) — plus a shared loss, evaluation metrics, and a reference
train/eval CLI (`e2e.ml.train`). Both models consume the exact same `[2*C, R, D]`
range-Doppler input and predict the same `[3, n_range, n_azimuth]` detection map, so they
are interchangeable on any dataset this package produces.

This package is a sibling of `e2e/comms/`: self-contained, torch-free at the geometry/scene
layer (`radar_config.py`, `scatterers.py`, `scenes.py` import no torch), with heavy tensor
work confined to `rd_synth.py`/`transforms.py`/`labels.py`/`models/`/`train.py`. It does not
touch the runtime S-parameter pipeline (`e2e/blocks.py`, `e2e/simulation.py`) or its `.pkl`
frame format — it is a separate track for training perception models on synthetic radar
scenes, not a consumer of ray-traced frames.

## Quickstart

```bash
# 1. Generate a dataset: 200 frames, TI IWR1443-class radar, difficulty tier D1.
python -m e2e.ml.dataset --config ti_iwr1443 --tier D1 --n 200 --seed 0
# --dry-run prints the generation plan (shapes, estimated size, output path) without
# synthesizing or writing anything -- useful for sizing a run before committing to it:
python -m e2e.ml.dataset --config ti_iwr1443 --tier D1 --n 200 --seed 0 --dry-run

# 2. Train a model against the manifest the previous step wrote.
python -m e2e.ml.train --manifest e2e/ml/datasets/ti_iwr1443_D1/manifest.json \
    --model fftradnet --epochs 25

# 3. Evaluate a checkpoint on the held-out test split.
python -m e2e.ml.train --manifest e2e/ml/datasets/ti_iwr1443_D1/manifest.json \
    --eval-only e2e/ml/datasets/ti_iwr1443_D1/runs/fftradnet/best.pt --split test
```

Equivalent Python (no CLI):

```python
from e2e.ml.dataset import generate_dataset, RadarFrameDataset
from e2e.ml.train import train, evaluate

manifest_path = generate_dataset("ti_iwr1443", "D1", n_frames=200, seed=0)
train_ds = RadarFrameDataset(manifest_path, split="train")

history = train(manifest_path, "fftradnet", epochs=25)
metrics = evaluate(manifest_path, manifest_path.parent / "runs" / "fftradnet" / "best.pt")
```

Generated datasets land under `e2e/ml/datasets/` (gitignored — regenerate rather than
commit); `--model` may be `fftradnet` or `ssmradnet`.

## Difficulty tiers (`e2e.ml.scenes.DIFFICULTY_TIERS`)

Each tier draws a random count of vehicles/pedestrians/clutter (uniform integer in the
given inclusive range) with speeds and RCS jitter drawn from the listed ranges; targets
are reject-sampled apart by `min_target_separation_m`. D2/D3 vehicles and pedestrians use
the "full" speed ranges (0–30 m/s / 0–3 m/s); a target's radial velocity is always clamped
to 80% of the radar config's `max_velocity_mps` so training frames never alias in Doppler.

| Tier | Vehicles | Pedestrians | Clutter | Vehicle speed (m/s) | Pedestrian speed (m/s) | RCS jitter (±dB) | Min separation (m) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D0 | 1–1 | 0–0 | 0–0 | 0.0–2.0 | 0.0–0.0 | 0.0 | 3.0 |
| D1 | 1–2 | 0–1 | 0–3 | 0.0–10.0 | 0.0–1.5 | 2.0 | 3.0 |
| D2 | 2–4 | 1–3 | 5–15 | 0.0–30.0 | 0.0–3.0 | 4.0 | 3.0 |
| D3 | 3–6 | 2–5 | 20–40 | 0.0–30.0 | 0.0–3.0 | 6.0 | 2.0 |

D0 is a single slow, stationary-ish vehicle — a sanity check, not a realistic scene. D3 is
a dense, fast, tightly-packed multi-target scene.

## Radar presets (`e2e.ml.radar_config.PRESETS`)

| Preset | MIMO | TX/RX (virtual) | Range res. | Max range | Velocity res. | Max velocity |
| --- | --- | --- | --- | --- | --- | --- |
| `ti_iwr1443` | TDM | 3×4 (12) | 0.0749 m | 38.37 m | 0.4004 m/s | 12.81 m/s |
| `radial_like` | DDMA | 12×16 (192) | 0.2000 m | 102.40 m | 0.1012 m/s | 1.06 m/s |

**`radial_like` is the default preset for detection work**, and the choice is forced by
geometry rather than taste. The detection label grid
(`LabelGrid.for_config`) has 192 azimuth bins and `metrics.MatchCriterion` matches a
detection to a target within 0.06 in sin(azimuth). An array of `n_virtual` elements
resolves no finer than ~`2 / n_virtual`: that is 0.0104 for `radial_like`'s 192 virtual
elements (one grid cell each, tolerance ≈ 6 resolution cells — the configuration the
upstream RADIal harness was built around), but 0.1667 for `ti_iwr1443`'s 12, where the
tolerance is **2.8× tighter than the array can resolve** and both AP and AR are capped by
geometry no matter which model or corpus is used. `e2e.ml.baseline.resolution_report`
prints this comparison; run it before trusting any AP on a new config. Measured
2026-08-10: on a `ti_iwr1443` corpus the classical CFAR baseline scored AP 0.0241 while a
trained FFTRadNet reached 0.0084 — the harness, not the data, set the ceiling.

`ti_iwr1443` remains available as a TI IWR1443BOOST-like mid-range profile (76–81 GHz
band) within the device's real operating envelope, and is the right choice for
signal-chain and ADC work where the detection grid is not involved.
`radial_like` reproduces the RADIal paper's
(Rebut et al., CVPR 2022) published resolution/FOV numbers (Table 5) — the paper never
states its RF chirp parameters, so `radar_config.py` solves for `f0_hz`/`fs_hz`/
`chirp_period_s` that reproduce the stated numbers; see the inline derivation comments in
`radar_config.py` for exactly which values are given vs. solved-for. Two deliberate
deviations from the paper's radar (both documented at the preset): `n_chirps=252` rather
than 256, so the DDMA replica spacing (`n_chirps/n_tx = 21`) lands on exact Doppler bins
(a fractional spacing both smears the replicas and defeats any uniformly-dilated demux,
FFTRadNet's pre-encoder included); and the max velocity honestly reflects DDMA's
code-division penalty — the unambiguous span is divided by `n_tx`, so alias-free
`radial_like` training data is limited to slow (≲1 m/s radial) targets unless you model
Doppler unfolding downstream (this package does not).

## Data format

**Network input**, `[2*C, R, D]` float32 (real channels then imaginary channels,
channel-first — see `transforms.rd_to_input`), where `R = cfg.n_samples` always, and
`C`/`D` depend on `cfg.mimo`:

| MIMO | `C` | `D` | `ti_iwr1443` shape | `radial_like` shape |
| --- | --- | --- | --- | --- |
| `"tdm"` | `n_virtual` (TX de-interleaved to the virtual array first) | `n_chirps_per_tx` | `(24, 512, 64)` | — |
| `"ddma"` / `"single"` | `n_rx` (raw ADC, no de-interleave) | `n_chirps` | — | `(32, 512, 252)` |

**Label map**, `[3, n_range, n_azimuth]` float32 (`e2e.ml.labels.LabelGrid` /
`encode_detection_labels`), e.g. `(3, 128, 192)` for the defaults above
(`range_stride=4` on `n_samples=512` → `n_range=128`; `n_azimuth=192` is a free parameter):

* channel 0 — objectness, a `1.0` 3×3 footprint centred on each target's `(range,
  sin-azimuth)` cell, `0.0` elsewhere.
* channels 1–2 — range/azimuth regression residuals, defined per footprint cell (not
  just the centre cell) so any of the 9 cells can reconstruct the exact target position.

The output grid is `(range, sin(azimuth))`, not `(range, angle_degrees)` — azimuth is
stored as the ULA direction cosine on a uniform `[-1, 1)` axis.

**Manifest** (`manifest.json`, one per `<config>_<tier>` dataset directory):

```json
{
  "config": { "...": "RadarConfig.to_dict()" },
  "tier": "D1",
  "grid": {"n_range": 128, "n_azimuth": 192, "max_range_m": 38.37},
  "snr_db": 30.0,
  "seed": 0,
  "files": {"train": ["frame_00000.npz", "..."], "val": ["..."], "test": ["..."]}
}
```

Each `frame_?????.npz` holds `input` `[2*C, R, D]`, `labels` `[3, n_range, n_azimuth]`, and
a `meta` 0-d unicode array (JSON: per-frame provenance + `targets` + `scene` summary). The
train/val/test split (default 80/10/10) is a deterministic, unshuffled slice over frame
index — reproducible for a given `(seed, n_frames)`, not random per run.

## Corpus v1 + baseline results

`corpus_v1` is a fixed, regeneratable four-tier `ti_iwr1443` corpus used for all the
numbers below (manifest_version 2, raw-ADC storage, scene-grouped `sequences` — see
"Data format"). ~14 GB on disk; regenerate any tier with:

| Tier | Frames | Seed | Regenerate |
| --- | --- | --- | --- |
| D0 | 400 | 1000 | `python -m e2e.ml.dataset --config ti_iwr1443 --tier D0 --n 400 --seed 1000 --out e2e/ml/datasets/corpus_v1` |
| D1 | 1000 | 2000 | `python -m e2e.ml.dataset --config ti_iwr1443 --tier D1 --n 1000 --seed 2000 --out e2e/ml/datasets/corpus_v1` |
| D2 | 1500 | 3000 | `python -m e2e.ml.dataset --config ti_iwr1443 --tier D2 --n 1500 --seed 3000 --out e2e/ml/datasets/corpus_v1` |
| D3 | 2000 | 4000 | `python -m e2e.ml.dataset --config ti_iwr1443 --tier D3 --n 2000 --seed 4000 --out e2e/ml/datasets/corpus_v1` |

`e2e.ml.stats_report` (run it on your regenerated corpus: `python -m e2e.ml.stats_report
--manifest <manifest.json>`; reports are local, not tracked) confirms the ladder is structurally
monotone as designed — targets/frame 1 → 7.9 and clutter/frame 0 → 29.6 from D0 to D3,
zero label-footprint overlaps at every tier — and surfaces the one caveat worth knowing
before training on D2/D3: 24.7%/22.5% of targets sit at the radial-velocity clamp cap
(`scenes.py`'s 80%-of-max-velocity anti-alias clamp), so the sampled speed range does
not translate into a proportionally wider realized Doppler distribution at those tiers.

### Recipe

A 10-trial sweep (see `e2e.ml.sweep`; run on a smaller pilot D1 — seed 500, 300 frames,
not `corpus_v1`'s D1) picked **`lr=3e-4`, `gamma=2` (focal loss), `reg_weight=100`
(upstream default, kept)** — `lr=3e-4` dominated `1e-4` at every `reg_weight` tried, and
`reg_weight` itself had little effect at that learning rate. More epochs beat more
tuning at this scale.

The sweep's `gamma=0` (plain BCE) trials looked *better* on validation — AP spikes up to
0.55–0.67 mid-training — but that is a mirage, not a signal: those gamma=0 checkpoints'
best-by-val-AP epoch has collapsed to near-zero recall (about one correct detection
across the whole split), and our AP metric excludes zero-detection confidence
thresholds from its mean instead of scoring them as zero, so a single lucky hit prints
as AP≈0.5–0.7. This reproduces on two independent datasets/sweeps, not just one run, and
the confirmed gamma=0 baseline below (trained on the full `corpus_v1`) shows the same
pattern on test. `gamma=2` was kept.

Note the sweep dataset is smaller than the confirm/baseline dataset below (pilot D1:
seed 500/300 frames vs. `corpus_v1` D1: seed 2000/1000 frames) — the recipe was picked
on the pilot set, then confirmed on the full corpus, not re-tuned on it.

### Baseline results (`corpus_v1`, confirmed)

FFTRadNet, `gamma=2`, 40 epochs / batch 8, one run per tier; SSMRadNet, same recipe
(unswept for this architecture), 12 epochs / batch 2, D1 only — a step-comparable
budget (FFTRadNet: ~4000 steps; SSMRadNet: ~4800 steps):

| Model | Tier | Test AP | Test AR | Range RMSE (matched) |
| --- | --- | --- | --- | --- |
| FFTRadNet (g2) | D0 | 0.177 | 0.183 | — |
| FFTRadNet (g2) | D1 | 0.030 | 0.127 | — |
| FFTRadNet (g2) | D2 | 0.079 | 0.103 | — |
| FFTRadNet (g2) | D3 | 0.093 | 0.095 | — |
| SSMRadNet | D1 | 0.196 | 0.304 | 0.070 m |

Matched detections still localize to the exact ground-truth range/azimuth bin at every
tier (the label/decode chain is exact); absolute AP/AR are the weak part.

Two caveats, stated exactly because it would be easy to over-read this table:

- **The D0–D3 AP ordering above is not a difficulty ranking.** Recall collapses to
  under 2% above confidence threshold 0.3 on every tier, so AP here is dominated by
  whether a handful of very-high-confidence detections per tier happen to be correct —
  it should not be read as "D1 is the hardest tier" or any other density-vs-accuracy
  claim.
- **SSMRadNet's D1 numbers are encouraging, not conclusive.** This is a single seed,
  single tier, with hyperparameters tuned for FFTRadNet (via the sweep above), not for
  SSMRadNet — an SSMRadNet-tuned recipe could move its number either direction. Its
  training curve is steady and mostly monotonic, versus FFTRadNet gamma=2's more
  volatile 40-epoch run on the same tier and a comparable step budget.
- **These are reference/pipeline-validation numbers, not a production-capable
  detector.** The best observed test AP across everything tried here is 0.196, AR at
  most ~0.37 — enough to demonstrate the dataset-generation → training → evaluation
  plumbing works end to end, not a tuned detector.

Reproduce with `e2e.ml.train`, e.g.:

```bash
python -m e2e.ml.train --manifest e2e/ml/datasets/corpus_v1/ti_iwr1443_D1/manifest.json \
    --model fftradnet --epochs 40 --batch-size 8 --lr 3e-4 --gamma 2
```

## Practical notes

* **Clutter is signal, not ground truth.** Scenes deliberately contain low-RCS static
  clutter points, but only `vehicle`/`pedestrian` objects become detection labels
  (`e2e.ml.dataset.LABEL_CLASSES`; recorded in each manifest) — a detector must learn to
  *reject* clutter. Pass `label_classes=None` to `generate_sample` if you explicitly
  want everything labeled.
* **AP semantics.** `metrics.evaluate_dataset` excludes thresholds where the model made
  no detections against existing ground truth from the AP mean (undefined precision,
  reported per-threshold as NaN) — an under-confident model does not harvest vacuous
  `precision=1.0` above its confidence ceiling. Compare AP together with AR. This
  exclusion has a sharp edge, though: when a model makes almost no detections, only 1-2
  thresholds have *any* defined precision, and AP becomes the mean of that razor-thin
  sample — a single lucky confident detection can print AP 0.5-1.0 while recall is
  near zero (this is exactly how an earlier gamma=0 checkpoint's validation AP spiked to
  0.667, see the recipe section above). `evaluate_dataset`'s result dict now carries
  `n_defined_precision_thresholds`, the count backing the AP mean — treat an AP resting
  on a small count as unstable, not as model quality.
* **SSMRadNet memory.** The pure-torch parallel selective scan (`e2e.ml.models.ssm`)
  materializes `[batch, L, d_inner, d_state]` state at every Hillis-Steele step (unlike a
  fused CUDA kernel, which never does), so it is memory-hungry: `batch_size=8` OOMs on an
  8 GB card. Use a small batch (the baseline run above used 2) or reduce `d_state`/
  sequence length.
* **`mamba_ssm` fast path is optional and off by default.** `MambaBlock(backend="auto")`
  prefers the real `mamba_ssm` fused-CUDA-kernel package when it is importable and CUDA is
  available, falling back to the pure-torch scan otherwise. `mamba_ssm` ships hand-written
  CUDA extensions (`selective_scan_cuda`, `causal_conv1d`) with **no Windows wheels**, so on
  this box the pure-torch path is what actually runs; a `mamba_ssm`-backed checkpoint is not
  portable to the `"torch"` backend (the two hold different parameter tensors) or vice versa.
* **TDM Doppler-phase residual.** `transforms.tdm_deinterleave` does not apply any
  per-target Doppler phase de-rotation across TX groups — a moving target's phase advances
  between one TX's chirps and the next (transmitted `n_tx` chirp-periods apart), so the
  de-interleaved virtual array is only exactly coherent for stationary targets. This is
  documented, deliberate, and matches the standard "raw TDM deinterleave" used for ML
  dataset generation (the network learns to cope with/exploit the residual phase, the same
  way FFTRadNet's `MIMO_PreEncoder` learns the DDMA/TDM demux end to end). See the docstring
  in `transforms.py` for the exact phase term.

## Attribution & licensing

* **`FFTRadNet`** (`e2e/ml/models/fftradnet.py`) is adapted from
  [valeoai/RADIal](https://github.com/valeoai/RADIal) (Rebut et al., "Raw High-Definition
  Radar for Multi-Task Learning", CVPR 2022).
* **`SSMRadNet`** (`e2e/ml/models/ssmradnet.py`, `e2e/ml/models/ssm.py`) is adapted from
  [AnuvabSen1/SSMRadNet](https://github.com/AnuvabSen1/SSMRadNet); the SSMRadNet authors
  requested this integration (training/evaluation on this simulator's synthetic data).

**Neither upstream repository ships a `LICENSE` file.** These adaptations carry
attribution headers identifying the upstream source and documenting every deviation
inline, and are used under research-community attribution norms. Status:

* **SSMRadNet — approved.** The authors gave written approval for this adaptation's
  redistribution (2026-08-07), on top of having requested the integration.
* **RADIal — inquiry sent (2026-08-07), proceeding with attribution.** The upstream
  project's code and paper evidence open-research intent; this non-commercial,
  attributed adaptation proceeds on that basis and **will be removed or relicensed
  promptly if the authors object**. (If you fork this repo for commercial use, obtain
  your own clarity from the RADIal authors first.)
