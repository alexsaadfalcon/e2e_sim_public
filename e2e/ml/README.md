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

`ti_iwr1443` is a TI IWR1443BOOST-like mid-range profile (76–81 GHz band) chosen to be
within the device's real operating envelope. `radial_like` reproduces the RADIal paper's
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

## Smoke-train results

CUDA smoke test on `ti_iwr1443` / tier D0 (160 frames, 128 train). These are
**plumbing-verification numbers, not benchmarks** — no hyperparameter search, small
batches (memory-bound for SSMRadNet, see below), and AP is measured with the honest
metric (vacuous thresholds excluded — see "AP semantics" below; an earlier draft
reported AP 0.78 for the same weights purely through that inflation, see CHANGELOG):

| Model | Epochs / batch | Loss (start → end) | Held-out test AP | Test AR | Range RMSE (matched) |
| --- | --- | --- | --- | --- | --- |
| FFTRadNet | 120 / 8 | 36.7k → 61 | 0.31 | 0.19 | 0.000 m |
| SSMRadNet | 40 / 2 | 5.3k → 53 | ~0.00 | 0.10 | 0.000 m |

What this smoke does and does not establish: losses fall by ~3 orders of magnitude,
recall is real, and matched detections localize to **exactly the right bin** (0.000 m
RMSE — the label/decode chain is exact); but absolute AP at this scale is weak, and the
loss balance inherited from upstream (`reg_weight=100`, tuned for RADIal's real data)
visibly starves the objectness term on these synthetic scenes. Proper training runs and
loss/hyperparameter tuning are deliberately downstream users' work (tracked as a
ROADMAP follow-up).

## Practical notes

* **Clutter is signal, not ground truth.** Scenes deliberately contain low-RCS static
  clutter points, but only `vehicle`/`pedestrian` objects become detection labels
  (`e2e.ml.dataset.LABEL_CLASSES`; recorded in each manifest) — a detector must learn to
  *reject* clutter. Pass `label_classes=None` to `generate_sample` if you explicitly
  want everything labeled.
* **AP semantics.** `metrics.evaluate_dataset` excludes thresholds where the model made
  no detections against existing ground truth from the AP mean (undefined precision,
  reported per-threshold as NaN) — an under-confident model does not harvest vacuous
  `precision=1.0` above its confidence ceiling. Compare AP together with AR.
* **SSMRadNet memory.** The pure-torch parallel selective scan (`e2e.ml.models.ssm`)
  materializes `[batch, L, d_inner, d_state]` state at every Hillis-Steele step (unlike a
  fused CUDA kernel, which never does), so it is memory-hungry: `batch_size=8` OOMs on an
  8 GB card. Use a small batch (the smoke test above used 2) or reduce `d_state`/sequence
  length.
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
