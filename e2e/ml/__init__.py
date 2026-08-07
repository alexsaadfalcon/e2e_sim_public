"""
Machine-learning dataset layer for the end-to-end array-processing simulator.

This package builds FMCW MIMO radar training data (range-Doppler / range-angle
tensors + labels) for radar perception models (e.g. RADIal / FFT-RadNet-style
architectures). It is a sibling of `e2e/comms/`: self-contained, and does NOT
require Sionna or any precomputed `.pkl` frames to build synthetic scenes.

Sub-modules
-----------
* `radar_config` -- dependency-free `RadarConfig` dataclass (chirp/frame timing,
  derived range/velocity resolution and limits) plus reference presets
  (`TI_IWR1443`, `RADIAL_LIKE`). No torch/numpy required.
* `scatterers`   -- point-scatterer scene generation for synthetic radar targets.
* `rd_synth`     -- raw-ADC cube synthesis (dechirped FMCW beat signal, shape
  [n_rx, n_chirps, n_samples]) from scatterers + RadarConfig; range-Doppler
  tensors are derived from it via `transforms`.
* `transforms`   -- ADC -> range-Doppler transforms, TDM virtual-array
  deinterleave, real/imag input packing, per-channel normalization stats.
* `labels`       -- LabelGrid (range x sin-azimuth output geometry) + FFTRadNet-
  style detection-label encoding (3x3 footprint + per-cell residuals) and the
  matching decoder.
* `scenes`       -- randomized vehicle/pedestrian/clutter scene sampler with
  difficulty tiers D0-D3 (`DIFFICULTY_TIERS`).
* `dataset`      -- end-to-end sample/dataset generation (scenario -> ADC -> RD
  input + labels -> .npz + manifest), `RadarFrameDataset`, and the
  `python -m e2e.ml.dataset` CLI. Datasets land in the gitignored
  `e2e/ml/datasets/`.

Mirrors `e2e/comms/__init__.py`: nothing heavy is imported at package import
time, so `import e2e.ml` and `import e2e.ml.radar_config` stay torch-free.
"""
