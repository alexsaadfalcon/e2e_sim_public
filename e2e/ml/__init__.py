"""
Machine-learning dataset layer for the end-to-end array-processing simulator.

This package builds FMCW MIMO radar training data (range-Doppler / range-angle
tensors + labels) for radar perception models (e.g. RADIal / FFT-RadNet-style
architectures). It is a sibling of `e2e/comms/`: self-contained, and does NOT
require Sionna or any precomputed `.pkl` frames to build synthetic scenes.

C1 layering (`notes/C1_MOVE_PLAN.md`): the core FMCW/RT signal model used to live
here but has moved out, so `e2e.ml` is a CONSUMER of that core, not its owner --
`e2e/chain` and `e2e/environment` never import `e2e.ml` at module scope. The old
`e2e.ml.<module>` import paths for everything that moved are kept as deprecated
`DeprecationWarning` shims through v1.1 (removal left to successors); import the new
paths below in new code.

Moved out of this package
--------------------------
* `radar_config`   -- now `e2e.radar_config` (dependency-free, sibling of
  `e2e/scenario.py`; not ML-specific).
* `geometry`, `scatterers` -- now `e2e.environment.{geometry,scatterers}` (scene
  geometry / scatterer-pose bridge; core to the RT path too, not ML-specific).
* `rd_synth`, `transforms`, `impairments`, `link_budget` -- now `e2e.chain.*` (the
  heavy-tensor signal model; `e2e/chain/receive.py` already wrapped them).
* `assets`, `rt_scene_build`, `rt_signal_chain`, `rt_doppler_study`, `rt_gen`,
  `rt_scenes` -- now `e2e.environment.*` (the Sionna RT scene-build/signal-chain
  sextet; `e2e/environment/` owns ray tracing).
* `render_scene` -- now `e2e.render_scene` (leaf visualization tool, beside
  `e2e/viz.py`).

Two lazy exceptions remain, both documented at their call site: `e2e.radar_config`
imports `e2e.ml.metrics.MatchCriterion` lazily (inside a function) for its
answerability check, and `e2e.environment.blocks.RTEnvironmentBlock` imports
`e2e.ml.labels` lazily for its label encoder -- neither is a module-scope import, so
the core-never-imports-ml rule holds.

Sub-modules remaining in this package
--------------------------------------
* `scenes`      -- randomized vehicle/pedestrian/clutter scene sampler with
  difficulty tiers D0-D3 (`DIFFICULTY_TIERS`); torch-free.
* `labels`      -- LabelGrid (range x sin-azimuth output geometry) + FFTRadNet-
  style detection-label encoding (3x3 footprint on the target's SURFACE + per-cell
  residuals toward its CENTRE) and the matching decoder.
* `dataset`      -- end-to-end sample/dataset generation (scenario -> ADC -> RD
  input + labels -> .npz + manifest), `RadarFrameDataset`, and the
  `python -m e2e.ml.dataset` CLI. Datasets land in the gitignored
  `e2e/ml/datasets/`.
* `chain_generate` -- generates the same dataset schema by running frames through
  the composed pipeline block chain (RT-consuming path), instead of `dataset.py`'s
  direct closed-form synthesis.
* `models`, `train`, `losses`, `metrics` -- `FFTRadNet`/`SSMRadNet` model ports, the
  shared loss, evaluation metrics, and the `python -m e2e.ml.train` train/eval CLI.
* `compare_detectors`, `stats_report`, `check_corpus_overlap`, `sweep`,
  `detect_viz`, `export_corpus`, `export_ssm` -- corpus/benchmark tooling: baseline
  comparison, corpus statistics, train/eval leak detection, hyperparameter
  sweeps, detection visualization, and dataset/checkpoint export.

Mirrors `e2e/comms/__init__.py`: nothing heavy is imported at package import
time, so `import e2e.ml` and `import e2e.ml.scenes` stay torch-free.
"""
