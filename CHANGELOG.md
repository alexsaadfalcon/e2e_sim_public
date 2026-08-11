# Changelog

All notable changes to this project are documented here. The format is loosely based
on [Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
semantic versioning.

## [Unreleased]

### Changed
- **The detection path now defaults to the `radial_like` preset (12 TX x 16 RX, 192
  virtual elements).** The 192-bin azimuth label grid is inherited from RADIal, where
  192 *is* the virtual-element count; the 0.06 sin(az) match tolerance is this project's
  own choice for a polar criterion (RADIal itself matches on cartesian box IoU >= 0.5).
  Pointing that harness at `ti_iwr1443` (12 virtual elements, Rayleigh 0.167) demanded
  azimuth accuracy 2.8x finer than the array can resolve, capping AP and AR by geometry
  rather than by model or corpus quality. `ti_iwr1443` remains available and remains
  right for signal-chain/ADC work. Measured cost of the switch, one D1 frame: solve
  0.27 s -> 2.47 s, ADC cube 3.1 MB -> 16.5 MB.
- **TX PA output mismatch is now modelled physically** (`ripple_model="mismatch"`, the
  multiple-reflection response parameterized by reflection coefficients and a physical
  length) instead of an ad hoc sinusoid whose default period implied a 50-150 cm
  electrical length at 77 GHz. The physical form also carries the group-delay ripple
  causality requires -- a range bias the magnitude-only model set to exactly zero. The
  old model is retained as `ripple_model="sinusoid"` since it is its small-mismatch
  linearization.
- Mixed precision in `train.py` (`amp`, default on for CUDA). Measured saving on
  SSMRadNet is ~5% of peak memory (6.02 -> 5.74 GiB on an 8 GiB card), not the halving
  an activation-memory argument predicts, and it does not buy a larger batch. Batch size
  is what decides whether that model fits.

### Added
- **`e2e/main/main_tx_nonideality.py` -- ideal vs non-ideal transmitter, side by side.**
  The TX PA was wired into the radar corpus generator and the webapp but into none of the
  comms/ISAC examples, so the domain where a high-PAPR waveform stresses a PA hardest
  never exercised it. Runs one OFDM frame through two otherwise-identical transmit paths:
  EVM 3.56% (ideal) vs 22.02% (non-ideal) at 0 dB backoff, converging to 3.71% at 12 dB;
  ACPR -148 dB vs -20.4 dB; sensing PSL 19.1 -> 18.3 dB; measured PAPR 9.27 dB. The
  printed summary and the PSD figure both state that the ~0 ACPR asymmetry is structural,
  because the PA model is memoryless, so the plot is not mistaken for a hardware
  prediction.
- **`e2e/ml/baseline.py` -- a classical (no-learning) CFAR detector**, scored through
  the identical metric as the networks, because a detection AP means nothing on its own.
  On the pre-switch `ti_iwr1443` corpus it scored AP 0.0241 where a trained FFTRadNet
  reached 0.0084: the learned detector was losing to an FFT, which had been invisible for
  the whole campaign. On a `radial_like` pilot corpus -- where the harness is physically
  answerable -- the ordering FLIPS: baseline AP 0.0044 against the model's 0.0168, a ~3.9x
  win for the model. That is the result that justifies the preset switch. (400 frames,
  12 epochs; an early number, not a headline.) `resolution_report()` states whether a config's evaluation harness is
  physically answerable at all and warns when it is not.
- **Full-vs-reduced dimension contract** (`frames.DIMENSION_*`, `require_dimension`)
  plus standalone `CompressBlock`/`DecompressBlock` (`e2e/chain/compress.py`).
  Quantization and compression were entangled in one stage that reconstructed
  immediately, so reduced-dimension processing was inexpressible and the reconstruction
  cost was unconditional. Compression is modelled as analog (pre-digitization, reducing
  ADC count); decompression is explicit because it is lossy for M < N.
- **`doppler_validity()` / `warn_if_doppler_invalid()`** -- the one-solve Doppler model
  omits intra-frame range migration, giving an error linear in chirp index at rate
  `2*pi*B*v/(sqrt(3)*c)` and usable chirps `N = eps*sqrt(3)*c/(2*pi*B*v*T_c)`. Measured
  on a stable-path-set scene (fitted exponent 0.93, R^2 0.998, matching the formula to
  2.3%). Note the consequence, pinned by a test: `radial_like`'s 252-chirp frame is
  outside a 5% bound even at 1 m/s.
- `cfr_sum_over_paths()` -- the closed-form per-path CFR (convention verified against
  Sionna's own `cfr()` at 2.3e-4) with an optional, not-yet-default range-migration
  correction.
- `detection_loss(cls_normalize=...)`, normalizing the focal term by positive-cell count
  so `reg_weight` is meaningful. Explicitly NOT a fix for low detection AP -- measurement
  refuted that, and the docstring records why.

### Licensing
- **RADIal: reuse and redistribution approved in writing by the publishing author and
  repository publisher (2026-08-10)**; the missing `LICENSE` file was an oversight. The
  previous "attribution, remove on request" posture and its scrub contingency are retired.

### Fixed
- **Adversarial-review fixes to `e2e/ml/` (pre-merge audit, 6 confirmed findings)**:
  - Clutter is no longer detection ground truth: label encoding/target listing take a
    class filter and the dataset generator labels only vehicles/pedestrians
    (`dataset.LABEL_CLASSES`, recorded in manifests). D2/D3 ground truth was previously
    70–80% background clutter.
  - `metrics.evaluate_dataset` no longer counts vacuous `precision=1.0` at thresholds
    where the model made no detections against existing ground truth — such thresholds
    are excluded from the AP mean (per-threshold NaN), so AP no longer rewards
    under-confident detectors.
  - DDMA now pays the physically correct `n_tx` unambiguous-velocity penalty in
    `RadarConfig.max_velocity_mps` (empirically, velocities one sub-band apart alias
    to identical replica sets); the `radial_like` preset moves to `n_chirps=252`
    (divisible by `n_tx=12`) so DDMA replicas land on exact Doppler bins, and the
    FFTRadNet DDMA pre-encoder rejects fractional replica spacings outright.
  - `RadarConfig` normalizes the `mimo` tag at construction (a case-mismatched `"TDM"`
    previously synthesized correctly while silently mis-computing the noise coherent
    gain by 10·log10(n_tx) dB).
  - `FFTRadNet`'s decoder now crops deconv outputs to each skip's range length,
    fixing forward-pass crashes for most non-power-of-two `n_range_in` values.
  - `train.py` keeps the later epoch on a val-AP tie (`>=`), and a new end-to-end
    seam test pins the input tensor's range axis to the label grid through
    `generate_sample`.

### Added
- **FMCW MIMO radar ML dataset + perception models (`e2e/ml/`)**: a self-contained,
  torch-free-at-the-geometry-layer package for generating labeled radar training data
  and training/evaluating perception models on it.
  - `radar_config.py` (dependency-free `RadarConfig` + reference presets `ti_iwr1443`
    3TX/4RX TDM and `radial_like` 12TX/16RX DDMA, the latter reproducing the RADIal
    paper's published resolution/FOV numbers) and `scenes.py` (four difficulty tiers
    D0–D3 of vehicle/pedestrian/clutter scenes, sampled from `e2e.scenario`).
  - Analytic raw-ADC synthesis (`rd_synth.py`) and pure-torch ADC→range-Doppler
    transforms including TDM de-interleave and network-input packing (`transforms.py`,
    with the residual TDM Doppler-phase term across TX groups documented rather than
    corrected).
  - FFTRadNet/RADIal-style dense detection labels — `(range, sin-azimuth)` grid, 3×3
    footprint + per-cell regression residuals, encode/decode — in `labels.py`.
  - `dataset.py`: `.npz` + manifest dataset generator, `RadarFrameDataset`, and a
    `python -m e2e.ml.dataset` CLI (with a `--dry-run` sizing mode).
  - Two ported detection models sharing one input/output contract: `FFTRadNet` (from
    valeoai/RADIal, Rebut et al., CVPR 2022) and `SSMRadNet` (a two-scale selective-
    state-space detector from AnuvabSen1/SSMRadNet, with a pure-torch Mamba selective
    scan since `mamba_ssm` ships no Windows wheels), plus `losses.py` (focal + masked
    regression) and `metrics.py` (confidence-sweep AP/AR, RADIal-style).
  - `train.py`: a reference `python -m e2e.ml.train` train/eval CLI. CUDA smoke test on
    `ti_iwr1443`/D0 (160 frames, `batch_size=2`): FFTRadNet loss 36.7k→1.8k over 25
    epochs (val AP ~0.45–0.56); SSMRadNet loss 5.3k→161 over 15 epochs (val AP 0.778,
    held-out test AP 0.778, range RMSE ~0 m) — plumbing verification, not benchmarks.
  - Neither upstream repo ships a `LICENSE` file; both ports carry attribution headers
    and are pending explicit license confirmation before public redistribution (see
    `e2e/ml/README.md`).
- **Corpus v1 + tuning infrastructure for `e2e/ml`**: `corpus_v1`, a fixed, regeneratable
  four-tier `ti_iwr1443` dataset (D0 400 / D1 1000 / D2 1500 / D3 2000 frames, ~14 GB;
  see `e2e/ml/README.md` for the exact regen commands per tier) plus the tooling built to
  generate, tune against, and validate it:
  - **v2 manifest format**: `dataset.py` now stores raw ADC (not a precomputed
    range-Doppler "input") plus scene-grouped `sequences` alongside the flat per-split
    file lists, so a future sequence-aware loader has frame-order/scene grouping for
    free; `manifest_version=1` corpora still load unchanged.
  - `sweep.py`: a `reg_weight`/`lr`/`gamma` hyperparameter sweep driver with a
    trailing-window objective and an AR-decline guard, plus `pick_best()` selection
    logic — used to derive the recipe below.
  - `stats_report.py`: a corpus-composition/statistical-validation reporter (targets-
    and clutter-per-frame, veh:ped ratio, footprint-overlap and radial-velocity
    clamp-saturation checks). Reports are generated locally (the `report/` directory is
    not tracked): `python -m e2e.ml.stats_report --manifest <manifest.json>`.
  - `render_scene.py`: bird's-eye + radar-view animated GIFs of `e2e.ml.scenes` scenes;
    three showcase GIFs are committed under `docs/media/` and embedded in the README.
  - **Tuned recipe + confirmed baseline numbers**: a 10-trial sweep on a smaller pilot
    D1 set picked `lr=3e-4`, `gamma=2`, `reg_weight=100` (kept); `gamma=0` was rejected
    after its validation-AP spikes (0.55-0.67) were traced to a near-zero-recall
    checkpoint whose AP the metric's zero-detection-threshold exclusion inflates — the
    same mirage reproduces independently in the confirmed `corpus_v1` baseline. FFTRadNet
    gamma=2 (40 epochs) reaches test AP 0.030-0.177 / AR 0.095-0.183 across the D0-D3
    tiers (not a difficulty ranking — see `e2e/ml/README.md`); SSMRadNet (12 epochs, same
    recipe, unswept for that architecture) reaches D1 test AP 0.196 / AR 0.304 / range
    RMSE 0.070 m on a step-comparable budget, with the appropriate single-seed/
    single-tier/hyperparameters-not-tuned-for-it caveats.
  - `metrics.evaluate_dataset` now also returns `n_defined_precision_thresholds`, the
    count of confidence thresholds with defined (non-NaN) precision backing the AP mean
    — the diagnostic that surfaced the gamma=0 mirage above (an AP resting on 1-2
    thresholds is a small-sample artifact, not a quality signal).
- **Comms head with spatial combining**: `ModemBlock` gains a `combining` mode
  (`"element0"` the historical single-tap SISO shortcut / `"mrc"` full-aperture
  maximum-ratio combining / `"subspace"` broadband combining using the AdaOja
  tracker's dominant direction, `state['U'][:, 0]`) plus `e2e/comms/beamforming.py`
  (per-element channel extraction, MRC/subspace weights, coherent combine). Both
  spatial modes inject independent per-element AWGN *before* combining, so the
  reported `comm_array_gain_db` is real coherent-combining gain, not a noise-averaging
  artifact (measured ~+30 dB at 1024 elements). New example
  `python -m e2e.main.main_comms_head` runs the SAME full pipeline (environment → RFFE →
  interconnect → AFE → AdaOja subspace) three times, once per combining mode, and
  reports mean BER/EVM/array-gain plus a radar-product map from the same run — making
  concrete that radar products and the comms link are two swappable *heads* on one
  pipeline. Also exposed in the web UI as an optional "Comms Head (OFDM)" block
  downstream of the subspace stage.
- **Self-describing frames (A1)**: generated `.pkl`s now carry a `meta` block —
  frequency plan, per-link RX array geometry, link kind, `tx_power_dbm`, and scale
  convention — alongside the per-link frame stacks. `SionnaIterator` reads all three
  formats (v2 / legacy multi-link dict / legacy bare array) and exposes the metadata;
  the environment block auto-derives its array shape from it; the web UI gains an
  RFFE "auto" scale mode and derives the true frequency span for range axes from the
  frames themselves. Validated end-to-end on the real GPU ray-tracing path.
- **Physical signal levels**: `Node.tx_power_dbm` (reference scenarios default to 12 dBm)
  switches generation to physically scaled S-parameters — Sionna's un-normalized CFR
  (real path) or an analytic free-space level (dry-run) converted to absolute receiver
  voltage via the transmit power and the 50 Ω system impedance — with
  `RFFEBlock(physical_scale=True)` consuming volts directly. `tx_power_dbm=None`
  preserves the legacy unit-energy convention end to end.
- Physics validation tests (`tests/test_rffe_physics.py`): thermal noise floor vs
  theory, 24 dB small-signal chain gain, monotonic compression, and a link-budget
  round-trip pinning the generation-to-volts conversion against a hand computation.
- Two new reference scenarios: `etoile_radar` (backs the runtime `etoile` base scene via
  `--out .../etoile.pkl`) and `munich_patrol` (motion-rich: waypoint patrol track plus
  cars exercising translation, orbit, and combined curve motion).
- A JSON scenario pack under `e2e/environment/scenarios/` — `munich_dense_traffic`,
  `munich_two_link_isac` (three links: radar + one TX to two RX), and `canyon_radar`
  (street-canyon base scene) — validated by a pack-wide test.
- `main_isac_multilink` example: consumes the multi-link `.pkl` export directly —
  link discovery via `SionnaIterator.available_links()`, radar range-profile leg and
  OFDM comm leg per link, real-frames-ready at the same cache path.
- Web block-diagram UI (`webapp/`): a Dash + Plotly + cytoscape app to compose,
  configure, and run pipelines and to place scenario nodes/objects on a 2D map.
- Declarative scenario spec (`e2e/scenario.py`) — `Scenario` / `Node` / `SceneObject`
  / `ArrayConfig` / `Motion` / `FrequencyPlan` + reference scenarios — plus an offline
  generator (`e2e/environment/scenario_runner.py`) with a GPU-free `--dry-run` mode and
  a real Sionna RT path.
- Communications and joint radar/comms (ISAC) layer (`e2e/comms/`): OFDM modem,
  channel estimation/equalization + metrics, ISAC sensing helpers, and
  pipeline-compatible `ModemBlock` / `BERBlock`.
- Example scripts: `main_comms_link`, `main_channel_estimation`, `main_isac` (each
  falls back to a synthetic channel when no frames are present).
- Multi-link ISAC frame export (one frame-stack per tx→rx link) and `SionnaIterator`
  link selection.
- Automated test suite (`tests/`) with marker-gated hardware/GUI tests and CI.
- Packaging: `pyproject.toml` with `[webapp]` / `[sionna]` / `[dev]` extras, console
  scripts, and an MIT `LICENSE`.
- `ROADMAP.md`, `CONTRIBUTING.md`, this changelog.

### Changed
- The online subspace tracker (AdaOja) now RMS-normalizes its input at the block
  boundary. This decouples its step size from absolute signal scale, but it also
  changes effective tracker dynamics for legacy (normalized) runs — if you had tuned
  `eta` against the old ~1e-5-RMS inputs, re-check it.
- Receive-array geometry is parameterized through `Simulation.array_shape` (the web
  runner now derives it from the environment block instead of assuming 32×32).
- Combined linear + angular node motion now curves about the entity's own origin
  ("drive forward while turning") instead of spiraling about the scene centroid.
- `Scenario.validate()` is stricter: it flags an unpaired comm link in either
  direction, rejects non-positive array dimensions/spacings, and rejects `num_freqs < 1`.
- **`mmse_equalize` now defaults to `unbiased=True`** (behavior change for any external
  caller): it rescales by the per-subcarrier estimator bias before hard decisions, so
  MMSE and ZF now agree on hard decisions (as scalar-MMSE theory predicts) and the
  shipped BER example reports MMSE's genuine advantage in soft-symbol EVM/MSE instead
  of an artifact of comparing a biased estimator against unbiased decision boundaries.
  Pass `unbiased=False` to recover the previous raw (biased) filter output.
- **The three radar display products (`FFTBlock`, `RangeAzBlock`, `RangeElBlock`) now
  return real power maps** (previously complex amplitudes accumulated by a coherent
  sum across the non-displayed axis/range). Output keys and `[bins, bins]` shapes are
  unchanged, but values are now honest non-coherent-integration power, and consumers
  that expected complex output (e.g. taking `.abs()` themselves) should drop that step.
- **The RFFE frequency-domain thermal-noise floor drops by ~`10*log10(n_freqs)` dB**
  compared to previous runs (verified ratio 64.08 at `n_freqs=64`, i.e. ~18 dB): the
  noise was inflated at the unnormalized-FFT seam and is now band-referenced correctly
  per frequency bin. Demo plots and any tuned SNR-dependent thresholds will look
  different (lower noise floor, more headroom) after this fix.

### Fixed
- **Non-square aperture ordering**: Sionna's `PlanarArray` numbers antennas
  column-first (row index varies fastest along a frame's flat RX axis), so the
  row-major aperture reshape needs the slow axis first. The env block's array-shape
  auto-derive from v2 metadata (`[num_rows, num_cols]`) now maps to
  `(num_cols, num_rows)` — grid dim 0 = columns (azimuth), dim 1 = rows (elevation),
  the convention the range/angle blocks assume. Square legacy arrays were unaffected;
  non-square auto-derived arrays would have had a scrambled aperture (wrong angle maps).
- The frame shape-contract guards raise a dedicated `frames.FrameContractError`
  (a `ValueError` subclass); the web UI maps it to its friendly
  "Pipeline constraint failed" message again (lost when the guards stopped being
  bare `assert`s).
- `Simulation` with `subspace_block=None` no longer crashes: the measurement stage
  is skipped, so FFT/range-map-only pipelines run without a subspace tracker.
- **RF front-end physics** (author-reviewed audit): the baseband stage's cubic
  nonlinearity had the wrong sign (expansive instead of compressive saturation); the
  thermal-noise constant was kT instead of 4kT (~6 dB); the noise floor is band-referenced
  to the receiver's 15 MHz IF bandwidth per frequency point (stepped-frequency measurement
  semantics — no longer coupled to a vestigial chirp duration, nor to the buffer sample
  rate, which would have inflated it ~23 dB); the LNA bias
  was retuned (8 mA → 15.6 dB real gain, ~1.8 dB NF) so the gain chain is physically
  distributed; the noise bandwidth is no longer silently inflated by an IF-filter
  workaround (the filter is an explicit `if_filter` flag); the total chain gain is
  interpreted as 24 dB *voltage* gain; and the online subspace tracker is
  RMS-normalized at its boundary so absolute signal levels don't retune its step size.
- **Range/azimuth and range/elevation maps** collapsed the range axis to a single bin
  (`torch.fft.fft(x, 1)` was read as `n=1`); both now produce full `[bins, bins]` maps.
- **Communications metrics**: `ModemBlock` drew identical AWGN on every frame (degenerate
  multi-frame averaging) — it now uses an independent per-frame noise realization;
  `BERBlock` EVM is now measured against the true transmitted symbols instead of a
  decision-directed reference (which biased EVM low at high error rates).
- `gen_A_ada` built its sensing matrix on a module-default device, crashing a CPU-resident
  basis on a GPU box; it now follows the input's device.
- Web app robustness: all-zero products no longer yield NaN heatmaps; strictly-positive
  parameters fall back to defaults instead of producing `0/0`; an empty upload is a no-op;
  editing a block parameter no longer drops the node selection.
- Generation robustness: per-link RNG streams (a link's dry-run frames are independent of
  other links), pickle files are opened via a context manager (no Windows file lock), and
  the real-Sionna CFR slice is cast to `complex64` to match the dry-run contract.
- **RFFE physics, round two** (author-reviewed audit, RF/EM panel follow-up):
  - Injected thermal noise was inflated by a factor of `N_FREQS` at the unnormalized-FFT
    seam (verified ratio 64.08 at `n_freqs=64`); per-sample variance is now pre-divided
    by `N_FREQS` so the per-frequency-bin floor lands on `NBB*BW` (stepped-frequency
    semantics), with a new end-to-end frequency-domain regression test.
  - The LNA/mixer cubic nonlinearity acted on the complex value directly (`v**3`),
    giving phase-dependent gain (5.28 vs 7.47 measured across phase) and unphysical
    expansion for Q-dominant inputs. Both stages now use the standard bandpass
    baseband-equivalent envelope form (`(3/4)*a3*|v|^2*v`) with a phase-preserving
    envelope clamp at the compressive peak; the baseband per-rail cubic (post-IQ-demod)
    is unchanged, since it was already physically correct there.
  - `CircuitStage` no longer duplicates the frame into a pol pair and discards half
    (was 2x compute plus a wasted noise draw); `apply_circuit` infers the chirp
    dimension instead.
- **Radar display products, non-coherent integration** (RF/EM panel follow-up):
  `FFTBlock` previously coherently summed raw frequency samples before the aperture
  FFT, which is only coherent at range 0 (verified a 63 dB collapse by 5 cm), so the
  az/el map showed zero-range leakage rather than the scene; it now range-transforms
  first, runs the coherent 2D aperture FFT, and power-sums over range so targets at any
  range appear. `RangeAzBlock`/`RangeElBlock` previously collapsed the orthogonal
  aperture axis by a coherent sum (an implicit un-steered broadside beam), which nulled
  targets more than ~3.6 degrees off broadside (verified peak 65515 -> 0.00); they now
  run the coherent FFT in the displayed axes and power-sum across the collapsed axis, so
  all angles contribute. `main_sionna_blocks` now auto-resolves `physical_scale` from the
  frames' metadata (like the webapp) instead of silently renormalizing physically-scaled
  frames. Regression tests pin analytic Parseval peak values for off-broadside and 10 m
  targets.
- **Comms MMSE estimator/equalizer honesty** (RF/EM panel follow-up): `mmse_estimate`'s
  Wiener prior was derived circularly from the same few noisy pilots it shrinks (worse
  than LS in 24/30 low-SNR Monte-Carlo trials); the prior and noise power are now pooled
  across all pilots and symbols, with tests pinning MMSE <= LS at 0-4 dB SNR and
  convergence to LS at high SNR. See the `mmse_equalize` unbiased-default entry above
  under Changed.
- **Generation guards** (RF/EM panel follow-up): `Scenario.validate()` and
  `ScenarioRunner` now reject dual-polarization arrays with a physical explanation.
  Sionna's `PlanarArray` reports 2x antenna ports for VH/cross-polarization, so
  dual-pol scenarios previously generated frames whose antenna axis contradicted the
  recorded `rx_array_shape`/`n_tx_ant` metadata (and dry-run mocked the wrong shape);
  dual-pol support is deferred until the frame contract is dual-pol aware (see
  `ROADMAP.md`). Separately, `_tx_power_amplitude_scale` never divided by the transmit
  element count, so a full TX array radiated `P_tx` *per element* (+`10*log10(n_tx)` dB
  EIRP, ~+30 dB at 1024 elements); `tx_power_dbm` is now the total radiated aperture
  power, split uniformly across elements, with array gain emerging from coherent
  combining rather than power double-counting (verified summed channel power ratio
  full-vs-single array = 0.9986).
- **AFE quantizer and subspace sensing-matrix bias** (RF/EM panel follow-up): the AFE
  floating-point quantizer floored the mantissa (truncation), a systematic ~0.52%
  downward magnitude bias on every quantized measurement matrix; it now rounds to
  nearest (ties-to-even) with proper carry into the exponent and saturation (bias
  measured 0.0008% after the fix). `gen_A_ada`'s random sensing rows were unnormalized
  complex Gaussians with row-norm ~sqrt(n) (~32 at n=1024) against the tracked basis's
  unit-norm rows; the random block is now column-normalized so both halves of the
  sensing matrix weight equally.

### Known limitations
See `ROADMAP.md`. In brief: the runtime pipeline currently assumes a single chirp and no
MIMO; frames are not yet self-describing (the planned A1 keystone); and the online
subspace-tracking and RF-front-end default operating points are slated for review in 1.1.
