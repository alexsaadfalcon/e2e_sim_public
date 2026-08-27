# Glossary

Shared vocabulary for this repo, aimed at someone who knows signals/arrays but has
never opened this codebase. Terms are grouped roughly by where you meet them: the
pipeline, the scenario layer, and the ML/benchmark layer. See also
[`CONTRIBUTING.md`](../CONTRIBUTING.md) and [`docs/FIRST_SCENARIO.md`](FIRST_SCENARIO.md).

## Core pipeline

- **Frame** — one snapshot of the receive array's channel response at one instant in
  time: an **S-parameter frame**, shape `[N_RX, N_TX, n_chirp, N_FREQS]` (torch
  complex64). A run of the simulator advances frame by frame.
- **S-parameter frame** — the tensor that flows through the pipeline. Despite the
  name, it is not always the dimensionless network-theory S21 ratio: in
  "physical-scale" mode (`tx_power_dbm` set at generation) the values are receiver
  voltages (V_rms across 50 ohms); in the legacy convention they are unit-energy.
  See `e2e/frames.py` for the shape contract in full.
- **Block** — one swappable pipeline stage (`e2e/blocks.py`), e.g. `RFFEBlock`,
  `InterconnectBlock`, `AFEBlock`, `AdaOjaBlock`, `FFTBlock`. `Simulation`
  (`e2e/simulation.py`) feeds a frame forward through whichever blocks are
  configured. Every block that consumes `s_pars` declares a `frame_capabilities`
  (see below) so `Simulation` can validate the frame against it before calling in.
- **`frame_capabilities`** — a `FrameCapabilities` dataclass (`e2e/frames.py`) a
  block/stage attaches as a class attribute, declaring what frame shapes it accepts:
  whether it handles MIMO (`accepts_mimo`), how it handles the chirp axis
  (`chirps`: native / broadcast over chirps / single-chirp-only), which signal
  domain it consumes (frequency-domain CFR, TX time-domain, or RX time-domain), and
  whether it needs the full aperture or is fine with a compressed measurement
  (`dimension`). The default is the historical contract: no MIMO, single chirp,
  frequency domain, full dimension — so a block that declares nothing behaves
  exactly as before. `Simulation` raises a `FrameContractError` (not a bare
  assertion) naming the offending block and shape when a frame violates a
  declaration, e.g. handing a multi-chirp frame to a block declared
  `CHIRP_SINGLE`.
- **RFFE** — Radio-Frequency Front End: the analog receive-chain circuit distortion
  model (`e2e/circuit/rffe_model.py`, applied via `RFFEBlock`) — gain, compression,
  and thermal-noise floor of the physical receiver electronics.
- **AFE** — Adaptive Feature Extraction: the quantized-matmul compression stage
  (`e2e/afe/afe_utils.py`, applied via `AFEBlock`) that projects the full aperture
  down to a smaller set of measurements before subspace tracking.
- **Subspace error** — the tracked-vs-true subspace distance `SubspaceErrorBlock`
  reports each frame (`e2e/subspace/subspace_utils.py:subspace_dist_frob`): how far
  the AdaOja tracker's estimated top-`k` subspace basis has drifted from the
  frame's true one. Lower is better; the two-honest-worked-example figure in the
  README gallery ("Adaptive subspace tracker...") plots exactly this.
- **`sv_gap_norm`** — a per-frame spectrum diagnostic, `(S[k-1] - S[k]) / S[0]`
  where `S` is the frame's singular-value spectrum and `k` the tracked rank: the
  absolute gap between the k-th and (k+1)-th singular value, normalized by the
  largest one. A small `sv_gap_norm` means the top-`k` subspace is nearly
  degenerate (rank collapse) and harder to track; `AdaOjaBlock(gap_response=...)`
  can spend extra refinement passes when it drops below a threshold. Computed in
  `e2e/simulation.py` (oracle, from the full spectrum) and estimated online in
  `e2e/subspace/spectrum_estimator.py`.
- **ISAC** — Integrated Sensing And Communications: running a radar product and an
  OFDM communications link off the same received aperture / pipeline run
  (`e2e/comms/isac.py`, `main_isac*`). "Swappable heads" is the same idea at the
  block level — `RangeAzBlock`/`FFTBlock` vs `ModemBlock`+`BERBlock` consuming the
  same upstream reconstruction.

## Two layers — don't confuse them

- **Runtime pipeline** — `e2e/blocks.py` + `e2e/simulation.py`. Consumes
  *precomputed* `.pkl` S-parameter frames. Fast, no Sionna/GPU needed to run.
- **Scenario generation** — `e2e/environment/`. Uses Sionna RT ray tracing (or a
  synthetic dry-run) to *produce* those `.pkl` files from a declarative `Scenario`.
  Heavy in its real mode: needs Sionna + DrJit + LLVM; the `--dry-run` mode needs
  neither.
- **scenario vs `Scenario`** — lowercase "scenario" is the general idea (a
  particular RF situation: which nodes, what motion, what scene); `Scenario` is the
  specific Python/JSON dataclass (`e2e/scenario.py`) that encodes one, shared by the
  UI, the generator (`scenario_runner.py`), and the examples. `REFERENCE_SCENARIOS`
  is the dict of named, ready-to-generate `Scenario` instances (e.g.
  `munich_radar`, `munich_isac`).
- **`n_freqs` vs `num_freqs`** — the same concept, named differently on each side of
  the boundary above, deliberately (not a bug, not being unified): the runtime
  pipeline layer (blocks/frames/`e2e/chain`) says `n_freqs`; the declarative
  `Scenario` layer (`FrequencyPlan`, `scenario_runner`, the `.pkl`'s `meta.freq_plan`)
  says `num_freqs`. Bridge code translates between them; neither name is being
  renamed out.
- **Two scenario *namespaces*, orthogonal to the above** — `munich`/`etoile` name a
  *base scene* the runtime pipeline loads precomputed frames for; `munich_radar`/
  `munich_isac`/etc. name *generation* `REFERENCE_SCENARIOS` entries that build on
  top of a base scene and that `scenario_runner` turns into frames. See the
  README's "Two scenario namespaces" callout for the full explanation.

## ML / benchmark layer (`e2e/ml/`)

- **Answerability tier** — whether a radar config's geometry/preset is physically
  capable of resolving the targets a corpus asks about (unaliased Doppler,
  sufficient angular resolution for the tolerance a metric uses). Checked by
  `e2e.radar_config.answerability_problems`; generation refuses to build an
  unanswerable corpus unless `--allow-unanswerable` is passed explicitly. An
  "unanswerable" harness makes any AP number meaningless — the model is being
  scored on distinctions its own front-end geometry cannot make.
- **Corpus tag** — the identifying string embedded in a generated dataset (e.g. by
  `render_scene.py`'s render path) that records which generation path/config
  produced it, used to keep provenance traceable and to catch train/eval overlap
  (`check_corpus_overlap.py`).
- **Chance floor / null arm** — the mandatory, data-blind baseline every AP table
  must report alongside a trained model's score (`e2e.ml.compare_detectors.
  score_null`): uniform-random confidence scores inside the train-split ground-truth
  bounding box, fit on train only, never touching the RF data. Established because
  the first learned-model comparison on this project shipped an AP *below* this
  floor — without it, an AP number alone is uninterpretable (ground truth occupies
  a small fraction of the map, so "beats zero" is not the same as "beats chance").
- **Operating point** — the single score threshold at which recall/precision
  numbers like `AR` are reported (as opposed to an averaged-over-all-thresholds
  quantity like AP). Different detector arms (classical CFAR vs a trained model)
  generally need *different* operating points to be compared fairly — a shared
  threshold across arms with different score distributions is a documented trap
  (`e2e/ml/detect_viz.py`, `e2e/ml/metrics.py`).
- **Objectness** — channel 0 of a detection label/prediction map: a `[0, 1]`
  probability (sigmoid output for a model, a CFAR ratio mapped to `[0, 1]` for the
  classical baseline) that a target occupies that grid cell. The other channels
  (range/azimuth residuals) only mean something where objectness says a target is.

## Where each term lives in code

| Term | Defined / grounded in |
|---|---|
| Frame shape contract, `frame_capabilities` | `e2e/frames.py` |
| Blocks | `e2e/blocks.py`, `e2e/simulation.py` |
| RFFE | `e2e/circuit/rffe_model.py` |
| AFE | `e2e/afe/afe_utils.py` |
| Subspace error, `sv_gap_norm` | `e2e/subspace/`, `e2e/simulation.py` |
| ISAC / swappable heads | `e2e/comms/isac.py`, `e2e/comms/blocks.py` |
| `Scenario`, `FrequencyPlan`, `REFERENCE_SCENARIOS` | `e2e/scenario.py` |
| Answerability tier, chance floor, operating point, objectness | `e2e/radar_config.py`, `e2e/ml/compare_detectors.py`, `e2e/ml/metrics.py`, `e2e/ml/labels.py` |
