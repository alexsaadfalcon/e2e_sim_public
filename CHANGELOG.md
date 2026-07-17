# Changelog

All notable changes to this project are documented here. The format is loosely based
on [Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
semantic versioning.

## [Unreleased]

### Added
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

### Fixed
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

### Known limitations
See `ROADMAP.md`. In brief: the runtime pipeline currently assumes a single chirp and no
MIMO; frames are not yet self-describing (the planned A1 keystone); and the online
subspace-tracking and RF-front-end default operating points are slated for review in 1.1.
