# Array Processing End-to-End Simulator

**Release v1.0**

## Getting Started

### Installation

The project uses a standard `pyproject.toml` (setuptools). Install the package plus
its optional extras as needed:

```bash
pip install -e .                  # core runtime (torch, numpy, matplotlib, tqdm, Pillow)
pip install -e ".[webapp]"        # + the Dash/Plotly web UI
pip install -e ".[sionna]"        # + Sionna RT / DrJit / Mitsuba (real frame generation; needs a GPU/LLVM)
pip install -e ".[dev]"           # + the test suite
```

The pinned `requirements.txt` / `requirements-dev.txt` files are still provided for
reproducible environments if you prefer `pip install -r`.

### Usage

#### Quickstart (no GPU required)

Everything below runs on a plain CPU machine with only the core install — no Sionna,
no GPU, no precomputed frames.

**1. Dry-run a scenario.** The scenario runner has a `--dry-run` mode that exercises
all scheduling / motion / serialization logic and emits *synthetic* frames, so it
needs neither Sionna nor a GPU:

```bash
python -m e2e.environment.scenario_runner --scenario munich_radar --dry-run
```

**2. Run the communications / ISAC examples.** Each example saves figures to
`e2e/main/figures/` and **falls back to a synthetic channel when frames are absent**,
so they work out of the box on CPU:

```bash
python -m e2e.main.main_comms_link            # OFDM link, BER vs SNR + constellation
python -m e2e.main.main_channel_estimation    # pilot-based estimation, MSE vs SNR
python -m e2e.main.main_isac                  # joint radar+comm in one multi-node scene
python -m e2e.main.main_comms_head            # SAME pipeline, swappable radar/comms heads
```

**3. Launch the web UI** (block diagram + scenario editor) — see
[Web UI](#web-ui-block-diagram--scenario-scheduling) below. The shell and scenario
editor build without torch/Sionna.

> **Note: frame `.pkl` files are not shipped.** The runtime radar pipeline
> (`python -m e2e.main.main_sionna_blocks`) consumes precomputed Sionna RT
> S-parameter frames (`.pkl` files under `e2e/environment/sionna_sims/`). Those files
> are **gitignored and not included in the repository** — you must generate them
> first (see [Advanced: Sionna RT frame generation](#advanced-sionna-rt-frame-generation-gpu)).

#### Advanced: Sionna RT frame generation (GPU)

To run the full radar pipeline you first generate real S-parameter frames with Sionna
RT ray tracing. This needs `pip install -e ".[sionna]"`, an LLVM toolchain, and (for
OptiX) a CUDA-12.x NVIDIA driver — see the [GPU / driver / LLVM](#gpu--driver--llvm)
section for the exact requirements.

```bash
# Generate the default Munich radar frames.
python -m e2e.environment.sionna_simple_channel
# ...or drop --dry-run on the scenario runner for a real ray-traced generation:
python -m e2e.environment.scenario_runner --scenario munich_radar --frames 10

# Then run the runtime radar pipeline against the generated frames.
python -m e2e.main.main_sionna_blocks
```

### GPU / driver / LLVM

**LLVM (CPU backend for DrJit).** DrJit needs an LLVM shared library on the CPU path.
Install it with conda/mamba and point `DRJIT_LIBLLVM_PATH` at the library — the file
extension differs per OS:

```bash
# macOS (libLLVM*.dylib)
export DRJIT_LIBLLVM_PATH=$(find "$CONDA_PREFIX/lib" -name 'libLLVM*.dylib' 2>/dev/null | head -1)

# Linux (libLLVM*.so)
export DRJIT_LIBLLVM_PATH=$(find "$CONDA_PREFIX/lib" -name 'libLLVM*.so*' 2>/dev/null | head -1)
```

```powershell
# Windows PowerShell (LLVM-C.dll, usually under <CONDA_PREFIX>\Library\bin)
$env:DRJIT_LIBLLVM_PATH = (Get-ChildItem "$env:CONDA_PREFIX\Library\bin" -Filter 'LLVM-C.dll' | Select-Object -First 1).FullName
```

Verify with `python -c "import drjit; print('DrJit loaded successfully!')"`.

**NVIDIA driver / CUDA (GPU OptiX backend).** DrJit 1.2 / Mitsuba 3.7 OptiX ray
tracing requires a **CUDA-12.x** NVIDIA driver. Keep the driver in the range
**`>= 570` and `< 580`**:

* **Validated:** driver `576.80` / CUDA `12.9`.
* **Known broken:** driver `610` / CUDA `13.3` fails with a `ptx2llvm` error —
  `Failed to translate PTX input to LLVM`. CUDA 13.x is not supported by this
  DrJit/Mitsuba combination; pin the driver below 580.

CPU-only generation (`--dry-run`, or the synthetic-channel example fallbacks) does not
need any NVIDIA driver.

## Web UI (block diagram + scenario scheduling)

A browser-based interface for building pipelines and scenarios visually:

```bash
python -m webapp.app      # serves at http://127.0.0.1:8050
```

It has three tabs:

- **Block Diagram** — a drag/connect node graph of the pipeline blocks (environment →
  RFFE → interconnect → AFE → subspace → FFT / range-az / range-el / subspace-error).
  Toggle blocks on/off, edit their parameters, and run the pipeline to view results.
- **Scenario** — place and edit antenna nodes (radar / comm TX / comm RX) and objects on
  a 2D map, set the base scene, frequency plan, and frame count, then validate, save/load,
  or **generate frames** for the scenario.
- **Results** — FFT / range-az / range-el heatmaps and the subspace-error curve.

The UI shell and scenario editor run without `torch`/Sionna; only *running* a pipeline
needs precomputed frames + `torch`.

## Scenario scheduling and generation

Scenarios are declarative, JSON-serializable specs (`e2e/scenario.py`): a base scene, a
set of antenna nodes with roles and motion, scene objects, and a frequency plan.
`e2e/environment/scenario_runner.py` turns a scenario into S-parameter frames via Sionna
RT ray tracing. A **dry-run** mode exercises all scheduling/motion/serialization logic
and emits synthetic frames without needing Sionna installed:

```bash
# Reference scenarios: munich_radar, munich_isac, etoile_radar, munich_patrol
python -m e2e.environment.scenario_runner --scenario munich_radar --dry-run
python -m e2e.environment.scenario_runner --scenario path/to/scenario.json --frames 10

# Drop --dry-run on a machine with Sionna RT + DrJit + LLVM to generate real frames.

# The etoile_radar scenario can back the runtime 'etoile' base scene directly:
python -m e2e.environment.scenario_runner --scenario etoile_radar \
    --out e2e/environment/sionna_sims/etoile.pkl
```

A **scenario pack** of preconfigured variations ships as JSON under
`e2e/environment/scenarios/` (dense traffic, a two-receiver ISAC scene, a street-canyon
radar, ...). Each is a ready-made `--scenario <path>` input and a template for your own:
copy one, edit, validate, generate.

### Physical signal levels

Setting `tx_power_dbm` on a transmitting node (`radar` / `comm_tx`) makes generation emit
**physically scaled S-parameters** (volts at the receiver): the real path requests Sionna's
un-normalized CFR (which carries free-space path loss, antenna patterns, and multipath),
the dry-run mock applies an analytic free-space level, and both are scaled to absolute
voltage via the configured transmit power and the 50 Ω system impedance. The reference
scenarios default to 12 dBm (IWR1443-class). Leaving `tx_power_dbm` unset (`None`) keeps
the legacy unit-energy convention. **The convention is per-link** (it follows each link's
TX node) — and generated frames are now **self-describing**: the `.pkl` carries a `meta`
block (frequency plan, per-link array geometry, `tx_power_dbm`, scale convention) that
`SionnaIterator` exposes and consumers read automatically — the environment block derives
its array shape from it, and the web UI's RFFE "auto" scale mode follows it. Legacy pkls
(bare arrays) still load; for those, consumers set `physical_scale` manually. On the
consumption side, `RFFEBlock(physical_scale=True)`
feeds those volts directly into the analog front-end — whose clamp, compression, and
thermal-noise floor are then a real operating point rather than an arbitrary
normalization — while the default (`physical_scale=False`) preserves the legacy
`signal_scaling` normalization for old frame sets.

> **Two scenario namespaces (different stages).** Do not confuse them:
> * **Runtime / precomputed-frame selection** — `munich`, `etoile`. These name a
>   *base scene* whose precomputed `.pkl` frames the runtime pipeline and the
>   `SionnaIterator` load (e.g. `base_scene="munich"`).
> * **Generation reference scenarios** — `munich_radar`, `munich_isac`. These are the
>   declarative `REFERENCE_SCENARIOS` entries in `e2e/scenario.py` that the *generator*
>   (`scenario_runner`) turns into frames. They build on top of a base scene.
>
> In short: `munich_radar`/`munich_isac` are generation inputs; `munich`/`etoile` are
> the runtime base scenes whose frames get consumed.

**Multi-link / ISAC scenarios.** A scenario with several simultaneous links — e.g. a
monostatic radar *and* a comm TX→RX link (`munich_isac`) — exports one frame-stack per
link. Single-link scenarios dump a plain array (as before); multi-link scenarios dump a
`dict` of `link_name -> frames`. Each link is generated in its own Sionna scene, so links
with different array sizes (a 32×32 radar RX and a 4×4 comm RX) coexist. Select a link
when loading:

```python
from e2e.environment.sionna_iterator import SionnaIterator
it = SionnaIterator("sionna_sims/munich_isac.pkl", link="building_comm_tx__car_comm_rx")
```

### Physical modeling scope & limitations

Honesty contract for this release: what's physically modeled, and what's a placeholder
or a link-level abstraction.

- **Diffuse reflection is off.** Ray tracing runs specular/LOS/refraction paths only
  (`max_depth=5`); diffuse scattering from rough surfaces or foliage is not represented,
  so clutter from those surfaces is absent from generated frames.
- **Comms SNR is enforced post-hoc at the receiver.** Noise is added to hit a target SNR
  directly, decoupled from the ray-traced path loss; the radar leg of the same scenarios
  *does* use the physical channel gain from the ray-traced/analytic path. The **pipeline
  comms head** (`ModemBlock`, see "Swappable heads" above) does perform real spatial
  combining across the array (`combining="mrc"`/`"subspace"`, with independent
  per-element noise injected before combining so the resulting array gain is honest) —
  it is only the per-element/per-combined-stream SNR target itself that is set post-hoc,
  not the combining. The standalone link-level examples (`main_comms_link`,
  `main_isac`, `main_isac_multilink`) remain single-spatial-channel (SISO) by design —
  they demonstrate the OFDM/channel-estimation machinery in isolation, not the array.
- **The OFDM path applies the channel as a per-subcarrier frequency-domain multiply**,
  not a time-domain convolution, so ISI/CP-overrun effects cannot occur in the shipped
  examples. `cp_len` is exercised only by the modem's own time-domain round-trip
  (`OFDMModem.modulate`/`demodulate`), not by any end-to-end example.
- **"S-parameters" are absolute voltages in physical-scale mode.** The stored per-link
  quantities are called `s_pars` throughout the code, but when generated with
  `tx_power_dbm` set they are receiver voltages (V_rms across 50 ohms), not the
  dimensionless network-theory S21 ratios the name suggests. The legacy (`tx_power_dbm
  = None`) convention is unit-energy, also not literal S21.
- **Frames are independent scene snapshots** with no frame-to-frame phase continuity, so
  Doppler processing across frames (as opposed to within a single frame/chirp) is not
  physically meaningful.
- **The interconnect model is a placeholder**: a fixed 11-tap boxcar frequency response
  (`InterconnectBlock`), independent of the scenario's `FrequencyPlan`. It stands in for
  a real interconnect filter but is not derived from one.

## Communications and joint radar/comms (ISAC) examples

Beyond the radar pipeline, the `e2e/comms/` package adds an OFDM modem, channel
estimation/equalization, and ISAC utilities. Example scripts (each saves figures to
`e2e/main/figures/`, and falls back to a synthetic channel when frames are absent):

```bash
python -m e2e.main.main_comms_link            # OFDM link, BER vs SNR + constellation
python -m e2e.main.main_channel_estimation    # pilot-based estimation, MSE vs SNR
python -m e2e.main.main_isac                  # joint radar+comm in one multi-node scene
python -m e2e.main.main_isac_multilink        # consume a multi-link .pkl: radar + comm legs per link
```

`main_isac_multilink` is the end-to-end demo of the **multi-link export**: it generates
(or reuses) one `.pkl` holding both links of `munich_isac`, discovers them with
`SionnaIterator.available_links()`, and drives the radar and comm legs off their own
frame stacks. With dry-run frames the numbers are plumbing checks; drop a real
Sionna-generated `.pkl` at the same path and both legs become physical with no code
changes.

The comms blocks are also **first-class pipeline stages**: add `ModemBlock` and `BERBlock`
to a `Simulation`'s `downstream_blocks` and they run alongside the radar products (FFT,
range-az/el, subspace error), consuming the same `s_pars` channel. Downstream blocks
compose — `ModemBlock`'s transmitted/received bits flow to `BERBlock` in the same step.

### Swappable heads: radar products or a comms link, off the same pipeline

The receive chain (environment → RFFE → interconnect → AFE → AdaOja subspace tracking)
ends in a reconstructed aperture grid; everything past that point is a swappable **head**.
`RangeAzBlock`/`RangeElBlock`/`FFTBlock` turn it into a radar product; `ModemBlock` +
`BERBlock` turn the SAME reconstruction into an OFDM communications link. `ModemBlock`'s
`combining` selects how the comms head uses the array:

- `"element0"` — the historical single-tap SISO shortcut (one spatial channel, no
  combining).
- `"mrc"` — full-aperture maximum-ratio combining: independent per-element noise is
  injected *before* combining (`e2e/comms/beamforming.py`), so the measured array gain
  (~`10*log10(N)` for `N` elements at high correlation) is real coherent-combining gain,
  not a noise-averaging artifact.
- `"subspace"` — reuses the AdaOja tracker's dominant tracked direction (`state['U'][:, 0]`)
  as a broadband beamformer weight instead of a fresh MRC solve, so the same online-tracked
  subspace serves both heads.

`python -m e2e.main.main_comms_head` runs the full pipeline three times (fresh
`Simulation` per run, same frames) — once per `combining` mode — and reports mean
BER/EVM/array-gain plus a radar-product map from the same run, to make the "one
pipeline, two heads" point concrete. The web UI exposes the same head as an optional
"Comms Head (OFDM)" block downstream of the subspace stage.

## Cookbook

Where to look when you want to...

| Goal | Start here |
| ---- | ---------- |
| **Run an example** | `e2e/main/` — e.g. `main_sionna_blocks.py` (radar pipeline), `main_comms_link.py`, `main_channel_estimation.py`, `main_isac.py`. Run via `python -m e2e.main.<name>`. |
| **Add a pipeline block** | `e2e/blocks.py` — implement an `apply(state_dict) -> dict` block class (see `RFFEBlock` / `FFTBlock`), then wire it into the feed-forward order in `e2e/simulation.py` (`Simulation`). Comms blocks live in `e2e/comms/blocks.py`. The web UI registry is `webapp/pipeline_registry.py`. |
| **Define a scenario** | `e2e/scenario.py` — build a `Scenario` (nodes, objects, `FrequencyPlan`, motion) or add an entry to `REFERENCE_SCENARIOS`. Generate frames with `e2e/environment/scenario_runner.py`. |
| **Extend the comms / ISAC layer** | `e2e/comms/` — `ofdm.py` (modem), `channel.py` (estimation/equalization/metrics + synthetic fallback), `isac.py` (sensing/comm split), `blocks.py` (pipeline blocks). |

## Testing

The project ships an automated test suite (`tests/`). The default run is fully
hands-off — synthetic data, no Sionna ray tracing, no display:

```bash
pip install -r requirements-dev.txt   # test deps (plus torch; see the file)
pytest
```

Tests that need hardware or a human are skipped by default and opt-in via env vars:
`RUN_SIONNA=1` (real Sionna RT generation), `RUN_SLOW=1` (full RF chain / sweeps),
`RUN_GUI=1` (live server). CI runs the default suite on every push/PR
(`.github/workflows/tests.yml`). See `tests/README.md` for details.