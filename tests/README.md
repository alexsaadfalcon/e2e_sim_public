# Test suite

Automated tests for the end-to-end array-processing simulator. The default run is
**fully hands-off**: it uses small synthetic data and needs no Sionna ray tracing, no
display, no network, and does not depend on the precomputed `.pkl` frames existing.

## Running

```bash
pip install -r requirements-dev.txt      # plus torch (see that file)
pytest                                    # the whole hands-off suite
pytest tests/test_comms_channel.py        # one file
pytest -k ofdm                            # by keyword
```

## What is covered

The suite spans ~59 files / 1100+ collected tests. Broad areas (one row per theme,
not per file — `ls tests` for the full list):

| Area | Files |
|------|-------|
| Scenario spec / motion / offline generation (dry-run) | `test_scenario.py`, `test_motion.py`, `test_scenario_runner.py` |
| Frame loading, self-describing frames, contracts | `test_sionna_iterator.py`, `test_frames.py` |
| Subspace tracking (Oja / reestimate / gap_response) & estimators | `test_subspace.py`, `test_subspace_tracking.py`, `test_spectrum_estimator.py` |
| AFE / compression chain / ADC-cube chain | `test_afe.py`, `test_chain_*.py`, `test_afe_sweep.py` |
| RF front end & TX PA physics | `test_rffe_physics.py`, `test_tx_pa.py`, `test_interconnect.py` |
| Pipeline blocks & Simulation feed-forward | `test_blocks.py`, `test_blocks_range_profile.py`, `test_simulation.py` |
| Communications: OFDM / channel / ISAC / beamforming / examples | `test_comms_*.py`, `test_isac_multilink*.py` |
| ML: datasets, models, training, baselines, assets, renders | `test_ml_*.py` |
| Web UI registry / layout / editor / runner | `test_webapp.py` |
| Browser journey tests (Playwright, opt-in) | `e2e_ui/test_journeys.py` |

## Markers (opt-in tests)

Some tests need hardware, a browser install, or a human and are **skipped by
default**. Enable them with an env var:

| Marker | Needs | Enable with |
|--------|-------|-------------|
| `sionna` | Sionna RT + DrJit + LLVM (real ray tracing) | `RUN_SIONNA=1 pytest -m sionna` |
| `slow` | extra wall-clock (full RF chain, SNR sweeps) | `RUN_SLOW=1 pytest -m slow` |
| `gui` | a live server / display | `RUN_GUI=1 pytest -m gui` |
| `browser` | Playwright + `playwright install chromium` | `RUN_BROWSER=1 pytest -m browser` |

PowerShell: `$env:RUN_SLOW=1; pytest -m slow`

## Conventions

- Shared fixtures live in `conftest.py` (synthetic frames, a drop-in environment block,
  a tiny scenario, the library device). Don't redefine them in test files.
- Tests run on whatever device the library is configured for (CUDA if present, else CPU),
  so the suite is green both on CPU-only CI and on a CUDA dev box.
- Two test paths need extra setup beyond `requirements-dev.txt`: **real Sionna ray
  tracing** (`@pytest.mark.sionna`, needs the Sionna/DrJit stack and a capable
  GPU/LLVM machine) and the **browser journey tests** (`@pytest.mark.browser`, need
  `playwright install chromium`). Everything else runs hands-off.
