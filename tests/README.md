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

| Area | Files |
|------|-------|
| Scenario spec (validate / JSON round-trip / references) | `test_scenario.py` |
| Motion & scheduling helpers | `test_motion.py` |
| Offline scenario generation (dry-run) | `test_scenario_runner.py` |
| Frame loader | `test_sionna_iterator.py` |
| Subspace tracking (Oja) & distances | `test_subspace.py` |
| AFE quantizer | `test_afe.py` |
| Pipeline blocks | `test_blocks.py` |
| Simulation feed-forward (all block combos) | `test_simulation.py` |
| Communications: OFDM / channel / ISAC / blocks | `test_comms_*.py` |
| Web UI registry / layout / editor / runner | `test_webapp.py` |

## Markers (opt-in tests)

Some tests need hardware or a human and are **skipped by default**. Enable them with an
env var:

| Marker | Needs | Enable with |
|--------|-------|-------------|
| `sionna` | Sionna RT + DrJit + LLVM (real ray tracing) | `RUN_SIONNA=1 pytest -m sionna` |
| `slow` | extra wall-clock (full RF chain, SNR sweeps) | `RUN_SLOW=1 pytest -m slow` |
| `gui` | a live server / display | `RUN_GUI=1 pytest -m gui` |

PowerShell: `$env:RUN_SLOW=1; pytest -m slow`

## Conventions

- Shared fixtures live in `conftest.py` (synthetic frames, a drop-in environment block,
  a tiny scenario, the library device). Don't redefine them in test files.
- Tests run on whatever device the library is configured for (CUDA if present, else CPU),
  so the suite is green both on CPU-only CI and on a CUDA dev box.
- The only test path that cannot be automated here is **real Sionna ray tracing**
  (`@pytest.mark.sionna`), which requires an LLVM toolchain; run it on a capable machine.
