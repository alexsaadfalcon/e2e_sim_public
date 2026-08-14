# Contributing

Thanks for your interest in the Array Processing End-to-End Simulator. This guide
covers the essentials for getting set up and submitting changes.

## Development setup

```bash
pip install -e ".[dev]"     # core + test deps (also install torch; see requirements-dev.txt)
# optional, for the web UI / real frame generation:
pip install -e ".[webapp]"
pip install -e ".[sionna]"
```

The runtime pipeline and the test suite need only `torch`; the web UI shell and the
scenario spec import without `torch`/Sionna. Heavy ray-traced generation needs the
`[sionna]` extra, an LLVM toolchain, and a CUDA-12.x driver — see the README's
"GPU / driver / LLVM" section.

## Running the tests

```bash
pytest                      # hands-off: synthetic data, no GPU, no display
```

Hardware/human tests are skipped by default and opt-in via environment variables:
`RUN_SIONNA=1` (real Sionna RT generation), `RUN_SLOW=1` (full RF chain / sweeps),
`RUN_GUI=1` (live server). CI runs the default suite on every push/PR.

Please keep the suite green and add tests for new behavior. A few conventions:

- Tests run on the library's device (CUDA if present, else CPU) — never hardcode CPU.
- Reuse the shared fixtures in `tests/conftest.py` (`synthetic_frame(s)_np`,
  `tmp_pkl_frames`, `make_env_block`, `small_scenario`, `torch_device`).
- To check that a module imports without `torch`, do it in a subprocess (see
  `tests/test_webapp.py`) — never pop/reload `torch` in-process.

## Architecture orientation

- **Two layers, kept distinct.** The runtime pipeline (`e2e/blocks.py`,
  `e2e/simulation.py`) consumes precomputed `.pkl` frames. Scenario generation
  (`e2e/environment/`) uses Sionna RT to produce them. See `CLAUDE.md` for a fuller map.
- **Blocks.** A pipeline stage implements `apply(state_dict) -> dict`; downstream
  blocks compose. The web UI registry is `webapp/pipeline_registry.py`.
- **Scenarios.** `e2e/scenario.py` is the dependency-free, JSON-serializable contract
  shared by the UI, the generator, and the examples.

See `ROADMAP.md` for where the project is headed if you're looking for high-leverage
places to contribute — currently: further compressed-domain (reduced-dimension)
processing blocks (a first one, `RangeProfileBlock`, shows the pattern), the
MIMO/multi-chirp shape-contract widening (A3), and sequence-aware dataset loading for
the ML models.

## Submitting changes

- Branch from `main`, keep commits focused, and write a clear commit message.
- Run `pytest` before opening a pull request and note any gated tests you exercised.
- Describe the change and its rationale in the PR; link any related issue.

## License

By contributing, you agree that your contributions are licensed under the project's
MIT License (see `LICENSE`).
