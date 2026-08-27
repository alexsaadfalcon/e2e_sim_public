# Contributing

Thanks for your interest in the Array Processing End-to-End Simulator. This guide
covers the essentials for getting set up and submitting changes. New here? Start
with [`docs/FIRST_SCENARIO.md`](docs/FIRST_SCENARIO.md) for a hands-on walkthrough,
and keep [`docs/GLOSSARY.md`](docs/GLOSSARY.md) open for terms you don't recognize.

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
pytest                      # hands-off: synthetic data, no GPU required, no display
```

The full default suite takes about 4 minutes; `tests/README.md` has the current
file/test-count breakdown by area (counts drift as tests are added — that file is
the source of truth, not this one). Run a single file or `-k` a keyword while
iterating (`pytest tests/test_blocks.py`, `pytest -k afe`) and the full suite before
opening a PR.

Hardware/human tests are skipped by default and opt-in via environment variables:
`RUN_SIONNA=1` (real Sionna RT generation — needs the Sionna/DrJit stack, not
runnable in most CI/dev environments), `RUN_SLOW=1` (full RF chain / SNR sweeps —
extra wall-clock, no special hardware), `RUN_GUI=1` (live server / display). CI runs
the default suite on every push/PR (`.github/workflows/tests.yml`).

Please keep the suite green and add tests for new behavior. A few conventions:

- **Device.** `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
  is the pattern everywhere in `e2e/` — never hardcode `"cpu"` in library code or a
  test. Tests run on whatever the library resolves to, so the suite is green both on
  CPU-only CI and a CUDA dev box; compare `tensor.device.type` (`"cuda"`), not an
  exact index (`"cuda:0"`), since the index isn't guaranteed.
- **Tolerances.** Tensors are torch `complex64`; use float32-friendly tolerances
  (e.g. `1e-2` for subspace-distance-style checks), not `1e-6`.
- Reuse the shared fixtures in `tests/conftest.py` (`synthetic_frame(s)_np`,
  `tmp_pkl_frames`, `make_env_block`, `small_scenario`, `torch_device`) — don't
  redefine them in a new test file.
- To check that a module imports without `torch`, do it in a subprocess (see
  `tests/test_webapp.py::_import_without_torch`) — never pop/reload `torch`
  in-process; it leaves torch half-initialized for every later test in the run.

## Architecture orientation

- **Two layers, kept distinct.** The runtime pipeline (`e2e/blocks.py`,
  `e2e/simulation.py`) consumes precomputed `.pkl` frames — fast, no Sionna/GPU
  needed. Scenario generation (`e2e/environment/`) uses Sionna RT (or a
  Sionna-free `--dry-run` mode) to produce those `.pkl` files from a declarative
  `Scenario`. See `CLAUDE.md` and `docs/GLOSSARY.md` for the fuller map, and
  `docs/FIRST_SCENARIO.md` for both layers exercised end to end.
- **Blocks.** A downstream/product block implements `apply(state_dict) -> dict`
  (`FFTBlock`, `RangeAzBlock`, `SubspaceErrorBlock`, ...); `Simulation` composes
  them, feeding each the running state dict. The web UI registry (what the block
  diagram UI shows/edits) is `webapp/pipeline_registry.py`.
- **The block/frame API contract (`frame_capabilities`).** Every block/stage that
  consumes `s_pars` declares a `FrameCapabilities` (`e2e/frames.py`) as a class
  attribute: whether it accepts MIMO (`accepts_mimo`), how it treats the chirp axis
  (native / broadcast-per-chirp / single-chirp-only), which signal domain it wants
  (frequency-domain CFR by default), and whether it needs the full aperture or
  tolerates a compressed measurement. The default is the historical contract — **no
  MIMO, single chirp** (`s_pars.shape[1] == n_tx == 1`, `shape[2] == n_chirp == 1`)
  — so a new block that declares nothing keeps exactly that behavior. `Simulation`
  validates every incoming frame against the callee's declaration and raises a
  named `FrameContractError` (not a bare `assert`) on a mismatch, so widening a
  block to accept more shapes is a one-line declaration change, not an assertion
  hunt. Adding a block that legitimately needs a shape the defaults reject (MIMO,
  multi-chirp, time-domain) means declaring the wider capability explicitly, not
  loosening the default for everyone.
- **Scenarios.** `e2e/scenario.py` is the dependency-free, JSON-serializable contract
  shared by the UI, the generator, and the examples.
- **Layering rule (core never imports `e2e.ml`).** `e2e/ml` is a *consumer* of the
  core signal model (`e2e/chain`, `e2e/environment`), not its owner — established by
  the C1 refactor (`notes/C1_MOVE_PLAN.md`, not shipped in this tree but referenced
  from `e2e/ml/__init__.py`). Concretely: `e2e/chain`, `e2e/environment`, and
  `e2e/blocks.py`/`e2e/simulation.py` must never import `e2e.ml` at module scope.
  Two documented exceptions import it *lazily* (inside a function, at the single
  call site that needs it) — `e2e.radar_config.answerability_problems` and
  `e2e.environment.blocks.RTEnvironmentBlock`'s label encoder — and both say so in a
  comment at the import. If you're adding a new dependency from core code into
  `e2e.ml`, that's the pattern to follow (or a sign the code you need belongs in
  core, not in `e2e.ml`, in the first place).

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
