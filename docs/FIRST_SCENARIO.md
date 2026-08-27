# Your first scenario

A copy-pasteable walkthrough from a clean clone to a running pipeline and a live web
UI — entirely on CPU, with no GPU and no Sionna install required. Every command below
was run against this repo to produce the output shown; if your output differs
materially, that's a bug report, not user error.

Unfamiliar term? Check [`docs/GLOSSARY.md`](GLOSSARY.md). For the deeper "why", see
[`CLAUDE.md`](../CLAUDE.md) and [`CONTRIBUTING.md`](../CONTRIBUTING.md).

## 0. Clone and install

```bash
git clone <this-repo-url> e2e_sim_public
cd e2e_sim_public
pip install -e ".[dev,webapp]"
# torch is best installed separately so you get the right CPU/CUDA build, e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Expected:** `pip install -e .` finishes with `Successfully installed e2e-sim-1.0.0`
(or similar); `python -c "import e2e, torch"` exits with no output/error. Nothing here
needs Sionna, DrJit, or a GPU — the `[sionna]` extra is a separate, optional install
(see the README's "GPU / driver / LLVM" section) and is not needed for anything below.

## 1. Dry-run a scenario (no Sionna, no GPU)

`scenario_runner` turns a declarative `Scenario` into S-parameter `.pkl` frames. Its
`--dry-run` flag exercises the full scheduling/motion/serialization path but
synthesizes the frame values analytically instead of ray tracing them, so it needs
neither Sionna nor a GPU:

```bash
python -m e2e.environment.scenario_runner --scenario munich_radar --dry-run --frames 5 \
    --out scratch/first_scenario_dryrun.pkl
```

> **Always pass `--out` for a dry run.** Without it the runner writes to the scenario's
> default path under `e2e/environment/sionna_sims/`, **overwriting** any real
> ray-traced `.pkl` you already have there with synthetic frames. Those files are
> gitignored, so git cannot bring one back — regenerating costs a full RT run. (This
> bit us while writing this page, 2026-08-27.)

**Expected output** (frame count and the final path will match what you passed):

```
======================================================================
scenario:   munich_radar  (base_scene=munich)
frames:     5
frequency:  28.500-31.500 GHz, 5000 bins (carrier 30.000 GHz)
objects:    []
links (1):
  - radar [radar]: tx=radar (n_tx_ant=1) -> rx=radar (n_rx_ant=1024); frame (1024, 1, 1, 5000)
dumped:     v2 payload: meta + 1 link array(s)
mode:       DRY-RUN (synthetic, no Sionna)
moving:     ['radar']
======================================================================
  frame 1/5
  ...
  frame 5/5
dumping to file .../e2e/environment/sionna_sims/munich_radar.pkl
done dumping  1 links {'radar': (5, 1024, 1, 1, 5000)}
----------------------------------------------------------------------
Generated 5 frames x 1 links -> .../e2e/environment/sionna_sims/munich_radar.pkl
  link radar: (5, 1024, 1, 1, 5000), dtype complex64
(dry-run: synthetic data; replace with real generation by dropping --dry-run on a
machine with Sionna RT installed.)
```

This writes (overwrites) `e2e/environment/sionna_sims/munich_radar.pkl`. That
directory is gitignored — the `.pkl` it just wrote is scratch output, not something
to commit. `munich_radar` is a *generation* scenario name (a `REFERENCE_SCENARIOS`
entry); it is not the same namespace as the *runtime* base-scene name `munich` used
in the next step (see `docs/GLOSSARY.md`'s "two scenario namespaces" entry, and the
README's callout of the same name, for why they're spelled differently on purpose).

## 2. Run the pipeline on the shipped Munich frames

The repo does **not** ship the `.pkl` frames the runtime radar pipeline reads by
default (they're gitignored — see the README's note under Quickstart) — you
generate them with real Sionna RT or, for the synthetic-fallback comms/ISAC
examples, run those directly (`python -m e2e.main.main_comms_link`, etc., which work
with no `.pkl` at all). If a `munich.pkl` *is* present (e.g. you generated one, or
you're working in a checkout that has it), run the full radar pipeline against it:

```bash
python -m e2e.main.main_sionna_blocks
```

This builds `SionnaEnvironmentBlock('munich')` -> `RFFEBlock` -> `InterconnectBlock`
-> `AFEBlock` -> `AdaOjaBlock` -> `FFTBlock`/`RangeAzBlock`/`RangeElBlock`/
`SubspaceErrorBlock`, runs 2 frames, and (since `__main__` calls `main(show=True)`)
saves the subspace-error curve and two azimuth-elevation maps under
`e2e/main/figures/`.

**Expected:** it completes in a few seconds on CPU and prints a `tqdm` progress bar
(`RUNNING ARRAY SIMULATION: 100%|##########| 2/2`). We verified the underlying call
directly (`main(scenario_name='munich', n_steps=2, k=8, show=False)`, i.e. the same
code path minus the plotting) against this repo's shipped `munich.pkl`:

```
RUNNING ARRAY SIMULATION: 100%|##########| 2/2 [00:02<00:00, 1.03it/s]
keys: ['effective_rank', 'sv_gap_at_k', 'sv_gap_norm', 'rank_ok', 'n_refine_used',
       'fft', 'range_az', 'range_el', 'subspace_err']
```

Each key is a length-2 list (one entry per frame). `subspace_err` is the number to
watch: the tracked-vs-true subspace distance each frame (see `docs/GLOSSARY.md`).

If frames aren't present, the script raises a `FileNotFoundError` with a message
that names both fixes (generate real frames, or run `scenario_runner --dry-run` as
in step 1) — that message is itself part of the tested contract
(`tests/test_main_sionna_blocks.py`).

## 3. Launch the web UI (shell only, no server left running)

```bash
python -m webapp.app      # serves at http://127.0.0.1:8050
```

Ctrl-C to stop it. It has three tabs: **Block Diagram** (drag/connect pipeline
blocks, edit params, run), **Scenario** (place nodes/objects, generate frames), and
**Results** (heatmaps + subspace-error curve). The shell and scenario editor import
without `torch`/Sionna; running a pipeline needs `torch` + frames.

**Verify without opening a browser or binding a port** (what this walkthrough
actually ran, and what `tests/test_webapp.py` does): import the app shell in a
subprocess and confirm it did not pull in `torch`:

```bash
python -c "import subprocess, sys; \
p = subprocess.run([sys.executable, '-c', \
  \"import importlib, sys; importlib.import_module('webapp.app'); \
    sys.exit(0 if 'torch' not in sys.modules else 3)\"], \
  capture_output=True); print('rc', p.returncode)"
```

**Expected:** `rc 0`.

## 4. Swap a block and see the output change

The webapp's "Run" button builds blocks from a `{block_id: {"enabled", "params"}}`
state dict and calls `webapp.pipeline_runner.run_pipeline(state)` — the same
function whether you click it in the browser or call it from a script. We'll swap
the **Interconnect** block's `case` param exactly as the UI's dropdown does: from
the default 11-tap boxcar placeholder to `case3` (an identity pass-through, see
`InterconnectBlock`'s docstring in `e2e/blocks.py`), and watch `subspace_err` move.

```bash
python - <<'PY'
import copy
from webapp.pipeline_registry import default_block_state
from webapp.pipeline_runner import run_pipeline

state = default_block_state()
state["interconnect"]["enabled"] = True
state["interconnect"]["params"]["case"] = "default"   # boxcar placeholder
out_default = run_pipeline(state, n_steps=2)

state2 = copy.deepcopy(state)
state2["interconnect"]["params"]["case"] = "case3"     # identity pass-through
out_case3 = run_pipeline(state2, n_steps=2)

print("subspace_err (default boxcar):  ", [float(x) for x in out_default["subspace_err"]])
print("subspace_err (case3 pass-through):", [float(x) for x in out_case3["subspace_err"]])
PY
```

**Expected (values will vary slightly run to run — RFFE injects thermal noise — but
the qualitative gap is consistent):**

```
subspace_err (default boxcar):   [1.547, 1.759]
subspace_err (case3 pass-through): [0.041, 0.059]
```

The boxcar placeholder's frequency-domain ripple visibly degrades subspace tracking
relative to the identity pass-through — the same comparison the README's
"Interconnect: placeholder vs. measured" section makes with real measured transfer
functions instead of the toy `case3` identity. To do the equivalent thing in the
browser: open the **Block Diagram** tab, toggle **Interconnect** on, change its
**Case** dropdown from `default` to `case3`, click **Run**, and compare the
**Results** tab's subspace-error curve before and after.

## Where to go next

- `CONTRIBUTING.md` — environment setup, test conventions, the block API contract.
- `docs/GLOSSARY.md` — definitions for every term used above.
- The root `README.md`'s Cookbook table — where to look to add a block, define a
  scenario, or extend comms/ISAC.
- `e2e/ml/README.md` — generating labeled radar ML data and training a detector.
