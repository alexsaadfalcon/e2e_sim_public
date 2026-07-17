# Web UI — Array Processing End-to-End Simulator

A Dash + Plotly + dash-cytoscape web interface for the simulator. It replaces the
old Tkinter GUI (`e2e/main/main_sionna_interactive.py`) and adds a scenario editor.

## Run

```bash
python -m webapp.app          # preferred
# or
python webapp/app.py
```

Then open <http://127.0.0.1:8050> (host `127.0.0.1`, port `8050`, debug off).

## Required dependencies

The web UI deps are packaged as the `webapp` extra (see the root `pyproject.toml`),
so install them through the project rather than pinning by hand:

```bash
pip install -e ".[webapp]"      # dash, plotly, dash-cytoscape via the extra
# or, with the requirements files:
pip install -r requirements-dev.txt   # also includes the dash/plotly/cytoscape set
```

`torch` (and the `e2e` package + its deps) are needed only for the **Run pipeline**
action. The app shell, block diagram, and scenario editor all build and render
without torch/sionna installed — heavy imports are done lazily inside callbacks.

## Tabs

### Block Diagram
A dash-cytoscape node graph of the runtime pipeline. The node list and the
dataflow edges are **derived from a single registry** (`pipeline_registry.py`,
the `BLOCKS`/`EDGES` definitions), mirroring `e2e/blocks.py` and the feed-forward
order in `e2e/simulation.py`:

```
Sionna Environment -> RFFE -> Interconnect -> AFE -> AdaOja Subspace
                                                         |-> FFT
                                                         |-> Range-Azimuth
                                                         |-> Range-Elevation
                                                         `-> Subspace Error
```

Click a block to toggle it on/off and edit its key params
(RFFE `signal_scaling`/`chirp_dur`, AFE `exp`/`mantissa`, FFT `bins`,
subspace `d`, environment scenario, interconnect case). Disabled blocks dim and
their edges dash. **Run pipeline** runs N frames and switches to **Results**.

### Scenario
Place/edit antenna nodes and scene objects on a 2D x/y map (Plotly scatter,
role-colored markers). Editing is done through the **Scenario JSON** text box
(schema = `e2e.scenario.Scenario`), which is the canonical shared spec:

* **Load reference** — load a `REFERENCE_SCENARIOS` entry (`munich_radar`,
  `munich_isac`).
* **Render / Preview** — redraw the map + summary from the JSON.
* **Validate** — run `Scenario.validate()` and list any problems.
* **Download / Upload JSON** — round-trip via `Scenario.to_json/from_json`.
* **Generate frames** — shells out to the offline generator:
  `python -m e2e.environment.scenario_runner --scenario <tmp.json> --dry-run`.
  Output is streamed back into the page. If that module is missing or errors, the
  UI degrades gracefully with a message (it never crashes the server). Dry-run
  mode produces synthetic frames so it works without Sionna.

### Results
Plotly figures from the most recent run: FFT, Range-Azimuth, Range-Elevation
(dB heatmaps) and the per-frame Subspace Error line plot.

## Module map

| File                   | Responsibility |
| ---------------------- | -------------- |
| `app.py`               | Dash app shell, tab layout, all callbacks, `python -m webapp.app` entry point. |
| `pipeline_registry.py` | Pure-data registry of blocks/edges/params (no heavy imports). Single source of truth. |
| `block_diagram.py`     | Cytoscape elements + stylesheet + parameter editor (derived from the registry). |
| `pipeline_runner.py`   | Lazily imports torch/e2e, builds blocks from UI state, runs the sim, makes figures. |
| `scenario_editor.py`   | Scenario tab layout, 2D map figure, JSON helpers (imports the dependency-free `e2e.scenario`). |

## Integration notes / limitations

* **Run pipeline needs frames + torch.** It loads precomputed `.pkl` S-parameter
  frames via `e2e.environment.sionna_iterator`. Missing frames (e.g. selecting
  `etoile` with no `etoile.pkl`) surface a clear "generate frames first" message
  instead of crashing. Generate frames first via the Scenario tab.
* **Backend block-combination constraint.** The current, unmodified
  `e2e/simulation.feed_forward()` (a) reads `PRX` unconditionally — only assigned
  when the RFFE block is present — and (b) takes an as-yet-unwired subspace-update
  path when AFE is off. So a clean run currently requires **RFFE + AFE + Subspace
  enabled** (the defaults). `pipeline_runner.py` detects the incompatible combos
  up front and explains, rather than letting the run crash. These constraints live
  in `e2e/` and are out of scope for this UI to change.
* The 2D editor edits positions/roles/arrays/motion through the JSON box (the
  source of truth) rather than drag-on-canvas, keeping the `e2e.scenario` schema
  authoritative and round-trippable.
