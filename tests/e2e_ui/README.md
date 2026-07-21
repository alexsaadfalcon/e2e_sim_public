# Browser-driven UI journey tests (`tests/e2e_ui`)

Procedural, **user-journey** tests for the Dash web UI (`webapp/`). Instead of
fine-grained `assert`s, each test drives a real headless browser through the exact
sequence of clicks a user performs — open the app, pick a scenario, load it, validate
it, run the pipeline — and checks state **at each step**, not just at the end. This is
the E2E / golden-path style, expressed with a small **Screenplay pattern** layer so a
test reads as a script of named actions.

## How to run

These tests are gated behind the `browser` marker and **skip by default** (like
`sionna`/`slow`/`gui`). They need Playwright's Chromium browser.

```bash
pip install playwright          # already a dev dep; the Python package
playwright install chromium     # one-time: fetch the browser binary
RUN_BROWSER=1 pytest tests/e2e_ui
```

Nothing to run manually — the harness boots the real Dash app in a background thread
on a free port (see `_ui_server.py`) and points the browser at it. The lighter
journeys (tabs, scenario authoring, validation) need no torch; the pipeline-Run
journey additionally skips unless torch **and** the `munich.pkl` frames are present.

## The pattern (why it reads the way it does)

Layers, smallest first:

| Layer | File | Role |
|-------|------|------|
| **Screenplay core** | `_ui_screenplay.py` | `Actor`, `Task`, `Question`, `Ensure` + matchers. App-agnostic. |
| **Page objects** | `_ui_app.py` | Every locator lives here (one file to fix on markup change), wrapped into named `Task`s (`load_reference_scenario()`) and `Question`s (`validation_text()`). |
| **Server** | `_ui_server.py` | Serves `webapp.app` in a daemon thread; polls it live before yielding. |
| **Journeys** | `test_journeys.py` | Compose Tasks + `Ensure` checkpoints into readable user stories. |

A journey reads like this — the sequence *is* the test:

```python
actor.attempts_to(
    ui.open_app(base_url),
    ui.go_to_tab("scenario"),
    ui.select_reference_scenario("etoile_radar"),
    ui.load_reference_scenario(),
    Ensure.that(ui.scenario_json_value(), contains("etoile"),
                "loading pulls the selected reference into the editor"),
    ui.validate_scenario(),
    Ensure.that(ui.validation_text(), contains("no problems"),
                "a reference scenario validates as OK"),
    ui.preview_scenario(),
)
```

Every step is named, so a failure is reported as
`[A user] while 'load the reference scenario into the editor': ...` rather than a bare
assertion deep in a helper. **Put `Ensure` checkpoints along the journey**, not only at
the end, so a failure localizes to the step that broke.

## Adding a journey

1. Need a new element? Add its selector to `SEL` in `_ui_app.py` (nowhere else).
2. Need a new action/state? Add a `Task`/`Question` factory in `_ui_app.py`.
3. Write the journey in `test_journeys.py` as `actor.attempts_to(...)` with checkpoints.

`playwright codegen http://127.0.0.1:8050` (against a manually launched `python -m
webapp.app`) records a click sequence you can crib selectors/order from — treat its
output as a rough skeleton, then express it through the Task/Question layer.

## Conventions that keep these stable

- **Never `time.sleep`; never wait on `networkidle`.** Dash's callback traffic can keep
  the network busy forever. Wait on the concrete DOM effect of the callback you
  triggered — an element visible, or its text/value *changing* (see
  `load_reference_scenario`, which waits for the editor to change, not merely to be
  non-empty; the old value already looked "valid").
- **Fresh browser context per test** (`page` fixture) so journeys don't leak state.
- **User-facing locators where possible** (`get_by_text`, `get_by_role("option", ...)`);
  Dash component `id`s are stable by design and are used as anchors.
- **Canvas caveat (dash-cytoscape).** Cytoscape draws nodes to a `<canvas>`, so there is
  no per-node DOM element and pixel clicks are brittle. `_tap_cytoscape_node` reaches the
  Cytoscape instance the component stores on its container (`._cyreg.cy`) and dispatches
  a `tap` event on the node by id — the exact event dash-cytoscape turns into
  `tapNodeData`, i.e. the same handler a user's click fires — then waits for the
  parameter editor to actually swap. Deterministic and node-targeted.

## CI

Add a dedicated job (kept out of the default fast suite):

```yaml
- run: pip install -r requirements-dev.txt playwright
- run: playwright install --with-deps chromium   # browser + OS libs, no xvfb needed
- run: RUN_BROWSER=1 pytest tests/e2e_ui
```

Playwright's Chromium is self-contained and behaves identically on Windows (dev) and
Linux (CI) — no ChromeDriver/Chrome version matching.
