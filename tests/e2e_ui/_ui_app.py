"""Page-object layer for the E2E Array Simulator web UI.

Every locator for the app lives HERE (one file to fix when markup changes), wrapped
into named Tasks/Questions built on the tiny Screenplay core in ``_ui_screenplay``.
Journeys in ``test_journeys.py`` compose these; they never touch a raw selector.

Waiting policy (from the tooling research): never sleep, never wait on
``networkidle`` (Dash's callback/hot-reload traffic can keep the network busy).
Wait on the concrete DOM effect of the callback we triggered -- an element becoming
visible, or its text/value changing.
"""

from __future__ import annotations

from typing import Any

from _ui_screenplay import Actor, Question, Task

# --------------------------------------------------------------------------- locators
# Component ids come straight from webapp/*.py (Dash ids are stable by design).
TABS = {
    "blocks": "Block Diagram",
    "scenario": "Scenario",
    "results": "Results",
}
SEL = {
    "title": "h2",
    # block diagram tab
    "cytoscape": "#block-cytoscape",
    "param_editor": "#block-param-editor",
    "nsteps": "#run-nsteps",
    "run_button": "#run-button",
    "run_status": "#run-status",
    # scenario tab
    "ref_dropdown": "#ref-scenario-dropdown",
    "load_button": "#load-ref-button",
    "scenario_json": "#scenario-json",
    "render_button": "#render-button",
    "validate_button": "#validate-button",
    "scenario_map": "#scenario-map",
    "scenario_summary": "#scenario-summary",
    "validation_output": "#validation-output",
    # results tab
    "results_content": "#results-tab-content",
    "plotly_graph": ".js-plotly-plot",
}

_DEFAULT_TIMEOUT = 15000  # ms


# ------------------------------------------------------------------------ interactions
# Thin helpers over the Playwright page; Tasks below call these.

def _goto(actor: Actor, url: str) -> None:
    actor.page.goto(url, wait_until="domcontentloaded")
    actor.page.wait_for_selector(SEL["title"], state="visible", timeout=_DEFAULT_TIMEOUT)


def _open_tab(actor: Actor, key: str) -> None:
    actor.page.get_by_text(TABS[key], exact=True).click()


def _fill(actor: Actor, sel: str, value: str) -> None:
    actor.page.fill(sel, value)


def _click(actor: Actor, sel: str) -> None:
    actor.page.click(sel, timeout=_DEFAULT_TIMEOUT)


def _wait_text_nonempty(actor: Actor, sel: str) -> None:
    actor.page.wait_for_function(
        """(s) => { const e = document.querySelector(s);
            return e && e.innerText.trim().length > 0; }""",
        arg=sel, timeout=_DEFAULT_TIMEOUT,
    )


def _select_dropdown_option(actor: Actor, sel: str, name: str) -> None:
    """Open a Dash dropdown and pick an option by its visible label (a real
    open-menu-then-click, like a user). Dash 4.x renders options as
    ``[role=option]`` in a listbox. After picking, wait until the trigger shows the
    chosen value, so the component's ``value`` prop has committed before a following
    action (e.g. clicking Load, which reads that value as callback State) fires."""
    actor.page.click(sel)
    actor.page.get_by_role("option", name=name, exact=True).click(timeout=_DEFAULT_TIMEOUT)
    actor.page.wait_for_function(
        """([s, n]) => { const e = document.querySelector(s + '-value');
            return e && e.innerText.includes(n); }""",
        arg=[sel, name], timeout=_DEFAULT_TIMEOUT,
    )


def _tap_cytoscape_node(actor: Actor, block_id: str) -> None:
    """Tap a canvas-rendered dash-cytoscape node by id.

    Cytoscape.js draws nodes onto a <canvas>, so there is no per-node DOM element to
    target with a CSS click, and pixel-coordinate clicks are brittle (they depend on
    pan/zoom and can hit an overlapping node). We instead reach the Cytoscape instance
    the component stores on the container (``._cyreg.cy``) and dispatch a ``tap`` event
    on the specific node -- the exact event dash-cytoscape listens for to populate
    ``tapNodeData`` (the same handler a user's click ultimately fires). This keeps the
    interaction node-targeted and deterministic. Selection through the real event path
    is verified by waiting for the parameter editor to actually swap.
    """
    page = actor.page
    # Wait until the cytoscape instance exists AND the target node has been added
    # (its elements arrive via a Dash callback shortly after the tab mounts).
    page.wait_for_function(
        """([sel, id]) => { const el = document.querySelector(sel);
            if (!(el && el._cyreg && el._cyreg.cy)) return false;
            const n = el._cyreg.cy.$id(id); return !!(n && n.length > 0); }""",
        arg=[SEL["cytoscape"], block_id], timeout=_DEFAULT_TIMEOUT,
    )
    # Capture the editor before the tap so we can wait for the tapNodeData callback
    # to actually swap its contents (rather than racing an immediate read).
    before = page.text_content(SEL["param_editor"]) or ""
    page.evaluate(
        """([sel, id]) => {
            const cy = document.querySelector(sel)._cyreg.cy;
            cy.$id(id).emit('tap');
        }""",
        [SEL["cytoscape"], block_id],
    )
    page.wait_for_function(
        """([sel, prev]) => { const e = document.querySelector(sel);
            return e && e.innerText.trim().length > 0 && e.innerText !== prev; }""",
        arg=[SEL["param_editor"], before], timeout=_DEFAULT_TIMEOUT,
    )


# ------------------------------------------------------------------------------- Tasks

def open_app(url: str) -> Task:
    return Task.where("open the app", lambda a: _goto(a, url))


def go_to_tab(key: str) -> Task:
    return Task.where(f"go to the {key} tab", lambda a: _open_tab(a, key))


def select_reference_scenario(name: str) -> Task:
    return Task.where(
        f"select reference scenario {name!r}",
        lambda a: _select_dropdown_option(a, SEL["ref_dropdown"], name),
    )


def load_reference_scenario() -> Task:
    def _do(a: Actor) -> None:
        # The editor already holds the default scenario JSON, so wait for the value to
        # *change* (the load callback replacing it), not merely to be non-empty.
        before = a.page.input_value(SEL["scenario_json"])
        _click(a, SEL["load_button"])
        a.page.wait_for_function(
            """([s, prev]) => { const e = document.querySelector(s);
                return e && e.value.length > 0 && e.value !== prev; }""",
            arg=[SEL["scenario_json"], before], timeout=_DEFAULT_TIMEOUT,
        )
    return Task.where("load the reference scenario into the editor", _do)


def set_scenario_json(text: str) -> Task:
    return Task.where("type scenario JSON into the editor",
                      lambda a: _fill(a, SEL["scenario_json"], text))


def validate_scenario() -> Task:
    def _do(a: Actor) -> None:
        _click(a, SEL["validate_button"])
        _wait_text_nonempty(a, SEL["validation_output"])
    return Task.where("validate the scenario", _do)


def preview_scenario() -> Task:
    def _do(a: Actor) -> None:
        _click(a, SEL["render_button"])
        # the map is a dcc.Graph; wait for Plotly to draw its SVG, and the summary text
        a.page.wait_for_selector(f"{SEL['scenario_map']} .main-svg",
                                 state="attached", timeout=_DEFAULT_TIMEOUT)
        _wait_text_nonempty(a, SEL["scenario_summary"])
    return Task.where("render / preview the scenario map", _do)


def select_block(block_id: str) -> Task:
    def _do(a: Actor) -> None:
        _tap_cytoscape_node(a, block_id)
    return Task.where(f"click the {block_id!r} block in the diagram", _do)


def _param_editor_checkbox() -> str:
    return f"{SEL['param_editor']} input[type=checkbox]"


def _param_editor_number_input() -> str:
    return f"{SEL['param_editor']} input[type=number]"


def _set_block_enabled(actor: Actor, block_id: str, desired: bool) -> None:
    """Select the block (tapping its node opens/rebuilds its param editor from the
    store), then drive its enabled checkbox to ``desired``. Idempotent: a no-op click
    is skipped so this is safe to call from either enable_block/disable_block."""
    _tap_cytoscape_node(actor, block_id)
    page = actor.page
    checkbox = _param_editor_checkbox()
    if page.is_checked(checkbox) != desired:
        page.click(checkbox)
    page.wait_for_function(
        """([s, want]) => { const e = document.querySelector(s);
            return e && e.checked === want; }""",
        arg=[checkbox, desired], timeout=_DEFAULT_TIMEOUT,
    )


def enable_block(block_id: str) -> Task:
    return Task.where(f"enable the {block_id!r} block",
                      lambda a: _set_block_enabled(a, block_id, True))


def disable_block(block_id: str) -> Task:
    return Task.where(f"disable the {block_id!r} block",
                      lambda a: _set_block_enabled(a, block_id, False))


def set_open_block_number_param(value: Any) -> Task:
    """Fill the currently-open editor's FIRST number param and commit it (the
    dcc.Input is debounce=True, so it only pushes to the store on Enter/blur)."""
    def _do(a: Actor) -> None:
        page = a.page
        sel = _param_editor_number_input()
        field = page.locator(sel).first
        field.fill(str(value))
        field.press("Enter")
        page.wait_for_function(
            """([s, want]) => { const e = document.querySelector(s);
                return e && e.value === want; }""",
            arg=[sel, str(value)], timeout=_DEFAULT_TIMEOUT,
        )
    return Task.where(f"set the open block's number param to {value!r}", _do)


def set_frames_to_run(n: int) -> Task:
    return Task.where(f"set frames-to-run = {n}",
                      lambda a: _fill(a, SEL["nsteps"], str(n)))


def run_pipeline() -> Task:
    def _do(a: Actor) -> None:
        _click(a, SEL["run_button"])
        # A successful run switches to the Results tab (unmounting run-status), so we
        # key completion on the results graphs appearing; a failed run leaves an error
        # in run-status instead. Wait for whichever happens. Generous timeout: the
        # first run pays a cold torch-import cost inside the server thread.
        a.page.wait_for_function(
            """() => {
                const g = document.querySelector(
                    '#results-tab-content .js-plotly-plot');
                if (g) return true;
                const s = document.querySelector('#run-status');
                return !!(s && /error|failed/i.test(s.innerText));
            }""",
            timeout=90000,
        )
    return Task.where("run the pipeline and wait for it to settle", _do)


# --------------------------------------------------------------------------- Questions

def title_text() -> Question:
    return Question("the page title", lambda a: a.page.text_content(SEL["title"]))


def scenario_json_value() -> Question:
    return Question("the scenario JSON editor",
                    lambda a: a.page.input_value(SEL["scenario_json"]))


def validation_text() -> Question:
    return Question("the validation output",
                    lambda a: a.page.text_content(SEL["validation_output"]))


def summary_text() -> Question:
    return Question("the scenario summary",
                    lambda a: a.page.text_content(SEL["scenario_summary"]))


def param_editor_text() -> Question:
    return Question("the block parameter editor",
                    lambda a: a.page.text_content(SEL["param_editor"]))


def run_status_text() -> Question:
    return Question("the run status",
                    lambda a: a.page.text_content(SEL["run_status"]))


def results_graph_count() -> Question:
    def _count(a: Actor) -> int:
        return a.page.locator(
            f"{SEL['results_content']} {SEL['plotly_graph']}"
        ).count()
    return Question("the number of result graphs", _count)


def results_content_visible() -> Question:
    return Question(
        "the results tab content is visible",
        lambda a: a.page.is_visible(SEL["results_content"]),
    )


def open_block_enabled() -> Question:
    return Question(
        "whether the open block's editor shows it enabled",
        lambda a: a.page.is_checked(_param_editor_checkbox()),
    )


def open_block_number_value() -> Question:
    return Question(
        "the open block's first number param value",
        lambda a: a.page.locator(_param_editor_number_input()).first.input_value(),
    )


def results_text() -> Question:
    return Question(
        "the results tab content text",
        lambda a: a.page.inner_text(SEL["results_content"]),
    )
