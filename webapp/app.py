"""
Web UI for the Array Processing End-to-End Simulator (Dash + Plotly + cytoscape).

Two capabilities across three tabs:

  * Block Diagram - a dash-cytoscape node graph of the runtime pipeline; toggle
    blocks, edit params, and Run the pipeline.
  * Scenario      - place/edit nodes & objects on a 2D map, edit/validate/load/
    save the Scenario JSON, and trigger offline frame generation.
  * Results       - Plotly figures from the most recent pipeline run.

Design rules (so the app imports/launches without torch or sionna):
  * NO heavy imports at module top. torch / e2e.blocks / e2e.simulation are
    imported lazily inside the Run callback (via webapp.pipeline_runner).
  * e2e.scenario IS imported (it is dependency-free by design).

Run it with:
    python -m webapp.app
    # or
    python webapp/app.py
Then open http://127.0.0.1:8050
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from typing import Any, Dict

from dash import (
    ALL,
    Dash,
    Input,
    Output,
    State,
    callback_context,
    ctx,
    dcc,
    html,
    no_update,
)

from webapp import block_diagram, scenario_editor
from webapp.pipeline_registry import BLOCKS_BY_ID, PRODUCT_IDS, default_block_state
from webapp.pipeline_runner import (
    PipelineError,
    figures_from_outputs,
    placeholder_figure,
    run_pipeline,
)
from webapp.scenario_editor import map_figure, scenario_from_json_safe, summarize

HOST = "127.0.0.1"
PORT = 8050

app = Dash(__name__, suppress_callback_exceptions=True, title="E2E Array Simulator")
server = app.server  # exposed for gunicorn/WSGI if ever needed

# Result figures are stored as Plotly figure dicts in this Store between runs.
EMPTY_RESULTS: Dict[str, Any] = {}


def _app_layout() -> Any:
    return html.Div([
        html.H2("Array Processing End-to-End Simulator",
                style={"marginBottom": "0"}),
        html.P("Block-diagram pipeline control and scenario scheduling.",
               style={"color": "#576574", "marginTop": "2px"}),

        # Client-side state stores.
        dcc.Store(id="block-state-store", data=default_block_state()),
        dcc.Store(id="results-store", data=EMPTY_RESULTS),
        # Per-session last-rendered diagram signature (see _render_diagram): starts
        # None each session/refresh so a fresh client always gets its first render.
        dcc.Store(id="diagram-sig-store", data=None),

        dcc.Tabs(id="tabs", value="tab-blocks", children=[
            dcc.Tab(label="Block Diagram", value="tab-blocks",
                    children=html.Div(block_diagram.layout(), style={"padding": "12px"})),
            dcc.Tab(label="Scenario", value="tab-scenario",
                    children=html.Div(scenario_editor.layout(), style={"padding": "12px"})),
            dcc.Tab(label="Results", value="tab-results",
                    children=html.Div(id="results-tab-content", style={"padding": "12px"})),
        ]),
    ], style={"maxWidth": "1280px", "margin": "0 auto", "fontFamily": "Segoe UI, Arial, sans-serif",
              "padding": "12px"})


app.layout = _app_layout


# =================================================================================
# Block Diagram tab callbacks
# =================================================================================

# Only the enabled-set changes the cytoscape graph structure (node "disabled"
# class + dashed/"inactive" edges). Param edits do NOT affect build_elements, yet
# re-emitting `elements` mid-edit makes cytoscape drop the current node selection,
# closing the param editor under the user. We therefore only re-output elements
# when the enabled-set actually changed, so param-only edits leave the graph (and
# the selection) intact.
#
# The last-rendered signature lives in a PER-SESSION dcc.Store, never module state:
# the cytoscape ships with elements=[] and a fresh client (new session OR page
# refresh) must always get its first render. A module-level cache is shared across
# sessions/refreshes, so the second client's first callback would compare equal and
# no_update into a permanently blank diagram.
def _enabled_signature(block_state: Dict[str, Any]) -> list:
    """Canonical, JSON-safe enabled-set signature (dcc.Store round-trips JSON,
    so use lists -- tuples would come back as lists and never compare equal)."""
    bs = block_state or {}
    return sorted(
        [bid, bool(st.get("enabled", False))]
        for bid, st in bs.items()
    )


@app.callback(
    Output("block-cytoscape", "elements"),
    Output("diagram-sig-store", "data"),
    Input("block-state-store", "data"),
    State("diagram-sig-store", "data"),
)
def _render_diagram(block_state, last_sig):
    """Redraw the cytoscape graph only when the enabled-set (graph structure) changes."""
    state = block_state or default_block_state()
    sig = _enabled_signature(state)
    if sig == last_sig:
        # Param-only edit (or no structural change): keep the graph + selection.
        return no_update, no_update
    return block_diagram.build_elements(state), sig


@app.callback(
    Output("block-param-editor", "children"),
    Input("block-cytoscape", "tapNodeData"),
    State("block-state-store", "data"),
)
def _show_param_editor(node_data, block_state):
    """Show the parameter editor for the tapped block."""
    block_state = block_state or default_block_state()
    block_id = (node_data or {}).get("id", PRODUCT_IDS[0])
    return block_diagram.param_editor(block_id, block_state)


@app.callback(
    Output("block-state-store", "data"),
    Input({"role": "block-enabled", "block": ALL}, "value"),
    Input({"role": "block-param", "block": ALL, "param": ALL}, "value"),
    State("block-state-store", "data"),
    prevent_initial_call=True,
)
def _update_block_state(enabled_values, param_values, block_state):
    """Persist edits from the (pattern-matched) param controls into the store."""
    block_state = block_state or default_block_state()
    triggered = callback_context.triggered_id
    if triggered is None:
        return no_update

    role = triggered.get("role")
    if role == "block-enabled":
        bid = triggered["block"]
        # find this control's value among the ALL list
        for inp, val in zip(callback_context.inputs_list[0], enabled_values):
            if inp["id"]["block"] == bid:
                block_state.setdefault(bid, {})["enabled"] = bool(val)
                break
    elif role == "block-param":
        bid, pkey = triggered["block"], triggered["param"]
        for inp, val in zip(callback_context.inputs_list[1], param_values):
            cid = inp["id"]
            if cid["block"] == bid and cid["param"] == pkey:
                block_state.setdefault(bid, {}).setdefault("params", {})[pkey] = val
                break
    return block_state


@app.callback(
    Output("results-store", "data"),
    Output("run-status", "children"),
    Output("tabs", "value"),
    Input("run-button", "n_clicks"),
    State("block-state-store", "data"),
    State("run-nsteps", "value"),
    prevent_initial_call=True,
)
def _run_pipeline(n_clicks, block_state, n_steps):
    """Run the pipeline (lazy heavy imports inside) and stash result figures."""
    block_state = block_state or default_block_state()
    try:
        outputs = run_pipeline(block_state, n_steps=int(n_steps or 10))
    except PipelineError as e:
        # Friendly, expected failure: stay on the diagram, show the message.
        return no_update, html.Span(str(e), style={"color": "#eb3b5a"}), no_update
    except Exception as e:  # unexpected — still don't crash the server
        return no_update, html.Span(f"Unexpected error: {e}",
                                    style={"color": "#eb3b5a"}), no_update

    figs = figures_from_outputs(outputs)
    # store as plain dicts (Plotly figures are JSON-serializable via to_dict)
    data = {k: f.to_dict() for k, f in figs.items()}
    note = ""
    # Physical scale mode trusts the frames to BE volts; nothing in a bare .pkl can
    # verify that (no metadata until the frames-carry-metadata refactor), so surface
    # the assumption instead of silently producing clipped/underdriven nonsense when
    # a legacy unit-energy pkl (e.g. the stock munich frames) is fed through it.
    scale_mode = block_state.get("rffe", {}).get("params", {}).get("scale_mode")
    if scale_mode == "physical":
        note = ("  [physical scale mode (forced): assumes frames were generated with "
                "tx_power_dbm set -- stock munich/etoile pkls are legacy-normalized]")
    elif scale_mode in (None, "auto") and (outputs.get("_axis_meta") or {}).get("from_meta"):
        # In auto mode the frames declare their own convention (v2 metadata), so no
        # assumption warning is needed -- but say what was detected, for transparency.
        note = "  [auto scale mode: following the frames' own metadata]"
    msg = html.Span(f"Run complete: {len(data)} product(s). See Results tab.{note}",
                    style={"color": "#20bf6b"})
    return data, msg, "tab-results"


# =================================================================================
# Results tab
# =================================================================================

@app.callback(
    Output("results-tab-content", "children"),
    Input("results-store", "data"),
    Input("tabs", "value"),
)
def _render_results(results_data, active_tab):
    """Render stored result figures as a grid of graphs."""
    if active_tab != "tab-results":
        return no_update
    if not results_data:
        return html.Div([
            html.H3("Results"),
            html.P("No results yet. Configure the pipeline on the Block Diagram "
                   "tab and click Run pipeline.", style={"color": "#576574"}),
        ])

    titles = {
        "fft": "FFT", "range_az": "Range-Azimuth",
        "range_el": "Range-Elevation", "subspace_err": "Subspace Error",
    }
    cards = []
    for key, fig_dict in results_data.items():
        cards.append(html.Div(
            dcc.Graph(figure=fig_dict),
            style={"flex": "1 1 45%", "minWidth": "420px", "margin": "6px",
                   "border": "1px solid #dfe4ea", "borderRadius": "6px",
                   "padding": "4px"},
        ))
    return html.Div([
        html.H3("Results"),
        html.Div(cards, style={"display": "flex", "flexWrap": "wrap"}),
    ])


# =================================================================================
# Scenario tab callbacks
# =================================================================================

@app.callback(
    Output("scenario-json", "value"),
    Input("load-ref-button", "n_clicks"),
    Input("upload-json", "contents"),
    State("ref-scenario-dropdown", "value"),
    prevent_initial_call=True,
)
def _load_scenario(load_clicks, upload_contents, ref_name):
    """Load a reference scenario or an uploaded JSON file into the editor."""
    from e2e.scenario import REFERENCE_SCENARIOS  # cheap, dependency-free

    # An upload event with empty/falsy contents must be a no-op: falling through
    # to the dropdown branch would silently clobber the editor with the reference
    # scenario even though the user only (e.g.) cleared the upload.
    if ctx.triggered_id == "upload-json":
        if not upload_contents:
            return no_update
        import base64
        try:
            header, b64 = upload_contents.split(",", 1)
            text = base64.b64decode(b64).decode("utf-8")
            # Load the uploaded JSON verbatim into the editor; the _render_scenario
            # callback (the editor's sole consumer) validates parseability and shows
            # any error. Parse errors aren't surfaced here because this callback's
            # only Output is the editor value, not a status field.
            return text
        except Exception:
            return no_update

    if ref_name in REFERENCE_SCENARIOS:
        return REFERENCE_SCENARIOS[ref_name]().to_json()
    return no_update


@app.callback(
    Output("scenario-map", "figure"),
    Output("scenario-summary", "children"),
    Input("render-button", "n_clicks"),
    Input("scenario-json", "value"),
    prevent_initial_call=False,
)
def _render_scenario(_clicks, json_text):
    """Preview the scenario layout + summary from the JSON editor contents."""
    sc, err = scenario_from_json_safe(json_text or "")
    if sc is None:
        return placeholder_figure(f"Invalid scenario JSON:\n{err}"), \
            html.Span(err, style={"color": "#eb3b5a"})
    return map_figure(sc), summarize(sc)


@app.callback(
    Output("validation-output", "children"),
    Input("validate-button", "n_clicks"),
    State("scenario-json", "value"),
    prevent_initial_call=True,
)
def _validate_scenario(_clicks, json_text):
    """Run Scenario.validate() and list any problems."""
    sc, err = scenario_from_json_safe(json_text or "")
    if sc is None:
        return html.Span(f"Cannot parse JSON: {err}", style={"color": "#eb3b5a"})
    problems = sc.validate()
    if not problems:
        return html.Span("Valid — no problems found.", style={"color": "#20bf6b"})
    return html.Ul([html.Li(p, style={"color": "#eb3b5a"}) for p in problems])


@app.callback(
    Output("scenario-download", "data"),
    Input("download-button", "n_clicks"),
    State("scenario-json", "value"),
    prevent_initial_call=True,
)
def _download_scenario(_clicks, json_text):
    """Offer the current (normalized) scenario JSON as a download."""
    sc, err = scenario_from_json_safe(json_text or "")
    if sc is None:
        return no_update
    fname = f"{sc.name or 'scenario'}.json"
    return dict(content=sc.to_json(), filename=fname)


@app.callback(
    Output("generate-output", "children"),
    Input("generate-button", "n_clicks"),
    State("scenario-json", "value"),
    prevent_initial_call=True,
)
def _generate_frames(_clicks, json_text):
    """
    Invoke the offline scenario_runner CLI in --dry-run mode as a subprocess.

    The teammate's module is `e2e.environment.scenario_runner` with a
    `--scenario <name|path> --dry-run` interface. We write the editor JSON to a
    temp file and pass that path. Everything is guarded; if the module doesn't
    exist yet we degrade with a clear message instead of crashing.
    """
    sc, err = scenario_from_json_safe(json_text or "")
    if sc is None:
        return f"Cannot generate: invalid scenario JSON.\n{err}"

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                         encoding="utf-8") as fh:
            fh.write(sc.to_json())
            tmp_path = fh.name

        cmd = [sys.executable, "-m", "e2e.environment.scenario_runner",
               "--scenario", tmp_path, "--dry-run"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=120)
        except FileNotFoundError:
            return ("Could not launch the Python interpreter to run "
                    "scenario_runner. (subprocess FileNotFoundError)")
        except subprocess.TimeoutExpired:
            return "scenario_runner --dry-run timed out after 120s."

        out = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0 and "No module named" in out:
            return ("scenario_runner is not available yet "
                    "(e2e.environment.scenario_runner not found). "
                    "A teammate is still writing it. Once present, this button "
                    "will run:\n  " + " ".join(cmd) + "\n\n--- output ---\n" + out)
        header = f"$ {' '.join(cmd)}\n(exit code {proc.returncode})\n\n"
        return header + (out if out.strip() else "(no output)")
    except Exception as e:  # never crash the server on this action
        return f"Generate frames failed: {type(e).__name__}: {e}"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def main():
    """Launch the development server on 127.0.0.1:8050 (debug off)."""
    app.run(host=HOST, port=PORT, debug=False)


if __name__ == "__main__":
    main()
