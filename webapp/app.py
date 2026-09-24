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
from typing import Any, Dict, List

import numpy as np
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
from webapp.demo_presets import PRESETS, PRESETS_BY_ID, DemoPreset, PresetError, apply_preset
from webapp.pipeline_registry import BLOCKS_BY_ID, MAX_N_STEPS, PRODUCT_IDS, default_block_state
from webapp.pipeline_runner import (
    PipelineError,
    figures_from_outputs,
    placeholder_figure,
    prewarm_tessera_interconnect,
    run_pipeline,
    scenario_topdown_figure,
)
from webapp.scenario_editor import map_figure, scenario_from_json_safe, summarize

HOST = "127.0.0.1"
PORT = 8050

app = Dash(__name__, suppress_callback_exceptions=True, title="E2E Array Simulator")
server = app.server  # exposed for gunicorn/WSGI if ever needed

# Result figures are stored as Plotly figure dicts in this Store between runs.
EMPTY_RESULTS: Dict[str, Any] = {}

# Cancel flags + in-progress locks, keyed by PER-TAB session id (see "session-id-store"
# below). Dash's threaded dev server lets the Cancel callback execute while the Run
# callback is still inside run_pipeline; before 2026-09-23 there was ONE flag,
# process-wide, so a second browser tab's Cancel button silently truncated an
# unrelated tab's run ("Cancelled after 5 of 20 frames" with no hint why -- found in a
# hands-on browser bug hunt, reproduced deterministically). A missing/falsy session id
# (a unit test calling a callback directly, or a client whose store has not yet
# seeded) shares one fallback key -- the pre-existing global behaviour. Under a
# multi-worker WSGI deployment neither dict would be shared across workers; that
# deployment does not exist and would need a background-callback manager anyway.
import threading  # noqa: E402  (deliberately next to the flags/locks it exists for)
_SESSION_LOCK = threading.Lock()  # guards both dicts below
_CANCEL_FLAGS: Dict[str, "threading.Event"] = {}
_RUN_LOCKS: Dict[str, "threading.Lock"] = {}


def _session_key(session_id) -> str:
    return session_id or "_default"


def _cancel_event(session_id) -> "threading.Event":
    """This session's Cancel flag, created on first use."""
    key = _session_key(session_id)
    with _SESSION_LOCK:
        ev = _CANCEL_FLAGS.get(key)
        if ev is None:
            ev = _CANCEL_FLAGS[key] = threading.Event()
        return ev


def _run_lock(session_id) -> "threading.Lock":
    """This session's reentrancy guard (a rapid double-click on Run, or two Run
    clicks before the button's client-side `disabled` state takes effect, must not
    dispatch two overlapping pipeline runs -- found in the same bug hunt)."""
    key = _session_key(session_id)
    with _SESSION_LOCK:
        lock = _RUN_LOCKS.get(key)
        if lock is None:
            lock = _RUN_LOCKS[key] = threading.Lock()
        return lock


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
        # A per-BROWSER-TAB id (sessionStorage, not shared across tabs like
        # localStorage or a cookie session would be), seeded once by the clientside
        # callback below. Scopes Cancel and the double-click guard to the tab that
        # actually clicked Run.
        dcc.Store(id="session-id-store", storage_type="session", data=None),
        # The RAW text of "Frames to run" at the moment Run was clicked, captured
        # client-side (see the clientside callback below): the browser's own number
        # input reports None for BOTH a blank field and an out-of-range one (0, 51,
        # ...), which read as the identical "blank or outside 1..50" message either
        # way (bug hunt, 2026-09-23) -- the raw text lets the refusal say which.
        dcc.Store(id="run-nsteps-raw", data=None),

        dcc.Tabs(id="tabs", value="tab-blocks", children=[
            dcc.Tab(label="Block Diagram", value="tab-blocks",
                    children=html.Div(block_diagram.layout(), style={"padding": "12px"})),
            dcc.Tab(label="Scenario", value="tab-scenario",
                    children=html.Div(scenario_editor.layout(), style={"padding": "12px"})),
            dcc.Tab(label="Results", value="tab-results",
                    children=html.Div(id="results-tab-content", style={"padding": "12px"})),
        ]),
    # 1600px, not the original 1280px: on the 1920x1080 conference monitor the narrower
    # cap wasted ~338px of margin per side and bought the lone-figure Thrust 3 screen
    # nothing from the bigger display (coordinator finding, 2026-09-23). Figures scale
    # with their container; the podium-distance font floor (pipeline_runner._make_legible)
    # is independent of this and unaffected.
    ], style={"maxWidth": "1600px", "margin": "0 auto", "fontFamily": "Segoe UI, Arial, sans-serif",
              "padding": "12px"})


app.layout = _app_layout


@app.callback(
    Output("session-id-store", "data"),
    Input("session-id-store", "data"),
)
def _ensure_session_id(existing):
    """Seed this tab's session id once (self-seeding dcc.Store idiom: Input and
    Output share a prop, so the callback fires once more after writing a value and
    then stops, since the second call sees `existing` already set)."""
    if existing:
        return no_update
    import uuid
    return str(uuid.uuid4())


# Captures the Frames-to-run field's RAW text at the moment Run is clicked, via the
# DOM directly rather than Dash's own number-input coercion (which reports None for
# both a blank field and an out-of-range one -- see "run-nsteps-raw" above).
app.clientside_callback(
    """
    function(n_clicks) {
        var el = document.getElementById('run-nsteps');
        return el ? el.value : null;
    }
    """,
    Output("run-nsteps-raw", "data"),
    Input("run-button", "n_clicks"),
    prevent_initial_call=True,
)


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
def _enabled_source_is_rt(block_state: Dict[str, Any]) -> bool:
    """True when the live ray-tracing source is the one feeding the run, i.e. the
    Scenario editor's JSON describes the frames on screen."""
    return bool((block_state or {}).get("rt_environment", {}).get("enabled"))


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
                return _with_param(block_state, bid, pkey, val)
    return block_state


def _with_param(block_state, bid, pkey, val):
    """The store after one param edit. A number input reports ``None`` for an empty
    or out-of-step field (a browser stepMismatch fires on blur even when nothing was
    typed); writing that null through let the runner fall back to the registry default
    while the field still displayed the preset's value (rehearsal 2026-09-22). Keep
    the last valid value instead."""
    if val is None:
        return no_update
    block_state.setdefault(bid, {}).setdefault("params", {})[pkey] = val
    return block_state


def _matching_preset(block_state: Dict[str, Any], *, require_ab: bool = False):
    """The loaded preset, IF `block_state` is exactly its as-loaded ("arm a") state --
    i.e. the operator has not hand-edited anything since Load preset. A manual edit
    changes `block_state` away from `apply_preset(p)`, so this returns None (used both
    to find an A/B pairing and, more generally, to attach a preset's `screen_note` to
    whatever is on the Results tab)."""
    for p in PRESETS:
        if require_ab and p.ab is None:
            continue
        try:
            if apply_preset(p) == block_state:
                return p
        except PresetError:
            continue
    return None


def _matching_ab_preset(block_state: Dict[str, Any]):
    """The loaded preset, IF it defines an A/B comparison (`DemoPreset.ab`, Change 1).
    A manual edit changes `block_state` away from `apply_preset(p)`, so this returns
    None and Run falls back to the ordinary single-run path (the "turn one knob and
    run again" flow every card also documents keeps working, ab-enabled preset or
    not)."""
    return _matching_preset(block_state, require_ab=True)


def _read_corpus_v_max(block_state: Dict[str, Any]):
    """The unambiguous Doppler velocity (+-v_max, m/s) of the Corpus Replay manifest
    `block_state` points at, read fresh at render time -- never typed into a preset,
    per the hostile-expert finding that a stored number drifts while the manifest does
    not. Returns None (never raises) when there is no manifest, the file cannot be
    read, or its config does not parse, so a screen note can drop the clause instead
    of showing a stale or crashed one."""
    manifest = ((block_state or {}).get("corpus_environment", {})
                .get("params", {}).get("manifest"))
    if not manifest:
        return None
    try:
        import json
        from pathlib import Path

        from e2e.radar_config import RadarConfig  # dependency-free, stdlib only
        from webapp.corpus_catalog import REPO_ROOT
        path = Path(manifest)
        if not path.is_absolute():
            path = REPO_ROOT / manifest
        manifest_dict = json.loads(path.read_text(encoding="utf-8"))
        cfg = RadarConfig.from_dict(manifest_dict["config"])
        return cfg.max_velocity_mps
    except Exception:
        return None


def _resolve_screen_note(preset: "DemoPreset", block_state: Dict[str, Any]) -> str:
    """`preset.screen_note` with its "{VMAX_CLAUSE}" token (if any) filled in from the
    corpus manifest, or dropped if the manifest cannot be read (see
    `_read_corpus_v_max`) -- so the sentence still reads cleanly with the clause gone."""
    note = preset.screen_note if preset is not None else ""
    if note and "{VMAX_CLAUSE}" in note:
        v_max = _read_corpus_v_max(block_state)
        # Kept short (hostile-expert fourth read, 2026-09-23): the Thrust 5 notes grew
        # a mandatory "frames: ..." prefix, and this clause has to leave room for it
        # -- and, on thrust5_detector_ml, a further per-preset clause -- on one line
        # at 16 px on the 1600 px results page.
        clause = f"; v_max ±{v_max:.2f} m/s" if v_max is not None else ""
        note = note.replace("{VMAX_CLAUSE}", clause)
    return note


def _ab_arm_line(preset: "DemoPreset", arm: str) -> str:
    """'A (as loaded): <label> <value> -- before' or 'B: <label> <value> -- after',
    naming the knob the way the editor labels it. ONE arm's value only -- before
    2026-09-23 both panels' banners printed BOTH arms' values with identical text, so
    a visitor reading a single panel could not tell which arm was on screen (defect
    found reading the rendered Results tab)."""
    bid, key, _value_b = preset.ab
    label = next((ps.label for ps in BLOCKS_BY_ID[bid].params if ps.key == key), key)
    if arm == "a":
        return f"A (as loaded): {label} {preset.ab_label_a or '?'} -- before"
    return f"B: {label} {preset.ab_label_b or '?'} -- after"


#: Results-tab-only truncation limit for one run note (wave 9 follow-up,
#: 2026-09-24): the Block Diagram status line (`_note_for` below) still carries the
#: full, untruncated text -- this is purely a Results-tab display shortening, so it
#: lives next to `_notes_line`, the function it gates, not next to `run_pipeline`
#: (which builds the notes themselves and knows nothing about either screen).
_NOTE_TRUNCATE_CHARS = 160


def _truncate_note(note: str) -> str:
    """One run note, cut at its first `" -- "` separator or `_NOTE_TRUNCATE_CHARS`
    characters, whichever comes first (with an ellipsis) -- the Results-tab-only
    shortening item 6's follow-up fixes (wave 9, 2026-09-24): the live-chain
    correctness gate's own note runs its headline number, then a `" -- "`, then an
    attribution clause that can run another 2-3 lines ("...the usual answer: it is
    the knob the A/B turns...") -- illegible at 11 px and pushing the figures down
    on every Thrust 5 arm, when the banner above already carries the gate's verdict.
    A note with neither (short, no `" -- "`, e.g. the interconnect Tessera
    surrogate's own note) is returned unchanged -- checked against both arms'
    "evaluated at 14.25-15.75 GHz" text, which sits at the FRONT of that note and
    survives either cut."""
    cut = len(note)
    sep = note.find(" -- ")
    if 0 <= sep < cut:
        cut = sep
    if _NOTE_TRUNCATE_CHARS < cut:
        cut = _NOTE_TRUNCATE_CHARS
    if cut >= len(note):
        return note
    return note[:cut].rstrip() + " ..."


def _notes_line(axis_meta: Dict[str, Any]) -> List[str]:
    """`axis_meta['notes']` (`pipeline_runner.run_pipeline`'s `run_notes` -- e.g. the
    interconnect Tessera surrogate's scale-model/frequency disclosure,
    `InterconnectBlock.describe()`), one TRUNCATED (`_truncate_note`) entry per note,
    for the Results tab (item 6, wave 9 hostile-expert read, 2026-09-23 + follow-up,
    2026-09-24): these reached only the Block Diagram tab's status line (via
    `_note_for` below, which keeps every note's FULL text), so a Thrust 4 Results
    screen carried no on-screen record of the frequency it was actually evaluated
    at -- a visitor reading only that tab, or a photograph of it, never saw the
    disclosure at all. `[]` when there are none."""
    return [_truncate_note(n) for n in (axis_meta.get("notes") or [])]


def _notes_block(notes_lines: List[str]):
    """The Results-tab rendering of `_notes_line`'s return: ONE small, muted line
    PER note, not one paragraph joined by "|" (item 6 follow-up, 2026-09-24): the
    live-chain gate's own multi-clause note otherwise ran 3-5 lines of 11 px text on
    every Thrust 5 arm, pushing the figures down and illegible on stage, when the
    banner above it already carries the gate's headline verdict. Same muted style
    as `screen_note` in `_render_results`; module-level (not nested in that
    callback) so it is directly testable and reusable for both the current and the
    `_previous` (arm B / before-after) block."""
    return html.Div(
        [html.Div(n, style={"color": "#576574", "fontSize": "14px"}) for n in notes_lines],
        style={"marginBottom": "6px"},
    )


def _arm_result(n_clicks, outputs, n_steps, block_state, scenario_json, note: str,
                arm: str = "a"):
    """Figures + banner + status message for ONE run's outputs -- shared by the
    ordinary single-run path and each arm of an A/B run. `note` is the caller's
    already-computed status-line suffix (see `_run_pipeline`'s `_note_for`). Returns
    None when nothing rendered (cancelled before the first frame).

    `arm` ("a" or "b") is which A/B panel this is, if any (ignored on a single-run
    path). Item 7 (hostile-expert read, 2026-09-23): the offline-scored PR panel
    below is scored on the fixed beat_cfar.json split and is IDENTICAL on both arms
    by design -- an A/B knob never touches it -- which read as a bug (two panels,
    same numbers) until arm B's copy said so in its own subtitle."""
    figs = figures_from_outputs(outputs)
    axis_meta = outputs.get("_axis_meta") or {}
    if not figs and axis_meta.get("cancelled"):
        return None
    # Geometry FIRST, when the Scenario tab holds a parseable scene: a stripe in
    # sin(azimuth) is only interpretable next to the layout that produced it (see
    # pipeline_runner.scenario_topdown_figure). Best-effort by design -- the editor
    # may hold half-typed JSON, and a results tab must never be lost to a preview
    # panel, so an unparseable or unrenderable scenario just omits the panel.
    # Only when the RT Environment source ray-traced that scenario: the precomputed
    # .pkl frames and the corpus replay carry their own (unrelated) geometry, and a
    # plan view of whatever the editor happens to hold -- a lone radar triangle, on
    # every preset -- was the first card on screen in the 2026-09-22 rehearsal.
    scene_fig = None
    if scenario_json and _enabled_source_is_rt(block_state):
        sc, _err = scenario_from_json_safe(scenario_json)
        if sc is not None:
            try:
                scene_fig = scenario_topdown_figure(sc)
            except Exception:
                scene_fig = None
    if scene_fig is not None:
        figs = {"scene_topdown": scene_fig, **figs}
    # Thrust 5 preset: the offline-scored PR curve (e2e/ml/runs/beat_cfar.json) this
    # run's operating point sits on, next to the live scoreboard figures_from_outputs
    # already added -- only when the on-screen detector maps to one of its scored arms.
    if axis_meta.get("detector"):
        from webapp import detector_scoreboard
        try:
            arm_name = detector_scoreboard.arm_name_for_detector(axis_meta["detector"])
        except (FileNotFoundError, ValueError):
            arm_name = None
        if arm_name is not None:
            pr_fig = detector_scoreboard.stored_pr_figure(highlight_arm=arm_name)
            if arm == "b":
                # `stored_pr_figure` already reserves top margin for
                # `detector_scoreboard._PR_MIN_TITLE_LINES` (4) lines on EVERY call, so
                # appending this 4th line to arm B's title needs no further margin bump
                # here (item 1, wave 9 hostile-expert read, 2026-09-23: a flat "+25"
                # bump here undercounted the true per-line cost and this line's own
                # closing tick overprinted the plot's y-axis; worse, arm A never got
                # the bump at all, so the two arms' plots did not share an axis
                # height). Margin now comes from `stored_pr_figure` alone, identical on
                # both arms regardless of which one appends this sentence.
                pr_fig.update_layout(title=dict(
                    text=pr_fig.layout.title.text
                        + "<br><sup>scored offline; identical on both arms, the "
                          "knob cannot move it</sup>"))
            figs = {**figs, "detector_pr_stored": pr_fig}
    n_products = len(figs)
    banner = _run_banner(n_clicks, axis_meta, int(n_steps or 10))
    if axis_meta.get("cancelled"):
        # Partial results are still shown, labelled as partial.
        msg = html.Span(f"Cancelled after {axis_meta.get('n_steps_run', '?')} of "
                        f"{int(n_steps or 10)} frames: {n_products} product(s) from the "
                        f"frames that ran. See Results tab.{note}",
                        style={"color": "#f39c12"})
    else:
        msg = html.Span(f"Run complete: {n_products} product(s). See Results tab.{note}",
                        style={"color": "#20bf6b"})
    return {"figs": figs, "banner": banner, "msg": msg, "n_products": n_products,
            "cancelled": bool(axis_meta.get("cancelled"))}


def _describe_bad_frame_count(raw) -> str:
    """A precise reason for refusing a Frames-to-run value, using the RAW text typed
    into the field (see the clientside callback that populates "run-nsteps-raw").
    The browser's own number-input coercion reports None for BOTH a blank field and
    an out-of-range one (0, 51, ...), which used to read as the identical "blank or
    outside 1..50" message either way (bug hunt, 2026-09-23)."""
    text = str(raw if raw is not None else "").strip()
    if not text:
        return "the field is blank"
    try:
        val = float(text)
    except ValueError:
        return f"{text!r} is not a number"
    if val < 1:
        return f"{text} is below the minimum of 1"
    if val > MAX_N_STEPS:
        return f"{text} exceeds the maximum of {MAX_N_STEPS}"
    return f"{text} is not a whole number"


@app.callback(
    Output("results-store", "data"),
    Output("run-status", "children"),
    Output("tabs", "value"),
    Output("run-sink", "children"),
    Input("run-button", "n_clicks"),
    State("block-state-store", "data"),
    State("run-nsteps", "value"),
    State("scenario-json", "value"),
    State("results-store", "data"),
    State("run-nsteps-raw", "data"),
    State("session-id-store", "data"),
    prevent_initial_call=True,
    # dash>=2.9 supports `running` on plain (non-background) callbacks: the
    # renderer flips these properties synchronously around the request, so the
    # button disables and the dcc.Loading spinner around "run-sink" (its only
    # child is now an Output of this callback, which is what makes dcc.Loading
    # notice it's "loading" in the first place) actually engages for the ~10s a
    # run takes, instead of both being dead decoration.
    running=[(Output("run-button", "disabled"), True, False),
             (Output("cancel-button", "disabled"), False, True)],
)
def _run_pipeline(n_clicks, block_state, n_steps, scenario_json, prev_results=None,
                  nsteps_raw=None, session_id=None):
    """Run the pipeline (lazy heavy imports inside) and stash result figures.

    The store holds the figures under their product keys plus two reserved entries:
    ``_banner`` (what produced these figures: run number, time, source, frames, the
    detector's operating point) and ``_previous`` (the last run's figures and banner),
    so the Results tab can show a before/after -- every demo card says "run, turn one
    knob, run again", and until 2026-09-22 the comparison lived only in the audience's
    memory of a screen that had been replaced.

    A/B presets (Change 1, `DemoPreset.ab`): when `block_state` is exactly an ab-preset's
    as-loaded state, this runs the pipeline TWICE -- run A as loaded, run B with the
    preset's single override -- and reuses this same before/after mechanism, but with A
    ON TOP: A is the current-run payload (rendered first), B becomes ``_previous``
    (rendered second, below the divider). Each arm's ``_banner`` names ONLY its own
    value ("A (as loaded): ... -- before" / "B: ... -- after"), and both carry
    ``_ab=True`` so `_render_results` skips the generic "This run"/"Previous run"
    prefixes (defect found reading the rendered Results tab, 2026-09-23: both panels
    printed identical "A: .. | B: .." text). One click, two runs, no manual re-run.

    `session_id` scopes Cancel and the double-click guard to the browser tab that
    clicked Run (see `_cancel_event`/`_run_lock`): a second click while a run for the
    SAME session is still in flight is ignored rather than dispatching an overlapping
    run (bug hunt, 2026-09-23).
    """
    block_state = block_state or default_block_state()
    # Unique per-invocation value so "run-sink" always changes -- dcc.Loading only
    # needs this Output to belong to a pending callback to spin, but a changing
    # value also makes the sink's own purpose (a loading anchor) legible in devtools.
    sink = f"run #{n_clicks}"
    # The spinner reports None for a blank or out-of-range value (min=1, max=50), and
    # `int(n_steps or 10)` silently ran 10 frames for 0, -1, 500 and blank while the
    # field kept showing the typed value (operator-flow review, 2026-09-22). Refuse.
    if n_steps is None or int(n_steps) < 1:
        return no_update, html.Span(
            f"Frames to run must be a whole number from 1 to {MAX_N_STEPS}; "
            f"{_describe_bad_frame_count(nsteps_raw)}. Fix it and press Run again.",
            style={"color": "#eb3b5a"}), no_update, sink
    n_steps = int(n_steps)

    lock = _run_lock(session_id)
    if not lock.acquire(blocking=False):
        # A second Run click for the SAME session while one is still in flight (a
        # forced rapid double-click can beat the button's client-side `disabled`
        # state to the server) -- ignore it rather than dispatching an overlapping
        # run that would render two result panels.
        return no_update, html.Span(
            "A run is already in progress for this session; ignoring the extra click.",
            style={"color": "#f39c12"}), no_update, sink

    try:
        def _note_for(state_arm: Dict[str, Any], axis_meta: Dict[str, Any]) -> str:
            """The status-line suffix: a scale-mode assumption, or blocks a source
            could not apply -- computed per arm (A and B can differ, e.g. scale_mode)."""
            note = ""
            # Physical scale mode trusts the frames to BE volts; nothing in a bare
            # .pkl can verify that (no metadata until the frames-carry-metadata
            # refactor), so surface the assumption instead of silently producing
            # clipped/underdriven nonsense when a legacy unit-energy pkl (e.g. stock
            # munich frames) is fed in.
            scale_mode = state_arm.get("rffe", {}).get("params", {}).get("scale_mode")
            if scale_mode == "physical":
                note = ("  [physical scale mode (forced): assumes frames were "
                        "generated with tx_power_dbm set -- stock munich/etoile "
                        "pkls are legacy-normalized]")
            elif scale_mode in (None, "auto") and axis_meta.get("from_meta"):
                # In auto mode the frames declare their own convention (v2
                # metadata), so no assumption warning is needed -- but say what was
                # detected.
                note = "  [auto scale mode: following the frames' own metadata]"
            # Run notes (blocks a source could not apply, a checkpoint without a
            # provenance stamp, ...) belong next to the result, not in a server log
            # nobody reads on stage.
            if axis_meta.get("notes"):
                note += "  [" + " | ".join(axis_meta["notes"]) + "]"
            return note

        ab_preset = _matching_ab_preset(block_state)
        state_b = apply_preset(ab_preset, arm="b") if ab_preset is not None else None
        # The loaded preset's screen note (if any), shown ONCE at the top of the
        # Results tab regardless of whether this is a single run or an A/B pair --
        # `block_state` is always arm A / the as-loaded state, so this only needs
        # computing once per click.
        matched_preset = ab_preset or _matching_preset(block_state)
        screen_note = (_resolve_screen_note(matched_preset, block_state)
                      if matched_preset is not None else "")
        cancel = _cancel_event(session_id)
        try:
            cancel.clear()
            outputs_a = run_pipeline(block_state, n_steps=n_steps, should_stop=cancel.is_set)
            outputs_b = None
            if state_b is not None and not (outputs_a.get("_axis_meta") or {}).get("cancelled"):
                # The frame ceiling (MAX_N_STEPS/MAX_PRESET_N_STEPS) is enforced inside
                # run_pipeline itself, so it applies to THIS call independently of run A.
                outputs_b = run_pipeline(state_b, n_steps=n_steps, should_stop=cancel.is_set)
        except PipelineError as e:
            # Friendly, expected failure: stay on the diagram, show the message -- and
            # relabel the figures still on the Results tab so they are not read as this run.
            return (_stale_after_failure(prev_results, n_clicks, str(e)),
                    html.Span(str(e), style={"color": "#eb3b5a"}), no_update, sink)
        except Exception as e:  # unexpected — still don't crash the server
            return (_stale_after_failure(prev_results, n_clicks, str(e)),
                    html.Span(f"Unexpected error: {e}", style={"color": "#eb3b5a"}),
                    no_update, sink)

        result_a = _arm_result(n_clicks, outputs_a, n_steps, block_state, scenario_json,
                               _note_for(block_state, outputs_a.get("_axis_meta") or {}))
        if result_a is None:
            # Cancelled before the first frame finished: nothing ran. Stay on the diagram
            # rather than send the presenter to a Results tab holding only a banner.
            return no_update, html.Span(
                "Cancelled before the first frame finished: nothing ran, nothing to show. "
                "The Results tab is unchanged.", style={"color": "#f39c12"}), no_update, sink

        if state_b is not None:
            # Arm A renders ON TOP (the as-loaded baseline, "before") and arm B below
            # ("after") -- both used to carry identical "A: .. | B: .." text on both
            # panels, so a visitor reading one panel could not tell which arm was on
            # screen (defect found reading the rendered Results tab, 2026-09-23). A is
            # the top-level store payload (rendered first); B is nested under
            # "_previous" (rendered second, below the divider) -- reusing the existing
            # before/after render mechanism, but each banner now names ONLY its own arm.
            line_a = _ab_arm_line(ab_preset, "a")
            result_b = (_arm_result(n_clicks, outputs_b, n_steps, state_b, scenario_json,
                                    _note_for(state_b, outputs_b.get("_axis_meta") or {}),
                                    arm="b")
                       if outputs_b is not None else None)
            if result_b is None:
                # Cancelled between A and B (or before B's first frame): show A alone,
                # exactly like an ordinary single run -- Cancel still leaves something.
                data_a = {k: f.to_dict() for k, f in result_a["figs"].items()}
                data_a["_banner"] = f"{line_a} -- B did not run (cancelled)  ||  {result_a['banner']}"
                data_a["_ab"] = True
                notes_a = _notes_line(outputs_a.get("_axis_meta") or {})
                if notes_a:
                    data_a["_notes"] = notes_a
                if screen_note:
                    data_a["_screen_note"] = screen_note
                if prev_results:
                    data_a["_previous"] = {k: v for k, v in prev_results.items() if k != "_previous"}
                # Keep the frame count in the status line: the Cancel journey (and the
                # presenter) need "how far did it get", not only "B did not run".
                meta_a = outputs_a.get("_axis_meta") or {}
                ran_a = meta_a.get("n_steps_run", "?")
                return data_a, html.Span(
                    f"Cancelled after {ran_a} of {n_steps} frames of run A; run B did not "
                    f"start: showing run A only. See Results tab.",
                    style={"color": "#f39c12"}), "tab-results", sink
            line_b = _ab_arm_line(ab_preset, "b")
            data_a = {k: f.to_dict() for k, f in result_a["figs"].items()}
            data_b = {k: f.to_dict() for k, f in result_b["figs"].items()}
            data_a["_banner"] = f"{line_a}  ||  {result_a['banner']}"
            data_a["_ab"] = True
            data_b["_banner"] = f"{line_b}  ||  {result_b['banner']}"
            data_b["_ab"] = True
            # Item 6 (wave 9 hostile-expert read, 2026-09-23): run notes belong on the
            # Results tab, per arm (they can differ, e.g. arm B turning the interconnect
            # to a different scale/frequency) -- see `_notes_line` and `_render_results`.
            notes_a = _notes_line(outputs_a.get("_axis_meta") or {})
            if notes_a:
                data_a["_notes"] = notes_a
            notes_b = _notes_line(outputs_b.get("_axis_meta") or {})
            if notes_b:
                data_b["_notes"] = notes_b
            if screen_note:
                # Shown ONCE, above both A and B -- on `data_a` (the top-level payload),
                # never duplicated onto `data_b`/`_previous`.
                data_a["_screen_note"] = screen_note
            data_a["_previous"] = data_b
            if result_b["cancelled"]:
                msg = html.Span(
                    f"A complete, B cancelled after "
                    f"{(outputs_b.get('_axis_meta') or {}).get('n_steps_run', '?')} of "
                    f"{n_steps} frames. See Results tab (partial B).",
                    style={"color": "#f39c12"})
            else:
                # The per-arm run notes belong here too (Change, 2026-09-23): they used
                # to be dropped on the A/B path, which is now EVERY Thrust 5 run -- and
                # the live chain's correctness gate ("max |diff| = N ADC codes") is a
                # run note. A number that only exists when nobody looks is not a gate.
                msg = html.Span(
                    f"A/B run complete ({ab_preset.ab_label_a} vs {ab_preset.ab_label_b}): "
                    f"{result_a['n_products']} / {result_b['n_products']} product(s). "
                    "See Results tab."
                    + _note_for(block_state, outputs_a.get("_axis_meta") or {})
                    + _note_for(state_b, outputs_b.get("_axis_meta") or {}),
                    style={"color": "#20bf6b"})
            return data_a, msg, "tab-results", sink

        # Ordinary single-run path: unchanged behaviour.
        data = {k: f.to_dict() for k, f in result_a["figs"].items()}
        data["_banner"] = result_a["banner"]
        notes = _notes_line(outputs_a.get("_axis_meta") or {})
        if notes:
            data["_notes"] = notes
        if screen_note:
            data["_screen_note"] = screen_note
        if prev_results:
            data["_previous"] = {k: v for k, v in prev_results.items() if k != "_previous"}
        return data, result_a["msg"], "tab-results", sink
    finally:
        # Not cleaned up from _RUN_LOCKS/_CANCEL_FLAGS: popping here would race a
        # concurrent `_run_lock`/`_cancel_event` lookup for the SAME session (it
        # could create a second Lock/Event object right as this one is removed,
        # letting a genuinely overlapping run slip past the guard). The dicts grow
        # by one entry per distinct browser tab that has ever clicked Run or
        # Cancel -- trivial for a demo session.
        lock.release()


def _stale_after_failure(prev_results, n_clicks, error: str):
    """After a failed run the Results tab still holds the last run's figures; relabel
    them so the banner does not assert currency for a run that did not happen."""
    if not prev_results:
        return no_update
    data = dict(prev_results)
    old = data.get("_banner") or "unlabelled"
    if not old.startswith("NOT this run"):
        data["_banner"] = f"NOT this run -- run #{n_clicks} failed ({error[:80]}). Still showing: {old}"
    return data


def _run_banner(n_clicks, axis_meta, n_steps: int) -> str:
    """One line saying what produced the figures on screen (see _run_pipeline)."""
    import time as _time
    n_run = axis_meta.get("n_steps_run", n_steps)
    frames = f"{n_run} of {n_steps} frames"
    if axis_meta.get("cancelled"):
        frames += " -- CANCELLED, partial"
    parts = [f"run #{n_clicks}", _time.strftime("%H:%M:%S"),
             axis_meta.get("source") or "", frames]
    # The live-chain correctness gate (pipeline_runner._StoredADCGateBlock), on the
    # Results tab rather than only in the status line: a visitor photographs this
    # banner, and whether the cube on screen is the corpus's own is part of what the
    # picture has to say. Absent on every other path, which leaves those banners
    # byte-identical to before.
    if axis_meta.get("gate"):
        parts.append(axis_meta["gate"])
    det = axis_meta.get("detector") or {}
    if det:
        parts.append(f"detector: {det.get('label', '?')}, detections at objectness >= "
                     f"{float(det.get('threshold', 0.0)):.2f}")
    return "  |  ".join(p for p in parts if p)


# =================================================================================
# Demo presets + Cancel
# =================================================================================

@app.callback(
    Output("block-state-store", "data", allow_duplicate=True),
    Output("run-nsteps", "value"),
    Output("preset-notes", "children"),
    Output("block-param-editor", "children", allow_duplicate=True),
    Output("run-status", "children", allow_duplicate=True),
    Output("results-store", "data", allow_duplicate=True),
    Input("preset-load", "n_clicks"),
    State("preset-select", "value"),
    State("block-cytoscape", "tapNodeData"),
    prevent_initial_call=True,
)
def _load_preset(n_clicks, preset_id, node_data):
    """Replace the block state and frame count with a demo preset's, and show its
    operator card. The param editor is re-rendered for the currently selected block so
    the spinners on screen show the preset's values, not the ones just replaced."""
    preset = PRESETS_BY_ID.get(preset_id or "")
    if preset is None:
        return (no_update, no_update,
                html.Span(f"Unknown preset {preset_id!r}", style={"color": "#eb3b5a"}),
                no_update, no_update, no_update)
    try:
        state = apply_preset(preset)
    except PresetError as e:
        # A preset that no longer fits the registry is a bug in the preset; say so
        # rather than loading half of it.
        return (no_update, no_update, html.Span(str(e), style={"color": "#eb3b5a"}),
                no_update, no_update, no_update)
    # Pay any Tessera surrogate cold-start cost NOW (preset load), not when Run is
    # pressed in front of an audience (build item 5). Both A/B arms, since either one
    # may be Run first; best-effort -- see `prewarm_tessera_interconnect`.
    prewarm_tessera_interconnect(state)
    if preset.ab is not None:
        try:
            prewarm_tessera_interconnect(apply_preset(preset, arm="b"))
        except PresetError:
            pass
    # Open the editor on the block whose knob the card says to turn, so the operator
    # is one click from the live demo; fall back to the tapped node.
    if preset.live_knobs:
        block_id = preset.live_knobs[0][0]
    else:
        block_id = (node_data or {}).get("id", PRODUCT_IDS[0])
    # The Results tab is cleared: its figures came from another preset, and the
    # before/after section would otherwise pair a Thrust 5 run with a Thrust 2 one.
    return (state, preset.n_steps, block_diagram.preset_notes(preset),
            block_diagram.param_editor(block_id, state),
            html.Span(f"Preset loaded: {preset.label}. Press Run pipeline.",
                      style={"color": "#3867d6"}),
            None)


@app.callback(
    Output("run-status", "children", allow_duplicate=True),
    Input("cancel-button", "n_clicks"),
    State("session-id-store", "data"),
    prevent_initial_call=True,
)
def _cancel_run(n_clicks, session_id=None):
    """Ask the run in progress to stop after the frame it is on. Scoped to THIS
    session (see `_cancel_event`) -- before 2026-09-23 one shared flag let a second
    browser tab's Cancel truncate an unrelated tab's run."""
    _cancel_event(session_id).set()
    return html.Span("Cancelling after the current frame...", style={"color": "#f39c12"})


# =================================================================================
# Results tab
# =================================================================================

def _decode_plotly_array(v) -> list:
    """A heatmap trace's x/y/z, after `go.Figure.to_dict()`, is EITHER a plain list OR
    plotly's compact typed-array encoding (`{"dtype": ..., "bdata": <base64>}` -- used
    for large numpy arrays since plotly 5.20+; every range_az/range_el/radar_cube axis
    here is a numpy array). Decode either into a flat list of floats. Found live in the
    2026-09-22 rehearsal: `_share_y_ranges` crashed reading 'dtype' as a coordinate
    because it assumed the plain-list form. Used only for axis bookkeeping -- the
    stored trace dict itself (what actually renders) is untouched."""
    if v is None:
        return []
    if isinstance(v, dict) and "bdata" in v:
        import base64
        raw = base64.b64decode(v["bdata"])
        return np.frombuffer(raw, dtype=np.dtype(v.get("dtype", "f8"))).tolist()
    return list(v)


def _union_fixed_range(pair, axis: str):
    """`[lo, hi]` spanning whatever explicit `layout.<axis>.range` the pair already
    carries, or None when neither figure fixed one. Shared by the heatmap and scatter
    branches of :func:`_share_y_ranges`: an axis a figure deliberately set (a display
    crop, a dB floor) is a claim about what should be on screen, and sharing must widen
    it to cover both arms rather than replace it with the raw data extent."""
    fixed = [((fig.get("layout") or {}).get(axis) or {}).get("range") for fig in pair]
    fixed = [r for r in fixed if r and len(r) == 2 and None not in r]
    if not fixed:
        return None
    return [min(float(r[0]) for r in fixed), max(float(r[1]) for r in fixed)]


def _share_y_ranges(figs, prev_figs) -> None:
    """Give a figure present in both runs of a before/after pair one set of axes, so
    the pair reads as a difference in the DATA, not two independently autoscaled
    panels (Thrust 2: 0.06 vs 0.63 drawn the same size, review 2026-09-22; A/B
    heatmaps with mismatched axes/colour scale, Change 1 review 2026-09-22).

    Scatter traces share one y-range (0 to the larger of the two curves' own peak,
    or a floor either figure already set -- see pipeline_runner's subspace-error
    minimum y-axis bound, which this must not shrink back down). Heatmap traces
    additionally share one x-range, y-range AND zmin/zmax, so an A/B pair's colour
    scale and axis extent cannot silently differ even if one arm's binning did."""
    for key in set(figs) & set(prev_figs):
        pair = (figs[key], prev_figs[key])
        datas = [fig.get("data") or [] for fig in pair]
        if not all(datas):
            continue
        kinds = [d[0].get("type", "scatter") for d in datas]
        if kinds[0] != kinds[1]:
            continue
        if kinds[0] == "heatmap":
            xs, ys, zmins, zmaxs = [], [], [], []
            for data in datas:
                for tr in data:
                    if tr.get("type") != "heatmap":
                        continue
                    xs.extend(v for v in _decode_plotly_array(tr.get("x")) if v is not None)
                    ys.extend(v for v in _decode_plotly_array(tr.get("y")) if v is not None)
                    if tr.get("zmin") is not None:
                        zmins.append(float(tr["zmin"]))
                    if tr.get("zmax") is not None:
                        zmaxs.append(float(tr["zmax"]))
            if xs and ys:
                xr, yr = [min(xs), max(xs)], [min(ys), max(ys)]
                # ...unless a figure already FIXED that axis, in which case union the
                # fixed ranges instead of recomputing from the data -- the same rule
                # the scatter branch below has carried since 2026-09-23, applied here
                # after the live-chain A/B pairs shipped: the detector's objectness
                # panel is deliberately cropped to 50 m (the labels and the offline
                # scoring both stop at 40 m -- pipeline_runner reads that crop from
                # beat_cfar.json), and recomputing from the cube's own 0-102 m extent
                # silently undid the crop the moment those screens gained an A/B arm
                # (found in the rehearsal PNGs, 2026-09-23). The range-Doppler panel
                # fixes nothing and still shares the data extent.
                xr = _union_fixed_range(pair, "xaxis") or xr
                yr = _union_fixed_range(pair, "yaxis") or yr
                for fig in pair:
                    fig.setdefault("layout", {}).setdefault("xaxis", {})["range"] = xr
                    fig.setdefault("layout", {}).setdefault("yaxis", {})["range"] = yr
            if zmins and zmaxs:
                zr = (min(zmins), max(zmaxs))
                for fig in pair:
                    for tr in fig.get("data") or []:
                        if tr.get("type") == "heatmap":
                            tr["zmin"], tr["zmax"] = zr
        else:
            if kinds[0] != "scatter":
                continue
            existing = [((fig.get("layout") or {}).get("yaxis") or {}).get("range")
                       for fig in pair]
            if any(existing):
                # Respect a range either figure ALREADY fixed explicitly (range_profile's
                # constant -60..2 dB display floor, subspace_err's own minimum-bound
                # range) -- union them instead of recomputing tozero from the raw data,
                # which clobbered range_profile's negative-dB scale into [0, 1] (found in
                # rehearsal, 2026-09-23: the A/B before/after pair went blank because the
                # error-curve convention below assumes non-negative, zero-anchored data,
                # which is true for subspace_err and false for a dB-scale line plot).
                lo = min(r[0] for r in existing if r)
                hi = max(r[1] for r in existing if r)
                for fig in pair:
                    fig.setdefault("layout", {}).setdefault("yaxis", {})["range"] = [lo, hi]
                continue
            # No figure set its own range (e.g. subspace_err before Change 3, or any
            # other non-negative error-like curve): fall back to a shared, zero-anchored
            # range computed from the data, exactly as before.
            ys = []
            for data in datas:
                for tr in data:
                    ys.extend(v for v in _decode_plotly_array(tr.get("y")) if v is not None)
            if not ys:
                continue
            top = max(ys) * 1.05 if max(ys) > 0 else 1.0
            for fig in pair:
                fig.setdefault("layout", {}).setdefault("yaxis", {})["range"] = [0.0, top]


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

    # Card order follows results_data's insertion order, which mirrors
    # figures_from_outputs' build order (webapp/pipeline_runner.py): fft ->
    # range_az -> range_el -> range_profile -> subspace_err -> comms products.
    # That already keeps Range Profile grouped with its FFT/range siblings, so no
    # re-sort is needed here; each figure carries its own title (set where it is
    # built) rather than a second, easily-stale title map duplicated in this tab.
    def _grid(figs):
        cards = []
        # A lone figure (Thrust 1) takes the whole row instead of half a screen.
        basis = "1 1 100%" if len(figs) == 1 else "1 1 45%"
        for key, fig_dict in figs.items():
            cards.append(html.Div(
                # No modebar: the zoom/export toolbar overlaps each card's title at
                # this card width, and none of its tools matter for read-only results.
                dcc.Graph(figure=fig_dict, config={"displayModeBar": False}),
                style={"flex": basis, "minWidth": "420px", "margin": "6px",
                       "border": "1px solid #dfe4ea", "borderRadius": "6px",
                       "padding": "4px"},
            ))
        return html.Div(cards, style={"display": "flex", "flexWrap": "wrap"})

    figs = {k: v for k, v in results_data.items() if not k.startswith("_")}
    prev = results_data.get("_previous") or {}
    prev_figs = {k: v for k, v in prev.items() if not k.startswith("_")}
    _share_y_ranges(figs, prev_figs)
    children = [html.H3("Results")]
    banner = results_data.get("_banner")
    if banner:
        # An A/B run's banner already names its own arm in full ("A (as loaded): ..
        # -- before" / "B: .. -- after", see `_ab_arm_line`); the generic "This run"/
        # "Previous run" prefix stays for the single-run and manual before/after
        # paths, where the banner does not name an arm.
        prefix = "" if results_data.get("_ab") else "This run: "
        children.append(html.Div(f"{prefix}{banner}",
                                 style={"color": "#2d3a4a", "fontWeight": "bold",
                                        "marginBottom": "4px"}))
    notes_lines = results_data.get("_notes")
    if notes_lines:
        children.append(_notes_block(notes_lines))
    screen_note = results_data.get("_screen_note")
    if screen_note:
        # The preset's own caveat, for whoever photographs this tab rather than hears
        # the presenter (hostile-expert third read, 2026-09-23): one legible, muted
        # line shown ONCE, shared by both A/B panels below it.
        children.append(html.Div(screen_note,
                                 style={"color": "#576574", "fontSize": "16px",
                                        "marginBottom": "8px"}))
    children.append(_grid(figs))
    if prev_figs:
        # The before/after every card asks for: the previous run stays on screen
        # under its own banner, so "turn one knob and run again" is a comparison
        # the audience can see rather than remember.
        children.append(html.Hr())
        prev_prefix = "" if prev.get("_ab") else "Previous run (for before/after): "
        children.append(html.Div(
            f"{prev_prefix}{prev.get('_banner') or 'unlabelled'}",
            style={"color": "#576574", "fontWeight": "bold", "marginTop": "6px",
                   "marginBottom": "4px"}))
        prev_notes_lines = prev.get("_notes")
        if prev_notes_lines:
            children.append(_notes_block(prev_notes_lines))
        children.append(_grid(prev_figs))
    return html.Div(children)


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
