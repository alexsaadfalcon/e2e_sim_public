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

import copy
import os
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, List

import numpy as np
from dash import (
    ALL,
    ClientsideFunction,
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
    CAPTION_SEP,
    FIGURE_HEIGHT,
    PANEL_HEIGHT,
    PANEL_ROW_MAP,
    PipelineError,
    apply_arm_style,
    decode_plotly_array,
    figures_from_outputs,
    _panel_dict,
    panel_of,
    placeholder_figure,
    prewarm_tessera_interconnect,
    run_pipeline,
    scenario_topdown_figure,
    note_differing_y_extents,
    reach_floor_single_arm,
    share_heatmap_z_limits,
    y_extent_lock_of,
)
from webapp.scenario_editor import map_figure, scenario_from_json_safe, summarize

HOST = "127.0.0.1"
PORT = 8050

#: Results-tab animation period, ms -- one frame step per tick for every animated
#: panel at once (see the "results-clock" Interval and webapp/assets/results_clock.js).
#: 700 ms, not Plotly's own 350 ms ▶ default: the presenter talks over these screens,
#: 5-frame loops at 350 ms restart every 1.75 s, and the owner measured the RDP link
#: as comfortable at this rate ("Framerate looks fine over RDP", live test
#: 2026-09-24) -- which is also what retired the old "step the slider, never press
#: Play" runbook rule (webapp/runbook.py).
RESULTS_CLOCK_MS = 700

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
        html.H2("Array Processing End-to-End Simulator", className="page-title"),
        html.P("Block-diagram pipeline control and scenario scheduling.",
               className="page-subtitle"),

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

        # ONE CLOCK for every animated panel on the Results tab (owner, live test
        # 2026-09-24: "both as if the play button was hit (and should loop
        # repeatedly)"; "Framerate looks fine over RDP"). Plotly's own ▶ button
        # animates ONE panel, once, at its own rate -- two A/B arms started by hand
        # drift apart immediately and neither loops. This Interval ticks outside the
        # tab content (so switching tabs never unmounts it) and the clientside
        # callback below steps EVERY animated figure to the same frame index.
        dcc.Interval(id="results-clock", interval=RESULTS_CLOCK_MS, n_intervals=0),
        dcc.Store(id="results-clock-tick", data=0),

        dcc.Tabs(id="tabs", value="tab-blocks", children=[
            dcc.Tab(label="Block Diagram", value="tab-blocks",
                    children=html.Div(block_diagram.layout(), style={"padding": "12px"})),
            dcc.Tab(label="Scenario", value="tab-scenario",
                    children=html.Div(scenario_editor.layout(), style={"padding": "12px"})),
            dcc.Tab(label="Results", value="tab-results",
                    # 8 px, not 12: the budget above the first panel is 150 px
                    # (acceptance check 2) and this padding is inside it.
                    children=html.Div(id="results-tab-content", style={"padding": "8px"})),
        ]),
    # 1600px, not the original 1280px: on the 1920x1080 conference monitor the narrower
    # cap wasted ~338px of margin per side and bought the lone-figure Thrust 3 screen
    # nothing from the bigger display (coordinator finding, 2026-09-23). Figures scale
    # with their container; the podium-distance font floor (pipeline_runner._make_legible)
    # is independent of this and unaffected.
    # 1560 px, centred, 20 px side padding -> 1520 px of content, which is what the
    # two-column A/B geometry (746 + 28 + 746) is built from (layout spec section 2.1).
    ], className="app-shell")


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
    """Show the parameter editor for the tapped diagram NODE.

    A node can stand for several registry blocks (the one-chain diagram collapses the
    three sources, the transmit tributary and the ADC sub-chain), so it carries the block
    the editor should open on in `data["block"]` and `param_editor` renders every member
    from there. `data["id"]` is the fallback for a node that is one block.
    """
    block_state = block_state or default_block_state()
    data = node_data or {}
    block_id = data.get("block") or data.get("id") or PRODUCT_IDS[0]
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


#: A screen-note clause that only makes sense when BOTH arms are on screen. Matched
#: case-insensitively against each ";"-separated clause by `_one_arm_screen_note`.
_TWO_ARM_CLAUSE = re.compile(
    r"both arms|two arms|other arm|either arm|arm A\b|arm B\b|A/B", re.IGNORECASE)


#: The page-foot screen note is capped at this many characters, cut at a CLAUSE
#: boundary, with the remainder reachable in both arms' Details (where the whole note
#: already lives, verbatim).
#:
#: Hostile round 12, item 16: Thrust 4's note is 968 characters and rendered as FIVE
#: lines of the smallest type on the page, carrying -- in that type -- the NEXT/FEXT
#: reference pair and the 3.53 dB scale-model admission. A caveat nobody can read at
#: podium distance is not a disclosure; a short line that says what the caveat is
#: about, over a disclosure one click away, is.
#:
#: 420, NOT a tighter number, and the value is not free: the three Thrust 5 notes carry
#: mandatory disclosures (both CFAR baselines, the chance floor, the scoring crop, the
#: corpus's v_max, the ADC's automatic gain) and are already held to a 400-character
#: budget by `tests/test_demo_presets.py`. A cap below that would start deciding which
#: of those reaches the screen, which is a content decision this function must not
#: make. At 420 every Thrust 5 note renders WHOLE and only the genuinely over-long
#: notes (T4 968, T6 749, T2 612) are cut.
PAGE_FOOT_NOTE_MAX_CHARS = 420


def _foot_note(note: str) -> str:
    """`note` cut to `PAGE_FOOT_NOTE_MAX_CHARS` at a clause boundary, with a pointer to
    where the rest is. Never cut mid-phrase: acceptance check 12 forbids a truncation
    mark in visible text, and the full text is in Details either way.

    THE ARRAY/SCENE DISCLOSURE SURVIVES THE CUT (hostile round 13, N14). It is the LAST
    clause of every card that carries it, so it was the first thing this function
    dropped -- measured on the 2026-09-25 render: Thrust 1's 493-character note was cut
    at 420 and the visible page-foot ended "...share one front-end config. Full note in
    each arm's Details.", with the array geometry, the band, the boresight offset and the
    diffuse-scattering assumption all gone from the screen. It is mandatory on any card
    that mentions the array (`demo_presets._ARRAY_DISCLOSURE`, the rule the cards are
    written to), so the BUDGET is reserved for it and the prose before it is what gives
    way: the total still fits `PAGE_FOOT_NOTE_MAX_CHARS`.
    """
    note = (note or "").strip()
    if len(note) <= PAGE_FOOT_NOTE_MAX_CHARS:
        return note
    pointer = " Full note in each arm's Details."
    tail = ""
    budget = PAGE_FOOT_NOTE_MAX_CHARS
    try:
        from webapp.demo_presets import _ARRAY_DISCLOSURE
        if _ARRAY_DISCLOSURE and _ARRAY_DISCLOSURE in note:
            tail = " " + _ARRAY_DISCLOSURE
            # The pointer costs its own characters too -- counted, so the RESULT fits the
            # budget rather than the intermediate cut.
            budget = max(80, PAGE_FOOT_NOTE_MAX_CHARS - len(tail) - len(pointer))
    except Exception:
        pass
    head = note[:budget]
    cuts = [head.rfind(sep) for sep in (". ", "; ", " -- ")]
    cut = max(cuts)
    if cut <= 0:
        cut = head.rfind(" ")
    kept = note[:cut].rstrip(" ;,-")
    if not kept.endswith("."):
        kept += "."
    return kept + pointer + tail


def _one_arm_screen_note(note: str) -> str:
    """`note` with every two-arm clause removed -- what a SINGLE-arm screen prints.

    The cancel journey renders one map, from one arm, under an amber "arm B did not
    run" chip, and then printed the preset's own foot note describing "the two maps
    share one colour scale ... the 0.5 mA arm's background is visibly brighter": a
    comparison that did not happen, on the one screen that says so itself (hostile
    round 11, C1). The note text belongs to the preset (webapp/demo_presets.py); the
    RENDERING rule is here. Clauses are dropped whole, never re-punctuated, and the
    clip rule / geometry / band clauses beside them are untouched."""
    clauses = [c for c in (note or "").split("; ") if not _TWO_ARM_CLAUSE.search(c)]
    return "; ".join(clauses)


def _ab_arm_line(preset: "DemoPreset", arm: str) -> str:
    """'A (as loaded): <label> <value> -- before' or 'B: <label> <value> -- after',
    naming the knob the way the editor labels it. ONE arm's value only -- before
    2026-09-23 both panels' banners printed BOTH arms' values with identical text, so
    a visitor reading a single panel could not tell which arm was on screen (defect
    found reading the rendered Results tab)."""
    bid, key, _value_b = preset.ab
    # The FULL registry label here, unit parenthetical and all: this line is the
    # verbatim record in each arm's Details, where there is room for it. Only the CHIP
    # drops a duplicated unit (`_chip_label_without_a_duplicated_unit`), because that is
    # the 20 px line a photograph identifies the arm by.
    label = next((ps.label for ps in BLOCKS_BY_ID[bid].params if ps.key == key), key)
    if arm == "a":
        return f"A (as loaded): {label} {preset.ab_label_a or '?'} -- before"
    return f"B: {label} {preset.ab_label_b or '?'} -- after"


#: Longest chip that fits ONE line at 20/700 in a 746 px column (measured on the
#: rendered page, 2026-09-24). Past it the VALUE moves to the caption -- see
#: `_ab_arm_chip` and `_ab_arm_chip_overflow`.
ARM_CHIP_MAX_CHARS = 58


_DIGIT_RE = re.compile(r"\d")
#: One (...) group, with its leading whitespace. Non-nested by construction: none of
#: the presets' arm labels nests parentheses, and `_balance_parens` cleans up after
#: any form that cuts one open.
_PAREN_RE = re.compile(r"\s*\(([^()]*)\)")


def _balance_parens(text: str) -> str:
    """`text` with unmatched parentheses removed -- a cut inside a parenthetical would
    otherwise leave a stray ")" on the largest type on the screen."""
    out, depth = [], 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            if depth == 0:
                continue
            depth -= 1
        out.append(ch)
    text = "".join(out)
    if depth and "(" in text:
        text = text[:text.rfind("(")]
    return text.strip(" ,;-")


def _chip_value_forms(text: str) -> List[str]:
    """Progressively shorter renderings of ONE arm's knob value, longest first. Every
    rung still CARRIES THE VALUE -- that is the whole point of this ladder.

    Before 2026-09-24 an over-long value was dropped from the chip entirely, so the
    largest type on the screen read "B — gap_response" / "B — Corner range (m)", and on
    Thrust 4 the two arms' chips were character-for-character identical over two
    different runs (hostile round 11, defect D1). The rungs, in order:

      1. the label verbatim;
      2. minus any parenthetical carrying no digit ("(as built)", "(largest skirt
         mover)") -- a qualifier, never the value;
      3. a digit-bearing parenthetical cut to the comma-clause holding its first
         number ("(shipped default, 10 passes/frame baseline)" -> "(10 passes/frame
         baseline)");
      4. from the first number onward, dropping the prose in front of it
         ("canonical Tessera geometry (50 um presented)" -> "50 um presented");
      5. (4) with every parenthetical gone;
      6. the bare number and its unit ("50 um", "25 m", "10 passes").

    A value with no digits at all (none ship today, but a preset may add one) walks
    a clause-head ladder instead, which also never empties.
    """
    text = (text or "?").strip()
    if not _DIGIT_RE.search(text):
        head = _clause_head(text, (": ", " -- ", ", ", " ("))
        return [text, text, head, head, head, head]
    keep_digit_parens = _PAREN_RE.sub(
        lambda m: m.group(0) if _DIGIT_RE.search(m.group(1)) else "", text).strip()

    def _tighten(m):
        inner = m.group(1)
        if not _DIGIT_RE.search(inner):
            return m.group(0)
        parts = [p.strip() for p in inner.split(",")]
        return " (" + next((p for p in parts if _DIGIT_RE.search(p)), inner) + ")"

    tightened = _PAREN_RE.sub(_tighten, keep_digit_parens).strip()
    start = _DIGIT_RE.search(tightened).start()
    while start > 0 and tightened[start - 1] not in " (":
        start -= 1
    from_number = _balance_parens(tightened[start:])
    no_parens = _balance_parens(_PAREN_RE.sub("", from_number))
    m = re.search(r"[-+]?\d[\d.,]*(\s*[A-Za-z%µ°/]+)?", no_parens or tightened)
    bare = (m.group(0).strip() if m else no_parens) or no_parens
    return [text, keep_digit_parens, tightened, from_number, no_parens, bare]


#: Display names for A/B knobs that have no registry ParamSpec -- `_INTERNAL_PARAMS`
#: in webapp/demo_presets.py, knobs deliberately kept off the parameter editor so no
#: operator types a refinement-pass count mid-demo. The chip's label lookup falls back
#: to the raw key for these, and Thrust 3's chip read "A -- gap_response fixed effort
#: (5 passes/frame)" on the 2026-09-25 render: a Python identifier, in 20 px, as the
#: one line identifying the arm.
_INTERNAL_PARAM_CHIP_LABELS = {
    ("subspace", "gap_response"): "Refinement effort",
}


def _chip_label_without_a_duplicated_unit(label: str, *values: str) -> str:
    """`label` with its trailing unit parenthetical dropped when the arm VALUES already
    carry that unit.

    The chip is "<param label> <value>", and a registry label names its unit for the
    parameter EDITOR, where there is no value beside it. On a chip there is: the pairs
    read "LNA bias current (mA) 8 mA", "Corner range (m) 1 m", "Tessera: TSV height
    (um) 50 um presented" -- the unit twice, in a 20 px chip that is the one thing a
    photograph of the screen is meant to identify the arm by (hostile round 12, item
    16). Dropped only when a value actually repeats it, so a label whose unit is NOT in
    the value keeps it and no arm is left unitless.
    """
    m = re.search(r"\s*\(([^()]{1,6})\)\s*$", label)
    if not m:
        return label
    unit = m.group(1).strip()
    if not unit or not all(re.search(r"\d\s*" + re.escape(unit) + r"\b", v or "")
                           for v in values if v):
        return label
    return label[:m.start()].rstrip()


def _ab_arm_chip(preset: "DemoPreset", arm: str) -> str:
    """The ARM CHIP (layout spec section 2.3): just the knob and its value, e.g.
    "A -- ADC 12 bit (as built)". The full banner line above is not deleted -- it is
    the first line of this arm's Details disclosure -- but at podium distance a 2-4
    line bold banner per column wrapped across the 13 px gutter and read as one
    garbled paragraph (hostile round 10, defect 2.6).

    The VALUE is never dropped (hostile round 11, D1): when the full label does not
    fit one line, BOTH arms step down `_chip_value_forms`' ladder together -- the same
    rung on both, so the two chips stay parallel and a viewer compares like with like
    -- until both fit. Whatever the rung leaves off is still on screen, at the front
    of that arm's one-line caption (`_ab_arm_chip_overflow` -> `_arm_caption`), and in
    full in Details."""
    bid, key, _value_b = preset.ab
    label = next((ps.label for ps in BLOCKS_BY_ID[bid].params if ps.key == key),
                 _INTERNAL_PARAM_CHIP_LABELS.get((bid, key), key))
    label = _chip_label_without_a_duplicated_unit(label, preset.ab_label_a or "",
                                                  preset.ab_label_b or "")
    forms_a = _chip_value_forms(preset.ab_label_a or "?")
    forms_b = _chip_value_forms(preset.ab_label_b or "?")
    rung = len(forms_a) - 1
    for i, (va, vb) in enumerate(zip(forms_a, forms_b)):
        if (len(f"A — {label} {va}") <= ARM_CHIP_MAX_CHARS
                and len(f"B — {label} {vb}") <= ARM_CHIP_MAX_CHARS):
            rung = i
            break
    value = (forms_a if arm == "a" else forms_b)[rung]
    return f"{arm.upper()} — {label} {value}"


def _ab_arm_chip_overflow(preset: "DemoPreset", arm: str) -> str:
    """WHAT THE CHIP DROPPED, or "" when the chip carries the value whole. Rendered at
    the front of that arm's one-line caption, so a value too long for a chip is MOVED,
    never lost -- on Thrust 4 it is what says which Tessera geometry this arm ran
    (hostile round 10, section 1.7).

    Only the part the chip does NOT already show (hostile round 12, item 16 residue,
    closed 2026-09-25). It used to return the WHOLE value whenever the chip had shortened
    it at all, so on Thrust 3 and Thrust 4 the 20 px chip read "A -- Gap response fixed
    effort" and the caption 15 px below it read "fixed effort (5 passes/frame)" -- the
    same words twice, in the two largest pieces of text on that arm, for nothing. Word
    filtering rather than a prefix cut, because `_chip_value_forms`' shorter rungs are not
    prefixes of the full value (rung 4 drops the prose in FRONT of the number)."""
    value = (preset.ab_label_a if arm == "a" else preset.ab_label_b) or "?"
    chip = _ab_arm_chip(preset, arm)
    if chip.endswith(value):
        return ""

    def _norm(word: str) -> str:
        return word.strip("().,;:-——").lower()

    shown = {_norm(w) for w in chip.split()} - {""}
    kept = [w for w in value.split() if _norm(w) not in shown]
    rest = " ".join(kept).strip(" ,;")
    # A ONE-WORD scrap is not a caption clause: "fixed", left over from "fixed effort
    # (5 passes/frame)" once the chip showed the parenthetical, says nothing on its own
    # and costs the line that carries the run's own fact. Nothing is lost -- the whole
    # value is in that arm's Details, and in the card.
    return rest if len(rest.split()) >= 2 else ""


#: Longest run-identity line that fits ONE line at 18/600 across 1520 px of content
#: minus the transport and (on the cancel path) the CANCELLED chip -- measured on the
#: rendered page, 2026-09-24. Past it the line is SHORTENED AT A CLAUSE BOUNDARY, never
#: cut mid-phrase: acceptance check 12 forbids a truncation mark in visible text, and
#: CSS `text-overflow: ellipsis` draws one that does not even appear in `innerText`, so
#: a clipped line passed an ellipsis-count check while visibly ending in "..." on
#: screen (found on the Thrust 5 render, 2026-09-24).
RUN_IDENTITY_MAX_CHARS = 100
#: Characters the CANCELLED chip costs this line on the cancel path.
RUN_IDENTITY_CHIP_COST = 26


def _clause_head(text: str, seps=(": ", " -- ", " vs ", " (")) -> str:
    """`text` up to its first clause separator, or `text` unchanged. The result is a
    complete phrase, so nothing needs marking as elided."""
    cuts = [c for c in (text.find(sep) for sep in seps) if c > 0]
    return text[:min(cuts)].rstrip() if cuts else text


#: Words carrying no identity, ignored when asking "does this clause already say what
#: the SOURCE slot says?" (`_preset_slot`).
_SLOT_STOPWORDS = frozenset(("the", "a", "an", "of", "from", "in", "on", "with"))


def _preset_slot(label: str, source: str) -> str:
    """Slot 2 of the run-identity line: this preset's OWN short name, under one
    convention on every screen (hostile round 11, C7 -- the three Thrust 5 screens
    used to put a provenance claim, a claim-plus-condition and a model name in the
    same slot).

    The rule: walk the label's clauses left to right and take the first one that is
    not already said by the SOURCE slot beside it. "live chain from the stored
    channel: classical CFAR" beside a source of "Corpus Replay (live chain from
    stored channel)" therefore prints "classical CFAR" -- the detector, which is what
    the other two Thrust 5 screens print too -- instead of spending the slot on a
    duplicate."""
    src_words = {w.strip(".,;:()'").lower() for w in (source or "").split()}
    src_words -= _SLOT_STOPWORDS
    for clause in re.split(r":\s+|,\s+", label):
        clause = _clause_head(clause.strip())
        if not clause:
            continue
        words = {w.strip(".,;:()'").lower() for w in clause.split()} - _SLOT_STOPWORDS
        if words and words <= src_words:
            continue  # this clause only repeats the source slot
        return clause
    return _clause_head(label)


#: A parenthetical clause that says only which BAND a source is on -- "Ka-band",
#: "30 GHz", "Ka corpus". Recognised so `_env_without_band_clauses` can drop exactly
#: those and keep whatever else the parenthetical distinguishes the file by.
_BAND_CLAUSE_RE = re.compile(r"^(ka|ka-band|ka corpus|[\d.]+\s*ghz([- ]band)?)$", re.I)


def _env_without_band_clauses(source: str) -> str:
    """`"munich (Ka-band, 30 GHz, LoS sweep ~2.0 deg/frame)"` ->
    `"munich (LoS sweep ~2.0 deg/frame)"`; `""` when the parenthetical is only the band
    (dropping it would leave a bare name that names a DIFFERENT file -- see the caller).
    """
    head = _clause_head(source, (": ",))
    if "(" not in head or not head.rstrip().endswith(")"):
        return ""
    name, _, inner = head.partition("(")
    kept = [c.strip() for c in inner.rstrip().rstrip(")").split(",")
            if c.strip() and not _BAND_CLAUSE_RE.match(c.strip())]
    return f"{name.strip()} ({', '.join(kept)})" if kept else ""


def _run_identity_line(preset, axis_meta: Dict[str, Any], n_clicks, n_steps: int) -> str:
    """The ONE full-width line that replaces the `Results` H3 and the run half of the
    banner (layout spec section 2.3):

        Thrust 5 - classical CFAR - munich (Ka-band, 30 GHz) - 5 frames - run #1 14:29:06

    Every part is read from the run itself; nothing is typed. When the preset's own
    label is long prose it is shortened at a clause boundary rather than clipped -- the
    full label, the full source string and the frame count all remain in the banner,
    which is the first line of each arm's Details disclosure.
    """
    import time as _time
    n_run = axis_meta.get("n_steps_run", n_steps)
    frames = (f"{n_run} of {n_steps} frames" if n_run != n_steps
              else f"{n_steps} frames")
    run = f"run #{n_clicks} {_time.strftime('%H:%M:%S')}"
    thrust = f"Thrust {preset.thrust}" if preset is not None else ""
    label = ""
    if preset is not None:
        # The label already starts "Thrust N - ..." (or "Thrust N (LEAD) - ...");
        # printing the thrust twice on one line is part of what pushed this past the
        # page width.
        label = re.sub(rf"^Thrust {preset.thrust}(?![0-9])[^-]*-\s*", "", preset.label)
    # "Sionna frames: munich (Ka-band, 30 GHz)" -> the environment IS the identity,
    # the loader is not.
    source = (axis_meta.get("source") or "").replace("Sionna frames: ", "")
    slot = _preset_slot(label, source)
    # The BAND, in the 18 px line rather than mid-sentence in the 15 px foot note
    # (hostile round 11, H4): three of seven screens run on a 77 GHz corpus while the
    # other four announce "munich (Ka-band, 30 GHz)" in this very line, and that
    # discrepancy was carried by the smallest type on the page. Empty on the Sionna
    # path, whose source string already names its band.
    band = axis_meta.get("band") or ""

    # THE ENVIRONMENT IS ONE FIELD, band included, on every screen (hostile round 12,
    # item 16: "identity line format identical on every screen"). Before this the band
    # was its own clause, so the corpus screens read
    # "Thrust 5 . classical CFAR . Corpus Replay . 30 GHz corpus . 5 frames . run #1"
    # -- six fields against the Sionna screens' five -- and the cancel screen, whose
    # budget is smaller, shortened "munich (Ka-band, 30 GHz)" to a bare "munich", so
    # one run's environment had three spellings across seven screens.
    def _with_band(name: str) -> str:
        return f"{name} ({band})" if band and band not in name else name

    env_full = _with_band(source)
    #: "Corpus Replay (live chain from stored channel): test split from frame 0"
    #: -> the environment without its run-specific tail.
    env_name = _with_band(_clause_head(source, (": ",)))
    #: ...and without its parenthetical, which is a mouthful on the corpus path. On the
    #: Sionna path the parenthetical IS the band, so it is kept by `_with_band`'s own
    #: test unless the band is already inside the name.
    env_short = _with_band(_clause_head(source, (": ", " (")))
    #: ...and the one in between: the name plus whatever its parenthetical says BESIDES
    #: the band. On the smaller cancel-screen budget Thrust 3 could otherwise only fit a
    #: bare "munich", which is not merely short -- it is the name of the OTHER munich
    #: file, the static one every other thrust runs, on the one screen whose whole
    #: subject is that the scene sweeps. "munich (LoS sweep ~2.0 deg/frame)" is shorter
    #: than the full form and still says which file ran. Empty when the parenthetical is
    #: ONLY the band, where dropping it would leave the bare name again.
    #: Only on the path whose band lives INSIDE the name (the Sionna sources). When the
    #: band is its own `axis_meta` field -- the corpus path -- dropping the
    #: parenthetical's clauses would drop the band with them, which is the exact defect
    #: the round-12 residue is about; those rungs then collapse to empty and are skipped.
    env_medium = _env_without_band_clauses(source) if not band else ""

    # A ladder of progressively shorter forms, each made of WHOLE clauses. The first
    # that fits wins; the last rung always fits. Every rung keeps the environment as
    # ONE field, and the frame count is given up BEFORE the environment's own name is
    # shortened: on the cancel path the amber chip beside this line already reads
    # "CANCELLED -- N of M frames", while nothing else on that screen says which
    # environment ran.
    candidates = [
        [thrust, label, env_full, frames, run],
        [thrust, slot, env_full, frames, run],
        [thrust, slot, env_name, frames, run],
        [thrust, slot, env_name, run],
        # THE ENVIRONMENT WHOLE, WITHOUT THE PRESET'S NAME, before any rung that
        # shortens the environment (2026-09-25, round-12 residue: one environment-name
        # format everywhere). Thrust 3's environment is the longest on the page --
        # "munich (Ka-band, 30 GHz, LoS sweep ~2.0 deg/frame)" -- and beside its own
        # 27-character slot it overran the budget by 2 characters, so the ladder fell
        # through to a bare "munich" ON THE ONE SCREEN WHOSE SUBJECT IS THE SWEEP. The
        # thrust number already identifies which screen this is, and the card above it
        # carries the preset's name; nothing else on the page names the FILE.
        [thrust, env_full, frames, run],
        [thrust, env_full, run],
        # The band dropped, the distinguishing clause kept (see `env_medium`). SKIPPED
        # ENTIRELY when `env_medium` is empty: a rung with an empty environment slot
        # joins to a line with no environment on it at all, which fits every budget and
        # therefore wins -- it dropped the corpus screens' environment outright the first
        # time these rungs were added (caught by re-running every preset's line).
        *([[thrust, slot, env_medium, frames, run],
           [thrust, env_medium, frames, run],
           # ...and without the frame count, which is the rung the CANCEL screen lands
           # on (its budget is 26 characters smaller because of the amber chip, and that
           # chip already reads "CANCELLED -- N of M frames").
           [thrust, env_medium, run]] if env_medium else []),
        [thrust, slot, env_short, frames, run],
        # Keep the ENVIRONMENT before giving up on the label: a screen that says only
        # "Thrust 5 . 5 frames . run #1" has lost the two facts a photograph needs
        # (which preset, which corpus). Measured on thrust5_detector_ml, 2026-09-24.
        [thrust, slot, env_short, run],
        [thrust, env_short, frames, run],
        [thrust, env_short, run],
    ]
    # A cancelled run also draws an amber `CANCELLED -- N of M frames` chip in this
    # same row, which takes ~230 px out of the line's own width (measured on
    # cancel_results.png, 2026-09-24) -- so the budget is smaller on that path.
    budget = (RUN_IDENTITY_MAX_CHARS - RUN_IDENTITY_CHIP_COST
              if axis_meta.get("cancelled") else RUN_IDENTITY_MAX_CHARS)
    for parts in candidates:
        line = CAPTION_SEP.join(x for x in parts if x)
        if len(line) <= budget:
            return line
    return CAPTION_SEP.join(x for x in candidates[-1] if x)


#: RETRACTED (layout spec section 2.3, 2026-09-24). Run notes used to be truncated at
#: their first `" -- "` or 160 characters for the Results tab, which is how "the ADC
#: bits, full scale, IF high-p ...", "...ring3x3 arran ..." and "...trained on
#: b1_bench_v3/benchmark_v1_D2 and these f ..." reached the screen -- the SMALLEST type
#: on the page was the only text that lost information, and mid-word truncation reads
#: as a crash to a non-expert (hostile round 10, defect 2.3; acceptance check 12).
#: The full note now lives in the per-arm Details disclosure, and the same first
#: clause -- the headline the presenter actually reads -- is the arm's one-line
#: CAPTION, which is why this function survives as `_arm_caption`'s helper rather
#: than as a display shortener.
_NOTE_HEADLINE_CHARS = 110


def _note_headline(note: str) -> str:
    """The first clause of one run note: up to its first `" -- "` or
    `_NOTE_HEADLINE_CHARS` characters. Used ONLY for the arm's one-line caption -- the
    full note is never shortened anywhere any more (see the comment above).

    The live-chain correctness gate's own note runs its headline number, then a
    `" -- "`, then an attribution clause that can run another 2-3 lines; the headline
    IS the number the presenter reads ("live vs stored ADC: max |diff| 0 of 4096 LSB"),
    so it is what the caption carries."""
    sep = note.find(" -- ")
    head = note[:sep].rstrip() if sep >= 0 else note
    if len(head) <= _NOTE_HEADLINE_CHARS:
        # A clean clause boundary: nothing is elided, so nothing is marked. Acceptance
        # check 12 forbids truncation marks in visible text, and a note cut at " -- "
        # is a complete sentence, not a cut-off one.
        return head
    # Still too long: cut at the last WORD boundary and mark it. Mid-word truncation
    # ("...ring3x3 arran ...") is what made the smallest type on the screen the only
    # text that lost information (hostile round 10, defect 2.3).
    clipped = head[:_NOTE_HEADLINE_CHARS]
    space = clipped.rfind(" ")
    if space > 40:
        clipped = clipped[:space]
    return clipped.rstrip(" ,;") + "…"


def _notes_line(axis_meta: Dict[str, Any]) -> List[str]:
    """`axis_meta['notes']` (`pipeline_runner.run_pipeline`'s `run_notes` -- e.g. the
    interconnect Tessera surrogate's scale-model/frequency disclosure,
    `InterconnectBlock.describe()`), one entry per note, UNTRUNCATED. These reached
    only the Block Diagram tab's status line before 2026-09-23, so a Thrust 4 Results
    screen carried no on-screen record of the frequency it was actually evaluated at --
    a visitor reading only that tab, or a photograph of it, never saw the disclosure at
    all. `[]` when there are none."""
    return list(axis_meta.get("notes") or [])


def _arm_caption(payload: Dict[str, Any]) -> str:
    """ONE line, <= 110 characters: the single fact this arm adds (layout spec
    section 2.3). The headline of this arm's first run note -- which on every Thrust 5
    screen is the live-vs-stored ADC gate's own number, the statistic the presenter
    must be able to read without opening anything."""
    parts = []
    overflow = payload.get("_arm_chip_value")
    if overflow:
        parts.append(overflow)
    notes = [n for n in (payload.get("_notes") or [])
             # The environment note ("Environment 'munich (Ka-band, 30 GHz)': frames
             # carry a 30 GHz carrier.", pipeline_runner.run_pipeline) is IDENTICAL on
             # both arms and repeats the band the run-identity line already prints in
             # 18 px bold -- so on Thrust 1-3 the one line reserved for "the fact this
             # arm adds" said nothing about the arm at all (hostile round 11, H7).
             # Skipped here only; it is unchanged in Details.
             if not str(n).startswith("Environment '")]
    head = _note_headline(notes[0]) if notes else ""
    if head:
        parts.append(head)
    line = CAPTION_SEP.join(parts)
    # ONE line, and the column clips at ~86 characters at 16 px.
    if len(line) <= 86 or len(parts) < 2:
        return line if len(line) <= 86 else parts[0]
    # Both facts, one line: something has to go. The run note wins when the chip
    # above ALREADY carries every number the knob value has -- on Thrust 5's
    # IF-corner A/B the chip reads "B -- Corner range (m) 25 m (attenuates ~4.3 dB at
    # 22 m)" and the leftover adds only the words "IF high-pass corner", which the
    # knob label itself says, while the note it was crowding out is this arm's OWN
    # live-vs-stored ADC gate ("max |diff| = 1 of 8 LSB") -- the standard the CFAR and
    # RADDetNet screens already meet on both arms (hostile round 11, H7).
    # Never when the note's own headline had to be elided: a caption ending in "…"
    # is the defect hostile round 10 (2.3) found, and the knob value is whole.
    chip_has_the_numbers = _chip_carries_the_numbers(
        payload.get("_arm_chip") or "", overflow)
    if chip_has_the_numbers and not head.endswith("…"):
        return head
    # Otherwise the knob value wins: the run note is in Details in full either way.
    return parts[0]


def _chip_carries_the_numbers(chip: str, value: str) -> bool:
    """True when every digit-bearing token of `value` is already printed on `chip` --
    i.e. the leftover text would add words, not numbers. Used by `_arm_caption` to
    decide which of two facts keeps the arm's one line."""
    numbers = [tok for tok in value.split() if any(c.isdigit() for c in tok)]
    return bool(numbers) and all(tok in chip for tok in numbers)


# =================================================================================
# Results page: the fixed-geometry layout (spec 2026-09-24, sections 2.1-2.3)
# =================================================================================

#: Content width inside the 1560 px page container, and the two-column A/B geometry.
#: The gutter went from 13 px to 28 (hostile round 10, defect 4: at 13 px and with no
#: rule between them, arm A's bold banner ran straight into arm B's and the two arms
#: did not read as separate objects at podium distance). The extra width is bought back
#: from the panel padding, not from the plot.
RESULTS_CONTENT_WIDTH = 1520
AB_COLUMN_WIDTH = 746
AB_GUTTER = 28
#: A lone product is never stretched across the full width: at 1523 px the thrust-1 map
#: rendered 1080x230, a 4.7:1 strip (hostile round 10, defect 10). One product -> one
#: 1008 px panel; two or more -> the same two-up grid as the A/B case.
SINGLE_PANEL_WIDTH = 1008
ROW_GAP = 16


def _panel_header(panel: Dict[str, Any]):
    """The HTML band above one plot: panel title (22/600, ONE line) and caption
    (16/400, ONE line). Both are `text-overflow: ellipsis`-free by construction -- the
    caption is built <= 110 characters where the figure is built, precisely so nothing
    has to be clipped here (acceptance check 12)."""
    caption = CAPTION_SEP.join(panel.get("caption") or [])
    return html.Div([
        html.Div(panel.get("title") or "", className="panel-title"),
        html.Div(caption, className="panel-caption"),
    ], className="panel-header")


#: Extra figure height a panel gets when it is the LONE product on a single-arm screen
#: (hostile round 11, D6). A 1008 px panel leaves ~842 px of plot width, and at the
#: shared 340 px plot height that is 2.47 : 1 -- past the spec's 2.2 : 1 cap for a
#: single-arm plot, measured at 2.49 : 1 on the cancel screen. +50 px of figure height
#: puts the plot at ~390 px, i.e. 2.16 : 1, without touching any two-up panel.
SINGLE_PANEL_EXTRA_HEIGHT = 50


def _panel_block(fig_dict, *, width: str = "100%", extra_height: int = 0):
    """One bordered product panel: fixed height for its row kind, HTML header, plot.

    No modebar: the zoom/export toolbar overlaps the header at this width and none of
    its tools matter for read-only results.

    `extra_height` grows BOTH the panel box and the figure inside it, so the plot area
    -- not the margins -- is what gets the extra pixels. Used only by the single-arm
    layout (see `SINGLE_PANEL_EXTRA_HEIGHT`); every A/B panel keeps the fixed row
    height that makes the same product the same size on every screen (check 20)."""
    panel = panel_of(fig_dict)
    row = panel.get("row") or PANEL_ROW_MAP
    height = PANEL_HEIGHT.get(row, PANEL_HEIGHT[PANEL_ROW_MAP]) + extra_height
    if extra_height:
        fig_dict = copy.deepcopy(fig_dict)
        layout = fig_dict.setdefault("layout", {})
        base = layout.get("height") or FIGURE_HEIGHT.get(row,
                                                         FIGURE_HEIGHT[PANEL_ROW_MAP])
        layout["height"] = float(base) + extra_height
    return html.Div([
        _panel_header(panel),
        dcc.Graph(figure=fig_dict, config={"displayModeBar": False},
                  className="panel-graph"),
    ], className="result-panel", style={"height": f"{height}px", "width": width})


def _empty_panel(row: str):
    """The opposite cell when only one arm produced a product: a panel-shaped hole, so
    the other arm's panels do not silently shift up a row."""
    height = PANEL_HEIGHT.get(row, PANEL_HEIGHT[PANEL_ROW_MAP])
    return html.Div(className="result-panel result-panel-empty",
                    style={"height": f"{height}px"})


def _row(left, right=None):
    """One product row. Arm A's cell left, arm B's right, both the same width and the
    same height whether or not either is empty (acceptance check 4)."""
    if right is None:
        return html.Div([html.Div(left, className="ab-cell ab-cell-single")],
                        className="ab-row")
    return html.Div([html.Div(left, className="ab-cell"),
                     html.Div(right, className="ab-cell")], className="ab-row")


def _grid(figs: Dict[str, Any], side_column=None):
    """The SINGLE-arm layout. One product gets ONE 1008 px panel (never the full
    width); two or more wrap into the same two-up grid the A/B case uses, so the same
    product has the same panel geometry on every screen (acceptance check 20).

    `side_column` is the Details disclosure, and with ONE product it goes BESIDE the
    panel rather than above it -- layout spec section 2.1, "with the details disclosure
    in the 480 px column beside it". Hostile round 11, D5: on the cancel screen the
    disclosure sat above and left a 516 x 540 px empty rectangle to the right of the
    only picture on the page. It is ignored in the multi-product case, where the grid
    is two-up and there is no spare column."""
    items = list(figs.values())
    if len(items) == 1:
        panel = _panel_block(items[0], width=f"{SINGLE_PANEL_WIDTH}px",
                             extra_height=SINGLE_PANEL_EXTRA_HEIGHT)
        if side_column is None:
            return html.Div([_row(panel)], className="results-grid")
        return html.Div([
            html.Div([html.Div(panel, className="ab-cell ab-cell-single"),
                      html.Div(side_column, className="single-side-column")],
                     className="ab-row"),
        ], className="results-grid")
    rows = []
    for i in range(0, len(items), 2):
        pair = items[i:i + 2]
        if len(pair) == 2:
            rows.append(_row(_panel_block(pair[0]), _panel_block(pair[1])))
        else:
            rows.append(_row(_panel_block(pair[0], width=f"{AB_COLUMN_WIDTH}px")))
    return html.Div(rows, className="results-grid")


def _ab_columns(header_a, header_b, figs_a: Dict[str, Any], figs_b: Dict[str, Any]):
    """Arm A left, arm B right: a header row (each arm's chip, caption and Details
    disclosure above its own column) then ONE ROW PER PRODUCT, so the two copies of the
    same panel sit at the same height, at the same width, next to each other.

    Product order follows arm A's insertion order (which mirrors
    `figures_from_outputs`' build order); a product only one arm produced still gets
    its own row, with a panel-shaped hole opposite it."""
    keys = list(figs_a) + [k for k in figs_b if k not in figs_a]
    rows = [_row(header_a, header_b), html.Div(className="section-rule")]
    for key in keys:
        row_kind = (panel_of(figs_a.get(key) or figs_b.get(key)).get("row")
                    or PANEL_ROW_MAP)
        rows.append(_row(
            _panel_block(figs_a[key]) if key in figs_a else _empty_panel(row_kind),
            _panel_block(figs_b[key]) if key in figs_b else _empty_panel(row_kind),
        ))
    return html.Div(rows, className="results-grid")

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
            # The "identical on both arms" clause is NOT added here any more: it is a
            # statement about a PAIR of panels, so it is attached at render time,
            # when the page knows whether there is a second arm at all, and it goes
            # on ARM A -- see `_mark_pr_identical_on_arm_a`.
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
                data_a["_arm_chip"] = _ab_arm_chip(ab_preset, "a")
                data_a["_arm_chip_value"] = _ab_arm_chip_overflow(ab_preset, "a")
                _meta_a = outputs_a.get("_axis_meta") or {}
                data_a["_run_identity"] = _run_identity_line(
                    ab_preset, _meta_a, n_clicks, n_steps)
                data_a["_cancelled_chip"] = (
                    f"CANCELLED -- {_meta_a.get('n_steps_run', '?')} of {n_steps} "
                    "frames; arm B did not run")
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
            data_a["_arm_chip"] = _ab_arm_chip(ab_preset, "a")
            data_a["_arm_chip_value"] = _ab_arm_chip_overflow(ab_preset, "a")
            data_b["_banner"] = f"{line_b}  ||  {result_b['banner']}"
            data_b["_ab"] = True
            data_b["_arm_chip"] = _ab_arm_chip(ab_preset, "b")
            data_b["_arm_chip_value"] = _ab_arm_chip_overflow(ab_preset, "b")
            _meta_a = outputs_a.get("_axis_meta") or {}
            data_a["_run_identity"] = _run_identity_line(ab_preset, _meta_a,
                                                         n_clicks, n_steps)
            if result_b["cancelled"] or _meta_a.get("cancelled"):
                data_a["_cancelled_chip"] = (
                    f"CANCELLED -- arm B ran "
                    f"{(outputs_b.get('_axis_meta') or {}).get('n_steps_run', '?')} "
                    f"of {n_steps} frames")
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
        _meta = outputs_a.get("_axis_meta") or {}
        data["_run_identity"] = _run_identity_line(matched_preset, _meta,
                                                   n_clicks, n_steps)
        if _meta.get("cancelled"):
            data["_cancelled_chip"] = (
                f"CANCELLED -- {_meta.get('n_steps_run', '?')} of {n_steps} frames")
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
    # Open the editor on the block whose knob the card says to turn, so the operator is
    # one click from the live demo; fall back to the tapped node. `focus` additionally
    # HOISTS that knob to the top of the column and chips it -- hostile round 11 D4: on
    # Thrusts 3 and 4 the A/B knob rendered below the container's bottom edge, so
    # "diagram, the knob and Run on the first screen" was not met.
    focus = None
    if preset.ab is not None:
        focus = (preset.ab[0], preset.ab[1])
    elif preset.live_knobs:
        focus = (preset.live_knobs[0][0], preset.live_knobs[0][1])
    if focus is not None:
        block_id = focus[0]
    else:
        data = node_data or {}
        block_id = data.get("block") or data.get("id") or PRODUCT_IDS[0]
    # The Results tab is cleared: its figures came from another preset, and the
    # before/after section would otherwise pair a Thrust 5 run with a Thrust 2 one.
    return (state, preset.n_steps, block_diagram.preset_notes(preset),
            block_diagram.param_editor(block_id, state, focus=focus),
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
    """See `webapp.pipeline_runner.decode_plotly_array`, which this delegates to --
    ONE authority for plotly's compact typed-array wire encoding, now that the
    colour-limit sharing in `pipeline_runner` needs the same decode this module's
    axis sharing has needed since the 2026-09-22 rehearsal. Kept as a name here
    because the tests that pinned that rehearsal regression call it."""
    return decode_plotly_array(v)


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
                # ...and unless the pair LOCKED its y-extent. A sensing map cropped to
                # its own unambiguous window (pipeline_runner's `_Y_EXTENT_LOCK`) has an
                # extent the A/B knob itself sets: with pilot spacing 2 vs 8 the windows
                # are 249.8 m and 62.4 m, and unioning them would redraw arm B's map on a
                # 249.8 m axis -- the very wrapped-copies picture the crop removes.
                # `note_differing_y_extents` then puts the pair on the screen in words.
                y_locked = any(y_extent_lock_of(fig) is not None for fig in pair)
                for fig in pair:
                    fig.setdefault("layout", {}).setdefault("xaxis", {})["range"] = xr
                    if not y_locked:
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


#: The `window.dash_clientside` namespace webapp/assets/results_clock.js registers,
#: and the function on it this callback drives. Named constants so the test that pins
#: the wiring reads the same two strings the app and the asset do.
RESULTS_CLOCK_NAMESPACE = "e2eResultsClock"
RESULTS_CLOCK_FUNCTION = "tick"

# Autoplay + loop, one clock (owner, live test 2026-09-24). Every animated figure
# inside #results-tab-content -- BOTH A/B arms, every product with Plotly frames --
# is stepped to the same frame index on every tick and wraps forever, starting by
# itself the moment the results render. The work is clientside because it is pure
# browser animation: a serverside callback would round-trip a full figure per frame
# over the link the owner is presenting across. See results_clock.js for the pause/
# resume handling (Plotly's own ▶/❚❚ buttons drive THIS clock instead of their own
# one-shot animation, so the two arms can never drift apart).
app.clientside_callback(
    ClientsideFunction(namespace=RESULTS_CLOCK_NAMESPACE,
                       function_name=RESULTS_CLOCK_FUNCTION),
    Output("results-clock-tick", "data"),
    Input("results-clock", "n_intervals"),
)


#: Ids the ONE transport control uses. `webapp/assets/results_clock.js` binds to these
#: three; they are named constants so the asset, the app and the test that pins the
#: wiring read the same strings.
TRANSPORT_TOGGLE_ID = "results-transport-toggle"
TRANSPORT_SLIDER_ID = "results-transport-slider"
TRANSPORT_LABEL_ID = "results-transport-label"


def _transport_bar():
    """ONE transport per screen, right-aligned in the run-identity row (layout spec
    section 4).

    Every animated figure on the page has been stepped by a single clock
    (`assets/results_clock.js`) since wave 11, so the per-figure `updatemenus`/`sliders`
    Plotly drew -- up to EIGHT copies on one screen, 130 px of bottom margin each --
    were decoration the presenter never touched (hostile round 10, defect 8;
    acceptance check 10). They are gone from the figures; this is the only one left.

    It also fixes the "dragging a slider parks ONE arm" defect (hostile round 10,
    section 3.3): there is one frame index now, so parking it parks both arms.

    Plain HTML rather than `dcc.Slider`: no Dash callback reads these, the clock
    asset drives them directly in the browser, and a server round-trip per frame over
    the link the owner presents across is exactly what the clientside clock exists to
    avoid."""
    return html.Div([
        html.Button("❚❚", id=TRANSPORT_TOGGLE_ID, n_clicks=0,
                    className="transport-btn", title="pause / play"),
        html.Span("frame 1", id=TRANSPORT_LABEL_ID, className="transport-label"),
        # `dcc.Input(type="range")`, not `dcc.Slider`: dash 4.x has no `html.Input`,
        # and a native range input is a DOM node `results_clock.js` can read and write
        # directly. `dcc.Slider` renders rc-slider, whose value lives in React state
        # the asset would have to reach through.
        dcc.Input(type="range", id=TRANSPORT_SLIDER_ID, min=1, max=1, step=1,
                  value=1, className="transport-slider"),
    ], className="transport")


def _cancelled_chip(axis_meta_like: str):
    """The amber `CANCELLED -- N of M frames` chip in the run-identity line."""
    return html.Span(axis_meta_like, className="chip chip-warn")


def _details_disclosure(label: str, lines: List[Any]):
    """The per-arm `Details` disclosure: closed by default, and labelled so a hostile
    reader knows the honesty text exists without opening it (layout spec section 2.3).

    Opening it pushes the panels down. That is correct: it is a deliberate act by the
    presenter, not the default state."""
    return html.Details([
        html.Summary(label, className="details-summary"),
        html.Div(lines, className="details-body"),
    ], open=False, className="details")


def _details_lines(payload: Dict[str, Any], figs: Dict[str, Any],
                   screen_note: str) -> List[Any]:
    """EVERYTHING this arm has to say, in one place: the full banner, every run note
    untruncated, the preset's screen note, and each panel's own provenance/band/clip
    clauses grouped under that panel's title.

    Nothing here is new text and nothing that used to be on screen is missing -- the
    disclosure is where the six-line panel subtitles, the arm banner, the run-notes
    lines and the screen note went (acceptance check 15)."""
    out: List[Any] = []
    banner = payload.get("_banner")
    if banner:
        out.append(html.Div(banner, className="details-line details-banner"))
    for note in (payload.get("_notes") or []):
        out.append(html.Div(note, className="details-line"))
    if screen_note:
        out.append(html.Div(screen_note, className="details-line"))
    for fig in figs.values():
        panel = panel_of(fig)
        details = panel.get("details") or []
        if not details:
            continue
        out.append(html.Div(panel.get("title") or "", className="details-heading"))
        for line in details:
            out.append(html.Div(line, className="details-line"))
    return out


#: What arm A's stored-PR panel says when BOTH arms are on screen.
PR_IDENTICAL_CLAUSE = "identical on both arms"


def _mark_pr_identical_on_arm_a(figs: Dict[str, Any], prev_figs: Dict[str, Any]) -> None:
    """Say ONCE, on arm A's caption, that the offline-scored PR panel is the same on
    both arms -- an A/B knob never touches it, and two panels showing the same numbers
    read as a bug until one of them says so (hostile round 10, section 5.8).

    ARM A, not arm B (hostile round 11, C6): the heat-map panels put their "same
    colour scale on both arms" clause on arm A, and two conventions for the same kind
    of statement on one screen make the ABSENCE of a clause carry meaning.

    At RENDER time, not in `_arm_result`: this is a fact about a PAIR of panels, and
    the single-run path builds arm A's figures with the same call -- a single-arm
    screen that printed "identical on both arms" would be describing a comparison it
    does not show (the defect C1 names on the cancel screen)."""
    key = "detector_pr_stored"
    if key not in figs or key not in prev_figs:
        return
    panel = _panel_dict(figs[key])
    caption = list(panel.get("caption") or [])
    if PR_IDENTICAL_CLAUSE not in caption:
        panel["caption"] = caption + [PR_IDENTICAL_CLAUSE]


#: The key of the offline-scored precision-recall panel, and the label of the
#: disclosure it now lives in.
PR_PANEL_KEY = "detector_pr_stored"


def offline_benchmark_label() -> str:
    """The summary text of the closed disclosure that holds the offline PR panel.

    ONE AUTHORITY, because the runbook has to tell the presenter what to CLICK and a
    second copy of this string is exactly what drifts (hostile round 13, N2: the runbook
    described the PR evidence as visible, so there was no click and it never showed).
    The split size is read from beat_cfar.json, never typed.

    It NAMES the curve now: "Offline benchmark" alone did not say the precision-recall
    evidence was inside, which is why a reader who wanted it did not open it.
    """
    label = "▸ Offline benchmark: precision-recall vs the classical baseline"
    try:
        from webapp import detector_scoreboard as _ds
        _, n_frames = _ds._load_recall_target_and_n_frames()
        if n_frames:
            label += f" ({int(n_frames)}-frame test split)"
    except Exception:
        pass
    return label


def _offline_benchmark_disclosure(figs: Dict[str, Any], prev_figs: Dict[str, Any]):
    """PULL the stored PR panel out of the product rows and render it ONCE, closed,
    below them -- and return that disclosure (or None).

    Why it moves, measured: the three Thrust 5 screens were 2480 px against the layout
    spec's 2200 (acceptance check 14), and that is arithmetic rather than a tweak --
    four product rows at the spec's own fixed heights (540 + 540 + 388 + 552) plus three
    16 px gaps plus a 303 px header IS 2480. One row had to go, and the PR panel is the
    one that is not this run's product: it is scored OFFLINE on a fixed test split, an
    A/B knob never touches it, so the two arms drew the SAME picture twice, and its
    headline numbers (AP per arm, the interval) are already rows in the scoreboard table
    beside it. Nothing is deleted -- the panel is one click away, with its own caption
    and Details, and the numbers stay on the default screen in the table.

    Mutates `figs`/`prev_figs`: the caller then builds its rows from what is left.
    """
    fig = figs.pop(PR_PANEL_KEY, None)
    prev_figs.pop(PR_PANEL_KEY, None)
    if fig is None:
        return None
    panel = panel_of(fig)
    label = offline_benchmark_label()
    return html.Details([
        html.Summary(label, className="details-summary"),
        html.Div([
            html.Div(panel.get("caption") and CAPTION_SEP.join(panel["caption"]) or "",
                     className="details-line"),
            _panel_block(fig, width=f"{SINGLE_PANEL_WIDTH}px"),
        ] + [html.Div(line, className="details-line")
             for line in (panel.get("details") or [])],
            className="details-body"),
    ], open=False, className="details offline-benchmark")


def _arm_header(payload: Dict[str, Any], figs: Dict[str, Any], *, arm: str,
                screen_note: str, fallback_label: str):
    """One arm's whole header block: the arm chip, its one-line caption, and its
    Details disclosure (layout spec section 2.3). Replaces the old bold, 2-4 line
    banner that wrapped across the A/B gutter and read as one garbled paragraph
    (hostile round 10, defect 2.6)."""
    return html.Div([
        _arm_summary(payload, arm=arm, fallback_label=fallback_label),
        _arm_details(payload, figs, screen_note=screen_note),
    ], className="arm-header")


def _arm_summary(payload: Dict[str, Any], *, arm: str, fallback_label: str):
    """The chip + one-line caption half of an arm header (see `_arm_header`)."""
    chip = payload.get("_arm_chip") or fallback_label
    caption = _arm_caption(payload)
    return html.Div([
        # The height cap that enforces "at most 4 lines and 150 px above the first
        # panel" belongs on the SUMMARY (chip + caption), never on the whole header:
        # capping the header clipped the OPENED Details to a 4 px sliver, i.e. the
        # honesty text was one click away and then invisible (found by reading
        # `--expand-details` PNG, 2026-09-24 -- no figure-dict test can see this).
        html.Div([html.Span(className=f"arm-dot arm-dot-{arm}"),
                  html.Span(chip, className="arm-chip-label")],
                 className=f"arm-chip arm-chip-{arm}"),
        html.Div(caption, className="arm-caption"),
    ], className="arm-summary")


def _arm_details(payload: Dict[str, Any], figs: Dict[str, Any], *, screen_note: str):
    """The Details disclosure half of an arm header (see `_arm_header`)."""
    return _details_disclosure("▸ Details (provenance, band, clip)",
                               _details_lines(payload, figs, screen_note))


@app.callback(
    Output("results-tab-content", "children"),
    Input("results-store", "data"),
    Input("tabs", "value"),
)
def _render_results(results_data, active_tab):
    """Render stored result figures at the fixed page geometry (layout spec sections
    2.1-2.3): a one-line run-identity row with the screen's ONE transport, a per-arm
    header (chip + one-line caption + collapsed Details), then one row per product with
    the panel's title and caption as HTML above the plot.

    The hard cap this enforces is "4 text lines and 150 px above the first panel". What
    used to sit there -- a 3-line screen note, a 2-4 line bold banner and 1-4 lines of
    run notes, 40-51 % of the first screen -- is all still reachable, in Details and in
    the page-foot note."""
    if active_tab != "tab-results":
        return no_update
    if not results_data:
        return html.Div([
            html.Div("Results", className="page-heading"),
            html.P("No results yet. Configure the pipeline on the Block Diagram "
                   "tab and click Run pipeline.", style={"color": "#576574"}),
        ], className="results-page")

    # Card order follows results_data's insertion order, which mirrors
    # figures_from_outputs' build order (webapp/pipeline_runner.py): fft ->
    # range_az -> range_el -> range_profile -> subspace_err -> comms products.
    figs = {k: v for k, v in results_data.items() if not k.startswith("_")}
    prev = results_data.get("_previous") or {}
    prev_figs = {k: v for k, v in prev.items() if not k.startswith("_")}
    # Colour limits FIRST, then axis extents: the limit pass rewrites each panel's clip
    # CAPTION clause (and appends the exact shared limits to its Details), while
    # `_share_y_ranges` only unions numbers -- running it second keeps its zmin/zmax
    # union a no-op over the already-equal pair rather than a second, weaker rule.
    if prev_figs:
        share_heatmap_z_limits(figs, prev_figs)
    else:
        # ONE ARM: the same floor-reaching rule, applied to this arm alone. Without it
        # a single-arm screen (the cancel journey, a preset with no B) kept the hard
        # -40 dB clip and drew a uniformly dark map of the same product the two-arm
        # screen draws at -61.3 dB (hostile round 12, item 8).
        reach_floor_single_arm(figs)
    _share_y_ranges(figs, prev_figs)
    # ...and then SAY which pairs `_share_y_ranges` deliberately left unshared, so the
    # one thing a photograph of two side-by-side maps cannot recover is on the screen.
    note_differing_y_extents(figs, prev_figs)
    # The arm's colour on its own statistic strip -- `figures_from_outputs` builds one
    # run's figures and has no idea which arm it will be shown as.
    apply_arm_style(figs, "a")
    if prev_figs:
        apply_arm_style(prev_figs, "b")
        # The scoreboard's two OFFLINE rows are knob-invariant, so they print once,
        # under arm A, and arm B keeps only the rows this run moved (H9). Done here
        # rather than in the figure builder for the same reason the arm colour is:
        # `figures_from_outputs` builds one run and does not know which arm it is.
        from webapp import detector_scoreboard as _ds
        for _key in set(figs) & set(prev_figs):
            if _key.endswith("_scoreboard"):
                _ds.fold_offline_rows_onto_arm_a(figs[_key], prev_figs[_key])
        _mark_pr_identical_on_arm_a(figs, prev_figs)

    # The offline PR panel leaves the product rows and becomes a closed disclosure at
    # the foot of the page (acceptance check 14 -- see `_offline_benchmark_disclosure`).
    # Done AFTER the sharing/marking passes above so the panel keeps every clause they
    # give it; it is rendered below, under the grid.
    offline_disclosure = _offline_benchmark_disclosure(figs, prev_figs)

    screen_note = results_data.get("_screen_note") or ""
    if screen_note and not prev_figs:
        # ONE arm on screen (a single-run preset, or the cancel journey's "arm B did
        # not run"): the preset's note is written for the two-panel case, so its
        # two-arm clauses describe a comparison this screen does not show (C1).
        screen_note = _one_arm_screen_note(screen_note)
    identity = results_data.get("_run_identity") or results_data.get("_banner") or ""
    cancelled = results_data.get("_cancelled_chip")

    identity_children = [html.Span(identity, className="run-identity")]
    if cancelled:
        identity_children.append(_cancelled_chip(cancelled))
    # A screen whose products carry no animation frames (Thrust 3: one line plot) gets
    # NO transport: a play button and a slider that do nothing are a control the
    # presenter can press and be ignored by, which is worse than their absence.
    animated = any((fig.get("frames") or []) for fig in list(figs.values())
                   + list(prev_figs.values()) if isinstance(fig, dict))
    header_row = html.Div(
        [html.Div(identity_children, className="run-identity-cell")]
        + ([_transport_bar()] if animated else []),
        className="run-identity-row")

    children = [header_row, html.Div(className="section-rule")]

    if not prev_figs:
        # ONE PRODUCT -> the Details disclosure moves into the column beside the panel
        # (spec 2.1; hostile round 11, D5). With more than one the grid is two-up and
        # that column does not exist, so the header keeps both halves as before.
        lone_product = len(figs) == 1
        children.append(_row(
            _arm_summary(results_data, arm="a", fallback_label="This run")
            if lone_product else
            _arm_header(results_data, figs, arm="a", screen_note=screen_note,
                        fallback_label="This run")))
        children.append(html.Div(className="section-rule"))
        children.append(_grid(
            figs,
            side_column=(_arm_details(results_data, figs, screen_note=screen_note)
                         if lone_product else None)))
    else:
        # SIDE BY SIDE (owner, live test 2026-09-24): "default should be side by side".
        # One row per product, arm A left, arm B right, headers above their own column.
        children.append(_ab_columns(
            _arm_header(results_data, figs, arm="a", screen_note=screen_note,
                        fallback_label="This run"),
            _arm_header(prev, prev_figs, arm="b", screen_note=screen_note,
                        fallback_label="Previous run (for before/after)"),
            figs, prev_figs))

    if offline_disclosure is not None:
        children.append(offline_disclosure)

    if screen_note:
        # The preset's own caveat, at the BOTTOM of the page (layout spec section 2.3):
        # a photograph of the screen still catches it; a viewer of the screen is no
        # longer made to read three lines of it before any data appears. It is also
        # inside both arms' Details, IN FULL, so nothing here is more than one click
        # away -- which is what lets the foot line be capped (see `_foot_note`).
        children.append(html.Div(_foot_note(screen_note),
                                 className="page-foot-note"))

    return html.Div(children, className="results-page")

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
