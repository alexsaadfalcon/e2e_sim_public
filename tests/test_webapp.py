"""Tests for the Dash web UI under ``webapp/``.

These verify the design contract spelled out in ``webapp/README.md``: the app
shell, block-diagram registry, and scenario editor all import and build WITHOUT
torch/Sionna and WITHOUT binding a network port. We test purely by importing
modules and calling their layout / helper functions -- never ``app.run`` /
``run_server`` and never opening a socket.

A real pipeline run (which needs torch + frames) is exercised only behind the
``gui``/``slow`` markers, which auto-skip unless RUN_GUI=1 / RUN_SLOW=1.

Shared fixtures come from tests/conftest.py (small_scenario, make_env_block,
tmp_pkl_frames, torch_device, n_freqs) and are not redefined here.
"""

import inspect
import subprocess
import sys
from pathlib import Path

import pytest

# dash/plotly/cytoscape are the only deps the shell needs; skip cleanly if absent.
pytest.importorskip("dash")
pytest.importorskip("dash_cytoscape")
pytest.importorskip("plotly")

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _import_without_torch(module_name):
    """Import ``module_name`` in a *fresh subprocess* and report whether it pulled
    in torch.

    Done out-of-process on purpose: deleting/reloading torch in the live test
    process leaves it half-initialized and breaks every later torch test. The
    subprocess exits 0 if the import succeeded WITHOUT importing torch, 3 if torch
    got imported, and non-zero/other on an import error (stderr captured).
    """
    code = (
        "import importlib, sys; "
        f"importlib.import_module({module_name!r}); "
        "sys.exit(0 if 'torch' not in sys.modules else 3)"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )


# =============================================================================
# pipeline_registry integrity (pure data; no torch)
# =============================================================================

def test_registry_imports_without_torch():
    """Importing the registry must not pull in torch (it is pure data)."""
    proc = _import_without_torch("webapp.pipeline_registry")
    assert proc.returncode == 0, (
        "importing webapp.pipeline_registry must succeed without importing torch "
        f"(rc={proc.returncode}); stderr:\n{proc.stderr}"
    )


def test_registry_block_ids_unique():
    from webapp.pipeline_registry import BLOCKS
    ids = [b.id for b in BLOCKS]
    assert len(ids) == len(set(ids)), f"duplicate block ids: {ids}"


def test_the_registry_no_longer_carries_a_second_answer_to_how_blocks_connect():
    """`EDGES` and `normalize_edge` are gone, deliberately (see the note in their place).

    They were a hand-maintained topology beside the real one, and by 2026-09-24 they
    disagreed with it: the list still ran ("quantizer", "radar_cube") with no range
    transform between them, and marked ("interconnect", "dechirp") as an "alt" path when
    the mixing block is on every chain. One authority per question -- the chain's order is
    the stage list the runner and `Simulation` build; the diagram draws a collapsed view of
    that. A future edit that reintroduces a parallel edge table should fail here first."""
    import webapp.pipeline_registry as reg

    assert not hasattr(reg, "EDGES")
    assert not hasattr(reg, "normalize_edge")


def test_registry_categories_are_known():
    from webapp.pipeline_registry import BLOCKS
    valid = {"source", "stage", "product"}
    for b in BLOCKS:
        assert b.category in valid, f"{b.id} has unknown category {b.category!r}"


def test_registry_param_defaults_match_kind():
    """Declared params have sane defaults/types consistent with their declared kind."""
    from webapp.pipeline_registry import BLOCKS
    for b in BLOCKS:
        pkeys = [p.key for p in b.params]
        assert len(pkeys) == len(set(pkeys)), f"{b.id} has duplicate param keys"
        for p in b.params:
            assert p.kind in {"number", "int", "choice", "text"}, (
                f"{b.id}.{p.key} has unknown kind {p.kind!r}"
            )
            assert p.default is not None, f"{b.id}.{p.key} has no default"
            if p.kind == "text":
                assert isinstance(p.default, str), (
                    f"{b.id}.{p.key} kind=text but default {p.default!r} is not a str"
                )
            if p.kind == "int":
                assert isinstance(p.default, int) and not isinstance(p.default, bool), (
                    f"{b.id}.{p.key} kind=int but default {p.default!r} is not an int"
                )
            elif p.kind == "number":
                assert isinstance(p.default, (int, float)) and not isinstance(p.default, bool), (
                    f"{b.id}.{p.key} kind=number but default {p.default!r} is not numeric"
                )
            elif p.kind == "choice":
                assert p.choices, f"{b.id}.{p.key} kind=choice but no choices listed"
                assert p.default in p.choices, (
                    f"{b.id}.{p.key} default {p.default!r} not among choices {p.choices}"
                )


def test_registry_subspace_k_has_min_one():
    """Subspace dim k=0 (or negative) crashes deep inside e2e.simulation's
    rank_diagnostic; the ParamSpec declares a floor so the rendered dcc.Input
    stops the user at the UI layer."""
    from webapp.pipeline_registry import BLOCKS_BY_ID
    k_spec = next(p for p in BLOCKS_BY_ID["subspace"].params if p.key == "k")
    assert k_spec.min == 1


def test_registry_subspace_k_has_max_below_m():
    """REGRESSION: k had a floor but no ceiling, so the GUI happily accepted k=512.
    AdaOjaBlock refuses k >= m, and it refuses at CONSTRUCTION -- which run_pipeline
    does outside its try/except -- so the ValueError bypassed every PipelineError
    handler and reached the user as a raw 'Unexpected error'. The bound must be tied
    to the same SUBSPACE_M the tracker is built with, so the two cannot drift."""
    from webapp.pipeline_registry import BLOCKS_BY_ID, SUBSPACE_M
    k_spec = next(p for p in BLOCKS_BY_ID["subspace"].params if p.key == "k")
    assert k_spec.max == SUBSPACE_M - 1


def test_subspace_m_is_the_value_the_tracker_is_actually_built_with(monkeypatch):
    """The UI's k ceiling is only honest if the tracker's real measurement count
    matches it. Checked BEHAVIOURALLY -- we capture the m that run_pipeline actually
    passes to AdaOjaBlock -- rather than by grepping the source for 'm=SUBSPACE_M',
    which a rename or a shadowing local could satisfy while the bug came back."""
    pytest.importorskip("torch")
    import e2e.blocks as blocks
    from webapp import pipeline_runner
    from webapp.pipeline_registry import default_block_state, SUBSPACE_M

    seen = {}
    real = blocks.AdaOjaBlock

    def spy(d, k, *args, **kwargs):
        seen["m"] = kwargs.get("m")
        seen["k"] = k
        return real(d, k, *args, **kwargs)

    monkeypatch.setattr(blocks, "AdaOjaBlock", spy)
    try:
        pipeline_runner.run_pipeline(default_block_state(), n_steps=1)
    except pipeline_runner.PipelineError:
        pass  # no frames on this machine is fine -- construction happens first

    assert seen.get("m") == SUBSPACE_M, (
        f"run_pipeline built the tracker with m={seen.get('m')}, but the UI bounds k "
        f"against SUBSPACE_M={SUBSPACE_M}; they have drifted apart"
    )
    assert seen["k"] < seen["m"], "the default k must satisfy the constraint it advertises"


def test_tessera_ka_scale_matches_the_block_it_presents_for():
    """`pipeline_registry._TESSERA_KA_SCALE` (a duplicate constant, kept local because
    `e2e.blocks` pulls in torch -- see that module's comment) must equal what
    `InterconnectBlock(source='tessera')` actually resolves for the pipeline's default
    Ka band, or the GUI's presented knob ranges silently drift from the model-space
    envelope the block itself validates against."""
    pytest.importorskip("torch")
    from e2e.blocks import _resolve_tessera_scale
    from webapp.pipeline_registry import _TESSERA_KA_SCALE

    # (28.5e9, 31.5e9): the munich frames' band, also what
    # `pipeline_runner._resolve_interconnect_band_hz` falls back to for a legacy pkl
    # (no `freq_plan`) at the registry's default 3 GHz rffe span.
    assert _resolve_tessera_scale(None, (28.5e9, 31.5e9)) == _TESSERA_KA_SCALE


def test_param_editor_k_input_has_min_and_max():
    from webapp import block_diagram
    from webapp.pipeline_registry import default_block_state, SUBSPACE_M
    from dash import dcc

    state = default_block_state()
    children = block_diagram.param_editor("subspace", state)
    k_inputs = [c for c in children
                if isinstance(c, dcc.Input) and c.id.get("param") == "k"]
    assert len(k_inputs) == 1
    assert k_inputs[0].min == 1
    assert k_inputs[0].max == SUBSPACE_M - 1


def test_default_block_state_covers_every_block():
    from webapp.pipeline_registry import BLOCKS, BLOCKS_BY_ID, default_block_state
    state = default_block_state()
    assert set(state) == {b.id for b in BLOCKS}
    for bid, st in state.items():
        spec = BLOCKS_BY_ID[bid]
        assert st["enabled"] == spec.enabled_default
        # every declared param appears with its default value
        assert set(st["params"]) == {p.key for p in spec.params}
        for p in spec.params:
            assert st["params"][p.key] == p.default


def test_product_and_serial_ids_partition_blocks():
    from webapp.pipeline_registry import BLOCKS, PRODUCT_IDS, SERIAL_IDS
    all_ids = {b.id for b in BLOCKS}
    assert set(PRODUCT_IDS).isdisjoint(SERIAL_IDS)
    assert set(PRODUCT_IDS) | set(SERIAL_IDS) == all_ids
    assert PRODUCT_IDS, "expected at least one product block"


# =============================================================================
# app shell: imports & builds a layout WITHOUT binding a port
# =============================================================================

def test_app_imports_and_builds_layout_without_server():
    import webapp.app as appmod

    # The Dash app object exists and exposes a WSGI server, but nothing has been
    # run/served at import time.
    assert appmod.app is not None
    assert appmod.app.layout is not None

    # The layout in this app is assigned as a callable (lazy). Resolve it to a
    # concrete component tree and confirm it is non-empty.
    layout = appmod.app.layout
    component = layout() if callable(layout) else layout
    assert component is not None
    # Dash components carry a `.children` attribute; the top div should have some.
    assert getattr(component, "children", None) is not None


def test_app_module_does_not_call_run_at_import():
    """No server is started at import time: app.run is only reachable via main()."""
    import webapp.app as appmod

    # The run entrypoint lives in main() / the __main__ guard, never at top level.
    src = inspect.getsource(appmod)
    main_src = inspect.getsource(appmod.main)
    assert "app.run(" in main_src, "expected main() to be the only place app.run is called"
    # app.run must not be invoked at module top-level. A top-level statement is
    # un-indented (column 0); anything inside main()/the __main__ guard is indented.
    for line in src.splitlines():
        if line.startswith("app.run(") or line.startswith("server.run("):
            pytest.fail("server appears to be started at module top level")


def test_app_imports_without_torch():
    """The shell imports cleanly even if torch is not available."""
    proc = _import_without_torch("webapp.app")
    assert proc.returncode == 0, (
        "webapp.app must import without importing torch "
        f"(rc={proc.returncode}); stderr:\n{proc.stderr}"
    )


# =============================================================================
# block_diagram: elements builder + param editor
# =============================================================================

def test_the_diagram_draws_ONE_CHAIN_and_every_block_is_reachable_on_it():
    """THE one-chain test (owner 2026-09-24: "there are two pipelines ... needs to be
    fixed immediately from the ground up"; contract section 4).

    Four properties, each of which the previous diagram violated:
      * no compound REGION groups -- four labelled boxes around a one-path graph are four
        statements that it has four paths;
      * no "alt" edges -- the dotted salmon "alternative source path" strokes were the
        loudest lines in the diagram and the least important;
      * the chain nodes form ONE simple path, in the spine's order, with no node having
        two chain predecessors (the waveform CLASS is the only branch, and it is a choice
        inside one block, not a fork in the graph);
      * every product hangs off exactly one chain node -- the point whose domain it reads.
    And the property that keeps the collapse honest: every registry block is reachable
    through some node's editor, so collapsing boxes did not hide a knob.
    """
    from webapp import block_diagram
    from webapp.pipeline_registry import BLOCKS, default_block_state

    state = default_block_state()
    elements = block_diagram.build_elements(state)
    nodes = [e for e in elements if "position" in e]
    edges = [e for e in elements if "source" in e["data"]]

    assert not hasattr(block_diagram, "_GROUPS"), "compound region groups are gone"
    assert all("group" not in (e.get("classes") or "") for e in elements)
    assert all("alt-path" not in (e.get("classes") or "") for e in edges)
    assert not any(sel["selector"] == "edge.alt-path"
                   for sel in block_diagram.CYTO_STYLESHEET)

    node_ids = {n["data"]["id"] for n in nodes}
    chain_ids = [nid for nid, _l, _c, _m in block_diagram._CHAIN]
    assert set(chain_ids) <= node_ids

    chain_edges = [(e["data"]["source"], e["data"]["target"]) for e in edges
                   if e["data"]["source"] in chain_ids and e["data"]["target"] in chain_ids]
    assert chain_edges == list(zip(chain_ids, chain_ids[1:])), (
        "the chain must be one simple path in the spine's order")

    for pid, tap in block_diagram._PRODUCT_TAP.items():
        assert pid in node_ids
        taps = [s for s, t in ((e["data"]["source"], e["data"]["target"]) for e in edges)
                if t == pid]
        assert taps == [tap], f"{pid} must tap exactly {tap!r}, got {taps}"

    reachable = {bid for nid in node_ids
                 for bid in block_diagram._NODE_MEMBERS.get(nid, [])}
    # The source node resolves to ONE backend at render time, but all three are members.
    assert {b.id for b in BLOCKS} <= reachable, (
        "a registry block that no diagram node stands for is a knob with no way to reach it")

    for e in edges:
        d = e["data"]
        assert d["source"] in node_ids and d["target"] in node_ids
        assert d["id"] == f"{d['source']}->{d['target']}"


def test_build_elements_marks_disabled_block():
    from webapp import block_diagram
    from webapp.pipeline_registry import default_block_state

    state = default_block_state()
    # interconnect is toggleable and off by default -> should be marked disabled
    elements = block_diagram.build_elements(state)
    by_id = {e["data"].get("id"): e for e in elements if "source" not in e["data"]}
    assert "disabled" in by_id["interconnect"]["classes"]

    # turning it on removes the disabled class
    state["interconnect"]["enabled"] = True
    elements = block_diagram.build_elements(state)
    by_id = {e["data"].get("id"): e for e in elements if "source" not in e["data"]}
    assert "disabled" not in by_id["interconnect"]["classes"]


def test_param_editor_runs_for_sample_blocks():
    from webapp import block_diagram
    from webapp.pipeline_registry import default_block_state

    state = default_block_state()
    # a block with params + toggle
    children = block_diagram.param_editor("rffe", state)
    assert isinstance(children, list) and children

    # a structural/no-param block also builds
    children_err = block_diagram.param_editor("subspace_err", state)
    assert isinstance(children_err, list) and children_err


def test_param_editor_unknown_block_is_graceful():
    from webapp import block_diagram
    children = block_diagram.param_editor("does-not-exist", {})
    assert isinstance(children, list) and children  # returns a "select a block" prompt


def _find_checklist(children, block=None):
    """Dig a dcc.Checklist out of a param_editor children list.

    The editor now renders every registry block a DIAGRAM NODE stands for (the one-chain
    diagram collapses the three sources, the transmit tributary and the ADC sub-chain), so
    a node's column can hold several enable checkboxes. `block=` names which one; with no
    `block` there must still be exactly one.
    """
    from dash import dcc
    matches = [c for c in children if isinstance(c, dcc.Checklist)]
    if block is not None:
        matches = [c for c in matches if (c.id or {}).get("block") == block]
    assert len(matches) == 1, f"expected one checklist (block={block!r}), got {len(matches)}"
    return matches[0]


def test_param_editor_subspace_checkbox_disabled_while_afe_enabled():
    """pipeline_runner always builds the AdaOja tracker whenever AFE is enabled
    (dead no-AFE-without-subspace guard notwithstanding), so the subspace 'Enabled'
    checkbox toggling it off is a no-op. The honest-UI fix: disable the checkbox
    (checked/locked on) and explain why, instead of letting the user believe it
    does something."""
    from webapp import block_diagram
    from webapp.pipeline_registry import default_block_state

    state = default_block_state()
    assert state["afe"]["enabled"] is True  # afe is on by default
    children = block_diagram.param_editor("subspace", state)

    checklist = _find_checklist(children, block="subspace")
    assert checklist.options[0]["disabled"] is True
    assert checklist.value == ["on"]  # locked "on" regardless of stored state
    caption_text = " ".join(
        c.children for c in children
        if hasattr(c, "children") and isinstance(c.children, str)
    )
    assert "always on while AFE is enabled" in caption_text


def test_param_editor_subspace_checkbox_normal_when_afe_disabled():
    """With AFE off, the subspace toggle is a real, honored control again."""
    from webapp import block_diagram
    from webapp.pipeline_registry import default_block_state

    state = default_block_state()
    state["afe"]["enabled"] = False
    state["subspace"]["enabled"] = False
    children = block_diagram.param_editor("subspace", state)

    checklist = _find_checklist(children, block="subspace")
    assert not checklist.options[0].get("disabled")
    assert checklist.value == []  # honors the stored (off) state


def test_block_diagram_layout_builds():
    from webapp import block_diagram
    layout = block_diagram.layout()
    assert layout is not None
    assert getattr(layout, "children", None) is not None


def test_positions_cover_every_diagram_node_without_collisions():
    """Every DIAGRAM NODE must have an explicit position -- otherwise it silently falls
    back to (0, 0) and overlaps another node -- AND no two nodes' rendered boxes may
    geometrically overlap. Tuple-equality alone is not enough: this is the regression that
    let range_profile (1020, 270) sit on top of quantizer (1000, 280) -- different tuples,
    overlapping boxes. (Positions are per NODE since the one-chain rewrite; several
    registry blocks share a node, and `_NODE_MEMBERS` is what keeps them reachable --
    tested in test_the_diagram_draws_ONE_CHAIN_...)"""
    from webapp import block_diagram

    ids = list(block_diagram._NODE_MEMBERS)
    missing = [bid for bid in ids if bid not in block_diagram._POSITIONS]
    assert not missing, f"nodes missing explicit positions: {missing}"

    # Node box size from CYTO_STYLESHEET's base "node" selector (width/height).
    node_style = next(s["style"] for s in block_diagram.CYTO_STYLESHEET
                       if s["selector"] == "node")
    w = float(node_style["width"].rstrip("px"))
    h = float(node_style["height"].rstrip("px"))

    def _box(bid):
        x, y = block_diagram._POSITIONS[bid]
        return (x - w / 2, x + w / 2, y - h / 2, y + h / 2)

    def _overlaps(a, b):
        ax0, ax1, ay0, ay1 = a
        bx0, bx1, by0, by1 = b
        return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1

    boxes = {bid: _box(bid) for bid in ids}
    colliding = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if _overlaps(boxes[a], boxes[b]):
                colliding.append((a, b))
    assert not colliding, f"node boxes ({w}x{h}) geometrically overlap: {colliding}"


def test_no_product_edge_crosses_a_node_it_does_not_touch():
    """SEAT'S READ OF THE 2026-09-24 RENDERS, item 1b: six of the nine products tap the
    CUBE, which is the last chain node, so with the products in two rows under the whole
    chain every one of those edges swept back across the row. Not merely untidy --
    measured on the old positions, the straight `cube -> fft` segment passed through the
    ADC node's own box, so the fan crossed the boxes it was drawn to explain.

    This models the geometry the stylesheet actually draws: chain edges are straight
    (same y, adjacent columns), product edges are TAXI (orthogonal) with `taxi-turn: 50%`
    -- trunk halfway between the two boxes, then a spur into the target. It asserts no
    segment of any edge enters a node box other than its own two endpoints'. A straight
    fan into a stacked column cannot pass this (the ray to the bottom product clips the
    left half of the boxes above it), which is why the routing is taxi and why this test
    is geometric rather than a list of expected positions.
    """
    from webapp import block_diagram
    from webapp.pipeline_registry import default_block_state

    node_style = next(st["style"] for st in block_diagram.CYTO_STYLESHEET
                      if st["selector"] == "node")
    w = float(node_style["width"].rstrip("px"))
    h = float(node_style["height"].rstrip("px"))
    pos = block_diagram._POSITIONS

    def box(nid):
        x, y = pos[nid]
        return (x - w / 2, x + w / 2, y - h / 2, y + h / 2)

    def segments(src, dst, taxi):
        """The polyline cytoscape draws, from source border to target border."""
        sx, sy = pos[src]
        tx, ty = pos[dst]
        if taxi == "tap-right":
            x0 = sx + w / 2
            x1 = tx - w / 2
            trunk = (x0 + x1) / 2.0
            return [((x0, sy), (trunk, sy)), ((trunk, sy), (trunk, ty)),
                    ((trunk, ty), (x1, ty))]
        if taxi == "tap-down":
            y0 = sy + h / 2
            y1 = ty - h / 2
            trunk = (y0 + y1) / 2.0
            return [((sx, y0), (sx, trunk)), ((sx, trunk), (tx, trunk)),
                    ((tx, trunk), (tx, y1))]
        return [((sx, sy), (tx, ty))]          # straight (chain edges)

    def hits(seg, b):
        """Does an axis-aligned OR diagonal segment intersect box `b`? (Liang-Barsky.)"""
        (x0, y0), (x1, y1) = seg
        bx0, bx1, by0, by1 = b
        dx, dy = x1 - x0, y1 - y0
        t0, t1 = 0.0, 1.0
        for p, q in ((-dx, x0 - bx0), (dx, bx1 - x0), (-dy, y0 - by0), (dy, by1 - y0)):
            if p == 0:
                if q < 0:
                    return False
                continue
            r = q / p
            if p < 0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
            if t0 > t1:
                return False
        return True

    elements = block_diagram.build_elements(default_block_state())
    edges = [e for e in elements if "source" in e["data"]]
    assert edges
    offences = []
    for e in edges:
        src, dst = e["data"]["source"], e["data"]["target"]
        classes = (e.get("classes") or "").split()
        taxi = next((c for c in classes if c.startswith("tap-")), None)
        for seg in segments(src, dst, taxi):
            for nid in pos:
                if nid in (src, dst):
                    continue
                if hits(seg, box(nid)):
                    offences.append((f"{src}->{dst}", nid))
    assert not offences, f"edges crossing a node box they do not touch: {sorted(set(offences))}"


def test_the_front_end_node_says_where_the_computation_applies_the_cascade():
    """SEAT'S READ OF THE 2026-09-24 RENDERS, item 1a. The diagram draws the PHYSICAL
    order (front end at the element, then the interconnect, then the mixing block) while
    the code applies the front-end cascade to the sampled BEAT record after the mixing
    block. A diagram that shows one order while the numbers come from the other is only
    honest if the node itself says so, so the sentence is part of the block's own blurb
    and is rendered in the editor column beside the diagram -- not in a comment."""
    from webapp import block_diagram
    from webapp.pipeline_registry import BLOCKS_BY_ID, default_block_state

    rffe = BLOCKS_BY_ID["rffe"].blurb.lower()
    assert "beat" in rffe and "after the mixing block" in rffe, (
        "the front-end block must say the cascade is applied to the beat record after "
        f"the mixing block; blurb reads: {rffe!r}")
    dechirp = BLOCKS_BY_ID["dechirp"].blurb.lower()
    assert "mixer" in dechirp, "the mixing node must say it is the front end's own mixer"

    # And it reaches the screen: the editor column for the front-end node renders it.
    def flat(component):
        if isinstance(component, str):
            return component
        kids = getattr(component, "children", None)
        if isinstance(kids, (list, tuple)):
            return " ".join(flat(c) for c in kids if c is not None)
        return flat(kids) if kids is not None else ""

    text = " ".join(flat(c) for c in
                    block_diagram.param_editor("rffe", default_block_state()))
    assert "beat record" in text.lower()


def test_no_help_or_caption_string_quotes_a_hand_PICKED_drive_LEVEL():
    """SEAT'S READ OF THE 2026-09-24 RENDERS, item 1c: the LNA-bias help still read
    "visible only when the signal sits near the front-end's own noise (signal scaling
    ~1e-7)" on a screen whose preset drives at 3e-5 -- a level that stopped being true
    when the front end moved onto the beat record, printed beside the setting that
    replaced it. A static registry string cannot know a preset's drive, so it may not
    quote one: either the screen computes it from the run, or it is not stated."""
    from webapp.pipeline_registry import BLOCKS

    import re
    # A drive level is a small power of ten in the signal-scaling range; the registry's
    # own legitimate numbers (1e-6 s chirp, 1e-7 s step, 3e9 Hz) are not in it.
    pattern = re.compile(r"(?<![\w.])[0-9](?:\.[0-9]+)?e-0?[5-9](?![\w])")
    offences = []
    for spec in BLOCKS:
        for text, where in ([(spec.blurb, f"{spec.id}.blurb")]
                            + [(ps.help, f"{spec.id}.{ps.key}.help") for ps in spec.params]):
            for hit in pattern.findall(text or ""):
                offences.append((where, hit))
    assert not offences, (
        "help/blurb strings quoting an absolute drive level (must be computed from the "
        f"run or deleted): {offences}")


def test_the_diagram_extent_leaves_enough_fit_zoom_for_legible_labels():
    """HOSTILE ROUND 11, D2 / acceptance check 17: node labels must render >= 12 px of
    INK after fit zoom, and they measured ~6 px.

    The canvas fits the whole graph, so the extent's WIDTH sets the zoom and the zoom
    times the font size is what the room sees. This pins the arithmetic the fix rests on,
    in one place, so that adding a chain column or a wider label cannot quietly halve the
    label ink again: the check that matters is still a measurement on a rendered PNG, and
    this is the cheap upstream guard that says which change broke it.

    (The two tests that used to live here checked that the four compound REGION boxes did
    not overlap each other. There are no regions any more -- see
    test_the_diagram_draws_ONE_CHAIN_and_every_block_is_reachable_on_it.)
    """
    from webapp import block_diagram

    node_style = next(s["style"] for s in block_diagram.CYTO_STYLESHEET
                      if s["selector"] == "node")
    w = float(node_style["width"].rstrip("px"))
    h = float(node_style["height"].rstrip("px"))
    font = float(str(node_style["font-size"]).rstrip("px"))

    xs = [x for x, _y in block_diagram._POSITIONS.values()]
    ys = [y for _x, y in block_diagram._POSITIONS.values()]
    extent_w = (max(xs) - min(xs)) + w
    extent_h = (max(ys) - min(ys)) + h

    # The panel the cytoscape canvas sits in, MEASURED in the browser on the Thrust 1
    # card (2026-09-24 23:0x): `#block-cytoscape` renders 996 x 620 at the 1600 px
    # viewport -- the flex basis is 1032 px but the container's border and padding come
    # off it, so 1032 flattered the zoom by 4 %. Cytoscape's own layout padding is 20 px
    # a side on top of that.
    panel_w, panel_h, pad = 996.0, 620.0, 20.0
    zoom = min((panel_w - 2 * pad) / extent_w, (panel_h - 2 * pad) / extent_h, 1.0)
    # Cap-height / em for the default sans stack. MEASURED, not assumed: the rendered
    # card of 2026-09-24 21:0x drew "FMCW" with row-runs of 17 and 21 px of ink at a
    # 0.813 fit zoom and a 30 px font -> 17 / (30 * 0.813) = 0.70. (Hostile round 11 read
    # ~6 px at 20 px/~0.6, i.e. ~0.5, from a crop of a wrapped two-line label where the
    # rows it sampled were the x-height of "Sionna"/"Environment" rather than a cap; 0.70
    # is the conservative-to-optimistic correction and this test is the upstream guard --
    # the check that decides is still a measurement on a fresh render.)
    ink = font * zoom * 0.70
    assert ink >= 12.0, (
        f"label ink {ink:.1f} px at fit zoom {zoom:.2f} (extent {extent_w:.0f}x"
        f"{extent_h:.0f}, font {font:.0f}px) -- acceptance check 17 wants >= 12 px")


def test_every_registry_block_belongs_to_exactly_one_diagram_node():
    """Every registered block must appear in exactly one diagram node's members -- not
    zero (unreachable knob), not two (two boxes claiming the same block)."""
    from webapp import block_diagram
    from webapp.pipeline_registry import BLOCKS

    count = {b.id: 0 for b in BLOCKS}
    for nid, members in block_diagram._NODE_MEMBERS.items():
        for bid in members:
            assert bid in count, f"node member {bid!r} is not a registered block"
            count[bid] += 1

    not_placed = [bid for bid, n in count.items() if n == 0]
    multi = [bid for bid, n in count.items() if n > 1]
    assert not not_placed, f"blocks no diagram node stands for: {not_placed}"
    assert not multi, f"blocks claimed by more than one node: {multi}"


# =============================================================================
# scenario_editor: JSON round-trip, map figure, validation
# =============================================================================

def test_reference_scenarios_round_trip_through_editor_helpers():
    from e2e.scenario import REFERENCE_SCENARIOS
    from webapp.scenario_editor import scenario_from_json_safe

    for name, factory in REFERENCE_SCENARIOS.items():
        sc = factory()
        text = sc.to_json()
        parsed, err = scenario_from_json_safe(text)
        assert err is None, f"{name}: unexpected parse error {err}"
        assert parsed is not None
        # round-trip preserves the full structure
        assert parsed.to_dict() == sc.to_dict(), f"{name} did not round-trip"


def test_scenario_from_json_safe_reports_error_not_crash():
    from webapp.scenario_editor import scenario_from_json_safe
    sc, err = scenario_from_json_safe("this is not json {")
    assert sc is None
    assert isinstance(err, str) and err  # a human-readable message, no exception


def test_default_scenario_json_is_a_reference():
    from webapp.scenario_editor import default_scenario_json, scenario_from_json_safe
    sc, err = scenario_from_json_safe(default_scenario_json())
    assert err is None and sc is not None


def test_map_figure_returns_plotly_figure(small_scenario):
    import plotly.graph_objects as go
    from webapp.scenario_editor import map_figure
    fig = map_figure(small_scenario)
    assert isinstance(fig, go.Figure)
    # at least one trace (the radar node) was placed
    assert len(fig.data) >= 1


def test_summarize_returns_component(small_scenario):
    from webapp.scenario_editor import summarize
    table = summarize(small_scenario)
    assert table is not None
    assert getattr(table, "children", None) is not None


def test_validate_surfaces_problems_for_invalid_scenario():
    from e2e.scenario import Scenario
    # empty scenario: no nodes -> validate() must report a problem
    sc = Scenario(name="empty")
    problems = sc.validate()
    assert problems, "expected validate() to flag a node-less scenario"
    assert any("no nodes" in p for p in problems)


def test_validate_clean_for_reference_scenario(small_scenario):
    assert small_scenario.validate() == []


def test_scenario_editor_layout_builds():
    from webapp import scenario_editor
    layout = scenario_editor.layout()
    assert layout is not None
    assert getattr(layout, "children", None) is not None


# =============================================================================
# pipeline_runner: lazy torch import + clean error paths
# =============================================================================

def test_pipeline_runner_does_not_import_torch_at_module_top():
    """pipeline_runner keeps torch lazy: importing it must not import torch."""
    proc = _import_without_torch("webapp.pipeline_runner")
    assert proc.returncode == 0, (
        "webapp.pipeline_runner must import torch lazily, not at module top "
        f"(rc={proc.returncode}); stderr:\n{proc.stderr}"
    )

    # And torch must appear inside run_pipeline's source, confirming it is lazy.
    from webapp import pipeline_runner
    run_src = inspect.getsource(pipeline_runner.run_pipeline)
    assert "import torch" in run_src


def _stale_guard_fired(exc) -> bool:
    """True iff the (now-removed) RFFE/AFE compatibility guard rejected the run.

    The old guard raised a PipelineError up front whose message named RFFE/AFE and
    told the user to "Enable ... then Run again". The backend now supports RFFE-off
    and AFE-off, so this must never fire. We match on the distinctive guard wording
    rather than just "RFFE"/"AFE" so an unrelated later failure that happens to
    mention those acronyms does not look like the stale guard.
    """
    msg = str(exc)
    return "then Run again" in msg and ("RFFE" in msg or "AFE" in msg)


def test_run_pipeline_maps_frame_contract_errors_to_constraint_message(
        monkeypatch, make_env_block):
    """A frame violating the shape contract (here: MIMO, n_tx=2) must surface as the
    friendly 'Pipeline constraint failed: ...' PipelineError, not the generic
    'Pipeline run failed'. The contract guards moved from bare asserts onto
    e2e.frames.FrameContractError; this pins the webapp mapping for the new type."""
    torch = pytest.importorskip("torch")
    from webapp import pipeline_runner
    from webapp.pipeline_registry import default_block_state
    import e2e.blocks as blocks

    env = make_env_block(n_frames=1, n_freqs=16)
    mimo = torch.cat([env.get_S_pars()] * 2, dim=1)  # [n_rx, 2, 1, F]
    env.get_S_pars = lambda: mimo
    monkeypatch.setattr(blocks, "SionnaEnvironmentBlock", lambda *a, **k: env)

    state = default_block_state()
    state["rffe"]["enabled"] = False  # fail at the shape guard, not inside RFFE
    with pytest.raises(pipeline_runner.PipelineError, match=r"Pipeline constraint failed.*MIMO"):
        pipeline_runner.run_pipeline(state, n_steps=1)


def test_run_pipeline_wires_every_classic_product_block(monkeypatch, make_env_block):
    """Regression for the range_profile gap: a block registered as a non-toggleable
    'product' in pipeline_registry MUST actually execute in run_pipeline and MUST get
    a figure from figures_from_outputs -- otherwise the UI presents a permanently-on
    node that silently never runs. Guards the classic-chain products; the ADC-cube
    chain trio (radar_cube/detector/sink) is registered-but-unwired by documented
    design (see pipeline_runner's module docstring) and excluded here."""
    pytest.importorskip("torch")
    from webapp import pipeline_runner
    from webapp.pipeline_registry import default_block_state
    import e2e.blocks as blocks

    env = make_env_block(n_frames=2, n_freqs=16)
    monkeypatch.setattr(blocks, "SionnaEnvironmentBlock", lambda *a, **k: env)

    outputs = pipeline_runner.run_pipeline(default_block_state(), n_steps=1)
    for key in ("fft", "range_az", "range_el", "range_profile_agg", "subspace_err"):
        assert outputs.get(key), f"classic product output {key!r} missing from run_pipeline"

    figs = pipeline_runner.figures_from_outputs(outputs)
    for fig_key in ("fft", "range_az", "range_el", "range_profile", "subspace_err"):
        assert fig_key in figs, f"figures_from_outputs has no figure for {fig_key!r}"


@pytest.mark.parametrize("k", [512, 600])
def test_run_pipeline_rejects_k_at_or_above_m_as_a_pipeline_error(k):
    """REGRESSION: the dcc.Input `max` is only a browser hint -- a typed value or a
    saved state can still carry k >= m. run_pipeline must reject it as a friendly
    PipelineError (which webapp/app.py renders in the diagram view) rather than
    letting AdaOjaBlock's construction-time ValueError escape. Deliberately checked
    WITHOUT frames or monkeypatching: the guard has to fire before any heavy work,
    which is also why a sponsor never waits 30 s to see this message."""
    from webapp.pipeline_runner import PipelineError, run_pipeline
    from webapp.pipeline_registry import default_block_state

    state = default_block_state()
    state["subspace"]["params"]["k"] = k
    with pytest.raises(PipelineError, match=r"Subspace dim k must be <"):
        run_pipeline(state, n_steps=1)


def test_run_pipeline_still_accepts_the_largest_valid_k():
    """The ceiling must be off-by-one correct: k = m-1 is legal (AdaOjaBlock only
    refuses k >= m), so the guard must not fire there."""
    from webapp.pipeline_runner import PipelineError, run_pipeline
    from webapp.pipeline_registry import default_block_state, SUBSPACE_M

    state = default_block_state()
    state["subspace"]["params"]["k"] = SUBSPACE_M - 1
    try:
        run_pipeline(state, n_steps=1)
    except PipelineError as e:
        assert "Subspace dim k must be <" not in str(e), (
            f"k={SUBSPACE_M - 1} is valid but the ceiling guard rejected it: {e}"
        )


def test_run_pipeline_no_longer_errors_when_rffe_disabled():
    """RFFE-off is now a valid config: the stale up-front RFFE guard must not fire.

    The backend initializes PRX to None, so a no-RFFE run is supported. We don't run
    a full pipeline here (that needs torch + frames); we only assert the run gets
    PAST the removed guard. With no frames it fails later (e.g. missing .pkl), but
    that failure must not be the old "enable RFFE ... then Run again" guard.
    """
    from webapp.pipeline_runner import PipelineError, run_pipeline
    from webapp.pipeline_registry import default_block_state

    state = default_block_state()
    state["rffe"]["enabled"] = False
    try:
        run_pipeline(state, n_steps=1)
    except PipelineError as e:
        assert not _stale_guard_fired(e), f"stale RFFE guard still fires: {e}"
    # No exception (e.g. torch + frames available) is also fine: the guard is gone.


def test_run_pipeline_no_longer_errors_when_afe_disabled():
    """AFE-off is now a valid config: the stale up-front AFE guard must not fire.

    The backend's no-AFE branch calls subspace.update(X, A) with two args, so a
    no-AFE run is supported. As above, we only assert we get past the removed guard;
    a later failure on missing frames is acceptable, the stale AFE guard is not.
    """
    from webapp.pipeline_runner import PipelineError, run_pipeline
    from webapp.pipeline_registry import default_block_state

    state = default_block_state()
    state["afe"]["enabled"] = False
    try:
        run_pipeline(state, n_steps=1)
    except PipelineError as e:
        assert not _stale_guard_fired(e), f"stale AFE guard still fires: {e}"


def test_run_pipeline_still_requires_subspace_when_afe_enabled():
    """A genuinely-still-real guard: AFE on requires the AdaOja Subspace block.

    feed_forward dereferences subspace_block.oja.U unconditionally and the backend
    raises if an AFE block is paired without a subspace block, so the runner must
    keep enforcing this. Asserted without torch/frames by checking the up-front
    guard message; gated behind no markers since it should fire before any heavy
    work. If torch/frames are unavailable the import/environment path may raise a
    different PipelineError first -- in that case we simply don't assert the
    subspace message (the contract is enforced in pipeline_runner regardless).
    """
    from webapp.pipeline_runner import PipelineError, run_pipeline
    from webapp.pipeline_registry import default_block_state

    state = default_block_state()
    state["afe"]["enabled"] = True
    state["subspace"]["enabled"] = False
    # The runner constructs a subspace block whenever AFE is on, so this combo is
    # reconciled rather than rejected -- it must NOT raise the stale RFFE/AFE guard.
    try:
        run_pipeline(state, n_steps=1)
    except PipelineError as e:
        assert not _stale_guard_fired(e), f"stale guard still fires: {e}"


def test_run_pipeline_k_zero_surfaces_friendly_message(monkeypatch, make_env_block):
    """A subspace k < 1 crashes deep inside e2e.simulation.rank_diagnostic with
    ValueError('rank_diagnostic requires k >= 1, got k=0'); the runner must
    translate that into an actionable message instead of leaking the internal
    function name."""
    torch = pytest.importorskip("torch")
    from webapp import pipeline_runner
    from webapp.pipeline_registry import default_block_state
    import e2e.blocks as blocks

    env = make_env_block(n_frames=1, n_freqs=16)
    monkeypatch.setattr(blocks, "SionnaEnvironmentBlock", lambda *a, **k: env)

    state = default_block_state()
    state["subspace"]["params"]["k"] = 0
    with pytest.raises(pipeline_runner.PipelineError, match=r"Subspace dim k must be >= 1"):
        pipeline_runner.run_pipeline(state, n_steps=1)


def test_figures_from_outputs_handles_empty_outputs():
    """The figure builder tolerates an empty/partial outputs dict without torch."""
    from webapp.pipeline_runner import figures_from_outputs
    assert figures_from_outputs({}) == {}


def test_figures_from_outputs_labels_axes_with_physical_units():
    """Result heatmaps carry physical axes (sin(theta) aperture / range), not raw
    bin indices, when run_pipeline's axis metadata is present and self-consistent
    (range-FFT bins == the raw frame's frequency-sample count)."""
    torch = pytest.importorskip("torch")
    import numpy as np
    from webapp.pipeline_runner import figures_from_outputs

    bins = 8
    outputs = {
        "fft": [torch.zeros((bins, bins), dtype=torch.complex64)],
        "range_az": [torch.zeros((bins, bins), dtype=torch.complex64)],
        "range_el": [torch.zeros((bins, bins), dtype=torch.complex64)],
        "_axis_meta": {
            "fft_bins": bins,
            "range_az_bins": bins,
            "range_el_bins": bins,
            "n_freqs": bins,  # matches range bins -> exact range mapping applies
            "freq_span_hz": 3e9,
        },
    }
    figs = figures_from_outputs(outputs)

    expected_u = (np.arange(bins) - bins // 2) / (bins / 2)
    np.testing.assert_allclose(figs["fft"].data[0].x, expected_u)
    np.testing.assert_allclose(figs["fft"].data[0].y, expected_u)
    assert figs["fft"].layout.xaxis.title.text == "azimuth sin(θ)"
    assert figs["fft"].layout.yaxis.title.text == "elevation sin(θ)"
    # Peak-relative WITH a stated clip; a bare "power (dB)" reads as absolute dB.
    # The colour bar itself carries no title any more (layout spec, 2026-09-24,
    # section 4): at 20 px "dB rel. peak (clipped at -67.1)" was ~310 px wide and
    # Plotly bought that width by shrinking the plot. The same two facts are one
    # caption clause pair above the plot, and they are still pinned here.
    from webapp.pipeline_runner import panel_caption
    assert figs["fft"].data[0].colorbar.title.text is None
    assert panel_caption(figs["fft"]) == "dB rel. peak · clipped at -40.0 dB"

    # ASCENDING FROM ZERO, not fftshifted-and-negated. The spine's RangeTransformBlock
    # keeps only the non-negative-delay half and hands every product a cube whose bin 0
    # is zero excess delay, so the display axis is `gate * per * m_per_bin` -- the
    # fftshift/negate/crop this test used to assert belonged to the products' own range
    # FFTs, which the one-chain contract deleted. The metre itself is the bistatic excess
    # path c*tau (owner ballot 2B) on the endpoint-inclusive grid (F97d), which is why the
    # expectation is computed by the same helper the figure code uses rather than retyped.
    from webapp.pipeline_runner import (_conform_range_axis, _display_range_axis,
                                        _range_meta_from_grid)
    _axis, _ = _display_range_axis(bins, _range_meta_from_grid(bins, 3e9))
    expected_range = _conform_range_axis(_axis, bins)
    np.testing.assert_allclose(figs["range_az"].data[0].x, expected_u)
    np.testing.assert_allclose(figs["range_az"].data[0].y, expected_range)
    assert figs["range_az"].layout.xaxis.title.text == "azimuth sin(θ)"
    # The axis title names the convention FIRST and then says the axis is the
    # non-negative half of the transform's period (hostile round 12 item 13); the exact
    # wording and the half-window number are pinned in test_webapp_figures_wave7.py.
    assert figs["range_az"].layout.yaxis.title.text.startswith("excess path (m)")

    np.testing.assert_allclose(figs["range_el"].data[0].x, expected_u)
    np.testing.assert_allclose(figs["range_el"].data[0].y, expected_range)
    assert figs["range_el"].layout.xaxis.title.text == "elevation sin(θ)"
    assert figs["range_el"].layout.yaxis.title.text.startswith("excess path (m)")


def test_figures_from_outputs_range_axis_valid_for_any_bins():
    """The range blocks compress over the FULL frequency band and power-bin to
    `bins` display gates, so the physical range axis is well-defined even when
    bins != the frame's frequency-sample count (range-per-gate = c*n_freqs /
    (2*B*bins)). Only a genuinely absent metadata dict falls back to raw indices."""
    torch = pytest.importorskip("torch")
    import numpy as np
    from webapp.pipeline_runner import figures_from_outputs

    bins = 8
    n_freqs = 64
    freq_span_hz = 3e9
    outputs = {
        "range_az": [torch.zeros((bins, bins), dtype=torch.complex64)],
        "_axis_meta": {
            "range_az_bins": bins,
            "n_freqs": n_freqs,  # != bins: full-band compression still gives a valid axis
            "freq_span_hz": freq_span_hz,
        },
    }
    figs = figures_from_outputs(outputs)
    from webapp.pipeline_runner import (_conform_range_axis, _display_range_axis,
                                        _range_meta_from_grid)
    _axis, _ = _display_range_axis(bins, _range_meta_from_grid(n_freqs, freq_span_hz))
    np.testing.assert_allclose(figs["range_az"].data[0].y,
                               _conform_range_axis(_axis, bins))
    assert figs["range_az"].layout.yaxis.title.text.startswith("excess path (m)")

    # No metadata at all (e.g. a hand-built outputs dict): fall back to raw gates.
    outputs_no_meta = {"range_el": [torch.zeros((bins, bins), dtype=torch.complex64)]}
    figs2 = figures_from_outputs(outputs_no_meta)
    np.testing.assert_allclose(figs2["range_el"].data[0].y, np.arange(bins))
    assert figs2["range_el"].layout.yaxis.title.text == "range (bins)"


def test_range_axis_mirrors_power_bin_grouping_when_nondivisible():
    """When `bins` does not divide the cube's range-bin count (the production case:
    2501 native bins into 256 gates), the axis must mirror `e2e.blocks._power_bin`'s
    CEIL grouping -- `per = ceil(n_range / bins)` native bins per gate -- not the exact
    ratio. The exact-multiple case above cannot catch this.

    What changed with the one chain: `n_range` is the SPINE's cube length (the kept
    non-negative half, `n_fft//2 + 1`), not `n_freqs`, and there is no zero-range gate to
    locate because bin 0 IS zero excess delay."""
    torch = pytest.importorskip("torch")
    import math
    import numpy as np
    from webapp.pipeline_runner import (_conform_range_axis, _display_range_axis,
                                        _range_meta_from_grid, figures_from_outputs)

    bins, n_freqs, freq_span_hz = 8, 100, 3e9   # 51 kept bins % 8 != 0
    outputs = {
        "range_az": [torch.zeros((bins, bins), dtype=torch.complex64)],
        "_axis_meta": {"range_az_bins": bins, "n_freqs": n_freqs, "freq_span_hz": freq_span_hz},
    }
    figs = figures_from_outputs(outputs)
    rmeta = _range_meta_from_grid(n_freqs, freq_span_hz)
    per = math.ceil(rmeta["range_n_bins"] / bins)
    assert rmeta["range_n_bins"] % bins != 0, "this test needs a non-divisible case"
    axis, gate = _display_range_axis(bins, rmeta)
    assert gate == pytest.approx(per * rmeta["range_m_per_bin"])
    np.testing.assert_allclose(figs["range_az"].data[0].y,
                               _conform_range_axis(axis, bins))
    assert figs["range_az"].layout.yaxis.title.text.startswith("excess path (m)")


def test_placeholder_figure_is_plotly_figure():
    import plotly.graph_objects as go
    from webapp.pipeline_runner import placeholder_figure
    fig = placeholder_figure("hello")
    assert isinstance(fig, go.Figure)


def test_param_helper_falls_back_to_default():
    from webapp.pipeline_runner import _p
    from webapp.pipeline_registry import default_block_state
    state = default_block_state()
    # explicit value is returned
    assert _p(state, "subspace", "k") == state["subspace"]["params"]["k"]
    # None / missing falls back to the registry default
    state["subspace"]["params"]["k"] = None
    from webapp.pipeline_registry import BLOCKS_BY_ID
    default_k = next(p.default for p in BLOCKS_BY_ID["subspace"].params if p.key == "k")
    assert _p(state, "subspace", "k") == default_k


# --- Fix #3: positive-only params fall back to default on <= 0 --------------------

def test_positive_param_helper_falls_back_when_non_positive():
    """signal_scaling/freq_span_hz/fft bins must reject <= 0 (would cause 0/0 NaNs)."""
    from webapp.pipeline_runner import _p_positive
    from webapp.pipeline_registry import BLOCKS_BY_ID, default_block_state

    default_scaling = next(
        p.default for p in BLOCKS_BY_ID["rffe"].params if p.key == "signal_scaling"
    )

    state = default_block_state()
    # a legitimate positive value is preserved
    state["rffe"]["params"]["signal_scaling"] = 2e-5
    assert _p_positive(state, "rffe", "signal_scaling") == 2e-5

    # zero falls back to the registry default (not 0 -> no downstream 0/0)
    state["rffe"]["params"]["signal_scaling"] = 0
    assert _p_positive(state, "rffe", "signal_scaling") == default_scaling
    # negative likewise
    state["rffe"]["params"]["signal_scaling"] = -1.0
    assert _p_positive(state, "rffe", "signal_scaling") == default_scaling
    # None likewise
    state["rffe"]["params"]["signal_scaling"] = None
    assert _p_positive(state, "rffe", "signal_scaling") == default_scaling


# --- Fix #2: all-zero product -> finite (not NaN) heatmap -------------------------

def test_to_numpy_abs_db_all_zero_is_finite():
    """An all-zero product must yield a finite map, not all-NaN (zero-division)."""
    import numpy as np

    torch = pytest.importorskip("torch")
    from webapp.pipeline_runner import _to_numpy_abs_db

    z = torch.zeros((4, 4), dtype=torch.complex64)
    out = _to_numpy_abs_db(z)
    assert np.isfinite(out).all(), "all-zero product produced non-finite dB values"


# --- Fix #1: array_shape is forwarded; N_RX derived from the env block ------------

@pytest.mark.gui
@pytest.mark.slow
def test_run_pipeline_forwards_array_shape(monkeypatch, make_env_block):
    """N_RX/AdaOja dim and Simulation.array_shape must follow the env block's shape.

    Uses a synthetic env block whose array_shape is non-(32,32). We capture the
    AdaOjaBlock dimension and Simulation's array_shape to prove both are derived
    from the env block rather than hardcoded 32*32. Gated (needs torch).
    """
    pytest.importorskip("torch")
    from webapp import pipeline_runner
    from webapp.pipeline_registry import default_block_state
    import e2e.blocks as blocks
    import e2e.simulation as simulation

    shape = (8, 4)  # n_rx = 32, deliberately != 32*32
    env = make_env_block(n_frames=2, n_freqs=16)
    env.array_shape = shape
    monkeypatch.setattr(blocks, "SionnaEnvironmentBlock", lambda *a, **k: env)

    seen = {}

    real_oja = blocks.AdaOjaBlock

    def spy_oja(d, k, *args, **kwargs):
        seen["oja_d"] = d
        return real_oja(d, k, *args, **kwargs)

    monkeypatch.setattr(blocks, "AdaOjaBlock", spy_oja)

    real_sim = simulation.Simulation

    def spy_sim(*a, **k):
        seen["array_shape"] = k.get("array_shape")
        return real_sim(*a, **k)

    monkeypatch.setattr(pipeline_runner, "Simulation", spy_sim, raising=False)
    # pipeline_runner imports Simulation locally inside run_pipeline, so patch the
    # source module too.
    monkeypatch.setattr(simulation, "Simulation", spy_sim)

    state = default_block_state()
    try:
        pipeline_runner.run_pipeline(state, n_steps=1)
    except Exception:
        pass  # we only care about what was constructed, not a full successful run

    assert seen.get("oja_d") == shape[0] * shape[1], (
        f"AdaOja dim {seen.get('oja_d')} should equal n_rx={shape[0] * shape[1]}"
    )
    assert seen.get("array_shape") == shape, (
        f"Simulation array_shape {seen.get('array_shape')} should be {shape}"
    )


# --- Fix #4: an empty upload is a no-op (does not clobber the editor) -------------

def test_empty_upload_is_no_op(monkeypatch):
    """An upload-json trigger with empty contents must return no_update, NOT the
    reference scenario (which would silently overwrite the editor)."""
    from dash import no_update
    import webapp.app as appmod

    # Force the trigger to look like the upload control with empty contents.
    class _Ctx:
        triggered_id = "upload-json"

    monkeypatch.setattr(appmod, "ctx", _Ctx)

    func = appmod._load_scenario.__wrapped__ if hasattr(
        appmod._load_scenario, "__wrapped__"
    ) else appmod._load_scenario

    # ref_name points at a real reference; if the bug were present, that would be
    # returned instead of no_update.
    from e2e.scenario import REFERENCE_SCENARIOS
    ref_name = next(iter(REFERENCE_SCENARIOS))

    result = func(load_clicks=0, upload_contents="", ref_name=ref_name)
    assert result is no_update


# =============================================================================
# Real run through the runner -- gated (needs torch + frames). Auto-skips.
# =============================================================================

@pytest.mark.gui
@pytest.mark.slow
def test_run_pipeline_real_run_with_synthetic_frames(monkeypatch, make_env_block):
    """End-to-end run via the UI runner using a synthetic env block (no .pkl/Sionna).

    Monkeypatches SionnaEnvironmentBlock so no precomputed frames are needed.
    Gated behind gui+slow so it only runs with RUN_GUI=1 RUN_SLOW=1.
    """
    pytest.importorskip("torch")
    from webapp import pipeline_runner
    from webapp.pipeline_runner import figures_from_outputs, run_pipeline
    from webapp.pipeline_registry import default_block_state
    import e2e.blocks as blocks

    env = make_env_block(n_frames=3, n_freqs=16)
    monkeypatch.setattr(blocks, "SionnaEnvironmentBlock", lambda *a, **k: env)

    state = default_block_state()
    outputs = run_pipeline(state, n_steps=2)
    assert isinstance(outputs, dict)
    figs = figures_from_outputs(outputs)
    assert isinstance(figs, dict)


# --- Diagram render must be per-session, not module-cached ------------------------

def test_render_diagram_fresh_session_always_renders():
    """A fresh client (new session or page refresh) starts with elements=[] and a
    None sig store, so the first callback fire must ALWAYS return elements — a
    module-level cache shared across sessions used to no_update the second
    client/refresh into a permanently blank diagram."""
    from dash import no_update
    from webapp.app import _render_diagram, _enabled_signature
    from webapp.pipeline_registry import default_block_state

    state = default_block_state()
    elements, sig = _render_diagram(state, None)   # fresh session: last_sig=None
    assert elements is not no_update and len(elements) > 0
    assert sig == _enabled_signature(state)

    # Same session, param-only edit (sig unchanged): keep graph + selection.
    elements2, sig2 = _render_diagram(state, sig)
    assert elements2 is no_update and sig2 is no_update

    # Structural change (toggle a block): re-render.
    state["rffe"]["enabled"] = not state["rffe"]["enabled"]
    elements3, sig3 = _render_diagram(state, sig)
    assert elements3 is not no_update and sig3 != sig


def test_scale_mode_param_reads_through():
    """The rffe scale_mode registry param must read through _p with the right key
    and default, since pipeline_runner maps it to RFFEBlock.physical_scale."""
    from webapp.pipeline_runner import _p
    from webapp.pipeline_registry import default_block_state

    state = default_block_state()
    assert _p(state, "rffe", "scale_mode") == "auto"
    state["rffe"]["params"]["scale_mode"] = "legacy"
    assert _p(state, "rffe", "scale_mode") == "legacy"
    state["rffe"]["params"]["scale_mode"] = "physical"
    assert _p(state, "rffe", "scale_mode") == "physical"


class _FakeEnvBlock:
    """Minimal stand-in for SionnaEnvironmentBlock's metadata pass-throughs."""

    def __init__(self, freq_plan=None, physical_scale=None):
        self.freq_plan = freq_plan
        self.physical_scale = physical_scale


class _LegacyEnvBlock:
    """No freq_plan/physical_scale attrs at all (pre-v2 env block)."""


def test_resolve_physical_scale_forced_modes():
    from webapp.pipeline_runner import _resolve_physical_scale

    env = _FakeEnvBlock(physical_scale=False)
    assert _resolve_physical_scale("physical", env) is True
    assert _resolve_physical_scale("legacy", env) is False

    env_true = _FakeEnvBlock(physical_scale=True)
    assert _resolve_physical_scale("legacy", env_true) is False
    assert _resolve_physical_scale("physical", env_true) is True


def test_resolve_physical_scale_auto_follows_env_metadata():
    from webapp.pipeline_runner import _resolve_physical_scale

    assert _resolve_physical_scale("auto", _FakeEnvBlock(physical_scale=True)) is True
    assert _resolve_physical_scale("auto", _FakeEnvBlock(physical_scale=False)) is False
    # None (v2 pkl with unset metadata) and a legacy block with no attr at all both
    # degrade to legacy behavior (False), unchanged from before this feature.
    assert _resolve_physical_scale("auto", _FakeEnvBlock(physical_scale=None)) is False
    assert _resolve_physical_scale("auto", _LegacyEnvBlock()) is False


def test_resolve_freq_span_hz_prefers_freq_plan():
    from webapp.pipeline_runner import _resolve_freq_span_hz
    from webapp.pipeline_registry import default_block_state

    state = default_block_state()
    plan = {"carrier_hz": 30e9, "start_hz": 28.5e9, "stop_hz": 31.5e9, "num_freqs": 512}
    env = _FakeEnvBlock(freq_plan=plan)
    assert _resolve_freq_span_hz(state, env) == pytest.approx(3e9)


def test_resolve_freq_span_hz_falls_back_to_param_when_no_freq_plan():
    from webapp.pipeline_runner import _resolve_freq_span_hz
    from webapp.pipeline_registry import default_block_state

    state = default_block_state()
    default_span = state["rffe"]["params"]["freq_span_hz"]

    # No freq_plan attr at all (legacy env block).
    assert _resolve_freq_span_hz(state, _LegacyEnvBlock()) == pytest.approx(default_span)
    # freq_plan attr present but None.
    assert _resolve_freq_span_hz(state, _FakeEnvBlock(freq_plan=None)) == pytest.approx(
        default_span
    )

    # Custom UI value is honored when there's no freq_plan to override it.
    state["rffe"]["params"]["freq_span_hz"] = 5e9
    assert _resolve_freq_span_hz(state, _LegacyEnvBlock()) == pytest.approx(5e9)


# =============================================================================
# comms head (opt-in "product" -- see webapp/pipeline_registry.py "comms")
# =============================================================================

def test_registry_comms_block_toggleable_and_disabled_by_default():
    from webapp.pipeline_registry import BLOCKS_BY_ID, default_block_state

    spec = BLOCKS_BY_ID["comms"]
    assert spec.category == "product"
    assert spec.toggleable is True
    assert spec.enabled_default is False
    pkeys = {p.key for p in spec.params}
    assert pkeys == {"combining", "snr_db", "fft_size"}

    # default block state mirrors the registry: present, off, with the 3 params
    state = default_block_state()
    assert state["comms"]["enabled"] is False
    assert set(state["comms"]["params"]) == {"combining", "snr_db", "fft_size"}


def test_comms_freqs_uses_freq_plan_metadata():
    from webapp.pipeline_runner import _comms_freqs
    from webapp.pipeline_registry import default_block_state
    import numpy as np

    state = default_block_state()
    plan = {"carrier_hz": 30e9, "start_hz": 28.5e9, "stop_hz": 31.5e9, "num_freqs": 16}
    env = _FakeEnvBlock(freq_plan=plan)
    freqs = _comms_freqs(state, env)
    assert len(freqs) == 16
    assert freqs[0] == pytest.approx(28.5e9)
    assert freqs[-1] == pytest.approx(31.5e9)


def test_comms_freqs_falls_back_to_rffe_span_centered_at_30ghz_for_legacy_env():
    from webapp.pipeline_runner import _comms_freqs
    from webapp.pipeline_registry import default_block_state

    state = default_block_state()
    state["rffe"]["params"]["freq_span_hz"] = 2e9
    freqs = _comms_freqs(state, _LegacyEnvBlock())
    assert freqs[0] == pytest.approx(30e9 - 1e9)
    assert freqs[-1] == pytest.approx(30e9 + 1e9)


@pytest.mark.gui
@pytest.mark.slow
def test_run_pipeline_comms_enabled_emits_ber_and_figures(monkeypatch, make_env_block):
    """Enabling the comms head appends ModemBlock/BERBlock and the run's outputs
    (and derived figures) include the new comms products."""
    pytest.importorskip("torch")
    from webapp import pipeline_runner
    from webapp.pipeline_runner import figures_from_outputs, run_pipeline
    from webapp.pipeline_registry import default_block_state
    import e2e.blocks as blocks

    env = make_env_block(n_frames=2, n_freqs=32)
    monkeypatch.setattr(blocks, "SionnaEnvironmentBlock", lambda *a, **k: env)

    state = default_block_state()
    state["comms"]["enabled"] = True
    outputs = run_pipeline(state, n_steps=2)

    assert "ber" in outputs and len(outputs["ber"]) == 2
    assert all(0.0 <= b <= 1.0 for b in outputs["ber"])
    assert "evm" in outputs
    assert "comm_data_eq" in outputs

    figs = figures_from_outputs(outputs)
    assert "ber" in figs
    assert "mrc" in figs["ber"].layout.title.text  # default combining
    assert figs["ber"].layout.yaxis.type == "log"
    assert "evm" in figs
    assert "comm_const" in figs


@pytest.mark.gui
@pytest.mark.slow
def test_run_pipeline_forwards_comms_combining(monkeypatch, make_env_block):
    """The comms 'combining' param must reach ModemBlock's constructor, spy-style
    like test_run_pipeline_forwards_array_shape."""
    pytest.importorskip("torch")
    from webapp import pipeline_runner
    from webapp.pipeline_registry import default_block_state
    import e2e.blocks as blocks
    import e2e.comms.blocks as comm_blocks

    env = make_env_block(n_frames=1, n_freqs=16)
    monkeypatch.setattr(blocks, "SionnaEnvironmentBlock", lambda *a, **k: env)

    seen = {}
    real_modem = comm_blocks.ModemBlock

    def spy_modem(*a, **k):
        seen["combining"] = k.get("combining")
        seen["snr_db"] = k.get("snr_db")
        seen["fft_size"] = k.get("fft_size")
        return real_modem(*a, **k)

    monkeypatch.setattr(comm_blocks, "ModemBlock", spy_modem)

    state = default_block_state()
    state["comms"]["enabled"] = True
    state["comms"]["params"]["combining"] = "subspace"
    state["comms"]["params"]["snr_db"] = 5.0
    state["comms"]["params"]["fft_size"] = 128  # > default n_active=52, avoids ValueError
    try:
        pipeline_runner.run_pipeline(state, n_steps=1)
    except Exception:
        pass  # we only care about what ModemBlock was constructed with

    assert seen.get("combining") == "subspace"
    assert seen.get("snr_db") == pytest.approx(5.0)
    assert seen.get("fft_size") == 128


def test_figures_from_outputs_ber_evm_constellation():
    """figures_from_outputs builds the comms figures directly from a hand-built
    outputs dict (no torch/run_pipeline needed for 'ber'/'evm', which are plain
    floats; 'comm_data_eq' exercises the torch-tolerant complex flattening)."""
    torch = pytest.importorskip("torch")
    from webapp.pipeline_runner import figures_from_outputs

    outputs = {
        "ber": [0.1, 0.01],
        "evm": [0.2, 0.05],
        "comm_array_gain_db": [10.0, 20.0],
        "comm_data_eq": [torch.tensor([1 + 1j, -1 - 1j], dtype=torch.complex64)],
        "_comms_meta": {"combining": "mrc"},
    }
    figs = figures_from_outputs(outputs)

    # Combining and array gain are CAPTION clauses now, not a parenthesised figure
    # title (the figure carries no title at all -- layout spec, 2026-09-24). Same two
    # facts, same computation; `panel_text` is title + caption + Details.
    from webapp.pipeline_runner import panel_of, panel_text
    assert figs["ber"].layout.title.text is None
    assert panel_of(figs["ber"])["title"] == "Comms head BER"
    assert "mrc combining" in panel_text(figs["ber"])
    assert "array gain 15.0 dB" in panel_text(figs["ber"])
    assert figs["ber"].layout.yaxis.type == "log"
    assert list(figs["ber"].data[0].y) == [0.1, 0.01]

    assert panel_of(figs["evm"])["title"] == "Comms head EVM per frame"

    const = figs["comm_const"].data[0]
    import numpy as np
    np.testing.assert_allclose(const.x, [1.0, -1.0])
    np.testing.assert_allclose(const.y, [1.0, -1.0])
    assert figs["comm_const"].layout.yaxis.scaleanchor == "x"


def test_figures_from_outputs_ber_without_array_gain_omits_it_from_the_caption():
    from webapp.pipeline_runner import figures_from_outputs

    from webapp.pipeline_runner import panel_caption
    outputs = {"ber": [0.3], "_comms_meta": {"combining": "element0"}}
    figs = figures_from_outputs(outputs)
    # No array gain -> no array-gain clause. (The clause moved from the figure title
    # to the panel caption with the 2026-09-24 layout change; the fact it pins --
    # "absent means absent, never a stale number" -- is unchanged.)
    assert "element0 combining" in panel_caption(figs["ber"])
    assert "array gain" not in panel_caption(figs["ber"])


# =============================================================================
# ADC-cube chain blocks (e2e/chain/*, e2e/environment/blocks.py, e2e/ml/blocks.py)
# registered in webapp/pipeline_registry.py: rt_environment / waveform / tx_pa /
# modulate / dechirp / impairment / quantizer / radar_cube / detector / sink.
# =============================================================================

_ADC_CHAIN_BLOCK_IDS = {
    "rt_environment", "waveform", "tx_pa", "modulate", "dechirp",
    "thermal_noise", "impairment", "if_hpf", "quantizer",
    "radar_cube", "detector", "sink",
}


def test_registry_adc_chain_blocks_all_present_and_unique():
    from webapp.pipeline_registry import BLOCKS, BLOCKS_BY_ID

    ids = [b.id for b in BLOCKS]
    assert len(ids) == len(set(ids)), f"duplicate block ids: {ids}"
    missing = _ADC_CHAIN_BLOCK_IDS - set(BLOCKS_BY_ID)
    assert not missing, f"new blocks missing from BLOCKS: {missing}"


def test_registry_adc_chain_blocks_default_off_and_categorized():
    """Every new block is opt-in (enabled_default False), so the existing
    radar/subspace/comms pipeline is unaffected until a user turns one on."""
    from webapp.pipeline_registry import BLOCKS_BY_ID

    expected_category = {
        "rt_environment": "source", "waveform": "source",
        "tx_pa": "stage", "modulate": "stage", "dechirp": "stage",
        "thermal_noise": "stage", "impairment": "stage",
        "if_hpf": "stage", "quantizer": "stage",
        "radar_cube": "product", "detector": "product", "sink": "product",
    }
    for block_id, category in expected_category.items():
        spec = BLOCKS_BY_ID[block_id]
        assert spec.enabled_default is False, f"{block_id} should default off"
        assert spec.toggleable is True, f"{block_id} should be toggleable"
        assert spec.category == category, f"{block_id} category {spec.category!r}"


def test_registry_dechirp_preset_and_mimo_params():
    from webapp.pipeline_registry import BLOCKS_BY_ID

    params = {p.key: p for p in BLOCKS_BY_ID["dechirp"].params}
    # Every choice must be a real PRESETS entry (the registry list is hand-maintained;
    # 2026-08-24 it gained benchmark_v1 + ddma_wide_v1, the two answerable presets).
    from e2e.radar_config import PRESETS
    assert set(params["preset"].choices) <= set(PRESETS)
    assert {"radial_like", "benchmark_v1", "ddma_wide_v1"} <= set(params["preset"].choices)
    # radial_like (12 TX x 16 RX = 192 virtual) is the default: the detection label grid's
    # 192 azimuth bins are only physically answerable at that array size (2026-08-10).
    assert params["preset"].default == "radial_like"
    assert params["mimo"].choices == ["tdm", "ddma", "single"]
    assert params["mimo"].default == "ddma"      # matches the radial_like preset


def test_registry_imports_without_torch_after_adc_chain_additions():
    """Re-assert the no-heavy-import invariant now that BLOCKS is bigger."""
    proc = _import_without_torch("webapp.pipeline_registry")
    assert proc.returncode == 0, (
        f"webapp.pipeline_registry must still import without torch; stderr:\n{proc.stderr}"
    )


def test_the_receive_segment_is_on_the_one_chain_not_a_second_band():
    """The blocks that used to be drawn as an "ADC-cube chain - mutually exclusive with
    the products above" band are now segments of the one chain: the mixing block is a
    chain node, the four ADC stages are collapsed into the single `adc` node (round-11 D2:
    five boxes there cost label ink and bought nothing a presenter clicks separately), and
    the transmit tributary is the waveform node. Their ORDER is still asserted -- it just
    lives in `_CHAIN` and in the stage list `pipeline_runner` builds, not in a set of
    presentational edges that could drift from either."""
    from webapp import block_diagram
    from webapp.pipeline_registry import default_block_state

    elements = block_diagram.build_elements(default_block_state())
    node_ids = {e["data"]["id"] for e in elements}
    edges = {(e["data"]["source"], e["data"]["target"])
             for e in elements if "source" in e["data"]}

    for bid in _ADC_CHAIN_BLOCK_IDS:
        assert block_diagram.resolve_node(bid) in node_ids, (
            f"{bid} is not reachable on the one chain")
    assert block_diagram._NODE_MEMBERS["adc"] == [
        "thermal_noise", "impairment", "if_hpf", "quantizer"], (
        "the ADC node's members are the corpus generator's stage order")
    assert block_diagram._NODE_MEMBERS["waveform"] == ["waveform", "tx_pa", "modulate"]
    # THE PHYSICAL ORDER, not the computation order (seat's read of the 2026-09-24
    # renders, item 1a): front end at the element, then the interconnect, then the mixing
    # block -- because rffe_model.py's cascade IS LNA -> mixer -> baseband amp, so a
    # diagram with "Mixing" upstream of "RF front end" says the signal is dechirped before
    # it is amplified. The computation still applies the cascade to the beat record AFTER
    # the mixing block (F97b licences it); each node's blurb says so, and
    # test_the_front_end_node_says_where_the_computation_applies_it pins that sentence.
    assert ("rffe", "interconnect") in edges
    assert ("interconnect", "dechirp") in edges      # the mixing block, on the chain
    assert ("dechirp", "adc") in edges
    assert ("adc", "cube") in edges                  # the ONE range transform
    assert ("adc", "detector") in edges              # scored detectors read `adc`
    assert ("cube", "radar_cube") in edges           # range-Doppler reads `cube`


def test_param_editor_runs_for_adc_chain_blocks():
    from webapp import block_diagram
    from webapp.pipeline_registry import default_block_state

    state = default_block_state()
    for block_id in _ADC_CHAIN_BLOCK_IDS:
        children = block_diagram.param_editor(block_id, state)
        assert isinstance(children, list) and children


def test_pipeline_runner_lazy_imports_adc_chain_backend():
    """dechirp/rt_environment must be imported lazily inside run_pipeline, not at
    module top -- same torch-free-shell requirement as the rest of the runner."""
    from webapp import pipeline_runner

    run_src = inspect.getsource(pipeline_runner.run_pipeline)
    assert "from e2e.chain.dechirp import DechirpBlock" in run_src
    assert "from e2e.environment.blocks import RTEnvironmentBlock" in run_src


def test_the_comms_head_is_a_tap_on_the_same_chain_as_the_receive_products(
        monkeypatch, make_env_block):
    """ONE CHAIN: the comms head and the mixing block are not rival pipelines.

    This test used to pin the opposite ("dechirp and comms ... cannot run together").
    That refusal existed because the head was a DOWNSTREAM block: downstream blocks run
    after the whole spine, by which point the chain is past the mixing block and the
    channel frequency response the head reads has been dropped at the crossing. The
    one-chain contract (2026-09-24, section 1.2) makes the head a TAP at the mixing
    block's INPUT instead, so there is nothing left for the rule to protect -- and the
    owner's directive was that the diagram must stop showing two pipelines, which a
    refusal like this one is the runtime half of.

    What is pinned here is the behaviour that replaces it: both are enabled, the run
    completes, the comms product (`ber`) is present BESIDE the receive chain's own, and
    the head's own scope note (F98: on a dechirp chain the front end's noise cannot
    reach a tap that reads a channel response, so the head is its own noise source
    there) is what the card has to say -- not a refusal.
    """
    torch = pytest.importorskip("torch")
    from webapp import pipeline_runner
    from webapp.pipeline_registry import default_block_state
    import e2e.blocks as blocks

    env = make_env_block(n_frames=1, n_freqs=16)
    monkeypatch.setattr(blocks, "SionnaEnvironmentBlock", lambda *a, **k: env)

    state = default_block_state()
    state["dechirp"]["enabled"] = True
    state["comms"]["enabled"] = True
    # A product on the receive side of the chain, so the run has one of each.
    state["sink"]["enabled"] = True

    outputs = pipeline_runner.run_pipeline(state, n_steps=1)
    assert outputs.get("ber"), "the comms head ran as a tap but its product never reached outputs"
    assert outputs.get("comm_noise_source"), "the head must record which floor it used"


def test_run_pipeline_detector_on_single_chirp_source_names_the_fix(
        monkeypatch, make_env_block):
    """The detector consumes an ADC cube, which the precomputed .pkl source cannot
    provide (single-chirp frames). Enabling it there must fail with a message naming
    the two sources that CAN feed it -- RT Environment or Corpus Replay -- rather than
    a shape mismatch from inside adc_to_rd. (Until 2026-09-22 this test pinned the
    detector's old behaviour of always raising for want of a checkpoint path; the
    block now runs in CFAR or ML mode, see tests/test_webapp_detector.py.)"""
    torch = pytest.importorskip("torch")
    from webapp import pipeline_runner
    from webapp.pipeline_registry import default_block_state
    import e2e.blocks as blocks

    env = make_env_block(n_frames=1, n_freqs=16)
    monkeypatch.setattr(blocks, "SionnaEnvironmentBlock", lambda *a, **k: env)

    state = default_block_state()
    state["dechirp"]["enabled"] = True
    state["detector"]["enabled"] = True
    with pytest.raises(pipeline_runner.PipelineError, match="Corpus Replay"):
        pipeline_runner.run_pipeline(state, n_steps=1)


@pytest.mark.gui
@pytest.mark.slow
def test_run_pipeline_dechirp_chain_replaces_frequency_domain_products(
        monkeypatch, make_env_block):
    """A real run with 'dechirp' (+quantizer+sink) enabled must succeed and must
    NOT produce the frequency-domain products (fft/range_az/...), since the chain
    crossed into RX time. 'sink' is enabled (not radar_cube/detector, see the
    checkpoint/multi-chirp guards tested elsewhere) purely so the downstream
    product list isn't empty -- see test_run_pipeline_dechirp_with_no_product_raises
    for what happens when it is. 'impairment' is deliberately left off: its
    phase-noise stage (e2e/chain/impairments.py, not owned by this shard) hits an
    unrelated pre-existing in-place aliasing RuntimeError on this synthetic-frame
    path (see handoff notes) -- a separate bug from the one this test covers."""
    pytest.importorskip("torch")
    from webapp import pipeline_runner
    from webapp.pipeline_registry import default_block_state
    import e2e.blocks as blocks

    env = make_env_block(n_frames=2, n_freqs=16)
    monkeypatch.setattr(blocks, "SionnaEnvironmentBlock", lambda *a, **k: env)

    state = default_block_state()
    state["dechirp"]["enabled"] = True
    state["quantizer"]["enabled"] = True
    state["sink"]["enabled"] = True
    outputs = pipeline_runner.run_pipeline(state, n_steps=2)

    assert isinstance(outputs, dict)
    for stale_key in ("fft", "range_az", "range_el", "subspace_err"):
        assert stale_key not in outputs, f"{stale_key} should not appear once dechirp is on"


def test_run_pipeline_dechirp_with_no_product_raises(monkeypatch, make_env_block):
    """Enabling only 'dechirp' (no radar_cube/detector/sink) used to run to
    completion with an EMPTY downstream_blocks list -- zero outputs, no error,
    indistinguishable in the Results tab from never having run. It must now raise
    a PipelineError naming the fix (enable a product, or disable dechirp)."""
    torch = pytest.importorskip("torch")
    from webapp import pipeline_runner
    from webapp.pipeline_registry import default_block_state
    import e2e.blocks as blocks

    env = make_env_block(n_frames=1, n_freqs=16)
    monkeypatch.setattr(blocks, "SionnaEnvironmentBlock", lambda *a, **k: env)

    state = default_block_state()
    state["dechirp"]["enabled"] = True
    with pytest.raises(pipeline_runner.PipelineError, match="Radar Cube"):
        pipeline_runner.run_pipeline(state, n_steps=1)




# --------------------------------------------------------------------------------
# D2: scenario geometry leads the Results tab
# --------------------------------------------------------------------------------
def _rt_source_state():
    from webapp.pipeline_registry import default_block_state
    state = default_block_state()
    state["rt_environment"]["enabled"] = True
    return state


def test_results_lead_with_scene_geometry_when_the_scenario_parses(monkeypatch):
    import plotly.graph_objects as go
    """A stripe in sin(azimuth) is only interpretable next to the geometry that made
    it, so when the RT Environment source ray-traced the Scenario editor's scene, that
    scene renders as the FIRST results card (D2)."""
    from webapp import app as webapp_app
    from e2e.scenario import REFERENCE_SCENARIOS

    monkeypatch.setattr(webapp_app, "run_pipeline", lambda *a, **k: {"fft": []})
    monkeypatch.setattr(webapp_app, "figures_from_outputs",
                        lambda outputs: {"fft": go.Figure()})

    scenario_json = REFERENCE_SCENARIOS["munich_radar"]().to_json()
    data, _status, _tab, _sink = webapp_app._run_pipeline(
        1, _rt_source_state(), 2, scenario_json)

    assert list(data)[0] == "scene_topdown", "geometry must come before signal plots"
    assert "fft" in data  # and it does not displace the products


@pytest.mark.parametrize("source", ["environment", "corpus_environment"])
def test_no_scene_panel_unless_the_rt_source_made_the_frames(monkeypatch, source):
    import plotly.graph_objects as go
    """Precomputed .pkl frames and corpus replay carry their own geometry; the
    Scenario editor's JSON says nothing about them. The 2026-09-22 rehearsal had a
    plan view of a lone radar triangle as the first card of every demo preset."""
    from webapp import app as webapp_app
    from webapp.pipeline_registry import default_block_state
    from e2e.scenario import REFERENCE_SCENARIOS

    monkeypatch.setattr(webapp_app, "run_pipeline", lambda *a, **k: {"fft": []})
    monkeypatch.setattr(webapp_app, "figures_from_outputs",
                        lambda outputs: {"fft": go.Figure()})

    state = default_block_state()
    state["rt_environment"]["enabled"] = False
    state[source]["enabled"] = True
    scenario_json = REFERENCE_SCENARIOS["munich_radar"]().to_json()
    data, *_ = webapp_app._run_pipeline(1, state, 2, scenario_json)

    assert "scene_topdown" not in data and "fft" in data


@pytest.mark.parametrize("scenario_json", ["", "{not json", None])
def test_results_survive_an_unparseable_scenario(monkeypatch, scenario_json):
    import plotly.graph_objects as go
    """The Scenario editor may hold half-typed JSON while a run is launched. The
    geometry panel is best-effort: its absence must never cost the user the results."""
    from webapp import app as webapp_app

    monkeypatch.setattr(webapp_app, "run_pipeline", lambda *a, **k: {"fft": []})
    monkeypatch.setattr(webapp_app, "figures_from_outputs",
                        lambda outputs: {"fft": go.Figure()})

    data, _status, _tab, _sink = webapp_app._run_pipeline(
        1, _rt_source_state(), 2, scenario_json)

    assert "scene_topdown" not in data
    assert "fft" in data


def test_results_survive_a_topdown_figure_that_raises(monkeypatch):
    import plotly.graph_objects as go
    """Same guarantee when the figure builder itself fails on an exotic scenario."""
    from webapp import app as webapp_app
    from e2e.scenario import REFERENCE_SCENARIOS

    monkeypatch.setattr(webapp_app, "run_pipeline", lambda *a, **k: {"fft": []})
    monkeypatch.setattr(webapp_app, "figures_from_outputs",
                        lambda outputs: {"fft": go.Figure()})

    def _boom(_sc):
        raise RuntimeError("exotic scenario")

    monkeypatch.setattr(webapp_app, "scenario_topdown_figure", _boom)
    data, *_ = webapp_app._run_pipeline(1, _rt_source_state(), 2,
                                        REFERENCE_SCENARIOS["munich_radar"]().to_json())

    assert "scene_topdown" not in data and "fft" in data


# --------------------------------------------------------------------------------
# Layering: core never imports e2e.ml at module scope (the C1 rule the docs assert)
# --------------------------------------------------------------------------------
def test_core_modules_do_not_import_e2e_ml_at_module_scope():
    """CONTRIBUTING.md and e2e/ml/__init__.py both promise core does not import the ML
    package at module scope. Nothing enforced it, and a module relocation quietly broke
    it (e2e/render_scene.py, 2026-08-27) while the docs kept asserting otherwise.

    Function-local (lazy) imports are the sanctioned escape hatch and are ignored here;
    only module-scope imports fail this test.
    """
    import ast

    root = Path(__file__).resolve().parents[1] / "e2e"
    ml_dir = root / "ml"
    offenders = []
    for path in root.rglob("*.py"):
        if ml_dir in path.parents or path == ml_dir:
            continue  # the ML package may of course import itself
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:  # module scope ONLY: top-level statements
            names = []
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("e2e.ml"):
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names if a.name.startswith("e2e.ml")]
            if names:
                offenders.append(f"{path.relative_to(root.parent)}:{node.lineno} -> {names}")

    assert not offenders, (
        "core modules import e2e.ml at module scope (make them function-local, or "
        "update the documented rule):\n  " + "\n  ".join(offenders))


def _write_fake_ka_pkl(path, carrier_hz=30e9):
    """A tiny but structurally real v2 `{"meta", "links"}` pkl (see
    e2e.environment.sionna_simple_channel / sionna_iterator), so `discover_sionna_scenarios`
    exercises its real `freq_plan` read without needing the ~0.8 GB production file."""
    import pickle

    import numpy as np

    meta = {
        "version": 2,
        "freq_plan": {"carrier_hz": carrier_hz, "start_hz": carrier_hz - 1.5e9,
                     "stop_hz": carrier_hz + 1.5e9, "num_freqs": 4},
        "links": {"munich": {"rx_array_shape": [2, 2], "physical_scale": False}},
    }
    arr = np.zeros((1, 1, 1, 1, 4), dtype=np.complex64)
    with open(path, "wb") as f:
        pickle.dump({"meta": meta, "links": {"munich": arr}}, f)


def test_gui_offers_only_loadable_sionna_scenarios(tmp_path):
    """The Scenario dropdown must list only scenarios whose backing .pkl exists
    (2026-09-23: 'etoile' was offered with no file and raised FileNotFoundError on
    selection)."""
    from webapp import corpus_catalog as cc
    assert cc.discover_sionna_scenarios(tmp_path) == []
    # A bare (legacy-shaped) munich.pkl offers only the legacy label -- 'munich_ka.pkl'
    # is what plain 'munich' now resolves to (F93).
    (tmp_path / "munich.pkl").write_bytes(b"legacy-bare-array-stand-in")
    assert cc.discover_sionna_scenarios(tmp_path) == [cc.MUNICH_LEGACY_LABEL]
    (tmp_path / "etoile.pkl").write_bytes(b"x")
    assert cc.discover_sionna_scenarios(tmp_path) == [cc.MUNICH_LEGACY_LABEL, "etoile"]

    from webapp.pipeline_registry import BLOCKS
    env = next(b for b in BLOCKS if b.id == "environment")
    spec = next(p for p in env.params if p.key == "scenario_name")
    for token in spec.choices:
        name, link = cc.resolve_sionna_scenario(token)
        expected_file = "munich_ka.pkl" if (name == "munich" and link is None) else f"{name}.pkl"
        assert (cc.SIONNA_SIMS_DIR / expected_file).is_file() or not cc.SIONNA_SCENARIOS


def test_discover_sionna_scenarios_ka_and_legacy_both_present(tmp_path):
    """Both munich files present -> two distinct tokens, Ka-band first (it is the
    default and what every preset uses), each carrying its own label."""
    from webapp import corpus_catalog as cc

    _write_fake_ka_pkl(tmp_path / "munich_ka.pkl", carrier_hz=30e9)
    (tmp_path / "munich.pkl").write_bytes(b"legacy-bare-array-stand-in")

    labels = cc.discover_sionna_scenarios(tmp_path)
    assert labels == ["munich (Ka-band, 30 GHz)", cc.MUNICH_LEGACY_LABEL]

    _labels, index = cc._discover_sionna_scenario_specs(tmp_path)
    assert index["munich (Ka-band, 30 GHz)"] == ("munich", None)
    from e2e.environment.sionna_iterator import MUNICH_LEGACY_LINK
    assert index[cc.MUNICH_LEGACY_LABEL] == ("munich", MUNICH_LEGACY_LINK)


def test_discover_sionna_scenarios_ka_only_when_legacy_absent(tmp_path):
    """Only munich_ka.pkl on disk -> only the Ka token is offered."""
    from webapp import corpus_catalog as cc

    _write_fake_ka_pkl(tmp_path / "munich_ka.pkl", carrier_hz=30e9)
    assert cc.discover_sionna_scenarios(tmp_path) == ["munich (Ka-band, 30 GHz)"]


def test_munich_ka_label_falls_back_when_metadata_unreadable(tmp_path):
    """A corrupt/unexpected munich_ka.pkl must not take discovery down with it."""
    from webapp import corpus_catalog as cc

    (tmp_path / "munich_ka.pkl").write_bytes(b"not a pickle")
    assert cc.discover_sionna_scenarios(tmp_path) == ["munich (Ka-band)"]


def test_resolve_sionna_scenario_passes_through_unknown_token():
    """An unrecognized token (a saved UI state predating the split, or a name typed
    directly) must still reach SionnaEnvironmentBlock, not be rejected here."""
    from webapp import corpus_catalog as cc

    assert cc.resolve_sionna_scenario("etoile") == ("etoile", None)
    assert cc.resolve_sionna_scenario("some_future_scenario") == ("some_future_scenario", None)


def test_environment_block_resolves_legacy_token_to_legacy_link(monkeypatch):
    """run_pipeline must resolve the legacy label to (scenario_name='munich',
    link=MUNICH_LEGACY_LINK) -- not pass the display label straight through as a
    scenario name, which SionnaEnvironmentBlock would reject."""
    pytest.importorskip("torch")
    import e2e.blocks as blocks
    from webapp import pipeline_runner
    from webapp.pipeline_registry import default_block_state
    from webapp.corpus_catalog import MUNICH_LEGACY_LABEL
    from e2e.environment.sionna_iterator import MUNICH_LEGACY_LINK

    seen = {}

    def spy(name, *args, **kwargs):
        seen["name"] = name
        seen["link"] = kwargs.get("link")
        raise FileNotFoundError("stand-in: stop before the rest of the pipeline builds")

    monkeypatch.setattr(blocks, "SionnaEnvironmentBlock", spy)
    state = default_block_state()
    state["environment"]["params"]["scenario_name"] = MUNICH_LEGACY_LABEL
    with pytest.raises(pipeline_runner.PipelineError):
        pipeline_runner.run_pipeline(state, n_steps=1)

    assert seen["name"] == "munich"
    assert seen["link"] == MUNICH_LEGACY_LINK


def test_environment_block_default_token_resolves_to_no_link(monkeypatch):
    """The default (Ka) token must resolve to link=None, i.e. build exactly as
    before the legacy/Ka split."""
    pytest.importorskip("torch")
    import e2e.blocks as blocks
    from webapp import pipeline_runner
    from webapp.pipeline_registry import default_block_state

    seen = {}

    def spy(name, *args, **kwargs):
        seen["name"] = name
        seen["link"] = kwargs.get("link")
        raise FileNotFoundError("stand-in: stop before the rest of the pipeline builds")

    monkeypatch.setattr(blocks, "SionnaEnvironmentBlock", spy)
    state = default_block_state()  # scenario_name defaults to DEFAULT_SIONNA_SCENARIO
    with pytest.raises(pipeline_runner.PipelineError):
        pipeline_runner.run_pipeline(state, n_steps=1)

    assert seen["name"] == "munich"
    assert seen["link"] is None


def test_run_notes_state_carrier_when_freq_plan_present(monkeypatch, make_env_block):
    """The run banner must say what carrier the frames actually are, when the
    environment block carries a freq_plan (v2 pkl)."""
    pytest.importorskip("torch")
    import e2e.blocks as blocks
    from webapp import pipeline_runner
    from webapp.pipeline_registry import default_block_state

    env = make_env_block(n_frames=1, n_freqs=4)
    env.freq_plan = {"carrier_hz": 30e9, "start_hz": 28.5e9,
                     "stop_hz": 31.5e9, "num_freqs": 4}
    monkeypatch.setattr(blocks, "SionnaEnvironmentBlock", lambda *a, **k: env)
    state = default_block_state()
    outputs = pipeline_runner.run_pipeline(state, n_steps=1)
    notes = outputs["_axis_meta"]["notes"]
    assert any("30 GHz" in n for n in notes), notes


def test_corpus_replay_split_excludes_train():
    """One click on 'train' would show the learned detector its own training frames."""
    from webapp.pipeline_registry import BLOCKS
    blk = next(b for b in BLOCKS if b.id == "corpus_environment")
    spec = next(p for p in blk.params if p.key == "split")
    assert "train" not in spec.choices and spec.default == "test"
