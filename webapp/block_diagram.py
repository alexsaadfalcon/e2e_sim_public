"""
Block-diagram tab: a dash-cytoscape node graph of the runtime pipeline plus a
parameter editor. Everything here derives from :mod:`webapp.pipeline_registry`,
so adding/removing a block or edge in the registry automatically updates the UI.

No heavy imports — pure layout + helpers.
"""

from __future__ import annotations

from typing import Any, Dict, List

import dash_cytoscape as cyto
from dash import dcc, html

from webapp.demo_presets import PRESETS, DemoPreset, resolved_say as _resolved_say
from webapp.pipeline_registry import (
    BLOCKS, BLOCKS_BY_ID, MAX_N_STEPS, PRODUCT_IDS,
)

# THE ONE CHAIN ---------------------------------------------------------------------
# The owner, 2026-09-24: "the block diagram shown is still very confusing from the fact
# that there are two pipelines. I've been complaining about this for ages. Needs to be
# fixed immediately from the ground up." What used to be here drew four compound REGIONS
# (a TX-time tributary, a main chain, a frequency-domain product column and an "ADC-cube
# chain - mutually exclusive with the products above" band) joined by dotted salmon
# "alternative source path" edges. That was a picture of two pipelines, and an accurate
# picture of the code at the time.
#
# It is not the code any more (notes/ONE_CHAIN_CONTRACT_2026-09-24.md): there is ONE
# serial spine, replay is a start index into it, and every product is a TAP on it. So the
# diagram is ONE ROW, in the spine's own order, with the products fanning out beneath it
# -- Justin's shape, relayed by the owner: stored channel, waveform, front end,
# interconnect, one mixing block with a mode, AFE/ADC, one cube, and the split at the
# application level. The waveform CLASS is the only branch, and it is a choice inside one
# block rather than a fork in the graph.
#
# A DIAGRAM NODE IS NOT ALWAYS ONE REGISTRY BLOCK. Three collapses, each for a reason:
#   * the three source blocks are one node, because only one of them ever feeds a run
#     (the runner ignores the others) -- which is what the dotted "alt" edges were
#     apologising for;
#   * the transmit tributary (waveform / TX PA / modulate) is one node, because it is one
#     decision: which waveform class this run transmits;
#   * the ADC sub-chain (thermal floor / impairments / IF high-pass / quantiser) is one
#     node -- hostile round 11's D2 fix: the graph has to be narrow enough that the fit
#     zoom leaves >= 12 px of rendered label ink (acceptance check 17), and five separate
#     boxes there bought nothing the presenter clicks separately.
# Every collapsed block stays reachable: a node's parameter editor renders every member
# block's controls (see `param_editor`), so no knob became unreachable.
#
#: (node_id, label, category, member registry block ids). ORDER IS THE SPINE'S ORDER.
# LABELS ARE SHORT ON PURPOSE. Two hard constraints meet here and they pull opposite
# ways: a label must fit INSIDE its box (a 30 px line is ~36 px tall, so three lines need
# a 130 px box), and the boxes set the extent, which sets the fit zoom, which sets how
# much ink the room sees. Measured on the render: at this geometry "FMCW" draws 17-21 px
# of cap-height ink against acceptance check 17's 12 px. Long labels spend that margin
# twice over -- once on the wider box and once on the smaller zoom -- so the detail goes
# in the node's editor, where it can be read, not on the node.
#
# PHYSICAL ORDER, NOT COMPUTATION ORDER (seat's read of the 2026-09-24 renders, item 1a).
# Until now this list was the order the stages RUN in, which puts the mixing block ahead
# of the front end -- `e2e/chain/frontend.py` applies the circuit cascade to the sampled
# BEAT record, which is exact below baseband clipping (F97b) and saves a 4.9 GB RF-rate
# tensor per frame. As a picture of a receiver that is wrong twice over: it says the
# signal is dechirped before it is amplified, and `e2e/circuit/rffe_model.py`'s cascade
# IS LNA -> mixer -> baseband amp, so the "Mixing" node is the front end's own mixer and
# cannot be drawn upstream of it. The diagram therefore draws the PHYSICAL signal order
# and each node's editor says where the computation applies it (see the `rffe` and
# `dechirp` blurbs in `webapp/pipeline_registry.py`).
#
# The interconnect sits between the element amplifier and the mixer: `InterconnectBlock`
# applies S21 per RF grid point over the frame's RF band, so it is an RF-band two-port and
# belongs on the RF side of the mixer. Whether it precedes or follows the element LNA is
# NOT fixed by this model (it is linear, and the model injects noise once inside the front
# end, so nothing here measures the difference; notes/ONE_CHAIN_CONTRACT_2026-09-24.md row
# 4 writes the equivalence as `interconnect->LNA->mixer`, which is a commutation statement,
# not an architecture claim). The owner's contract sentence -- "stored channel, waveform,
# front end, interconnect, one mixing block with a mode, AFE/ADC, one cube" -- is the
# tiebreak, and it is the Tessera architecture Thrust 4 tells on stage: LNA at the element,
# TSVs carrying the amplified RF to the RFIC tier, the mixer there.
_CHAIN: List[tuple] = [
    ("source", "Stored\nchannel", "source",
     ["environment", "rt_environment", "corpus_environment"]),
    ("waveform", "Waveform\nFMCW", "stage",
     ["waveform", "tx_pa", "modulate"]),
    ("rffe", "RF front\nend", "stage", ["rffe"]),
    ("interconnect", "Inter-\nconnect", "stage", ["interconnect"]),
    ("dechirp", "Mixing\n(dechirp)", "stage", ["dechirp"]),
    ("adc", "ADC\nchain", "stage",
     ["thermal_noise", "impairment", "if_hpf", "quantizer"]),
    ("cube", "One cube\n+ AFE", "stage",
     ["afe", "subspace"]),
]

#: Short diagram labels for the product nodes -- the registry's own labels ("Radar Cube
#: (Range-Doppler)") are written for the editor's heading and are two to three times too
#: long for a box. Same reason as the chain labels above.
_PRODUCT_LABELS: Dict[str, str] = {
    "fft": "FFT",
    "range_az": "Range-Az",
    "range_el": "Range-El",
    "range_profile": "Range\nprofile",
    "subspace_err": "Subspace\nerror",
    "radar_cube": "Range-\nDoppler",
    "detector": "Detector",
    "sink": "Frame\nsink",
    "comms": "Comms\nhead",
}

#: Which chain node each product TAPS -- the point on the one chain whose domain that
#: product reads. This is the runtime fact (`Simulation`'s product taps), not a drawing
#: convention: the scored detectors read the digitised beat record at the ADC, the
#: range-Doppler product reads the cube, the images read the cube after the compressor,
#: and the comms head reads the channel response at the mixing block's input.
_PRODUCT_TAP: Dict[str, str] = {
    "fft": "cube",
    "range_az": "cube",
    "range_el": "cube",
    "range_profile": "cube",
    "subspace_err": "cube",
    "radar_cube": "cube",
    "detector": "adc",
    "sink": "adc",
    "comms": "interconnect",
}

#: Members of each diagram node, and the reverse map.
_NODE_MEMBERS: Dict[str, List[str]] = {nid: list(members)
                                       for nid, _, _, members in _CHAIN}
_NODE_MEMBERS.update({pid: [pid] for pid in _PRODUCT_TAP})
_BLOCK_TO_NODE: Dict[str, str] = {bid: nid
                                  for nid, members in _NODE_MEMBERS.items()
                                  for bid in members}

#: Chain geometry. Node boxes are 200x130 (see CYTO_STYLESHEET), so a 215 px pitch leaves
#: a 15 px gap and nothing overlaps. These numbers matter for ONE reason: the canvas fits
#: the whole extent, so the extent's WIDTH sets the fit zoom, and the fit zoom times the
#: font size is how much label ink the room actually sees. 7 chain columns at 215 plus the
#: product column give an extent of 1705 px against the ~1032 px panel -- a fit zoom near
#: 0.58, which at the 34 px font in CYTO_STYLESHEET is the ink acceptance check 17 bounds
#: at 12 px. Add a column, or lengthen a label so a box has to grow, and that margin has to
#: be re-measured on a render, not re-argued.
_CHAIN_PITCH_X = 215
_CHAIN_Y = 40

# PRODUCT PLACEMENT (seat's read of the 2026-09-24 renders, item 1b). The products used to
# sit in two rows of five UNDER the whole chain, with every edge leaving the last node. Six
# of the nine tap the cube, which is the RIGHTMOST chain node, so six edges swept back
# across the row -- and the geometry is worse than untidy: the straight line from the cube
# to a product under the first columns enters the chain's own y band immediately (measured
# on the old positions: the `cube -> fft` segment passes through the ADC node's box at
# x 975-1175), so the fan crossed the boxes it was meant to explain.
#
# Two placements, by which point each product taps:
#   * the six CUBE products go in a right-hand COLUMN beside the cube, reached by
#     orthogonal (taxi) edges: one vertical trunk in the 15 px gutter between the cube and
#     the column, then a horizontal spur into each product's left edge. A fan of straight
#     lines into a stacked column always clips the boxes it passes (the ray to the bottom
#     product crosses the left half of the ones above it); the trunk-and-spur routing is
#     the standard bus fan-out and crosses nothing.
#   * the three products that tap EARLIER points (comms at the mixing block's input,
#     detector and frame sink at the ADC) go in one row beneath their own tap, reached by
#     downward taxi edges. They are the only long-ish product edges left and both are
#     short: `adc -> sink` drops one column to the left, entirely below the chain row.
# The chain's width is unchanged, so this costs ~0.09 of fit zoom (extent 1490 -> 1705 px);
# the font goes 30 -> 34 px to pay it back. Both numbers are re-measured on the render, not
# argued here -- see the CYTO_STYLESHEET font comment.
_PRODUCT_COL_X = 6 * _CHAIN_PITCH_X + _CHAIN_PITCH_X   # one pitch right of the cube
_PRODUCT_COL_PITCH_Y = 150
_PRODUCT_ROW_Y = 300
#: The six cube products, top to bottom in the right-hand column: the two images first
#: (they are what Thrusts 1-4 show), then the profile, then range-Doppler, then the two
#: subspace-chain products.
_PRODUCT_COLUMN = ("range_az", "range_el", "fft", "range_profile", "radar_cube",
                   "subspace_err")
#: The products that tap an earlier point, and the chain column each sits under: directly
#: below their own tap where that column is free. `sink` sits one column LEFT of the ADC
#: rather than doubling up on it, because a vertical stack under one node puts the lower
#: box across the upper one's edge; its taxi edge turns above the product row, so it
#: crosses neither a chain node nor another product.
_PRODUCT_ROW = {"comms": "interconnect", "detector": "adc", "sink": "dechirp"}

_POSITIONS: Dict[str, tuple] = {}
for _i, (_nid, _lbl, _cat, _members) in enumerate(_CHAIN):
    _POSITIONS[_nid] = (_i * _CHAIN_PITCH_X, _CHAIN_Y)
for _i, _pid in enumerate(_PRODUCT_COLUMN):
    _POSITIONS[_pid] = (_PRODUCT_COL_X, _CHAIN_Y + _i * _PRODUCT_COL_PITCH_Y)
for _pid, _under in _PRODUCT_ROW.items():
    _POSITIONS[_pid] = (_POSITIONS[_under][0], _PRODUCT_ROW_Y)

# Entry point: the ONE source. There is no second "start here", because there is no
# second path to start on.
_ENTRY_IDS = {"source"}

#: Nodes that are ALWAYS on the chain, whatever the checkboxes say, because
#: `Simulation._build_spine` always builds them (one-chain contract section 1.2: "the
#: dechirp and the range transform are always present, because they are what make the
#: chain one chain"). Drawing them dimmed would be the diagram's own version of the
#: two-pipeline picture: it would say the chain stops here on a Thrust 1 run, and it does
#: not -- the mixing block and the range transform run on every run, and the images are
#: computed from THAT cube.
#:
#: A consequence worth stating where it can be read: the `dechirp` checkbox no longer
#: switches the mixing block on and off. It switches the RECEIVE SEGMENT after it (floor,
#: impairments, IF high-pass, quantiser and their products) -- see the block's own blurb
#: in `webapp/pipeline_registry.py`.
_STRUCTURAL_NODES = {"source", "waveform", "dechirp", "cube"}

#: Source backends, most specific first: which registry source a run actually uses when
#: several are enabled. Mirrors `webapp.pipeline_runner.run_pipeline`'s own precedence --
#: the corpus source wins, then live ray tracing, then the precomputed .pkl frames.
_SOURCE_PRECEDENCE = ("corpus_environment", "rt_environment", "environment")


def active_source(block_state: Dict[str, Dict[str, Any]]) -> str:
    """Which source block this run reads -- the one the `source` node stands for."""
    for bid in _SOURCE_PRECEDENCE[:-1]:
        if block_state.get(bid, {}).get("enabled", False):
            return bid
    return "environment"


def node_members(node_id: str, block_state: Dict[str, Dict[str, Any]]) -> List[str]:
    """The registry blocks a diagram node stands for, in editor order.

    The source node resolves to the ONE backend in use rather than listing all three:
    offering the operator a manifest path for a corpus this run is not reading is how a
    knob gets turned on the wrong block in front of an audience.
    """
    if node_id == "source":
        return [active_source(block_state)]
    return list(_NODE_MEMBERS.get(node_id, [node_id]))


def resolve_node(block_or_node_id: str) -> str:
    """The diagram node a registry block id belongs to (identity for a node id)."""
    if block_or_node_id in _NODE_MEMBERS:
        return block_or_node_id
    return _BLOCK_TO_NODE.get(block_or_node_id, block_or_node_id)


def waveform_kind(block_state: Dict[str, Dict[str, Any]]) -> str:
    """The waveform CLASS this run transmits -- the diagram's one branch point."""
    if not block_state.get("waveform", {}).get("enabled", False):
        return "fmcw"
    return str(block_state.get("waveform", {}).get("params", {}).get("kind") or "fmcw")


#: What the mixing node IS, per waveform class. A class is (source, MIXING MODE,
#: products), so the mixing block is not the same block on all three -- and a diagram
#: that says "dechirp" on a JSAC run tells the room the frame was dechirped when it was
#: DIVIDED by the transmitted grid (read on the first JSAC card render, 2026-09-24).
#: `ofdm` has no mixing block at all: a comms receiver equalises the grid and never
#: forms a cube, which is exactly what makes the difference between `ofdm` and `jsac` a
#: screenshot rather than a claim.
_MIXING_LABELS = {
    "fmcw": "Mixing\n(dechirp)",
    "wideband": "Mixing\n(dechirp)",
    "ofdm": "Mixing\n(none)",
    "jsac": "Mixing\n(symbol\ndivision)",
}


def _node_label(node_id: str, label: str, block_state: Dict[str, Dict[str, Any]]) -> str:
    """The label as rendered: what the node IS DOING on this run, which is the
    difference between a diagram and a wiring list."""
    if node_id == "source":
        src = active_source(block_state)
        backend = {"environment": ".pkl frames",
                   "rt_environment": "live RT",
                   "corpus_environment": "corpus"}.get(src, src)
        return "Stored\nchannel\n(%s)" % backend
    if node_id == "dechirp":
        return _MIXING_LABELS.get(waveform_kind(block_state), label)
    if node_id == "waveform":
        kind = waveform_kind(block_state)
        # With the transmit tributary off there is still a waveform: for a unit-modulus
        # chirp the dechirp identity IS the modulation (contract section 1.2 row 3), so
        # `s_pars = H` is the FMCW case, not the absence of one. Say which it is rather
        # than dimming a block that is doing something.
        tail = ("TX chain" if block_state.get("waveform", {}).get("enabled", False)
                else "identity")
        return "Waveform\n%s\n(%s)" % (kind.upper(), tail)
    return label


# Category colors, shared by the cytoscape stylesheet below AND the legend in
# layout() -- keep these two in sync when changing either.
CATEGORY_COLORS: Dict[str, str] = {
    "source": "#0fb9b1",
    "stage": "#4b6584",
    "product": "#8854d0",
    "disabled": "#a5b1c2",
}

# Cytoscape stylesheet: color by category, dim disabled toggleable blocks. Sized
# and weighted to stay legible projected at demo scale (a shared screen, not a
# close-up laptop), not just on a dev monitor.
CYTO_STYLESHEET: List[Dict[str, Any]] = [
    {
        "selector": "node",
        "style": {
            "label": "data(label)",
            "text-valign": "center",
            "text-halign": "center",
            "color": "#fff",
            # HOSTILE ROUND 11, D2 (a BLOCKER): the previous 20 px at the then
            # ~0.6 width-bound fit zoom measured ~6 px of rendered cap-height ink on
            # `thrust1_circuit_knobs_card.png` -- half of acceptance check 17's 12 px,
            # and the same as the pre-redesign state the spec diagnosed. Two changes
            # together, because neither is enough alone: the graph lost six columns and
            # four compound regions (see _CHAIN above), and the font went up.
            # 30 px at the resulting 0.67 zoom MEASURED 13-17 px of cap-height ink on the
            # 2026-09-24 21:0x render (so ~0.65-0.85 ink per em, not the 0.5 an earlier
            # version of this comment assumed). Moving the products to their own column
            # (see _POSITIONS) widened the extent to 1705 px and dropped the zoom to
            # ~0.58, so the font goes to 34 px to hold the same rendered ink -- measured
            # again on the render, because that is the only place this product exists.
            "font-size": "34px",
            "font-weight": 600,
            "text-wrap": "wrap",
            "text-max-width": "180px",
            # 200x130 holds three 30 px lines (~36 px each) with padding. The previous
            # 160x76 did not: at 30 px every multi-line label overflowed its own box and
            # collided with its neighbours' -- visible on the 21:0x render, which is why
            # the labels above are short and these numbers are not the old ones.
            "width": "200px",
            "height": "130px",
            "shape": "round-rectangle",
            "background-color": CATEGORY_COLORS["stage"],
            "border-width": 2,
            "border-color": "#2d3a4a",
        },
    },
    {"selector": "node.source", "style": {"background-color": CATEGORY_COLORS["source"]}},
    {"selector": "node.product", "style": {"background-color": CATEGORY_COLORS["product"]}},
    {"selector": "node.disabled", "style": {"background-color": CATEGORY_COLORS["disabled"],
                                            "border-color": "#7f8c9b",
                                            "border-style": "dashed",
                                            "color": "#3c4858",
                                            "opacity": 0.7}},
    {
        "selector": "node:selected",
        "style": {"border-color": "#f7b731", "border-width": 4, "border-style": "solid"},
    },
    {"selector": "node.entry", "style": {"border-color": "#20bf6b", "border-width": 5,
                                         "border-style": "solid"}},
    # The compound-region styles that used to sit here are gone with the regions
    # themselves (see _CHAIN): four labelled boxes around a graph that has one path are
    # four statements that it has four. Their captions were also the least legible ink
    # on the card (hostile round 11, D2: ~5-6 px).
    {
        "selector": "edge",
        "style": {
            "curve-style": "bezier",
            "target-arrow-shape": "triangle",
            "arrow-scale": 1.3,
            "line-color": "#778ca3",
            "target-arrow-color": "#778ca3",
            "width": 2.5,
        },
    },
    {"selector": "edge.inactive", "style": {"line-color": "#d1d8e0",
                                            "target-arrow-color": "#d1d8e0",
                                            "line-style": "dashed",
                                            "width": 2}},
    # PRODUCT EDGES ARE ORTHOGONAL (taxi), the chain's stay straight. A straight fan from
    # one node into a stacked column clips the boxes it passes (see the _POSITIONS note);
    # a trunk-and-spur bus does not. `taxi-turn: 50%` puts the trunk halfway between the
    # two boxes -- in the 15 px gutter beside the cube for the column, and above the
    # product row for the three that tap an earlier point.
    {"selector": "edge.tap-right", "style": {"curve-style": "taxi",
                                             "taxi-direction": "rightward",
                                             "taxi-turn": "50%",
                                             "taxi-turn-min-distance": "5px"}},
    {"selector": "edge.tap-down", "style": {"curve-style": "taxi",
                                            "taxi-direction": "downward",
                                            "taxi-turn": "50%",
                                            "taxi-turn-min-distance": "5px"}},
]


def _block_active(block_state, block_id: str) -> bool:
    """Whether a REGISTRY BLOCK takes part in the run the diagram describes."""
    spec = BLOCKS_BY_ID.get(block_id)
    default = spec.enabled_default if spec is not None else True
    if block_id in _SOURCE_PRECEDENCE:
        # Only one source feeds a run, whatever the checkboxes say -- so the one the
        # runner would actually read is the active one, and the others are not drawn at
        # all any more (they were the dotted "alternative source path" edges).
        return block_id == active_source(block_state)
    if spec is not None and not spec.toggleable:
        return True
    return bool(block_state.get(block_id, {}).get("enabled", default))


def _node_active(block_state, node_id: str) -> bool:
    """A collapsed node is ACTIVE if any block it stands for is.

    "Any" and not "all", on purpose: the ADC node stands for four optional stages, and a
    run with the quantiser on and the impairments off is still a run whose ADC stage does
    something. What each member is doing is in the node's editor, one click away; what
    the graph carries is whether the chain passes through here.
    """
    if node_id in ("dechirp", "cube") and waveform_kind(block_state) == "ofdm":
        # The one class with no mixing block and therefore no cube. Structural for the
        # other two, absent here -- and drawing it lit would be the diagram claiming a
        # radar product this run cannot produce (the runner refuses those by name).
        return False
    if node_id in _STRUCTURAL_NODES:
        return True
    return any(_block_active(block_state, bid)
               for bid in node_members(node_id, block_state))


def build_elements(block_state: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Cytoscape elements for THE ONE CHAIN: seven chain nodes in the spine's order, the
    products fanned out beneath, one edge per real dataflow. No compound regions, no
    "alt" edges, no second path -- see _CHAIN.
    """
    elements: List[Dict[str, Any]] = []

    def _add_node(node_id: str, label: str, category: str) -> None:
        classes = [category]
        if not _node_active(block_state, node_id):
            classes.append("disabled")
        if node_id in _ENTRY_IDS:
            classes.append("entry")
        x, y = _POSITIONS.get(node_id, (0, 0))
        elements.append({
            "data": {"id": node_id,
                     "label": _node_label(node_id, label, block_state),
                     # Which registry block the parameter editor opens on. The editor
                     # renders every member (see `param_editor`); this is where it starts.
                     "block": node_members(node_id, block_state)[0]},
            "position": {"x": x, "y": y},
            "classes": " ".join(classes),
        })

    for node_id, label, category, _members in _CHAIN:
        _add_node(node_id, label, category)
    for pid in _PRODUCT_TAP:
        _add_node(pid, _PRODUCT_LABELS.get(pid, pid), "product")

    def _add_edge(src: str, dst: str, active: bool, extra: str = "") -> None:
        classes = ([] if active else ["inactive"]) + ([extra] if extra else [])
        elements.append({
            "data": {"source": src, "target": dst, "id": f"{src}->{dst}"},
            "classes": " ".join(classes),
        })

    # THE CHAIN'S EDGES ARE ALWAYS ACTIVE. The frame passes through every one of these
    # points on every run; what a disabled stage means is that the stage APPLIES nothing
    # there, which is what the dimmed NODE says. Dashing the edges too would draw a broken
    # chain -- on Thrust 1, where the interconnect is off, it drew the one chain in three
    # disconnected pieces (21:0x render), which is the picture this whole rewrite exists
    # to stop drawing.
    for (src, _l, _c, _m), (dst, _l2, _c2, _m2) in zip(_CHAIN, _CHAIN[1:]):
        _add_edge(src, dst, True)
    # A product's edge, on the other hand, carries real information: this product did not
    # run.
    for pid, tap in _PRODUCT_TAP.items():
        route = "tap-right" if pid in _PRODUCT_COLUMN else "tap-down"
        _add_edge(tap, pid, _node_active(block_state, pid), route)
    return elements


#: How much of a parameter's help string the editor shows without opening "more".
#: ~4 lines at 15 px in the editor column, measured on the rendered card
#: (2026-09-24). The budget is in CHARACTERS because the cut has to land on a clause
#: boundary, which is a property of the text, not of the box.
_HELP_HEAD_CHARS = 170
#: Boundaries `_help_head` is allowed to cut at, longest marker first. Each leaves a
#: complete, readable unit behind -- so nothing on the card needs an ellipsis.
_HELP_BOUNDARIES = (". ", "; ", " -- ")


def _help_head(text: str, budget: int = _HELP_HEAD_CHARS) -> str:
    """As much of one help string as fits `budget` characters while ending on a
    sentence or clause boundary -- the whole string when it fits, which is the common
    case. Never cuts inside a word; only a single unit longer than the budget on its
    own falls back to a word-boundary cut with "…" (and then "more" is always there).
    """
    text = (text or "").strip()
    if len(text) <= budget:
        return text
    cut = -1
    for marker in _HELP_BOUNDARIES:
        pos = text.rfind(marker, 0, budget + len(marker))
        # +len(marker)-1: keep the boundary punctuation itself ("...skirt of the
        # five;"), drop the space after it.
        cut = max(cut, pos + len(marker) - 1 if pos > 0 else -1)
    if cut > 0:
        # A cut at " -- " leaves the dashes dangling at the end of the visible line;
        # a cut at ". " or "; " keeps its own punctuation, which reads as written.
        return text[:cut].rstrip(" -")
    space = text.rfind(" ", 0, budget)
    return (text[:space] if space > 0 else text[:budget]).rstrip(" ,;-") + "…"


#: Chip marking the one knob a preset's card tells the presenter to turn.
_DEMOED_CHIP_STYLE = {"fontSize": "14px", "fontWeight": "bold", "color": "#ffffff",
                      "backgroundColor": "#20bf6b", "borderRadius": "3px",
                      "padding": "1px 6px", "marginLeft": "8px",
                      "whiteSpace": "nowrap"}


def param_editor(block_id: str, block_state: Dict[str, Dict[str, Any]],
                 focus: Any = None) -> List[Any]:
    """The editor column for the selected DIAGRAM NODE.

    A node can stand for several registry blocks (see `_CHAIN`), so this renders each
    member block's controls in turn, under its own heading. `block_id` may be either a
    node id or any member's block id; either way the whole node is rendered.

    `focus=(block_id, param_key)` -- the knob this preset's card tells the presenter to
    turn -- is hoisted to the TOP of the column and chipped. HOSTILE ROUND 11, D4: on 2 of
    7 presets the A/B knob was below the fold (Thrust 4's `Tessera: TSV height` sat under
    two other Tessera params, clipped mid-glyph at the container's bottom edge; Thrust 3's
    `gap_response` likewise), so the spec's "diagram, the knob and Run on the first screen"
    was met only for "diagram and Run". Ordering the column by what the screen demos --
    rather than by registry declaration order -- fixes that for every preset at once,
    including presets that do not exist yet.
    """
    node_id = resolve_node(block_id)
    members = node_members(node_id, block_state)
    members = [bid for bid in members if bid in BLOCKS_BY_ID]
    if not members:
        return [html.P("Select a block to edit its parameters.")]

    focus_block, focus_param = (focus if focus else (None, None))
    if focus_block in members:
        members = [focus_block] + [bid for bid in members if bid != focus_block]

    children: List[Any] = []
    for i, member in enumerate(members):
        if i:
            children.append(html.Hr(style={"border": "none", "borderTop": "1px solid #dfe4ea",
                                           "margin": "12px 0 8px"}))
        children.extend(_block_controls(
            member, block_state,
            focus_param=(focus_param if member == focus_block else None)))
    return children


def _block_controls(block_id: str, block_state: Dict[str, Dict[str, Any]],
                    focus_param: Any = None) -> List[Any]:
    """The controls for ONE registry block (a section of the node's editor column)."""
    spec = BLOCKS_BY_ID.get(block_id)
    if spec is None:
        return [html.P("Select a block to edit its parameters.")]

    st = block_state.get(block_id, {})
    children: List[Any] = [
        html.H4(spec.label, style={"marginBottom": "2px"}),
        html.P(spec.blurb, style={"fontSize": "16px", "color": "#576574"}),
    ]

    # Enable/disable toggle for toggleable blocks. The subspace block is a special
    # case: pipeline_runner always builds an AdaOjaBlock whenever AFE is enabled
    # (the AFE draws its combining weights from the tracker, and the results view
    # always needs the tracker's 'U' for SubspaceErrorBlock), so the checkbox would
    # otherwise be a no-op that lies about being toggleable. Render it disabled with
    # an explanatory caption instead of letting the user "turn off" something that
    # stays on.
    afe_enabled = block_state.get("afe", {}).get("enabled", True)
    subspace_locked_on = block_id == "subspace" and afe_enabled
    if spec.toggleable:
        children.append(
            dcc.Checklist(
                id={"role": "block-enabled", "block": block_id},
                options=[{"label": " Enabled", "value": "on",
                          "disabled": subspace_locked_on}],
                value=["on"] if (subspace_locked_on or st.get("enabled", spec.enabled_default)) else [],
                style={"marginBottom": "8px"},
            )
        )
        if subspace_locked_on:
            children.append(html.P(
                "(always on while AFE is enabled -- the AFE draws its combining "
                "weights from the tracker)",
                style={"fontSize": "15px", "color": "#8395a7"},
            ))
    else:
        children.append(html.P("(structural block — always on)",
                               style={"fontSize": "15px", "color": "#8395a7"}))

    params = st.get("params", {})
    ordered = list(spec.params)
    if focus_param is not None:
        ordered.sort(key=lambda ps: 0 if ps.key == focus_param else 1)
    for ps in ordered:
        val = params.get(ps.key, ps.default)
        label_children: List[Any] = [ps.label]
        if focus_param is not None and ps.key == focus_param:
            label_children.append(html.Span("this screen's knob",
                                            style=_DEMOED_CHIP_STYLE))
        children.append(html.Label(label_children,
                                   style={"fontWeight": "bold", "display": "block",
                                          "marginTop": "6px"}))
        # Control first, help underneath (was help-then-control -- the thing the
        # presenter must click was the least findable object in the column).
        cid = {"role": "block-param", "block": block_id, "param": ps.key}
        if ps.kind == "choice":
            children.append(dcc.Dropdown(
                id=cid,
                options=[{"label": str(c), "value": c} for c in (ps.choices or [])],
                value=val, clearable=False, style={"marginBottom": "4px"},
            ))
        elif ps.kind == "text":
            # Free-form string (a path). Debounced like the numbers so the store is
            # not rewritten on every keystroke.
            children.append(dcc.Input(
                id=cid, type="text", value="" if val is None else str(val),
                debounce=True, style={"width": "100%", "marginBottom": "4px"},
            ))
        else:
            step = ps.step if ps.step is not None else (1 if ps.kind == "int" else "any")
            input_kwargs = {}
            if ps.min is not None:
                input_kwargs["min"] = ps.min
            if ps.max is not None:
                input_kwargs["max"] = ps.max
            children.append(dcc.Input(
                id=cid, type="number", value=val, step=step,
                debounce=True, style={"width": "100%", "marginBottom": "4px"},
                **input_kwargs,
            ))
        if ps.help:
            # WRAPS, never clips (hostile round 11, D13): this used to be a 2-line
            # `-webkit-line-clamp`, which cut every help string mid-phrase -- the one
            # on the LNA knob Thrust 1 is entirely about ended "...visible only when
            # the signal sits near the front-end's…" on the rendered card. What shows
            # now is `_help_head`: whole sentences/clauses up to a character budget,
            # wrapped by the browser, with no ellipsis and no cut inside a word. The
            # rest (only the longest few help strings have one) stays behind "more",
            # which still holds the string in full.
            head = _help_head(ps.help)
            block = [html.P(head, style={"fontSize": "15px", "color": "#8395a7",
                                         "margin": "2px 0 0"})]
            if head != ps.help:
                block.append(html.Details([
                    html.Summary("▸ more", className="no-marker",
                                 style={"fontSize": "15px", "color": "#8395a7",
                                        "cursor": "pointer"}),
                    html.P(ps.help, style={"fontSize": "15px", "color": "#8395a7",
                                           "marginTop": "2px"}),
                ]))
            children.append(html.Div(block, style={"marginBottom": "2px"}))
    return children


def preset_notes(preset: DemoPreset) -> Any:
    """The operator's card for a loaded preset: what it shows, the knob(s) to turn live,
    what to say, and what not to say. Rendered into `preset-notes` on load."""
    def _list(title: str, items: List[str], color: str) -> Any:
        if not items:
            return None
        return html.Div([
            html.Div(title, style={"fontWeight": "bold", "color": color,
                                   "marginTop": "6px"}),
            html.Ul([html.Li(t, style={"marginBottom": "2px"}) for t in items],
                    style={"marginTop": "2px", "paddingLeft": "20px"}),
        ])

    # Name the knob the way the editor labels it ("LNA bias current (mA)"), not by
    # its param key ("lna_bias_ma"), which appears nowhere on screen.
    def _param_label(bid: str, key: str) -> str:
        return next((ps.label for ps in BLOCKS_BY_ID[bid].params if ps.key == key), key)

    knobs = [f"{BLOCKS_BY_ID[b].label} -> {_param_label(b, k)}: {how}"
             for b, k, how in preset.live_knobs]
    body = html.Div([
        html.Div(f"Loaded: {preset.label}  (Thrust {preset.thrust}, {preset.n_steps} frames)",
                 style={"fontWeight": "bold"}),
        html.P(preset.blurb, style={"marginTop": "4px", "marginBottom": "2px"}),
        _list("Turn live", knobs, "#3867d6"),
        # `resolved_say`, not `preset.say`: a bullet may carry the corpus's own v_max as
        # a token, filled in from the manifest (webapp/demo_presets.py `resolved_say`).
        _list("Say", _resolved_say(preset), "#20bf6b"),
        _list("Do NOT say or show", preset.do_not_say, "#eb3b5a"),
    ], style={"fontSize": "16px", "color": "#2d3a4a", "marginTop": "8px"})
    # Collapsed by default (was a 750px wall of text between the tab strip and the
    # diagram); the summary names the thrust so a collapsed card still orients.
    return html.Details([
        # `no-marker` (hostile round 11, D12): the native <details> triangle plus the
        # literal "▸" rendered as "▶ ▸ Presenter notes" on every card, while the
        # Results disclosures (which already suppress the native marker) render one.
        html.Summary(f"▸ Presenter notes (Thrust {preset.thrust})",
                     className="no-marker",
                     style={"fontWeight": "bold", "cursor": "pointer",
                            "fontSize": "16px", "color": "#2d3a4a"}),
        body,
    ], style={"padding": "12px", "border": "1px solid #dfe4ea",
             "borderRadius": "6px", "backgroundColor": "#f7f9fb"})


def _legend_swatch(color: str, label: str) -> Any:
    """One legend entry: a small filled square + label (readable at demo scale --
    colored text alone is low-contrast for the lighter category colors)."""
    return html.Span([
        html.Span(style={
            "display": "inline-block", "width": "12px", "height": "12px",
            "backgroundColor": color, "border": "1px solid #2d3a4a",
            "borderRadius": "3px", "marginRight": "5px", "verticalAlign": "middle",
        }),
        html.Span(label, style={"verticalAlign": "middle", "color": "#2d3a4a"}),
    ], style={"marginRight": "16px", "whiteSpace": "nowrap"})


def _legend_line(label: str, dashed: bool = False, style: str = None,
                  color: str = "#778ca3") -> Any:
    """One legend entry for an edge style: solid = active, dashed = inactive,
    dotted (with `style`/`color` overrides) = the alt-path edges from (C)."""
    line_style = style or ("dashed" if dashed else "solid")
    return html.Span([
        html.Span(style={
            "display": "inline-block", "width": "22px", "height": "0px",
            "borderTop": f"3px {line_style} {color}",
            "marginRight": "5px", "verticalAlign": "middle",
        }),
        html.Span(label, style={"verticalAlign": "middle", "color": "#2d3a4a"}),
    ], style={"marginRight": "16px", "whiteSpace": "nowrap"})


def layout() -> Any:
    """The Block Diagram tab layout.

    Structure, top to bottom: (1) a sticky one-line control bar -- preset picker,
    Run pipeline, Cancel -- so the button the whole demo depends on never sits a
    thousand px below the fold; (2) the diagram + parameter editor work row;
    (3) the edge/color legend; (4) the operator card, collapsed by default (see
    preset_notes). Every existing component id is unchanged -- app.py's callbacks
    wire to these ids and are not part of this file.
    """
    control_bar = html.Div([
        html.Label("Demo preset:", style={"fontWeight": "bold", "marginRight": "8px",
                                          "whiteSpace": "nowrap"}),
        dcc.Dropdown(
            id="preset-select",
            options=[{"label": p.label, "value": p.id} for p in PRESETS],
            value=PRESETS[0].id if PRESETS else None, clearable=False,
            style={"width": "520px", "flex": "0 0 520px"},
        ),
        # FIXED WIDTH AND NO SHRINK on every control (hostile round 11, D11): these
        # are flex items, so with the default `flex-shrink: 1` the bar re-divided
        # itself around each preset's status string and "Run pipeline" rendered one
        # line on two cards and wrapped to "Run" / "pipeline" on the other five --
        # the same button at seven widths across the deck. Only the status span
        # (flex: 1 1 auto) absorbs the leftover width now.
        html.Button("Load preset", id="preset-load", n_clicks=0,
                    style={"marginLeft": "8px", "width": "110px", "padding": "6px 0",
                           "flex": "0 0 110px", "whiteSpace": "nowrap",
                           "fontWeight": "bold", "backgroundColor": "#3867d6",
                           "color": "white", "border": "none",
                           "borderRadius": "4px", "cursor": "pointer"}),
        html.Div(className="control-divider", style={"margin": "0 16px"}),
        html.Label("Frames to run:", style={"fontWeight": "bold", "marginRight": "8px",
                                            "whiteSpace": "nowrap"}),
        # Bounded to MAX_N_STEPS on both ends of the wire: here (advisory, a typed
        # value can still exceed it) and in run_pipeline (enforced).
        dcc.Input(id="run-nsteps", type="number", value=10, min=1, max=MAX_N_STEPS,
                  step=1, style={"width": "90px", "flex": "0 0 90px"}),
        html.Button("Run pipeline", id="run-button", n_clicks=0,
                    style={"marginLeft": "12px", "width": "150px", "padding": "6px 0",
                           "flex": "0 0 150px", "whiteSpace": "nowrap",
                           "fontWeight": "bold", "backgroundColor": "#20bf6b",
                           "color": "white", "border": "none",
                           "borderRadius": "4px", "cursor": "pointer"}),
        # Enabled only while a run is in progress (see app._run_pipeline's
        # `running=`); sets a flag the simulation polls before each frame.
        html.Button("Cancel", id="cancel-button", n_clicks=0, disabled=True,
                    style={"marginLeft": "8px", "width": "100px", "padding": "6px 0",
                           "flex": "0 0 100px", "whiteSpace": "nowrap",
                           "backgroundColor": "#eb3b5a", "color": "white",
                           "border": "none", "borderRadius": "4px",
                           "cursor": "pointer"}),
    ], className="control-bar",
       style={"display": "flex", "alignItems": "center",
              "flexWrap": "nowrap", "padding": "0 12px", "overflowX": "auto"})

    # THE STATUS LINE GETS ITS OWN ROW (hostile round 13, N15). It used to be the last
    # flex item in the bar above, with `text-overflow: ellipsis` and `white-space:
    # nowrap`, absorbing whatever width the fixed-width controls left over -- which on
    # every rendered card was not enough: the line read "Preset loaded: T..." and the
    # sentence that tells the operator WHAT to press next ("Press Run pipeline.") was the
    # part that got cut. It is also where a PipelineError's message lands, so truncating
    # it truncates error messages (acceptance check 12: no truncation marks in visible
    # text). On its own full-width row it wraps instead of eliding, and the bar above
    # keeps its one-line geometry and its fixed-width buttons exactly as round 11's D11
    # fix left them.
    status_row = html.Div(
        html.Span(id="run-status", style={"fontSize": "16px", "color": "#576574"}),
        style={"padding": "6px 12px 0 12px", "minHeight": "24px",
               "lineHeight": "1.35"})

    legend = html.Div([
        _legend_swatch(CATEGORY_COLORS["source"], "source"),
        _legend_swatch(CATEGORY_COLORS["stage"], "stage"),
        _legend_swatch(CATEGORY_COLORS["product"], "product"),
        _legend_swatch(CATEGORY_COLORS["disabled"], "disabled"),
        _legend_line("active edge"),
        _legend_line("inactive edge", dashed=True),
        # One chain, so there is no "alternative source path" legend entry any more --
        # the dotted salmon edges it described are gone with the second pipeline.
        html.Span("one chain: the waveform class is the only branch; products tap the "
                  "point whose domain they read",
                  style={"marginLeft": "12px", "color": "#576574"}),
    ], style={"marginTop": "8px", "marginBottom": "8px", "fontSize": "15px",
              "display": "flex", "flexWrap": "nowrap", "overflowX": "auto",
              "alignItems": "center"})

    work_row = html.Div([
        html.Div(
            cyto.Cytoscape(
                id="block-cytoscape",
                elements=[],            # populated by callback from store
                stylesheet=CYTO_STYLESHEET,
                # fit=True re-frames the viewport to the current extent on every
                # (re)render. The full ~13-column pipeline is wide enough that a
                # width-bound fit at this panel size lands well under 1.0 (~0.6) --
                # an earlier revision floored zoom at 0.85 via minZoom, but that
                # cropped the source node on the left and ran the whole ADC-cube
                # tributary off the right edge of the canvas (confirmed in a real
                # browser render at 1600x1000), which is a worse demo defect than
                # smaller labels. No minZoom: the whole graph always fits: the
                # readability loss from the resulting ~0.6 zoom is compensated by
                # the larger font-size in CYTO_STYLESHEET instead (see its comment).
                layout={"name": "preset", "fit": True, "padding": 20},
                style={"width": "100%", "height": "620px"},
                userZoomingEnabled=True,
                userPanningEnabled=True,
            ),
            style={"flex": "1 1 1032px", "border": "1px solid #dfe4ea",
                   "borderRadius": "6px", "padding": "4px"},
        ),
        html.Div(
            id="block-param-editor",
            children=param_editor(PRODUCT_IDS[0], {}),
            style={"flex": "0 0 460px", "marginLeft": "28px", "padding": "12px",
                   "border": "1px solid #dfe4ea", "borderRadius": "6px",
                   "overflowY": "auto", "maxHeight": "640px"},
        ),
    ], style={"display": "flex", "marginTop": "12px"})

    return html.Div([
        control_bar,
        status_row,
        html.H3("Pipeline Block Diagram", style={"marginTop": "12px", "marginBottom": "4px"}),
        html.P("Click a block to edit its parameters or toggle it on/off. "
               "Dashed edges feed a disabled block.",
               style={"color": "#576574", "marginTop": "0"}),
        work_row,
        legend,
        # Demo presets (notes/DEMO_DEFENSE.md): one click configures every block, the
        # frame count and the operator notes for one thrust. Loading a preset REPLACES
        # the block state; edits made afterwards are the operator's own. The card
        # itself (collapsed disclosure, summary names the thrust) is built by
        # preset_notes() -- before any preset is loaded this div is empty.
        html.Div(id="preset-notes", style={"marginTop": "16px"}),
        dcc.Loading(html.Div(id="run-sink", style={"display": "none"}), type="default"),
    ])
