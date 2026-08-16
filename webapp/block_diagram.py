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

from webapp.pipeline_registry import (
    BLOCKS, BLOCKS_BY_ID, EDGES, PRODUCT_IDS, normalize_edge,
)

# Manual positions so the graph reads left-to-right as a pipeline. Every id in
# webapp.pipeline_registry.BLOCKS must appear here (see test_webapp.py), or it
# falls back to the origin and overlaps other nodes. Product column keeps a
# consistent 100px pitch; the ADC-cube tributary gets its own band (y=580-740)
# well clear of the main chain and product column (node boxes are 160x60, see
# CYTO_STYLESHEET, so anything closer than 100px on a shared axis can touch).
_POSITIONS = {
    # Main radar/subspace chain: environment -> rffe -> interconnect -> afe ->
    # subspace -> products (fanned out on the right).
    "environment": (0, 160),
    "rffe": (200, 160),
    "interconnect": (400, 160),
    "afe": (600, 160),
    "subspace": (800, 160),
    "fft": (1020, 20),
    "range_az": (1020, 120),
    "range_el": (1020, 220),
    "range_profile": (1020, 320),
    "subspace_err": (1020, 420),
    "comms": (1020, 520),

    # TX-time tributary (above the main chain): waveform -> PA -> modulate,
    # which bridges back into the main chain at rffe. Pitch bumped 150 -> 170
    # (deviation from the original untouched values): at 150 the 160px-wide
    # boxes actually overlapped by 10px -- caught by the new geometric-overlap
    # test in (E), not the original tuple-equality one.
    "waveform": (0, 40),
    "tx_pa": (170, 40),
    "modulate": (340, 40),

    # Live-ray-tracing source, an alternative to "environment" (below the main
    # chain); it also feeds modulate and rffe directly.
    "rt_environment": (0, 280),

    # RX-time ADC-cube tributary, on its own band well below everything else,
    # branching off interconnect: dechirp -> impairment -> quantizer -> products.
    "dechirp": (600, 620),
    "impairment": (800, 620),
    "quantizer": (1000, 620),
    "radar_cube": (1240, 580),
    "detector": (1240, 660),
    "sink": (1240, 740),
}

# Compound region groups (Cytoscape native `data.parent`; see build_elements).
# Parent/container nodes get no explicit position -- Cytoscape auto-computes
# the compound box from the children's preset positions above.
_GROUPS: Dict[str, Dict[str, Any]] = {
    "grp_txtime": {
        "members": ["waveform", "tx_pa", "modulate"],
        "label": "TX-time waveform tributary (opt-in)",
    },
    "grp_main": {
        "members": ["environment", "rt_environment", "rffe", "interconnect",
                    "afe", "subspace"],
        "label": "Main radar/subspace chain",
    },
    "grp_products": {
        "members": ["fft", "range_az", "range_el", "range_profile",
                    "subspace_err", "comms"],
        "label": "Frequency-domain products",
    },
    "grp_adc": {
        "members": ["dechirp", "impairment", "quantizer", "radar_cube",
                    "detector", "sink"],
        "label": "ADC-cube chain - mutually exclusive with the products above",
    },
}

_BLOCK_TO_GROUP: Dict[str, str] = {
    bid: gid for gid, spec in _GROUPS.items() for bid in spec["members"]
}

# Entry points into the diagram: a "start here" affordance for the two source
# blocks that begin the two mutually-exclusive main-chain paths.
_ENTRY_IDS = {"environment", "rt_environment"}

# Diagram-only label overrides for clean line breaks. Deliberately local to
# this module -- webapp.pipeline_registry.BlockSpec.label is also used by the
# param editor, which should keep the single-line label.
_DIAGRAM_LABEL_OVERRIDES: Dict[str, str] = {
    "rt_environment": "RT Environment\n(live ray tracing)",
    "modulate": "Modulate\n(TX -> channel)",
}

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
            "font-size": "12px",
            "font-weight": 600,
            "text-wrap": "wrap",
            "text-max-width": "130px",
            "width": "160px",
            "height": "60px",
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
    # Compound group containers (see _GROUPS / build_elements): a faint labeled
    # box behind their children, drawn from Cytoscape's native :parent pseudo-class.
    {
        "selector": "node:parent",
        "style": {
            "background-opacity": 0.06,
            "border-style": "dashed",
            "border-width": 1,
            "border-color": "#576574",
            "text-valign": "top",
            "text-halign": "left",
            "font-size": "11px",
            "font-weight": "bold",
            "color": "#576574",
            "padding": "18px",
        },
    },
    # grp_adc's label calls out a real constraint (mutually exclusive with the
    # frequency-domain products), not just a region name like the other three
    # group labels -- give it the same amber the app already uses for :selected
    # emphasis, so it reads as a warning rather than blending in as grey chrome.
    {
        "selector": "#grp_adc",
        "style": {"color": "#f7b731", "font-weight": "bold"},
    },
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
    {"selector": "edge.alt-path", "style": {"line-color": "#e17055",
                                            "target-arrow-color": "#e17055",
                                            "line-style": "dotted",
                                            "width": 2.5}},
]


def build_elements(block_state: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build cytoscape elements (nodes + edges) reflecting enabled/disabled state."""
    elements: List[Dict[str, Any]] = []

    # Compound region containers first (see _GROUPS). No "position" -- Cytoscape
    # auto-computes the compound box from the children's preset positions.
    for gid, spec in _GROUPS.items():
        elements.append({"data": {"id": gid, "label": spec["label"]}, "classes": "group"})

    for b in BLOCKS:
        enabled = block_state.get(b.id, {}).get("enabled", b.enabled_default)
        classes = [b.category]
        if b.toggleable and not enabled:
            classes.append("disabled")
        if b.id in _ENTRY_IDS:
            classes.append("entry")
        x, y = _POSITIONS.get(b.id, (0, 0))
        data = {"id": b.id, "label": _DIAGRAM_LABEL_OVERRIDES.get(b.id, b.label)}
        if b.id in _BLOCK_TO_GROUP:
            data["parent"] = _BLOCK_TO_GROUP[b.id]
        elements.append({
            "data": data,
            "position": {"x": x, "y": y},
            "classes": " ".join(classes),
        })

    for edge in EDGES:
        src, dst, kind = normalize_edge(edge)
        src_on = block_state.get(src, {}).get("enabled", True)
        dst_on = block_state.get(dst, {}).get("enabled", True)
        classes = [] if (src_on and dst_on) else ["inactive"]
        if kind == "alt":
            classes.append("alt-path")
        elements.append({
            "data": {"source": src, "target": dst, "id": f"{src}->{dst}"},
            "classes": " ".join(classes),
        })
    return elements


def param_editor(block_id: str, block_state: Dict[str, Dict[str, Any]]) -> List[Any]:
    """Build the editor controls for a single selected block."""
    spec = BLOCKS_BY_ID.get(block_id)
    if spec is None:
        return [html.P("Select a block to edit its parameters.")]

    st = block_state.get(block_id, {})
    children: List[Any] = [
        html.H4(spec.label, style={"marginBottom": "2px"}),
        html.P(spec.blurb, style={"fontSize": "12px", "color": "#576574"}),
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
                style={"fontSize": "11px", "color": "#8395a7"},
            ))
    else:
        children.append(html.P("(structural block — always on)",
                               style={"fontSize": "11px", "color": "#8395a7"}))

    params = st.get("params", {})
    for ps in spec.params:
        val = params.get(ps.key, ps.default)
        children.append(html.Label(ps.label, style={"fontWeight": "bold",
                                                     "display": "block",
                                                     "marginTop": "6px"}))
        if ps.help:
            children.append(html.Span(ps.help, style={"fontSize": "11px",
                                                       "color": "#8395a7"}))
        cid = {"role": "block-param", "block": block_id, "param": ps.key}
        if ps.kind == "choice":
            children.append(dcc.Dropdown(
                id=cid,
                options=[{"label": str(c), "value": c} for c in (ps.choices or [])],
                value=val, clearable=False, style={"marginBottom": "4px"},
            ))
        else:
            step = ps.step if ps.step is not None else (1 if ps.kind == "int" else "any")
            input_kwargs = {}
            if ps.min is not None:
                input_kwargs["min"] = ps.min
            children.append(dcc.Input(
                id=cid, type="number", value=val, step=step,
                debounce=True, style={"width": "100%", "marginBottom": "4px"},
                **input_kwargs,
            ))
    return children


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
    """The Block Diagram tab layout."""
    legend = html.Div([
        _legend_swatch(CATEGORY_COLORS["source"], "source"),
        _legend_swatch(CATEGORY_COLORS["stage"], "stage"),
        _legend_swatch(CATEGORY_COLORS["product"], "product"),
        _legend_swatch(CATEGORY_COLORS["disabled"], "disabled"),
        _legend_line("active edge"),
        _legend_line("inactive edge", dashed=True),
        _legend_line("alternative source path - only one used per run",
                     style="dotted", color="#e17055"),
    ], style={"marginBottom": "8px", "fontSize": "12px", "display": "flex",
              "flexWrap": "wrap", "alignItems": "center"})

    return html.Div([
        html.H3("Pipeline Block Diagram"),
        html.P("Click a block to edit its parameters or toggle it on/off. "
               "Dashed edges feed a disabled block. Then hit Run pipeline.",
               style={"color": "#576574"}),
        legend,
        html.Div([
            html.Div(
                cyto.Cytoscape(
                    id="block-cytoscape",
                    elements=[],            # populated by callback from store
                    stylesheet=CYTO_STYLESHEET,
                    # fit=True re-frames the viewport to the current extent on
                    # every (re)render. Height is sized to the diagram's OWN
                    # aspect (~1400x840 layout units): at typical panel widths
                    # the fit zoom is width-bound (~0.7), so the fitted content
                    # is ~600px tall -- an 820px canvas just added a dead-space
                    # band below the diagram (flagged in the showcase capture).
                    layout={"name": "preset", "fit": True, "padding": 20},
                    style={"width": "100%", "height": "620px"},
                    userZoomingEnabled=True,
                    userPanningEnabled=True,
                ),
                style={"flex": "3", "border": "1px solid #dfe4ea",
                       "borderRadius": "6px", "padding": "4px"},
            ),
            html.Div(
                id="block-param-editor",
                children=param_editor(PRODUCT_IDS[0], {}),
                style={"flex": "1", "marginLeft": "12px", "padding": "10px",
                       "border": "1px solid #dfe4ea", "borderRadius": "6px",
                       "minWidth": "240px", "maxWidth": "320px",
                       "overflowY": "auto", "maxHeight": "820px"},
            ),
        ], style={"display": "flex"}),
        html.Div([
            html.Label("Frames to run (n_steps): ",
                       style={"fontWeight": "bold", "marginRight": "6px"}),
            dcc.Input(id="run-nsteps", type="number", value=10, min=1, step=1,
                      style={"width": "80px"}),
            html.Button("Run pipeline", id="run-button", n_clicks=0,
                        style={"marginLeft": "12px", "padding": "6px 16px",
                               "fontWeight": "bold", "backgroundColor": "#20bf6b",
                               "color": "white", "border": "none",
                               "borderRadius": "4px", "cursor": "pointer"}),
            html.Span(id="run-status", style={"marginLeft": "12px",
                                              "color": "#576574"}),
        ], style={"marginTop": "12px"}),
        dcc.Loading(html.Div(id="run-sink", style={"display": "none"}), type="default"),
    ])
