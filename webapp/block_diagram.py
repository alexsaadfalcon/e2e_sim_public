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

from webapp.pipeline_registry import BLOCKS, BLOCKS_BY_ID, EDGES, PRODUCT_IDS

# Manual positions so the graph reads left-to-right as a pipeline. Every id in
# webapp.pipeline_registry.BLOCKS must appear here (see test_webapp.py), or it
# falls back to the origin and overlaps other nodes.
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
    "range_profile": (1020, 270),
    "subspace_err": (1020, 320),
    "comms": (1020, 420),

    # TX-time tributary (above the main chain): waveform -> PA -> modulate,
    # which bridges back into the main chain at rffe.
    "waveform": (0, 40),
    "tx_pa": (150, 40),
    "modulate": (300, 40),

    # Live-ray-tracing source, an alternative to "environment" (below the main
    # chain); it also feeds modulate and rffe directly.
    "rt_environment": (0, 280),

    # RX-time ADC-cube tributary (below the main chain), branching off
    # interconnect: dechirp -> impairment -> quantizer -> products.
    "dechirp": (600, 280),
    "impairment": (800, 280),
    "quantizer": (1000, 280),
    "radar_cube": (1240, 200),
    "detector": (1240, 300),
    "sink": (1240, 400),
}

# Cytoscape stylesheet: color by category, dim disabled toggleable blocks.
CYTO_STYLESHEET: List[Dict[str, Any]] = [
    {
        "selector": "node",
        "style": {
            "label": "data(label)",
            "text-valign": "center",
            "text-halign": "center",
            "color": "#fff",
            "font-size": "11px",
            "text-wrap": "wrap",
            "text-max-width": "120px",
            "width": "150px",
            "height": "56px",
            "shape": "round-rectangle",
            "background-color": "#4b6584",
            "border-width": 2,
            "border-color": "#2d3a4a",
        },
    },
    {"selector": "node.source", "style": {"background-color": "#0fb9b1"}},
    {"selector": "node.product", "style": {"background-color": "#8854d0"}},
    {"selector": "node.disabled", "style": {"background-color": "#a5b1c2",
                                            "border-color": "#7f8c9b",
                                            "color": "#3c4858"}},
    {
        "selector": "node:selected",
        "style": {"border-color": "#f7b731", "border-width": 4},
    },
    {
        "selector": "edge",
        "style": {
            "curve-style": "bezier",
            "target-arrow-shape": "triangle",
            "line-color": "#778ca3",
            "target-arrow-color": "#778ca3",
            "width": 2,
        },
    },
    {"selector": "edge.inactive", "style": {"line-color": "#d1d8e0",
                                            "target-arrow-color": "#d1d8e0",
                                            "line-style": "dashed"}},
]


def build_elements(block_state: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build cytoscape elements (nodes + edges) reflecting enabled/disabled state."""
    elements: List[Dict[str, Any]] = []
    for b in BLOCKS:
        enabled = block_state.get(b.id, {}).get("enabled", b.enabled_default)
        classes = [b.category]
        if b.toggleable and not enabled:
            classes.append("disabled")
        x, y = _POSITIONS.get(b.id, (0, 0))
        elements.append({
            "data": {"id": b.id, "label": b.label},
            "position": {"x": x, "y": y},
            "classes": " ".join(classes),
        })

    for src, dst in EDGES:
        src_on = block_state.get(src, {}).get("enabled", True)
        dst_on = block_state.get(dst, {}).get("enabled", True)
        cls = "" if (src_on and dst_on) else "inactive"
        elements.append({
            "data": {"source": src, "target": dst, "id": f"{src}->{dst}"},
            "classes": cls,
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

    # Enable/disable toggle for toggleable blocks.
    if spec.toggleable:
        children.append(
            dcc.Checklist(
                id={"role": "block-enabled", "block": block_id},
                options=[{"label": " Enabled", "value": "on"}],
                value=["on"] if st.get("enabled", spec.enabled_default) else [],
                style={"marginBottom": "8px"},
            )
        )
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
            children.append(dcc.Input(
                id=cid, type="number", value=val, step=step,
                debounce=True, style={"width": "100%", "marginBottom": "4px"},
            ))
    return children


def layout() -> Any:
    """The Block Diagram tab layout."""
    legend = html.Div([
        html.Span("source ", style={"color": "#0fb9b1", "fontWeight": "bold"}),
        html.Span("stage ", style={"color": "#4b6584", "fontWeight": "bold"}),
        html.Span("product ", style={"color": "#8854d0", "fontWeight": "bold"}),
        html.Span("disabled", style={"color": "#a5b1c2", "fontWeight": "bold"}),
    ], style={"marginBottom": "6px", "fontSize": "12px"})

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
                    layout={"name": "preset"},
                    style={"width": "100%", "height": "440px"},
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
                       "overflowY": "auto", "maxHeight": "440px"},
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
