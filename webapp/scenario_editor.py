"""
Scenario tab: place/edit antenna nodes and objects on a 2D (x/y) map, edit
top-level scenario settings (base scene, frequency plan, num_frames), validate,
load/save JSON via :mod:`e2e.scenario`, and trigger offline frame generation
through the ``scenario_runner`` CLI (in --dry-run mode).

``e2e.scenario`` is dependency-free (no torch / sionna), so importing it here is
safe for the app shell. The Generate-frames action shells out to
``python -m e2e.environment.scenario_runner --scenario <path> --dry-run`` and
degrades gracefully if that module doesn't exist yet.
"""

from __future__ import annotations

from typing import Any, Dict, List

import plotly.graph_objects as go
from dash import dcc, html

# Safe to import at module top: scenario.py imports nothing heavy.
from e2e.scenario import (
    REFERENCE_SCENARIOS,
    Scenario,
    NodeRole,
)

# Marker color per node role + objects.
ROLE_COLORS = {
    NodeRole.RADAR.value: "#eb3b5a",
    NodeRole.COMM_TX.value: "#2d98da",
    NodeRole.COMM_RX.value: "#20bf6b",
    NodeRole.MONITOR.value: "#8854d0",
}
OBJECT_COLOR = "#f7b731"


def default_scenario_json() -> str:
    """JSON for the first reference scenario, used as the editor's initial value."""
    first = next(iter(REFERENCE_SCENARIOS.values()))()
    return first.to_json()


def scenario_from_json_safe(text: str):
    """Parse JSON -> Scenario. Returns (scenario, error_message)."""
    try:
        return Scenario.from_json(text), None
    except Exception as e:  # noqa: BLE001 - surface any parse/shape error to UI
        return None, f"{type(e).__name__}: {e}"


def map_figure(scenario: Scenario) -> go.Figure:
    """2D scatter of node and object positions (x vs y), role-colored."""
    fig = go.Figure()

    # group nodes by role for a clean legend
    by_role: Dict[str, List] = {}
    for n in scenario.nodes:
        by_role.setdefault(n.role.value, []).append(n)
    for role, nodes in by_role.items():
        fig.add_trace(go.Scatter(
            x=[n.position[0] for n in nodes],
            y=[n.position[1] for n in nodes],
            mode="markers+text",
            text=[n.name for n in nodes],
            textposition="top center",
            marker=dict(size=16, symbol="square",
                        color=ROLE_COLORS.get(role, "#778ca3"),
                        line=dict(width=1, color="#2d3a4a")),
            name=role,
        ))

    if scenario.objects:
        fig.add_trace(go.Scatter(
            x=[o.position[0] for o in scenario.objects],
            y=[o.position[1] for o in scenario.objects],
            mode="markers+text",
            text=[o.name for o in scenario.objects],
            textposition="bottom center",
            marker=dict(size=12, symbol="circle", color=OBJECT_COLOR,
                        line=dict(width=1, color="#2d3a4a")),
            name="objects",
        ))

    fig.update_layout(
        title=f"Scene layout: {scenario.name}  (base_scene={scenario.base_scene})",
        xaxis_title="x (m)", yaxis_title="y (m)",
        height=460, margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", y=-0.18),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def summarize(scenario: Scenario) -> Any:
    """A compact human summary of the scenario for the side panel."""
    rows = [
        html.Tr([html.Td("name"), html.Td(scenario.name)]),
        html.Tr([html.Td("base_scene"), html.Td(scenario.base_scene)]),
        html.Tr([html.Td("num_frames"), html.Td(str(scenario.num_frames))]),
        html.Tr([html.Td("nodes"), html.Td(str(len(scenario.nodes)))]),
        html.Tr([html.Td("objects"), html.Td(str(len(scenario.objects)))]),
        html.Tr([html.Td("isac"), html.Td(str(scenario.is_isac))]),
        html.Tr([html.Td("carrier_hz"), html.Td(f"{scenario.frequency.carrier_hz:g}")]),
        html.Tr([html.Td("freq band"),
                 html.Td(f"{scenario.frequency.start_hz:g} - "
                         f"{scenario.frequency.stop_hz:g} "
                         f"({scenario.frequency.num_freqs} pts)")]),
    ]
    return html.Table(rows, style={"fontSize": "13px", "width": "100%"})


def layout() -> Any:
    """The Scenario tab layout."""
    ref_options = [{"label": k, "value": k} for k in REFERENCE_SCENARIOS]
    return html.Div([
        html.H3("Scenario Scheduling"),
        html.P("Edit the scenario JSON (schema = e2e.scenario.Scenario), preview "
               "the 2D layout, validate, and generate frames.",
               style={"color": "#576574"}),
        html.Div([
            # Left: controls + JSON editor
            html.Div([
                html.Div([
                    html.Label("Load reference: ", style={"fontWeight": "bold"}),
                    dcc.Dropdown(id="ref-scenario-dropdown", options=ref_options,
                                 value=ref_options[0]["value"], clearable=False,
                                 style={"width": "220px", "display": "inline-block",
                                        "verticalAlign": "middle"}),
                    html.Button("Load", id="load-ref-button", n_clicks=0,
                                style={"marginLeft": "8px"}),
                ], style={"marginBottom": "8px"}),
                html.Label("Scenario JSON", style={"fontWeight": "bold"}),
                dcc.Textarea(
                    id="scenario-json",
                    value=default_scenario_json(),
                    style={"width": "100%", "height": "360px",
                           "fontFamily": "monospace", "fontSize": "12px"},
                ),
                html.Div([
                    html.Button("Render / Preview", id="render-button", n_clicks=0,
                                style={"marginRight": "8px"}),
                    html.Button("Validate", id="validate-button", n_clicks=0,
                                style={"marginRight": "8px"}),
                    html.Button("Generate frames", id="generate-button", n_clicks=0,
                                style={"backgroundColor": "#fa8231", "color": "white",
                                       "border": "none", "borderRadius": "4px",
                                       "padding": "6px 12px", "cursor": "pointer"}),
                ], style={"marginTop": "8px"}),
                dcc.Download(id="scenario-download"),
                html.Div([
                    html.Button("Download JSON", id="download-button", n_clicks=0,
                                style={"marginRight": "8px", "marginTop": "8px"}),
                    dcc.Upload(id="upload-json",
                               children=html.Button("Upload JSON"),
                               multiple=False,
                               style={"display": "inline-block"}),
                ]),
            ], style={"flex": "1", "minWidth": "380px", "marginRight": "12px"}),

            # Right: map + summary + messages
            html.Div([
                dcc.Loading(dcc.Graph(id="scenario-map"), type="default"),
                html.Div(id="scenario-summary", style={"marginTop": "8px"}),
                html.H4("Validation", style={"marginBottom": "2px"}),
                html.Div(id="validation-output",
                         style={"fontSize": "13px", "minHeight": "24px"}),
                html.H4("Generate frames output", style={"marginBottom": "2px"}),
                html.Pre(id="generate-output",
                         style={"fontSize": "12px", "whiteSpace": "pre-wrap",
                                "background": "#f1f2f6", "padding": "8px",
                                "borderRadius": "4px", "maxHeight": "200px",
                                "overflowY": "auto"}),
            ], style={"flex": "1.2"}),
        ], style={"display": "flex"}),
    ])
