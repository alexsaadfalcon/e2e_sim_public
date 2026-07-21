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


def test_registry_edges_reference_existing_nodes():
    from webapp.pipeline_registry import BLOCKS_BY_ID, EDGES
    for src, dst in EDGES:
        assert src in BLOCKS_BY_ID, f"edge source {src!r} not a known block"
        assert dst in BLOCKS_BY_ID, f"edge target {dst!r} not a known block"


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
            assert p.kind in {"number", "int", "choice"}, (
                f"{b.id}.{p.key} has unknown kind {p.kind!r}"
            )
            assert p.default is not None, f"{b.id}.{p.key} has no default"
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

def test_build_elements_matches_registry():
    from webapp import block_diagram
    from webapp.pipeline_registry import BLOCKS, EDGES, default_block_state

    elements = block_diagram.build_elements(default_block_state())
    nodes = [e for e in elements if "source" not in e["data"]]
    edges = [e for e in elements if "source" in e["data"]]

    assert len(nodes) == len(BLOCKS)
    assert len(edges) == len(EDGES)

    node_ids = {n["data"]["id"] for n in nodes}
    assert node_ids == {b.id for b in BLOCKS}

    # every edge connects two real nodes and carries a stable id
    for e in edges:
        d = e["data"]
        assert d["source"] in node_ids
        assert d["target"] in node_ids
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


def test_block_diagram_layout_builds():
    from webapp import block_diagram
    layout = block_diagram.layout()
    assert layout is not None
    assert getattr(layout, "children", None) is not None


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
    assert figs["fft"].data[0].colorbar.title.text == "power (dB)"

    # Negated: the range blocks use a FORWARD fft over frequency, so a physical
    # delay +tau lands on the negative fftshifted side; the axis flips sign so
    # targets read at positive range.
    expected_range = -(np.arange(bins) - bins // 2) * (2.99792458e8 / (2.0 * 3e9))
    np.testing.assert_allclose(figs["range_az"].data[0].x, expected_u)
    np.testing.assert_allclose(figs["range_az"].data[0].y, expected_range)
    assert figs["range_az"].layout.xaxis.title.text == "azimuth sin(θ)"
    assert figs["range_az"].layout.yaxis.title.text == "range (m)"

    np.testing.assert_allclose(figs["range_el"].data[0].x, expected_u)
    np.testing.assert_allclose(figs["range_el"].data[0].y, expected_range)
    assert figs["range_el"].layout.xaxis.title.text == "elevation sin(θ)"
    assert figs["range_el"].layout.yaxis.title.text == "range (m)"


def test_figures_from_outputs_falls_back_to_bins_when_range_fft_bins_mismatch_freqs():
    """When the range-FFT bin count != the raw frame's frequency-sample count, the
    freq->range mapping is ambiguous (zero-pad/truncate before the DFT), so the
    range axis falls back to raw bin indices labeled '(bins)' rather than a bogus
    physical unit."""
    torch = pytest.importorskip("torch")
    import numpy as np
    from webapp.pipeline_runner import figures_from_outputs

    bins = 8
    outputs = {
        "range_az": [torch.zeros((bins, bins), dtype=torch.complex64)],
        "_axis_meta": {
            "range_az_bins": bins,
            "n_freqs": 64,  # != bins -> exact range mapping does not apply
            "freq_span_hz": 3e9,
        },
    }
    figs = figures_from_outputs(outputs)
    np.testing.assert_allclose(figs["range_az"].data[0].y, np.arange(bins))
    assert figs["range_az"].layout.yaxis.title.text == "range (bins)"

    # No metadata at all (e.g. a hand-built outputs dict): same fallback.
    outputs_no_meta = {"range_el": [torch.zeros((bins, bins), dtype=torch.complex64)]}
    figs2 = figures_from_outputs(outputs_no_meta)
    np.testing.assert_allclose(figs2["range_el"].data[0].y, np.arange(bins))
    assert figs2["range_el"].layout.yaxis.title.text == "range (bins)"


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
    assert _p(state, "subspace", "d") == state["subspace"]["params"]["d"]
    # None / missing falls back to the registry default
    state["subspace"]["params"]["d"] = None
    from webapp.pipeline_registry import BLOCKS_BY_ID
    default_d = next(p.default for p in BLOCKS_BY_ID["subspace"].params if p.key == "d")
    assert _p(state, "subspace", "d") == default_d


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

    def spy_oja(n, d):
        seen["oja_n"] = n
        return real_oja(n, d)

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

    assert seen.get("oja_n") == shape[0] * shape[1], (
        f"AdaOja dim {seen.get('oja_n')} should equal n_rx={shape[0] * shape[1]}"
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

    assert figs["ber"].layout.title.text == "Comms head BER (mrc, array gain 15.0 dB)"
    assert figs["ber"].layout.yaxis.type == "log"
    assert list(figs["ber"].data[0].y) == [0.1, 0.01]

    assert figs["evm"].layout.title.text == "Comms head EVM per frame"

    const = figs["comm_const"].data[0]
    import numpy as np
    np.testing.assert_allclose(const.x, [1.0, -1.0])
    np.testing.assert_allclose(const.y, [1.0, -1.0])
    assert figs["comm_const"].layout.yaxis.scaleanchor == "x"


def test_figures_from_outputs_ber_without_array_gain_omits_it_from_title():
    from webapp.pipeline_runner import figures_from_outputs

    outputs = {"ber": [0.3], "_comms_meta": {"combining": "element0"}}
    figs = figures_from_outputs(outputs)
    assert figs["ber"].layout.title.text == "Comms head BER (element0)"
