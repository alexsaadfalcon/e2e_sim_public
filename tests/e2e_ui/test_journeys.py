"""User-journey tests: each drives the real web UI through the exact click sequence
a user performs, checking state at each step (not just at the end).

These are procedural/Screenplay-style, deliberately NOT fine-grained asserts: a test
reads as "the actor opens the app, goes to the Scenario tab, loads a reference,
validates it, previews the map" -- and any failure is annotated with the actor and
the named step that broke. See README.md for the pattern.

Run them with:  RUN_BROWSER=1 pytest tests/e2e_ui
"""

from __future__ import annotations

import pytest

import _ui_app as ui
from _ui_screenplay import Ensure, at_least, contains, equals, is_true

# All journeys drive a real headless browser -> gate them behind RUN_BROWSER=1
# (gate lives in tests/conftest.py alongside sionna/slow/gui). Module-level so it
# binds during collection.
pytestmark = pytest.mark.browser


# ----------------------------------------------------------------- journey 1: tabs
def test_journey_navigate_all_tabs(actor, base_url):
    """A user opens the app and walks through all three tabs, seeing each one's
    landmark content. (Pure shell -- no torch needed.)

    The block-diagram interaction comes first: it is the default tab on load, and
    dcc.Tabs unmounts an inactive tab's content, so the cytoscape graph is populated
    and clickable here on the initial render.
    """
    actor.attempts_to(
        ui.open_app(base_url),
        Ensure.that(ui.title_text(), contains("End-to-End Simulator"),
                    "the app title is shown"),
        ui.select_block("comms"),
        Ensure.that(ui.param_editor_text(), contains("Comms Head"),
                    "clicking the comms block opens its parameter editor"),
        ui.go_to_tab("scenario"),
        Ensure.that(ui.scenario_json_value(), contains("nodes"),
                    "the Scenario tab shows a JSON editor pre-filled with a scenario"),
        ui.go_to_tab("results"),
        Ensure.that(ui.results_content_visible(), is_true(),
                    "the Results tab renders its content area"),
    )


# ------------------------------------------------- journey 2: author a scenario
def test_journey_author_and_preview_scenario(actor, base_url):
    """A user selects a reference scenario, loads it, validates it, and previews the
    2D map -- the core 'scenario scheduling' workflow. (No torch needed.)"""
    actor.attempts_to(
        ui.open_app(base_url),
        ui.go_to_tab("scenario"),
        ui.select_reference_scenario("etoile_radar"),
        ui.load_reference_scenario(),
        Ensure.that(ui.scenario_json_value(), contains("etoile"),
                    "loading pulls the selected reference into the editor"),
        ui.validate_scenario(),
        Ensure.that(ui.validation_text(), contains("no problems"),
                    "a reference scenario validates as OK"),
        ui.preview_scenario(),
    )
    # preview populated the summary (asked as a follow-up so the matcher stays simple)
    assert actor.asks(ui.summary_text()).strip(), "expected a non-empty summary"


# --------------------------------------- journey 3: invalid scenario (negative)
def test_journey_invalid_scenario_is_reported_not_crashed(actor, base_url):
    """A user types malformed JSON and validates: the app must surface an error
    gracefully (not crash, not silently pass). Negative-path journey."""
    actor.attempts_to(
        ui.open_app(base_url),
        ui.go_to_tab("scenario"),
        ui.set_scenario_json("{ this is not valid json "),
        ui.validate_scenario(),
        Ensure.that(ui.validation_text(), contains("Cannot parse"),
                    "malformed JSON is reported as a parse error, not accepted"),
    )


# ------------------------------------------------- journey 4: run the pipeline
def test_journey_configure_and_run_pipeline(actor, base_url, run_capable):
    """A user sets the frame count on the Block Diagram tab, runs the pipeline, and
    lands on the Results tab with rendered figures. Needs torch + munich frames."""
    if not run_capable:
        pytest.skip("pipeline Run needs torch + munich.pkl frames")

    actor.attempts_to(
        ui.open_app(base_url),
        ui.set_frames_to_run(2),
        ui.run_pipeline(),
        Ensure.that(ui.results_graph_count(), at_least(1),
                    "the Results tab shows at least one product figure after the run"),
        Ensure.that(ui.results_content_visible(), is_true(),
                    "the app landed on the Results tab"),
    )


# --------------------------------------- journey 5: enabling a block persists
def test_journey_block_enable_persists(actor, base_url):
    """A user enables a disabled-by-default block, then navigates away and back:
    the enabled state must have been persisted into the store, not just the DOM of
    the editor that was open at the time. (No torch needed.)"""
    actor.attempts_to(
        ui.open_app(base_url),
        ui.enable_block("interconnect"),
        Ensure.that(ui.open_block_enabled(), is_true(),
                    "toggling enables the block"),
        ui.select_block("afe"),
        ui.select_block("interconnect"),
        Ensure.that(ui.open_block_enabled(), is_true(),
                    "the enabled state persisted across re-selection"),
    )


# -------------------------------------------- journey 6: editing a param persists
def test_journey_param_edit_persists(actor, base_url):
    """A user edits a block's number parameter, navigates away and back: the edited
    value must have been persisted into the store. (No torch needed.)"""
    actor.attempts_to(
        ui.open_app(base_url),
        ui.select_block("subspace"),
    )
    current = actor.asks(ui.open_block_number_value())
    # Use a value different from whatever is currently shown (default is 16).
    new_value = 8 if current != "8" else 12
    actor.attempts_to(
        ui.set_open_block_number_param(new_value),
        ui.select_block("fft"),
        ui.select_block("subspace"),
        Ensure.that(ui.open_block_number_value(), equals(str(new_value)),
                    "the edited param persisted in the store"),
    )


# ------------------------------------------------- journey 7: comms head via UI
def test_journey_comms_head_via_ui(actor, base_url, run_capable):
    """A user enables the (opt-in, disabled-by-default) comms head, runs the
    pipeline, and sees a BER figure among the results. Needs torch + munich frames."""
    if not run_capable:
        pytest.skip("comms head run needs torch + munich.pkl frames")

    actor.attempts_to(
        ui.open_app(base_url),
        ui.enable_block("comms"),
        ui.set_frames_to_run(2),
        ui.run_pipeline(),
        Ensure.that(ui.results_text(), contains("BER"),
                    "enabling the comms head produces a BER figure in the results"),
    )


# ------------------------------------------------ journey 8: load every demo preset
def _preset_ids():
    from webapp.demo_presets import PRESETS
    return [p.id for p in PRESETS]


@pytest.mark.parametrize("preset_id", _preset_ids())
def test_journey_load_demo_preset(actor, base_url, preset_id):
    """The presenter picks a demo preset and loads it: the operator card appears,
    the frame count becomes the preset's, and the status line says what to do next.
    (No torch needed -- loading only rewrites the block-state store.)"""
    from webapp.demo_presets import PRESETS_BY_ID
    preset = PRESETS_BY_ID[preset_id]
    actor.attempts_to(
        ui.open_app(base_url),
        ui.select_preset(preset.label),
        ui.load_preset(),
        Ensure.that(ui.preset_notes_text(), contains(preset.label),
                    "the operator card names the loaded preset"),
        Ensure.that(ui.preset_notes_text(), contains("Do NOT say"),
                    "the card carries its do-not-say list"),
        Ensure.that(ui.frames_to_run_value(), equals(str(preset.n_steps)),
                    "the frame count is the preset's"),
        Ensure.that(ui.run_status_text(), contains("Preset loaded"),
                    "the status line tells the presenter to press Run"),
    )


# --------------------------------------- journey 9: a Thrust 5 preset end to end
def test_journey_run_cfar_preset(actor, base_url, corpus_capable):
    """The presenter loads the classical-CFAR preset and runs it: the Results tab
    shows the radar cube and the CFAR objectness map -- and NOT the Scenario plan
    view, which describes the editor's scene, not the replayed corpus (2026-09-22
    rehearsal: a lone radar triangle led every preset's results)."""
    if not corpus_capable:
        pytest.skip("Thrust 5 presets need torch + the benchmark corpus")
    from webapp.demo_presets import PRESETS_BY_ID
    preset = PRESETS_BY_ID["thrust5_detector_cfar"]
    actor.attempts_to(
        ui.open_app(base_url),
        ui.select_preset(preset.label),
        ui.load_preset(),
        ui.run_pipeline(),
        Ensure.that(ui.results_text(), contains("CFAR objectness"),
                    "the CFAR objectness map is on screen"),
        Ensure.that(ui.results_text(), contains("Range-Doppler"),
                    "the radar cube is on screen"),
    )
    titles = actor.asks(ui.results_titles())
    assert not any("plan view" in t for t in titles), titles


# ------------------------------------------------------- journey 10: Cancel
def test_journey_cancel_a_long_run(actor, base_url, run_capable):
    """The presenter starts a 20-frame run, presses Cancel a moment later, and gets
    the frames that ran as partial results with a status line saying so; the app
    then runs normally again."""
    if not run_capable:
        pytest.skip("a pipeline run needs torch + munich.pkl frames")
    from webapp.demo_presets import PRESETS_BY_ID
    preset = PRESETS_BY_ID["thrust3_cold_start_acquisition"]
    actor.attempts_to(
        ui.open_app(base_url),
        ui.select_preset(preset.label),
        ui.load_preset(),
        ui.set_frames_to_run(20),
        ui.start_run_then_cancel(),
        Ensure.that(ui.results_graph_count(), at_least(1),
                    "the frames that ran still produce figures"),
        ui.return_to_block_diagram(),
        Ensure.that(ui.run_status_text(), contains("Cancelled after"),
                    "the status line reports the cancel and the frame count"),
        ui.set_frames_to_run(2),
        ui.run_pipeline(),
        Ensure.that(ui.results_graph_count(), at_least(1),
                    "a run after a cancel completes normally"),
    )
