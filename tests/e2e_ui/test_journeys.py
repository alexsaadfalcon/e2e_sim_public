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
