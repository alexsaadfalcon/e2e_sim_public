"""Tests for the generated demo runbook (webapp/runbook.py).

The doc used to be hand-copied from webapp/demo_presets.py and went stale on every
review wave. These tests pin: (a) generation covers every preset in stage order,
(b) `--check` passes on the committed docs/DEMO_RUNBOOK.md (i.e. it really is
regenerated output, not hand-edited since), (c) the module never imports torch.

Fixed 2026-09-24 (cross-shard bug): the doc used to embed a per-preset wall time
read from a rehearsal summary.json, which drifts every time anyone reruns
`webapp.rehearse` -- so (b) failed on every rehearsal, independent of any preset
change. `render()` no longer takes a `summary` argument and the CLI no longer has
`--summary`; see test_render_does_not_embed_wall_times.
"""

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUNBOOK = _REPO_ROOT / "docs" / "DEMO_RUNBOOK.md"


def _flat(doc: str) -> str:
    """Collapse whitespace (including the 79-col wrap's line breaks) so a
    substring check isn't sensitive to where `webapp.runbook._wrap` happened to
    break a line -- the wrap point is cosmetic, not semantic content."""
    return " ".join(doc.split())


def _import_without_torch(module_name):
    """Import `module_name` in a fresh subprocess; exit 0 if torch was never
    imported, 3 if it was -- mirrors tests/test_webapp.py::_import_without_torch.
    Done out-of-process so a torch import here can't corrupt the shared process."""
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


def test_runbook_module_imports_without_torch():
    proc = _import_without_torch("webapp.runbook")
    assert proc.returncode == 0, (
        "importing webapp.runbook must succeed without importing torch "
        f"(rc={proc.returncode}); stderr:\n{proc.stderr}"
    )


def test_render_mentions_every_preset_in_stage_order():
    from webapp.demo_presets import PRESETS
    from webapp.runbook import render

    doc = render(PRESETS)

    positions = [doc.index(p.label) for p in PRESETS]
    assert positions == sorted(positions), "presets must appear in PRESETS order"
    # Every preset's numbered heading is present.
    for i, p in enumerate(PRESETS, 1):
        assert f"## {i}. {p.label}" in doc


def test_render_does_not_embed_wall_times():
    """Fixed 2026-09-24 (cross-shard bug): a per-preset wall_s read from a
    rehearsal summary.json drifted on every rehearsal, independent of any preset
    change, so `--check` (and test_check_passes_on_committed_runbook) failed
    whenever anyone rehearsed after the doc was last generated -- a drifting
    value baked into a durable document. The doc must instead print one fixed
    sentence, the same for every preset, naming where the CURRENT number lives
    (the rehearsal summary and the preflight timing pass).

    Wave 10 (2026-09-24, item 3.2, hostile round 9): RETRACTED the "presets are
    sized to stay under the 15 s WARN budget" clause this test used to pin --
    false on 5 of 7 presets (rehearsal summary: 14-27 s with both arms).
    Replaced with an honest range ("roughly 15-30 s for both arms"); this test no
    longer imports webapp.preflight.WARN_SECONDS since nothing in the rendered
    doc derives from it any more."""
    from webapp.demo_presets import PRESETS
    from webapp.runbook import render

    doc = render(PRESETS)
    flat = _flat(doc)
    # No per-run number anywhere: neither a fabricated wall time nor the old
    # "measure" fallback (both were per-preset; the new sentence is not).
    assert "measure (no entry for this preset" not in flat
    assert "from the rehearsal summary)" not in flat
    # The one fixed sentence, once per preset, naming both places the current
    # number lives. (`doc.count`, not `flat.count`: the sentence starts a new
    # numbered list item in the wrapped source, so counting on the flattened
    # text would also match the two module-docstring mentions of "rehearsal".)
    assert doc.count("Wall time: read the last rehearsal's") == len(PRESETS)
    assert "summary.json" in flat and "`wall_s`" in flat
    assert "preflight timing pass" in flat
    # The retracted claim must not reappear, on this run or any future one.
    assert "warn budget" not in flat
    assert "roughly 15-30 s for both arms" in flat


def test_ab_lines_matches_the_rendered_arm_chip():
    """`_ab_lines` must quote exactly the chip `webapp.app._ab_arm_chip` renders
    on screen, including its rung-ladder shortening -- a hand-rebuilt
    `f"A — {label} {value}"` here silently drifted from the real chip once that
    ladder shipped (wave 13 beautification pass, item 2)."""
    from webapp.app import _ab_arm_chip
    from webapp.demo_presets import PRESETS
    from webapp.runbook import _ab_lines

    checked_ab = False
    for p in PRESETS:
        if p.ab is None:
            assert _ab_lines(p) is None, p.id
            continue
        checked_ab = True
        chip_a, chip_b = _ab_lines(p)
        assert chip_a == _ab_arm_chip(p, "a"), p.id
        assert chip_b == _ab_arm_chip(p, "b"), p.id
    assert checked_ab, "no A/B preset found -- this test would pass vacuously"


def test_render_has_no_previous_run_divider_label():
    """Wave 10 (2026-09-24, item 3.1, hostile round 9): webapp/app.py only prints
    the "Previous run (for before/after): " label when `_ab` is false; every
    preset here sets `ab`, so the divider that actually renders is a bare
    horizontal rule. The runbook must not tell the presenter to look for a label
    the app never draws for these seven presets."""
    from webapp.demo_presets import PRESETS
    from webapp.runbook import render

    doc = render(PRESETS)
    assert '"Previous run" divider' not in doc
    assert "no label" in _flat(doc)


def test_render_gives_a_prepared_line_for_every_preset_while_it_runs():
    """Wave 10 (2026-09-24, item 3.2, hostile round 9): each Run takes 14-27 s
    (rehearsal summary, both arms); the three Thrust 5 screens back to back are
    ~77 s of dead air. Every preset's click sequence must carry a "While it runs,
    say:" line, and it must be that preset's own FIRST `say` bullet (never
    hand-typed, so it cannot drift from the card)."""
    from webapp.demo_presets import PRESETS
    from webapp.runbook import render

    doc = render(PRESETS)
    flat = _flat(doc)
    assert doc.count("**While it runs, say:**") == len(PRESETS)
    for p in PRESETS:
        assert p.say, p.id  # every preset has a say list (test_demo_presets.py)
        # `_flat`, not raw `doc`: `_bullet` wraps at 79 cols, so a long say
        # bullet's exact text may straddle a line break in the rendered source.
        assert _flat(p.say[0]) in flat, p.id


def test_render_gives_the_rdp_slider_rule_a_thrust_5_exception():
    """Wave 10 (2026-09-24, item 3.3, hostile round 9): the RDP slider rule is
    actively harmful on Thrust 5 -- the objectness/scoreboard/PR panels have no
    slider (pinned to the last frame); dragging the Range-Doppler panel's slider
    desyncs it from the frozen detections."""
    from webapp.demo_presets import PRESETS
    from webapp.runbook import render

    doc = render(PRESETS)
    flat = _flat(doc)
    assert "thrust 5 exception" in flat.lower()
    assert "desyncs the cube" in flat.lower()


# wave 12 (2026-09-24), hostile round 10 items 3.1-3.4.
def test_render_gives_the_rdp_slider_rule_a_thrust_4_exception_too():
    """Item 3.2: the loop-vs-static desync isn't Thrust-5-only -- Thrust 4's
    range-profile panel is built from the LAST frame and never animates while
    the range-azimuth map above it keeps looping (pipeline_runner's
    `range_profile_agg[-1]`); the runbook used to disclose only the Thrust 5
    case."""
    from webapp.demo_presets import PRESETS
    from webapp.runbook import render

    doc = render(PRESETS)
    flat = _flat(doc)
    assert "thrust 4 exception" in flat.lower()
    assert "range-profile panel renders once from the last frame" in flat.lower()


def test_render_says_the_one_shared_slider_parks_both_arms():
    """Item 3.3, RETRACTED by the beautification pass (2026-09-24): the old
    per-panel sliders `results_clock.js` used to pause on drag could park ONE
    arm and leave the other running (item 3.3's original finding). Those sliders
    are gone -- there is now exactly one shared transport (pause/play + "frame N
    of M" + one slider) for the whole screen, so scrubbing it pauses and parks
    BOTH arms together. The runbook must state the current behaviour, not the
    retired one-arm-parking warning."""
    from webapp.demo_presets import PRESETS
    from webapp.runbook import render

    doc = render(PRESETS)
    flat = _flat(doc)
    assert "parks only the one arm" not in flat.lower()
    assert "parks both arms" in flat.lower()
    assert "one slider for the whole screen" in flat.lower()


def test_render_has_a_general_pause_before_reading_a_number_rule():
    """Item 3.4: three demo cards say "read it off the screen" for a printed
    statistic that is rebuilt every frame while the clock loops -- the runbook's
    general mechanics must carry one rule that applies to every preset, not
    only the Thrust 5 exception paragraph.

    Wave 13 (2026-09-24, beautification pass): the pinned phrase now names WHERE
    to pause -- the one transport's button lives in the run-identity row, not on
    a per-panel slider any more."""
    from webapp.demo_presets import PRESETS
    from webapp.runbook import render

    doc = render(PRESETS)
    flat = _flat(doc)
    assert ("pause with the button in the run-identity row before reading a "
           "per-frame number" in flat.lower())
    assert "not just thrust 5" in flat.lower()


def test_render_states_shared_colour_limits_only_when_a_heatmap_is_on_screen():
    """Item 3.1: "on shared colour limits" used to be claimed for every A/B
    preset, including Thrust 3, whose only product (subspace_err) is a line
    plot -- there is no colour scale on that screen to share."""
    from webapp.demo_presets import PRESETS, PRESETS_BY_ID
    from webapp.runbook import render

    doc = render(PRESETS)
    t3 = PRESETS_BY_ID["thrust3_cold_start_acquisition"]
    i3 = PRESETS.index(t3) + 1
    start = doc.index(f"## {i3}. {t3.label}")
    section = doc[start:doc.index("### Second knob", start)]
    assert "shared colour limits" not in _flat(section).lower()
    assert "no shared colour scale" in _flat(section).lower()
    # A heat-map preset (Thrust 1) still gets the claim.
    t1 = PRESETS_BY_ID["thrust1_circuit_knobs"]
    i1 = PRESETS.index(t1) + 1
    start1 = doc.index(f"## {i1}. {t1.label}")
    section1 = doc[start1:doc.index("### Second knob", start1)]
    assert "shared colour limits" in _flat(section1).lower()


def test_render_lists_all_four_thrust5_panels():
    """Wave 10 (2026-09-24, item 3.4, hostile round 9): two product blocks
    (radar_cube, detector) render FOUR panels on every Thrust 5 screen -- the
    scoreboard and PR-curve panels are not separately toggleable blocks, so the
    generic per-product enumeration undercounted them by half."""
    from webapp.demo_presets import PRESETS
    from webapp.runbook import render

    doc = render(PRESETS)
    for p in PRESETS:
        if p.thrust != 5:
            continue
        section = doc[doc.index(f'## {PRESETS.index(p) + 1}. {p.label}'):]
        section = section[:section.index("### Second knob")]
        assert "Range-Doppler power" in section, p.id
        assert "Detector scoreboard" in section, p.id
        assert "scored offline" in section, p.id
        assert ("CFAR objectness" in section
               or "Neural detector objectness" in section), p.id


def test_render_quotes_rendered_panel_titles_not_block_diagram_labels():
    """Wave 10 (2026-09-24, item 3.5, hostile round 9): the "What you are looking
    at" section used to quote BLOCKS_BY_ID node labels ("Radar Cube
    (Range-Doppler)", "Detector (CFAR | ML)") -- strings that never appear on the
    rendered Results tab. It must quote the actual panel titles there instead.
    (The block label is still correct, and kept, in "Second knob (optional)" --
    that section names the BLOCK the knob lives on, not a panel.)"""
    from webapp.demo_presets import PRESETS
    from webapp.runbook import render

    doc = render(PRESETS)
    for i, p in enumerate(PRESETS, 1):
        start = doc.index(f"## {i}. {p.label}")
        section = doc[start:doc.index("### Second knob", start)]
        assert "Radar Cube (Range-Doppler)" not in section, p.id
        assert "Detector (CFAR | ML)" not in section, p.id
    assert '"Range-azimuth power"' in doc
    assert '"Range-Doppler power"' in doc


def test_panel_titles_match_the_rendered_source():
    """Wave 10 (2026-09-24, item 3.5, hostile round 9): the panel-title constants
    in webapp/runbook.py are hand-typed (the true strings live inside
    torch-tolerant modules this torch-free generator does not import) -- so this
    test, which MAY import torch, greps those modules' own source text for the
    literal substrings, and catches a rename there that generation time cannot
    see.

    Wave 13 (2026-09-24, beautification pass): RETRACTED the "not anchored to a
    closing quote" extractor -- these titles used to be the first part of a
    longer f-string title, so a bare substring check was the most it could pin.
    They are now `set_panel(fig, title="...")` literals (five directly, or via a
    `title` loop variable fed by a literal tuple two lines above its own
    `set_panel(` call -- see `pipeline_runner.py`'s range_az/range_el loop), so
    the extractor reads the actual set_panel( titles: it anchors to the quoted
    literal (`"<title>"`), not a loose substring that could also match inside a
    comment or an unrelated string in this heavily-commented file."""
    pytest.importorskip("torch")
    from webapp.runbook import _DETECTOR_PANEL_TITLES, _PANEL_TITLES

    pipeline_runner_src = (_REPO_ROOT / "webapp" / "pipeline_runner.py").read_text(encoding="utf-8")
    scoreboard_src = (_REPO_ROOT / "webapp" / "detector_scoreboard.py").read_text(encoding="utf-8")
    for bid, title in _PANEL_TITLES.items():
        assert f'"{title}"' in pipeline_runner_src, (bid, title)
    for mode, title in _DETECTOR_PANEL_TITLES.items():
        assert f'"{title}"' in pipeline_runner_src, (mode, title)
    assert '"Detector scoreboard' in scoreboard_src
    assert '"scored offline: ' in scoreboard_src


def test_render_tells_thrust_1_and_4_to_scroll_the_param_pane():
    """Wave 10 (2026-09-24, item 3.6, hostile round 9): the Thrust 1 and Thrust 4
    param editors clip mid-sentence at the bottom of the pane; the knob the click
    sequence just named is below the fold on both."""
    from webapp.demo_presets import PRESETS
    from webapp.runbook import render

    doc = render(PRESETS)
    for p in PRESETS:
        section = doc[doc.index(f'## {PRESETS.index(p) + 1}. {p.label}'):]
        section = section[:section.index("### What you are looking at")]
        if p.thrust in (1, 4):
            assert "Scroll the param pane" in _flat(section), p.id
        else:
            assert "Scroll the param pane" not in _flat(section), p.id


def test_render_names_the_block_to_click_when_second_knob_differs_from_opening_block():
    """Cold-read item 1 (2026-09-24): a "Second knob (optional)" step that names a
    knob on a block OTHER than the one Load preset auto-opened (e.g. Thrust 5's
    "Detector (CFAR | ML)" -> "Decode threshold" while the open panel is "ADC
    Quantizer" or "IF High-Pass") must tell the operator to click that block
    first -- the value step is otherwise unreachable. A second knob on the SAME
    block as the opening one (e.g. Thrust 1's two `rffe` knobs) must not."""
    from webapp.demo_presets import PRESETS
    from webapp.runbook import (
        _block_label, _opening_block, _second_knobs, render,
    )

    doc = render(PRESETS)
    for i, p in enumerate(PRESETS, 1):
        start = doc.index(f"## {i}. {p.label}")
        knob_start = doc.index("### Second knob", start)
        section = doc[knob_start:doc.index("\n\n", knob_start)]
        open_block, _ = _opening_block(p)
        for bid, key, _how in _second_knobs(p):
            knob_block = _block_label(bid)
            instruction = f"Click the **{knob_block}** node in the block diagram to"
            if knob_block != open_block:
                assert instruction in section, (p.id, knob_block, open_block)
            else:
                assert instruction not in section, (p.id, knob_block, open_block)


def test_render_adds_a_step_back_to_the_block_diagram_tab():
    """Cold-read item 2 (2026-09-24): the app auto-switches to Results after a
    run, but the **Demo preset:** dropdown lives on the **Block Diagram** tab --
    every preset's click sequence must tell the operator to switch back before
    loading the next one."""
    from webapp.demo_presets import PRESETS
    from webapp.runbook import render

    doc = render(PRESETS)
    expected = ("click the **Block Diagram** tab to return to the preset picker")
    assert _flat(doc).count(expected) == len(PRESETS)


def test_render_quotes_the_click_affordance_text_once():
    """Cold-read item 1 (2026-09-24): the general mechanics section should quote
    `webapp.block_diagram`'s own click-affordance text once, verbatim, rather
    than the runbook inventing its own paraphrase of what a block click does."""
    pytest.importorskip("dash")
    pytest.importorskip("dash_cytoscape")
    from dash import html

    from webapp import block_diagram
    from webapp.demo_presets import PRESETS
    from webapp.runbook import render

    def _find_p(node):
        if isinstance(node, html.P):
            return node
        children = getattr(node, "children", None)
        if children is None:
            return None
        if not isinstance(children, list):
            children = [children]
        for c in children:
            found = _find_p(c)
            if found is not None:
                return found
        return None

    p = _find_p(block_diagram.layout())
    assert p is not None, "no html.P found in block_diagram.layout()"
    affordance_text = p.children

    doc = render(PRESETS)
    # `_flat`, not raw `doc`: `_wrap` breaks this long sentence across lines in
    # the rendered source.
    assert _flat(doc).count(affordance_text) == 1


def test_render_puts_while_it_runs_between_run_click_and_results_switch():
    """Cold-read item 4 (2026-09-24): "While it runs, say:" must sit between the
    numbered "Click Run pipeline" step and the numbered "app switches to
    Results" step, not after both."""
    from webapp.demo_presets import PRESETS
    from webapp.runbook import render

    doc = render(PRESETS)
    for i, p in enumerate(PRESETS, 1):
        if not p.say:
            continue
        start = doc.index(f"## {i}. {p.label}")
        section = doc[start:doc.index("### What you are looking at", start)]
        run_idx = section.index("2. Click **Run pipeline**")
        say_idx = section.index("**While it runs, say:**")
        results_idx = section.index("3. The app switches to the **Results** tab")
        assert run_idx < say_idx < results_idx, p.id


def test_trouble_section_covers_the_generic_error_message():
    """Cold-read item 3 (2026-09-24): a `PipelineError` or bare `Exception` from
    the Run callback (webapp/app.py, the two `except` clauses ~line 643) paints
    the run-status text red instead of switching to Results. The runbook must
    say what that looks like and the recovery path: `preflight --quick` in a
    second terminal, reload the preset, run again, and the fallback-deck escape
    hatch if it persists."""
    from webapp.demo_presets import PRESETS
    from webapp.runbook import render

    doc = render(PRESETS)
    flat = _flat(doc)
    assert "Unexpected error: " in flat
    assert "python -m webapp.preflight --quick" in flat
    assert "fallback_deck.pdf" in flat
    assert "python -m webapp.fallback_deck" in flat
    # It must live in the trouble section, not be a stray mention elsewhere.
    trouble = doc[doc.index("## If something goes wrong"):]
    assert "Unexpected error: " in trouble
    assert "preflight --quick" in trouble


def test_render_drops_element_id_parentheticals():
    from webapp.demo_presets import PRESETS
    from webapp.runbook import render

    doc = render(PRESETS)
    for leaked_id in ("(preset-select)", "(preset-load)", "(run-nsteps)",
                      "(run-button)", "(cancel-button)", "(run-status)"):
        assert leaked_id not in doc


def test_render_has_no_commit_sha():
    """A SHA embedded in a file committed AT that SHA can never match the commit
    that contains it -- `--check` must not depend on git state, only on the
    presets, so the same render() output is expected regardless of which
    commit produced it (or when it was regenerated -- wall time is no longer
    an input, see test_render_does_not_embed_wall_times)."""
    from webapp.demo_presets import PRESETS
    from webapp.runbook import render

    doc = render(PRESETS)
    assert "HEAD" not in doc
    assert "Regenerate after any change" in _flat(doc)


def test_render_literalises_play_control():
    from webapp.demo_presets import PRESETS
    from webapp.runbook import render

    doc = render(PRESETS)
    assert "▶ (Play) control" in doc


def test_render_drops_sources_note():
    from webapp.demo_presets import PRESETS
    from webapp.runbook import render

    doc = render(PRESETS)
    assert "Sources note" not in doc


def test_check_flag_detects_staleness(tmp_path):
    from webapp.runbook import main

    out = tmp_path / "runbook.md"
    assert main(["--out", str(out)]) == 0
    assert out.is_file()
    # Freshly generated -> --check passes.
    assert main(["--out", str(out), "--check"]) == 0
    # Hand-edit it -> --check must fail.
    out.write_text(out.read_text() + "\nstray hand edit\n")
    assert main(["--out", str(out), "--check"]) == 1


def test_no_summary_cli_argument():
    """The --summary flag was removed with the wall-time embedding it fed --
    nothing else read it (fixed 2026-09-24, cross-shard bug)."""
    from webapp.runbook import main

    with pytest.raises(SystemExit):
        main(["--summary", "whatever.json"])


@pytest.mark.skipif(not _RUNBOOK.is_file(), reason="docs/DEMO_RUNBOOK.md not present")
def test_check_passes_on_committed_runbook():
    """The committed doc must be exactly what `python -m webapp.runbook` produces
    right now -- regenerate it (see the task instructions) before trusting this."""
    proc = subprocess.run(
        [sys.executable, "-m", "webapp.runbook", "--check"],
        cwd=str(_REPO_ROOT), capture_output=True, text=True,
    )
    assert proc.returncode == 0, (
        f"docs/DEMO_RUNBOOK.md is stale; regenerate with `python -m webapp.runbook`.\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
