"""Generate ``docs/DEMO_RUNBOOK.md`` from the demo presets -- never hand-copy them.

    python -m webapp.runbook [--out docs/DEMO_RUNBOOK.md] [--check]

``docs/DEMO_RUNBOOK.md`` used to be hand-written from ``webapp/demo_presets.py`` --
copying each preset's blurb / live_knobs / say / do_not_say verbatim. That is two
authorities for one text: the moment a card changes in ``demo_presets.py`` (which
happens after every review wave) the hand-written copy goes stale silently. This
module is the one authority; the doc is generated output.

Per preset the doc renders: a numbered click sequence (the click MECHANICS are a
template; the preset's own label/n_steps/ab labels/first live-knob are substituted
in), "What you are looking at" (the enabled product panels + the A/B arm labels,
both read from the registry/dataclass, never typed), "Second knob (optional)" (the
``live_knobs`` entries that are not themselves the built-in A/B knob, with the
registry's own label/min/max/default), then the preset's ``blurb``, ``say`` and
``do_not_say`` exactly as the dataclass states them today.

Wall time is NOT embedded (fixed 2026-09-24, cross-shard bug): a rehearsal
``summary.json``'s ``wall_s`` changes every time anyone reruns ``webapp.rehearse``,
so a committed doc that quoted it went stale on every rehearsal, independent of any
preset change -- a drifting value baked into a durable document. Every preset's
click sequence instead prints one fixed sentence pointing at the two places the
CURRENT number lives (the rehearsal summary and the preflight timing pass).

Wave 10 (2026-09-24, item 3.2, hostile round 9): the previous fixed sentence
claimed "presets are sized to stay under the 15 s WARN budget" -- false on five of
seven presets (measured 14-27 s with both arms in the same rehearsal summary this
generator points at). Replaced with an honest range ("roughly 15-30 s for both
arms; talk over it") and a "While it runs, say:" line per preset, rendered from
that preset's own FIRST ``say`` bullet -- so a ~15-77 s dead-air gap (the three
Thrust 5 screens back to back) always has a prepared line, not a fixed-budget claim
the data itself contradicts.

The "Before the audience" / "If something goes wrong" sections describe UI
mechanics (button labels, status strings), not preset data; they are template text
in this module, checked against ``webapp/app.py`` / ``webapp/block_diagram.py`` by
hand, with drift-prone counts (frame ceilings, how many presets set `ab`, the
frame-count range presets ship at) computed at generation time. Panel TITLES quoted
in "What you are looking at" (wave 10, item 3.5) are hand-typed constants, not
imported: the true strings are f-string title expressions with computed numbers
inside ``webapp/pipeline_runner.py`` / ``webapp/detector_scoreboard.py`` (torch-
tolerant modules this one deliberately does not import), so
``tests/test_runbook.py``'s ``test_panel_titles_match_the_rendered_source`` (which
IS allowed to import torch) greps their source for these exact literal substrings,
catching a rename here that this module cannot see at generation time.

Deliberately torch-free: only ``webapp.demo_presets`` and
``webapp.pipeline_registry`` and the standard library are imported, so the doc can
be regenerated (and ``--check``ed) on any machine, CI included, without installing
torch/Sionna.
"""
from __future__ import annotations

import argparse
import difflib
import sys
import textwrap
from pathlib import Path
from typing import List, Optional, Tuple

from webapp.demo_presets import PRESETS, DemoPreset, apply_preset
from webapp.pipeline_registry import (
    BLOCKS_BY_ID,
    MAX_N_STEPS,
    MAX_PRESET_N_STEPS,
    PRODUCT_IDS,
    ParamSpec,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUT = _REPO_ROOT / "docs" / "DEMO_RUNBOOK.md"
_CLI = "python -m webapp.runbook"
_WIDTH = 79

#: Fixed 2026-09-24 (cross-shard bug): this used to be a per-preset number read from
#: a rehearsal summary.json, which changes every time anyone reruns
#: `webapp.rehearse` -- so the committed doc went stale on every rehearsal, with no
#: change to any preset. One fixed sentence, same for every preset, pointing at
#: where the CURRENT number lives instead of embedding one.
#: wave 10 (2026-09-24, item 3.2, hostile round 9): RETRACTED the "15 s WARN
#: budget" claim -- false on 5 of 7 presets (rehearsal summary: 14-27 s with both
#: arms). Replaced with an honest range; the "while it runs, say:" line (rendered
#: per preset from that preset's own first `say` bullet, see `_render_preset`)
#: gives the presenter something prepared for the dead air instead.
_WALL_TIME_NOTE = (
    "Wall time: read the last rehearsal's "
    "`e2e/main/figures/rehearsal/summary.json` (`wall_s`) or the preflight timing "
    "pass; a Run takes roughly 15-30 s for both arms -- talk over it."
)


def _wrap(text: str, indent: str = "", subsequent_indent: Optional[str] = None) -> str:
    """Podium-readable wrapping at `_WIDTH` cols -- long single lines are hard to
    scan in a raw-markdown read even though they render fine, and the doc this
    replaces was hand-wrapped the same way."""
    subsequent_indent = indent if subsequent_indent is None else subsequent_indent
    return textwrap.fill(text, width=_WIDTH, initial_indent=indent,
                         subsequent_indent=subsequent_indent,
                         break_long_words=False, break_on_hyphens=False)


def _bullet(text: str) -> str:
    return _wrap(text, indent="- ", subsequent_indent="  ")


def _numbered(n: int, text: str) -> str:
    prefix = f"{n}. "
    return _wrap(text, indent=prefix, subsequent_indent=" " * len(prefix))


# =====================================================================================
# Small lookups over the registry / a preset -- kept here rather than duplicated per
# call site, so a rename in the registry (a param's label, say) only has to be found
# once by whoever renders the doc.
# =====================================================================================

def _block_label(bid: str) -> str:
    spec = BLOCKS_BY_ID.get(bid)
    return spec.label if spec is not None else bid


def _param_spec(bid: str, key: str) -> Optional[ParamSpec]:
    spec = BLOCKS_BY_ID.get(bid)
    if spec is None:
        return None
    return next((ps for ps in spec.params if ps.key == key), None)


def _param_label(bid: str, key: str) -> str:
    ps = _param_spec(bid, key)
    return ps.label if ps is not None else key


def _param_bounds(bid: str, key: str) -> str:
    """"choices [...], default X" or "min a, max b, default X" (only the parts that
    are actually declared) for a registry ParamSpec -- "" if the param has no
    registry entry at all (an internal, non-UI knob such as `subspace.gap_response`,
    which is validated in `demo_presets._INTERNAL_PARAMS`, not the registry)."""
    ps = _param_spec(bid, key)
    if ps is None:
        return ""
    if ps.kind == "choice":
        return f"choices {ps.choices}, default {ps.default!r}"
    parts = []
    if ps.min is not None:
        parts.append(f"min {ps.min:g}" if isinstance(ps.min, (int, float)) else f"min {ps.min}")
    if ps.max is not None:
        parts.append(f"max {ps.max:g}" if isinstance(ps.max, (int, float)) else f"max {ps.max}")
    parts.append(f"default {ps.default!r}")
    return ", ".join(parts)


def _opening_block(preset: DemoPreset) -> Tuple[str, str]:
    """(block label, param label) the param editor opens on after Load preset --
    the block named by the preset's FIRST `live_knobs` entry (see
    `webapp.app._load_preset`); a preset with no `live_knobs` falls back to
    whichever block the operator last tapped, which generation time cannot know."""
    if not preset.live_knobs:
        return ("whichever block was last selected (this preset sets no `live_knobs`)", "")
    bid, key, _how = preset.live_knobs[0]
    return (_block_label(bid), _param_label(bid, key))


def _ab_lines(preset: DemoPreset) -> Optional[Tuple[str, str]]:
    """(A's banner line, B's banner line), matching `webapp.app._ab_arm_line` --
    None if this preset has no built-in A/B."""
    if preset.ab is None:
        return None
    bid, key, _value_b = preset.ab
    label = _param_label(bid, key)
    return (f"A (as loaded): {label} {preset.ab_label_a or '?'} -- before",
            f"B: {label} {preset.ab_label_b or '?'} -- after")


def _second_knobs(preset: DemoPreset) -> List[Tuple[str, str, str]]:
    """`live_knobs` entries that are NOT the knob the built-in A/B already turns --
    the manual, additional path each preset's card offers on top of one-click A/B."""
    ab_key = (preset.ab[0], preset.ab[1]) if preset.ab is not None else None
    return [(b, k, how) for (b, k, how) in preset.live_knobs if (b, k) != ab_key]


# =====================================================================================
# wave 10 (2026-09-24, item 3.5, hostile round 9): the RENDERED panel titles, not
# the block-diagram node labels (`BLOCKS_BY_ID[bid].label`, e.g. "Radar Cube
# (Range-Doppler)", "Detector (CFAR | ML)") the previous version of this module
# quoted. Hand-typed, not imported: the true strings are f-string title
# expressions with computed numbers inline (clip dB, corpus name, frame counts)
# inside webapp/pipeline_runner.py and webapp/detector_scoreboard.py, which this
# module deliberately does not import (both are torch-tolerant, not torch-free at
# call time). tests/test_runbook.py::test_panel_titles_match_the_rendered_source
# (which IS allowed to import torch) greps those two files' source for these exact
# literal substrings, so a rename there is caught here rather than silently
# quoting a string nobody sees on screen.
# =====================================================================================
_PANEL_TITLES = {
    "range_az": "Range-azimuth power",
    "range_el": "Range-elevation power",
    "range_profile": "Range profile (non-coherent over channels)",
    "radar_cube": "Range-Doppler power",
    "subspace_err": "Subspace error (Frobenius) per frame",
}
#: The two rendered titles for the "detector" product block -- which one shows
#: depends on the preset's own `detector.mode` override, not the block itself.
_DETECTOR_PANEL_TITLES = {"cfar": "CFAR objectness", "ml": "Neural detector objectness"}


def _panel_title(bid: str, preset: DemoPreset) -> str:
    """The rendered panel title for product block `bid` under `preset` -- the
    constant from `_PANEL_TITLES`/`_DETECTOR_PANEL_TITLES`, falling back to the
    registry's own block label only for a product this runbook has not been
    taught the rendered title of yet (so a future product block degrades to the
    old behaviour instead of crashing generation)."""
    if bid == "detector":
        mode = apply_preset(preset)["detector"]["params"].get("mode", "cfar")
        return _DETECTOR_PANEL_TITLES.get(mode, _block_label(bid))
    return _PANEL_TITLES.get(bid, _block_label(bid))


# =====================================================================================
# Per-preset section
# =====================================================================================

#: wave 10 (2026-09-24, item 3.6, hostile round 9): the Thrust 1 and Thrust 4
#: param editors clip mid-sentence at the bottom of the pane (rehearsal PNG,
#: round 9 item 2.5) -- the knob the click sequence just named (Tessera: TSV
#: height (um) on T4; LNA bias current (mA) on T1, though that one opens
#: visible -- the SECOND control below it does not) is below the fold on both.
#: Keyed by thrust number, not preset id, since the fold is a param-pane-height
#: fact, not a per-preset one.
_SCROLL_PARAM_PANE_THRUSTS = {1, 4}


def _render_preset(i: int, preset: DemoPreset) -> str:
    lines: List[str] = []
    lines.append(f"## {i}. {preset.label}\n")

    open_block, open_param = _opening_block(preset)
    open_desc = (f"**{open_block}**" + (f" (first knob: **{open_param}**)" if open_param else ""))
    ab = _ab_lines(preset)
    arm_sentence = (f" Both arms run in one click (A = {preset.ab_label_a}, "
                    f"B = {preset.ab_label_b})." if preset.ab is not None else "")
    scroll_note = (" Scroll the param pane; the knob is below the fold."
                   if preset.thrust in _SCROLL_PARAM_PANE_THRUSTS else "")

    lines.append("### Click sequence")
    lines.append(_numbered(1,
        f'Open **Demo preset:**, select "{preset.label}", click **Load preset**. '
        f'The param editor opens on {open_desc}; the operator card shows "Loaded: '
        f'{preset.label} (Thrust {preset.thrust}, {preset.n_steps} frames)".'
        f'{scroll_note}'))
    lines.append(_numbered(2,
        f"Click **Run pipeline**.{arm_sentence} {_WALL_TIME_NOTE}"))
    if preset.say:
        # wave 10 (2026-09-24, item 3.2, hostile round 9): a prepared line for the
        # 15-30 s dead air, always the preset's own first `say` bullet -- never
        # hand-typed, so it cannot drift from the card. Sits between the Run
        # click and the tab switch (item 4, round-9 cold read) -- it names what
        # to say WHILE the run is in flight, not after it lands.
        lines.append(_bullet(f"**While it runs, say:** {preset.say[0]}"))
    lines.append(_numbered(3, "The app switches to the **Results** tab automatically."))
    lines.append(_numbered(4,
        "Before loading the next preset: click the **Block Diagram** tab to "
        "return to the preset picker (the app auto-switched to **Results** in "
        "the step above; the **Demo preset:** dropdown lives on **Block "
        "Diagram**)."))
    lines.append("")

    lines.append("### What you are looking at")
    if preset.thrust == 5:
        # wave 10 (2026-09-24, item 3.4, hostile round 9): two product blocks
        # (radar_cube, detector) render FOUR panels on every Thrust 5 screen --
        # the scoreboard and PR-curve panels are not separately toggleable
        # blocks, so the generic per-product enumeration below would undercount
        # them by half.
        det_title = _panel_title("detector", preset)
        lines.append(_bullet(
            f'Four panels render: **"{_panel_title("radar_cube", preset)}"** '
            f'(Range-Doppler, has its own frame slider), **"{det_title}"** '
            '(no slider -- pinned to the last frame), **"Detector scoreboard"**, '
            'and the offline PR-curve panel (**"scored offline: ... test '
            'frames"**).'))
        # wave 11 (2026-09-24, owner live test): the Results-tab clock now animates
        # the cube WITHOUT a click, so the desync the wave-10 "leave the slider
        # alone" note warned about is the default state of these three screens.
        lines.append(_bullet(
            'The Range-Doppler panel loops by itself on the Results-tab clock; the '
            'other three hold the LAST frame. Press pause on the cube before '
            "talking about one frame's detections."))
    else:
        state = apply_preset(preset)
        enabled_bids = [bid for bid in PRODUCT_IDS if state[bid]["enabled"]]
        if enabled_bids:
            titles = [_panel_title(bid, preset) for bid in enabled_bids]
            lines.append(_bullet("Product panel(s) this preset enables: "
                                 + ", ".join(f'**"{t}"**' for t in titles) + "."))
        else:
            lines.append(_bullet("This preset enables no product panel (check the block state)."))
    if ab is not None:
        lines.append(_bullet(
            f'Arm banners on screen: "{ab[0]}" (LEFT column) / "{ab[1]}" '
            f'(RIGHT column) -- each product renders once per column, on the same '
            f'row, on shared colour limits.'))
    lines.append("")

    lines.append("### Second knob (optional)")
    extra = _second_knobs(preset)
    if extra:
        for bid, key, how in extra:
            bounds = _param_bounds(bid, key)
            bounds_part = f" ({bounds})" if bounds else ""
            knob_block = _block_label(bid)
            if knob_block != open_block:
                # wave 11 (2026-09-24, cold-read item 1): this knob lives on a
                # DIFFERENT block than the one the param editor auto-opened on
                # (step 1) -- e.g. Thrust 5's "Detector (CFAR | ML)" second
                # knob while step 1 opened "ADC Quantizer"/"IF High-Pass". The
                # operator has to click the other node first or the value step
                # below is unreachable.
                lines.append(_bullet(
                    f"Click the **{knob_block}** node in the block diagram to "
                    "open its panel."))
            lines.append(_bullet(f"**{knob_block}** -> **{_param_label(bid, key)}**"
                                 f"{bounds_part}: {how}"))
    else:
        lines.append(_bullet(
            "None: the only `live_knobs` entry for this preset is the knob the "
            "built-in A/B already turns; nothing further to change manually "
            "before re-running."))
    lines.append("")

    lines.append(_wrap(preset.blurb) + "\n")

    if preset.say:
        lines.append("### Say")
        for s in preset.say:
            lines.append(_bullet(s))
        lines.append("")

    if preset.do_not_say:
        lines.append("### Do NOT say")
        for s in preset.do_not_say:
            lines.append(_bullet(s))
        lines.append("")

    return "\n".join(lines)


# =====================================================================================
# Template sections -- UI mechanics, not preset data. Cross-checked by hand against
# webapp/app.py and webapp/block_diagram.py; the counts embedded below (frame
# ceilings, how many presets set `ab`, the frame range presets ship at) are computed
# from the registry/presets at generation time rather than typed here, since those
# ARE preset/registry data and can drift.
# =====================================================================================

_BEFORE_AUDIENCE = """## Before the audience

1. Preflight: `python -m webapp.preflight` (full run, ~2 min: assets, torch/CUDA,
   port, a warm-up run, every preset, a timing pass). Fix any `FAIL` before
   continuing.
2. Launch: `python -m webapp.app`
3. Open: `http://127.0.0.1:8050`
4. Throwaway warm-up: pick any one preset, click **Load preset**, click
   **Run pipeline**, let it finish. This pays the ~10s torch cold start now
   instead of in front of the room.
5. The Results tab PLAYS ITSELF: every animated panel on it -- both A/B arms,
   every product that has frames -- steps together on one 700 ms clock and loops
   forever, starting by itself when the results render. Nothing to click.
   RETIRED (owner, live test 2026-09-24): the old rule to advance frames with a
   figure's own frame slider and never the ▶ (Play) control. It assumed Play's
   ~350 ms/frame animation stuttered over the RDP link; the owner measured the
   link and it does not.
   To HOLD a frame while you talk about it, press the pause button on any panel --
   one clock, so every panel stops together -- and ▶ to resume. Dragging a
   slider also pauses the clock, so you can park a panel on a chosen frame.
   **Thrust 5 exception**: only the Range-Doppler panel has frames. The
   objectness/scoreboard/PR panels are pinned to the LAST frame by design, so the
   clock now desyncs the cube from those frozen detections by itself, with no
   drag at all. Pause the cube before talking about a specific frame's detections.
"""


def _click_mechanics(n_presets: int, n_ab: int) -> str:
    if n_ab == n_presets:
        # wave 11 (2026-09-24, owner live test): A/B arms render SIDE BY SIDE,
        # one row per product (webapp/app.py `_ab_columns`). Supersedes the wave-10
        # wording ("A on top, B below, under a plain divider line"), which itself
        # RETRACTED an earlier "under a 'Previous run' divider" -- that label only
        # ever printed when `_ab` was false, and every preset here sets `ab`.
        ab_sentence = (f"Every one of the {n_presets} presets below sets `ab`, so one "
                       "click on **Run pipeline** runs BOTH arms A and B and renders "
                       "both on the Results tab SIDE BY SIDE: one row per product, "
                       "arm A in the left column, arm B in the right, each column "
                       "under its own banner (no divider line, no label) -- there "
                       "is no second Run click "
                       "needed for the built-in A/B; the \"Second knob\" step in each "
                       "section below is a *manual, additional* change on top of "
                       "that.")
    else:
        ab_sentence = (f"{n_ab} of the {n_presets} presets below set `ab` (one click "
                       "on **Run pipeline** then runs BOTH arms and renders both on "
                       "the Results tab); the rest are single-run. The \"Second knob\" "
                       "step in each section is a *manual, additional* change on top "
                       "of whatever the preset already does.")
    bullets = [
        "The preset picker is the **Demo preset:** dropdown on the **Block Diagram** "
        "tab; **Load preset** applies it.",
        # wave 11 (2026-09-24, cold-read item 1): the exact affordance text under
        # the diagram (webapp/block_diagram.py) -- quoted once here rather than
        # per-preset, since it applies to every block click in every section
        # below (the auto-opened block on Load AND the "Second knob" clicks).
        'Every block in the diagram is clickable: "Click a block to edit its '
        'parameters or toggle it on/off. Dashed edges feed a disabled block. '
        'Then hit Run pipeline."',
        "Loading a preset REPLACES the whole block state (edits made afterwards are "
        "the operator's own), fills **Frames to run (n_steps)** with the preset's "
        'frame count, prints an operator card under the dropdown ("Loaded: '
        "`<label>` (Thrust `<n>`, `<n_steps>` frames)\" followed by **Turn live** / "
        "**Say** / **Do NOT say or show** lists -- copied verbatim from the "
        "preset), and auto-opens the block-param editor on the block named by the "
        "preset's first `live_knobs` entry.",
        "**Run pipeline** runs it; a run in progress enables **Cancel** (otherwise "
        "disabled) and shows the run-status text. A completed run switches the "
        "browser to the **Results** tab automatically.",
        ab_sentence,
    ]
    body = "\n".join(_bullet(b) for b in bullets)
    header = _wrap("General click mechanics that apply to every preset below "
                   "(from `webapp/app.py`, `webapp/block_diagram.py`):")
    return header + "\n\n" + body + "\n"


def _trouble(n_min: int, n_max: int) -> str:
    bullets = [
        "**Cancel:** click **Cancel** (enabled only while a run is in progress) to "
        'stop after the current frame. The status immediately shows "Cancelling '
        'after the current frame...". If nothing had finished yet: "Cancelled '
        "before the first frame finished: nothing ran, nothing to show. The "
        'Results tab is unchanged." (single-run path) or "Cancelled after `<n>` '
        "of `<n>` frames of run A; run B did not start: showing run A only. See "
        'Results tab." (A/B path, cancelled during arm A). If at least one frame '
        'finished: "Cancelled after `<n>` of `<n>` frames: `<n>` product(s) from '
        'the frames that ran."',
        f"**Frame ceiling:** **Frames to run (n_steps)** accepts 1 to "
        f"{MAX_N_STEPS}; typing outside that range or leaving it blank refuses "
        f'with "Frames to run must be a whole number from 1 to {MAX_N_STEPS}; '
        '`<reason>`. Fix it and press Run again." Every preset here ships at '
        f"{n_min}-{n_max} frames, well under both that ceiling and the separate "
        f"{MAX_PRESET_N_STEPS}-frame cap presets themselves are validated against.",
        "**Reloading a preset:** clicking **Load preset** again fully replaces "
        "the block state and clears the Results tab (so a stale before/after "
        "pair from a different preset cannot linger), then re-opens the param "
        "editor on the block named by that preset's first `live_knobs` entry.",
        "**PDF/PNG fallback (LAST RESORT ONLY):** `e2e/main/figures/rehearsal/`, "
        "generated by `python -m webapp.rehearse`. Preflight passing on every "
        "check is what is supposed to make this unnecessary.",
        # wave 11 (2026-09-24, cold-read item 3): the two exception paths in
        # webapp/app.py's Run callback (~lines 643-651) -- PipelineError (a
        # named, anticipated failure) prints its own message with no fixed
        # prefix; anything else falls to the bare `except Exception` and always
        # carries the "Unexpected error: " prefix.
        "**A red error message after clicking Run pipeline:** the run-status "
        "text (same place as the frame-ceiling message above) turns red "
        'instead of the app switching to Results. A named failure prints its '
        'own reason with no fixed prefix (e.g. "Corpus manifest not found: '
        '...", "Unknown detector mode ..."); anything unanticipated prints '
        '"Unexpected error: `<exception>`". Either way: in a second terminal '
        "run `python -m webapp.preflight --quick`, then reload the preset "
        "(**Load preset** again) and click **Run pipeline** again. If it "
        "still fails, present the fallback deck for this preset instead "
        "(`e2e/main/figures/rehearsal/fallback_deck.pdf`, built by "
        "`python -m webapp.fallback_deck`).",
    ]
    body = "\n".join(_bullet(b) for b in bullets)
    return "## If something goes wrong\n\n" + body + "\n"


# =====================================================================================
# Top-level render + CLI
# =====================================================================================

def render(presets: List[DemoPreset], cmd: str = _CLI) -> str:
    """Render the full runbook markdown for `presets`.

    No commit SHA is embedded: a SHA inside a file committed AT that SHA can
    never equal the commit that contains it, so a `--check` gate tied to one
    would fail at every commit that doesn't touch the presets -- exactly the
    drifting-value-in-a-durable-document failure this generator exists to
    avoid. Regenerate after any change to `webapp/demo_presets.py`; `--check`
    is what enforces that, not a SHA comparison. Wall time is deliberately NOT
    an input here either, for the same reason (see the module docstring and
    `_WALL_TIME_NOTE`) -- a rehearsal run is not a `render()` input."""
    n_steps_vals = [p.n_steps for p in presets] or [0]
    n_ab = sum(1 for p in presets if p.ab is not None)

    parts: List[str] = []
    parts.append("# Demo runbook -- CogniSense Annual Review\n\n" + _wrap(
        f"Generated by `{cmd}` from `webapp/demo_presets.py` (every preset's "
        "blurb / live_knobs / say / do_not_say -- the source of truth) and "
        "`webapp/pipeline_registry.py` (product-panel labels, knob ranges). "
        "Wall time is not embedded (it drifts on every rehearsal); each click "
        "sequence instead points at where the current number lives. Every UI "
        "label quoted below is the exact string in `webapp/app.py` or "
        "`webapp/block_diagram.py`. Regenerate after any change to "
        "`webapp/demo_presets.py`.") + "\n")
    parts.append(_BEFORE_AUDIENCE)
    parts.append("Preset stage order (`PRESETS` in `webapp/demo_presets.py`):\n")
    parts.append("\n".join(f"{i}. {p.label}" for i, p in enumerate(presets, 1)) + "\n")
    parts.append(_click_mechanics(len(presets), n_ab))
    parts.append("---\n")
    for i, p in enumerate(presets, 1):
        parts.append(_render_preset(i, p))
        parts.append("---\n")
    parts.append(_trouble(min(n_steps_vals), max(n_steps_vals)))
    return "\n".join(parts).rstrip() + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(_DEFAULT_OUT),
                    help="path to write the generated runbook")
    ap.add_argument("--check", action="store_true",
                    help="don't write; exit 1 (and print a diff) if --out differs "
                         "from what would be generated")
    args = ap.parse_args(argv)

    out_path = Path(args.out)
    content = render(PRESETS)

    if args.check:
        existing = out_path.read_text(encoding="utf-8") if out_path.is_file() else ""
        if content == existing:
            print(f"{out_path} is up to date.")
            return 0
        diff = difflib.unified_diff(
            existing.splitlines(keepends=True), content.splitlines(keepends=True),
            fromfile=str(out_path), tofile=f"{args.out} (generated)")
        sys.stdout.writelines(diff)
        print(f"\n{out_path} is STALE -- regenerate with `{_CLI}`.")
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    print(f"wrote {out_path} ({len(content.splitlines())} lines).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
