"""Generate ``docs/DEMO_RUNBOOK.md`` from the demo presets -- never hand-copy them.

    python -m webapp.runbook [--out docs/DEMO_RUNBOOK.md] \\
        [--summary e2e/main/figures/rehearsal/summary.json] [--check]

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
``do_not_say`` exactly as the dataclass states them today. Wall time comes from a
rehearsal ``summary.json`` when that preset has an entry there, else "measure".

The "Before the audience" / "If something goes wrong" sections describe UI
mechanics (button labels, status strings), not preset data; they are template text
in this module, checked against ``webapp/app.py`` / ``webapp/block_diagram.py`` by
hand, with drift-prone counts (frame ceilings, how many presets set `ab`, the
frame-count range presets ship at) computed at generation time.

Deliberately torch-free: only ``webapp.demo_presets``, ``webapp.pipeline_registry``
and the standard library are imported, so the doc can be regenerated (and
``--check``ed) on any machine, CI included, without installing torch/Sionna.
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
_DEFAULT_SUMMARY = _REPO_ROOT / "e2e" / "main" / "figures" / "rehearsal" / "summary.json"
_CLI = "python -m webapp.runbook"
_WIDTH = 79


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


def _enabled_product_labels(preset: DemoPreset) -> List[str]:
    """Product-block labels this preset's as-loaded (arm A) state enables, in
    registry order -- read from `apply_preset`, never from a hand-kept list, so a
    preset that stops disabling a product is reflected here automatically."""
    state = apply_preset(preset)
    return [_block_label(bid) for bid in PRODUCT_IDS if state[bid]["enabled"]]


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


def _wall_time(preset: DemoPreset, summary: Dict[str, Any]) -> str:
    entry = summary.get(preset.id)
    if not entry or entry.get("wall_s") is None:
        return f"measure (no entry for this preset in the rehearsal summary; n_steps={preset.n_steps})"
    n = entry.get("n_steps", preset.n_steps)
    return f"**{entry['wall_s']:.2f}s** (n_steps={n}, from the rehearsal summary)"


# =====================================================================================
# Per-preset section
# =====================================================================================

def _render_preset(i: int, preset: DemoPreset, summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"## {i}. {preset.label}\n")

    open_block, open_param = _opening_block(preset)
    open_desc = (f"**{open_block}**" + (f" (first knob: **{open_param}**)" if open_param else ""))
    ab = _ab_lines(preset)
    arm_sentence = (f" Both arms run in one click (A = {preset.ab_label_a}, "
                    f"B = {preset.ab_label_b})." if preset.ab is not None else "")

    lines.append("### Click sequence")
    lines.append(_numbered(1,
        f'Open **Demo preset:**, select "{preset.label}", click **Load preset**. '
        f'The param editor opens on {open_desc}; the operator card shows "Loaded: '
        f'{preset.label} (Thrust {preset.thrust}, {preset.n_steps} frames)".'))
    lines.append(_numbered(2,
        f"Click **Run pipeline**.{arm_sentence} Wall time: {_wall_time(preset, summary)}."))
    lines.append(_numbered(3, "The app switches to the **Results** tab automatically."))
    lines.append("")

    lines.append("### What you are looking at")
    products = _enabled_product_labels(preset)
    if products:
        lines.append(_bullet("Product panel(s) this preset enables: "
                             + ", ".join(f"**{p}**" for p in products) + "."))
    else:
        lines.append(_bullet("This preset enables no product panel (check the block state)."))
    if ab is not None:
        lines.append(_bullet(f'Arm banners on screen: "{ab[0]}" / "{ab[1]}".'))
    lines.append("")

    lines.append("### Second knob (optional)")
    extra = _second_knobs(preset)
    if extra:
        for bid, key, how in extra:
            bounds = _param_bounds(bid, key)
            bounds_part = f" ({bounds})" if bounds else ""
            lines.append(_bullet(f"**{_block_label(bid)}** -> **{_param_label(bid, key)}**"
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
5. Presenting over RDP: advance frames with the Results-tab figure's own frame
   **slider** (the widget Plotly draws under each heatmap/animation), never the
   ▶ (Play) control -- its ~350 ms/frame animation stutters over the link.
"""


def _click_mechanics(n_presets: int, n_ab: int) -> str:
    if n_ab == n_presets:
        ab_sentence = (f"Every one of the {n_presets} presets below sets `ab`, so one "
                       "click on **Run pipeline** runs BOTH arms A and B and renders "
                       "both on the Results tab (A on top, B below, under a "
                       "\"Previous run\" divider) -- there is no second Run click "
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
    ]
    body = "\n".join(_bullet(b) for b in bullets)
    return "## If something goes wrong\n\n" + body + "\n"


# =====================================================================================
# Top-level render + CLI
# =====================================================================================

def render(presets: List[DemoPreset], summary: Dict[str, Any], cmd: str = _CLI) -> str:
    """Render the full runbook markdown for `presets` given a rehearsal `summary`
    dict (preset id -> {"wall_s": ..., "n_steps": ...}, as written by
    `webapp.rehearse`; missing/empty is fine, every preset just reads "measure").

    No commit SHA is embedded: a SHA inside a file committed AT that SHA can
    never equal the commit that contains it, so a `--check` gate tied to one
    would fail at every commit that doesn't touch the presets -- exactly the
    drifting-value-in-a-durable-document failure this generator exists to
    avoid. Regenerate after any change to `webapp/demo_presets.py` or the
    rehearsal summary; `--check` is what enforces that, not a SHA comparison."""
    n_steps_vals = [p.n_steps for p in presets] or [0]
    n_ab = sum(1 for p in presets if p.ab is not None)

    parts: List[str] = []
    parts.append("# Demo runbook -- CogniSense Annual Review\n\n" + _wrap(
        f"Generated by `{cmd}` from `webapp/demo_presets.py` (every preset's "
        "blurb / live_knobs / say / do_not_say -- the source of truth) and "
        "`webapp/pipeline_registry.py` (product-panel labels, knob ranges); "
        "wall times come from a rehearsal summary when a preset has an entry "
        'there, else "measure". Every UI label quoted below is the exact '
        "string in `webapp/app.py` or `webapp/block_diagram.py`. Regenerate "
        "after any change to `webapp/demo_presets.py` or the rehearsal "
        "summary.") + "\n")
    parts.append(_BEFORE_AUDIENCE)
    parts.append("Preset stage order (`PRESETS` in `webapp/demo_presets.py`):\n")
    parts.append("\n".join(f"{i}. {p.label}" for i, p in enumerate(presets, 1)) + "\n")
    parts.append(_click_mechanics(len(presets), n_ab))
    parts.append("---\n")
    for i, p in enumerate(presets, 1):
        parts.append(_render_preset(i, p, summary))
        parts.append("---\n")
    parts.append(_trouble(min(n_steps_vals), max(n_steps_vals)))
    return "\n".join(parts).rstrip() + "\n"


def _load_summary(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(_DEFAULT_OUT),
                    help="path to write the generated runbook")
    ap.add_argument("--summary", default=str(_DEFAULT_SUMMARY),
                    help="rehearsal summary.json to read wall times from")
    ap.add_argument("--check", action="store_true",
                    help="don't write; exit 1 (and print a diff) if --out differs "
                         "from what would be generated")
    args = ap.parse_args(argv)

    out_path = Path(args.out)
    summary = _load_summary(Path(args.summary))
    content = render(PRESETS, summary)

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
