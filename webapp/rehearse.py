"""Rehearse the demo presets: drive the real web UI in a headless browser, load and
run every preset, and write what the audience will SEE -- one PNG per preset of the
operator card and one of the Results tab -- plus a JSON summary.

    python -m webapp.rehearse [--out DIR] [--only PRESET_ID ...] [--no-cancel]

Why this exists (2026-09-22): the presets had unit tests on their block state, tests on
the figure dictionaries, two code reviews and measured cards -- and the first card on
every Results page was still a "Scenario, plan view" of a lone radar triangle that had
nothing to do with the frames, because nothing in that chain ever rendered the page.
A figure dictionary is not the screen. Run this after any change to the presets, the
runner's figures or the app layout, and READ THE PNGS; the summary alone does not
count as having looked.

With ``--only``, an existing ``summary.json`` in ``--out`` is merged: only the entries
for the presets actually run (and ``cancel_journey`` only if the cancel journey ran) are
replaced, the rest of the previous full run's entries are kept. A full run (no ``--only``)
always writes a fresh file.

Needs Playwright and its Chromium (``playwright install chromium``), torch and the
frames each preset replays. The two GPU jobs a training campaign may have in flight
are unaffected: this only reads checkpoints.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import sys
import time
from typing import Any, Dict, List

_TIMEOUT_MS = 20000
_RUN_TIMEOUT_MS = 240000


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _merge_summary(existing: Dict[str, Any], new: Dict[str, Any],
                    preset_order: List[str]) -> Dict[str, Any]:
    """Combine a previous full (or partial) run's summary with the entries from this
    run, keeping every existing entry this run didn't touch. Result is ordered by
    ``preset_order`` (the registry's stage order), with ``cancel_journey`` last if
    present in either input."""
    merged = {**existing, **new}
    ordered: Dict[str, Any] = {}
    for pid in preset_order:
        if pid in merged:
            ordered[pid] = merged[pid]
    if "cancel_journey" in merged:
        ordered["cancel_journey"] = merged["cancel_journey"]
    return ordered


def _pick_preset(page, label: str) -> None:
    page.click("#preset-select")
    page.get_by_role("option", name=label, exact=True).click(timeout=_TIMEOUT_MS)
    page.wait_for_function(
        """([s, n]) => { const e = document.querySelector(s + '-value');
            return e && e.innerText.includes(n); }""",
        arg=["#preset-select", label], timeout=_TIMEOUT_MS)


def _load_preset(page) -> None:
    before = page.text_content("#preset-notes") or ""
    page.click("#preset-load")
    page.wait_for_function(
        """(prev) => { const e = document.querySelector('#preset-notes');
            return e && e.innerText.trim().length > 0 && e.innerText !== prev; }""",
        arg=before, timeout=_TIMEOUT_MS)


def _run_and_wait(page) -> float:
    # Completion = the result graphs appear, or the status line CHANGES (a failure
    # message). Not a regex on "error": the Thrust 2 label contains that word and
    # sits in the status line after a preset load.
    before = page.text_content("#run-status") or ""
    t0 = time.time()
    page.click("#run-button")
    page.wait_for_function(
        """(prev) => {
            const g = document.querySelector('#results-tab-content .js-plotly-plot');
            if (g) return true;
            const s = document.querySelector('#run-status');
            return !!(s && s.innerText !== prev && !/Cancelling/.test(s.innerText)); }""",
        arg=before, timeout=_RUN_TIMEOUT_MS)
    return time.time() - t0


def _figure_titles(page) -> List[str]:
    """The panel titles as the AUDIENCE sees them.

    Reads `.panel-title`, not Plotly's `.gtitle` (layout spec section 3, 2026-09-24):
    figures carry no title any more -- the title and a one-line caption are HTML above
    the plot -- so a `.gtitle` query would report zero figures on a page full of them.
    """
    return page.evaluate(
        """() => Array.from(document.querySelectorAll(
            '#results-tab-content .panel-title')).map(e => e.textContent)""")


def _expand_details(page) -> int:
    """Open every `Details` disclosure on the Results tab and return how many were
    opened -- so `--expand-details` can capture a SECOND screenshot proving that the
    honesty text the default render hides is one click away and photographable
    (acceptance check 15)."""
    return page.evaluate(
        """() => { const d = Array.from(document.querySelectorAll(
            '#results-tab-content details')); d.forEach(e => e.open = true);
            return d.length; }""")


def _details_text(page) -> List[str]:
    """The text of every expanded `Details` body, for the honesty diff."""
    return page.evaluate(
        """() => Array.from(document.querySelectorAll(
            '#results-tab-content .details-body')).map(e => e.innerText)""")


#: What `_geometry` measures, in the browser, on the real page. A figure-dict test
#: cannot see any of it (memory: "RENDER THE PAGE ... figure-dict tests and code
#: reviews cannot see the screen"), and a pixel-hunt on the PNG has to guess where a
#: panel ends. The DOM knows exactly.
_GEOMETRY_JS = """() => {
  const root = document.getElementById('results-tab-content');
  if (!root) return null;
  const R = e => { const b = e.getBoundingClientRect();
    return {x: b.x + window.scrollX, y: b.y + window.scrollY,
            w: b.width, h: b.height}; };
  const panels = Array.from(root.querySelectorAll('.result-panel')).map(p => {
    const plot = p.querySelector('.js-plotly-plot .nsewdrag');
    const title = p.querySelector('.panel-title');
    const cap = p.querySelector('.panel-caption');
    return {rect: R(p), plot: plot ? R(plot) : null,
            title: title ? title.textContent : '',
            caption: cap ? cap.textContent : '',
            titleClipped: title ? title.scrollWidth > title.clientWidth + 1 : false,
            captionClipped: cap ? cap.scrollWidth > cap.clientWidth + 1 : false};
  });
  // Every text node's rendered font size, so "no text under 15 px" is measured on
  // what the browser actually drew (HTML and SVG both).
  const sizes = {};
  const walk = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  let n;
  while ((n = walk.nextNode())) {
    const t = (n.textContent || '').trim();
    if (!t) continue;
    const el = n.parentElement;
    if (!el) continue;
    const st = window.getComputedStyle(el);
    if (st.display === 'none' || st.visibility === 'hidden') continue;
    // Inside a <details> that is closed: not visible text.
    let d = el.closest('details');
    if (d && !d.open) continue;
    const px = Math.round(parseFloat(st.fontSize) * 10) / 10;
    const inFigure = !!el.closest('.js-plotly-plot');
    const key = px + (inFigure ? '|figure' : '|page');
    (sizes[key] = sizes[key] || []).push(t.slice(0, 60));
  }
  const firstPanel = panels.length ? panels[0].rect.y : null;
  return {
    page: {w: document.documentElement.scrollWidth,
           h: document.documentElement.scrollHeight},
    rootTop: R(root).y,
    firstPanelTop: firstPanel,
    panels: panels,
    fontSizes: sizes,
    nDetails: root.querySelectorAll('details').length,
    nDetailsOpen: root.querySelectorAll('details[open]').length,
    nTransportButtons: root.querySelectorAll('button').length,
    nTransportSliders: root.querySelectorAll('input[type=range]').length,
    nPlotlySliders: root.querySelectorAll('.slider-container').length,
    nPlotlyButtons: root.querySelectorAll('.updatemenu-button').length,
    nFigureTitles: root.querySelectorAll('.js-plotly-plot .gtitle').length,
    plotBg: Array.from(root.querySelectorAll('.js-plotly-plot .bg'))
              .map(e => e.getAttribute('style') || ''),
    visibleText: (root.innerText || '')
  };
}"""


def _geometry(page):
    """Measured page geometry (see `_GEOMETRY_JS`)."""
    return page.evaluate(_GEOMETRY_JS)


def _status_after(page) -> str:
    # A finished run lands on Results, unmounting the diagram tab; re-mounting it
    # restores the last callback outputs, the status line included.
    page.get_by_text("Block Diagram", exact=True).click()
    page.wait_for_selector("#run-status", state="attached", timeout=_TIMEOUT_MS)
    return page.text_content("#run-status") or ""


def rehearse(out: pathlib.Path, only: List[str] | None = None,
             cancel_journey: bool = True, viewport=(1600, 1000),
             expand_details: bool = False) -> Dict[str, Any]:
    """Load, run and screenshot every preset (or those in ``only``); optionally end
    with a 20-frame run cancelled after a few seconds. Returns the summary dict.

    ``expand_details`` additionally opens every ``Details`` disclosure and writes a
    second ``<id>_results_details.png`` plus the disclosure text into the summary --
    the capture the layout spec's acceptance check 15 ("opening all of them loses no
    string that is present in today's screens") is run against."""
    from playwright.sync_api import sync_playwright

    from webapp.demo_presets import PRESETS

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tests" / "e2e_ui"))
    from _ui_server import LiveDashServer  # the same harness the journey tests use

    out.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {}
    presets = [p for p in PRESETS if not only or p.id in only]
    with LiveDashServer() as srv, sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        for p in presets:
            ctx = browser.new_context(viewport={"width": viewport[0], "height": viewport[1]})
            page = ctx.new_page()
            page.set_default_timeout(_TIMEOUT_MS)
            page.goto(srv.url, wait_until="domcontentloaded")
            page.wait_for_selector("h2", state="visible")
            _pick_preset(page, p.label)
            _load_preset(page)
            n_steps = page.input_value("#run-nsteps")
            page.screenshot(path=str(out / f"{p.id}_card.png"), full_page=True)
            wall = _run_and_wait(page)
            page.wait_for_timeout(1500)  # let Plotly finish drawing every card
            titles = _figure_titles(page)
            page.screenshot(path=str(out / f"{p.id}_results.png"), full_page=True)
            geometry = _geometry(page)
            (out / f"{p.id}_geometry.json").write_text(
                json.dumps(geometry, indent=1), encoding="utf-8")
            details_text: List[str] = []
            if expand_details:
                n_open = _expand_details(page)
                page.wait_for_timeout(400)
                details_text = _details_text(page)
                page.screenshot(path=str(out / f"{p.id}_results_details.png"),
                                full_page=True)
                print(f"{p.id}: expanded {n_open} Details disclosure(s)", flush=True)
            status = _status_after(page)
            summary[p.id] = {"label": p.label, "n_steps": n_steps,
                             "wall_s": round(wall, 2), "figures": titles,
                             "status": status, "rendered_at": _now_iso()}
            if expand_details:
                summary[p.id]["details_text"] = details_text
            print(f"{p.id}: n={n_steps} {wall:.1f}s figs={len(titles)} "
                  f"status={status[:100]}", flush=True)
            ctx.close()

        if cancel_journey and presets:
            p = presets[0]
            ctx = browser.new_context(viewport={"width": viewport[0], "height": viewport[1]})
            page = ctx.new_page()
            page.set_default_timeout(_TIMEOUT_MS)
            page.goto(srv.url, wait_until="domcontentloaded")
            page.wait_for_selector("h2")
            _pick_preset(page, p.label)
            _load_preset(page)
            page.fill("#run-nsteps", "20")
            before = page.text_content("#run-status") or ""
            t0 = time.time()
            page.click("#run-button")
            page.wait_for_function(
                "() => !document.querySelector('#cancel-button').disabled",
                timeout=_TIMEOUT_MS)
            page.wait_for_timeout(2500)
            page.click("#cancel-button")
            page.wait_for_function(
                """(prev) => {
                    const g = document.querySelector('#results-tab-content .js-plotly-plot');
                    if (g) return true;
                    const s = document.querySelector('#run-status');
                    return !!(s && s.innerText !== prev && !/Cancelling/.test(s.innerText)); }""",
                arg=before, timeout=_RUN_TIMEOUT_MS)
            wall = time.time() - t0
            page.wait_for_timeout(1000)
            page.screenshot(path=str(out / "cancel_results.png"), full_page=True)
            (out / "cancel_geometry.json").write_text(
                json.dumps(_geometry(page), indent=1), encoding="utf-8")
            status = _status_after(page)
            summary["cancel_journey"] = {"preset": p.id, "wall_s": round(wall, 2),
                                         "status": status, "rendered_at": _now_iso()}
            print(f"cancel ({p.id}, 20 frames): {wall:.1f}s status={status}", flush=True)
            ctx.close()
        browser.close()

    preset_order = [p.id for p in PRESETS]
    summary_path = out / "summary.json"
    if only and summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text())
        except (json.JSONDecodeError, OSError):
            existing = {}
        summary = _merge_summary(existing, summary, preset_order)
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="e2e/main/figures/rehearsal",
                    help="directory for the PNGs and summary.json (gitignored figures dir)")
    ap.add_argument("--only", nargs="*", default=None, help="preset ids to rehearse")
    ap.add_argument("--no-cancel", action="store_true",
                    help="skip the 20-frame run-then-Cancel journey at the end")
    ap.add_argument("--expand-details", action="store_true",
                    help="also open every Details disclosure and write "
                         "<id>_results_details.png + its text into summary.json")
    args = ap.parse_args(argv)
    os.environ.setdefault("MPLBACKEND", "Agg")
    summary = rehearse(pathlib.Path(args.out), only=args.only,
                       cancel_journey=not args.no_cancel,
                       expand_details=args.expand_details)
    print(f"\n{len(summary)} entries -> {args.out}/summary.json. Now READ the PNGs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
