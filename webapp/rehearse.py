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

Needs Playwright and its Chromium (``playwright install chromium``), torch and the
frames each preset replays. The two GPU jobs a training campaign may have in flight
are unaffected: this only reads checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Any, Dict, List

_TIMEOUT_MS = 20000
_RUN_TIMEOUT_MS = 240000


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
    return page.evaluate(
        """() => Array.from(document.querySelectorAll(
            '#results-tab-content .js-plotly-plot .gtitle')).map(e => e.textContent)""")


def _status_after(page) -> str:
    # A finished run lands on Results, unmounting the diagram tab; re-mounting it
    # restores the last callback outputs, the status line included.
    page.get_by_text("Block Diagram", exact=True).click()
    page.wait_for_selector("#run-status", state="attached", timeout=_TIMEOUT_MS)
    return page.text_content("#run-status") or ""


def rehearse(out: pathlib.Path, only: List[str] | None = None,
             cancel_journey: bool = True, viewport=(1600, 1000)) -> Dict[str, Any]:
    """Load, run and screenshot every preset (or those in ``only``); optionally end
    with a 20-frame run cancelled after a few seconds. Returns the summary dict."""
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
            status = _status_after(page)
            summary[p.id] = {"label": p.label, "n_steps": n_steps,
                             "wall_s": round(wall, 2), "figures": titles,
                             "status": status}
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
            status = _status_after(page)
            summary["cancel_journey"] = {"preset": p.id, "wall_s": round(wall, 2),
                                         "status": status}
            print(f"cancel ({p.id}, 20 frames): {wall:.1f}s status={status}", flush=True)
            ctx.close()
        browser.close()

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="e2e/main/figures/rehearsal",
                    help="directory for the PNGs and summary.json (gitignored figures dir)")
    ap.add_argument("--only", nargs="*", default=None, help="preset ids to rehearse")
    ap.add_argument("--no-cancel", action="store_true",
                    help="skip the 20-frame run-then-Cancel journey at the end")
    args = ap.parse_args(argv)
    os.environ.setdefault("MPLBACKEND", "Agg")
    summary = rehearse(pathlib.Path(args.out), only=args.only,
                       cancel_journey=not args.no_cancel)
    print(f"\n{len(summary)} entries -> {args.out}/summary.json. Now READ the PNGs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
