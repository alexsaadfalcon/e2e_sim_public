"""Generate the README's web-UI walkthrough GIF from a scripted browser journey.

    python -m webapp.walkthrough [--out docs/media/ui_walkthrough.gif] [--preset ID]

The journey is what a presenter does: open the app, pick a demo preset, load it (the
operator card and the knob's editor appear), run it, land on the Results tab, turn the
card's live knob, run again and see the before/after pair. Each step is one frame, held
long enough to read. The GIF used to be a hand-made screen recording with no generator;
this module is the generator, so the picture cannot silently drift from the app.

Needs Playwright + Chromium (`playwright install chromium`), torch, and the frames the
chosen preset replays. Uses the same in-thread server as the browser journey tests.
"""

from __future__ import annotations

import argparse
import io
import os
import pathlib
import sys
from typing import List, Optional

_T = 20000


def _frames_for(preset_id: str, viewport=(1280, 1250)) -> List["PIL.Image.Image"]:
    from PIL import Image
    from playwright.sync_api import sync_playwright

    from webapp.demo_presets import PRESETS_BY_ID
    from webapp.rehearse import _load_preset, _pick_preset, _run_and_wait, _status_after

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tests" / "e2e_ui"))
    from _ui_server import LiveDashServer

    preset = PRESETS_BY_ID[preset_id]
    shots: List[Image.Image] = []

    def shot(page, full_page=False):
        png = page.screenshot(full_page=full_page)
        shots.append(Image.open(io.BytesIO(png)).convert("P", palette=Image.ADAPTIVE))

    with LiveDashServer() as srv, sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_context(viewport={"width": viewport[0], "height": viewport[1]}).new_page()
        page.set_default_timeout(_T)
        page.goto(srv.url, wait_until="domcontentloaded")
        page.wait_for_selector("h2")
        page.wait_for_timeout(1500)
        shot(page)                                   # 1. the block diagram
        page.click("#preset-select")
        page.wait_for_timeout(400)
        shot(page)                                   # 2. the preset menu open
        page.get_by_role("option", name=preset.label, exact=True).click()
        page.wait_for_function(
            """([s, n]) => { const e = document.querySelector(s + '-value');
                return e && e.innerText.includes(n); }""",
            arg=["#preset-select", preset.label])
        _load_preset(page)
        page.wait_for_timeout(600)
        shot(page)                                   # 3. card + knob editor
        _run_and_wait(page)
        page.wait_for_timeout(1500)
        shot(page)                                   # 4. results
        _status_after(page)
        page.wait_for_timeout(800)
        if preset.live_knobs:
            # Turn the first live knob: the card's "a -> b" pair names the start and
            # end values; the field currently showing `a` gets `b`.
            import re as _re
            _bid, _key, how = preset.live_knobs[0]
            m = _re.search(r"([0-9.eE+-]+)\s*->\s*([0-9.eE+-]+)", how)
            changed = False
            if m:
                start, target = m.group(1), m.group(2)
                num = page.locator("#block-param-editor input[type=number]")
                for i in range(num.count()):
                    cur = num.nth(i).input_value()
                    try:
                        same = abs(float(cur) - float(start)) < 1e-9
                    except ValueError:
                        same = False
                    if same:
                        num.nth(i).fill(target)
                        num.nth(i).press("Enter")
                        changed = True
                        break
            if changed:
                page.wait_for_timeout(600)
                shot(page)                           # 5. the knob turned
                _run_and_wait(page)
                page.wait_for_timeout(1500)
                shot(page)                           # 6. before/after on one screen
        browser.close()
    return shots


def write_gif(frames, out: pathlib.Path, hold_ms: int = 1800, width: int = 900) -> pathlib.Path:
    from PIL import Image

    resized = []
    for f in frames:
        f = f.convert("RGB")
        h = int(f.height * width / f.width)
        resized.append(f.resize((width, h), Image.LANCZOS)
                       .convert("P", palette=Image.ADAPTIVE, colors=128))
    out.parent.mkdir(parents=True, exist_ok=True)
    resized[0].save(out, save_all=True, append_images=resized[1:], duration=hold_ms, loop=0,
                    optimize=True)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="docs/media/ui_walkthrough.gif")
    ap.add_argument("--preset", default="thrust2_feature_reduction_error",
                    help="preset to walk through (its live knob gives the before/after)")
    ap.add_argument("--hold-ms", type=int, default=1800)
    args = ap.parse_args(argv)
    os.environ.setdefault("MPLBACKEND", "Agg")
    frames = _frames_for(args.preset)
    out = write_gif(frames, pathlib.Path(args.out), hold_ms=args.hold_ms)
    print(f"wrote {out} ({len(frames)} frames, {out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
