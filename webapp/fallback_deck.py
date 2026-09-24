"""webapp.fallback_deck -- build a PDF slide deck from the demo's rehearsal screenshots.

LAST RESORT ONLY: `webapp.preflight` exists precisely so this should never be needed on
the day. If the live demo cannot run and there is no time left to fix it, this turns the
already-captured rehearsal PNGs (written by `python -m webapp.rehearse` to
`e2e/main/figures/rehearsal/`: `<preset_id>_card.png`, `<preset_id>_results.png`,
`cancel_results.png`, `summary.json`) into a single PDF that can be presented instead.

Command:
    python -m webapp.fallback_deck [--in e2e/main/figures/rehearsal] \\
        [--out e2e/main/figures/rehearsal/fallback_deck.pdf]

Torch-free: the only project import is `webapp.demo_presets` (itself torch-free, see its
own module docstring), used only as a fallback for preset order/labels when
`summary.json` is missing or does not name a preset. Pillow (already a dependency) does
all PDF writing, so no new dependency is added.

Pagination policy (why a threshold, and which one): Playwright's `full_page`
screenshots capture the entire scrolled page, so a Results-tab screenshot is far taller
than a landscape page is wide -- shrinking it to fit BOTH page dimensions would, past a
point, shrink its on-screen UI text (Dash's ~14-16px copy) below legibility. The rule
used here: compute the scale that fits the image inside the page's image area on BOTH
axes; if that scale is below MIN_FIT_SCALE = 0.5 (i.e. it would need to shrink by more
than 2x), the image is instead split into consecutive pages, each a horizontal band
sized so that scaling it to fit the page WIDTH exactly fills the page height (that
per-band scale is >= MIN_FIT_SCALE by construction, since screenshots are exactly the
browser viewport width and never wider than the page's image area). Short images (the
operator cards, and any results page that isn't unusually tall) get one page each, no
split, no cropping.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont

# Landscape "page" canvas in pixels -- roughly a landscape US Letter at 150 dpi
# (11in x 8.5in -> 1650 x 1275), rendered directly as a raster page rather than vector
# so all page-building (title pages and image pages alike) goes through one Pillow
# multi-page PDF write.
PAGE_W, PAGE_H = 1650, 1275
MARGIN = 50
TITLE_H = 110  # space reserved at the top of image pages for the title line(s)

#: See "Pagination policy" in the module docstring.
MIN_FIT_SCALE = 0.5

_FONT_CANDIDATES = ("arial.ttf", "DejaVuSans.ttf", r"C:\Windows\Fonts\arial.ttf",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


def _font(size: int) -> ImageFont.FreeTypeFont:
    for name in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def _avail_area() -> Tuple[int, int]:
    return PAGE_W - 2 * MARGIN, PAGE_H - TITLE_H - MARGIN


def _paginate_image(img: Image.Image) -> List[Image.Image]:
    """Split ``img`` into one or more horizontal bands, one per output page. See the
    module docstring's "Pagination policy" for the threshold and reasoning."""
    avail_w, avail_h = _avail_area()
    w, h = img.size
    scale_w = avail_w / w
    scale_h = avail_h / h
    if min(scale_w, scale_h) >= MIN_FIT_SCALE:
        return [img]
    band_h_px = max(1, int(round(avail_h / scale_w)))
    bands = []
    y = 0
    while y < h:
        bands.append(img.crop((0, y, w, min(h, y + band_h_px))))
        y += band_h_px
    return bands


def _make_image_page(title: str, img: Image.Image) -> Image.Image:
    page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    draw = ImageDraw.Draw(page)
    draw.text((MARGIN, MARGIN // 2), title, fill="black", font=_font(22))
    avail_w, avail_h = _avail_area()
    w, h = img.size
    scale = min(avail_w / w, avail_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    x = (PAGE_W - new_w) // 2
    y = TITLE_H + (avail_h - new_h) // 2
    page.paste(resized, (x, y))
    return page


def _git_head(repo_root: Path) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(repo_root),
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown (git rev-parse failed)"


def _make_title_page(in_dir: Path) -> Image.Image:
    page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    draw = ImageDraw.Draw(page)
    lines = [
        "Demo fallback deck (last resort -- see webapp.preflight)",
        f"generated: {datetime.now().isoformat(timespec='seconds')}",
        f"git HEAD: {_git_head(Path(__file__).resolve().parents[1])}",
        f"source: {in_dir}",
    ]
    y = PAGE_H // 3
    for i, line in enumerate(lines):
        size = 34 if i == 0 else 22
        draw.text((MARGIN, y), line, fill="black", font=_font(size))
        y += size + 20
    return page


def _load_order_and_labels(in_dir: Path) -> Tuple[List[str], Dict[str, str], bool]:
    """Preset id order + id->label map, from summary.json if present and non-empty,
    else from webapp.demo_presets.PRESETS (stage order). Also reports whether a cancel
    page should be looked for."""
    summary: Dict = {}
    summary_path = in_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except Exception as exc:
            print(f"warning: could not parse {summary_path}: {exc}")
            summary = {}

    order = [k for k in summary if k != "cancel_journey"]
    labels = {k: v.get("label", k) for k, v in summary.items()
             if k != "cancel_journey" and isinstance(v, dict)}

    if not order:
        from webapp.demo_presets import PRESETS
        order = [p.id for p in PRESETS]
        labels = {p.id: p.label for p in PRESETS}
    else:
        # Fill any id present in the order but missing a label (summary entry with no
        # "label" key) from the registry, best effort.
        missing = [pid for pid in order if pid not in labels]
        if missing:
            from webapp.demo_presets import PRESETS
            preset_labels = {p.id: p.label for p in PRESETS}
            for pid in missing:
                labels[pid] = preset_labels.get(pid, pid)

    has_cancel = "cancel_journey" in summary
    cancel_label = ""
    if has_cancel and isinstance(summary.get("cancel_journey"), dict):
        cancel_pid = summary["cancel_journey"].get("preset", "")
        cancel_label = labels.get(cancel_pid, cancel_pid)
    return order, labels, cancel_label


def build_deck(in_dir: Path, out_path: Path) -> Tuple[int, int]:
    """Build the fallback PDF from PNGs under ``in_dir`` into ``out_path``.

    Returns ``(n_pngs_found, n_pages_written)``. Missing PNGs are skipped with a
    printed warning, never raised; if NO PNG is found at all, nothing is written and
    ``(0, 0)`` is returned -- the caller (``main``) is the one that turns that into a
    nonzero exit code.
    """
    order, labels, cancel_label = _load_order_and_labels(in_dir)

    pages: List[Image.Image] = []
    n_pngs = 0
    for pid in order:
        label = labels.get(pid, pid)
        for suffix in ("card", "results"):
            png_path = in_dir / f"{pid}_{suffix}.png"
            if not png_path.exists():
                print(f"warning: missing {png_path}, skipping")
                continue
            n_pngs += 1
            img = Image.open(png_path).convert("RGB")
            bands = _paginate_image(img)
            for i, band in enumerate(bands):
                part = f" (part {i + 1}/{len(bands)})" if len(bands) > 1 else ""
                title = f"{label} -- {suffix}{part} -- {png_path.name}"
                pages.append(_make_image_page(title, band))

    cancel_path = in_dir / "cancel_results.png"
    if cancel_path.exists():
        n_pngs += 1
        img = Image.open(cancel_path).convert("RGB")
        bands = _paginate_image(img)
        for i, band in enumerate(bands):
            part = f" (part {i + 1}/{len(bands)})" if len(bands) > 1 else ""
            prefix = f"Cancel journey ({cancel_label})" if cancel_label else "Cancel journey"
            title = f"{prefix}{part} -- {cancel_path.name}"
            pages.append(_make_image_page(title, band))
    elif cancel_label:
        print(f"warning: missing {cancel_path}, skipping")

    if n_pngs == 0:
        print(f"error: no rehearsal PNGs found under {in_dir}")
        return 0, 0

    all_pages = [_make_title_page(in_dir)] + pages
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_pages[0].save(str(out_path), save_all=True, append_images=all_pages[1:])
    return n_pngs, len(all_pages)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--in", dest="in_dir", default="e2e/main/figures/rehearsal",
                    help="directory containing the rehearsal PNGs + summary.json")
    ap.add_argument("--out", dest="out_path",
                    default="e2e/main/figures/rehearsal/fallback_deck.pdf",
                    help="output PDF path")
    args = ap.parse_args(argv)

    in_dir = Path(args.in_dir)
    out_path = Path(args.out_path)
    n_pngs, n_pages = build_deck(in_dir, out_path)
    if n_pngs == 0:
        return 1
    print(f"wrote {out_path} ({n_pages} pages from {n_pngs} PNGs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
