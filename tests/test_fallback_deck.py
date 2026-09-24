"""Tests for webapp.fallback_deck: the last-resort PDF built from rehearsal PNGs.

No PDF-reading library is installed in this environment (checked: no PyPDF2/pypdf), so
page count is verified with a byte-level regex on the PDF's own `/Type /Page` object
markers (excluding the `/Type /Pages` catalog object via a negative lookahead) -- this
was cross-checked by hand against a known-page-count PDF written by the same Pillow
`save_all` path (3 images -> 3 `/Type /Page` matches, 1 `/Type /Pages`).
"""

import json
import re

from PIL import Image

from webapp.fallback_deck import build_deck, _paginate_image

_PAGE_RE = re.compile(rb"/Type\s*/Page(?!s)")


def _count_pdf_pages(pdf_bytes: bytes) -> int:
    return len(_PAGE_RE.findall(pdf_bytes))


def _write_png(path, size, color):
    Image.new("RGB", size, color).save(path)


def test_build_deck_pages_and_split(tmp_path):
    """Two presets (one with a results screenshot taller than a page) plus a cancel
    page; page count = title + each PNG's own pagination (>1 page for the tall one)."""
    _write_png(tmp_path / "demo_one_card.png", (800, 600), "red")
    _write_png(tmp_path / "demo_one_results.png", (800, 600), "blue")
    _write_png(tmp_path / "demo_two_card.png", (800, 600), "green")
    tall = (1600, 6000)  # taller than a page -> must split, per the module's own policy
    _write_png(tmp_path / "demo_two_results.png", tall, "yellow")
    _write_png(tmp_path / "cancel_results.png", (800, 600), "purple")
    summary = {
        "demo_one": {"label": "Demo One"},
        "demo_two": {"label": "Demo Two"},
        "cancel_journey": {"preset": "demo_one"},
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))

    out_pdf = tmp_path / "fallback_deck.pdf"
    n_pngs, n_pages = build_deck(tmp_path, out_pdf)

    assert n_pngs == 5  # 2 cards + 2 results + 1 cancel
    assert out_pdf.exists()
    assert out_pdf.stat().st_size > 0

    tall_bands = len(_paginate_image(Image.open(tmp_path / "demo_two_results.png")))
    assert tall_bands > 1, "test fixture must actually exercise the split path"
    expected_pages = 1 + 1 + 1 + 1 + tall_bands + 1  # title + 3 single pages + split + cancel
    assert n_pages == expected_pages

    pdf_bytes = out_pdf.read_bytes()
    assert _count_pdf_pages(pdf_bytes) == expected_pages


def test_missing_png_warns_not_raises(tmp_path, capsys):
    """A preset named in summary.json with no PNGs on disk is skipped with a warning;
    the deck still builds from whatever PNGs ARE present."""
    _write_png(tmp_path / "present_card.png", (400, 300), "red")
    _write_png(tmp_path / "present_results.png", (400, 300), "blue")
    summary = {
        "present": {"label": "Present Preset"},
        "ghost": {"label": "Ghost Preset"},
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))

    out_pdf = tmp_path / "fallback_deck.pdf"
    n_pngs, n_pages = build_deck(tmp_path, out_pdf)

    captured = capsys.readouterr()
    assert "warning" in captured.out.lower()
    assert "ghost" in captured.out
    assert n_pngs == 2
    assert n_pages == 1 + 2  # title + the two present-preset pages
    assert out_pdf.exists()


def test_no_pngs_found_returns_zero_without_crashing(tmp_path, capsys):
    (tmp_path / "summary.json").write_text(json.dumps({"ghost": {"label": "Ghost"}}))
    out_pdf = tmp_path / "fallback_deck.pdf"

    n_pngs, n_pages = build_deck(tmp_path, out_pdf)

    captured = capsys.readouterr()
    assert n_pngs == 0
    assert n_pages == 0
    assert not out_pdf.exists()
    assert "no rehearsal png" in captured.out.lower()


def test_fallback_to_demo_presets_order_without_summary(tmp_path):
    """No summary.json at all: order/labels fall back to webapp.demo_presets.PRESETS
    (torch-free import) rather than crashing on a missing file."""
    from webapp.demo_presets import PRESETS

    first = PRESETS[0]
    _write_png(tmp_path / f"{first.id}_card.png", (400, 300), "red")
    _write_png(tmp_path / f"{first.id}_results.png", (400, 300), "blue")

    out_pdf = tmp_path / "fallback_deck.pdf"
    n_pngs, n_pages = build_deck(tmp_path, out_pdf)

    assert n_pngs == 2
    assert n_pages == 3  # title + card + results
    assert out_pdf.exists()
    assert _count_pdf_pages(out_pdf.read_bytes()) == 3
