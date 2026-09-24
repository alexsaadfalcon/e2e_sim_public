"""Tests for the generated demo runbook (webapp/runbook.py).

The doc used to be hand-copied from webapp/demo_presets.py and went stale on every
review wave. These tests pin: (a) generation covers every preset in stage order,
(b) `--check` passes on the committed docs/DEMO_RUNBOOK.md (i.e. it really is
regenerated output, not hand-edited since), (c) the module never imports torch.
"""

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUNBOOK = _REPO_ROOT / "docs" / "DEMO_RUNBOOK.md"


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

    doc = render(PRESETS, summary={"thrust2_feature_reduction_error":
                                   {"wall_s": 16.8, "n_steps": 6}})

    positions = [doc.index(p.label) for p in PRESETS]
    assert positions == sorted(positions), "presets must appear in PRESETS order"
    # Every preset's numbered heading is present.
    for i, p in enumerate(PRESETS, 1):
        assert f"## {i}. {p.label}" in doc


def test_render_uses_summary_wall_time_and_measure_fallback():
    from webapp.demo_presets import PRESETS
    from webapp.runbook import render

    target = PRESETS[0].id
    other = PRESETS[1].id if len(PRESETS) > 1 else target
    doc = render(PRESETS, summary={target: {"wall_s": 12.34, "n_steps": PRESETS[0].n_steps}})
    assert "12.34s" in doc
    # A preset with no summary entry falls back to "measure", never a fabricated number.
    if other != target:
        assert "measure" in doc


def test_render_drops_element_id_parentheticals():
    from webapp.demo_presets import PRESETS
    from webapp.runbook import render

    doc = render(PRESETS, summary={})
    for leaked_id in ("(preset-select)", "(preset-load)", "(run-nsteps)",
                      "(run-button)", "(cancel-button)", "(run-status)"):
        assert leaked_id not in doc


def test_render_has_no_commit_sha():
    """A SHA embedded in a file committed AT that SHA can never match the commit
    that contains it -- `--check` must not depend on git state, only on the
    presets/summary, so the same render() output is expected regardless of
    which commit produced it."""
    from webapp.demo_presets import PRESETS
    from webapp.runbook import render

    doc = render(PRESETS, summary={})
    assert "HEAD" not in doc
    assert "Regenerate after any change" in doc


def test_render_literalises_play_control():
    from webapp.demo_presets import PRESETS
    from webapp.runbook import render

    doc = render(PRESETS, summary={})
    assert "▶ (Play) control" in doc


def test_render_drops_sources_note():
    from webapp.demo_presets import PRESETS
    from webapp.runbook import render

    doc = render(PRESETS, summary={})
    assert "Sources note" not in doc


def test_check_flag_detects_staleness(tmp_path):
    from webapp.runbook import main

    out = tmp_path / "runbook.md"
    assert main(["--out", str(out), "--summary", str(tmp_path / "missing.json")]) == 0
    assert out.is_file()
    # Freshly generated -> --check passes.
    assert main(["--out", str(out), "--summary", str(tmp_path / "missing.json"), "--check"]) == 0
    # Hand-edit it -> --check must fail.
    out.write_text(out.read_text() + "\nstray hand edit\n")
    assert main(["--out", str(out), "--summary", str(tmp_path / "missing.json"), "--check"]) == 1


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
