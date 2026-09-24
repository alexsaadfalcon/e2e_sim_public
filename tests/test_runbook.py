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
    (the rehearsal summary and the preflight timing pass) with the WARN budget
    read from webapp.preflight.WARN_SECONDS, never a number that came from an
    actual run."""
    from webapp.demo_presets import PRESETS
    from webapp.preflight import WARN_SECONDS
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
    # The WARN budget is read from webapp.preflight, not retyped.
    assert f"{WARN_SECONDS:g} s WARN budget" in flat


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
