"""Tests for `e2e.main.main_subspace_refine` -- the subspace-collapse/recovery figure.

The pipeline runs themselves need `munich.pkl` and minutes of compute, so these tests
exercise the parts that decide what the figure SAYS: which stretches count as collapsed,
which count as recoveries, and whether the summary numbers are computed over the right
frames. Those are the parts that can be silently wrong while still producing a picture.
"""

import json

import pytest

pytest.importorskip("matplotlib")

from e2e.main import main_subspace_refine as msr


# --------------------------------------------------------------------------------
# _degenerate_runs / _recovered_windows
# --------------------------------------------------------------------------------
def test_degenerate_runs_splits_at_a_recovery_instead_of_spanning_it():
    """The whole point of the figure is the stretch where the gap comes BACK.

    An earlier version took the outer first-to-last span of sub-threshold frames, which on
    a real 100-frame munich run (collapse at 22, recovery at 91-96, collapse again at 97)
    shaded straight over the recovery and hid it.
    """
    gaps = [0.2] * 5 + [1e-3] * 10 + [0.2] * 4 + [1e-3] * 3
    runs = msr._degenerate_runs(gaps, threshold=0.01)
    assert runs == [(5, 14), (19, 21)]


def test_degenerate_runs_ignores_single_frame_flickers():
    """One frame dipping below threshold is noise, not a collapse; shading it would turn
    the figure into a barcode."""
    gaps = [0.2, 0.2, 1e-3, 0.2, 0.2, 1e-3, 1e-3, 0.2]
    assert msr._degenerate_runs(gaps, threshold=0.01) == [(5, 6)]
    # ...but with min_len=1 the flicker is visible, so the filter is a choice, not a bug.
    assert msr._degenerate_runs(gaps, threshold=0.01, min_len=1) == [(2, 2), (5, 6)]


def test_degenerate_runs_treats_nan_as_not_collapsed():
    """`sv_gap_norm` is NaN when the spectrum is degenerate enough that the gap is
    undefined. NaN comparisons are always False, so NaN must not silently read as
    'below threshold' -- that would invent a collapse."""
    gaps = [0.2, float("nan"), float("nan"), 0.2]
    assert msr._degenerate_runs(gaps, threshold=0.01) == []


def test_degenerate_runs_closes_a_run_that_reaches_the_last_frame():
    gaps = [0.2, 0.2, 1e-3, 1e-3, 1e-3]
    assert msr._degenerate_runs(gaps, threshold=0.01) == [(2, 4)]


def test_recovered_windows_are_only_the_gaps_BETWEEN_collapses():
    """The clean stretch before the first collapse is not a 'recovery' -- nothing had
    broken yet. Counting it would overstate what the gate achieves."""
    gaps = [0.2] * 5 + [1e-3] * 10 + [0.2] * 4 + [1e-3] * 3
    assert msr._recovered_windows(gaps, threshold=0.01) == [(15, 18)]


def test_recovered_windows_empty_when_the_gap_never_comes_back():
    gaps = [0.2] * 5 + [1e-3] * 10
    assert msr._recovered_windows(gaps, threshold=0.01) == []


# --------------------------------------------------------------------------------
# summarize
# --------------------------------------------------------------------------------
def _arms():
    """Two synthetic arms: identical before the collapse, then the gate wins 10x."""
    gaps = [0.2] * 4 + [1e-3] * 6 + [0.2] * 4 + [1e-3] * 3
    base = {"subspace_err": [0.5] * 4 + [1.0] * 6 + [0.6] * 4 + [1.0] * 3,
            "sv_gap_norm": gaps, "n_refine_used": [1.0] * 17}
    refine = {"subspace_err": [0.5] * 4 + [0.1] * 6 + [0.3] * 4 + [0.1] * 3,
              "sv_gap_norm": gaps,
              "n_refine_used": [1.0] * 4 + [60.0] * 6 + [1.0] * 4 + [60.0] * 3}
    return base, refine


def test_summarize_compares_the_arms_only_over_collapsed_frames():
    """Averaging over the whole run would dilute the effect with frames on which the two
    arms are identical by construction, understating the gate."""
    base, refine = _arms()
    s = msr.summarize(base, refine, threshold=0.01)
    # 4 clean, 6 collapsed, 4 clean, 3 collapsed -> 17 frames
    assert s["degenerate_runs"] == [(4, 9), (14, 16)]
    assert s["recovery_windows"] == [(10, 13)]
    assert s["degenerate_err_baseline"] == pytest.approx(1.0)
    assert s["degenerate_err_refine"] == pytest.approx(0.1)
    assert s["degenerate_improvement"] == pytest.approx(0.9)


def test_summarize_reports_pre_collapse_error_separately():
    """"Recovered" only means something against what the arm managed BEFORE the collapse."""
    base, refine = _arms()
    s = msr.summarize(base, refine, threshold=0.01)
    assert s["pre_collapse_err_refine"] == pytest.approx(0.5)
    assert s["recovered_err_refine"] == pytest.approx(0.3)


def test_summarize_survives_a_run_with_no_collapse_at_all():
    """A healthy scene must produce a figure and a summary, not a KeyError."""
    arm = {"subspace_err": [0.1] * 6, "sv_gap_norm": [0.2] * 6, "n_refine_used": [1.0] * 6}
    s = msr.summarize(arm, arm, threshold=0.01)
    assert s["degenerate_runs"] == [] and s["recovery_windows"] == []
    assert s["frames_degenerate"] == 0
    assert "degenerate_improvement" not in s


# --------------------------------------------------------------------------------
# figure + cache
# --------------------------------------------------------------------------------
def test_build_figure_writes_a_png_and_creates_parent_dirs(tmp_path):
    base, refine = _arms()
    out = tmp_path / "nested" / "refine.png"
    assert msr.build_figure(base, refine, out, threshold=0.01) == out
    assert out.is_file() and out.stat().st_size > 0


def test_build_figure_handles_a_run_with_no_collapse(tmp_path):
    arm = {"subspace_err": [0.1] * 6, "sv_gap_norm": [0.2] * 6, "n_refine_used": [1.0] * 6}
    out = tmp_path / "flat.png"
    msr.build_figure(arm, arm, out, threshold=0.01)
    assert out.is_file() and out.stat().st_size > 0


def test_cache_refuses_to_plot_a_run_it_does_not_describe(tmp_path, capsys):
    """The cache exists so figure iteration is free. That makes it dangerous: silently
    plotting a 60-frame munich run under a `--n-steps 100 --scenario etoile` command line
    would be exactly the kind of plausible-looking wrong result that is hardest to catch.
    """
    base, refine = _arms()
    cache = tmp_path / "runs.json"
    cache.write_text(json.dumps({"scenario": "munich", "n_steps": 60,
                                 "base": base, "refine": refine}))
    with pytest.raises(SystemExit, match="munich"):
        msr.main(["--scenario", "etoile", "--n-steps", "100",
                  "--cache", str(cache), "--out", str(tmp_path / "x.png")])


def test_cache_round_trips_and_replots_without_running_the_pipeline(tmp_path, capsys):
    base, refine = _arms()
    cache = tmp_path / "runs.json"
    cache.write_text(json.dumps({"scenario": "munich", "n_steps": 17,
                                 "base": base, "refine": refine}))
    out = tmp_path / "cached.png"
    assert msr.main(["--scenario", "munich", "--n-steps", "17",
                     "--cache", str(cache), "--out", str(out)]) == 0
    assert out.is_file() and out.stat().st_size > 0
    assert "reusing cached" in capsys.readouterr().out
