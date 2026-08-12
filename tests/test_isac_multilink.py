"""Smoke test for the multi-link ISAC example (e2e.main.main_isac_multilink).

Runs the synthetic-fallback path in-process (no real Sionna .pkl on disk), with
the cache path and figures directory redirected into ``tmp_path`` -- never
touches ``e2e/environment/sionna_sims/`` or ``e2e/main/figures/`` (matches the
convention used by ``tests/test_isac_multilink_demo.py`` and
``tests/test_comms_head_example.py``). This complements (does not duplicate)
``test_isac_multilink_demo.py``'s deeper per-function coverage by focusing on
the module's top-level contract: it runs end to end with zero preexisting
assets, clearly labels the synthetic fallback, and produces one figure per
link comparison with finite per-link BER/EVM/range metrics.
"""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import matplotlib
matplotlib.use("Agg")            # headless, matches the example's own convention

from e2e.main.main_isac_multilink import main


N_FRAMES = 3
N_FREQS = 16


def test_synthetic_fallback_runs_with_zero_preexisting_assets(tmp_path):
    """No cache_path is reused from a previous run (fresh tmp dir) and no real
    munich_isac.pkl exists at the canonical path, so this must exercise the
    in-process synthetic-fallback generation end to end."""
    cache_path = str(tmp_path / "isac_multilink.pkl")
    fig_dir = str(tmp_path / "figures")
    assert not os.path.isfile(cache_path)

    result = main(cache_path=cache_path, fig_dir=fig_dir,
                 num_frames=N_FRAMES, num_freqs=N_FREQS, verbose=False)

    assert os.path.isfile(cache_path)
    assert result["data_source"] == "synthetic dry-run demo"


def test_figures_created_in_redirected_dir_only(tmp_path):
    fig_dir = str(tmp_path / "figures")
    result = main(cache_path=str(tmp_path / "cache.pkl"), fig_dir=fig_dir,
                 num_frames=N_FRAMES, num_freqs=N_FREQS, verbose=False)

    figures = result["figures"]
    assert len(figures) >= 1
    for fig_path in figures:
        assert fig_path.startswith(fig_dir)
        assert os.path.isfile(fig_path)
        assert os.path.getsize(fig_path) > 0


def test_per_link_metrics_are_finite(tmp_path):
    """Both legs (sensing off the radar link, comm BER/EVM off the comm link)
    read from the SAME multi-link payload and produce finite, sane metrics."""
    result = main(cache_path=str(tmp_path / "cache.pkl"),
                 fig_dir=str(tmp_path / "figures"),
                 num_frames=N_FRAMES, num_freqs=N_FREQS, verbose=False)

    assert result["radar_link"] != result["comm_link"]

    # sensing leg
    assert np.all(np.isfinite(result["ranges"]))
    assert np.all(np.isfinite(result["profiles"]))
    assert np.all(np.isfinite(result["peak_ranges"]))
    assert result["peak_ranges"].shape == (N_FRAMES,)

    # comm leg: BER/EVM table (per SNR) + per-frame BER
    comm = result["comm"]
    assert len(comm["ber"]) == len(comm["snr_list"]) > 0
    assert all(np.isfinite(b) and 0.0 <= b <= 1.0 for b in comm["ber"])
    assert all(np.isfinite(e) for e in comm["evm_pct"])
    assert result["ber_per_frame"].shape == (N_FRAMES,)
    assert np.all(np.isfinite(result["ber_per_frame"]))
    assert np.all((result["ber_per_frame"] >= 0.0) & (result["ber_per_frame"] <= 1.0))


def test_prefers_real_pkl_over_synthetic_when_present_at_canonical_path(tmp_path, monkeypatch):
    """When a (fake, dry-run-generated here) 'real' pkl already sits at the
    canonical scenario_runner output path, the demo must use it as-is instead of
    generating its own synthetic fallback -- see main_isac_multilink.main()'s
    canonical-path check."""
    import e2e.main.main_isac_multilink as m
    from e2e.environment.scenario_runner import ScenarioRunner
    from e2e.scenario import munich_isac_scenario

    fake_sims_dir = tmp_path / "sionna_sims"
    fake_sims_dir.mkdir()
    monkeypatch.setattr(m, "default_out_path",
                        lambda name: str(fake_sims_dir / f"{name}.pkl"))

    real_path = m.default_out_path(m.REAL_SCENARIO_NAME)
    sc = munich_isac_scenario()
    sc.num_frames = N_FRAMES
    sc.frequency.num_freqs = N_FREQS
    ScenarioRunner(sc, dry_run=True, seed=1).run(out_path=real_path, verbose=False)

    result = m.main(cache_path=None, fig_dir=str(tmp_path / "figures"),
                    num_frames=N_FRAMES, num_freqs=N_FREQS, verbose=False)

    assert result["cache_path"] == real_path
    assert result["data_source"] == "REAL Sionna-generated"


def test_main_exits_cleanly_as_a_module(tmp_path):
    """Mirrors ``python -m e2e.main.main_isac_multilink`` (must exit 0) without
    touching the real repo tree: same call the ``if __name__ == '__main__'``
    guard makes, just with I/O redirected."""
    result = main(cache_path=str(tmp_path / "cache.pkl"),
                 fig_dir=str(tmp_path / "figures"),
                 num_frames=N_FRAMES, num_freqs=N_FREQS, verbose=True)
    assert result is not None
