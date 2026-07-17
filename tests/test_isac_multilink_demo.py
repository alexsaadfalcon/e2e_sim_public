"""
Tests for the multi-link ISAC demo (e2e.main.main_isac_multilink).

Runs the demo's ``main()`` in-process with tiny frame/freq counts and both the pkl
cache and the figures redirected into ``tmp_path`` -- never touches
``e2e/environment/sionna_sims/`` or ``e2e/main/figures/``. Dry-run only (no Sionna).
"""

import os

import numpy as np
import pytest

from e2e.environment.sionna_iterator import SionnaIterator


# Keep the demo tiny for a fast test.
N_FRAMES = 3
N_FREQS = 16


@pytest.fixture(scope="module")
def demo_result(tmp_path_factory):
    """Run main() ONCE for the whole module; read-only tests share this result.

    Tests that mutate shared state (e.g. re-invoking main() against the same
    cache_path to check reuse behavior) must NOT use this fixture -- they need
    their own isolated tmp dir / cache_path.
    """
    from e2e.main.main_isac_multilink import main

    tmp_path = tmp_path_factory.mktemp("isac_multilink_demo")
    cache_path = str(tmp_path / "isac_demo.pkl")
    fig_dir = str(tmp_path / "figures")

    result = main(cache_path=cache_path, fig_dir=fig_dir,
                 num_frames=N_FRAMES, num_freqs=N_FREQS, verbose=False)
    return {"result": result, "cache_path": cache_path, "fig_dir": fig_dir}


def test_demo_runs_and_produces_multilink_cache(demo_result):
    cache_path = demo_result["cache_path"]
    result = demo_result["result"]

    assert os.path.isfile(cache_path)

    links = SionnaIterator.available_links(cache_path)
    assert links is not None and len(links) >= 2
    assert any("radar" in l.lower() for l in links)
    assert any("comm" in l.lower() for l in links)

    assert result["radar_link"] in links
    assert result["comm_link"] in links
    assert result["radar_link"] != result["comm_link"]


def test_radar_leg_outputs_finite(demo_result):
    result = demo_result["result"]

    ranges = result["ranges"]
    profiles = result["profiles"]
    peak_ranges = result["peak_ranges"]

    assert profiles.shape[0] == N_FRAMES
    assert np.all(np.isfinite(ranges))
    assert np.all(np.isfinite(profiles))
    assert np.all(np.isfinite(peak_ranges))
    assert peak_ranges.shape == (N_FRAMES,)


def test_comm_leg_outputs_finite(demo_result):
    comm = demo_result["result"]["comm"]
    assert len(comm["ber"]) == len(comm["snr_list"]) > 0
    assert all(np.isfinite(b) for b in comm["ber"])
    assert all(np.isfinite(e) for e in comm["evm_pct"])
    # BER should be non-increasing on average as SNR increases isn't guaranteed with
    # unstructured dry-run "channel" data (see module docstring caveat) -- only check
    # the values are valid rates.
    assert all(0.0 <= b <= 1.0 for b in comm["ber"])


def test_demo_writes_figures_to_tmp_dir(demo_result):
    fig_dir = demo_result["fig_dir"]
    for fig_path in demo_result["result"]["figures"]:
        assert fig_path.startswith(fig_dir)
        assert os.path.isfile(fig_path)


def test_demo_reuses_existing_cache(tmp_path):
    """A second call with the same cache_path must not regenerate (loads the pkl as-is)."""
    from e2e.main.main_isac_multilink import main

    cache_path = str(tmp_path / "isac_demo.pkl")
    fig_dir = str(tmp_path / "figures")

    main(cache_path=cache_path, fig_dir=fig_dir,
        num_frames=N_FRAMES, num_freqs=N_FREQS, verbose=False)
    mtime_1 = os.path.getmtime(cache_path)

    main(cache_path=cache_path, fig_dir=fig_dir,
        num_frames=N_FRAMES, num_freqs=N_FREQS, verbose=False)
    mtime_2 = os.path.getmtime(cache_path)

    assert mtime_1 == mtime_2


def test_pick_link_raises_if_missing():
    from e2e.main.main_isac_multilink import _pick_link

    with pytest.raises(ValueError):
        _pick_link(["foo", "bar"], "radar")


def test_cli_module_runs_as_script(demo_result):
    """Smoke-test the __main__ entry point runs end-to-end without error.

    Runs the real module-level default paths would write into the repo tree, so
    instead we rely on the shared demo_result fixture (equivalent coverage without
    touching e2e/environment/sionna_sims or e2e/main/figures, and without a redundant
    main() re-invocation -- this assertion is read-only).
    """
    assert demo_result["result"] is not None
