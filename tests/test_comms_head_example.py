"""Fast smoke test for the swappable-heads example (e2e.main.main_comms_head).

Drives `main()`'s core logic directly (no subprocess) with a tiny synthetic
config -- fast, deterministic, no display, no reliance on munich.pkl.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import matplotlib
matplotlib.use("Agg")            # no display, matches the example's own convention

from e2e.main.main_comms_head import main, MODES


def test_main_returns_finite_metrics_for_all_modes():
    results = main(n_frames=2, show=False, force_synthetic=True, n_freqs=64,
                   n_symbols=4, seed=0)
    assert set(results.keys()) == set(MODES)
    for mode in MODES:
        r = results[mode]
        assert r["ber"].shape == (2,)
        assert r["evm"].shape == (2,)
        assert np.all(np.isfinite(r["ber"]))
        assert np.all(np.isfinite(r["evm"]))
        assert np.all((r["ber"] >= 0.0) & (r["ber"] <= 1.0))
    # element0 never reports an array gain (no spatial combining); mrc/subspace do.
    assert results["element0"]["gain_db"] is None
    assert results["mrc"]["gain_db"] is not None
    assert results["subspace"]["gain_db"] is not None


def test_mrc_ber_not_worse_than_element0():
    """MRC is the SNR-maximizing combiner (with independent per-element noise
    injected before combining, per beamforming.combine's docstring), so its mean
    BER should never exceed the single-tap element0 baseline."""
    results = main(n_frames=2, show=False, force_synthetic=True, n_freqs=64,
                   n_symbols=4, seed=0)
    assert results["mrc"]["ber"].mean() <= results["element0"]["ber"].mean()


def test_main_does_not_touch_disk_when_show_is_false(tmp_path, monkeypatch):
    """show=False (the test default) must not write any figures -- keeps the smoke
    test fast and free of filesystem side effects."""
    import e2e.main.main_comms_head as mch
    monkeypatch.setattr(mch, "FIG_DIR", str(tmp_path))
    main(n_frames=2, show=False, force_synthetic=True, n_freqs=64, n_symbols=4)
    assert list(tmp_path.iterdir()) == []
