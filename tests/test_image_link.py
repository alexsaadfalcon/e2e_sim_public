"""Fast smoke test for the image-through-the-link demo (e2e.main.main_image_link).

Uses a tiny synthetic 32x32 gradient image (NOT scipy.datasets.ascent() -- no
network/pooch download in CI) through the Sionna-free synthetic channel
fallback (`force_synthetic=True`), mirroring the redirect-FIG_DIR pattern in
`tests/test_comms_head_example.py`.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import matplotlib
matplotlib.use("Agg")            # no display, matches the example's own convention

import e2e.main.main_image_link as mil


def _tiny_gradient_image(size=32):
    """A deterministic 32x32 uint8 gradient -- cheap, no download, has enough
    bit-level variety (both low- and high-order bits vary) to exercise BER."""
    row = np.linspace(0, 255, size, dtype=np.uint8)
    return np.tile(row, (size, 1)).astype(np.uint8)


def test_main_runs_and_reconstructs_image_shape():
    img = _tiny_gradient_image()
    out = mil.main(image=img, show=False, force_synthetic=True, seed=0)
    assert out["source"] == "synthetic"
    assert set(out["results"].keys()) == {p["name"] for p in mil.OPERATING_POINTS}
    for point in mil.OPERATING_POINTS:
        r = out["results"][point["name"]]
        assert np.isfinite(r["ber"])
        assert 0.0 <= r["ber"] <= 1.0
        assert r["image"].shape == img.shape
        assert r["image"].dtype == np.uint8


def test_ber_monotonically_increases_as_snr_drops():
    """The three noise-only operating points (clean/mid/low) are ordered by
    decreasing SNR; BER must not decrease as SNR drops."""
    img = _tiny_gradient_image()
    out = mil.main(image=img, show=False, force_synthetic=True, seed=0)
    snr_points = [p for p in mil.OPERATING_POINTS if not p.get("pa", False)]
    snr_points_sorted = sorted(snr_points, key=lambda p: -p["snr_db"])
    bers = [out["results"][p["name"]]["ber"] for p in snr_points_sorted]
    assert all(np.isfinite(b) for b in bers)
    assert all(b2 >= b1 - 1e-9 for b1, b2 in zip(bers, bers[1:]))


def test_pa_operating_point_composes_and_is_finite():
    """The TX-PA-nonideality point runs through the same oversample/de-alias
    path and returns a finite in-range BER."""
    img = _tiny_gradient_image()
    out = mil.main(image=img, show=False, force_synthetic=True, seed=0)
    r = out["results"]["pa"]
    assert np.isfinite(r["ber"])
    assert 0.0 <= r["ber"] <= 1.0


def test_main_writes_figure_when_show_is_true(tmp_path, monkeypatch):
    monkeypatch.setattr(mil, "FIG_DIR", str(tmp_path))
    img = _tiny_gradient_image()
    mil.main(image=img, show=True, force_synthetic=True, seed=0)
    fig_path = tmp_path / "image_link.png"
    assert fig_path.exists()
    assert fig_path.stat().st_size > 0


def test_main_does_not_touch_disk_when_show_is_false(tmp_path, monkeypatch):
    monkeypatch.setattr(mil, "FIG_DIR", str(tmp_path))
    img = _tiny_gradient_image()
    mil.main(image=img, show=False, force_synthetic=True, seed=0)
    assert list(tmp_path.iterdir()) == []
