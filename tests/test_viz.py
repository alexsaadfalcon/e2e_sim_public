"""Tests for `e2e.viz` -- the shared RA-map dB-normalize + imshow-orientation helpers
consolidated out of `e2e.ml.render_scene` and five `e2e/main/main_*.py` example
scripts (see `e2e.viz`'s module docstring)."""
import os

import numpy as np
import pytest

from e2e import viz

torch = pytest.importorskip("torch")


# --------------------------------------------------------------------------------
# fig_dir
# --------------------------------------------------------------------------------
def test_fig_dir_creates_and_returns_figures_subdir(tmp_path):
    module_file = str(tmp_path / "some_module.py")
    d = viz.fig_dir(module_file)

    assert d == os.path.join(str(tmp_path), "figures")
    assert os.path.isdir(d)


def test_fig_dir_is_idempotent(tmp_path):
    module_file = str(tmp_path / "some_module.py")
    d1 = viz.fig_dir(module_file)
    d2 = viz.fig_dir(module_file)  # must not raise on an already-existing dir

    assert d1 == d2
    assert os.path.isdir(d2)


# --------------------------------------------------------------------------------
# to_db
# --------------------------------------------------------------------------------
@pytest.mark.parametrize("as_torch", [False, True])
def test_to_db_peak_is_zero_db(as_torch):
    power = np.array([0.01, 1.0, 0.25], dtype=np.float64)
    if as_torch:
        power = torch.as_tensor(power, dtype=torch.float32)

    db = viz.to_db(power, floor_db=None)

    peak_db = float(db[1])
    assert peak_db == pytest.approx(0.0, abs=1e-5)
    # -100x below peak -> -20 dB, well above the eps floor so unaffected by it.
    assert float(db[2]) == pytest.approx(10 * np.log10(0.25), abs=1e-4)


@pytest.mark.parametrize("as_torch", [False, True])
def test_to_db_floor_clamps_deep_values(as_torch):
    power = np.array([1.0, 0.0, 1e-9], dtype=np.float64)
    if as_torch:
        power = torch.as_tensor(power, dtype=torch.float32)

    db = viz.to_db(power, floor_db=-40.0)

    assert float(db[0]) == pytest.approx(0.0, abs=1e-5)
    assert float(db[1]) == pytest.approx(-40.0, abs=1e-5)   # exact zero -> floored
    assert float(db[2]) == pytest.approx(-40.0, abs=1e-5)   # -90 dB -> floored to -40


@pytest.mark.parametrize("as_torch", [False, True])
def test_to_db_floor_none_leaves_deep_values_unclamped(as_torch):
    power = np.array([1.0, 1e-9], dtype=np.float64)
    if as_torch:
        power = torch.as_tensor(power, dtype=torch.float32)

    db = viz.to_db(power, floor_db=None)

    assert float(db[1]) == pytest.approx(-90.0, abs=1e-2)


def test_to_db_all_zero_power_is_finite_not_nan():
    power = np.zeros(5, dtype=np.float64)
    db = viz.to_db(power, floor_db=-40.0)
    assert np.all(np.isfinite(db))
    assert np.all(db == pytest.approx(-40.0))


def test_to_db_returns_same_container_type():
    power_np = np.array([1.0, 0.5], dtype=np.float64)
    assert isinstance(viz.to_db(power_np), np.ndarray)

    power_t = torch.tensor([1.0, 0.5], dtype=torch.float32)
    assert isinstance(viz.to_db(power_t), torch.Tensor)


def test_to_db_works_on_cuda_tensor_if_available():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    power = torch.rand(4, 4, device="cuda")
    db = viz.to_db(power, floor_db=-40.0)
    assert db.device.type == "cuda"
    assert torch.isfinite(db).all()


# --------------------------------------------------------------------------------
# imshow_ra -- orientation contract (hot-cell probe, mirrors
# tests/test_ml_render.py::test_draw_radar_view_imshow_orientation_matches_extent)
# --------------------------------------------------------------------------------
def test_imshow_ra_orientation_matches_extent():
    """`ra` is `[n_angle, n_range]`, but with azimuth on x / range on y the array
    fed to `imshow` must be `[n_range, n_angle]` -- regression for the transpose bug
    reintroduced independently at least three times in this project's history."""
    import matplotlib.pyplot as plt

    n_angle, n_range = 8, 5
    angle_idx, range_idx = 2, 4  # deliberately distinct so a transpose is detectable
    ra = torch.full((n_angle, n_range), -40.0)
    ra[angle_idx, range_idx] = 0.0
    sin_az_axis = np.linspace(-1.0, 1.0, n_angle)
    range_axis_m = np.arange(n_range, dtype=float) * 2.0

    fig, ax = plt.subplots()
    try:
        image = viz.imshow_ra(ax, ra, sin_az_axis, range_axis_m)
        arr = image.get_array()

        assert arr.shape == (n_range, n_angle)  # [row=range, col=angle]
        hot_row, hot_col = np.unravel_index(np.argmax(arr), arr.shape)
        assert (hot_row, hot_col) == (range_idx, angle_idx)

        extent = image.get_extent()
        assert extent == pytest.approx(
            [sin_az_axis[0], sin_az_axis[-1], range_axis_m[0], range_axis_m[-1]])
    finally:
        plt.close(fig)


def test_imshow_ra_accepts_numpy_input():
    import matplotlib.pyplot as plt

    ra = np.zeros((4, 3))
    ra[1, 2] = 1.0
    sin_az_axis = np.linspace(-1.0, 1.0, 4)
    range_axis_m = np.arange(3, dtype=float)

    fig, ax = plt.subplots()
    try:
        image = viz.imshow_ra(ax, ra, sin_az_axis, range_axis_m)
        arr = image.get_array()
        assert arr.shape == (3, 4)
        hot_row, hot_col = np.unravel_index(np.argmax(arr), arr.shape)
        assert (hot_row, hot_col) == (2, 1)
    finally:
        plt.close(fig)


def test_imshow_ra_without_axes_falls_back_to_pixel_extent():
    """No `sin_az_axis`/`range_axis_m` (e.g. a bin-indexed az/el map) -> no `extent`
    kwarg passed, matching a bare `ax.imshow(arr, origin="lower")` call (imshow_ra's
    own default `origin`)."""
    import matplotlib.pyplot as plt

    ra = np.zeros((4, 3))
    fig, ax = plt.subplots()
    try:
        image = viz.imshow_ra(ax, ra)
        plain = ax.imshow(np.zeros((3, 4)), origin="lower")
        assert image.get_extent() == plain.get_extent()
    finally:
        plt.close(fig)


def test_imshow_ra_passes_through_imshow_kwargs():
    import matplotlib.pyplot as plt

    ra = np.ones((2, 2))
    fig, ax = plt.subplots()
    try:
        image = viz.imshow_ra(ax, ra, cmap="viridis", vmin=-30, vmax=0)
        assert image.get_cmap().name == "viridis"
        assert image.get_clim() == (-30, 0)
    finally:
        plt.close(fig)


def test_imshow_ra_works_on_cuda_tensor_if_available():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    import matplotlib.pyplot as plt

    ra = torch.zeros(4, 3, device="cuda")
    fig, ax = plt.subplots()
    try:
        image = viz.imshow_ra(ax, ra)
        assert image.get_array().shape == (3, 4)
    finally:
        plt.close(fig)
