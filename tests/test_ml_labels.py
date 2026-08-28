"""
Tests for `e2e.ml.labels` (ground-truth detection-label encode/decode).

Uses the real `Scatterer`/`RadarPose` from `e2e.environment.scatterers` (not stubs,
they're a dependency-free core module) with the default pose: position at the origin,
boresight +x, so `array_axis` is +y and `sin_azimuth == y / range` for an in-plane (z=0)
target -- see `e2e.chain.rd_synth.array_axis`'s docstring for the convention.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from e2e.environment.scatterers import RadarPose, Scatterer
from e2e.ml.labels import (
    LabelGrid,
    cross_range_half_extent_m,
    decode_detections,
    encode_detection_labels,
    targets_in_grid,
)


def _target(r, sin_az, object_class="vehicle"):
    """A Scatterer at the given (range, sin_azimuth) w.r.t. the default RadarPose, z=0."""
    y = r * sin_az
    x = math.sqrt(max(r * r - y * y, 0.0))
    return Scatterer(position=(x, y, 0.0), velocity=(0.0, 0.0, 0.0), rcs_dbsm=0.0,
                      object_class=object_class)


class _CfgStub:
    def __init__(self, n_samples=512, max_range_m=64.0):
        self.n_samples = n_samples
        self.max_range_m = max_range_m


# --------------------------------------------------------------------------------
# LabelGrid
# --------------------------------------------------------------------------------
def test_for_config_derives_bins():
    grid = LabelGrid.for_config(_CfgStub(n_samples=512, max_range_m=64.0), range_stride=4, n_azimuth=192)
    assert grid.n_range == 128
    assert grid.n_azimuth == 192
    assert grid.max_range_m == 64.0
    assert grid.range_bin_m == pytest.approx(64.0 / 128)
    assert grid.az_bin == pytest.approx(2.0 / 192)


# --------------------------------------------------------------------------------
# encode: footprint shape
# --------------------------------------------------------------------------------
def test_interior_target_has_full_3x3_footprint(torch_device):
    grid = LabelGrid(n_range=40, n_azimuth=40, max_range_m=40.0)
    pose = RadarPose()
    sc = _target(r=20.3, sin_az=0.05)   # well interior in both axes

    label = encode_detection_labels(grid, [sc], pose)
    assert label.shape == (3, grid.n_range, grid.n_azimuth)
    assert label.dtype == torch.float32
    assert label.device.type == torch_device.type
    assert label[0].sum().item() == pytest.approx(9.0)


def test_edge_target_footprint_clipped_without_error(torch_device):
    grid = LabelGrid(n_range=40, n_azimuth=40, max_range_m=40.0)
    pose = RadarPose()
    # r just above 0 -> range-cell index 0 (corner in range); sin_az near -1 -> az-cell
    # index 0 (corner in azimuth). Footprint should clip to the 2x2 in-bounds corner.
    sc = _target(r=0.05, sin_az=-0.999)

    label = encode_detection_labels(grid, [sc], pose)
    assert label[0].sum().item() == pytest.approx(4.0)
    assert torch.all(label[0, :2, :2] >= 0.0)  # no exception, sane values


# --------------------------------------------------------------------------------
# encode/decode round trip
# --------------------------------------------------------------------------------
def test_encode_decode_round_trip_recovers_subbin_precision(torch_device):
    grid = LabelGrid(n_range=40, n_azimuth=40, max_range_m=40.0)
    pose = RadarPose()

    # 4 targets on well-separated (range_idx, az_idx) cells (spacing well beyond the 3x3
    # footprint + NMS suppression radius) with random sub-bin offsets.
    import random
    rng = random.Random(1234)
    cells = [(5, 5), (5, 30), (30, 5), (30, 30)]
    targets = []
    for (ri, ai) in cells:
        r = (ri + rng.uniform(0.1, 0.9)) * grid.range_bin_m
        sin_az = -1.0 + (ai + rng.uniform(0.1, 0.9)) * grid.az_bin
        targets.append((r, sin_az))

    scatterers = [_target(r, sin_az) for r, sin_az in targets]
    label = encode_detection_labels(grid, scatterers, pose)

    decoded = decode_detections(grid, label, threshold=0.5)
    assert len(decoded) == len(targets)

    # match each decoded detection to its nearest true target and check sub-bin accuracy
    remaining = list(targets)
    for r_dec, sin_az_dec, score, _surface in decoded:
        best = min(remaining, key=lambda t: abs(t[0] - r_dec) + abs(t[1] - sin_az_dec))
        remaining.remove(best)
        r_true, sin_az_true = best
        assert abs(r_dec - r_true) < grid.range_bin_m / 2.0
        assert abs(sin_az_dec - sin_az_true) < grid.az_bin / 2.0
        assert score == pytest.approx(1.0)


def test_decode_beats_raw_cell_quantization(torch_device):
    """The regression refinement must do better than just reporting the cell centre."""
    grid = LabelGrid(n_range=40, n_azimuth=40, max_range_m=40.0)
    pose = RadarPose()
    # deliberately offset far from its cell centre (near the cell's edge)
    r_true, sin_az_true = 10.95 * grid.range_bin_m, -1.0 + 20.95 * grid.az_bin
    sc = _target(r_true, sin_az_true)

    label = encode_detection_labels(grid, [sc], pose)
    decoded = decode_detections(grid, label, threshold=0.5)
    assert len(decoded) == 1
    r_dec, sin_az_dec, _score, _surface = decoded[0]

    raw_cell_r = (10 + 0.5) * grid.range_bin_m
    raw_cell_az = -1.0 + (20 + 0.5) * grid.az_bin
    assert abs(r_dec - r_true) < abs(raw_cell_r - r_true)
    assert abs(sin_az_dec - sin_az_true) < abs(raw_cell_az - sin_az_true)
    assert abs(r_dec - r_true) < 1e-3
    assert abs(sin_az_dec - sin_az_true) < 1e-3


# --------------------------------------------------------------------------------
# out-of-grid handling
# --------------------------------------------------------------------------------
def test_out_of_grid_target_gives_empty_map_and_is_excluded(torch_device):
    grid = LabelGrid(n_range=40, n_azimuth=40, max_range_m=40.0)
    pose = RadarPose()
    far = _target(r=1000.0, sin_az=0.0)     # way beyond max_range_m

    label = encode_detection_labels(grid, [far], pose)
    assert label[0].sum().item() == 0.0
    assert torch.all(label == 0.0)
    assert targets_in_grid(grid, [far], pose) == []


def test_targets_in_grid_filters_mixed_scene():
    grid = LabelGrid(n_range=40, n_azimuth=40, max_range_m=40.0)
    pose = RadarPose()
    near = _target(r=10.0, sin_az=0.1, object_class="vehicle")
    far = _target(r=1000.0, sin_az=0.1, object_class="pedestrian")

    result = targets_in_grid(grid, [near, far], pose)
    assert len(result) == 1
    r, sin_az, cls, surface_r, cross_half = result[0]
    assert r == pytest.approx(10.0, abs=1e-6)
    assert sin_az == pytest.approx(0.1, abs=1e-6)
    assert cls == "vehicle"
    # point scatterer (no extent): surface == centre, no cross-range footprint
    assert surface_r == pytest.approx(10.0, abs=1e-6)
    assert cross_half == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------------
# decode: multiple targets + threshold
# --------------------------------------------------------------------------------
def test_two_well_separated_targets_decode_to_two_detections(torch_device):
    grid = LabelGrid(n_range=40, n_azimuth=40, max_range_m=40.0)
    pose = RadarPose()
    a = _target(r=8.5, sin_az=-0.5)
    b = _target(r=30.5, sin_az=0.5)

    label = encode_detection_labels(grid, [a, b], pose)
    decoded = decode_detections(grid, label, threshold=0.5)
    assert len(decoded) == 2


def test_decode_threshold_filters_low_confidence_cells(torch_device):
    grid = LabelGrid(n_range=20, n_azimuth=20, max_range_m=20.0)
    label = torch.zeros((3, grid.n_range, grid.n_azimuth), dtype=torch.float32, device=torch_device)
    # two isolated single-cell "objects" at different confidences, far enough apart that
    # NMS suppression never merges them
    label[0, 3, 3] = 0.8
    label[0, 15, 15] = 0.3

    assert len(decode_detections(grid, label, threshold=0.5)) == 1
    assert len(decode_detections(grid, label, threshold=0.2)) == 2
    detections = decode_detections(grid, label, threshold=0.2)
    assert detections[0][2] >= detections[1][2]   # sorted by score, descending


# --------------------------------------------------------------------------------
# device / dtype
# --------------------------------------------------------------------------------
def test_encode_device_and_dtype(torch_device):
    grid = LabelGrid(n_range=16, n_azimuth=16, max_range_m=16.0)
    pose = RadarPose()
    label = encode_detection_labels(grid, [_target(8.0, 0.0)], pose)
    assert label.dtype == torch.float32
    assert label.device.type == torch_device.type


def test_decode_works_on_given_device(torch_device):
    grid = LabelGrid(n_range=16, n_azimuth=16, max_range_m=16.0)
    pose = RadarPose()
    label = encode_detection_labels(grid, [_target(8.0, 0.0)], pose).to(torch_device)
    decoded = decode_detections(grid, label, threshold=0.5)
    assert len(decoded) == 1


# --------------------------------------------------------------------------------
# Surface footprint / centre regression (2026-08-17 convention)
# --------------------------------------------------------------------------------
def _extended_target(r_centre, sin_az, extent_m, yaw_rad=0.0, object_class="vehicle"):
    """A Scatterer with real geometry, centred at `(r_centre, sin_az)` in the z=0 plane."""
    y = r_centre * sin_az
    x = math.sqrt(max(r_centre * r_centre - y * y, 0.0))
    return Scatterer(position=(x, y, 0.0), velocity=(0.0, 0.0, 0.0), rcs_dbsm=10.0,
                     object_class=object_class, extent_m=extent_m, yaw_rad=yaw_rad)


def test_point_target_encoding_is_unchanged_by_the_surface_convention(torch_device):
    """A scatterer with no known extent IS a point: its footprint must still sit on its
    own cell and its target tuple must still report that same range twice. This is the
    guarantee that the analytic (`rd_synth`) path is bit-for-bit unaffected."""
    grid = LabelGrid(n_range=40, n_azimuth=40, max_range_m=40.0)
    pose = RadarPose()
    r_true = 17.37
    label = encode_detection_labels(grid, [_target(r_true, 0.1)], pose)
    rows = torch.nonzero(label[0])[:, 0]
    ci = int(r_true / grid.range_bin_m)
    assert int(rows.min()) == ci - 1 and int(rows.max()) == ci + 1

    (r_centre, _sin_az, _cls, r_surface, cross_half), = targets_in_grid(
        grid, [_target(r_true, 0.1)], pose)
    assert r_surface == pytest.approx(r_centre)
    assert r_centre == pytest.approx(r_true, abs=1e-6)
    assert cross_half == pytest.approx(0.0, abs=1e-9)


def test_footprint_sits_on_the_surface_and_regression_points_at_the_centre(torch_device):
    """The whole convention in one assertion pair: objectness where the energy is, the
    regression channels still reconstructing the object's CENTRE exactly."""
    grid = LabelGrid(n_range=128, n_azimuth=192, max_range_m=38.4)   # ti_iwr1443 geometry
    pose = RadarPose()
    length = 4.4                                   # low_poly_car, end-on (yaw 0, target at +x)
    r_centre = 20.0
    sc = _extended_target(r_centre, 0.0, (length, 1.8, 1.5), yaw_rad=0.0)

    r_surface_expected = r_centre - length / 2.0
    (r_c, _sin_az, _cls, r_s, _cross_half), = targets_in_grid(grid, [sc], pose)
    assert r_c == pytest.approx(r_centre, abs=1e-6)
    assert r_s == pytest.approx(r_surface_expected, abs=1e-6)

    label = encode_detection_labels(grid, [sc], pose)
    rows = torch.nonzero(label[0])[:, 0]
    ci_surface = int(r_surface_expected / grid.range_bin_m)
    assert int(rows.min()) == ci_surface - 1 and int(rows.max()) == ci_surface + 1
    # ... and NOT on the centre cell: 2.2 m is more than a footprint away.
    assert int(r_centre / grid.range_bin_m) > int(rows.max())

    # Every footprint cell independently reconstructs the CENTRE (the RADIal per-cell
    # residual guarantee, unchanged -- only what it points at moved).
    for i, j in torch.nonzero(label[0]).tolist():
        r_cell = (i + 0.5) * grid.range_bin_m
        assert r_cell + float(label[1, i, j]) * grid.range_bin_m == pytest.approx(
            r_centre, abs=2e-3)

    decoded = decode_detections(grid, label, threshold=0.5)
    assert len(decoded) == 1
    r_dec, _sin_dec, _score, r_surface_dec = decoded[0]
    assert r_dec == pytest.approx(r_centre, abs=1e-3)              # centre, sub-bin exact
    assert abs(r_surface_dec - r_surface_expected) <= 1.5 * grid.range_bin_m


def test_broadside_and_end_on_footprints_differ(torch_device):
    """Yaw is not decoration: the same car labelled end-on and broadside puts its
    footprint in different range cells, because it really does reflect from different
    places. An axis-aligned implementation cannot produce this."""
    grid = LabelGrid(n_range=128, n_azimuth=192, max_range_m=38.4)
    pose = RadarPose()
    end_on = _extended_target(20.0, 0.0, (4.4, 1.8, 1.5), yaw_rad=0.0)
    broadside = _extended_target(20.0, 0.0, (4.4, 1.8, 1.5), yaw_rad=math.pi / 2)

    (_c1, _s1, _k1, surf_end, cross_end), = targets_in_grid(grid, [end_on], pose)
    (_c2, _s2, _k2, surf_broad, cross_broad), = targets_in_grid(grid, [broadside], pose)
    assert surf_end == pytest.approx(20.0 - 2.2, abs=1e-6)
    assert surf_broad == pytest.approx(20.0 - 0.9, abs=1e-6)
    # Same yaw-awareness for the CROSS-RANGE extent: end-on presents its half-WIDTH
    # (0.9 m) across azimuth, broadside its half-LENGTH (2.2 m) -- the two footprint
    # dimensions swap roles exactly as the range-axis surface point does above.
    assert cross_end == pytest.approx(0.9, abs=1e-6)
    assert cross_broad == pytest.approx(2.2, abs=1e-6)

    rows_end = torch.nonzero(encode_detection_labels(grid, [end_on], pose)[0])[:, 0]
    rows_broad = torch.nonzero(encode_detection_labels(grid, [broadside], pose)[0])[:, 0]
    assert int(rows_broad.min()) > int(rows_end.max())


# --------------------------------------------------------------------------------
# Cross-range half-extent (added 2026-08-27: the azimuth-axis counterpart of the
# surface-range split above, consumed by e2e.ml.metrics as a widened azimuth
# match tolerance for don't-care regions)
# --------------------------------------------------------------------------------
def test_cross_range_half_extent_zero_for_point_target():
    """No extent -> no footprint to be off-centre by."""
    pose = RadarPose()
    sc = _target(20.0, 0.1)   # built with no extent_m
    assert cross_range_half_extent_m(sc, pose) == 0.0


def test_cross_range_half_extent_matches_ellipse_support_function():
    """Oracle: independently recompute the ellipse support-function projection
    (`sqrt((a*n_x)^2 + (b*n_y)^2)`, `n` = the object's own local frame direction of
    `array_axis(pose)`) for an arbitrary (non-axis-aligned) yaw, and check the
    module's value against it -- not just the two special-cased 0/90 degree yaws the
    other tests above use."""
    pose = RadarPose()   # boresight +x -> array_axis == +y
    a, b = 2.2, 0.9       # half-length, half-width
    yaw = math.pi / 6     # 30 degrees, deliberately not a special angle
    sc = _extended_target(20.0, 0.0, (2 * a, 2 * b, 1.5), yaw_rad=yaw)

    # array_axis is world +y here; express it in the object's own (yawed) frame the
    # same way the module docstring describes.
    c, s = math.cos(yaw), math.sin(yaw)
    nx, ny = c * 0.0 + s * 1.0, -s * 0.0 + c * 1.0
    expected = math.hypot(a * nx, b * ny)

    assert cross_range_half_extent_m(sc, pose) == pytest.approx(expected, abs=1e-9)


def test_cross_range_half_extent_never_exceeds_the_half_diagonal():
    """Sanity bound: the yaw-aware projection must never exceed the conservative
    `sqrt(half_len^2 + half_width^2)` half-diagonal fallback the task brief calls out
    as an acceptable (but cruder) alternative -- it should always be tighter or equal,
    across arbitrary yaws."""
    pose = RadarPose()
    a, b = 2.2, 0.9
    half_diagonal = math.hypot(a, b)
    for yaw_deg in range(0, 181, 5):
        sc = _extended_target(20.0, 0.0, (2 * a, 2 * b, 1.5), yaw_rad=math.radians(yaw_deg))
        assert cross_range_half_extent_m(sc, pose) <= half_diagonal + 1e-9


def test_range_residual_bound_and_regression_loss_scale(torch_device):
    """The widened range-residual bound, and the loss-scale consequence of widening it.

    Residuals are still in BIN units (a unit slip to metres is the failure mode this
    catches) and span `[-1.5, 1.5 + delta/range_bin]`. Because a 15.7 m semi at
    `ti_iwr1443`'s 0.3 m output bins puts that upper end near 27 bins, the smooth-L1
    regression term now starts deep in its LINEAR regime instead of its quadratic one --
    measured here, not assumed -- and the exact-prediction case must still be zero, i.e.
    the widened target stays representable.
    """
    from e2e.ml.losses import masked_regression_loss

    grid = LabelGrid(n_range=128, n_azimuth=192, max_range_m=38.4)
    pose = RadarPose()
    r_centre = 30.0
    for length in (4.4, 15.7):
        sc = _extended_target(r_centre, 0.0, (length, 2.5, 3.0), yaw_rad=0.0)
        label = encode_detection_labels(grid, [sc], pose)
        mask = label[0]
        res = label[1][mask > 0]
        delta_bins = (length / 2.0) / grid.range_bin_m
        assert float(res.min()) >= delta_bins - 1.5 - 1e-3
        assert float(res.max()) <= delta_bins + 1.5 + 1e-3

        # Zero-prediction (an untrained head): smooth-L1 is |x| - 0.5 out here.
        zero = torch.zeros_like(label[1:])
        loss_zero = float(masked_regression_loss(zero[None], label[None, 1:], mask[None]))
        assert loss_zero == pytest.approx(delta_bins - 0.5, abs=1.6)
        # Exact prediction: still exactly zero, so the target remains learnable.
        assert float(masked_regression_loss(label[None, 1:], label[None, 1:],
                                            mask[None])) == 0.0


def test_widened_residual_does_not_change_the_regression_gradient_scale(torch_device):
    """The loss-scale check behind the widened bound.

    Smooth-L1 SATURATES its gradient at 1 per element, so pushing the range residual from
    ~1.5 bins out to ~26 grows the loss VALUE (MEASURED at `reg_weight=100`, zero-
    regression prediction, 3 targets: reg term 0.76 -> 6.6 for a car -> 17.6 for a 15.7 m
    semi) while leaving the optimization signal alone. Pinned here so a future switch to a
    plain L2 regression -- which does NOT saturate, and where a 26-bin residual would be
    ~300x the old gradient -- cannot land silently.
    """
    from e2e.ml.losses import detection_loss

    grid = LabelGrid(n_range=128, n_azimuth=192, max_range_m=38.4)
    pose = RadarPose()
    grads = {}
    for name, extent in (("point", None), ("semi", (15.7, 2.9, 3.9))):
        sc = (_target(30.0, 0.1) if extent is None
              else _extended_target(30.0, 0.1, extent))
        label = encode_detection_labels(grid, [sc], pose).cpu()
        pred = torch.zeros_like(label)
        pred[0] = 1e-3
        pred = pred[None].clone().requires_grad_(True)
        loss, _parts = detection_loss(pred, label[None])
        loss.backward()
        grads[name] = (float(pred.grad[0, 1:].abs().max()),
                       float(pred.grad[0, 1:].abs().sum()))

    assert grads["semi"][0] == pytest.approx(grads["point"][0], rel=1e-6)   # per-cell cap
    assert grads["semi"][1] <= 1.5 * grads["point"][1]                      # measured 1.25x
