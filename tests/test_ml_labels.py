"""
Tests for `e2e.ml.labels` (ground-truth detection-label encode/decode).

Uses the real `Scatterer`/`RadarPose` from `e2e.ml.scatterers` (not stubs, they're a
sibling shard's dependency-free module) with the default pose: position at the origin,
boresight +x, so `array_axis` is +y and `sin_azimuth == y / range` for an in-plane (z=0)
target -- see `e2e.ml.rd_synth.array_axis`'s docstring for the convention.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from e2e.ml.labels import LabelGrid, decode_detections, encode_detection_labels, targets_in_grid
from e2e.ml.scatterers import RadarPose, Scatterer


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
    for r_dec, sin_az_dec, score in decoded:
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
    r_dec, sin_az_dec, _ = decoded[0]

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
    r, sin_az, cls = result[0]
    assert r == pytest.approx(10.0, abs=1e-6)
    assert sin_az == pytest.approx(0.1, abs=1e-6)
    assert cls == "vehicle"


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
