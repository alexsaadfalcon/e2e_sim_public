"""Tests for the classical (no-learning) CFAR baseline detector."""

import math

import pytest
import torch

from e2e.ml.baseline import (
    CFAR_MAX_DB,
    CFAR_MIN_DB,
    cfar_objectness,
    classical_detection_map,
    range_azimuth_power,
    resolution_report,
)
from e2e.ml.labels import LabelGrid
from e2e.ml.metrics import MatchCriterion
from e2e.ml.radar_config import PRESETS


# --------------------------------------------------------------------------------
# resolution_report -- the harness-answerability check
# --------------------------------------------------------------------------------
def test_resolution_report_flags_the_ti_preset_as_unanswerable():
    """3 TX x 4 RX resolves 0.167 in sin(az); the default tolerance is 0.06. The metric
    asks for ~2.8x finer azimuth accuracy than the array can deliver (measured 2026-08-10,
    the reason every AP in the campaign was capped)."""
    cfg = PRESETS["ti_iwr1443"]
    grid = LabelGrid.for_config(cfg)
    r = resolution_report(cfg, grid)

    assert r["n_virtual"] == 12
    assert r["rayleigh_sin_az"] == pytest.approx(2.0 / 12)
    assert r["cells_per_beamwidth"] == pytest.approx(16.0)
    assert r["tolerance_over_resolution"] < 1.0
    assert r["answerable"] is False


def test_resolution_report_accepts_the_radial_like_preset():
    """12 TX x 16 RX = 192 virtual elements is what the 192-bin grid and the 0.06
    tolerance were designed around upstream: one grid cell per virtual element, and a
    tolerance of ~6 resolution cells."""
    cfg = PRESETS["radial_like"]
    grid = LabelGrid.for_config(cfg)
    r = resolution_report(cfg, grid)

    assert r["n_virtual"] == 192
    assert r["cells_per_beamwidth"] == pytest.approx(1.0)
    assert r["tolerance_over_resolution"] > 1.0
    assert r["answerable"] is True


def test_resolution_report_honours_a_custom_criterion():
    cfg = PRESETS["ti_iwr1443"]
    grid = LabelGrid.for_config(cfg)
    widened = MatchCriterion(max_sin_az_err=2.0 / cfg.n_virtual)
    assert resolution_report(cfg, grid, widened)["answerable"] is True


# --------------------------------------------------------------------------------
# CA-CFAR
# --------------------------------------------------------------------------------
def test_cfar_objectness_isolates_a_point_target(torch_device):
    power = torch.full((48, 48), 1.0, device=torch_device)
    power[20, 30] = 1e4                            # 40 dB over the noise floor

    obj = cfar_objectness(power)

    assert obj.shape == power.shape
    assert float(obj.min()) >= 0.0 and float(obj.max()) <= 1.0
    assert float(obj[20, 30]) == pytest.approx(1.0)     # saturates the [0,1] map
    # A cell well outside the target's guard+training window sees flat noise -> ~0 dB.
    assert float(obj[5, 5]) < 0.05


def test_cfar_objectness_is_scale_invariant(torch_device):
    """CFAR compares a cell to its own neighbourhood, so multiplying the whole map by a
    constant (a gain change anywhere upstream) must not move the objectness."""
    power = torch.rand((40, 40), device=torch_device) + 0.1
    power[15, 15] = 500.0

    a = cfar_objectness(power)
    b = cfar_objectness(power * 137.0)
    assert torch.allclose(a, b, atol=1e-5)


def test_cfar_objectness_flat_map_is_near_zero(torch_device):
    """Every cell equals its local noise estimate -> 0 dB ratio -> objectness 0, so a
    featureless frame yields no detections at any threshold."""
    obj = cfar_objectness(torch.full((32, 32), 7.0, device=torch_device))
    assert float(obj.max()) == pytest.approx((0.0 - CFAR_MIN_DB) / (CFAR_MAX_DB - CFAR_MIN_DB),
                                             abs=1e-5)


def test_cfar_objectness_edge_cells_are_not_padding_biased(torch_device):
    """`count_include_pad=False`: a corner cell averages only real neighbours. With a flat
    map its ratio must still be 0 dB, not inflated by zero padding."""
    obj = cfar_objectness(torch.full((30, 30), 3.0, device=torch_device))
    assert float(obj[0, 0]) == pytest.approx(float(obj[15, 15]), abs=1e-5)


# --------------------------------------------------------------------------------
# End-to-end on synthesized ADC
# --------------------------------------------------------------------------------
def test_classical_map_localizes_a_synthesized_target_in_range(torch_device):
    """A single strong scatterer at a known range must produce the map's peak in the
    correct range bin. Azimuth is deliberately not asserted -- a 12-element array cannot
    place it to grid-cell precision, which is the whole point of this module."""
    from e2e.ml.rd_synth import synthesize_adc
    from e2e.ml.scatterers import RadarPose, Scatterer

    cfg = PRESETS["ti_iwr1443"]
    grid = LabelGrid.for_config(cfg)
    range_m = 12.0
    target = Scatterer(position=(range_m, 0.0, 0.0), velocity=(0.0, 0.0, 0.0),
                       rcs_dbsm=20.0, object_class="vehicle")

    # 40 dB: this test is about the processing chain being wired to the right bin, not
    # about detection sensitivity. At 30 dB a 12-element array's own sidelobes inflate the
    # CFAR training cells enough that noise can take the peak -- a real property of the
    # baseline, covered by the corpus-level AP rather than pinned here.
    adc = synthesize_adc(cfg, [target], RadarPose(), snr_db=40.0, seed=0)
    out = classical_detection_map(cfg, adc, grid)

    assert out.shape == (3, grid.n_range, grid.n_azimuth)
    assert torch.all(out[1:] == 0.0)               # no sub-cell regression, by design
    peak_range_bin = int(torch.argmax(out[0].max(dim=1).values))
    assert abs(peak_range_bin * grid.range_bin_m - range_m) <= 2.0 * grid.range_bin_m


def test_range_azimuth_power_shape_matches_the_virtual_array(torch_device):
    """No zero-padding by default: the angle axis has exactly one bin per virtual element,
    since interpolation would place peaks between resolution cells without adding
    information."""
    from e2e.ml.rd_synth import synthesize_adc
    from e2e.ml.scatterers import RadarPose, Scatterer

    cfg = PRESETS["ti_iwr1443"]
    adc = synthesize_adc(cfg, [Scatterer(position=(10.0, 1.0, 0.0), velocity=(0.0, 0.0, 0.0),
                                          rcs_dbsm=10.0, object_class="vehicle")],
                         RadarPose(), snr_db=30.0, seed=1)
    power = range_azimuth_power(cfg, adc)

    assert power.shape[0] == cfg.n_virtual
    assert power.ndim == 2
    assert torch.all(power >= 0.0)
    assert math.isfinite(float(power.sum()))
