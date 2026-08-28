"""Tests for the classical (no-learning) CFAR baseline detector."""

import json
import math

import numpy as np
import pytest
import torch

from e2e.ml import baseline
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
from e2e.radar_config import PRESETS


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
    from e2e.environment.scatterers import RadarPose, Scatterer
    from e2e.chain.rd_synth import synthesize_adc

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
    from e2e.environment.scatterers import RadarPose, Scatterer
    from e2e.chain.rd_synth import synthesize_adc

    cfg = PRESETS["ti_iwr1443"]
    adc = synthesize_adc(cfg, [Scatterer(position=(10.0, 1.0, 0.0), velocity=(0.0, 0.0, 0.0),
                                          rcs_dbsm=10.0, object_class="vehicle")],
                         RadarPose(), snr_db=30.0, seed=1)
    power = range_azimuth_power(cfg, adc)

    assert power.shape[0] == cfg.n_virtual
    assert power.ndim == 2
    assert torch.all(power >= 0.0)
    assert math.isfinite(float(power.sum()))


# --------------------------------------------------------------------------------
# score_manifest -- ground-truth target counting (regression: must be the
# deduplicated per-frame target list, not a footprint-cell count off the dense label
# map; see e2e.ml.labels.encode_detection_labels's 3x3-footprint convention)
# --------------------------------------------------------------------------------
def _write_tiny_baseline_corpus(tmp_path, cfg):
    """A 2-frame, 2-well-separated-targets-per-frame synthetic corpus, split entirely
    into "val" -- built by hand (not `e2e.ml.dataset.generate_dataset`) so the ADC
    payload is deliberately pre-quantized and round-trips exactly through
    `e2e.ml.storage`'s `CODEC_INT16`, guaranteeing the on-disk `adc_code_re`/
    `adc_code_im` keys `score_manifest` needs to beamform.

    4 real targets total, range bins far enough apart (40 bins, footprint half-width
    1) that every 3x3 footprint stays disjoint and unclipped by the grid boundary:
    exactly 4*9 = 36 positive label-map cells, vs. 4 deduplicated targets.
    """
    from e2e.ml import dataset as ml_dataset
    from e2e.ml import storage
    from e2e.environment.scatterers import RadarPose, Scatterer
    from e2e.ml.labels import encode_detection_labels, targets_in_grid

    grid = LabelGrid.for_config(cfg)
    pose = RadarPose()
    range_bin_idx_per_frame = [(20, 60), (30, 70)]  # far apart -> disjoint footprints

    dataset_dir = tmp_path / "tiny_baseline_corpus"
    dataset_dir.mkdir()
    rng = np.random.default_rng(0)

    sequences = []
    for i, idxs in enumerate(range_bin_idx_per_frame):
        scatterers = []
        for ri in idxs:
            r = (ri + 0.5) * grid.range_bin_m
            # sin_az == 0 (mid-grid azimuth bin, unclipped) -- azimuth is not what
            # this test is about (see the module's existing range-localization test
            # for the same "assert azimuth deliberately not" convention).
            scatterers.append(Scatterer(position=(r, 0.0, 0.0), velocity=(0.0, 0.0, 0.0),
                                        rcs_dbsm=20.0, object_class="vehicle"))
        labels = encode_detection_labels(grid, scatterers, pose, classes=("vehicle",))
        targets = targets_in_grid(grid, scatterers, pose, classes=("vehicle",))
        assert len(targets) == 2   # sanity: both scatterers landed in-grid

        # Integer-valued ADC codes round-trip byte-exact through int16 storage (see
        # e2e.ml.storage's SAFETY note) -- content is irrelevant to this test, only
        # that adc_code_re/adc_code_im land on disk.
        codes = rng.integers(-1000, 1000, size=(cfg.n_rx, cfg.n_chirps, cfg.n_samples))
        adc = codes.astype(np.complex64)

        fname = f"frame_{i:05d}.npz"
        storage.write_sample_npz(
            dataset_dir / fname, {"adc": adc, "labels": labels.cpu().numpy()},
            {"targets": targets}, payload_key="adc", full_scale=float(2 ** 15),
        )
        sequences.append([fname])

    # Every sequence into "val" (score_manifest's own default split).
    manifest_path = ml_dataset.write_manifest(
        dataset_dir, cfg, "test_tier", sequences, grid=grid, splits=(0.0, 1.0, 0.0),
    )
    return manifest_path


def test_score_manifest_targets_are_deduplicated_not_footprint_cells(tmp_path, monkeypatch):
    """Regression for the ~9x target-inflation bug: `score_manifest` must score
    against `RadarFrameDataset.targets()` (deduplicated, one entry per real object),
    not the dense label map's positive-cell count (each of a target's 3x3 footprint
    cells counted separately)."""
    from e2e.ml.dataset import RadarFrameDataset

    cfg = PRESETS["ti_iwr1443"]
    manifest_path = _write_tiny_baseline_corpus(tmp_path, cfg)

    captured = {}
    real_evaluate_dataset = baseline.evaluate_dataset

    def _spy(pred_maps, target_lists, grid, **kwargs):
        captured["target_lists"] = target_lists
        return real_evaluate_dataset(pred_maps, target_lists, grid, **kwargs)

    monkeypatch.setattr(baseline, "evaluate_dataset", _spy)

    res = baseline.score_manifest(manifest_path, split="val")
    assert res["n_frames"] == 2

    ds = RadarFrameDataset(manifest_path, split="val")
    expected = sum(len(ds.targets(i)) for i in range(len(ds)))
    assert expected == 4   # 2 targets/frame x 2 frames, deduplicated

    got = sum(len(t) for t in captured["target_lists"])
    assert got == expected

    # Pin against the removed footprint-cell counting method directly: it would have
    # thresholded each frame's dense label map instead.
    footprint_cell_count = 0
    files = json.loads(manifest_path.read_text())["files"]["val"]
    for fn in files:
        with np.load(manifest_path.parent / fn) as z:
            footprint_cell_count += int((z["labels"][0] > 0.5).sum())
    assert footprint_cell_count == 36
    assert got != footprint_cell_count


# ------------------------------------------------------------------------------------
# Doppler reduction -- the axis added 2026-08-19 after an RF consultant flagged that
# collapsing Doppler with `max` BEFORE CFAR changes the noise statistics the threshold
# depends on.
#
# The batched-vs-looped tests are the load-bearing ones: the per-Doppler path exists only
# because `cfar_objectness` and `_to_grid` grew a batch axis, and a batch axis that
# disagrees with the loop it replaces would corrupt every number quietly.
# ------------------------------------------------------------------------------------
def test_batched_cfar_objectness_agrees_with_looping_the_2d_version():
    """The batch axis must be exactly a loop, not approximately one."""
    torch.manual_seed(0)
    cube = torch.rand(5, 24, 32) + 0.05
    batched = baseline.cfar_objectness(cube)
    looped = torch.stack([baseline.cfar_objectness(cube[d]) for d in range(cube.shape[0])])
    assert batched.shape == cube.shape
    assert torch.equal(batched, looped)


def test_unbatched_cfar_objectness_is_unchanged_by_the_batch_axis():
    """Regression guard: the 2-D entry point must still return 2-D, bit for bit."""
    torch.manual_seed(1)
    power = torch.rand(24, 32) + 0.05
    out = baseline.cfar_objectness(power)
    assert out.shape == power.shape
    assert torch.equal(out, baseline.cfar_objectness(power[None])[0])


def test_batched_to_grid_agrees_with_looping():
    torch.manual_seed(2)
    cfg = PRESETS["ti_iwr1443"]
    grid = LabelGrid.for_config(cfg)
    cube = torch.rand(4, 16, 64)
    batched = baseline._to_grid(cube, cfg, grid)
    looped = torch.stack([baseline._to_grid(cube[d], cfg, grid)
                          for d in range(cube.shape[0])])
    assert torch.equal(batched, looped)


def test_keep_doppler_returns_the_uncollapsed_cube_whose_max_is_the_collapsed_one():
    """`keep_doppler` must be the SAME computation, just stopped one step earlier."""
    torch.manual_seed(3)
    cfg = PRESETS["ti_iwr1443"]
    adc = torch.complex(torch.randn(cfg.n_rx, cfg.n_chirps, cfg.n_samples),
                        torch.randn(cfg.n_rx, cfg.n_chirps, cfg.n_samples))
    flat = baseline.range_azimuth_power(cfg, adc)
    cube = baseline.range_azimuth_power(cfg, adc, keep_doppler=True)
    assert cube.dim() == 3 and cube.shape[:2] == flat.shape
    assert torch.equal(cube.max(dim=2).values, flat)


def test_default_doppler_reduce_is_sum():
    """v1.1 default flip (release-plan A4): `sum` measured better than `max` at
    matched recall on both corpora tried (F48, F50), owner-approved. The default must
    be exactly DOPPLER_SUM -- and must DIFFER from the pre-v1.1 `max` on generic data,
    so a silent revert cannot pass."""
    torch.manual_seed(4)
    cfg = PRESETS["ti_iwr1443"]
    grid = LabelGrid.for_config(cfg)
    adc = torch.complex(torch.randn(cfg.n_rx, cfg.n_chirps, cfg.n_samples),
                        torch.randn(cfg.n_rx, cfg.n_chirps, cfg.n_samples))
    default = baseline.classical_detection_map(cfg, adc, grid)
    explicit_sum = baseline.classical_detection_map(cfg, adc, grid,
                                                    doppler_reduce=baseline.DOPPLER_SUM)
    explicit_max = baseline.classical_detection_map(cfg, adc, grid,
                                                    doppler_reduce=baseline.DOPPLER_MAX)
    assert torch.equal(default, explicit_sum)
    assert not torch.equal(default, explicit_max)


@pytest.mark.parametrize("reduce", ["max", "sum", "cfar_first"])
def test_every_doppler_reduction_produces_the_detector_output_format(reduce):
    torch.manual_seed(5)
    cfg = PRESETS["ti_iwr1443"]
    grid = LabelGrid.for_config(cfg)
    adc = torch.complex(torch.randn(cfg.n_rx, cfg.n_chirps, cfg.n_samples),
                        torch.randn(cfg.n_rx, cfg.n_chirps, cfg.n_samples))
    out = baseline.classical_detection_map(cfg, adc, grid, doppler_reduce=reduce)
    assert out.shape == (3, grid.n_range, grid.n_azimuth)
    assert torch.isfinite(out).all()
    assert float(out[0].min()) >= 0.0 and float(out[0].max()) <= 1.0


def test_unknown_doppler_reduction_raises_rather_than_silently_defaulting():
    cfg = PRESETS["ti_iwr1443"]
    grid = LabelGrid.for_config(cfg)
    adc = torch.zeros(cfg.n_rx, cfg.n_chirps, cfg.n_samples, dtype=torch.complex64)
    with pytest.raises(ValueError, match="doppler_reduce"):
        baseline.classical_detection_map(cfg, adc, grid, doppler_reduce="mean")


def test_cfar_first_is_worse_calibrated_than_max_on_pure_noise():
    """Pins a MEASURED own-goal so it cannot be quietly reintroduced as an improvement.

    "CFAR each Doppler slice, then take the max of the objectness" sounds like the
    physically correct fix -- detect first, collapse second -- and it is not: the max over
    K slices is an UNCORRECTED multiple-hypothesis test, so with K tries per cell
    something crosses threshold almost every time. MEASURED on benchmark_v1 (K=64), false
    alarms per frame on noise-only input barely move with threshold: 1427 at 0.05 and
    1417 at 0.3, against 822 -> 0.0 for `max`.

    Here the effect is reproduced on a small config, where the assertion that matters is
    the ORDERING (cfar_first strictly worse), not the absolute counts.
    """
    torch.manual_seed(6)
    cfg = PRESETS["ti_iwr1443"]
    grid = LabelGrid.for_config(cfg)
    # Pure complex Gaussian: every detection below is a false alarm by construction.
    adc = torch.complex(torch.randn(cfg.n_rx, cfg.n_chirps, cfg.n_samples),
                        torch.randn(cfg.n_rx, cfg.n_chirps, cfg.n_samples)) / (2 ** 0.5)

    def fa(reduce, thr):
        obj = baseline.classical_detection_map(cfg, adc, grid, doppler_reduce=reduce)[0]
        return int((obj > thr).sum())

    assert fa("cfar_first", 0.3) > fa("max", 0.3)


@pytest.mark.parametrize("offset_bins", [0, 1, 2, 3])
def test_classical_map_localizes_a_point_target_at_every_fine_bin_offset(offset_bins, torch_device):
    """Regression for the _to_grid decimation defect (found 2026-08-23 when the A4
    default flip exposed it): nearest-neighbour range decimation sampled only fine
    bins stride*i + stride//2, so a point target in any OTHER fine bin of its cell
    was invisible to the detector. Peak-pooling must localize it at every offset."""
    from e2e.environment.scatterers import RadarPose, Scatterer
    from e2e.chain.rd_synth import synthesize_adc

    cfg = PRESETS["ti_iwr1443"]
    grid = LabelGrid.for_config(cfg)
    range_m = (160 + offset_bins) * float(cfg.range_resolution_m)
    target = Scatterer(position=(range_m, 0.0, 0.0), velocity=(0.0, 0.0, 0.0),
                       rcs_dbsm=20.0, object_class="vehicle")
    adc = synthesize_adc(cfg, [target], RadarPose(), snr_db=40.0, seed=0)
    out = classical_detection_map(cfg, adc, grid)
    peak_range_bin = int(torch.argmax(out[0].max(dim=1).values))
    assert abs(peak_range_bin * grid.range_bin_m - range_m) <= 2.0 * grid.range_bin_m




def _peak_azimuth_sidelobe_db(cfg, adc, **kw):
    """Peak azimuth sidelobe of the strongest range bin, dB below the mainlobe.

    Zero-pads the ANGLE axis only to sample the underlying continuous pattern -- this
    measures a pattern, it does not detect anything, so interpolation here adds no
    resolution claim (contrast `_to_grid`, where it would).
    """
    import numpy as np
    from e2e.ml.baseline import range_azimuth_power

    p = range_azimuth_power(cfg, adc, n_angle_fft=1024, **kw).cpu().numpy()
    ri = int(np.unravel_index(p.argmax(), p.shape)[1])
    cut = p[:, ri] / p[:, ri].max()
    db = 10.0 * np.log10(cut + 1e-300)
    c = int(np.argmax(db))
    k = c
    while k + 1 < len(db) and db[k + 1] < db[k]:
        k += 1                                  # walk down out of the mainlobe, right
    j = c
    while j - 1 >= 0 and db[j - 1] < db[j]:
        j -= 1                                  # and left
    return float(max(db[k:].max(), db[:j + 1].max()))


class _PointTarget:
    """A true point scatterer (no extent), so the response IS the array's own."""

    def __init__(self, range_m, sin_az, speed_mps, rcs_dbsm=10.0):
        y = range_m * sin_az
        self.position = (float((range_m ** 2 - y ** 2) ** 0.5), float(y), 0.0)
        self.velocity = (-float(speed_mps), 0.0, 0.0)
        self.rcs_dbsm = float(rcs_dbsm)
        self.extent_m = None
        self.object_class = "vehicle"


def test_tdm_doppler_compensation_restores_the_aperture_for_moving_targets(torch_device):
    """TDM fires its transmitters in sequence, so a moving target's phase advances
    between one TX's chirps and the next. Uncorrected, that corrupts the virtual aperture
    and raises azimuth sidelobes in proportion to speed. Measured on one point target
    through `range_azimuth_power`: -31.5 dB at rest (the ideal Hann level for a
    64-element ULA) degrading to -18.7 dB at 8 m/s -- and the corpus's spurious CFAR
    detections sit at a median -17.2 dB below their own range's peak.

    `tdm_deinterleave` declines to correct this because the per-target Doppler "is not
    known before detection". That is true BEFORE the Doppler FFT; `range_azimuth_power`
    applies it after, where every Doppler bin's f_D is known exactly and the correction
    is well posed per bin.

    Pins the PROPERTY rather than the exact decibels: uncompensated must degrade with
    speed, compensated must stay near the stationary level.
    """
    from e2e.chain import rd_synth

    cfg = PRESETS["benchmark_v1"]
    at_rest = _peak_azimuth_sidelobe_db(
        cfg, rd_synth.synthesize_adc(cfg, [_PointTarget(20.0, 0.0, 0.0)], snr_db=None,
                                     seed=0, device=torch_device, random_phase=False))

    fast = rd_synth.synthesize_adc(cfg, [_PointTarget(20.0, 0.0, 8.0)], snr_db=None,
                                   seed=0, device=torch_device, random_phase=False)
    uncompensated = _peak_azimuth_sidelobe_db(cfg, fast)
    compensated = _peak_azimuth_sidelobe_db(cfg, fast, tdm_doppler_comp=True)

    assert at_rest < -30.0, f"a STATIONARY target should already be clean, got {at_rest:.1f} dB"
    assert uncompensated > at_rest + 8.0, (
        f"expected motion to wreck the aperture: at rest {at_rest:.1f} dB, "
        f"moving {uncompensated:.1f} dB -- if this now fails the coupling was fixed "
        f"upstream and this test should move with it")
    assert compensated < at_rest + 1.0, (
        f"compensation should restore the stationary sidelobe level: at rest "
        f"{at_rest:.1f} dB, compensated {compensated:.1f} dB")


def test_tdm_doppler_compensation_refuses_a_non_tdm_config():
    """It de-rotates by TX index; on a non-TDM config that is meaningless, so it must
    raise rather than silently return a differently-wrong map."""
    from e2e.chain import rd_synth
    from e2e.ml.baseline import range_azimuth_power

    cfg = PRESETS["ddma_wide_v1"]
    adc = rd_synth.synthesize_adc(cfg, [_PointTarget(20.0, 0.0, 3.0)], snr_db=None, seed=0)
    with pytest.raises(ValueError, match="tdm_doppler_comp"):
        range_azimuth_power(cfg, adc, tdm_doppler_comp=True)
