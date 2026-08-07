"""End-to-end integration of the e2e.ml radar-dataset chain.

Scenario -> frame_scatterers/radar_pose -> synthesize_adc -> transforms, using the
REAL presets (not stubs): a target with known range / velocity / azimuth must land
in the analytically expected range, Doppler, and virtual-array angle bins after the
full pipeline. This is the cross-shard contract test: config, scatterer bridge,
synthesizer, and transforms all have to agree on conventions for it to pass.
"""
import dataclasses
import math

import pytest

torch = pytest.importorskip("torch")

from e2e.ml.radar_config import RADIAL_LIKE, TI_IWR1443
from e2e.ml.rd_synth import C_LIGHT, synthesize_adc
from e2e.ml.scatterers import frame_scatterers, radar_pose, vehicle
from e2e.ml.transforms import adc_to_rd, rd_to_input, tdm_deinterleave
from e2e.scenario import Node, NodeRole, Scenario


def _scene(target_pos, target_vel):
    return Scenario(
        name="ml_integration",
        nodes=[Node(name="radar", role=NodeRole.RADAR, position=(0.0, 0.0, 0.0),
                    look_at=(10.0, 0.0, 0.0))],
        objects=[vehicle("car", target_pos, velocity=target_vel)],
    )


def test_ti_tdm_scenario_hits_expected_range_doppler_angle_bins(torch_device):
    cfg = TI_IWR1443
    pos, vel = (20.0, 8.0, 0.0), (-5.0, 0.0, 0.0)
    sc = _scene(pos, vel)

    pose = radar_pose(sc, 0)
    scats = frame_scatterers(sc, 0, dt=1.0 / cfg.frame_rate_hz)
    assert len(scats) == 1 and scats[0].object_class == "vehicle"

    adc = synthesize_adc(cfg, scats, pose, snr_db=None, seed=0, device=torch_device)
    assert adc.shape == (cfg.n_rx, cfg.n_chirps, cfg.n_samples)

    # TDM: deinterleave to the virtual array, then RD-transform one TX's chirps.
    virt = tdm_deinterleave(cfg, adc)
    assert virt.shape == (cfg.n_virtual, cfg.n_chirps_per_tx, cfg.n_samples)
    sub = dataclasses.replace(cfg, n_tx=1, mimo="single", n_chirps=cfg.n_chirps_per_tx)
    rd = adc_to_rd(sub, virt)                     # [n_virtual, range, doppler]

    # Analytic expectations (same math the physics tests use, now through the
    # full scenario -> transforms chain).
    r = math.dist(pos, pose.position)
    e_los = tuple((p - q) / r for p, q in zip(pos, pose.position))
    v_r = sum(v * e for v, e in zip(vel, e_los))          # receding-positive
    sin_th = e_los[1]                                     # boresight +x -> ULA +y

    k_range = 2.0 * r * cfg.bandwidth_hz / C_LIGHT
    t_eff = cfg.n_tx * cfg.chirp_period_s                 # TDM slow-time period
    k_dopp = (cfg.n_chirps_per_tx // 2
              + 2.0 * v_r * t_eff / cfg.wavelength_m * cfg.n_chirps_per_tx)

    power = rd.abs() ** 2
    rbin = int(power.sum(dim=(0, 2)).argmax())
    assert abs(rbin - k_range) <= 1.0
    dbin = int(power.sum(dim=0)[rbin].argmax())
    assert abs(dbin - k_dopp) <= 1.0

    # Angle: FFT across the 12-element lambda/2 virtual ULA at the target bin.
    spec = torch.fft.fftshift(torch.fft.fft(rd[:, rbin, dbin]), dim=0).abs()
    k_ang = cfg.n_virtual // 2 + cfg.n_virtual * sin_th / 2.0
    assert abs(int(spec.argmax()) - k_ang) <= 1.0


def test_generate_sample_input_peak_lands_in_label_footprint(torch_device):
    """The seam test: nothing else ties the INPUT tensor's range axis to the LABEL
    grid's range axis through the actual generate_sample path. A refactor that
    changed range_stride handling, FFT bin ordering, or LabelGrid.for_config would
    silently desynchronize input from labels without this check."""
    from e2e.ml.dataset import generate_sample
    from e2e.ml.labels import LabelGrid
    from e2e.scenario import Motion, SceneObject

    cfg = TI_IWR1443
    sc = _scene((20.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    # Add background clutter close to the vehicle: it must contribute SIGNAL but
    # never LABELS (a detector rejects clutter, it does not report it).
    sc.objects.append(SceneObject(name="clutter", position=(12.0, 3.0, 0.0),
                                  motion=Motion(), object_class="scatterer",
                                  rcs_dbsm=-10.0))
    grid = LabelGrid.for_config(cfg)
    sample = generate_sample(cfg, sc, grid, snr_db=None, seed=0, device=torch_device)

    # Ground truth contains ONLY the vehicle.
    assert [t[2] for t in sample["targets"]] == ["vehicle"]

    # The input's range-power peak, downsampled by the grid's stride, must land
    # inside the label's 3x3 objectness footprint.
    x, labels = sample["input"], sample["labels"]
    stride = cfg.n_samples // grid.n_range
    peak_range_bin = int((x ** 2).sum(dim=(0, 2)).argmax()) // stride
    label_range_bins = labels[0].amax(dim=1).nonzero().flatten().tolist()
    assert peak_range_bin in label_range_bins, (
        f"input peak at label-grid range bin {peak_range_bin}, "
        f"but labels mark {label_range_bins}")


def test_radial_like_ddma_produces_fftradnet_input_contract(torch_device):
    cfg = RADIAL_LIKE
    sc = _scene((30.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    adc = synthesize_adc(cfg, frame_scatterers(sc, 0), radar_pose(sc, 0),
                         snr_db=None, seed=0, device=torch_device)
    rd = adc_to_rd(cfg, adc)
    x = rd_to_input(rd)

    # FFTRadNet-style input contract: (2*n_rx, range, doppler) float32. The vendor
    # radar uses 256 chirps; our preset deliberately uses 252 (divisible by n_tx=12,
    # so the DDMA replicas land on exact bins -- see radar_config.py) and the models
    # are parameterized over the Doppler width, so 252 is the contract here.
    assert x.shape == (2 * cfg.n_rx, 512, cfg.n_chirps) == (32, 512, 252)
    assert x.dtype == torch.float32

    # Boresight static target: range bin at 2*R*B/c regardless of MIMO scheme.
    k_range = 2.0 * 30.0 * cfg.bandwidth_hz / C_LIGHT
    power = rd.abs() ** 2
    rbin = int(power.sum(dim=(0, 2)).argmax())
    assert abs(rbin - k_range) <= 1.0

    # DDMA signature: a single target shows n_tx Doppler replicas spaced
    # n_chirps/n_tx bins apart; every strong Doppler peak must sit on that comb.
    dprof = power.sum(dim=0)[rbin]
    comb = {int((cfg.n_chirps // 2 + round(t * cfg.n_chirps / cfg.n_tx))
                % cfg.n_chirps) for t in range(cfg.n_tx)}
    strong = (dprof > 0.25 * dprof.max()).nonzero().flatten().tolist()
    assert strong, "no Doppler peaks found at the target's range bin"
    for b in strong:
        assert any(abs(b - c) <= 1 or abs(b - c) >= cfg.n_chirps - 1 for c in comb)
