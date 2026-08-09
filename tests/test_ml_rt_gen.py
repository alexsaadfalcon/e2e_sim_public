"""Ray-traced ADC generation (`e2e.ml.rt_gen`) against the analytic dataset contract.

Every test here runs REAL Sionna RT ray tracing on the GPU, so the whole module is
gated behind `@pytest.mark.sionna` (RUN_SIONNA=1) -- CI has no GPU and, on this box,
no working LLVM/CPU DrJit backend either.

Deviation from the brief, deliberately: the Sionna import is done inside a session
fixture (`sionna_rt`) rather than as a module-level `pytest.importorskip`. A
module-level import runs at COLLECTION time, i.e. during a plain `pytest` run too --
which would initialise DrJit/CUDA in the same process as the rest of the suite even
though every test in this file is about to be skipped. The fixture keeps the default
run completely untouched while still skipping cleanly when Sionna is absent.

Scenes are kept tiny on purpose (free space or one ground plane, one or two small
scatterers, `max_depth <= 2`, tens of chirps) so the gated module still finishes in
well under a minute.
"""
import dataclasses
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from e2e.ml.radar_config import C_MPS, TI_IWR1443
from e2e.ml.transforms import adc_to_rd, tdm_deinterleave
from e2e.scenario import Node, NodeRole, Scenario

pytestmark = pytest.mark.sionna


# --------------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def sionna_rt():
    """Import Sionna RT once per session (skips the module if it is unavailable)."""
    return pytest.importorskip("sionna.rt")


@pytest.fixture(scope="module")
def cfg():
    """A TI_IWR1443 preset shrunk to a test-sized CPI (same 3TX/4RX TDM geometry).

    n_samples=128 -> max_range 9.6 m, n_chirps=48 -> 16 chirps/TX for the Doppler FFT.
    """
    return dataclasses.replace(TI_IWR1443, name="ti_test", n_chirps=48, n_samples=128)


RADAR_POS = (0.0, 0.0, 1.5)

# Sionna's `sphere.ply` primitive, measured from its bounding box (+-0.987 m).
SPHERE_RADIUS_M = 0.987231
TARGET_SCALING = 0.15


def _scene(cfg, target_pos, velocity_mps, *, scaling=TARGET_SCALING, n_frames=2):
    """One small metal sphere with an explicit velocity, radar at the origin facing +x.

    The sphere is scaled to ~15 cm because ray tracing reflects off the SURFACE, not
    the centre: at full size it would peak a whole metre (13 range bins) closer than
    the point-target model predicts. Even at 15 cm the offset is ~2 bins, which the
    range expectation corrects for explicitly (see `_expected_bins`). Shrinking it
    further instead would starve the Monte-Carlo diffuse sampling of ray hits.
    """
    from e2e.ml.scatterers import vehicle

    return Scenario(
        name="rt_gen_test", base_scene="free", num_frames=n_frames,
        nodes=[Node(name="radar", role=NodeRole.RADAR, position=RADAR_POS,
                    look_at=(10.0, 0.0, 1.5))],
        objects=[dataclasses.replace(
            vehicle("car", target_pos, velocity=velocity_mps), scaling=scaling)],
    )


def _expected_bins(cfg, target_pos, velocity_mps, surface_offset_m=0.0):
    """Analytic (range, doppler, angle) bins -- the same math as test_ml_integration.

    `surface_offset_m` is the ONLY concession to ray tracing: the echo comes off the
    scatterer's near surface, so the expected range is the centre range minus the
    object's radius. Doppler and angle need no such correction (the near-surface point
    shares the centre's radial velocity and, to within a milliradian at these ranges,
    its bearing).
    """
    r = math.dist(target_pos, RADAR_POS)
    e_los = tuple((p - q) / r for p, q in zip(target_pos, RADAR_POS))
    v_r = sum(v * e for v, e in zip(velocity_mps, e_los))     # receding-positive
    sin_th = e_los[1]                                         # boresight +x -> ULA +y
    k_range = 2.0 * (r - surface_offset_m) * cfg.bandwidth_hz / C_MPS
    t_eff = cfg.n_tx * cfg.chirp_period_s
    k_dopp = (cfg.n_chirps_per_tx // 2
              + 2.0 * v_r * t_eff / cfg.wavelength_m * cfg.n_chirps_per_tx)
    k_ang = cfg.n_virtual // 2 + cfg.n_virtual * sin_th / 2.0
    return k_range, k_dopp, k_ang


def _rd(cfg, adc):
    """TDM deinterleave + range-Doppler, exactly as `e2e.ml.dataset.generate_sample` does."""
    sub = dataclasses.replace(cfg, n_tx=1, mimo="single", n_chirps=cfg.n_chirps_per_tx)
    return adc_to_rd(sub, tdm_deinterleave(cfg, adc))


# --------------------------------------------------------------------------------
# Dataset contract
# --------------------------------------------------------------------------------
def test_adc_matches_the_analytic_dataset_contract(sionna_rt, cfg, torch_device):
    """Shape/dtype/device must be indistinguishable from `rd_synth.synthesize_adc`."""
    from e2e.ml.rt_gen import rt_synthesize_adc

    sc = _scene(cfg, (5.0, 2.0, 1.5), (-5.0, 0.0, 0.0))
    adc = rt_synthesize_adc(cfg, sc, base_scene="free", snr_db=None, device=torch_device)

    assert adc.shape == (cfg.n_rx, cfg.n_chirps, cfg.n_samples)
    assert adc.dtype == torch.complex64
    assert adc.device.type == torch_device.type
    assert torch.isfinite(adc.real).all() and torch.isfinite(adc.imag).all()
    assert adc.abs().sum() > 0, "ray tracing found no paths to the target"


def test_beat_frequency_grid_spans_the_ramp(sionna_rt, cfg):
    """Equation (3): the CFR grid is the ramp, expressed as offsets from f0 + B/2."""
    from e2e.ml.rt_gen import beat_frequencies

    f = beat_frequencies(cfg)
    assert f.shape == (cfg.n_samples,)
    assert f[0] == pytest.approx(-cfg.bandwidth_hz / 2.0)
    # last sample is one step short of the far end (n runs 0..n_samples-1)
    step = cfg.ramp_slope_hzps / cfg.fs_hz
    assert f[-1] == pytest.approx(cfg.bandwidth_hz / 2.0 - step)


# --------------------------------------------------------------------------------
# THE validation: a known scatterer through the EXISTING transforms
# --------------------------------------------------------------------------------
def test_known_scatterer_lands_in_expected_range_doppler_angle_bins(
        sionna_rt, cfg, torch_device):
    """A ray-traced target must land where the analytic model says it should.

    This is the whole point of the module: same scene, same transforms, same bin math
    as `tests/test_ml_integration.py`, but the ADC came out of Sionna RT rather than
    the closed-form synthesizer. Range, Doppler AND angle, all within one bin.
    """
    from e2e.ml.rt_gen import rt_synthesize_adc

    pos, vel = (5.0, 2.0, 1.5), (-5.0, 0.0, 0.0)
    sc = _scene(cfg, pos, vel)
    adc = rt_synthesize_adc(cfg, sc, base_scene="free", snr_db=None, device=torch_device)
    rd = _rd(cfg, adc)

    k_range, k_dopp, k_ang = _expected_bins(
        cfg, pos, vel, surface_offset_m=SPHERE_RADIUS_M * TARGET_SCALING)
    power = rd.abs() ** 2

    rbin = int(power.sum(dim=(0, 2)).argmax())
    assert abs(rbin - k_range) <= 1.0, f"range bin {rbin}, expected ~{k_range:.2f}"

    dbin = int(power.sum(dim=0)[rbin].argmax())
    assert abs(dbin - k_dopp) <= 1.0, f"doppler bin {dbin}, expected ~{k_dopp:.2f}"

    # Angle: FFT across the 12-element lambda/2 virtual ULA at the target's cell.
    spec = torch.fft.fftshift(torch.fft.fft(rd[:, rbin, dbin]), dim=0).abs()
    abin = int(spec.argmax())
    assert abs(abin - k_ang) <= 1.0, f"angle bin {abin}, expected ~{k_ang:.2f}"


def test_static_scene_puts_all_energy_at_zero_doppler(sionna_rt, cfg, torch_device):
    from e2e.ml.rt_gen import rt_synthesize_adc

    pos = (5.0, 2.0, 1.5)
    sc = _scene(cfg, pos, (0.0, 0.0, 0.0))
    adc = rt_synthesize_adc(cfg, sc, base_scene="free", snr_db=None, device=torch_device)
    rd = _rd(cfg, adc)

    power = rd.abs() ** 2
    rbin = int(power.sum(dim=(0, 2)).argmax())
    zero_bin = cfg.n_chirps_per_tx // 2          # adc_to_rd fftshifts zero Doppler to the centre
    dprof = power.sum(dim=0)[rbin]
    assert int(dprof.argmax()) == zero_bin
    # ...and the zero-Doppler cell must actually dominate, not merely win.
    assert dprof[zero_bin] > 0.5 * dprof.sum()


# --------------------------------------------------------------------------------
# MIMO handling
# --------------------------------------------------------------------------------
def test_tdm_selects_one_transmitter_per_chirp(sionna_rt, cfg, torch_device):
    """`mimo_combine` must reproduce exactly `beat[:, c % n_tx, c, :]` for TDM.

    Checked against the raw per-TX beat cube (not a re-derivation of it), which is
    what `transforms.tdm_deinterleave` assumes when it gathers chirp `t::n_tx` into
    virtual row `t*n_rx + r`.
    """
    from e2e.ml.rt_gen import _beat_from_paths, _solve, build_rt_scene, mimo_combine

    sc = _scene(cfg, (5.0, 2.0, 1.5), (-5.0, 0.0, 0.0))
    rt_scene = build_rt_scene(sc, cfg, base_scene="free")
    paths = _solve(rt_scene, max_depth=2, include_leakage=False, diffuse_reflection=True,
                   specular_reflection=True, refraction=False, seed=41)
    beat = _beat_from_paths(paths, cfg, n_chirps=cfg.n_chirps)
    assert beat.shape == (cfg.n_rx, cfg.n_tx, cfg.n_chirps, cfg.n_samples)

    adc = mimo_combine(cfg, beat)
    assert adc.shape == (cfg.n_rx, cfg.n_chirps, cfg.n_samples)
    for c in range(cfg.n_chirps):
        np.testing.assert_array_equal(adc[:, c, :], beat[:, c % cfg.n_tx, c, :])

    # DDMA on the same beat cube: every TX on every chirp, with the 2pi t c / n_tx code.
    ddma_cfg = dataclasses.replace(cfg, mimo="ddma")
    ddma = mimo_combine(ddma_cfg, beat)
    c = 5
    ref = sum(beat[:, t, c, :] * np.exp(2j * np.pi * t * c / cfg.n_tx)
              for t in range(cfg.n_tx))
    np.testing.assert_allclose(ddma[:, c, :], ref, rtol=1e-4, atol=1e-6 * np.abs(ref).max())


# --------------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------------
def test_repeated_generation_is_reproducible(sionna_rt, cfg, torch_device):
    """Same solver seed -> same frame; a different solver seed -> a different one.

    RT nondeterminism, measured on this box: repeating a solve with the SAME
    `solver_seed` reproduces the ADC to ~5e-7 relative (float32 / GPU reduction-order
    noise, not exact bit equality -- do not assert `torch.equal`). Sionna's DIFFUSE
    reflections are Monte-Carlo sampled, so changing `solver_seed` (or moving geometry
    by a fraction of a millimetre) fully re-randomises them: a different seed gives a
    ~100% relative difference, which is why `doppler_error_study` reports an explicit
    Monte-Carlo noise floor.
    """
    from e2e.ml.rt_gen import build_rt_scene, rt_synthesize_adc

    sc = _scene(cfg, (5.0, 2.0, 1.5), (-5.0, 0.0, 0.0))
    rt_scene = build_rt_scene(sc, cfg, base_scene="free")
    kw = dict(rt_scene=rt_scene, snr_db=None, device=torch_device)

    a = rt_synthesize_adc(cfg, sc, solver_seed=41, **kw)
    b = rt_synthesize_adc(cfg, sc, solver_seed=41, **kw)
    rel = ((a - b).abs().pow(2).sum().sqrt() / a.abs().pow(2).sum().sqrt()).item()
    assert rel < 1e-4, f"same-seed solves differ by {rel:.2e}"

    c = rt_synthesize_adc(cfg, sc, solver_seed=99, **kw)
    rel_seed = ((a - c).abs().pow(2).sum().sqrt() / a.abs().pow(2).sum().sqrt()).item()
    assert rel_seed > 1e-3, "diffuse sampling looks seed-independent -- check the solver"

    # AWGN is seeded independently of the solver.
    n1 = rt_synthesize_adc(cfg, sc, solver_seed=41, snr_db=20.0, seed=7,
                           rt_scene=rt_scene, device=torch_device)
    n2 = rt_synthesize_adc(cfg, sc, solver_seed=41, snr_db=20.0, seed=7,
                           rt_scene=rt_scene, device=torch_device)
    rel_noise = ((n1 - n2).abs().pow(2).sum().sqrt() / n1.abs().pow(2).sum().sqrt()).item()
    assert rel_noise < 1e-4


def test_noise_raises_the_floor_without_moving_the_peak(sionna_rt, cfg, torch_device):
    """`snr_db` must add energy but leave the target's range/Doppler cell where it was."""
    from e2e.ml.rt_gen import build_rt_scene, rt_synthesize_adc

    pos, vel = (5.0, 2.0, 1.5), (-5.0, 0.0, 0.0)
    sc = _scene(cfg, pos, vel)
    rt_scene = build_rt_scene(sc, cfg, base_scene="free")

    clean = rt_synthesize_adc(cfg, sc, rt_scene=rt_scene, snr_db=None, device=torch_device)
    noisy = rt_synthesize_adc(cfg, sc, rt_scene=rt_scene, snr_db=25.0, seed=0,
                              device=torch_device)
    assert noisy.abs().pow(2).sum() > clean.abs().pow(2).sum()

    p_clean, p_noisy = _rd(cfg, clean).abs() ** 2, _rd(cfg, noisy).abs() ** 2
    rb_c = int(p_clean.sum(dim=(0, 2)).argmax())
    rb_n = int(p_noisy.sum(dim=(0, 2)).argmax())
    assert abs(rb_c - rb_n) <= 1


def test_snr_calibration_matches_rd_synths_convention(sionna_rt, cfg, torch_device):
    """`snr_db` really is the post-2-D-FFT peak SNR of the strongest target.

    `rt_gen` cannot read a scatterer amplitude off a ray-traced scene, so it inverts
    rd_synth's relation from the measured FFT peak instead. This checks the round trip:
    inject noise at 25 dB, measure the injected variance from the clean/noisy
    difference, and recover the SNR with the same estimator the generator used.
    """
    from e2e.ml.rt_gen import (_coherent_gain, _peak_reference_amplitude,
                               build_rt_scene, rt_synthesize_adc)

    sc = _scene(cfg, (5.0, 2.0, 1.5), (-5.0, 0.0, 0.0))
    rt_scene = build_rt_scene(sc, cfg, base_scene="free")
    clean = rt_synthesize_adc(cfg, sc, rt_scene=rt_scene, snr_db=None, device=torch_device)
    noisy = rt_synthesize_adc(cfg, sc, rt_scene=rt_scene, snr_db=25.0, seed=3,
                              device=torch_device)

    sigma2 = float((noisy - clean).abs().pow(2).mean())
    clean_np = clean.cpu().numpy()
    a_max = _peak_reference_amplitude(cfg, clean_np, 3.0 * cfg.range_resolution_m)
    gain = _coherent_gain(cfg, clean_np)
    snr_db = 10.0 * math.log10(a_max ** 2 * gain / sigma2)
    assert abs(snr_db - 25.0) < 0.5, f"requested 25 dB, measured {snr_db:.2f} dB"


# --------------------------------------------------------------------------------
# Scene building
# --------------------------------------------------------------------------------
def test_build_rt_scene_wires_arrays_objects_and_velocity(sionna_rt, cfg):
    from e2e.ml.rt_gen import build_rt_scene

    sc = _scene(cfg, (5.0, 2.0, 1.5), (-5.0, 0.0, 0.0))
    for base in ("free", "flat"):
        rts = build_rt_scene(sc, cfg, base_scene=base)
        assert rts.scene.frequency[0] == pytest.approx(cfg.f0_hz + cfg.bandwidth_hz / 2.0,
                                                       rel=1e-6)
        assert rts.scene.tx_array.array_size == cfg.n_tx
        assert rts.scene.rx_array.array_size == cfg.n_rx
        assert set(rts.objects) == {"car"}
        v = rts.objects["car"].velocity
        assert float(v.x[0]) == pytest.approx(-5.0)
        # a non-zero scattering coefficient is what makes a small target visible at all
        assert float(rts.materials["car"].scattering_coefficient[0]) > 0.0

    # "flat" adds a ground plane; "free" does not.
    n_free = len(build_rt_scene(sc, cfg, base_scene="free").scene.objects)
    n_flat = len(build_rt_scene(sc, cfg, base_scene="flat").scene.objects)
    assert n_flat == n_free + 1


# --------------------------------------------------------------------------------
# Per-chirp re-trace reference + the Doppler error study
# --------------------------------------------------------------------------------
def test_retrace_reference_equals_native_for_a_static_scene(sionna_rt, cfg, torch_device):
    """With nothing moving, re-tracing per chirp must reproduce the native cube.

    This pins the re-trace path's bookkeeping (per-chirp displacement, TDM selection,
    `num_time_steps=1` slicing): any indexing slip would show up here even though the
    physics is trivial.
    """
    from e2e.ml.rt_gen import build_rt_scene, rt_retrace_reference, rt_synthesize_adc

    n_cap = 6
    small = dataclasses.replace(cfg, n_chirps=n_cap)
    sc = _scene(cfg, (5.0, 2.0, 1.5), (0.0, 0.0, 0.0))
    rt_scene = build_rt_scene(sc, cfg, base_scene="free")

    native = rt_synthesize_adc(small, sc, rt_scene=rt_scene, snr_db=None,
                               device=torch_device)
    ref = rt_retrace_reference(small, sc, rt_scene=rt_scene, n_chirps_cap=n_cap,
                               snr_db=None, device=torch_device)
    assert ref.shape == native.shape == (cfg.n_rx, n_cap, cfg.n_samples)
    rel = ((native - ref).abs().pow(2).sum().sqrt() / ref.abs().pow(2).sum().sqrt()).item()
    assert rel < 1e-4, f"static-scene re-trace disagrees with the native cube ({rel:.2e})"


def test_retrace_reference_first_chirp_matches_native(sionna_rt, cfg, torch_device):
    """Chirp 0 is the same geometry with Doppler phase exp(0) = 1 in both paths."""
    from e2e.ml.rt_gen import build_rt_scene, rt_retrace_reference, rt_synthesize_adc

    n_cap = 6
    small = dataclasses.replace(cfg, n_chirps=n_cap)
    sc = _scene(cfg, (5.0, 2.0, 1.5), (-5.0, 0.0, 0.0))
    rt_scene = build_rt_scene(sc, cfg, base_scene="free")

    native = rt_synthesize_adc(small, sc, rt_scene=rt_scene, snr_db=None,
                               device=torch_device)
    ref = rt_retrace_reference(small, sc, rt_scene=rt_scene, n_chirps_cap=n_cap,
                               snr_db=None, device=torch_device)
    d = (native[:, 0, :] - ref[:, 0, :]).abs().pow(2).sum().sqrt()
    assert (d / ref[:, 0, :].abs().pow(2).sum().sqrt()).item() < 1e-4


def test_doppler_error_study_smoke(sionna_rt, cfg, torch_device):
    """One frame, 12-chirp CPI, specular-only box target (deterministic, no MC floor)."""
    from e2e.ml.rt_gen import _demo_scenario, doppler_error_study, format_error_study

    small = dataclasses.replace(cfg, n_chirps=12)
    sc = _demo_scenario(1, small, target="box")
    res = doppler_error_study(small, sc, n_frames=1, base_scene="free", n_chirps_cap=12,
                              diffuse_reflection=False, device=torch_device)

    assert res["n_chirps"] == 12 and res["solves_retrace"] == 12
    f = res["frames"][0]
    assert len(f["per_chirp_rel_err"]) == 12
    assert f["per_chirp_rel_err"][0] < 1e-5      # chirp 0: identical geometry
    assert f["rel_rmse"] >= f["per_chirp_rel_err"][0]
    assert f["mc_noise_floor"] < 0.05            # specular-only is near-deterministic
    assert res["cost_multiplier"] > 1.0          # re-tracing is the expensive path
    assert "peak-bin agreement" in format_error_study(res)
