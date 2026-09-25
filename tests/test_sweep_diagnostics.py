"""Tests for `e2e.environment.sweep_diagnostics` -- the measurement that decides whether
a stored frame file rotates its principal direction while holding its rank.

Synthetic frames with a KNOWN answer (a rank-1 outer product of a steering vector at a
chosen direction cosine and a random spectrum), so the diagnostic is checked against
geometry rather than against another measurement. No Sionna, no .pkl on disk except the
one the round-trip test writes itself.
"""
import pickle

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from e2e.environment.sweep_diagnostics import (  # noqa: E402
    angular_peak,
    diagnose_file,
    frame_matrix,
    pipeline_subspace_err,
    principal_angle_deg,
    range_az_peak_minus_median_db,
    subspace_angle_deg,
)

_N_AZ = _N_EL = 32
_N_RX = _N_AZ * _N_EL


def _steering(u_az, u_el=0.0, n_az=_N_AZ, n_el=_N_EL):
    """Unit-norm lambda/2 array response at direction cosines `(u_az, u_el)`, flattened
    in the aperture order `SionnaEnvironmentBlock` assumes (dim 0 = azimuth = the SLOW
    axis of the flat rx index)."""
    az = np.exp(1j * np.pi * u_az * np.arange(n_az))
    el = np.exp(1j * np.pi * u_el * np.arange(n_el))
    a = np.outer(az, el).reshape(-1)
    return (a / np.linalg.norm(a)).astype(np.complex64)


def _rank1_frame(u_az, n_freqs=64, seed=0):
    """A `[n_rx, 1, 1, n_freqs]` frame that is EXACTLY rank 1: one steering vector times
    one random frequency response."""
    r = np.random.default_rng(seed)
    spec = (r.standard_normal(n_freqs) + 1j * r.standard_normal(n_freqs)).astype(np.complex64)
    return np.outer(_steering(u_az), spec).reshape(_N_RX, 1, 1, n_freqs).astype(np.complex64)


def test_frame_matrix_folds_a_stored_frame_to_rx_by_freq():
    f = _rank1_frame(0.25, n_freqs=16)
    m = frame_matrix(f)
    assert tuple(m.shape) == (_N_RX, 16)


#: Angle floor of this measurement at complex64: `acos` of an inner product that rounds
#: to 1 - O(1e-7) is already ~0.04 deg, and the SVD of a (numerically) rank-1 frame adds
#: backend-dependent error on top -- measured 2026-09-25 on this box: 0.04 deg for the
#: direct inner product, 0.41 deg between the u1 of two CUDA (cuSOLVER, complex64) SVDs
#: of the same steering direction. 1 deg covers both backends and still sits two orders
#: of magnitude below the per-frame rotation the swept file shows (~50 deg).
_ANGLE_FLOOR_DEG = 1.0


def test_principal_angle_is_zero_for_the_same_direction_and_sign_invariant():
    a = torch.as_tensor(_steering(0.25))
    assert principal_angle_deg(a, a) == pytest.approx(0.0, abs=_ANGLE_FLOOR_DEG)
    # A singular vector's phase is arbitrary -- a global phase must not register as a
    # direction change.
    assert principal_angle_deg(a, a * np.exp(1j * 1.234)) == pytest.approx(
        0.0, abs=_ANGLE_FLOOR_DEG)


def test_principal_angle_grows_with_the_angular_separation():
    a = torch.as_tensor(_steering(0.0))
    small = principal_angle_deg(a, torch.as_tensor(_steering(0.02)))
    large = principal_angle_deg(a, torch.as_tensor(_steering(0.20)))
    assert 0.0 < small < large
    # 0.0625 in u is one native bin of a 32-element axis: separations well past that are
    # essentially orthogonal directions.
    assert large > 60.0


def test_subspace_angle_zero_for_the_same_basis_and_ninety_for_orthogonal_ones():
    a = torch.as_tensor(_steering(0.0)).reshape(-1, 1)
    b = torch.as_tensor(_steering(0.125)).reshape(-1, 1)   # two native bins away
    basis1 = torch.cat([a, b], dim=1)
    q1, _ = torch.linalg.qr(basis1)
    assert subspace_angle_deg(q1, q1) == pytest.approx(0.0, abs=_ANGLE_FLOOR_DEG)
    # An orthonormal basis of two other array directions, made orthogonal to q1 by
    # construction: the largest principal angle must be 90 deg.
    rest = torch.linalg.qr(torch.cat([q1, torch.as_tensor(
        np.random.default_rng(1).standard_normal((_N_RX, 4))
        + 1j * np.random.default_rng(2).standard_normal((_N_RX, 4))).to(q1.dtype)], dim=1))[0]
    q2 = rest[:, 2:4]
    assert subspace_angle_deg(q1, q2) == pytest.approx(90.0, abs=1e-2)


@pytest.mark.parametrize("u_az", [-0.5, -0.25, 0.0, 0.25, 0.5])
def test_angular_peak_finds_the_steering_direction_on_the_native_grid(u_az):
    """The peak must land in the native bin the direction cosine falls in (bin width
    1/(N/2) = 0.0625 for 32 elements) -- and for a pure steering vector essentially all
    the angular power is in that one bin."""
    peak_az, peak_el, share = angular_peak(torch.as_tensor(_steering(u_az)),
                                           (_N_AZ, _N_EL))
    assert peak_az == pytest.approx(u_az, abs=0.0626)
    assert peak_el == pytest.approx(0.0, abs=0.0626)
    assert share > 0.9


def test_angular_peak_moves_with_the_steering_direction():
    a = angular_peak(torch.as_tensor(_steering(-0.5)), (_N_AZ, _N_EL))[0]
    b = angular_peak(torch.as_tensor(_steering(+0.5)), (_N_AZ, _N_EL))[0]
    assert b - a == pytest.approx(1.0, abs=0.13)


def test_peak_minus_median_db_is_positive_and_finite_on_a_rank1_frame():
    pm = range_az_peak_minus_median_db(_rank1_frame(0.25, n_freqs=256), (_N_AZ, _N_EL),
                                       bins=64)
    assert np.isfinite(pm) and pm > 0.0


def test_diagnose_file_reports_rank1_and_a_rotating_direction(tmp_path):
    """End to end on a written pkl: three EXACTLY rank-1 frames whose steering direction
    steps by two native bins per frame. Rank must read 1 every frame while the measured
    consecutive principal angle is large -- the "rank doesn't change, direction rotates"
    signature the swept file has to show."""
    frames = np.stack([_rank1_frame(u, n_freqs=64, seed=i)
                       for i, u in enumerate((-0.25, 0.0, 0.25))], axis=0)
    path = tmp_path / "rank1_sweep.pkl"
    with open(path, "wb") as f:
        pickle.dump({"meta": {"version": 2, "links": {"munich": {"rx_array_shape": [32, 32]}}},
                     "links": {"munich": frames}}, f)
    rows, meta = diagnose_file(str(path), k=1)
    assert [r["effective_rank"] for r in rows] == [1, 1, 1]
    assert [r["u_az"] for r in rows] == pytest.approx([-0.25, 0.0, 0.25], abs=0.0626)
    angles = [r["angle_u1_prev_deg"] for r in rows[1:]]
    assert all(a > 45.0 for a in angles)
    assert np.isnan(rows[0]["angle_u1_prev_deg"])   # no previous frame
    assert meta["version"] == 2


def test_diagnose_file_frames_limit_and_static_direction(tmp_path):
    """Control: the SAME direction every frame must give ~zero consecutive angle, so a
    large angle in the test above is evidence about the data, not an artifact of the
    measurement."""
    frames = np.stack([_rank1_frame(0.25, n_freqs=64, seed=i) for i in range(4)], axis=0)
    path = tmp_path / "static.pkl"
    with open(path, "wb") as f:
        pickle.dump({"meta": {"version": 2, "links": {"munich": {"rx_array_shape": [32, 32]}}},
                     "links": {"munich": frames}}, f)
    rows, _meta = diagnose_file(str(path), n_frames=2, k=1)
    assert len(rows) == 2
    assert rows[1]["angle_u1_prev_deg"] == pytest.approx(0.0, abs=_ANGLE_FLOOR_DEG)


#: The synthetic fixtures' frequency plan. It is not decoration: `Simulation`'s
#: default composition puts the front end on the beat record and derives the beat
#: sample rate from the SOURCE's plan when no `radar_cfg=` is given, refusing by name
#: when there is neither (e2e/simulation.py `_build_spine`). Every real file this
#: module runs on is a v2 pkl with a plan; the fixture has to be one too, or it is
#: testing a source shape that does not exist. Endpoint-inclusive over the frames'
#: own `n_freqs`, the same convention the generator writes (F97d).
def _freq_plan(n_freqs):
    return {"carrier_hz": 30e9, "start_hz": 28.5e9, "stop_hz": 31.5e9,
            "num_freqs": int(n_freqs)}


def _write_pkl(path, frames):
    with open(path, "wb") as f:
        pickle.dump({"meta": {"version": 2,
                              # TOP-LEVEL: `SionnaIterator.freq_plan` reads
                              # `meta["freq_plan"]`, not the link's own entry.
                              "freq_plan": _freq_plan(frames.shape[-1]),
                              "links": {"munich": {"rx_array_shape": [32, 32],
                                                   "physical_scale": False}}},
                     "links": {"munich": frames}}, f)
    return str(path)


def test_pipeline_subspace_err_runs_the_real_chain_and_restores_the_path(tmp_path):
    """`--pipeline` must actually drive `Simulation` (RFFE -> interconnect -> AFE ->
    AdaOja) on the given file and hand back one error per frame per repeat -- and must
    leave `SIONNA_MUNICH_PATH` as it found it, since that attribute is global and the
    rest of the process (and the rest of the suite) reads it."""
    import e2e.environment.sionna_iterator as sionna_iterator

    before = sionna_iterator.SIONNA_MUNICH_PATH
    path = _write_pkl(tmp_path / "chain.pkl",
                      np.stack([_rank1_frame(u, n_freqs=64, seed=i)
                                for i, u in enumerate((-0.25, 0.0, 0.25))], axis=0))
    runs = pipeline_subspace_err(path, n_steps=3, k=1, m=64, n_refine=2, repeats=2)
    assert sionna_iterator.SIONNA_MUNICH_PATH == before
    assert len(runs) == 2
    assert all(len(r) == 3 for r in runs)
    assert all(np.isfinite(e) and e >= 0.0 for r in runs for e in r)


def test_pipeline_subspace_err_restores_the_path_even_when_the_run_raises(tmp_path):
    import e2e.environment.sionna_iterator as sionna_iterator

    before = sionna_iterator.SIONNA_MUNICH_PATH
    # k larger than the frame's rx count is a loud failure inside the run, not a silent
    # one -- whatever it raises, the global path must be put back.
    path = _write_pkl(tmp_path / "bad.pkl",
                      np.stack([_rank1_frame(0.0, n_freqs=8, seed=0)], axis=0))
    with pytest.raises(Exception):
        pipeline_subspace_err(path, n_steps=1, k=4096, m=8, n_refine=1)
    assert sionna_iterator.SIONNA_MUNICH_PATH == before


def test_pipeline_subspace_err_separates_a_rotating_file_from_a_static_one(tmp_path):
    """The claim the T3 choice rests on, in miniature: a file whose principal direction
    jumps every frame must leave the tracker at a HIGHER error than one whose direction
    stands still, with everything else equal. Synthetic rank-1 frames, so the only
    difference between the two files is whether the direction moves."""
    static = _write_pkl(tmp_path / "static_chain.pkl",
                        np.stack([_rank1_frame(0.25, n_freqs=64, seed=i)
                                  for i in range(4)], axis=0))
    moving = _write_pkl(tmp_path / "moving_chain.pkl",
                        np.stack([_rank1_frame(u, n_freqs=64, seed=i)
                                  for i, u in enumerate((-0.375, -0.125, 0.125, 0.375))],
                                 axis=0))
    err_static = pipeline_subspace_err(static, n_steps=4, k=1, m=64, n_refine=2)[0][1:]
    err_moving = pipeline_subspace_err(moving, n_steps=4, k=1, m=64, n_refine=2)[0][1:]
    assert np.mean(err_moving) > np.mean(err_static)
