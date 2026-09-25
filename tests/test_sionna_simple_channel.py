"""Tests for `e2e.environment.sionna_simple_channel` (the Ka-band munich re-trace).

Sionna-free tests (`build_frequencies`, `parse_args`, the pkl writer/reader round-trip)
run unconditionally; the one test that needs a real Sionna scene is gated behind
`@pytest.mark.sionna` (RUN_SIONNA=1) -- see CLAUDE.md / tests/README.md.
"""
import pickle

import numpy as np
import pytest

from e2e.environment.sionna_iterator import SionnaIterator
from e2e.environment.sionna_simple_channel import (
    _CFR_TENSOR_BUDGET,
    _LOS_DELAY_TOL_NS,
    _cfr_chunk_size,
    _los_receipt,
    _path_power_share,
    _synthesize_cfr,
    boresight_sin_az,
    build_frequencies,
    build_scene,
    generate,
    los_az_deg_from_sin,
    los_sweep_offsets,
    parse_args,
    street_lateral_axis,
    tx_sweep_positions,
    unambiguous_range_m,
    write_payload,
)

#: The munich scene's tx/rx placement (`build_scene`) and its per-frame receiver step --
#: the geometry the sweep tests below reproduce in plain numpy, no Sionna.
_TX0 = np.array([8.5, 21.0, 27.0])
_RX0 = np.array([45.0, 90.0, 1.5])
_RX_STEP = np.array([1.0, 0.0, 0.0])


def _look_at_orientation(rx_pos, tx_pos, offset_deg=0.0):
    """`(alpha, beta, gamma)` for `rx.look_at(tx)` followed by `aim_receiver`'s yaw --
    the closed form Sionna's `look_at` sets (alpha=phi, beta=theta-pi/2, gamma=0), with
    `offset_deg` SUBTRACTED from alpha, as `aim_receiver` does."""
    d = np.asarray(tx_pos, dtype=float) - np.asarray(rx_pos, dtype=float)
    theta = np.arccos(d[2] / np.linalg.norm(d))
    phi = np.arctan2(d[1], d[0])
    return (phi - np.radians(offset_deg), theta - np.pi / 2, 0.0)


def test_build_frequencies_relative_to_carrier_and_symmetric_for_default_band():
    freqs = build_frequencies(30e9, (28.5e9, 31.5e9), 1000)
    assert freqs.shape == (1000,)
    assert freqs[0] == pytest.approx(-1.5e9)
    assert freqs[-1] == pytest.approx(1.5e9)
    # Default band is centred on the carrier -> the relative grid is symmetric.
    assert freqs[0] == pytest.approx(-freqs[-1])


def test_build_frequencies_num_points():
    freqs = build_frequencies(30e9, (28.5e9, 31.5e9), 7)
    assert freqs.shape == (7,)


def test_unambiguous_range_m_1000pt_default_band():
    # 1000 points over 3 GHz -> ~25 m (measured, munich_physics investigation 2026-09-23:
    # too short for this scene's 37/68 m non-LoS returns, which then alias). The
    # investigation's "25.0 m" used c=3e8; exact c=299792458 gives 24.98 m.
    assert unambiguous_range_m(1000, (28.5e9, 31.5e9)) == pytest.approx(25.0, rel=1e-2)


def test_unambiguous_range_m_5000pt_default_band():
    # 5000 points over 3 GHz -> ~125 m.
    assert unambiguous_range_m(5000, (28.5e9, 31.5e9)) == pytest.approx(125.0, rel=1e-2)


def test_parse_args_defaults():
    args = parse_args([])
    assert args.carrier_hz == 30e9
    assert args.band_hz == [28.5e9, 31.5e9]
    assert args.num_freqs == 1000
    assert args.num_frames == 100
    assert args.out.endswith("munich_ka.pkl")
    assert args.seed == 41


def test_parse_args_overrides():
    args = parse_args(["--carrier-hz", "77e9", "--band-hz", "76e9", "78e9",
                       "--num-freqs", "16", "--num-frames", "3", "--seed", "7"])
    assert args.carrier_hz == 77e9
    assert args.band_hz == [76e9, 78e9]
    assert args.num_freqs == 16
    assert args.num_frames == 3
    assert args.seed == 7


def test_parse_args_diffuse_and_boresight_defaults():
    args = parse_args([])
    assert args.diffuse is False
    assert args.scattering_coefficient == 0.0
    assert args.boresight_offset_deg == 0.0


def test_parse_args_diffuse_and_boresight_overrides():
    args = parse_args(["--diffuse", "--scattering-coefficient", "0.4",
                       "--boresight-offset-deg", "35"])
    assert args.diffuse is True
    assert args.scattering_coefficient == 0.4
    assert args.boresight_offset_deg == 35.0


# --------------------------------------------------------------------------- CFR chunking


def test_cfr_chunk_size_stays_under_tensor_budget():
    # A path-rich solve (synthetic_array=True + diffuse: tens of thousands of paths,
    # see generate()'s "lost paths" fix) must not ask for the full frequency count.
    num_paths, n_rx_ant, num_freqs = 33_000, 1024, 5000
    chunk = _cfr_chunk_size(num_freqs, num_paths, n_rx_ant)
    assert chunk * num_paths * n_rx_ant <= _CFR_TENSOR_BUDGET
    assert chunk < num_freqs


def test_cfr_chunk_size_no_chunking_needed_for_small_problems():
    # Few paths -> the whole frequency axis fits in one call.
    assert _cfr_chunk_size(num_freqs=5000, num_paths=10, n_rx_ant=1024) == 5000


def test_cfr_chunk_size_at_least_one():
    # Pathological (huge) path count must still return a usable (>=1) chunk, not 0.
    assert _cfr_chunk_size(num_freqs=5000, num_paths=10_000_000, n_rx_ant=1024) >= 1


class _FakePaths:
    """Sionna `Paths`-shaped test double: `.tau` for path count, `.cfr()` returning a
    DETERMINISTIC, frequency-dependent (not chunk-boundary-uniform) response so a
    per-chunk-normalize bug (this function's regression) shows up as a boundary
    discontinuity, not just a benign global scale error."""

    def __init__(self, num_paths, n_rx_ant, seed=0):
        self.tau = np.zeros((1, 1, num_paths))
        rng = np.random.default_rng(seed)
        self._n_rx_ant = n_rx_ant
        # A fixed per-(antenna, frequency-index) response so cfr() is a pure function
        # of which absolute frequencies are requested -- values vary smoothly but
        # non-uniformly so a per-chunk rescale would visibly kink at chunk boundaries.
        self._table = {}
        self._rng = rng

    def _value(self, f, ant):
        key = (float(f), int(ant))
        if key not in self._table:
            # Deterministic pseudo-random magnitude in [0.5, 1.5), phase in [0, 2pi).
            h = hash(key) % (2**32)
            r = np.random.default_rng(h)
            mag = 0.5 + r.random()
            phase = 2 * np.pi * r.random()
            self._table[key] = mag * np.exp(1j * phase)
        return self._table[key]

    def cfr(self, frequencies, normalize, normalize_delays, out_type):
        assert normalize is False  # _synthesize_cfr must never request per-chunk normalize
        n_f = len(frequencies)
        out = np.zeros((self._n_rx_ant, n_f), dtype=np.complex64)
        for j, f in enumerate(frequencies):
            for a in range(self._n_rx_ant):
                out[a, j] = self._value(f, a)
        # [num_rx=1, num_rx_ant, num_tx=1, num_tx_ant=1, num_time=1, num_freqs]
        return out.reshape(1, self._n_rx_ant, 1, 1, 1, n_f)


def test_synthesize_cfr_matches_unchunked_call_up_to_global_normalization():
    """Regression test for the per-chunk-normalize bug: chunked synthesis (forced via a
    tiny fake tensor budget) must reproduce the SAME values (up to one global scale
    factor) as a single un-chunked call -- not independently-rescaled chunks."""
    n_rx_ant = 4
    num_paths = 2
    frequencies = np.linspace(-1.5e9, 1.5e9, 37)  # deliberately not a multiple of any chunk

    paths_chunked = _FakePaths(num_paths, n_rx_ant)
    import e2e.environment.sionna_simple_channel as mod
    # Force a tiny chunk size so >1 chunk is exercised.
    old_budget = mod._CFR_TENSOR_BUDGET
    try:
        mod._CFR_TENSOR_BUDGET = num_paths * n_rx_ant * 5  # chunk size ~5
        chunked = _synthesize_cfr(paths_chunked, frequencies, n_rx_ant,
                                  normalize=True, normalize_delays=True)
    finally:
        mod._CFR_TENSOR_BUDGET = old_budget

    # Reference: one call covering everything at once (normalize=False), then the
    # SAME global normalization _synthesize_cfr applies.
    paths_ref = _FakePaths(num_paths, n_rx_ant)
    ref = paths_ref.cfr(frequencies=frequencies, normalize=False,
                        normalize_delays=True, out_type="numpy")[0, :, 0, :, :, :]
    ref = ref / np.sqrt(np.mean(np.abs(ref) ** 2))

    np.testing.assert_allclose(chunked, ref, rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------- boresight geometry


def test_boresight_sin_az_zero_at_plain_look_at():
    """F94: with orientation aimed exactly at the target (alpha=phi, beta=theta-pi/2,
    gamma=0 -- what `Receiver.look_at` sets), the direct path is dead broadside."""
    rx_pos = np.array([46.0, 90.0, 1.5])
    tx_pos = np.array([8.5, 21.0, 27.0])
    d = tx_pos - rx_pos
    theta0 = np.arccos(d[2] / np.linalg.norm(d))
    phi0 = np.arctan2(d[1], d[0])
    orientation = (phi0, theta0 - np.pi / 2, 0.0)
    assert boresight_sin_az(rx_pos, tx_pos, orientation) == pytest.approx(0.0, abs=1e-9)


def test_boresight_sin_az_positive_offset_gives_positive_sin_az():
    """Subtracting `boresight_offset_deg` (converted to radians) from alpha -- the sign
    `build_scene` uses -- must put the transmitter at POSITIVE sin(az), roughly
    sin(35 deg)=0.57 (exact value differs from the idealized planar case because this
    scene's direct path also has a ~18 deg elevation component -- see build_scene)."""
    rx_pos = np.array([46.0, 90.0, 1.5])
    tx_pos = np.array([8.5, 21.0, 27.0])
    d = tx_pos - rx_pos
    theta0 = np.arccos(d[2] / np.linalg.norm(d))
    phi0 = np.arctan2(d[1], d[0])
    beta0 = theta0 - np.pi / 2
    offset_rad = np.radians(35.0)
    orientation = (phi0 - offset_rad, beta0, 0.0)
    sin_az = boresight_sin_az(rx_pos, tx_pos, orientation)
    assert sin_az > 0.0
    assert sin_az == pytest.approx(np.sin(offset_rad), abs=0.05)


def test_boresight_sin_az_negative_offset_gives_negative_sin_az():
    rx_pos = np.array([46.0, 90.0, 1.5])
    tx_pos = np.array([8.5, 21.0, 27.0])
    d = tx_pos - rx_pos
    theta0 = np.arccos(d[2] / np.linalg.norm(d))
    phi0 = np.arctan2(d[1], d[0])
    beta0 = theta0 - np.pi / 2
    offset_rad = np.radians(-35.0)
    orientation = (phi0 - offset_rad, beta0, 0.0)
    assert boresight_sin_az(rx_pos, tx_pos, orientation) < 0.0


# ----------------------------------------------------------------- moving LoS (sweeps)
# The 2026-09-24 owner directive: "the line of sight path change angle so that even if
# the rank doesn't change, the direction changes". These tests cover the PER-FRAME
# GEOMETRY the two sweep options compute -- no Sionna, no solve.


def test_parse_args_sweeps_default_off():
    """Both sweeps must default off, so the shipped `munich_ka.pkl` recipe (a bare
    `--boresight-offset-deg 35`) is unchanged by their existence."""
    args = parse_args([])
    assert args.los_sweep_deg is None
    assert args.tx_lateral_m is None


def test_parse_args_sweep_overrides():
    args = parse_args(["--los-sweep-deg", "-30", "30", "--tx-lateral-m", "-100", "32"])
    assert args.los_sweep_deg == [-30.0, 30.0]
    assert args.tx_lateral_m == [-100.0, 32.0]


def test_los_sweep_offsets_endpoints_uniform_and_monotonic():
    offs = los_sweep_offsets(30, (-30.0, 30.0))
    assert offs.shape == (30,)
    assert offs[0] == pytest.approx(-30.0)
    assert offs[-1] == pytest.approx(30.0)
    steps = np.diff(offs)
    assert np.allclose(steps, 60.0 / 29.0)  # uniform
    assert (steps > 0).all()                # monotonic


def test_los_sweep_offsets_single_frame_sits_at_the_start():
    assert los_sweep_offsets(1, (-30.0, 30.0)) == pytest.approx(np.array([-30.0]))


def test_los_az_deg_from_sin_clips_to_the_physical_domain():
    assert los_az_deg_from_sin(0.5) == pytest.approx(30.0)
    assert los_az_deg_from_sin(-0.5) == pytest.approx(-30.0)
    # A projection of a unit vector can round off past 1.0; arcsin must not go nan.
    assert los_az_deg_from_sin(1.0 + 1e-15) == pytest.approx(90.0)
    assert los_az_deg_from_sin(-1.0 - 1e-15) == pytest.approx(-90.0)


def test_los_sweep_puts_the_los_at_the_commanded_azimuth_every_frame():
    """The point of the array-pan sweep: re-aiming at the transmitter and yawing by
    `offsets[i]` puts the LoS at sin(offsets[i]) in the array frame, frame by frame.

    Exact equality does not hold because this scene's direct path has a ~18 deg
    elevation component (see `build_scene`), which is why the tolerance is 0.05 in
    sin -- the same tolerance the fixed-offset test above uses."""
    offs = los_sweep_offsets(30, (-30.0, 30.0))
    sins = []
    for i, off in enumerate(offs):
        rx = _RX0 + (i + 1) * _RX_STEP          # rx translates every frame ...
        orientation = _look_at_orientation(rx, _TX0, off)  # ... and is re-aimed
        sins.append(boresight_sin_az(rx, _TX0, orientation))
    sins = np.array(sins)
    assert np.allclose(sins, np.sin(np.radians(offs)), atol=0.05)
    az = np.degrees(np.arcsin(np.clip(sins, -1, 1)))
    assert (np.diff(az) > 0).all()                          # monotonic sweep
    assert az.max() - az.min() == pytest.approx(60.0, abs=3.0)  # ~the commanded span


def test_no_sweep_keeps_the_fixed_attitude_and_barely_moves_the_azimuth():
    """CONTROL for the test above: the shipped recipe (aim once, then translate only)
    leaves the LoS azimuth drifting by ~14 deg over 30 frames -- an order of magnitude
    less per frame than the swept file, and driven by the receiver's own 30 m of travel,
    not by the scene.

    Note the aim happens at `_RX0` itself, BEFORE the loop's first `+= [1,0,0]` step
    (`build_scene` aims; `generate` then steps and solves), so frame i sits at
    `_RX0 + (i+1)*step` with an attitude fixed at `_RX0`. The two endpoint values below
    are what the generator itself printed for a 2-frame run in this configuration on
    2026-09-25 (+32.46 / +31.89 for frames 0 and 1)."""
    orientation = _look_at_orientation(_RX0, _TX0, 35.0)  # aimed ONCE, pre-step
    az = []
    for i in range(30):
        rx = _RX0 + (i + 1) * _RX_STEP
        az.append(np.degrees(np.arcsin(np.clip(boresight_sin_az(rx, _TX0, orientation),
                                               -1, 1))))
    az = np.array(az)
    assert az[0] == pytest.approx(32.46, abs=0.05)
    assert az[1] == pytest.approx(31.89, abs=0.05)
    assert az[-1] == pytest.approx(18.28, abs=0.05)
    assert az.max() - az.min() == pytest.approx(14.19, abs=0.05)
    assert abs(np.diff(az)).max() < 0.6  # deg/frame


def test_street_lateral_axis_is_horizontal_unit_and_perpendicular_to_the_los():
    lat = street_lateral_axis(_RX0 + _RX_STEP, _TX0)
    assert lat[2] == 0.0                                   # never changes tx height
    assert np.linalg.norm(lat) == pytest.approx(1.0)
    d = _TX0 - (_RX0 + _RX_STEP)
    assert float(np.dot(lat, [d[0], d[1], 0.0])) == pytest.approx(0.0, abs=1e-9)


def test_street_lateral_axis_raises_when_nodes_are_vertically_aligned():
    with pytest.raises(ValueError):
        street_lateral_axis([0.0, 0.0, 0.0], [0.0, 0.0, 10.0])


def test_tx_sweep_positions_walk_the_endpoints_at_constant_height():
    pos = tx_sweep_positions(_TX0, _RX0 + _RX_STEP, (-100.0, 32.0), 30)
    assert pos.shape == (30, 3)
    lat = street_lateral_axis(_RX0 + _RX_STEP, _TX0)
    assert pos[0] == pytest.approx(_TX0 - 100.0 * lat)
    assert pos[-1] == pytest.approx(_TX0 + 32.0 * lat)
    assert np.allclose(pos[:, 2], _TX0[2])                  # height untouched
    # Uniform steps of (32 - -100)/29 m along the lateral axis.
    steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    assert np.allclose(steps, 132.0 / 29.0)


def test_tx_lateral_sweep_moves_the_azimuth_with_a_fixed_attitude():
    """The transmitter-motion option: with the receiver's attitude fixed as the shipped
    file has it (aimed once, 35 deg offset), walking the transmitter -100 -> +32 m across
    the LoS sweeps the arrival azimuth monotonically by ~50 deg. -100/+32 m is the window
    in which the direct path survives in this scene (measured 2026-09-24, real solves)."""
    orientation = _look_at_orientation(_RX0 + _RX_STEP, _TX0, 35.0)
    tx_pos = tx_sweep_positions(_TX0, _RX0 + _RX_STEP, (-100.0, 32.0), 30)
    az = []
    for i in range(30):
        rx = _RX0 + (i + 1) * _RX_STEP
        az.append(np.degrees(np.arcsin(np.clip(
            boresight_sin_az(rx, tx_pos[i], orientation), -1, 1))))
    az = np.array(az)
    assert (np.diff(az) > 0).all()
    assert az.max() - az.min() == pytest.approx(51.0, abs=4.0)


# ------------------------------------------------------- the per-frame LoS receipt
# `_los_receipt` decides `meta["frames"][i]["los_present"]`, i.e. the claim "the direct
# path is in every frame". It reads a Sionna `Paths` object, but only through `.tau` and
# `.a`, so a duck-typed stand-in exercises it with no Sionna and with delays chosen to
# put the answer beyond doubt.

_C_MPS = 299792458.0


class _StubDrJitArray:
    def __init__(self, arr):
        self._arr = np.asarray(arr)
        self.shape = self._arr.shape

    def numpy(self):
        return self._arr


class _StubPaths:
    """Minimal stand-in for `sionna.rt.Paths`: `tau` in SECONDS (invalid paths <= 0, as
    Sionna marks them) and `a` as the `(real, imag)` pair sionna-rt 1.2.2 returns."""

    def __init__(self, tau_s, amp=None):
        self.tau = _StubDrJitArray(tau_s)
        if amp is None:
            self.a = None
        else:
            amp = np.asarray(amp, dtype=complex)
            self.a = (_StubDrJitArray(amp.real), _StubDrJitArray(amp.imag))


def _pair_at(distance_m, extra_ns=0.0):
    """rx/tx positions `distance_m` apart, and the delay of a path that arrives
    `extra_ns` after the geometric direct one."""
    rx = np.array([0.0, 0.0, 0.0])
    tx = np.array([distance_m, 0.0, 0.0])
    tau_s = distance_m / _C_MPS + extra_ns * 1e-9
    return rx, tx, tau_s


def test_los_receipt_sees_the_direct_path_when_the_shortest_delay_matches_geometry():
    rx, tx, tau_s = _pair_at(82.568)
    paths = _StubPaths([tau_s, tau_s * 1.3, -1.0], amp=[3.0, 1.0, 0.0])
    min_ns, direct_ns, present, share = _los_receipt(paths, rx, tx)
    assert present is True
    assert min_ns == pytest.approx(direct_ns, abs=_LOS_DELAY_TOL_NS)
    assert direct_ns == pytest.approx(82.568 / _C_MPS * 1e9)
    # 3^2 / (3^2 + 1^2): invalid (tau <= 0) paths are excluded from the shortest-path
    # search but the share is over the solved power, as the docstring says.
    assert share == pytest.approx(9.0 / 10.0, rel=1e-6)


def test_los_receipt_reports_a_blocked_direct_path():
    """The occluded case measured on this scene: the shortest solved delay sits tens to
    hundreds of ns past d/c, and it carries almost no power."""
    rx, tx, tau_s = _pair_at(90.1, extra_ns=245.0)
    paths = _StubPaths([tau_s, tau_s * 1.1], amp=[0.01, 1.0])
    min_ns, direct_ns, present, share = _los_receipt(paths, rx, tx)
    assert present is False
    assert min_ns - direct_ns == pytest.approx(245.0, abs=1e-3)
    assert share == pytest.approx(1e-4 / (1e-4 + 1.0), rel=1e-6)


def test_los_receipt_handles_a_frame_with_no_valid_paths():
    rx, tx, _tau = _pair_at(50.0)
    min_ns, direct_ns, present, share = _los_receipt(_StubPaths([-1.0, 0.0]), rx, tx)
    assert (min_ns, present, share) == (None, False, None)
    assert direct_ns == pytest.approx(50.0 / _C_MPS * 1e9)


def test_path_power_share_returns_none_instead_of_raising_on_an_unreadable_tensor():
    """A receipt must never fail a multi-hour generation: an unexpected `a` layout gives
    None, not an exception."""
    rx, tx, tau_s = _pair_at(60.0)
    paths = _StubPaths([tau_s])          # a is None -> unreadable
    assert _path_power_share(paths, 0) is None
    min_ns, _direct, present, share = _los_receipt(paths, rx, tx)
    assert present is True and share is None


def test_los_receipt_tolerance_is_tight_enough_to_reject_a_one_metre_error():
    """The 0.3 ns tolerance is 9 cm of path length: a path a metre longer than the direct
    one must NOT be accepted as the LoS."""
    rx, tx, tau_s = _pair_at(82.568, extra_ns=1.0 / _C_MPS * 1e9)
    assert _los_receipt(_StubPaths([tau_s], amp=[1.0]), rx, tx)[2] is False


def test_both_sweeps_together_compose_the_two_schedules():
    """`--los-sweep-deg` and `--tx-lateral-m` are independent schedules and may be given
    together: the transmitter walks AND the array is re-aimed at wherever it now is, so
    the commanded azimuth still lands (within the elevation-coupling tolerance) while the
    range follows the transmitter."""
    offs = los_sweep_offsets(30, (-30.0, 30.0))
    tx_pos = tx_sweep_positions(_TX0, _RX0 + _RX_STEP, (-100.0, 32.0), 30)
    sins, ranges = [], []
    for i in range(30):
        rx = _RX0 + (i + 1) * _RX_STEP
        orientation = _look_at_orientation(rx, tx_pos[i], offs[i])   # re-aimed at the NEW tx
        sins.append(boresight_sin_az(rx, tx_pos[i], orientation))
        ranges.append(float(np.linalg.norm(tx_pos[i] - rx)))
    assert np.allclose(sins, np.sin(np.radians(offs)), atol=0.05)
    # The transmitter's own walk shows up as range, not as azimuth.
    assert max(ranges) - min(ranges) > 20.0


# --------------------------------------------------------------------------- writer/reader


def _synthetic_meta(carrier_hz=30e9, start_hz=28.5e9, stop_hz=31.5e9, num_freqs=8):
    return {
        "version": 2,
        "scenario_name": "munich",
        "scene": "munich",
        "carrier_hz": carrier_hz,
        "freq_plan": {"carrier_hz": carrier_hz, "start_hz": start_hz, "stop_hz": stop_hz,
                     "num_freqs": num_freqs},
        "rx_spacing_m": 0.5 * 299792458.0 / carrier_hz,
        "aperture_m": 31 * 0.5 * 299792458.0 / carrier_hz,
        "normalize": True,
        "normalize_delays": True,
        "sionna_version": "test",
        "git_head": None,
        "generated_at": "2026-09-23T00:00:00Z",
        "seed": 41,
        "links": {"munich": {"tx_node": "tx", "rx_node": "rx", "rx_array_shape": [32, 32],
                             "n_tx_ant": 1, "kind": "radar", "tx_power_dbm": None,
                             "physical_scale": False}},
    }


def test_write_payload_round_trips_through_sionna_iterator(tmp_path):
    r = np.random.default_rng(0)
    n_frames, n_rx, n_freqs = 2, 4, 8
    arr = (r.standard_normal((n_frames, n_rx, 1, 1, n_freqs))
          + 1j * r.standard_normal((n_frames, n_rx, 1, 1, n_freqs))).astype(np.complex64)
    meta = _synthetic_meta(num_freqs=n_freqs)
    meta["links"]["munich"]["rx_array_shape"] = [2, 2]  # matches n_rx=4 for this synthetic test

    out_path = tmp_path / "munich_ka_test.pkl"
    write_payload(arr, meta, str(out_path))

    it = SionnaIterator(str(out_path))
    assert len(it) == n_frames
    np.testing.assert_array_equal(np.asarray(it[0]), arr[0])
    assert it.link == "munich"
    assert it.freq_plan == meta["freq_plan"]
    assert it.freq_plan is not None
    assert it.rx_array_shape == (2, 2)
    assert it.physical_scale is False
    assert it.meta["carrier_hz"] == 30e9
    assert it.meta["rx_spacing_m"] == pytest.approx(0.5 * 299792458.0 / 30e9)
    assert it.meta["normalize"] is True
    assert it.meta["normalize_delays"] is True


def test_write_payload_creates_output_directory(tmp_path):
    arr = np.zeros((1, 4, 1, 1, 4), dtype=np.complex64)
    meta = _synthetic_meta(num_freqs=4)
    out_path = tmp_path / "nested" / "munich_ka_test.pkl"
    write_payload(arr, meta, str(out_path))
    assert out_path.exists()
    with open(out_path, "rb") as f:
        payload = pickle.load(f)
    assert set(payload.keys()) == {"meta", "links"}
    assert set(payload["links"].keys()) == {"munich"}


# --------------------------------------------------------------------------- registry
#
# `SionnaMunichIterator`'s default (non-legacy-link) branch consults ONLY the plain
# `SIONNA_MUNICH_PATH` module attribute -- never a live ka-vs-legacy existence check --
# because `tests/test_blocks.py` monkeypatches that exact attribute to point at a temp
# multi-link pkl and relies on it being the sole authority (an earlier version of this
# file re-checked `SIONNA_MUNICH_KA_PATH.exists()` at call time, which silently loaded
# the real generated munich_ka.pkl over the monkeypatched path and broke that contract).
# So the ka-preferred/legacy-fallback RESOLUTION LOGIC is tested directly against the
# pure `_resolve_munich_default_path` helper, and the FACTORY's use of the resulting
# `SIONNA_MUNICH_PATH` attribute (plus the legacy-link override) is tested by
# monkeypatching that attribute -- exactly as tests/test_blocks.py does.


def test_resolve_munich_default_path_prefers_ka_when_present(tmp_path):
    from e2e.environment.sionna_iterator import _resolve_munich_default_path

    ka_path = tmp_path / "munich_ka.pkl"
    ka_path.write_bytes(b"x")
    legacy_path = tmp_path / "munich.pkl"  # deliberately absent
    resolved = _resolve_munich_default_path(str(ka_path), str(legacy_path))
    assert resolved == str(ka_path)


def test_resolve_munich_default_path_falls_back_to_legacy_when_ka_absent(tmp_path):
    from e2e.environment.sionna_iterator import _resolve_munich_default_path

    ka_path = tmp_path / "does_not_exist.pkl"
    legacy_path = tmp_path / "munich.pkl"
    legacy_path.write_bytes(b"x")
    resolved = _resolve_munich_default_path(str(ka_path), str(legacy_path))
    assert resolved == str(legacy_path)


def test_munich_iterator_default_uses_sionna_munich_path(tmp_path, monkeypatch):
    """SionnaMunichIterator(link=None) must consult SIONNA_MUNICH_PATH directly -- the
    same attribute tests/test_blocks.py monkeypatches for its own multi-link fixtures."""
    from e2e.environment import sionna_iterator as si

    arr = np.zeros((1, 4, 1, 1, 4), dtype=np.complex64)
    meta = _synthetic_meta(num_freqs=4)
    meta["links"]["munich"]["rx_array_shape"] = [2, 2]
    ka_path = tmp_path / "munich_ka.pkl"
    write_payload(arr, meta, str(ka_path))

    monkeypatch.setattr(si, "SIONNA_MUNICH_PATH", str(ka_path))

    it = si.SionnaMunichIterator()
    assert it.freq_plan is not None
    np.testing.assert_array_equal(np.asarray(it[0]), arr[0])


def test_munich_iterator_legacy_link_selects_legacy_file_regardless_of_default_path(
        tmp_path, monkeypatch):
    """link=MUNICH_LEGACY_LINK always selects SIONNA_MUNICH_LEGACY_PATH, independent of
    whatever SIONNA_MUNICH_PATH currently resolves to."""
    from e2e.environment import sionna_iterator as si

    ka_path = tmp_path / "munich_ka.pkl"
    arr = np.zeros((1, 4, 1, 1, 4), dtype=np.complex64)
    meta = _synthetic_meta(num_freqs=4)
    meta["links"]["munich"]["rx_array_shape"] = [2, 2]
    write_payload(arr, meta, str(ka_path))

    legacy_path = tmp_path / "munich.pkl"
    legacy_arr = np.ones((1, 4, 1, 1, 4), dtype=np.complex64)
    with open(legacy_path, "wb") as f:
        pickle.dump(legacy_arr, f)

    monkeypatch.setattr(si, "SIONNA_MUNICH_PATH", str(ka_path))
    monkeypatch.setattr(si, "SIONNA_MUNICH_LEGACY_PATH", str(legacy_path))

    it = si.SionnaMunichIterator(link=si.MUNICH_LEGACY_LINK)
    assert it.freq_plan is None  # legacy pkl carries no metadata
    np.testing.assert_array_equal(np.asarray(it[0]), legacy_arr[0])


# --------------------------------------------------------------------------- real Sionna


@pytest.mark.sionna
def test_built_scene_frequency_and_spacing_match_carrier():
    """Real Sionna: the scene actually solved has frequency == carrier, and the rx
    array spacing is 0.5 wavelengths at that carrier -- the exact defect F93 found
    (frequency assigned to a scene later discarded; array built at the wrong frequency)."""
    carrier_hz = 30e9
    scene, tx, rx, wavelength, rx_spacing_m, aperture_m = build_scene(carrier_hz)

    assert float(scene.frequency.numpy()[0]) == pytest.approx(carrier_hz)
    expected_wavelength = 299792458.0 / carrier_hz
    expected_spacing = 0.5 * expected_wavelength
    assert wavelength == pytest.approx(expected_wavelength, rel=1e-6)
    assert rx_spacing_m == pytest.approx(expected_spacing, rel=1e-6)


@pytest.mark.sionna
def test_build_scene_boresight_offset_gives_expected_sign_and_order_of_magnitude():
    """Real Sionna: --boresight-offset-deg 35 must move the direct path to a POSITIVE
    sin(az) of roughly 0.5-0.6 (F94 target ~sin(35deg)=0.57; exact value differs because
    of this scene's nonzero elevation -- see boresight_sin_az)."""
    scene, tx, rx, wavelength, rx_spacing_m, aperture_m = build_scene(
        30e9, boresight_offset_deg=35.0)

    orientation_rad = tuple(float(c.numpy()[0]) for c in
                            (rx.orientation.x, rx.orientation.y, rx.orientation.z))
    sin_az = boresight_sin_az(np.asarray(rx.position.numpy()).reshape(3),
                              np.asarray(tx.position.numpy()).reshape(3), orientation_rad)
    assert 0.3 < sin_az < 0.8


@pytest.mark.sionna
def test_generate_produces_expected_frame_shape():
    """One frame at a small num_freqs -- shape must stay (n_frames, 1024, 1, 1, F), the
    legacy layout `SionnaEnvironmentBlock`/`SionnaIterator` expect."""
    args = parse_args(["--carrier-hz", "30e9", "--num-freqs", "4", "--num-frames", "1"])
    all_s_pars, meta = generate(args)

    assert meta["carrier_hz"] == 30e9
    expected_spacing = 0.5 * 299792458.0 / 30e9
    assert meta["rx_spacing_m"] == pytest.approx(expected_spacing, rel=1e-6)
    assert all_s_pars.shape == (1, 1024, 1, 1, 4)
