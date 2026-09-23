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
    _cfr_chunk_size,
    _synthesize_cfr,
    boresight_sin_az,
    build_frequencies,
    build_scene,
    generate,
    parse_args,
    unambiguous_range_m,
    write_payload,
)


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
