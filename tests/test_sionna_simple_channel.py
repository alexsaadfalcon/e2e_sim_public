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
    build_frequencies,
    build_scene,
    generate,
    parse_args,
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


def test_munich_iterator_prefers_ka_file_when_present(tmp_path, monkeypatch):
    from e2e.environment import sionna_iterator as si

    arr = np.zeros((1, 4, 1, 1, 4), dtype=np.complex64)
    meta = _synthetic_meta(num_freqs=4)
    meta["links"]["munich"]["rx_array_shape"] = [2, 2]
    ka_path = tmp_path / "munich_ka.pkl"
    write_payload(arr, meta, str(ka_path))

    legacy_path = tmp_path / "munich.pkl"
    legacy_arr = np.ones((1, 4, 1, 1, 4), dtype=np.complex64)
    with open(legacy_path, "wb") as f:
        pickle.dump(legacy_arr, f)

    monkeypatch.setattr(si, "SIONNA_MUNICH_KA_PATH", str(ka_path))
    monkeypatch.setattr(si, "SIONNA_MUNICH_LEGACY_PATH", str(legacy_path))

    it = si.SionnaMunichIterator()
    assert it.freq_plan is not None
    np.testing.assert_array_equal(np.asarray(it[0]), arr[0])


def test_munich_iterator_legacy_link_selects_legacy_file(tmp_path, monkeypatch):
    from e2e.environment import sionna_iterator as si

    arr = np.zeros((1, 4, 1, 1, 4), dtype=np.complex64)
    meta = _synthetic_meta(num_freqs=4)
    meta["links"]["munich"]["rx_array_shape"] = [2, 2]
    ka_path = tmp_path / "munich_ka.pkl"
    write_payload(arr, meta, str(ka_path))

    legacy_path = tmp_path / "munich.pkl"
    legacy_arr = np.ones((1, 4, 1, 1, 4), dtype=np.complex64)
    with open(legacy_path, "wb") as f:
        pickle.dump(legacy_arr, f)

    monkeypatch.setattr(si, "SIONNA_MUNICH_KA_PATH", str(ka_path))
    monkeypatch.setattr(si, "SIONNA_MUNICH_LEGACY_PATH", str(legacy_path))

    it = si.SionnaMunichIterator(link=si.MUNICH_LEGACY_LINK)
    assert it.freq_plan is None  # legacy pkl carries no metadata
    np.testing.assert_array_equal(np.asarray(it[0]), legacy_arr[0])


def test_munich_iterator_falls_back_to_legacy_when_ka_absent(tmp_path, monkeypatch):
    from e2e.environment import sionna_iterator as si

    legacy_path = tmp_path / "munich.pkl"
    legacy_arr = np.ones((1, 4, 1, 1, 4), dtype=np.complex64)
    with open(legacy_path, "wb") as f:
        pickle.dump(legacy_arr, f)

    monkeypatch.setattr(si, "SIONNA_MUNICH_KA_PATH", str(tmp_path / "does_not_exist.pkl"))
    monkeypatch.setattr(si, "SIONNA_MUNICH_LEGACY_PATH", str(legacy_path))

    it = si.SionnaMunichIterator()
    assert it.freq_plan is None
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
def test_generate_produces_expected_frame_shape():
    """One frame at a small num_freqs -- shape must stay (n_frames, 1024, 1, 1, F), the
    legacy layout `SionnaEnvironmentBlock`/`SionnaIterator` expect."""
    args = parse_args(["--carrier-hz", "30e9", "--num-freqs", "4", "--num-frames", "1"])
    all_s_pars, meta = generate(args)

    assert meta["carrier_hz"] == 30e9
    expected_spacing = 0.5 * 299792458.0 / 30e9
    assert meta["rx_spacing_m"] == pytest.approx(expected_spacing, rel=1e-6)
    assert all_s_pars.shape == (1, 1024, 1, 1, 4)
