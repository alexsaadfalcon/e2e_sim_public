"""Tests for the precomputed-frame loader (e2e.environment.sionna_iterator)."""

import os
import pickle

import numpy as np
import pytest

from e2e.environment.sionna_iterator import SionnaIterator


def test_iterator_len_and_getitem(tmp_pkl_frames):
    path, arr = tmp_pkl_frames(n_frames=5, n_freqs=8)
    it = SionnaIterator(path)
    assert len(it) == 5
    np.testing.assert_array_equal(np.asarray(it[0]), arr[0])
    np.testing.assert_array_equal(np.asarray(it[4]), arr[4])


def test_iterator_iterates_all_frames(tmp_pkl_frames):
    path, arr = tmp_pkl_frames(n_frames=3, n_freqs=8)
    frames = list(SionnaIterator(path))
    assert len(frames) == 3
    for got, want in zip(frames, arr):
        np.testing.assert_array_equal(np.asarray(got), want)


def test_iterator_frame_shape_matches_pipeline_expectation(tmp_pkl_frames):
    # Per-frame layout must be (n_rx, 1, 1, F) so the pipeline can view it as (32,32,1,F).
    path, _ = tmp_pkl_frames(n_frames=2, n_rx=1024, n_freqs=16)
    frame = np.asarray(SionnaIterator(path)[0])
    assert frame.shape == (1024, 1, 1, 16)
    assert frame.reshape(32, 32, 1, 16).shape == (32, 32, 1, 16)


# --------------------------------------------------------------------------- multi-link


def _write_multilink_pkl(tmp_path, links, n_frames=3, n_freqs=8, n_rx=4):
    """Write a dict {link_name: frames_array} pkl and return (path, {name: array})."""
    r = np.random.default_rng(0)
    data = {}
    for i, name in enumerate(links):
        arr = (r.standard_normal((n_frames, n_rx, 1, 1, n_freqs))
               + 1j * r.standard_normal((n_frames, n_rx, 1, 1, n_freqs))).astype(np.complex64)
        data[name] = arr
    path = tmp_path / "multilink.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return str(path), data


def test_iterator_multilink_default_first_link(tmp_path):
    path, data = _write_multilink_pkl(tmp_path, ["tx0", "tx1"])
    it = SionnaIterator(path)
    assert it.links == ["tx0", "tx1"]
    assert it.link == "tx0"
    np.testing.assert_array_equal(np.asarray(it[0]), data["tx0"][0])


def test_iterator_multilink_explicit_link(tmp_path):
    path, data = _write_multilink_pkl(tmp_path, ["tx0", "tx1"])
    it = SionnaIterator(path, link="tx1")
    assert it.link == "tx1"
    np.testing.assert_array_equal(np.asarray(it[0]), data["tx1"][0])


def test_iterator_multilink_unknown_link_raises(tmp_path):
    path, _ = _write_multilink_pkl(tmp_path, ["tx0", "tx1"])
    with pytest.raises(KeyError):
        SionnaIterator(path, link="nope")


def test_iterator_single_array_ignores_link(tmp_pkl_frames):
    path, arr = tmp_pkl_frames(n_frames=3, n_freqs=8)
    # link is ignored for legacy single-array pkls.
    it = SionnaIterator(path, link="anything")
    assert it.links is None
    assert it.link is None
    np.testing.assert_array_equal(np.asarray(it[0]), arr[0])


def test_available_links(tmp_path, tmp_pkl_frames):
    path, _ = _write_multilink_pkl(tmp_path, ["a", "b", "c"])
    assert SionnaIterator.available_links(path) == ["a", "b", "c"]
    single_path, _ = tmp_pkl_frames(n_frames=2, n_freqs=8)
    assert SionnaIterator.available_links(single_path) is None


# --------------------------------------------------------------------------- handle release

def test_constructing_iterator_releases_file_handle(tmp_path, tmp_pkl_frames):
    """SionnaIterator must close the .pkl after loading (no leaked handle).

    On Windows a leaked read handle keeps a lock on the file, so overwriting the same
    path (e.g. regenerating frames) fails. Constructing the iterator and then rewriting
    + deleting the file proves the handle was released.
    """
    path, _ = tmp_pkl_frames(n_frames=2, n_freqs=8)
    SionnaIterator(path)  # loads and (with the fix) closes the handle
    # If the handle leaked, these would raise PermissionError on Windows.
    with open(path, "wb") as f:
        pickle.dump(np.zeros((1, 4, 1, 1, 8), dtype=np.complex64), f)
    os.remove(path)
    assert not os.path.exists(path)


def test_available_links_releases_file_handle(tmp_path):
    """available_links must also close the .pkl after loading."""
    path, _ = _write_multilink_pkl(tmp_path, ["a", "b"])
    SionnaIterator.available_links(path)
    with open(path, "wb") as f:
        pickle.dump({"a": np.zeros((1, 4, 1, 1, 8), dtype=np.complex64)}, f)
    os.remove(path)
    assert not os.path.exists(path)


# --------------------------------------------------------------------------- v2 payload


def _write_v2_pkl(tmp_path, links, n_frames=3, n_freqs=8, n_rx=4, name="v2.pkl",
                   scenario_name="etoile"):
    """Write a self-describing v2 payload per the pinned format contract."""
    r = np.random.default_rng(0)
    links_arrays = {}
    links_meta = {}
    for name_ in links:
        arr = (r.standard_normal((n_frames, n_rx, 1, 1, n_freqs))
               + 1j * r.standard_normal((n_frames, n_rx, 1, 1, n_freqs))).astype(np.complex64)
        links_arrays[name_] = arr
        links_meta[name_] = {
            "tx_node": f"{name_}_tx",
            "rx_node": f"{name_}_rx",
            "rx_array_shape": [2, 2],
            "n_tx_ant": 1,
            "kind": "radar",
            "tx_power_dbm": 20.0,
            "physical_scale": True,
        }
    payload = {
        "meta": {
            "version": 2,
            "scenario_name": scenario_name,
            "freq_plan": {
                "carrier_hz": 3.5e9,
                "start_hz": 3.4e9,
                "stop_hz": 3.6e9,
                "num_freqs": n_freqs,
            },
            "links": links_meta,
        },
        "links": links_arrays,
    }
    path = tmp_path / name
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return str(path), links_arrays, links_meta


def test_iterator_v2_default_first_link(tmp_path):
    path, arrays, links_meta = _write_v2_pkl(tmp_path, ["radar0", "comm0"])
    it = SionnaIterator(path)
    assert it.links == ["radar0", "comm0"]
    assert it.link == "radar0"
    np.testing.assert_array_equal(np.asarray(it[0]), arrays["radar0"][0])


def test_iterator_v2_explicit_link(tmp_path):
    path, arrays, links_meta = _write_v2_pkl(tmp_path, ["radar0", "comm0"])
    it = SionnaIterator(path, link="comm0")
    assert it.link == "comm0"
    np.testing.assert_array_equal(np.asarray(it[0]), arrays["comm0"][0])


def test_iterator_v2_unknown_link_raises(tmp_path):
    path, _, _ = _write_v2_pkl(tmp_path, ["radar0", "comm0"])
    with pytest.raises(KeyError):
        SionnaIterator(path, link="nope")


def test_iterator_v2_meta_and_link_meta_populated(tmp_path):
    path, _, links_meta = _write_v2_pkl(tmp_path, ["radar0", "comm0"])
    it = SionnaIterator(path, link="comm0")
    assert it.meta["version"] == 2
    assert it.meta["scenario_name"] == "etoile"
    assert it.link_meta == links_meta["comm0"]


def test_iterator_v2_convenience_properties(tmp_path):
    path, _, _ = _write_v2_pkl(tmp_path, ["radar0"])
    it = SionnaIterator(path)
    assert it.freq_plan == {
        "carrier_hz": 3.5e9,
        "start_hz": 3.4e9,
        "stop_hz": 3.6e9,
        "num_freqs": 8,
    }
    assert it.rx_array_shape == (2, 2)
    assert it.physical_scale is True


def test_iterator_legacy_meta_properties_are_none(tmp_pkl_frames, tmp_path):
    # Legacy single-array pkl.
    path, _ = tmp_pkl_frames(n_frames=2, n_freqs=8)
    it = SionnaIterator(path)
    assert it.meta is None
    assert it.link_meta is None
    assert it.freq_plan is None
    assert it.rx_array_shape is None
    assert it.physical_scale is None

    # Legacy multi-link pkl.
    ml_path, _ = _write_multilink_pkl(tmp_path, ["tx0", "tx1"])
    it_ml = SionnaIterator(ml_path)
    assert it_ml.meta is None
    assert it_ml.link_meta is None
    assert it_ml.freq_plan is None
    assert it_ml.rx_array_shape is None
    assert it_ml.physical_scale is None


def test_available_links_v2(tmp_path):
    path, _, _ = _write_v2_pkl(tmp_path, ["radar0", "comm0", "comm1"])
    assert SionnaIterator.available_links(path) == ["radar0", "comm0", "comm1"]
