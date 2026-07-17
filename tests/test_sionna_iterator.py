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
