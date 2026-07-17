"""
Shared pytest fixtures and collection config for the e2e simulator test suite.

Design goals:
- The default `pytest` run is fully hands-off: no Sionna ray tracing, no display, no
  network, no reliance on the precomputed `.pkl` frames existing on disk. Everything uses
  small synthetic data so it is fast and deterministic.
- Tests that genuinely need hardware/human input are gated behind markers and skipped
  unless the corresponding env var is set:
      RUN_SIONNA=1  -> run @pytest.mark.sionna  (real Sionna RT generation)
      RUN_SLOW=1    -> run @pytest.mark.slow     (SNR sweeps, etc.)
      RUN_GUI=1     -> run @pytest.mark.gui      (live server / display)

Fixtures provided to all test modules (do NOT redefine these in test files):
- `n_freqs`              -> small default frequency-bin count for fast tests
- `synthetic_frame_np`   -> factory: one S-parameter frame, np.complex64, shape (n_rx,1,1,F)
- `synthetic_frames_np`  -> factory: stack of frames, shape (n_frames,n_rx,1,1,F)
- `tmp_pkl_frames`       -> factory: writes synthetic frames to a temp .pkl, returns (path, array)
- `make_env_block`       -> factory: a drop-in environment block (get_S_pars/step/reset) for Simulation
- `small_scenario`       -> a tiny valid Scenario (few frames/freqs) for fast scenario tests
"""

import os
import pickle

import numpy as np
import pytest


# Conventional array geometry the runtime pipeline assumes (32x32 -> 1024 elements).
N_RX = 1024
DEFAULT_F = 64


def _rng(seed=0):
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------- markers

def pytest_collection_modifyitems(config, items):
    """Auto-skip hardware/slow/gui tests unless explicitly opted in via env vars."""
    gates = {
        "sionna": (os.environ.get("RUN_SIONNA") == "1", "needs Sionna RT; set RUN_SIONNA=1"),
        "slow": (os.environ.get("RUN_SLOW") == "1", "slow test; set RUN_SLOW=1"),
        "gui": (os.environ.get("RUN_GUI") == "1", "needs display/server; set RUN_GUI=1"),
    }
    for item in items:
        for marker, (enabled, reason) in gates.items():
            if marker in item.keywords and not enabled:
                item.add_marker(pytest.mark.skip(reason=reason))


# --------------------------------------------------------------------------- data fixtures

@pytest.fixture
def n_freqs():
    return DEFAULT_F


@pytest.fixture
def synthetic_frame_np():
    """Factory for a single synthetic S-parameter frame, shape (n_rx, 1, 1, F), complex64.

    Matches the per-frame layout produced by the Sionna generation scripts
    (`cfr[0, :, 0, :, :, :]`), which the runtime pipeline reshapes to (32, 32, 1, F).
    """
    def _make(n_rx=N_RX, n_freqs=DEFAULT_F, seed=0):
        r = _rng(seed)
        arr = r.standard_normal((n_rx, 1, 1, n_freqs)) + 1j * r.standard_normal((n_rx, 1, 1, n_freqs))
        return arr.astype(np.complex64)
    return _make


@pytest.fixture
def synthetic_frames_np(synthetic_frame_np):
    """Factory for a stack of synthetic frames, shape (n_frames, n_rx, 1, 1, F)."""
    def _make(n_frames=4, n_rx=N_RX, n_freqs=DEFAULT_F, seed=0):
        return np.stack([synthetic_frame_np(n_rx, n_freqs, seed + i) for i in range(n_frames)], axis=0)
    return _make


@pytest.fixture
def tmp_pkl_frames(tmp_path, synthetic_frames_np):
    """Factory that writes synthetic frames to a temp .pkl (the format SionnaIterator expects).

    Returns (path_str, ndarray).
    """
    def _make(n_frames=4, n_rx=N_RX, n_freqs=DEFAULT_F, seed=0, name="frames.pkl"):
        arr = synthetic_frames_np(n_frames, n_rx, n_freqs, seed)
        path = tmp_path / name
        with open(path, "wb") as f:
            pickle.dump(arr, f)
        return str(path), arr
    return _make


@pytest.fixture
def torch_device():
    """The device the library is configured to use (cuda if available, else cpu).

    Tests place their tensors on this device so the suite is green both on CPU-only CI
    and on CUDA dev machines, and exercises the same device path as production.
    """
    pytest.importorskip("torch")
    from e2e.blocks import device
    return device


@pytest.fixture
def make_env_block(synthetic_frames_np):
    """Factory for a drop-in environment block usable as Simulation's environment_block.

    Mirrors SionnaEnvironmentBlock's interface (get_S_pars/step/reset + frame_counter)
    but serves synthetic frames on the library device, so Simulation can be exercised
    without any .pkl or Sionna.
    """
    torch = pytest.importorskip("torch")
    from e2e.blocks import device

    class _DummyEnvBlock:
        def __init__(self, frames, array_shape):
            self._frames = frames
            self.frame_counter = 0
            # advertise geometry so Simulation can auto-derive (n_rx_x, n_rx_y)
            self.array_shape = array_shape

        def __len__(self):
            return len(self._frames)

        def step(self):
            self.frame_counter = (self.frame_counter + 1) % len(self._frames)

        def reset(self):
            self.frame_counter = 0

        def get_S_pars(self):
            arr = np.ascontiguousarray(self._frames[self.frame_counter])
            return torch.from_numpy(arr).to(device)

    def _make(n_frames=4, n_freqs=DEFAULT_F, seed=0, n_rx=N_RX, array_shape=(32, 32)):
        assert array_shape[0] * array_shape[1] == n_rx, "array_shape must factor n_rx"
        return _DummyEnvBlock(synthetic_frames_np(n_frames, n_rx, n_freqs, seed), array_shape)

    return _make


@pytest.fixture
def small_scenario():
    """A tiny but valid Scenario for fast scenario/runner tests."""
    from e2e.scenario import munich_radar_scenario
    sc = munich_radar_scenario()
    sc.num_frames = 3
    sc.frequency.num_freqs = 16
    return sc
