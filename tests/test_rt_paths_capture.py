"""Tests for the `--store-paths` capture hook.

Three things, none needing Sionna:
  * `rt_signal_chain._fill_rt_paths_capture` -- the out-parameter dict `rt_cfr_frame`
    fills when given a `capture=` dict, against a stub `Paths`/`RTScene` (mirrors
    `tests/test_rt_coherent.py`'s `_stub_rt_scene`, extended with `a`/`doppler` since
    the diffuse term needs them too).
  * `RTEnvironmentBlock.get_S_pars()`/`get_state_updates()` wiring that capture into
    `e2e.ml.blocks.PATHS_CAPTURE_KEY` -- same monkeypatch pattern as
    `test_rt_coherent.py::test_rt_environment_block_forwards_the_fix_to_rt_cfr_frame`.
  * `e2e.ml.rebuild_manifest.group_frames`'s glob fix: a `.paths.npz`/`.cfr.npy`
    sidecar sitting next to real sample frames must not be mistaken for one (or
    raise, per the pre-fix regex-mismatch behaviour).

The real end-to-end RT path (a genuine Sionna solve producing this dict) is exercised
manually against a 2-scene corpus (`RUN_SIONNA=1`, see the shard report) rather than in
this gated-by-default file, to keep the module fast and GPU-free by default.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from e2e.environment import rt_signal_chain as rsc
from e2e.radar_config import RadarConfig
from e2e.scenario import (ArrayConfig, Motion, Node, NodeRole, ObjectKind, Scenario,
                          SceneObject)

_C = 299792458.0

_CFG = RadarConfig(
    name="test_paths_capture_cfg", f0_hz=77e9, bandwidth_hz=1e9, n_tx=2, n_rx=4,
    n_chirps=6, n_samples=8, fs_hz=5e6, chirp_period_s=20e-6, mimo="tdm",
)


def _scenario(position=(12.0, 3.0, 0.5)):
    return Scenario(
        name="paths_capture", base_scene="flat", num_frames=1,
        nodes=[Node(name="radar", role=NodeRole.RADAR, position=(0.0, 0.0, 1.5),
                    look_at=(1.0, 0.0, 1.5), array=ArrayConfig(num_rows=1, num_cols=1))],
        objects=[SceneObject(name="t", kind=ObjectKind.SPHERE, position=position,
                             scaling=0.5, material="metal", object_class="vehicle",
                             rcs_dbsm=10.0, motion=Motion(),
                             velocity_mps=(0.0, 0.0, 0.0))],
    )


class _FakeDrJitArray:
    """Minimal stand-in for a DrJit array: only `.numpy()`/`.shape` are touched by
    `_fill_rt_paths_capture` (same contract `test_rt_coherent.py` relies on)."""

    def __init__(self, arr):
        self._a = np.asarray(arr)

    def numpy(self):
        return self._a

    @property
    def shape(self):
        return self._a.shape


def _stub_paths_and_scene(cfg, centre_world, *, n_paths=5, object_id=7,
                          half=(0.5, 0.5, 0.5), antenna_pattern="tr38901"):
    """A `Paths`/`RTScene` pair carrying every array `_fill_rt_paths_capture` reads:
    `a`/`tau`/`doppler` (diffuse term, `cfr_from_paths`), `objects`/`vertices`
    (coherent term's first-interaction slice, `_rt_phase_centres`), and one placed
    object exposing a world-space AABB (`_object_bbox`)."""
    c = np.asarray(centre_world, dtype=float)
    h = np.asarray(half, dtype=float)
    bbox = types.SimpleNamespace(min=list(c - h), max=list(c + h))
    so = types.SimpleNamespace(object_id=object_id,
                               mi_mesh=types.SimpleNamespace(bbox=lambda: bbox))
    rt_scene = types.SimpleNamespace(scene=None, objects={"t": so}, cfg=cfg,
                                     antenna_pattern=antenna_pattern)

    shape = (1, int(cfg.n_rx), 1, int(cfg.n_tx), n_paths)
    a_re = np.full(shape, 1.0, dtype=np.float32)
    a_im = np.full(shape, 0.5, dtype=np.float32)
    tau = np.full(shape, float(np.linalg.norm(c) * 2.0 / _C), dtype=np.float32)
    doppler = np.zeros(shape, dtype=np.float32)
    paths = types.SimpleNamespace(
        a=(_FakeDrJitArray(a_re), _FakeDrJitArray(a_im)),
        tau=_FakeDrJitArray(tau),
        doppler=_FakeDrJitArray(doppler),
        objects=_FakeDrJitArray(np.full((2,) + shape, object_id)),
        vertices=_FakeDrJitArray(np.broadcast_to(c, (2,) + shape + (3,)).copy()),
    )
    return rt_scene, paths


# --------------------------------------------------------------------------------
# _fill_rt_paths_capture: the arrays a re-synthesis needs
# --------------------------------------------------------------------------------
def test_capture_holds_the_diffuse_and_coherent_term_inputs():
    rt_scene, paths = _stub_paths_and_scene(_CFG, (12.0, 3.0, 0.5))
    capture: dict = {}
    rsc._fill_rt_paths_capture(capture, paths, rt_scene, range_migration=True,
                               coherent_targets=True, scattering_coefficient=0.3)

    # Diffuse term (cfr_from_paths' inputs).
    assert capture["a"].dtype == np.complex64
    assert capture["a"].shape == (1, _CFG.n_rx, 1, _CFG.n_tx, 5)
    assert capture["tau"].dtype == np.float32
    assert capture["doppler"].dtype == np.float32
    np.testing.assert_allclose(capture["a"].real, 1.0)
    np.testing.assert_allclose(capture["a"].imag, 0.5)

    # Coherent term's first-interaction slice -- the SAME slice _rt_phase_centres reads.
    assert capture["first_object_id"].shape == (5,)
    assert np.all(capture["first_object_id"] == 7)
    assert capture["first_vertex"].shape == (5, 3)
    np.testing.assert_allclose(capture["first_vertex"][0], [12.0, 3.0, 0.5])
    assert capture["tau_ref"].shape == (5,)

    # Per-object id + world AABB, and the antenna pattern name.
    assert list(capture["object_names"]) == ["t"]
    assert capture["object_ids"].tolist() == [7]
    np.testing.assert_allclose(capture["object_bbox_centre"][0], [12.0, 3.0, 0.5])
    np.testing.assert_allclose(capture["object_bbox_half"][0], [0.5, 0.5, 0.5])
    assert str(capture["antenna_pattern"]) == "tr38901"

    # Scalars the closed-form branch used for this frame.
    assert bool(capture["range_migration"]) is True
    assert bool(capture["coherent_targets"]) is True
    assert float(capture["scattering_coefficient"]) == pytest.approx(0.3)


def test_capture_round_trips_through_the_paths_sidecar(tmp_path):
    """The whole point: `storage.write_paths_sidecar`/`read_paths_sidecar` on exactly
    this dict, unmodified -- the storage half already tested in `test_ml_store_cfr.py`,
    exercised here against the REAL capture contents rather than a hand-rolled stub."""
    from e2e.ml.storage import read_paths_sidecar, write_paths_sidecar

    rt_scene, paths = _stub_paths_and_scene(_CFG, (12.0, 3.0, 0.5))
    capture: dict = {}
    rsc._fill_rt_paths_capture(capture, paths, rt_scene, range_migration=True,
                               coherent_targets=True, scattering_coefficient=0.3)

    npz_path = tmp_path / "sample_frame_00000.npz"
    npz_path.write_bytes(b"")  # write_paths_sidecar only needs the sibling PATH
    name = write_paths_sidecar(npz_path, capture)
    got = read_paths_sidecar(npz_path, {"paths_sidecar": name})

    assert set(got) == set(capture)
    np.testing.assert_array_equal(got["a"], capture["a"])
    np.testing.assert_array_equal(got["object_bbox_centre"], capture["object_bbox_centre"])
    assert str(got["antenna_pattern"]) == "tr38901"


# --------------------------------------------------------------------------------
# rt_cfr_frame(capture=...): inert when None, filled when given
# --------------------------------------------------------------------------------
def test_rt_cfr_frame_capture_none_is_inert(monkeypatch):
    """`capture=None` (the default) must not change `rt_cfr_frame`'s return at all."""
    rt_scene, paths = _stub_paths_and_scene(_CFG, (12.0, 3.0, 0.5))
    monkeypatch.setattr(rsc, "_solve", lambda *a, **k: paths)
    scn = _scenario()

    out_none = rsc.rt_cfr_frame(_CFG, scn, frame_idx=0, rt_scene=rt_scene, device="cpu",
                                coherent_targets=False, capture=None)
    capture: dict = {}
    out_filled = rsc.rt_cfr_frame(_CFG, scn, frame_idx=0, rt_scene=rt_scene, device="cpu",
                                  coherent_targets=False, capture=capture)
    assert torch.equal(out_none, out_filled)
    assert set(capture) >= {
        "a", "tau", "doppler", "object_names", "object_ids", "object_bbox_centre",
        "object_bbox_half", "antenna_pattern", "range_migration", "coherent_targets",
        "scattering_coefficient",
    }
    # scattering_coefficient was not passed -- must resolve to rt_scene_build's default,
    # not surface as an un-resolved None.
    from e2e.environment.rt_scene_build import DEFAULT_SCATTERING_COEFFICIENT
    assert float(capture["scattering_coefficient"]) == pytest.approx(
        DEFAULT_SCATTERING_COEFFICIENT)


# --------------------------------------------------------------------------------
# RTEnvironmentBlock wiring: get_S_pars() -> get_state_updates()[PATHS_CAPTURE_KEY]
# --------------------------------------------------------------------------------
def test_rt_environment_block_emits_the_paths_capture(monkeypatch):
    """`get_S_pars()` is the only seam corpus generation goes through, so the capture
    dict has to arrive there and travel out via `get_state_updates()`. Sionna-free:
    `get_S_pars` resolves `build_rt_scene`/`rt_cfr_frame` from `e2e.environment.rt_gen`
    at call time (same pattern as `test_rt_environment_block_forwards_the_fix_to_rt_cfr_frame`
    in `test_rt_coherent.py`)."""
    import e2e.environment.rt_gen as rt_gen
    from e2e.environment.blocks import RTEnvironmentBlock
    from e2e.ml.blocks import PATHS_CAPTURE_KEY

    def fake_build_rt_scene(scenario, cfg, **kwargs):
        return types.SimpleNamespace(objects={}, scene=None, cfg=cfg)

    def fake_rt_cfr_frame(cfg, scenario, *, capture=None, **kwargs):
        if capture is not None:
            capture["a"] = np.array([1.0, 2.0, 3.0], dtype=np.complex64)
        return torch.zeros((int(cfg.n_rx), int(cfg.n_tx), int(cfg.n_chirps),
                            int(cfg.n_samples)), dtype=torch.complex64)

    monkeypatch.setattr(rt_gen, "build_rt_scene", fake_build_rt_scene)
    monkeypatch.setattr(rt_gen, "rt_cfr_frame", fake_rt_cfr_frame)

    blk = RTEnvironmentBlock(_scenario(), _CFG, device="cpu")
    assert blk.get_state_updates() == {}          # nothing before the first get_S_pars()

    blk.get_S_pars()
    updates = blk.get_state_updates()
    assert PATHS_CAPTURE_KEY in updates
    np.testing.assert_array_equal(updates[PATHS_CAPTURE_KEY]["a"],
                                  np.array([1.0, 2.0, 3.0], dtype=np.complex64))

    blk.reset()
    assert blk.last_rt_paths is None
    assert blk.get_state_updates() == {}


# --------------------------------------------------------------------------------
# rebuild_manifest.group_frames: a sidecar must not be mistaken for a sample frame
# --------------------------------------------------------------------------------
def test_group_frames_skips_paths_and_cfr_sidecars(tmp_path):
    from e2e.ml.rebuild_manifest import group_frames
    from e2e.ml.storage import CFR_SIDECAR_SUFFIX, PATHS_SIDECAR_SUFFIX

    (tmp_path / "sample_scene00000_frame_00000.npz").write_bytes(b"x")
    (tmp_path / "sample_scene00000_frame_00001.npz").write_bytes(b"x")
    # Both sidecar suffixes for frame 0 -- the `.paths.npz` one is what the pre-fix
    # `*.npz` glob would have swept in (it never matches `_FNAME`, so group_frames used
    # to raise ValueError on any corpus generated with --store-paths); `.cfr.npy` never
    # matched `*.npz` in the first place, kept here for symmetry with the fix.
    (tmp_path / f"sample_scene00000_frame_00000{PATHS_SIDECAR_SUFFIX}").write_bytes(b"x")
    (tmp_path / f"sample_scene00000_frame_00000{CFR_SIDECAR_SUFFIX}").write_bytes(b"x")

    scenes = group_frames(tmp_path)
    assert scenes == [["sample_scene00000_frame_00000.npz",
                       "sample_scene00000_frame_00001.npz"]]


def test_group_frames_still_raises_on_a_genuinely_unmatched_npz(tmp_path):
    """The fix must exclude ONLY the two known sidecar suffixes, not loosen the naming
    check generally."""
    from e2e.ml.rebuild_manifest import group_frames

    (tmp_path / "sample_scene00000_frame_00000.npz").write_bytes(b"x")
    (tmp_path / "not_a_corpus_frame.npz").write_bytes(b"x")
    with pytest.raises(ValueError, match="does not match"):
        group_frames(tmp_path)
