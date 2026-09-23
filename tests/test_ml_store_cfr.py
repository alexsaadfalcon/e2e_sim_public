"""Tests for the CFR store: corpora that keep the expensive ray-traced channel.

The architecture change (owner-directed 2026-09-23) is that a corpus stores the frame
ENTERING the chain -- `s_pars`, the ray-traced CFR -- next to the ADC cube it produced,
so everything downstream of the ray tracer can be re-run live instead of being frozen
at generation time. Ray tracing is ~11-13 s/scene; the rest of the chain is ~2 s.

Ungated and synthetic: a deterministic CFR-emitting stand-in replaces
`RTEnvironmentBlock` (as in `test_ml_chain_generate.py`), so the whole write ->
read -> re-run loop is exercised in CI in seconds, with no Sionna and no `.pkl`
frames. What is NOT covered here is the real ray-traced corpus; see the acceptance
run recorded in the shard report.
"""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from e2e import frames
from e2e.ml import chain_generate
from e2e.ml.blocks import (CFR_CAPTURE_KEY, PATHS_CAPTURE_KEY, CFRCaptureStage,
                           CorpusSourceBlock, SinkBlock, SourceBlock)
from e2e.ml.dataset import write_manifest
from e2e.ml.labels import LabelGrid
from e2e.radar_config import RadarConfig
from e2e.ml.storage import CFR_SIDECAR_SUFFIX

# Same shape of config as test_ml_chain_generate's: single-TX (sidesteps the known
# RadarCubeBlock TDM blocker) and tiny, so a full chain run costs milliseconds.
_CFG = RadarConfig(
    name="test_store_cfr_cfg", f0_hz=77e9, bandwidth_hz=500e6, n_tx=1, n_rx=4,
    n_chirps=8, n_samples=16, fs_hz=5e6, chirp_period_s=10e-6, mimo="single",
)


class _FakeRTEnvironment:
    """Deterministic CFR-emitting stand-in for `RTEnvironmentBlock` (no Sionna)."""

    def __init__(self, cfg, grid, n_frames=2, device=None, seed=0):
        self.cfg = cfg
        self.grid = grid
        self.n_frames = n_frames
        self.device = device
        self.seed = seed
        self.frame_counter = 0
        self.array_shape = (cfg.n_rx, 1)
        self.last_labels = None

    def step(self):
        self.frame_counter = (self.frame_counter + 1) % self.n_frames

    def reset(self):
        self.frame_counter = 0

    def frame(self, idx):
        """The CFR this environment emits for frame `idx` -- the oracle every
        sidecar assertion below compares against."""
        g = torch.Generator(device="cpu").manual_seed(self.seed + idx)
        shape = (self.cfg.n_rx, self.cfg.n_tx, self.cfg.n_chirps, self.cfg.n_samples)
        s = torch.complex(torch.randn(shape, generator=g),
                          torch.randn(shape, generator=g)).to(torch.complex64)
        return s if self.device is None else s.to(self.device)

    def get_S_pars(self):
        labels = torch.zeros(3, self.grid.n_range, self.grid.n_azimuth)
        labels[0, min(2, self.grid.n_range - 1), min(2, self.grid.n_azimuth - 1)] = 1.0
        self.last_labels = labels
        return self.frame(self.frame_counter)

    def get_state_updates(self):
        return {} if self.last_labels is None else {"labels": self.last_labels}


@pytest.fixture
def grid():
    return LabelGrid.for_config(_CFG, range_stride=1, n_azimuth=8)


@pytest.fixture
def env(grid, torch_device):
    return _FakeRTEnvironment(_CFG, grid, n_frames=2, device=torch_device)


def _run(out_dir, env, *, store_cfr, n=2, use_rffe=True, environment_block=None):
    sim = chain_generate.build_chain_simulation(
        scenario=None, cfg=_CFG, out_dir=out_dir,
        environment_block=environment_block if environment_block is not None else env,
        impairment_seed=7, store_cfr=store_cfr, use_rffe=use_rffe,
    )
    sim.run(n_steps=n)
    return sim


def _meta(npz_path):
    with np.load(npz_path, allow_pickle=False) as data:
        return json.loads(str(data["meta"].item()))


# --------------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------------
def test_sidecar_written_only_when_store_cfr_asked(tmp_path, env):
    off, on = tmp_path / "off", tmp_path / "on"
    _run(off, env, store_cfr=False)
    env.reset()
    _run(on, env, store_cfr=True)

    assert sorted(p.name for p in off.glob("*.npz")) == \
           sorted(p.name for p in on.glob("*.npz")) != []
    assert list(off.glob(f"*{CFR_SIDECAR_SUFFIX}")) == []
    assert len(list(on.glob(f"*{CFR_SIDECAR_SUFFIX}"))) == 2

    # ...and a corpus written without the flag says nothing about a sidecar at all,
    # which is what makes every pre-existing corpus read exactly as before.
    assert "cfr_sidecar" not in _meta(off / "sample_frame_00000.npz")


def test_meta_names_the_sidecar_and_it_holds_the_entering_frame(tmp_path, env):
    _run(tmp_path, env, store_cfr=True)
    for idx in (0, 1):
        npz = tmp_path / f"sample_frame_{idx:05d}.npz"
        meta = _meta(npz)
        assert meta["cfr_sidecar"] == f"sample_frame_{idx:05d}{CFR_SIDECAR_SUFFIX}"
        stored = np.load(tmp_path / meta["cfr_sidecar"])
        expected = env.frame(idx).cpu().numpy()
        assert stored.dtype == np.complex64
        assert stored.shape == (_CFG.n_rx, _CFG.n_tx, _CFG.n_chirps, _CFG.n_samples)
        # Exactly the frame the environment emitted: uncompressed, unquantized,
        # captured before the RFFE (not a lossy or post-front-end copy).
        assert np.array_equal(stored, expected)


def test_storing_the_cfr_does_not_change_the_sample(tmp_path, env):
    """The whole flag must be inert with respect to the corpus itself."""
    off, on = tmp_path / "off", tmp_path / "on"
    _run(off, env, store_cfr=False)
    env.reset()
    _run(on, env, store_cfr=True)

    with np.load(off / "sample_frame_00000.npz") as a, \
            np.load(on / "sample_frame_00000.npz") as b:
        assert set(a.files) == set(b.files)
        for key in a.files:
            if key == "meta":
                continue
            assert np.array_equal(a[key], b[key]), key
        meta_off, meta_on = _meta(off / "sample_frame_00000.npz"), \
            _meta(on / "sample_frame_00000.npz")
        assert {k: v for k, v in meta_on.items() if k != "cfr_sidecar"} == meta_off


def test_sink_without_a_capture_stage_fails_loudly(tmp_path):
    sink = SinkBlock(tmp_path, store_cfr=True)
    state = {"signal_domain": frames.DOMAIN_RX_TIME,
             "adc": torch.zeros(2, 2, 2, dtype=torch.complex64)}
    with pytest.raises(KeyError, match="CFRCaptureStage"):
        sink.apply(state)


def test_capture_stage_copies_rather_than_aliases():
    stage = CFRCaptureStage()
    s_pars = torch.ones(2, 1, 1, 4, dtype=torch.complex64)
    out = stage.apply({"s_pars": s_pars})
    s_pars.mul_(0)          # a later stage writing into the frame in place
    assert torch.all(out[CFR_CAPTURE_KEY] == 1)


# --------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------
def test_source_block_default_replay_is_unchanged(tmp_path, env):
    _run(tmp_path, env, store_cfr=True)
    src = SourceBlock(tmp_path)
    assert src.signal_domain == frames.DOMAIN_RX_TIME
    assert src.get_S_pars().shape == (_CFG.n_rx, _CFG.n_chirps, _CFG.n_samples)


def test_source_block_yields_the_cfr_frame_on_request(tmp_path, env):
    _run(tmp_path, env, store_cfr=True)
    src = SourceBlock(tmp_path, domain=frames.DOMAIN_CFR)
    assert src.signal_domain == frames.DOMAIN_CFR
    assert src.payload_key == frames.DOMAIN_PAYLOAD_KEY[frames.DOMAIN_CFR]
    for idx in (0, 1):
        got = src.get_S_pars()
        assert got.dtype == torch.complex64
        assert tuple(got.shape) == (_CFG.n_rx, _CFG.n_tx, _CFG.n_chirps, _CFG.n_samples)
        assert torch.equal(got.cpu(), env.frame(idx).cpu())
        src.step()
    # The labels still ride along with the frame.
    assert "labels" in src.get_state_updates()


def test_cfr_replay_of_a_corpus_without_sidecars_is_refused(tmp_path, env):
    _run(tmp_path, env, store_cfr=False)
    with pytest.raises(FileNotFoundError, match="store-cfr"):
        SourceBlock(tmp_path, domain=frames.DOMAIN_CFR)


def test_unknown_replay_domain_is_refused(tmp_path, env):
    _run(tmp_path, env, store_cfr=True)
    with pytest.raises(ValueError, match="not available"):
        SourceBlock(tmp_path, domain=frames.DOMAIN_TX_TIME)


def test_corpus_source_block_replays_the_cfr(tmp_path, env, grid):
    _run(tmp_path, env, store_cfr=True)
    manifest = write_manifest(
        tmp_path, _CFG, "D0", [["sample_frame_00000.npz"], ["sample_frame_00001.npz"]],
        grid=grid, splits=(0.5, 0.5, 0.0),
    )
    src = CorpusSourceBlock(manifest, split="train", domain=frames.DOMAIN_CFR)
    assert src.signal_domain == frames.DOMAIN_CFR
    assert torch.equal(src.get_S_pars().cpu(), env.frame(0).cpu())
    # Default (ADC replay) is untouched by the option existing.
    assert CorpusSourceBlock(manifest, split="train").signal_domain == frames.DOMAIN_RX_TIME


# --------------------------------------------------------------------------------
# The point of the exercise: re-run the chain live from a stored frame
# --------------------------------------------------------------------------------
def test_live_rerun_from_the_sidecar_reproduces_the_stored_adc(tmp_path, env):
    """Replay a stored CFR through the SAME chain and land on the same ADC cube.

    `use_rffe=False`: at HEAD the RFFE's thermal-noise draw is unseeded (a separate
    shard is fixing that), so with the front end in the chain two identical runs
    differ. Every other stage -- link budget, impairments, IF high-pass, quantizer --
    is seeded, so with the front end out this is exact.
    """
    gen_dir, replay_dir = tmp_path / "gen", tmp_path / "replay"
    _run(gen_dir, env, store_cfr=True, n=2, use_rffe=False)

    src = SourceBlock(gen_dir, domain=frames.DOMAIN_CFR)
    _run(replay_dir, env, store_cfr=False, n=2, use_rffe=False, environment_block=src)

    for idx in (0, 1):
        with np.load(gen_dir / f"sample_frame_{idx:05d}.npz") as a, \
                np.load(replay_dir / f"sample_frame_{idx:05d}.npz") as b:
            for key in ("adc_code_re", "adc_code_im"):
                assert np.array_equal(a[key], b[key]), f"frame {idx}, {key}"


# --------------------------------------------------------------------------------
# The paths sidecar: the cheap, durable form of the same channel
# --------------------------------------------------------------------------------
class _FakeRTEnvironmentWithPaths(_FakeRTEnvironment):
    """...plus the ray-traced path list the real RT block would emit.

    Stands in for the environment-side capture hook that does not exist yet (see
    `build_chain_simulation(store_paths=...)`): the contract this test pins down is
    that the environment emits a flat `{name: ndarray}` mapping under
    `PATHS_CAPTURE_KEY` from `get_state_updates()`, and the sink writes it verbatim.
    """

    n_paths = 7

    def get_state_updates(self):
        updates = super().get_state_updates()
        p = self.n_paths
        updates[PATHS_CAPTURE_KEY] = {
            "a": np.full((1, self.cfg.n_rx, 1, self.cfg.n_tx, p), self.frame_counter,
                         dtype=np.complex64),
            "tau": np.zeros((1, self.cfg.n_rx, 1, self.cfg.n_tx, p), dtype=np.float32),
            "doppler": np.zeros((1, self.cfg.n_rx, 1, self.cfg.n_tx, p), dtype=np.float32),
        }
        return updates


def test_paths_sidecar_written_and_read_back(tmp_path, grid, torch_device):
    env = _FakeRTEnvironmentWithPaths(_CFG, grid, n_frames=2, device=torch_device)
    sim = chain_generate.build_chain_simulation(
        scenario=None, cfg=_CFG, out_dir=tmp_path, environment_block=env,
        impairment_seed=7, store_cfr=True, store_paths=True,
    )
    sim.run(n_steps=2)

    from e2e.ml.storage import PATHS_SIDECAR_SUFFIX, read_paths_sidecar

    for idx in (0, 1):
        npz = tmp_path / f"sample_frame_{idx:05d}.npz"
        meta = _meta(npz)
        assert meta["paths_sidecar"] == f"sample_frame_{idx:05d}{PATHS_SIDECAR_SUFFIX}"
        got = read_paths_sidecar(npz, meta)
        assert set(got) == {"a", "tau", "doppler"}
        assert got["a"].shape == (1, _CFG.n_rx, 1, _CFG.n_tx, env.n_paths)
        assert got["a"].dtype == np.complex64
        assert np.all(got["a"] == idx)      # frame idx's own paths, not frame 0's


def test_store_paths_without_an_emitting_environment_fails_loudly(tmp_path, env):
    """No silent empty sidecar: a corpus either has the path list or says so."""
    sim = chain_generate.build_chain_simulation(
        scenario=None, cfg=_CFG, out_dir=tmp_path, environment_block=env,
        impairment_seed=7, store_paths=True,
    )
    with pytest.raises(KeyError, match="get_state_updates"):
        sim.run(n_steps=1)


def test_paths_sidecar_is_absent_by_default(tmp_path, grid, torch_device):
    env = _FakeRTEnvironmentWithPaths(_CFG, grid, n_frames=2, device=torch_device)
    _run(tmp_path, env, store_cfr=True, n=1)
    from e2e.ml.storage import PATHS_SIDECAR_SUFFIX

    assert list(tmp_path.glob(f"*{PATHS_SIDECAR_SUFFIX}")) == []
    assert "paths_sidecar" not in _meta(tmp_path / "sample_frame_00000.npz")
