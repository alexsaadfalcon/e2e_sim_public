"""Tests for `e2e.ml.chain_generate`: the radar-ML corpus generator composed as an
`e2e.simulation.Simulation` run (RFFE -> interconnect -> dechirp -> impairments ->
quantizer -> radar cube -> sink), per `report/chain_integration_design.html`.

Ungated: a deterministic synthetic environment block stands in for
`RTEnvironmentBlock` (no Sionna/DrJit needed), so the COMPOSITION itself is exercised
in CI even though real corpus generation (`generate_chain_corpus`) needs Sionna RT.
"""

import dataclasses
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from e2e import frames
from e2e.ml import storage
from e2e.blocks import CircuitStage, InterconnectBlock, InterconnectStage, RFFEBlock
from e2e.chain.dechirp import DechirpBlock
from e2e.chain.receive import ImpairmentBlock, QuantizerBlock, RadarCubeBlock
from e2e.ml import chain_generate
from e2e.ml.blocks import SinkBlock
from e2e.ml.dataset import RadarFrameDataset, write_manifest
from e2e.ml.labels import LabelGrid
from e2e.ml.radar_config import RadarConfig


# Small single-TX config: fast, and sidesteps a pre-existing TDM bug in
# e2e.chain.receive.RadarCubeBlock (see test_radar_cube_block_tdm_bug_is_a_known_blocker
# below) that is out of scope for this shard.
_CFG = RadarConfig(
    name="test_chain_cfg",
    f0_hz=77e9,
    bandwidth_hz=500e6,
    n_tx=1,
    n_rx=4,
    n_chirps=8,
    n_samples=16,
    fs_hz=5e6,
    chirp_period_s=10e-6,
    mimo="single",
)

# TDM config matching the shape of the project's flagship preset (ti_iwr1443), used
# only to document the RadarCubeBlock blocker.
_TDM_CFG = dataclasses.replace(_CFG, name="test_chain_tdm_cfg", n_tx=3, n_chirps=9, mimo="tdm")


class _FakeRTEnvironment:
    """Deterministic CFR-emitting stand-in for `e2e.environment.blocks.
    RTEnvironmentBlock` -- same interface (`get_S_pars`/`step`/`reset`/
    `get_state_updates`/`array_shape`), no Sionna. Frame `i`'s CFR is drawn from a
    seed derived from `i`, so distinct frames actually differ (matters for the
    per-frame-impairment-differs test, and is a more honest stand-in than a fixed
    frame repeated).
    """

    def __init__(self, cfg, grid, n_frames=4, device=None, seed=0):
        self.cfg = cfg
        self.grid = grid
        self.n_frames = n_frames
        self.device = device
        self.seed = seed
        self.frame_counter = 0
        self.array_shape = (cfg.n_rx, 1)
        self.last_labels = None
        self.last_targets = None

    def step(self):
        self.frame_counter = (self.frame_counter + 1) % self.n_frames

    def reset(self):
        self.frame_counter = 0

    def get_S_pars(self):
        g = torch.Generator(device="cpu").manual_seed(self.seed + self.frame_counter)
        shape = (self.cfg.n_rx, self.cfg.n_tx, self.cfg.n_chirps, self.cfg.n_samples)
        re = torch.randn(shape, generator=g)
        im = torch.randn(shape, generator=g)
        s_pars = torch.complex(re, im).to(torch.complex64)
        if self.device is not None:
            s_pars = s_pars.to(self.device)

        # Deterministic ground truth that varies with the frame index, so per-frame
        # provenance (labels/targets) is checkably distinct too.
        labels = torch.zeros(3, self.grid.n_range, self.grid.n_azimuth)
        i = min(2 + self.frame_counter, self.grid.n_range - 1)
        j = min(2 + self.frame_counter, self.grid.n_azimuth - 1)
        labels[0, i, j] = 1.0
        r = 5.0 + self.frame_counter
        self.last_labels = labels
        self.last_targets = [(r, 0.1, "vehicle")]
        return s_pars

    def get_state_updates(self):
        if self.last_labels is None:
            return {}
        return {"labels": self.last_labels, "targets": self.last_targets}


@pytest.fixture
def grid():
    return LabelGrid.for_config(_CFG, range_stride=1, n_azimuth=8)


@pytest.fixture
def fake_env(grid, torch_device):
    return _FakeRTEnvironment(_CFG, grid, n_frames=4, device=torch_device, seed=0)


# --------------------------------------------------------------------------------
# Composition: RFFE/interconnect really are on the path by default
# --------------------------------------------------------------------------------
def test_rffe_and_interconnect_present_in_default_composition(tmp_path, fake_env):
    sim = chain_generate.build_chain_simulation(
        scenario=None, cfg=_CFG, out_dir=tmp_path, environment_block=fake_env,
    )
    stage_types = [type(s) for s in sim.serial_stages]
    assert CircuitStage in stage_types
    assert InterconnectStage in stage_types
    assert DechirpBlock in stage_types
    assert QuantizerBlock in stage_types

    circuit_stage = next(s for s in sim.serial_stages if isinstance(s, CircuitStage))
    assert isinstance(circuit_stage.rffe_block, RFFEBlock)
    # RFFEBlock defaults to the imaging array's element count -- the composition MUST
    # override it to the radar's actual receive-channel count, or apply_circuit's
    # view() raises on the very first frame (see build_chain_simulation's docstring).
    assert circuit_stage.rffe_block.n == _CFG.n_rx

    interconnect_stage = next(s for s in sim.serial_stages if isinstance(s, InterconnectStage))
    assert isinstance(interconnect_stage.interconnect_block, InterconnectBlock)

    # Ordering: RFFE and interconnect precede the dechirp bridge (they operate in the
    # frequency domain, before the chain crosses into RX time).
    dechirp_idx = stage_types.index(DechirpBlock)
    assert stage_types.index(CircuitStage) < dechirp_idx
    assert stage_types.index(InterconnectStage) < dechirp_idx

    downstream_types = [type(b) for b in sim.downstream_blocks]
    assert RadarCubeBlock in downstream_types
    assert SinkBlock in downstream_types


def test_rffe_and_interconnect_are_config_gated_off(tmp_path, fake_env):
    sim = chain_generate.build_chain_simulation(
        scenario=None, cfg=_CFG, out_dir=tmp_path, environment_block=fake_env,
        use_rffe=False, use_interconnect=False,
    )
    stage_types = [type(s) for s in sim.serial_stages]
    assert CircuitStage not in stage_types
    assert InterconnectStage not in stage_types
    assert DechirpBlock in stage_types  # the bridge itself is never gated


# --------------------------------------------------------------------------------
# End-to-end composition run (no Sionna)
# --------------------------------------------------------------------------------
def test_composition_builds_and_runs_end_to_end(tmp_path, fake_env):
    sim = chain_generate.build_chain_simulation(
        scenario=None, cfg=_CFG, out_dir=tmp_path, environment_block=fake_env,
        impairment_chain_params=chain_generate.default_domain_randomizer(),
    )
    outputs = sim.run(n_steps=3)

    assert len(outputs["radar_cube"]) == 3
    assert outputs["radar_cube"][0].shape == (_CFG.n_rx, _CFG.n_samples, _CFG.n_chirps)

    files = sorted(tmp_path.glob("sample_frame_*.npz"))
    assert len(files) == 3
    with np.load(files[0]) as data:
        meta = json.loads(str(data["meta"].item()))
        # Read through the storage layer rather than the raw key: a chain that runs
        # QuantizerBlock lands as integer codes plus a scale, which is exactly lossless
        # and 2.7x smaller, so there is no literal "adc" array to index.
        adc = storage.read_payload(data, meta, "adc")
        assert adc.shape == (_CFG.n_rx, _CFG.n_chirps, _CFG.n_samples)
        assert "labels" in data
        assert meta["domain"] == frames.DOMAIN_RX_TIME
        assert meta["payload_key"] == "adc"
        assert "targets" in meta
        assert meta["codec"] == storage.CODEC_INT16


# --------------------------------------------------------------------------------
# Written samples load back through the EXISTING RadarFrameDataset
# --------------------------------------------------------------------------------
def test_written_sample_loads_via_radar_frame_dataset(tmp_path, fake_env, grid):
    dataset_dir = tmp_path / f"{_CFG.name}_D0"
    dataset_dir.mkdir()
    sim = chain_generate.build_chain_simulation(
        scenario=None, cfg=_CFG, out_dir=dataset_dir, environment_block=fake_env,
        tag="sample_scene00000",
        impairment_chain_params=chain_generate.default_domain_randomizer(),
    )
    sim.run(n_steps=2)

    sequences = [[f"sample_scene00000_frame_{t:05d}.npz" for t in range(2)]]
    manifest_path = write_manifest(dataset_dir, _CFG, "D0", sequences, grid=grid,
                                   frames_per_scene=2, splits=(1.0, 0.0, 0.0))
    assert manifest_path.is_file()

    ds = RadarFrameDataset(manifest_path, split="train")
    assert len(ds) == 2
    x, y = ds[0]
    assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
    assert x.dtype == torch.float32
    assert y.shape == (3, grid.n_range, grid.n_azimuth)

    ds_adc = RadarFrameDataset(manifest_path, split="train", input_format="adc")
    x_adc, _ = ds_adc[0]
    assert x_adc.shape == (2 * _CFG.n_rx, _CFG.n_samples, _CFG.n_chirps)

    targets = ds.targets(0)
    assert isinstance(targets, list) and len(targets) == 1
    assert targets[0][2] == "vehicle"


# --------------------------------------------------------------------------------
# Per-frame impairment domain randomization reaches the written sample
# --------------------------------------------------------------------------------
def test_impairment_params_differ_across_frames_and_are_in_written_samples(tmp_path, fake_env):
    sim = chain_generate.build_chain_simulation(
        scenario=None, cfg=_CFG, out_dir=tmp_path, environment_block=fake_env,
        impairment_chain_params=chain_generate.default_domain_randomizer(), impairment_seed=0,
    )
    sim.run(n_steps=3)

    files = sorted(tmp_path.glob("sample_frame_*.npz"))
    assert len(files) == 3
    psd_values = []
    for f in files:
        with np.load(f) as data:
            meta = json.loads(str(data["meta"].item()))
        params = meta["impairment_params"]
        # Converted to plain JSON-serializable dicts (see chain_generate._ImpairmentStage);
        # each stage's resolved params, plus the per-frame seed actually used.
        assert set(params) >= {"phase_noise", "leakage", "clutter", "seed"}
        assert isinstance(params["phase_noise"], dict)
        psd_values.append(params["phase_noise"]["psd_dbc_hz_at_ref"])

    assert len(set(psd_values)) == len(psd_values)  # every frame drew a distinct value

    seeds = []
    for f in files:
        with np.load(f) as data:
            meta = json.loads(str(data["meta"].item()))
        seeds.append(meta["impairment_params"]["seed"])
    assert seeds == [0, 1, 2]


def test_fixed_impairment_params_are_identical_across_frames(tmp_path, fake_env):
    """Sanity check on the OTHER end of the requirement: a FIXED (non-callable)
    chain_params must NOT vary frame to frame -- proves the differences above come
    from the randomizer, not from e.g. seed alone."""
    sim = chain_generate.build_chain_simulation(
        scenario=None, cfg=_CFG, out_dir=tmp_path, environment_block=fake_env,
        impairment_chain_params={"phase_noise": {"psd_dbc_hz_at_ref": -80.0}},
    )
    sim.run(n_steps=2)
    files = sorted(tmp_path.glob("sample_frame_*.npz"))
    values = []
    for f in files:
        with np.load(f) as data:
            meta = json.loads(str(data["meta"].item()))
        values.append(meta["impairment_params"]["phase_noise"]["psd_dbc_hz_at_ref"])
    assert values == [-80.0, -80.0]


# --------------------------------------------------------------------------------
# Known blocker: RadarCubeBlock mishandles TDM mimo (out-of-scope file)
# --------------------------------------------------------------------------------
def test_tdm_config_runs_the_whole_chain_and_produces_a_radar_cube(tmp_path, torch_device):
    """The project's flagship preset is TDM, so the composed chain must survive it.

    RadarCubeBlock used to de-interleave the cube -- collapsing the transmit
    multiplexing into one virtual array with n_chirps_per_tx slow-time samples -- and
    then describe it to adc_to_rd with the ORIGINAL config, whose chirp count no longer
    matched. That raised, blocking the downstream product for every TDM configuration.
    """
    grid_tdm = LabelGrid.for_config(_TDM_CFG, range_stride=1, n_azimuth=8)
    env = _FakeRTEnvironment(_TDM_CFG, grid_tdm, n_frames=1, device=torch_device, seed=0)
    sim = chain_generate.build_chain_simulation(
        scenario=None, cfg=_TDM_CFG, out_dir=tmp_path, environment_block=env,
    )
    sim.run(n_steps=1)

    cube = sim.get_outputs()["radar_cube"][0]
    # One virtual array of n_rx*n_tx elements; Doppler axis is the per-transmit count.
    assert cube.shape == (_TDM_CFG.n_rx * _TDM_CFG.n_tx, _TDM_CFG.n_samples,
                          _TDM_CFG.n_chirps_per_tx)
