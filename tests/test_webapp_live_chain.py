"""The GUI's LIVE-CHAIN replay path: a corpus frame re-entering the chain as the
stored ray-traced channel, with the ADC chain re-run from the knobs on screen.

Owner directive 2026-09-23 ("store the ray tracing, compute everything else live with
the knobs", notes/STATE.md §0.1; plan D3 in notes/RT_LIVE_PLAN_2026-09-23.md). Before
this, `webapp/pipeline_runner.py` bypassed every serial stage for a corpus source --
the frame was already digitized -- so no front-end setting could reach the detector on
a Thrust 5 screen.

THE CORRECTNESS GATE, and why it is a test AND a run note: a live chain that drifted
from the chain that generated the corpus would still draw a plausible picture. So the
runner re-encodes the live cube in each stored frame's own int16 code space and reports
the largest difference on every run (`_StoredADCGateBlock`), and the tests below pin
both directions of that gate -- zero at the frames' own settings, non-zero as soon as a
knob moves (a gate that can only say "identical" is not a gate).

Ungated and synthetic by default: a deterministic CFR-emitting stand-in drives the real
corpus generator (`e2e.ml.chain_generate.build_chain_simulation`, the same composition
the real corpora were written with), so the whole generate -> replay -> re-run loop runs
in CI in seconds with no Sionna and no ray tracing. The tests that need the real demo
corpus (3.4 GB, not tracked) skip when it is absent.
"""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from e2e import frames
from e2e.ml import chain_generate, storage
from e2e.ml.dataset import write_manifest
from e2e.ml.labels import LabelGrid
from e2e.radar_config import RadarConfig
from webapp.pipeline_registry import default_block_state
from webapp.pipeline_runner import PipelineError, run_pipeline

# Single-TX and tiny: a full chain run (front end -> ... -> quantizer -> cube -> CFAR)
# costs milliseconds, and single-TX sidesteps the known RadarCubeBlock TDM blocker.
_CFG = RadarConfig(name="live_chain_cfg", f0_hz=77e9, bandwidth_hz=500e6, n_tx=1, n_rx=4,
                   n_chirps=8, n_samples=16, fs_hz=5e6, chirp_period_s=10e-6, mimo="single")
_SEED = 17

#: The real demo corpus the Thrust 5 presets point at (generated with --store-cfr).
_DEMO_MANIFEST = (Path(__file__).resolve().parents[1] / "e2e" / "ml" / "datasets"
                  / "b1_demo_cfr" / "benchmark_v1_D2" / "manifest.json")
_needs_demo_corpus = pytest.mark.skipif(
    not _DEMO_MANIFEST.is_file(),
    reason="the b1_demo_cfr corpus (3.4 GB, generated locally) is not on this machine")


class _FakeRTEnvironment:
    """Deterministic CFR-emitting stand-in for `RTEnvironmentBlock` (no Sionna).

    Mirrors the one in tests/test_ml_store_cfr.py rather than importing it: that file
    owns the storage half's contract, this one owns the GUI's, and a shared fixture
    would couple the two shards' tests for four lines of tensor.
    """

    def __init__(self, cfg, grid, n_frames=2, device=None, seed=0):
        self.cfg, self.grid, self.n_frames = cfg, grid, n_frames
        self.device, self.seed = device, seed
        self.frame_counter = 0
        self.array_shape = (cfg.n_rx, 1)
        self.last_labels = None

    def step(self):
        self.frame_counter = (self.frame_counter + 1) % self.n_frames

    def reset(self):
        self.frame_counter = 0

    def get_S_pars(self):
        g = torch.Generator(device="cpu").manual_seed(self.seed + self.frame_counter)
        shape = (self.cfg.n_rx, self.cfg.n_tx, self.cfg.n_chirps, self.cfg.n_samples)
        s = torch.complex(torch.randn(shape, generator=g),
                          torch.randn(shape, generator=g)).to(torch.complex64)
        labels = torch.zeros(3, self.grid.n_range, self.grid.n_azimuth)
        labels[0, min(2, self.grid.n_range - 1), min(2, self.grid.n_azimuth - 1)] = 1.0
        self.last_labels = labels
        return s if self.device is None else s.to(self.device)

    def get_state_updates(self):
        return {} if self.last_labels is None else {"labels": self.last_labels}


@pytest.fixture
def grid():
    return LabelGrid.for_config(_CFG, range_stride=1, n_azimuth=8)


def _generate_corpus(out_dir, grid, torch_device, *, n_frames=2, store_cfr=True,
                     **build_kwargs):
    """A corpus written by the REAL generator composition -- with or without the
    stored channel -- plus a manifest. Returns the manifest path.

    `build_kwargs` go straight to `build_chain_simulation`, so a test can generate a
    corpus whose CHAIN differs from the generator's defaults (`if_hpf_kwargs=`,
    `use_rffe=False`, ...) -- the case the live-chain replay cannot fully reconstruct
    and must therefore describe honestly."""
    env = _FakeRTEnvironment(_CFG, grid, n_frames=n_frames, device=torch_device)
    sim = chain_generate.build_chain_simulation(
        scenario=None, cfg=_CFG, out_dir=out_dir, environment_block=env,
        impairment_seed=_SEED, store_cfr=store_cfr, label_grid=grid, **build_kwargs,
    )
    sim.run(n_steps=n_frames)
    names = [[f"sample_frame_{i:05d}.npz"] for i in range(n_frames)]
    return write_manifest(out_dir, _CFG, "D0", names, grid=grid, splits=(0.0, 0.0, 1.0))


def _state(manifest, *, domain="auto", bits=12, corner_range_m=1.0, full_scale=0.0):
    """The block state a Thrust 5 live-chain preset builds, at this corpus's geometry."""
    st = default_block_state()
    st["corpus_environment"]["enabled"] = True
    st["corpus_environment"]["params"].update(manifest=str(manifest), split="test",
                                              start_frame=0, domain=domain)
    for bid in ("rffe", "interconnect", "dechirp", "thermal_noise", "impairment",
                "if_hpf", "quantizer", "radar_cube", "detector"):
        st[bid]["enabled"] = True
    for bid in ("afe", "subspace", "fft", "range_az", "range_el", "range_profile",
                "subspace_err", "comms", "sink", "waveform", "tx_pa", "modulate"):
        st[bid]["enabled"] = False
    st["if_hpf"]["params"].update(corner_range_m=corner_range_m, order=2)
    st["quantizer"]["params"].update(bits=bits, full_scale=full_scale)
    st["detector"]["params"].update(mode="cfar", threshold=0.5, cfar_guard=1, cfar_train=2)
    return st


def _notes(outputs):
    return " | ".join(outputs["_axis_meta"]["notes"])


def _gate_code_diff(outputs):
    """The `max |diff| = N ADC codes` figure the runner prints on every live run."""
    note = next(n for n in outputs["_axis_meta"]["notes"] if "live chain vs stored ADC" in n)
    return int(note.split("max |diff| = ")[1].split(" ")[0])


# ------------------------------------------------------------------------------------
# The correctness gate
# ------------------------------------------------------------------------------------
def test_live_chain_reproduces_the_stored_cube_bit_for_bit(tmp_path, grid, torch_device):
    """THE gate: with every knob at the value the frames were generated with, the cube
    the GUI computes live from the stored channel is the cube the corpus holds -- not
    close to it, identical to it."""
    manifest = _generate_corpus(tmp_path, grid, torch_device)
    outputs = run_pipeline(_state(manifest), n_steps=2)

    assert outputs["_axis_meta"]["source"].startswith(
        "Corpus Replay (live chain from stored channel)")
    assert _gate_code_diff(outputs) == 0
    assert "bit-identical" in _notes(outputs)
    assert len(outputs["radar_cube"]) == 2 and len(outputs["cfar_detection"]) == 2


def test_the_gate_is_not_vacuous_a_moved_knob_shows_up_as_a_difference(tmp_path, grid,
                                                                      torch_device):
    """A gate that can only report "identical" proves nothing. Re-digitising the same
    stored channel at 4 bits must move the number off zero -- that difference IS the
    demo (an A/B arm), and the run note says so in the same words."""
    manifest = _generate_corpus(tmp_path, grid, torch_device)
    outputs = run_pipeline(_state(manifest, bits=4), n_steps=2)

    assert _gate_code_diff(outputs) > 0
    assert "DIFFERS" in _notes(outputs) and "ADC 4-bit" in _notes(outputs)


def test_the_live_cube_matches_what_the_adc_replay_path_serves(tmp_path, grid,
                                                               torch_device):
    """The two paths, same frames, same products: replaying the stored ADC cube and
    recomputing it live from the stored channel must put the SAME detections on
    screen. This is the claim the Thrust 5 cards make when they say the live count is
    the corpus's own, and it is checked here on the products, not only on the cube."""
    manifest = _generate_corpus(tmp_path, grid, torch_device)
    live = run_pipeline(_state(manifest, domain="cfr"), n_steps=2)
    replay = run_pipeline(_state(manifest, domain="adc"), n_steps=2)

    assert replay["_axis_meta"]["source"].startswith("Corpus Replay (ADC replay)")
    for a, b in zip(live["cfar_detection"], replay["cfar_detection"]):
        assert torch.equal(a, b)
    assert ([len(d) for d in live["cfar_detections"]]
            == [len(d) for d in replay["cfar_detections"]])


def test_the_stored_channel_is_the_frame_the_sidecar_holds(tmp_path, grid, torch_device):
    """The payload the live chain starts from is the ray-traced frame on disk, read by
    `storage.read_cfr_sidecar` -- pinned here so the GUI path cannot silently start
    from something else (a re-derived or re-scaled channel would still "work")."""
    manifest = _generate_corpus(tmp_path, grid, torch_device)
    from e2e.ml.blocks import CorpusSourceBlock

    src = CorpusSourceBlock(manifest, split="test", domain=frames.DOMAIN_CFR)
    npz = src._files[0]
    with np.load(npz, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"].item()))
    expected = storage.read_cfr_sidecar(npz, meta)

    assert src.signal_domain == frames.DOMAIN_CFR
    assert np.array_equal(src.get_S_pars().cpu().numpy(), expected)


# ------------------------------------------------------------------------------------
# Per-frame identity: the seeds and severities that belong to the frame, not the UI
# ------------------------------------------------------------------------------------
def test_each_frame_is_rerun_with_its_own_stored_seed(tmp_path, grid, torch_device):
    """Every frame carries the base seed and the domain-randomised impairment
    severities it was generated with, and the live chain uses those (one UI seed
    cannot reproduce two frames). The UI seed is deliberately left at a value that is
    NOT the corpus's, so a run that ignored the stored one would fail this."""
    manifest = _generate_corpus(tmp_path, grid, torch_device)
    st = _state(manifest)
    st["thermal_noise"]["params"]["seed"] = 999
    st["impairment"]["params"]["seed"] = 999

    outputs = run_pipeline(st, n_steps=2)

    assert _gate_code_diff(outputs) == 0
    assert "stored noise seeds" in _notes(outputs)


# ------------------------------------------------------------------------------------
# Which path ran, and the legacy path that must not change
# ------------------------------------------------------------------------------------
def test_a_corpus_without_a_stored_channel_keeps_todays_adc_replay(tmp_path, grid,
                                                                   torch_device):
    """'auto' falls back: every corpus generated before 2026-09-23 has no sidecars, and
    its screens must behave exactly as they did -- the chain bypassed, and the run note
    naming the blocks that were skipped."""
    manifest = _generate_corpus(tmp_path, grid, torch_device, store_cfr=False)
    outputs = run_pipeline(_state(manifest), n_steps=2)

    assert outputs["_axis_meta"]["source"].startswith("Corpus Replay (ADC replay)")
    notes = _notes(outputs)
    assert "skipped the enabled blocks" in notes
    assert "live chain vs stored ADC" not in notes
    assert len(outputs["radar_cube"]) == 2


def test_asking_for_the_stored_channel_when_there_is_none_says_what_is_missing(
        tmp_path, grid, torch_device):
    manifest = _generate_corpus(tmp_path, grid, torch_device, store_cfr=False)
    with pytest.raises(PipelineError, match="store-cfr"):
        run_pipeline(_state(manifest, domain="cfr"), n_steps=1)


def test_the_live_chain_needs_the_dechirp_bridge_and_says_so(tmp_path, grid, torch_device):
    """Turning the bridge off leaves the chain in the frequency domain with RX-time
    products attached -- refused by name, not by a shape error three stages later."""
    manifest = _generate_corpus(tmp_path, grid, torch_device)
    st = _state(manifest, domain="cfr")
    st["dechirp"]["enabled"] = False
    with pytest.raises(PipelineError, match="Dechirp"):
        run_pipeline(st, n_steps=1)


def test_a_fixed_full_scale_is_still_honoured(tmp_path, grid, torch_device):
    """full_scale 0 means automatic gain (the generator's setting); a positive value
    must still reach `QuantizerBlock` as a fixed clip level -- which on this corpus
    changes the cube, so the gate reports a difference rather than zero."""
    manifest = _generate_corpus(tmp_path, grid, torch_device)
    outputs = run_pipeline(_state(manifest, full_scale=1e-3), n_steps=1)
    assert _gate_code_diff(outputs) > 0


# ------------------------------------------------------------------------------------
# The real demo corpus the Thrust 5 presets point at (skipped when it is not here)
# ------------------------------------------------------------------------------------
@_needs_demo_corpus
@pytest.mark.slow
def test_demo_preset_live_chain_reproduces_the_demo_corpus_exactly():
    """The gate on the frames the demo actually shows: the A arm of every Thrust 5
    preset reproduces the stored cube bit for bit, and its detections are the ones the
    ADC-replay path serves from the same frames. Measured 2026-09-23: max |diff| = 0
    ADC codes over the 5 test frames."""
    from webapp.demo_presets import PRESETS_BY_ID, apply_preset

    state = apply_preset(PRESETS_BY_ID["thrust5_detector_cfar"])
    live = run_pipeline(state, n_steps=5)
    assert _gate_code_diff(live) == 0

    replay = dict(state)
    replay["corpus_environment"] = {**state["corpus_environment"],
                                    "params": {**state["corpus_environment"]["params"],
                                               "domain": "adc"}}
    served = run_pipeline(replay, n_steps=5)
    for a, b in zip(live["cfar_detection"], served["cfar_detection"]):
        assert torch.equal(a, b)


@_needs_demo_corpus
@pytest.mark.slow
def test_demo_preset_b_arm_changes_the_detection_count():
    """The B arm must actually move the picture: re-digitising the same stored channel
    at 4 bits changed the CFAR count 61 -> 55 crosses over the 5 frames (measured
    2026-09-23, two identical runs per arm). Pinned as a direction, not as the exact
    pair -- the cards carry the numbers, this pins that the knob reaches the detector."""
    from webapp.demo_presets import PRESETS_BY_ID, apply_preset

    preset = PRESETS_BY_ID["thrust5_detector_cfar"]
    a = run_pipeline(apply_preset(preset), n_steps=5)
    b = run_pipeline(apply_preset(preset, arm="b"), n_steps=5)

    assert _gate_code_diff(a) == 0 and _gate_code_diff(b) > 0
    n_a = sum(len(d) for d in a["cfar_detections"])
    n_b = sum(len(d) for d in b["cfar_detections"])
    assert n_a != n_b, "the ADC bits knob did not reach the detector"


# ------------------------------------------------------------------------------------
# A corpus whose CHAIN differs from the generator's defaults. The live path rebuilds
# the chain from the GUI plus what each frame records -- and a frame records its
# seeds, its impairment severities and its IF high-pass, but NOT whether the front
# end / interconnect / link budget ran, nor the ADC's bit depth. Found by review
# (2026-09-23): before these tests, such a corpus produced a non-zero gate reading
# that the run note attributed to "a knob off the value the frames were generated
# with" -- blaming an operator who had touched nothing.
# ------------------------------------------------------------------------------------
def test_a_frames_own_if_high_pass_is_named_when_it_is_not_this_runs(tmp_path, grid,
                                                                    torch_device):
    """The one piece of chain topology a frame DOES record. Generated at a 3 m corner,
    replayed at the UI's 1 m: the run must name both numbers, not blame the operator."""
    manifest = _generate_corpus(tmp_path, grid, torch_device,
                                if_hpf_kwargs={"corner_range_m": 3.0})
    outputs = run_pipeline(_state(manifest, corner_range_m=1.0), n_steps=2)
    notes = _notes(outputs)

    assert _gate_code_diff(outputs) > 0
    assert "IF high-pass: these frames were filtered at" in notes
    # The stored corner is 3x the live one; both are stated, in Hz, from the frame.
    stored_hz = 2.0 * _CFG.ramp_slope_hzps * 3.0 / 299792458.0
    assert f"{stored_hz:.0f} Hz" in notes
    # ...and the gate no longer asserts a cause it cannot know.
    assert "a knob is off" not in notes and "expected exactly when a knob" not in notes


def test_matching_the_frames_own_if_high_pass_puts_the_gate_back_to_zero(tmp_path, grid,
                                                                        torch_device):
    """The mirror: the corner is a live knob, never overridden from meta, so setting it
    to what the frames record reproduces them exactly."""
    manifest = _generate_corpus(tmp_path, grid, torch_device,
                                if_hpf_kwargs={"corner_range_m": 3.0})
    outputs = run_pipeline(_state(manifest, corner_range_m=3.0), n_steps=2)

    assert _gate_code_diff(outputs) == 0
    assert "IF high-pass: these frames were filtered at" not in _notes(outputs)


def test_an_unrecorded_chain_difference_is_described_not_blamed(tmp_path, grid,
                                                                torch_device):
    """A corpus generated WITHOUT the RF front end: nothing on disk says so, so the
    live chain cannot reconstruct it. The run must say the difference lies in what a
    frame does not record -- and must not claim the operator moved something."""
    manifest = _generate_corpus(tmp_path, grid, torch_device, use_rffe=False)
    outputs = run_pipeline(_state(manifest), n_steps=2)
    notes = _notes(outputs)

    assert _gate_code_diff(outputs) > 0
    assert "does NOT record" in notes
    assert "a knob is off" not in notes and "expected exactly when a knob" not in notes


def test_the_live_interconnect_matches_the_generators_own(tmp_path, grid, torch_device):
    """Drift guard (reviewed 2026-09-23): the runner types the interconnect's band
    (75-81 GHz) as a literal, because `e2e.ml.chain_generate` has no named constant to
    import for it -- only the CSV path. If the generator's default band or CSV moves
    and the runner's literal does not, replay parity breaks silently. Pin both here,
    read off a chain the generator itself built, so the literal cannot drift alone."""
    from e2e.blocks import InterconnectStage
    from e2e.ml.chain_generate import DEFAULT_INTERCONNECT_CSV

    env = _FakeRTEnvironment(_CFG, grid, n_frames=1, device=torch_device)
    sim = chain_generate.build_chain_simulation(
        scenario=None, cfg=_CFG, out_dir=tmp_path, environment_block=env,
        impairment_seed=_SEED, store_cfr=True, label_grid=grid)
    generator_ic = next(s.interconnect_block for s in sim.serial_stages
                        if isinstance(s, InterconnectStage))

    # The two values webapp/pipeline_runner.py re-types in its `corpus_live_cfr`
    # interconnect branch. Change them there and here together, or not at all.
    assert tuple(generator_ic.band_hz) == (75e9, 81e9)
    assert Path(generator_ic.transfer_csv) == Path(DEFAULT_INTERCONNECT_CSV)
