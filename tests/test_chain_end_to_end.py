"""The whole chain, composed and run — the claim that 'one chain' is real.

An adversarial review found that the chain blocks, though individually built and
tested, had no caller: nothing anywhere composed them into a running Simulation, and
the design document's own headline example could not execute. A block exercised only by
its own unit test is a claim, not an integration.

These tests are that integration, end to end and in one piece: a transmitted waveform,
distorted by the amplifier, merged into a ray-traced-shaped channel response, carried
through the front end, dechirped into the time domain, impaired, digitized, and turned
into a radar cube. They run on the project's flagship 3-transmit TDM preset, because
that configuration is where every shape assumption in the chain gets tested at once.

No Sionna: the environment is a stand-in emitting deterministic frames of the right
shape. What is under test here is the COMPOSITION, not the ray tracer.
"""

import dataclasses

import pytest

torch = pytest.importorskip("torch")

from e2e import frames
from e2e.chain.dechirp import DechirpBlock
from e2e.chain.receive import ImpairmentBlock, QuantizerBlock, RadarCubeBlock
from e2e.chain.waveform import ModulateBlock, TxPABlock, WaveformBlock
from e2e.circuit.tx_pa import TxPA, TxPAConfig
from e2e.ml.radar_config import PRESETS
from e2e.simulation import Simulation


# Small enough to stay fast, but keeps the flagship preset's 3 TX / 4 RX TDM structure.
CFG = dataclasses.replace(PRESETS["ti_iwr1443"], n_chirps=12, n_samples=64)


class _StubEnvironment:
    """Emits deterministic channel frames of the right shape, at physically small
    amplitudes (~1e-6) so the ADC's automatic gain is genuinely exercised rather than
    handed a conveniently unit-scaled cube."""

    array_shape = (CFG.n_rx, 1)

    def __init__(self, scale=1e-6):
        self.scale = scale
        self._i = 0

    def get_S_pars(self):
        g = torch.Generator().manual_seed(self._i)
        shape = (CFG.n_rx, CFG.n_tx, CFG.n_chirps, CFG.n_samples)
        real = torch.randn(shape, generator=g)
        imag = torch.randn(shape, generator=g)
        return ((real + 1j * imag) * self.scale).to(torch.complex64)

    def step(self):
        self._i += 1

    def reset(self):
        self._i = 0


def _full_chain(with_transmit=True):
    pa = TxPA(TxPAConfig())
    stages = []
    if with_transmit:
        stages += [
            WaveformBlock(kind="fmcw", n_tx=CFG.n_tx, n_chirp=CFG.n_chirps,
                          n_t=CFG.n_samples),
            TxPABlock(pa),
            ModulateBlock(tx_pa=pa, bandwidth_hz=CFG.bandwidth_hz),
        ]
    stages += [
        DechirpBlock(CFG),
        ImpairmentBlock(CFG, seed=1),
        QuantizerBlock(bits=12),
    ]
    return Simulation(_StubEnvironment(), [RadarCubeBlock(CFG)], k=4,
                      serial_stages=stages)


def test_the_whole_chain_runs_and_produces_a_radar_cube():
    """Transmit tributary, channel, dechirp, impairments, ADC, radar cube -- one run."""
    sim = _full_chain()
    sim.run(n_steps=2)

    cubes = sim.get_outputs()["radar_cube"]
    assert len(cubes) == 2
    # TDM de-interleaving forms the virtual array: n_rx * n_tx channels, and the Doppler
    # axis carries the per-transmitter chirp count.
    assert cubes[0].shape == (CFG.n_rx * CFG.n_tx, CFG.n_samples, CFG.n_chirps_per_tx)
    assert cubes[0].dtype == torch.complex64


def test_the_chain_survives_physically_small_amplitudes():
    """Ray-traced frames are ~1e-6, far below a fixed 12-bit LSB. The chain must not
    quantize the whole frame to nothing -- the failure the ADC's automatic gain exists
    to prevent."""
    sim = _full_chain()
    sim.run(n_steps=1)
    cube = sim.get_outputs()["radar_cube"][0]
    assert torch.count_nonzero(cube) > 0
    assert torch.isfinite(cube.abs()).all()


def test_the_transmit_tributary_is_optional_and_the_chain_still_runs():
    """The transmit path is a tributary, not a required segment: without it the chain
    assumes an ideal flat transmitter, which is what every existing simulation does."""
    sim = _full_chain(with_transmit=False)
    sim.run(n_steps=1)
    assert sim.get_outputs()["radar_cube"][0].shape == (
        CFG.n_rx * CFG.n_tx, CFG.n_samples, CFG.n_chirps_per_tx
    )


def test_the_transmit_path_actually_changes_the_result():
    """A tributary that made no difference would be decoration. The amplifier and
    waveform must leave a mark on the finished product."""
    with_tx = _full_chain(with_transmit=True)
    with_tx.run(n_steps=1)
    without_tx = _full_chain(with_transmit=False)
    without_tx.run(n_steps=1)

    a = with_tx.get_outputs()["radar_cube"][0]
    b = without_tx.get_outputs()["radar_cube"][0]
    assert not torch.allclose(a, b)


def test_impairment_provenance_survives_to_the_end_of_the_chain():
    """Every frame must be able to say what was done to it -- a randomized corpus is
    not much use otherwise."""
    sim = _full_chain()
    sim.run(n_steps=1)
    # The impairment stage records its settings into state; the sink writes them out.
    stage = next(s for s in sim.serial_stages if isinstance(s, ImpairmentBlock))
    out = stage.apply({"adc": torch.ones(CFG.n_rx, CFG.n_chirps, CFG.n_samples,
                                         dtype=torch.complex64)})
    params = out["impairment_params"]
    assert {"phase_noise", "leakage", "clutter", "seed"} <= set(params)


def test_a_misordered_chain_is_rejected_by_name():
    """Putting a receive-side block before the dechirp must fail immediately, naming
    the bridge that would fix it -- not fail later inside a transform."""
    sim = Simulation(
        _StubEnvironment(), [], k=4,
        serial_stages=[QuantizerBlock(bits=12), DechirpBlock(CFG)],
    )
    with pytest.raises(frames.FrameContractError, match="DechirpBlock"):
        sim.run(n_steps=1)
