"""RFFEBlock's thermal-noise draw is reproducible when seeded, unchanged when not.

Before this module existed, `circuit_model_bb_approx`'s noise draw
(`torch.randn_like`) came from the GLOBAL torch RNG, unseeded: replaying the same
`s_pars` through the composed ML-corpus chain (`e2e.ml.chain_generate.
build_chain_simulation`) twice -- same scene, same `impairment_seed` -- produced a
DIFFERENT ADC output each time (measured rel-RMSE ~0.6, entirely from this one
unseeded draw, since every other noise source in that chain is already seeded).
`RFFEBlock(seed=...)` / `circuit_model_batch(..., generator=...)` fix that without
touching the noise's statistics; `seed=None` (the default) is bit-for-bit the old
unseeded behaviour, verified by test_rffe_default_seed_unseeded_differs below.
"""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from e2e.blocks import RFFEBlock, device


N_RX = 8
N_FREQS = 32


def _s_pars(seed=0, n_rx=N_RX, n_freqs=N_FREQS):
    g = torch.Generator(device="cpu").manual_seed(seed)
    re = torch.randn(n_rx, 1, 1, n_freqs, generator=g)
    im = torch.randn(n_rx, 1, 1, n_freqs, generator=g)
    return torch.complex(re, im).to(torch.complex64).to(device)


# --------------------------------------------------------------------------------
# RFFEBlock-level: the unit this shard owns
# --------------------------------------------------------------------------------
def test_rffe_seeded_reproducible():
    """Two fresh RFFEBlocks, same seed, same input -> bit-identical output."""
    s_pars = _s_pars()
    out_a, prx_a = RFFEBlock(n=N_RX, seed=123).apply_circuit(s_pars)
    out_b, prx_b = RFFEBlock(n=N_RX, seed=123).apply_circuit(s_pars)
    assert torch.equal(out_a, out_b)
    assert torch.equal(prx_a, prx_b)


def test_rffe_different_seed_differs():
    s_pars = _s_pars()
    out_a, _ = RFFEBlock(n=N_RX, seed=1).apply_circuit(s_pars)
    out_b, _ = RFFEBlock(n=N_RX, seed=2).apply_circuit(s_pars)
    assert not torch.equal(out_a, out_b)


def test_rffe_default_seed_unseeded_differs():
    """seed=None (default) must stay RANDOM -- the whole point is that nothing
    changes for existing callers unless they opt in."""
    s_pars = _s_pars()
    out_a, _ = RFFEBlock(n=N_RX).apply_circuit(s_pars)
    out_b, _ = RFFEBlock(n=N_RX).apply_circuit(s_pars)
    assert not torch.equal(out_a, out_b)


def test_rffe_seed_advances_per_frame_and_reset_rewinds():
    """Same instance, two calls: frame 0 and frame 1 must draw different noise
    (seed + frame_idx), and reset() must rewind the sequence to frame 0's draw."""
    s_pars = _s_pars()
    rffe = RFFEBlock(n=N_RX, seed=7)
    frame0, _ = rffe.apply_circuit(s_pars)
    frame1, _ = rffe.apply_circuit(s_pars)
    assert not torch.equal(frame0, frame1)

    rffe.reset()
    frame0_again, _ = rffe.apply_circuit(s_pars)
    assert torch.equal(frame0, frame0_again)


# --------------------------------------------------------------------------------
# Composed-chain level: the pilot's own scoping check, as a regression test.
# --------------------------------------------------------------------------------
class _FixedCFREnvironment:
    """Stand-in for `RTEnvironmentBlock` returning the SAME `s_pars` every call, so
    two chain runs isolate exactly the downstream noise sources -- what the
    determinism claim below is about (mirrors the scoping pilot's `Stub`)."""

    def __init__(self, cfg, s_pars):
        self.cfg = cfg
        self._s_pars = s_pars
        self.array_shape = (cfg.n_rx, 1)
        self.frame_counter = 0

    def step(self):
        self.frame_counter += 1

    def reset(self):
        self.frame_counter = 0

    def get_S_pars(self):
        return self._s_pars.clone()


def _run_chain_adc(cfg, s_pars, out_dir, tag, seed):
    from e2e.ml import chain_generate, storage
    from e2e.ml.labels import LabelGrid

    grid = LabelGrid.for_config(cfg, range_stride=1, n_azimuth=8)
    sim = chain_generate.build_chain_simulation(
        scenario=None, cfg=cfg, out_dir=out_dir, tag=tag,
        environment_block=_FixedCFREnvironment(cfg, s_pars),
        impairment_chain_params=chain_generate.default_domain_randomizer(),
        impairment_seed=seed, label_grid=grid, device=device,
    )
    sim.run(n_steps=1)
    path = out_dir / f"{tag}_frame_00000.npz"
    with np.load(path) as data:
        meta = json.loads(str(data["meta"]))
        return storage.read_payload(data, meta, "adc")


def test_composed_chain_seeded_reproduces_adc(tmp_path):
    """The scoping pilot's own check: run the composed chain (RFFE -> interconnect
    -> dechirp -> thermal noise -> impairments -> quantizer) twice from ONE fixed
    `s_pars` with the same seed -- the ADC output must match exactly (rel-RMSE 0.0),
    not the ~0.6 measured before RFFEBlock's noise draw was seedable."""
    from e2e.radar_config import RadarConfig

    cfg = RadarConfig(
        name="test_rffe_determinism_cfg", f0_hz=77e9, bandwidth_hz=500e6,
        n_tx=1, n_rx=N_RX, n_chirps=4, n_samples=16, fs_hz=5e6,
        chirp_period_s=10e-6, mimo="single",
    )
    s_pars = _s_pars(seed=0, n_rx=cfg.n_rx, n_freqs=cfg.n_samples).view(
        cfg.n_rx, cfg.n_tx, 1, cfg.n_samples).expand(cfg.n_rx, cfg.n_tx,
                                                     cfg.n_chirps, cfg.n_samples).contiguous()

    adc_a = _run_chain_adc(cfg, s_pars, tmp_path, "run_a", seed=42)
    adc_b = _run_chain_adc(cfg, s_pars, tmp_path, "run_b", seed=42)
    assert np.array_equal(adc_a, adc_b)
    rel_rmse = np.sqrt(np.mean(np.abs(adc_a - adc_b) ** 2)) / np.sqrt(np.mean(np.abs(adc_a) ** 2))
    assert rel_rmse == 0.0

    adc_c = _run_chain_adc(cfg, s_pars, tmp_path, "run_c", seed=99)
    assert not np.array_equal(adc_a, adc_c)
