"""Does the FULL composition break the stored corpora? Measured, on the real Ka corpus.

The one-chain contract moves the RF front end off `ifft(CFR)` and onto the beat record,
moves `sqrt(P_tx)` to the source, and injects thermal noise once. Two questions follow,
and they have DIFFERENT answers:

1. **Bit parity.** `composition="legacy_impulse"` exists so the stored corpora's gates
   (`tests/test_ml_store_cfr.py`, `tests/test_webapp_live_chain.py`) keep reading
   max |diff| = 0 CODES. A zero tolerance fails on any difference in RNG consumption
   order, so this has to be exact, not close.
2. **Physical fidelity.** Whether the FULL composition would have produced a materially
   different corpus. Measured here as the residual between the two compositions' cubes
   with every stochastic injection off.

MEASURED 2026-09-24, 5 scenes of `b1_demo_cfr_ka` (`benchmark_v1_ka`, 16 RX, 4 TX TDM,
256 chirps, 512 samples, 12 bits), replaying each scene's stored `.cfr.npy` sidecar with
its OWN recorded seeds, impairment severities and IF-HPF corner fed back in:

| arm | result |
|---|---|
| legacy, full chain, vs the stored `adc` | `torch.equal`, max \\|diff\\| = 0.0, all 5 |
| legacy, full chain, vs the stored cube | `torch.equal`, max \\|diff\\| = 0.0, all 5 |
| FULL vs legacy, SIGNAL PATH (all injections off) | rel-RMSE 3.63e-05 .. 4.82e-05, mean 4.08e-05; max \\|diff\\| 3.8e-07 .. 3.4e-06 |
| one LSB, referred into cube units | 2.52e-05 .. 4.05e-05 RMS (0.26%-0.76% of the cube's own RMS) |

So the signal-path residual sits AT or BELOW one LSB in RMS and one to two orders below
it in peak: the placement move is not a numerical change to the corpus, and the legacy
flag is load-bearing for BIT parity only -- "these files on disk were made that way",
not "the physics differ". That is F97c's finding, re-measured on the FULL default.

**The control, and why the measurement is worthless without it.** With noise ON, the two
compositions draw INDEPENDENT realisations (different tensor shapes at different points
in the chain), so the same-variance floors differ by ~sqrt(2) x the floor. Measured that
way the two arms come out at rel-RMSE ~0.97 -- four orders larger, and entirely a
property of the RNG rather than of the placement. An earlier measurement of this exact
question reported 1.7e-1 for precisely that reason. Any future re-measurement that skips
the control is measuring noise.

**One number NOT run to ground, recorded rather than smoothed over.** With noise on, the
ADC-level gap between the FULL replay and the stored corpus (~3.6e-4) is larger than two
independent draws of a 1.26e-6-sigma floor can explain. A plausible contributor: under
the FULL composition the floor is the front end's OWN Friis cascade, not
`cfg.noise_figure_db = 15 dB`, so the two placements' floors may genuinely differ in
LEVEL as well as in realisation, and the leakage/clutter severities are referenced to
whatever floor is local to them. `tests/test_ml_link_budget.py::
test_cfg_noise_figure_is_inert_behind_a_front_end` pins the mechanism; the MAGNITUDE of
the resulting floor difference on this corpus is not measured, and no claim is made
about it here.

Gated on the corpus being present (it is a junction to a data tree on this machine, not
something a clean clone has) and marked `slow`.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from e2e.chain import transforms as T  # noqa: E402
from e2e.chain import link_budget as LB  # noqa: E402
from e2e.chain.frontend import FrontEndBlock  # noqa: E402
from e2e.chain.link_budget import ThermalNoiseBlock  # noqa: E402
from e2e.chain.receive import IFHighPassBlock, ImpairmentBlock, QuantizerBlock  # noqa: E402
from e2e.frames import DOMAIN_CFR  # noqa: E402
from e2e.ml import storage  # noqa: E402
from e2e.ml.blocks import SourceBlock  # noqa: E402
from e2e.ml.chain_generate import build_chain_simulation  # noqa: E402
from e2e.radar_config import PRESETS  # noqa: E402

CORPUS = (Path(__file__).resolve().parent.parent / "e2e" / "ml" / "datasets"
          / "b1_demo_cfr_ka" / "benchmark_v1_ka_D2" / "benchmark_v1_ka_D2")
CFG = PRESETS["benchmark_v1_ka"]

#: Two scenes, not the five the headline numbers were measured on. The per-scene spread
#: of the signal-path residual is 3.63e-05..4.82e-05 -- tight enough that two scenes
#: exercise the same property, and the run is ~40 s/scene on GPU.
N_SCENES = 2

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not CORPUS.is_dir(),
                       reason=f"the Ka demo corpus is not present at {CORPUS}"),
]


def _stored(i, device):
    """`(meta, adc)` for scene `i`, decoded from the corpus's own int16 codec."""
    with np.load(CORPUS / f"sample_scene{i:05d}_frame_00000.npz",
                 allow_pickle=False) as data:
        meta = json.loads(str(data["meta"].item()))
        adc = storage.read_payload(data, meta, meta["payload_key"])
    return meta, torch.as_tensor(np.array(adc)).to(device).to(torch.complex64)


def _stored_cube(adc):
    """`RadarCubeBlock`'s OWN computation, applied outside the block.

    NOT `transforms.adc_to_rd`, and the difference is real rather than pedantic: this
    config is `mimo="tdm"`, and `RadarCubeBlock` de-interleaves the TDM chirps into the
    virtual array before the Doppler FFT while the literal `adc_to_rd` does not. Their
    outputs are `[64, 512, 64]` and `[16, 512, 256]` -- not comparable at all, so a
    comparison written against `adc_to_rd` would fail on a shape rather than on the
    quantity it meant to measure. That asymmetry predates this work; it is named here
    so the substitution is explicit rather than silent.
    """
    cube = T.range_transform_for(CFG).apply({"adc": adc})["cube"]
    cube = T.tdm_deinterleave(CFG, cube)
    cfg = dataclasses.replace(CFG, n_tx=1, mimo="single", n_chirps=CFG.n_chirps_per_tx)
    return T.rd_from_cube(cfg, cube)


def _replay(i, meta, composition, out_dir, *, noiseless, device):
    """Replay scene `i`'s stored CFR through `composition`, returning its radar cube
    and the `adc` its sink wrote."""
    src = SourceBlock(str(CORPUS), tag=f"sample_scene{i:05d}", domain=DOMAIN_CFR)
    sim = build_chain_simulation(
        None, CFG, out_dir, tag=f"replay{i:05d}",
        environment_block=src, composition=composition, device=device,
        quant_bits=meta["quant_bits"],
        rffe_kwargs=({"inject_noise": False} if noiseless else {}),
        if_hpf_kwargs={"corner_hz": meta["if_hpf_corner_hz"],
                       "order": meta.get("if_hpf_order", 2)},
        impairment_chain_params=(
            {"phase_noise": None, "leakage": None, "clutter": None} if noiseless
            else {k: v for k, v in meta["impairment_params"].items()
                  if k in ("phase_noise", "leakage", "clutter")}),
        impairment_seed=int(meta["impairment_params"]["base_seed"]),
    )
    if noiseless:
        for stage in sim.serial_stages:
            if isinstance(stage, ThermalNoiseBlock) and stage.mode == "once":
                # A clean no-op in "once" mode: `sqrt(P_tx)` lives in `TxPowerStage`
                # under the FULL composition, so disabling the floor drops only the
                # floor. The LEGACY arm is handled by the caller instead, because its
                # "legacy" mode bundles `sqrt(P_tx)` INTO the same call as the noise
                # draw and `enabled=False` would silently drop the transmit-power
                # scaling too -- putting the two arms on different absolute footings
                # and turning a placement measurement into an amplitude one.
                stage.enabled = False
    quant = next(s for s in sim.serial_stages if isinstance(s, QuantizerBlock))
    sim.run(n_steps=1)
    return sim.get_outputs()["radar_cube"][0], quant


def _rel_rmse(a, b):
    den = torch.linalg.norm(b.flatten())
    return float(torch.linalg.norm((a - b).flatten()) / den) if den > 0 else float("nan")


def test_the_legacy_composition_reproduces_the_stored_ka_corpus_bit_for_bit(
        tmp_path, torch_device):
    """max |diff| = 0.0 on the stored corpus's own cube, per scene. Not a tolerance.

    This is the gate the `legacy_impulse` flag exists for, run against real stored
    frames rather than against a freshly generated pair. If it ever needs a tolerance,
    the flag has stopped doing its one job and every corpus-backed number (F85, F95, the
    T5 live-vs-stored screen) is about to move for a reason nobody asked for.
    """
    for i in range(N_SCENES):
        meta, stored_adc = _stored(i, torch_device)
        cube, _ = _replay(i, meta, "legacy_impulse", tmp_path / f"legacy{i}",
                          noiseless=False, device=torch_device)
        want = _stored_cube(stored_adc)
        assert cube.shape == want.shape
        diff = float((cube - want).abs().max())
        assert diff == 0.0, f"scene {i}: max |diff| = {diff:.3e} codes, expected 0"


def test_the_full_compositions_signal_path_agrees_with_the_legacy_one_below_one_lsb(
        tmp_path, torch_device):
    """The physical-fidelity half. EVERY stochastic injection is off on both arms --
    front-end noise, thermal floor, phase noise, leakage, clutter -- because with them
    on the two arms draw independent realisations and the measurement returns ~0.97
    rel-RMSE, which is a fact about the RNG and not about the placement.

    Measured 2026-09-24 on 5 scenes: rel-RMSE 3.63e-05..4.82e-05 (mean 4.08e-05),
    max |diff| 3.8e-07..3.4e-06, against one LSB referred into cube units at
    2.52e-05..4.05e-05 RMS. The threshold below is the LSB reference computed for the
    run, not a literal.
    """
    from unittest import mock

    for i in range(N_SCENES):
        meta, _ = _stored(i, torch_device)
        full, quant = _replay(i, meta, "full", tmp_path / f"full{i}",
                              noiseless=True, device=torch_device)
        # The legacy arm's noise draw is patched to identity rather than disabled, so
        # the `sqrt(P_tx)` its ThermalNoiseBlock also applies survives -- see `_replay`.
        with mock.patch.object(LB, "add_thermal_noise", lambda adc, cfg, **kw: adc):
            legacy, _ = _replay(i, meta, "legacy_impulse", tmp_path / f"legsig{i}",
                                noiseless=True, device=torch_device)

        # One LSB, referred into the cube's units by pushing a constant-MAGNITUDE,
        # random-PHASE lsb perturbation through the same transform. Constant magnitude
        # with random phase, not a DC constant: the scored range protocol removes the
        # fast-time mean, which would zero a literal DC probe and report an LSB of
        # nothing.
        g = torch.Generator(device="cpu").manual_seed(0)
        shape = (CFG.n_rx, CFG.n_chirps, CFG.n_samples)
        phase = torch.rand(shape, generator=g) * 2 * np.pi
        probe = (quant.lsb * torch.exp(1j * phase)).to(torch_device).to(torch.complex64)
        lsb_cube_rms = float(torch.linalg.norm(_stored_cube(probe))
                             / np.sqrt(_stored_cube(probe).numel()))

        rel = _rel_rmse(full, legacy)
        peak = float((full - legacy).abs().max())
        rms = float(torch.linalg.norm((full - legacy).flatten())
                    / np.sqrt(full.numel()))
        assert rms <= lsb_cube_rms, (
            f"scene {i}: the two placements' signal paths differ by {rms:.3e} RMS "
            f"against one LSB at {lsb_cube_rms:.3e} -- the placement move is no longer "
            f"below the quantiser, so the stored corpora would have to be regenerated "
            f"rather than merely replayed under the legacy flag (rel-RMSE {rel:.3e}, "
            f"peak {peak:.3e})")
