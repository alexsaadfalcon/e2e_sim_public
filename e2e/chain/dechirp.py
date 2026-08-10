"""``DechirpBlock`` -- THE BRIDGE from the frequency domain (CFR) to RX time (ADC).

This is the receive-side half of the chain's two domain bridges (the other is the
transmit-side modulate block): a channel frequency response ``s_pars``
``[n_rx, n_tx, n_chirp, n_freqs]`` in, a dechirped ADC cube ``adc``
``[n_rx, n_chirp, n_samples]`` out. See ``e2e/frames.py`` (``DOMAIN_CFR`` /
``DOMAIN_RX_TIME``) for the domain contract this declares and enforces.

The math is ``e2e.ml.rt_gen``'s (see that module's docstring, "The CFR -> beat mapping"
and equation (3), for the full derivation): sampling a CFR on the FMCW ramp's frequency
grid and conjugating it IS the dechirped beat sample; the antenna index is reversed to
match this project's ULA handedness convention (see rt_gen's "Element ordering / array
handedness" section); TDM/DDMA MIMO combining then collapses the TX axis. That
derivation was validated against re-traced ground truth -- the conjugate, the ``f_c =
f0 + B/2`` chirp-centre choice baked into the frequency grid the caller samples at, and
the antenna-index reversal must not drift. This module is now the ONE implementation of
the beat-mapping and MIMO-combine steps; ``e2e.ml.rt_gen`` builds the raw CFR (the
Sionna-specific half: chunked ``Paths.cfr`` calls) and delegates the rest here so the
math exists in exactly one place. See ``tests/test_chain_dechirp.py`` for the
bit-exactness check against the pre-refactor reference math.
"""

from __future__ import annotations

import math

import torch

from e2e.frames import CHIRP_NATIVE, DOMAIN_CFR, DOMAIN_RX_TIME, FrameCapabilities

# Mirrors `e2e.ml.rt_gen._ANTENNA_INDEX_REVERSED` -- see that module's docstring
# ("Element ordering / array handedness") for the derivation. This is now where the
# reversal is actually applied; rt_gen's own flag stays in sync (see its module
# docstring) but no longer performs the reversal itself.
ANTENNA_INDEX_REVERSED = True


def beat_from_cfr(s_pars: torch.Tensor) -> torch.Tensor:
    """CFR -> dechirped beat cube: complex-conjugate, with the RX/TX antenna axes
    (dims 0/1) reversed (see the module docstring). Shape-preserving; works on any
    tensor whose leading two axes are antenna indices (the raw
    ``[n_rx, n_tx, n_chirp, n_freqs]`` frame, or an intermediate per-chunk cube of the
    same rank) -- callers needing MIMO combining follow this with `mimo_combine`.
    """
    beat = s_pars.conj()
    if ANTENNA_INDEX_REVERSED:
        beat = torch.flip(beat, dims=(0, 1))
    return beat


def mimo_combine(cfg, beat: torch.Tensor) -> torch.Tensor:
    """Beat cube `[n_rx, n_tx, n_chirps, n_samples]` -> ADC cube `[n_rx, n_chirps, n_samples]`.

    Mirrors `e2e.ml.rd_synth.synthesize_adc`'s per-chirp TX factor exactly (see
    `e2e.ml.rt_gen`'s module docstring):

    * `"tdm"` / `"single"`: chirp `c` is transmitted by TX `c % n_tx` alone, so only
      that TX's column survives -- this is the selection `e2e.ml.transforms.
      tdm_deinterleave` inverts.
    * `"ddma"`: every TX transmits on every chirp, TX `t` carrying the extra per-chirp
      phase `2pi t c / n_tx`; the TX columns are summed with that code applied.

    `cfg.mimo` names the scheme (case-insensitive); an unrecognized value raises
    `ValueError`.
    """
    mimo = str(cfg.mimo).lower()
    n_rx, n_tx, n_chirps, n_samples = beat.shape
    if mimo in ("tdm", "single"):
        tx_of_chirp = torch.arange(n_chirps, device=beat.device) % n_tx
        chirp_idx = torch.arange(n_chirps, device=beat.device)
        return beat[:, tx_of_chirp, chirp_idx, :]
    if mimo == "ddma":
        t_idx = torch.arange(n_tx, device=beat.device).view(1, -1, 1)
        c_idx = torch.arange(n_chirps, device=beat.device).view(1, 1, -1)
        code = torch.exp(2j * math.pi * t_idx * c_idx / n_tx).to(beat.dtype)
        return torch.einsum("rtcn,xtc->rcn", beat, code)
    raise ValueError(f"unsupported mimo scheme {cfg.mimo!r}")


class DechirpBlock:
    """The frequency-domain -> RX-time bridge: `s_pars` (DOMAIN_CFR) -> `adc`
    (DOMAIN_RX_TIME).

    Consumes the current-frame `s_pars` (assumed already sampled on the FMCW ramp's
    beat-frequency grid, e.g. by `RTEnvironmentBlock` / `e2e.ml.rt_gen.rt_cfr_frame`),
    applies `beat_from_cfr` then `mimo_combine`, and emits `adc` +
    `signal_domain=DOMAIN_RX_TIME`. `accepts_mimo=True` because MIMO combining (the TX
    axis, dim 1) is exactly this block's job, not something upstream must have already
    resolved; `chirps=CHIRP_NATIVE` because it consumes the chirp axis directly (every
    chirp's TX selection/code depends on its own chirp index).

    `cfg` is a `e2e.ml.radar_config.RadarConfig`-like object; only `cfg.mimo` is read
    (the frequency grid `s_pars` was sampled at is the caller's concern, not this
    block's -- see `beat_from_cfr`'s docstring).
    """

    frame_capabilities = FrameCapabilities(
        domain=DOMAIN_CFR, emits_domain=DOMAIN_RX_TIME,
        accepts_mimo=True, chirps=CHIRP_NATIVE,
    )

    def __init__(self, cfg):
        self.cfg = cfg

    def apply(self, state):
        s_pars = state["s_pars"]
        beat = beat_from_cfr(s_pars)
        adc = mimo_combine(self.cfg, beat)
        return {"adc": adc.to(torch.complex64), "signal_domain": DOMAIN_RX_TIME}
