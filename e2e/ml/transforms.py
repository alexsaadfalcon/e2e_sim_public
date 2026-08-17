"""
Signal-processing transforms: raw FMCW ADC -> network-input tensors.

Pure torch (complex64/float32), device-preserving, matching the rest of the
simulator's tensor conventions. Nothing here imports Sionna or touches the
precomputed frames; it operates purely on the raw ADC cube produced by
`e2e.ml.rd_synth` (or any array with the same shape/dtype).

Adapted (structure only, not code) from the RADIal / FFTRadNet reference
`SignalProcessing/rpl.py` 'RD' processing chain (per-chirp DC removal -> Hann
window -> range FFT -> Hann window -> Doppler FFT), reimplemented here in pure
torch instead of `mkl_fft`/`cupy` so it runs on the library's device.

Conventions
-----------
* Raw ADC: `[n_rx, n_chirps, n_samples]` complex64 (one frame).
* Range-Doppler cube ("RD"): `[n_rx, range_bin, doppler_bin]` complex64 --
  `range_bin` has `n_samples` entries (fast-time FFT output), `doppler_bin` has
  `n_chirps` entries (slow-time FFT output, zero-Doppler centred via fftshift).
* Network input: `[2*n_rx, range_bin, doppler_bin]` float32, real channels
  first then imaginary channels (FFTRadNet convention), channel axis first
  (not channel-last as in the numpy reference).

`cfg` arguments are duck-typed (see `RadarConfig` in `radar_config.py`): only
the attributes actually used are read, so tests can pass a plain stub with
matching field names without importing torch-heavy sibling modules.
"""

import torch


# --------------------------------------------------------------------------------
# Range-Doppler processing
# --------------------------------------------------------------------------------
def adc_to_rd(cfg, adc):
    """Raw ADC -> Range-Doppler cube.

    adc : complex64 `[n_rx, n_chirps, n_samples]`.
    Returns RD complex64 `[n_rx, range_bin, doppler_bin]` (range_bin count ==
    n_samples, doppler_bin count == n_chirps, zero-Doppler at the centre bin
    `n_chirps // 2` after `fftshift`).

    Chain (matches rpl.py's 'RD' method, pure torch):
      1. remove per-(rx,chirp) DC offset (mean over the sample axis);
      2. Hann window on the sample axis, FFT over samples (range);
      3. Hann window on the chirp axis, FFT over chirps + fftshift (Doppler).
    """
    adc = torch.as_tensor(adc, dtype=torch.complex64)
    if adc.dim() != 3:
        raise ValueError(f"adc must be [n_rx, n_chirps, n_samples], got shape {tuple(adc.shape)}")
    n_rx, n_chirps, n_samples = adc.shape
    if cfg is not None:
        cfg_chirps = getattr(cfg, "n_chirps", None)
        cfg_samples = getattr(cfg, "n_samples", None)
        if cfg_chirps is not None and cfg_chirps != n_chirps:
            raise ValueError(f"adc has {n_chirps} chirps but cfg.n_chirps={cfg_chirps}")
        if cfg_samples is not None and cfg_samples != n_samples:
            raise ValueError(f"adc has {n_samples} samples but cfg.n_samples={cfg_samples}")

    # 1. per-chirp DC removal (mean over fast-time/sample axis)
    x = adc - adc.mean(dim=-1, keepdim=True)

    # 2. range FFT: Hann window over samples, FFT over the sample axis
    range_win = torch.hann_window(n_samples, periodic=False, dtype=torch.float32, device=x.device)
    x = x * range_win.to(x.dtype)
    range_fft = torch.fft.fft(x, n=n_samples, dim=-1)              # [n_rx, n_chirps, range_bin]

    # 3. doppler FFT: Hann window over chirps, FFT + fftshift over the chirp axis
    doppler_win = torch.hann_window(n_chirps, periodic=False, dtype=torch.float32, device=x.device)
    y = range_fft * doppler_win.to(x.dtype)[None, :, None]
    doppler_fft = torch.fft.fft(y, n=n_chirps, dim=1)
    doppler_fft = torch.fft.fftshift(doppler_fft, dim=1)           # zero-Doppler -> centre bin

    # swap (chirp, range) -> (range, doppler) so the returned axis order matches the docstring
    rd = doppler_fft.transpose(1, 2).contiguous()                  # [n_rx, range_bin, doppler_bin]
    return rd.to(torch.complex64)


# --------------------------------------------------------------------------------
# TDM-MIMO virtual array formation
# --------------------------------------------------------------------------------
def tdm_deinterleave(cfg, adc):
    """De-interleave TDM-MIMO chirps into a virtual-array ADC cube.

    Requires `cfg.mimo == "tdm"`. Chirp `c` was fired by TX `c % cfg.n_tx`
    (round-robin firing order); the chirps belonging to TX `t` are gathered
    into virtual-array row `t * n_rx + rx` for each physical Rx `rx`.

    adc : complex64 `[n_rx, n_chirps, n_samples]`.
    Returns complex64 `[n_virtual, n_chirps_per_tx, n_samples]` where
    `n_virtual = cfg.n_tx * n_rx` and `n_chirps_per_tx = n_chirps // cfg.n_tx`.
    Raises ValueError if `n_chirps` is not divisible by `cfg.n_tx`.

    Honesty note (no per-target Doppler compensation): a target's radial
    velocity advances its phase between one TX's chirps and the next TX's
    chirps (they were transmitted `n_tx` chirp-periods apart), so the
    synthesized virtual array is only exactly coherent for stationary targets.
    Correcting this properly requires the per-target Doppler, which is not
    known before detection (chicken-and-egg with the very Doppler processing
    this cube feeds into). This function therefore does NOT apply any
    per-target phase de-rotation -- it is a pure reorganization of chirps --
    and leaves a residual `exp(-j*2*pi*f_D*t*T_c)` phase term across the
    `n_tx` virtual-array groups for moving targets. This is the standard
    "raw TDM deinterleave" used for ML dataset generation: the network is
    expected to learn to cope with (or exploit) this residual phase, the same
    way FFTRadNet's `MIMO_PreEncoder` learns the DDMA/TDM demux end to end
    rather than having it hand-corrected upstream.
    """
    mimo = getattr(cfg, "mimo", None)
    if mimo != "tdm":
        raise ValueError(f"tdm_deinterleave requires cfg.mimo == 'tdm', got {mimo!r}")
    adc = torch.as_tensor(adc, dtype=torch.complex64)
    if adc.dim() != 3:
        raise ValueError(f"adc must be [n_rx, n_chirps, n_samples], got shape {tuple(adc.shape)}")
    n_rx, n_chirps, n_samples = adc.shape
    n_tx = int(cfg.n_tx)
    if n_chirps % n_tx != 0:
        raise ValueError(f"n_chirps ({n_chirps}) is not divisible by n_tx ({n_tx})")
    n_chirps_per_tx = n_chirps // n_tx
    n_virtual = n_tx * n_rx

    out = torch.empty((n_virtual, n_chirps_per_tx, n_samples), dtype=adc.dtype, device=adc.device)
    for t in range(n_tx):
        # chirps fired by TX t, in firing order: c = t, t+n_tx, t+2*n_tx, ...
        out[t * n_rx:(t + 1) * n_rx, :, :] = adc[:, t::n_tx, :]
    return out


def ddma_demux(cfg, rd):
    """De-multiplex a DDMA Range-Doppler cube into a virtual-array cube.

    The TDM counterpart above works on raw ADC, because TDM separates transmitters in
    TIME and you can just pick out chirps. DDMA separates them in DOPPLER -- every TX
    fires on every chirp, TX `t` carrying an extra per-chirp phase `2*pi*t*c/n_tx` -- so
    the separation only exists after the Doppler FFT. Hence this takes an RD cube, not
    ADC.

    rd : complex `[n_rx, n_range, n_doppler]` as `adc_to_rd` returns it, i.e. with
    zero-Doppler at the centre bin (`fftshift` applied).
    Returns complex `[n_tx * n_rx, n_range, n_doppler // n_tx]`, virtual element
    `v = t * n_rx + r`, matching the ULA indexing `rd_synth` synthesizes against
    (TX `t` sits at `t * n_rx * lambda/2`, so the pair (t, r) is virtual element
    `t*n_rx + r` of a uniform lambda/2 array). Each sub-band keeps the centre-bin
    zero-Doppler convention of its input.

    Why it matters: without this, an angle FFT over a DDMA cube runs across the
    `n_rx` PHYSICAL receivers instead of the `n_tx * n_rx` virtual elements. For the
    `radial_like` configuration that is 16 elements standing in for 192 -- a 12x loss
    of angular resolution, which shows up as targets smeared into horizontal ridges
    across the whole range-azimuth map.

    The extraction: TX `t`'s echo is shifted by `t/n_tx` of the Doppler PRF, so its
    replica of a target at true Doppler bin `k` lands at `k + t*n_doppler/n_tx`. Taking
    the `n_doppler/n_tx` bins CENTRED on `t*n_doppler/n_tx` recovers TX `t` alone, with
    zero Doppler back at the middle of the band. Centred, not `[t*n_sub, (t+1)*n_sub)`:
    a negative Doppler wraps to the top of the natural-order axis and would otherwise
    be attributed to the wrong transmitter.

    Unambiguous Doppler shrinks by `n_tx`, which is inherent to DDMA and already
    reflected in `RadarConfig.max_velocity_mps`; a target beyond it aliases into a
    neighbouring transmitter's band and this function cannot tell the difference.
    """
    mimo = getattr(cfg, "mimo", None)
    if mimo != "ddma":
        raise ValueError(f"ddma_demux requires cfg.mimo == 'ddma', got {mimo!r}")
    rd = torch.as_tensor(rd)
    if rd.dim() != 3:
        raise ValueError(f"rd must be [n_rx, n_range, n_doppler], got shape {tuple(rd.shape)}")
    n_rx, n_range, n_doppler = rd.shape
    n_tx = int(cfg.n_tx)
    if n_tx < 1:
        raise ValueError(f"n_tx must be >= 1, got {n_tx}")
    if n_doppler % n_tx != 0:
        raise ValueError(
            f"n_doppler ({n_doppler}) is not divisible by n_tx ({n_tx}); the DDMA "
            "sub-bands would not be equal sizes")
    n_sub = n_doppler // n_tx

    # Undo adc_to_rd's fftshift so bin 0 is zero Doppler and the t*n_sub offsets are
    # literal indices; each extracted band is then re-centred by construction.
    rd_nat = torch.fft.ifftshift(rd, dim=-1)
    offsets = torch.arange(n_sub, device=rd.device) - n_sub // 2

    out = torch.empty((n_tx * n_rx, n_range, n_sub), dtype=rd.dtype, device=rd.device)
    for t in range(n_tx):
        idx = (t * n_sub + offsets) % n_doppler
        out[t * n_rx:(t + 1) * n_rx] = rd_nat.index_select(-1, idx)
    return out


# --------------------------------------------------------------------------------
# Network-input formatting
# --------------------------------------------------------------------------------
def rd_to_input(rd):
    """Complex RD cube `[C, R, D]` -> real/imag-stacked float32 `[2*C, R, D]`.

    FFTRadNet convention: real channels first, then imaginary channels
    (channel-first here, matching this repo's `[rx/virtual, ...]`-leading
    layout rather than the numpy reference's channel-last layout).
    """
    rd = torch.as_tensor(rd)
    if not torch.is_complex(rd):
        raise ValueError("rd_to_input expects a complex input tensor")
    return torch.cat([rd.real, rd.imag], dim=0).to(torch.float32)


def input_stats(x):
    """Per-channel mean/std of a `[C, ...]` tensor, reduced over all other dims.

    Returns `(mean, std)`, each a 1-D float32 tensor of length C.
    """
    x = torch.as_tensor(x, dtype=torch.float32)
    flat = x.reshape(x.shape[0], -1)
    return flat.mean(dim=1), flat.std(dim=1, unbiased=False)


def normalize(x, mean, std):
    """Per-channel `(x - mean) / std` for a `[C, ...]` tensor.

    `mean`/`std` are 1-D tensors of length C (as returned by `input_stats`).
    """
    x = torch.as_tensor(x, dtype=torch.float32)
    mean = torch.as_tensor(mean, dtype=torch.float32, device=x.device)
    std = torch.as_tensor(std, dtype=torch.float32, device=x.device)
    shape = (x.shape[0],) + (1,) * (x.dim() - 1)
    return (x - mean.reshape(shape)) / std.reshape(shape)


# --------------------------------------------------------------------------------
# Debug / visualization helper
# --------------------------------------------------------------------------------
def rd_power_db(rd):
    """|RD|^2 in dB, normalized to the peak bin (peak -> 0 dB). For debug/plotting only."""
    rd = torch.as_tensor(rd)
    power = torch.abs(rd) ** 2
    peak = power.max()
    eps = torch.finfo(torch.float32).tiny
    return 10.0 * torch.log10((power / peak).clamp_min(eps)).to(torch.float32)
