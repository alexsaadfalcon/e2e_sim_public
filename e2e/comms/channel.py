"""
Channel application, estimation and equalization for the communications layer.

Everything operates in the frequency domain on a per-subcarrier basis, which is
exactly how the rest of the simulator represents the propagation channel
(S-parameters = channel frequency response). This keeps the comms path
consistent with the radar path: both consume a complex frequency response.

Provided here
-------------
* `synthetic_multipath_cfr`  -- a few random multipath taps -> frequency response
  over an arbitrary frequency grid. Used as the *fallback* channel when the
  precomputed Sionna `.pkl` frames are not available (Sionna cannot run here).
* `frame_to_cfr`             -- shared frame->CFR extraction (reshape + per-row
  interpolation onto a target frequency grid) used by every caller that reads a
  raw pipeline/Sionna frame (`load_or_synthesize_cfr`, `comms.blocks`, `main_isac`).
* `cfr_to_subcarriers`       -- resample a (dense) channel frequency response onto
  the OFDM subcarrier grid.
* `apply_channel`            -- multiply an OFDM frequency grid by the per-subcarrier
  channel and add complex AWGN at a target SNR.
* `ls_estimate` / `mmse_estimate` -- pilot-based channel estimation.
* `zf_equalize` / `mmse_equalize` -- per-subcarrier equalization.
* metrics: `ber`, `evm`, `channel_mse`.

All tensors are torch complex64 on the shared `device`.
"""

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------------
# Synthetic channel (Sionna-free fallback)
# --------------------------------------------------------------------------------
def synthetic_multipath_cfr(freqs, n_taps=6, max_delay_s=20e-9, rng=None,
                            rician_k_db=None):
    """Generate a channel frequency response from random multipath taps.

    Parameters
    ----------
    freqs : array of frequency-grid points (Hz), e.g. ``FrequencyPlan.linspace()``.
    n_taps : number of multipath components.
    max_delay_s : maximum tap delay (seconds); delays are drawn uniformly in [0, max].
    rng : optional ``np.random.Generator`` for reproducibility.
    rician_k_db : if given, makes the first tap a dominant LoS component with the
                  specified Rician K-factor (dB); otherwise the channel is Rayleigh.

    Returns
    -------
    H : complex64 torch tensor, shape [len(freqs)] -- the channel frequency response
        H(f) = sum_l a_l * exp(-j 2 pi f tau_l).
    """
    if rng is None:
        rng = np.random.default_rng()
    freqs = np.asarray(freqs, dtype=np.float64)

    delays = rng.uniform(0.0, max_delay_s, size=n_taps)
    delays[0] = 0.0                                  # first tap at zero delay
    # complex Gaussian gains, exponentially decaying power profile
    powers = np.exp(-delays / (0.5 * max_delay_s + 1e-12))
    gains = (rng.standard_normal(n_taps) + 1j * rng.standard_normal(n_taps))
    gains = gains * np.sqrt(powers / 2.0)

    if rician_k_db is not None:
        k = 10 ** (rician_k_db / 10.0)
        # split total power into LoS (deterministic) + scattered (the random taps)
        gains = gains * np.sqrt(1.0 / (k + 1.0))
        gains[0] = np.sqrt(k / (k + 1.0)) + gains[0]   # dominant specular component

    # H(f) = sum_l gain_l exp(-j 2 pi f tau_l)
    phase = np.exp(-2j * np.pi * np.outer(freqs, delays))    # [F, taps]
    H = phase @ gains
    H = H.astype(np.complex64)
    return torch.from_numpy(H).to(device)


def cfr_to_subcarriers(H_dense, freqs_dense, fft_size, carrier_hz, subcarrier_spacing_hz):
    """Resample a dense channel frequency response onto OFDM subcarrier frequencies.

    The OFDM subcarriers sit at ``carrier_hz + (k - fft_size/2) * subcarrier_spacing``
    for k in [0, fft_size). We linearly interpolate the dense CFR (magnitude/phase
    via real/imag parts) onto those subcarrier frequencies.

    Returns a complex64 tensor of length fft_size (fftshift-ordered to match
    ``OFDMModem.demodulate``).
    """
    H_dense = torch.as_tensor(H_dense, dtype=torch.complex64).cpu().numpy()
    freqs_dense = np.asarray(freqs_dense, dtype=np.float64)

    k = np.arange(fft_size)
    sc_freqs = carrier_hz + (k - fft_size / 2.0) * subcarrier_spacing_hz

    # clip to the available band, then interpolate real & imag separately
    sc_clip = np.clip(sc_freqs, freqs_dense.min(), freqs_dense.max())
    Hr = np.interp(sc_clip, freqs_dense, H_dense.real)
    Hi = np.interp(sc_clip, freqs_dense, H_dense.imag)
    H_sc = (Hr + 1j * Hi).astype(np.complex64)
    return torch.from_numpy(H_sc).to(device)


# --------------------------------------------------------------------------------
# Channel application
# --------------------------------------------------------------------------------
def apply_channel(tx_freq, H_sc, snr_db, rng_seed=None):
    """Apply a per-subcarrier channel and add complex AWGN.

    Parameters
    ----------
    tx_freq : [n_symbols, fft_size] transmitted frequency grid.
    H_sc    : [fft_size] channel frequency response (per subcarrier).
    snr_db  : SNR in dB, measured on the *active* (non-zero) subcarriers.

    Returns rx_freq : [n_symbols, fft_size].
    """
    tx_freq = torch.as_tensor(tx_freq, dtype=torch.complex64, device=device)
    H_sc = torch.as_tensor(H_sc, dtype=torch.complex64, device=device)
    rx_clean = tx_freq * H_sc[None, :]

    # noise power referenced to the average received power on used subcarriers
    used = torch.abs(rx_clean) > 0
    if used.any():
        sig_pow = torch.mean(torch.abs(rx_clean[used]) ** 2).item()
    else:
        sig_pow = 1.0
    noise_pow = sig_pow / (10 ** (snr_db / 10.0))

    gen = torch.Generator(device="cpu")
    if rng_seed is not None:
        gen.manual_seed(int(rng_seed))
    shape = tx_freq.shape
    noise = (torch.randn(shape, generator=gen) + 1j * torch.randn(shape, generator=gen))
    noise = noise.to(device) * np.sqrt(noise_pow / 2.0)
    return rx_clean + noise, noise_pow


# --------------------------------------------------------------------------------
# Channel estimation (from pilots)
# --------------------------------------------------------------------------------
def ls_estimate(rx_pilots, tx_pilots, pilot_idx, fft_size):
    """Least-squares channel estimate, interpolated across all subcarriers.

    Parameters
    ----------
    rx_pilots : [n_symbols, n_pilots] received pilot symbols.
    tx_pilots : [n_symbols, n_pilots] known transmitted pilot symbols.
    pilot_idx : [n_pilots] subcarrier indices (fftshift-ordered) of the pilots.
    fft_size  : full subcarrier count.

    Returns H_est : [fft_size] complex64. LS at pilots (H = Y/X), averaged over
    symbols, then linearly interpolated (real/imag) onto the full grid.
    """
    rx_pilots = torch.as_tensor(rx_pilots, dtype=torch.complex64, device=device)
    tx_pilots = torch.as_tensor(tx_pilots, dtype=torch.complex64, device=device)
    H_at_pilots = (rx_pilots / tx_pilots).mean(dim=0)           # [n_pilots]

    pidx = torch.as_tensor(pilot_idx, device=device).cpu().numpy()
    Hp = H_at_pilots.cpu().numpy()
    grid = np.arange(fft_size)
    Hr = np.interp(grid, pidx, Hp.real)
    Hi = np.interp(grid, pidx, Hp.imag)
    H_est = (Hr + 1j * Hi).astype(np.complex64)
    return torch.from_numpy(H_est).to(device)


def mmse_estimate(rx_pilots, tx_pilots, pilot_idx, fft_size, snr_db):
    """Pooled empirical-Bayes Wiener (diagonal MMSE) channel estimate.

    For each pilot subcarrier we have several noisy LS observations (one per OFDM
    symbol); their mean over symbols is the LS estimate at that pilot. The Wiener
    shrinkage ``g = sigma_H^2 / (sigma_H^2 + sigma_n^2 / n_sym)`` needs a signal-
    power prior ``sigma_H^2`` and a noise-power estimate. Deriving either one from
    the *same* handful of noisy pilots being shrunk is circular: at low SNR / few
    pilots that self-estimate is high-variance and can even go negative, making
    this "MMSE" estimator worse than plain LS. Instead both statistics are pooled
    across every pilot subcarrier AND every OFDM symbol in the frame -- the
    maximal averaging available -- giving a single, low-variance, frame-wide
    prior and noise floor. The same pooled shrinkage factor is then applied to
    every pilot's own LS mean (a flat prior across subcarriers -- estimating a
    per-subcarrier power profile without reusing the same noisy samples would
    need more pilots than a comb typically has). The de-noised pilots are
    interpolated onto the full grid exactly as in LS.

    `snr_db` is accepted for API symmetry but the noise power is measured, not
    assumed, so the estimate is robust to the channel not being unit-power.
    """
    rx_pilots = torch.as_tensor(rx_pilots, dtype=torch.complex64, device=device)
    tx_pilots = torch.as_tensor(tx_pilots, dtype=torch.complex64, device=device)
    H_obs = rx_pilots / tx_pilots                              # [n_symbols, n_pilots]
    H_mean = H_obs.mean(dim=0)                                 # LS per pilot

    n_sym = H_obs.shape[0]
    if n_sym > 1:
        # per-pilot variance across symbols -> noise power of a single observation,
        # pooled (averaged) over pilots too: one low-variance, frame-wide estimate
        # instead of n_pilots separate high-variance self-estimates.
        var_per_pilot = torch.mean(torch.abs(H_obs - H_mean[None, :]) ** 2, dim=0)
        sigma_n2_obs = torch.mean(var_per_pilot)                    # scalar
        sigma_n2 = sigma_n2_obs / n_sym                             # noise var of the mean
    else:
        # single symbol: nothing to pool across -> fall back to the assumed SNR
        rho = 10 ** (snr_db / 10.0)
        sigma_n2 = torch.mean(torch.abs(H_mean) ** 2) / rho

    # pooled empirical-Bayes signal-power prior: average |H_mean|^2 over ALL
    # pilots (maximal averaging) then remove the (now scalar, low-variance)
    # noise-of-the-mean bias; clamp so a noisy frame never yields sigma_H2 < 0.
    mean_pow = torch.mean(torch.abs(H_mean) ** 2)
    floor = 1e-6 * mean_pow + 1e-12
    sigma_H2 = torch.clamp(mean_pow - sigma_n2, min=floor)

    gain = sigma_H2 / (sigma_H2 + sigma_n2 + 1e-12)           # single pooled Wiener gain
    H_wiener = H_mean * gain

    pidx = torch.as_tensor(pilot_idx, device=device).cpu().numpy()
    Hp = H_wiener.cpu().numpy()
    grid = np.arange(fft_size)
    Hr = np.interp(grid, pidx, Hp.real)
    Hi = np.interp(grid, pidx, Hp.imag)
    H_est = (Hr + 1j * Hi).astype(np.complex64)
    return torch.from_numpy(H_est).to(device)


# --------------------------------------------------------------------------------
# Equalization
# --------------------------------------------------------------------------------
def zf_equalize(rx_freq, H_est):
    """Zero-forcing equalizer: divide received symbols by the channel estimate."""
    rx_freq = torch.as_tensor(rx_freq, dtype=torch.complex64, device=device)
    H_est = torch.as_tensor(H_est, dtype=torch.complex64, device=device)
    return rx_freq / (H_est[None, :] + 1e-12)


def mmse_equalize(rx_freq, H_est, snr_db, unbiased=True):
    """MMSE equalizer: ``conj(H) / (|H|^2 + 1/SNR)``.

    The raw scalar-MMSE filter ``w = H* / (|H|^2 + 1/rho)`` is *biased*:
    ``E[w H] = b_k = |H_k|^2 / (|H_k|^2 + 1/rho) < 1``, i.e. it shrinks the
    equalized point toward zero, more so on weak subcarriers. A hard-decision
    demapper slices against the *unbiased*, unit-power constellation, so with
    ``unbiased=True`` (default) the per-subcarrier output is rescaled by
    ``1/b_k`` (guarded away from 0) before being handed to the demapper.
    Algebraically ``w / b_k == 1/H_k``, so the unbiased-MMSE decision is
    identical to ZF: for uncoded per-subcarrier hard decisions, unbiased scalar
    MMSE and ZF make the *same* decision on the same received sample -- MMSE's
    real advantage over ZF is lower estimation MSE / noise enhancement at
    weak subcarriers (visible in soft-symbol MSE or EVM, not in uncoded hard-
    decision BER). Pass ``unbiased=False`` to get the raw, biased filter
    output (e.g. to compute that soft-metric advantage).
    """
    rx_freq = torch.as_tensor(rx_freq, dtype=torch.complex64, device=device)
    H_est = torch.as_tensor(H_est, dtype=torch.complex64, device=device)
    rho = 10 ** (snr_db / 10.0)
    h2 = torch.abs(H_est) ** 2
    w = torch.conj(H_est) / (h2 + 1.0 / rho)
    eq = rx_freq * w[None, :]
    if unbiased:
        bias = h2 / (h2 + 1.0 / rho)
        eq = eq / torch.clamp(bias, min=1e-6)[None, :]
    return eq


# --------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------
def ber(tx_bits, rx_bits):
    """Bit-error rate between two flat bit tensors of equal length."""
    tx_bits = torch.as_tensor(tx_bits, device=device).reshape(-1)
    rx_bits = torch.as_tensor(rx_bits, device=device).reshape(-1)
    n = min(tx_bits.numel(), rx_bits.numel())
    if n == 0:
        return float("nan")
    return (tx_bits[:n] != rx_bits[:n]).float().mean().item()


def evm(rx_syms, ref_syms):
    """Error-vector magnitude (RMS, as a fraction) between equalized and ref symbols."""
    rx_syms = torch.as_tensor(rx_syms, dtype=torch.complex64, device=device).reshape(-1)
    ref_syms = torch.as_tensor(ref_syms, dtype=torch.complex64, device=device).reshape(-1)
    n = min(rx_syms.numel(), ref_syms.numel())
    err = rx_syms[:n] - ref_syms[:n]
    ref_pow = torch.mean(torch.abs(ref_syms[:n]) ** 2).item()
    if ref_pow == 0:
        return float("nan")
    return float(np.sqrt(torch.mean(torch.abs(err) ** 2).item() / ref_pow))


def evm_db(rx_syms, ref_syms):
    """EVM expressed in dB (20*log10 of the RMS EVM)."""
    e = evm(rx_syms, ref_syms)
    return 20.0 * np.log10(e + 1e-12)


def channel_mse(H_est, H_true, active_idx=None):
    """Mean-squared error between an estimated and true channel response.

    If `active_idx` is given the MSE is computed only over those subcarriers
    (the band actually used), which is the meaningful comparison.
    """
    H_est = torch.as_tensor(H_est, dtype=torch.complex64, device=device).reshape(-1)
    H_true = torch.as_tensor(H_true, dtype=torch.complex64, device=device).reshape(-1)
    if active_idx is not None:
        active_idx = torch.as_tensor(active_idx, device=device)
        H_est = H_est[active_idx]
        H_true = H_true[active_idx]
    return torch.mean(torch.abs(H_est - H_true) ** 2).item()


# --------------------------------------------------------------------------------
# Frame -> CFR: the shared reshape/interpolate helper
# --------------------------------------------------------------------------------
def frame_to_cfr(frame, target_freqs, src_band=None, element=0):
    """Extract and resample spatial channel(s) from a raw pipeline frame.

    Shared helper behind every frame->CFR extraction in the comms layer
    (`load_or_synthesize_cfr`, `comms.blocks._cfr_from_state`,
    `main_isac._radar_s_pars`) -- previously three independent copies of the
    same reshape + per-row interpolation logic.

    Parameters
    ----------
    frame : array-like (numpy or torch), shape [n_rx, n_tx, chirp, F], or an
        already-flat [n_rx, F]. Any axes between the leading (spatial) axis and
        the trailing (frequency) axis are collapsed and index 0 (tx=0, chirp=0)
        is kept, matching the pipeline's single-chirp/no-MIMO convention.
    target_freqs : the frequency grid (Hz) to interpolate each row onto.
    src_band : optional ``(f_start_hz, f_stop_hz)`` stating the *actual* frequency
        band the frame's samples span. The frame stores only sample values, not
        their frequencies, so the source grid must be supplied or assumed. When
        ``None`` (default) we ASSUME the frame spans exactly the requested
        ``target_freqs`` band, i.e. ``src_band = (target_freqs[0], target_freqs[-1])``.
        If the frame was actually sampled over a different band this default
        silently mis-maps frequency -- pass the true band explicitly whenever
        known (e.g. from a v2 frame's `freq_plan` metadata).
    element : spatial row to return -- an int index into the leading (n_rx) axis,
        or ``None`` to return every row.

    Returns
    -------
    complex64 torch tensor on the library device: shape [len(target_freqs)] for
    an int `element`, or [n_rx, len(target_freqs)] for ``element=None``.
    """
    if torch.is_tensor(frame):
        frame = frame.detach().cpu().numpy()
    arr = np.asarray(frame, dtype=np.complex64)
    if arr.ndim > 2:
        # collapse everything between the leading spatial axis and the trailing
        # frequency axis, keeping index 0 (tx=0, chirp=0)
        arr = arr.reshape(arr.shape[0], -1, arr.shape[-1])[:, 0, :]

    target_freqs = np.asarray(target_freqs, dtype=np.float64)
    f0, f1 = (src_band if src_band is not None
              else (target_freqs[0], target_freqs[-1]))
    src_freqs = np.linspace(f0, f1, arr.shape[-1])

    def _interp_row(row):
        Hr = np.interp(target_freqs, src_freqs, row.real)
        Hi = np.interp(target_freqs, src_freqs, row.imag)
        return Hr + 1j * Hi

    if element is None:
        out = np.stack([_interp_row(arr[i]) for i in range(arr.shape[0])], axis=0)
    else:
        out = _interp_row(arr[element])
    out = out.astype(np.complex64)
    return torch.from_numpy(out).to(device)


# --------------------------------------------------------------------------------
# Channel sourcing: precomputed Sionna frame if present, else synthetic fallback
# --------------------------------------------------------------------------------
def load_or_synthesize_cfr(scenario_name, freqs, frame=0, n_taps=6, rng=None,
                           prefer_pkl=True, src_band=None):
    """Return ``(cfr_dense, source_str)`` over `freqs`.

    Attempts to read a precomputed Sionna `.pkl` frame via
    ``e2e.environment.sionna_iterator`` for `scenario_name`. The S-parameters
    there are read and resampled onto `freqs` by `frame_to_cfr` (single spatial
    channel, element 0). If the `.pkl` is absent (or Sionna frames can't be
    loaded), falls back to ``synthetic_multipath_cfr`` so examples always run.

    Parameters
    ----------
    src_band : optional ``(f_start_hz, f_stop_hz)``, forwarded to `frame_to_cfr`
        (see its docstring for the band-guess assumption when omitted). If
        omitted AND the loaded iterator exposes v2 frame metadata (a
        `freq_plan` with `start_hz`/`stop_hz`), that band is used automatically
        instead of guessing -- legacy (pre-v2) pkls still fall back to the
        "assume it spans `freqs`" default.

    `source_str` is 'sionna:<name>' or 'synthetic', for logging.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    if prefer_pkl:
        try:
            from e2e.environment import sionna_iterator as si
            iters = {"munich": getattr(si, "SionnaMunichIterator", None),
                     "etoile": getattr(si, "SionnaEtoileIterator", None)}
            factory = iters.get(scenario_name)
            if factory is not None:
                it = factory()                          # raises if .pkl missing
                arr = it[frame % len(it)]
                band = src_band
                if band is None:
                    freq_plan = getattr(it, "freq_plan", None)   # v2 meta, or None
                    if freq_plan is not None:
                        band = (freq_plan["start_hz"], freq_plan["stop_hz"])
                H = frame_to_cfr(arr, freqs, src_band=band, element=0)
                return H, f"sionna:{scenario_name}"
        except Exception:
            pass   # any load failure -> synthetic fallback

    H = synthetic_multipath_cfr(freqs, n_taps=n_taps, rng=rng)
    return H, "synthetic"
