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
    """Simple per-pilot Wiener (diagonal MMSE) channel estimate.

    For each pilot subcarrier we have several noisy LS observations (one per OFDM
    symbol). Their sample mean is the LS estimate; their sample variance across
    symbols is a direct, *unbiased* estimate of the per-pilot noise power. The
    Wiener gain ``g = sigma_H^2 / (sigma_H^2 + sigma_n^2)`` then shrinks each pilot
    toward zero only as much as the measured noise warrants -- so strong, clean
    pilots are left untouched and the estimate never floors above LS at high SNR.
    The de-noised pilots are interpolated onto the full grid exactly as in LS.

    `snr_db` is accepted for API symmetry but the noise power is measured, not
    assumed, so the estimate is robust to the channel not being unit-power.
    """
    rx_pilots = torch.as_tensor(rx_pilots, dtype=torch.complex64, device=device)
    tx_pilots = torch.as_tensor(tx_pilots, dtype=torch.complex64, device=device)
    H_obs = rx_pilots / tx_pilots                              # [n_symbols, n_pilots]
    H_mean = H_obs.mean(dim=0)                                 # LS per pilot

    n_sym = H_obs.shape[0]
    if n_sym > 1:
        # variance across symbols -> noise power of a single observation;
        # the mean of n_sym observations has 1/n_sym of that variance.
        var_obs = torch.mean(torch.abs(H_obs - H_mean[None, :]) ** 2, dim=0)
        sigma_n2 = var_obs / n_sym
    else:
        # single symbol: fall back to the assumed SNR
        rho = 10 ** (snr_db / 10.0)
        sigma_n2 = torch.mean(torch.abs(H_mean) ** 2) / rho * torch.ones_like(torch.abs(H_mean))

    sigma_H2 = torch.clamp(torch.abs(H_mean) ** 2 - sigma_n2, min=0.0)   # signal power
    gain = sigma_H2 / (sigma_H2 + sigma_n2 + 1e-12)           # per-pilot Wiener gain
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


def mmse_equalize(rx_freq, H_est, snr_db):
    """MMSE equalizer: ``conj(H) / (|H|^2 + 1/SNR)``."""
    rx_freq = torch.as_tensor(rx_freq, dtype=torch.complex64, device=device)
    H_est = torch.as_tensor(H_est, dtype=torch.complex64, device=device)
    rho = 10 ** (snr_db / 10.0)
    w = torch.conj(H_est) / (torch.abs(H_est) ** 2 + 1.0 / rho)
    return rx_freq * w[None, :]


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
# Channel sourcing: precomputed Sionna frame if present, else synthetic fallback
# --------------------------------------------------------------------------------
def load_or_synthesize_cfr(scenario_name, freqs, frame=0, n_taps=6, rng=None,
                           prefer_pkl=True, src_band=None):
    """Return ``(cfr_dense, source_str)`` over `freqs`.

    Attempts to read a precomputed Sionna `.pkl` frame via
    ``e2e.environment.sionna_iterator`` for `scenario_name`. The S-parameters
    there are sampled on the scenario's own dense frequency grid; we take a single
    spatial channel (element 0) and interpolate it onto `freqs`. If the `.pkl` is
    absent (or Sionna frames can't be loaded), falls back to
    ``synthetic_multipath_cfr`` so examples always run.

    Parameters
    ----------
    src_band : optional ``(f_start_hz, f_stop_hz)`` stating the *actual* frequency
        band the loaded `.pkl`'s S-parameters span. The frame stores only sample
        values, not their frequencies, so the source grid must be supplied or
        assumed. When ``None`` (default, preserving the historical behaviour) we
        ASSUME the pkl spans exactly the requested ``freqs`` band, i.e.
        ``src_band = (freqs[0], freqs[-1])``. If the pkl was actually sampled over
        a different band this default silently mis-maps frequency, so pass the
        true band explicitly whenever it is known.

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
                arr = np.asarray(it[frame % len(it)], dtype=np.complex64)
                flat = arr.reshape(-1, arr.shape[-1])   # [N_RX*..., N_FREQS]
                cfr = flat[0]
                # build the source frequency grid: explicit band if given, else
                # assume the pkl spans the requested band (see src_band docstring).
                f0, f1 = (src_band if src_band is not None
                          else (freqs[0], freqs[-1]))
                src_freqs = np.linspace(f0, f1, cfr.shape[-1])
                Hr = np.interp(freqs, src_freqs, cfr.real)
                Hi = np.interp(freqs, src_freqs, cfr.imag)
                H = (Hr + 1j * Hi).astype(np.complex64)
                return torch.from_numpy(H).to(device), f"sionna:{scenario_name}"
        except Exception:
            pass   # any load failure -> synthetic fallback

    H = synthetic_multipath_cfr(freqs, n_taps=n_taps, rng=rng)
    return H, "synthetic"
