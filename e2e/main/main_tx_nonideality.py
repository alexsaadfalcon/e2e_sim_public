"""
TX power-amplifier (PA) non-ideality A/B: ideal transmitter vs `TxPA`-distorted
transmitter, on the same OFDM/ISAC waveform.

Why this example exists
------------------------
`e2e/circuit/tx_pa.py`'s `TxPA` (Rapp AM/AM + saturating AM/PM + a frequency-domain
mismatch ripple) is already wired into the radar corpus generator and the webapp, but
every comms/ISAC example script (`main_comms_link`, `main_isac`, `main_channel_estimation`,
`main_comms_head`, `main_isac_multilink`) runs an ideal, distortion-free transmitter.
That is exactly backwards: constant-envelope radar waveforms barely notice a PA's
envelope nonlinearity (see `tx_pa.py`'s module docstring), while OFDM's high peak-to-
average power ratio (PAPR, typically 8-12 dB for dense QAM) drives the PA across a wide
swing of its AM/AM curve every symbol -- comms/ISAC is precisely the case the PA model
was built to matter for, and it was untested there. This script makes that cost visible.

What it does
------------
Builds ONE OFDM frame (`e2e.comms.ofdm.OFDMModem`) and pushes it through two transmit
paths that are identical except for the PA:

* **ideal**  -- `X(f)` unmodified (a flat, distortion-free transmitter).
* **non-ideal** -- the same time-domain waveform run through `TxPA.apply()` (memoryless
  AM/AM + AM/PM), plus `TxPA.frequency_response()`'s linear mismatch ripple, before it
  re-enters the frequency domain.

Both arms then see the SAME propagation channel (a precomputed Sionna munich frame if
present, else the synthetic multipath fallback -- see `e2e.comms.channel`) and the SAME
AWGN realization (fixed seed, independent of backoff/arm, for a paired A/B), then the
SAME pilot-LS estimate + zero-forcing equalizer. Nothing here reimplements modulation,
demodulation, equalization or the EVM/BER metrics -- all of that is `e2e.comms.ofdm` /
`e2e.comms.channel` verbatim, exactly per this repo's "don't reimplement the modem"
convention.

Why not `e2e.chain.waveform`'s `WaveformBlock`/`TxPABlock`/`ModulateBlock`?
----------------------------------------------------------------------------
Those blocks are the right way to wire a PA into the `Simulation` state-dict pipeline,
but they operate at the modem's own critically-sampled rate (`n_freqs`-point DFT tied
to `s_pars`'s grid). A memoryless nonlinearity generates spectral regrowth OUTSIDE the
original occupied band; sampled at the modem's own rate that regrowth aliases straight
back in-band, which would overstate in-band EVM and make the PSD plot meaningless (no
room to see the regrowth next to the main lobe -- it would already be folded in). This
script instead zero-pads `OFDMModem.modulate`'s frequency grid and takes a longer IFFT
(`_oversample_time` / `_undersample_freq` below) to get a properly oversampled TX-time
signal, applies `TxPA` there, and de-aliases back down to the modem's own subcarrier
grid for equalization/demapping -- the same underlying `TxPA` engine, just exercised
at a rate that can actually show what it does to the spectrum. `_oversample_time` and
`_undersample_freq` are EXACT inverses of each other when nothing is done to the signal
in between (zero-padding a spectrum then IFFT/FFT-ing at the padded length is a lossless
sinc interpolation) -- checked by this module's own tests.

Backoff convention
-------------------
Input backoff (IBO) is defined relative to `a_knee = TxPAConfig.a_sat /
10**(small_signal_gain_db/20)`, the input amplitude at which the LINEAR (small-signal)
extrapolation of the AM/AM curve would just reach saturation -- the standard textbook
IBO reference point, even though the smooth Rapp knee means the amplifier is already
visibly compressing a couple dB before that point. The OFDM waveform is normalized to
unit average power, then scaled so its RMS amplitude sits `backoff_db` below `a_knee`;
by construction (verified algebraically in this module and empirically in its tests)
`10*log10(a_knee**2 / mean(|scaled|**2)) == backoff_db`.

Honesty about what this PSD/ACPR plot is NOT
-----------------------------------------------
`TxPA` is MEMORYLESS (see `tx_pa.py`'s "KNOWN LIMITATIONS"): the output at time t
depends only on the input envelope at t, so the third/fifth-order intermodulation
products it generates are, by construction, symmetric about the carrier. A real PA's
spectral regrowth is NOT generally symmetric (memory effects -- thermal drift, bias-
network dynamics -- break the symmetry). This script measures and PRINTS the upper/
lower ACPR difference for the non-ideal arm specifically so that near-zero number is
visible as a red flag, not mistaken for a hardware validation: DO NOT read the PSD
plot here as an ACPR/spectral-mask compliance prediction for a real amplifier.

Run:
    python -m e2e.main.main_tx_nonideality

Outputs (e2e/main/figures/, not committed):
    tx_nonideality_evm_vs_backoff.png   headline: EVM vs input backoff, ideal vs non-ideal
    tx_nonideality_constellation.png    constellation at the most aggressive backoff
    tx_nonideality_psd.png              PSD / spectral regrowth, ACPR annotated
    tx_nonideality_range_profile.png    ISAC sensing-side cost at the same backoff
"""

import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")            # headless: write figures to files, no display
import matplotlib.pyplot as plt

from e2e.scenario import munich_radar_scenario
from e2e.comms.ofdm import OFDMModem, random_bits
from e2e.comms import channel as ch
from e2e.comms import isac
from e2e.circuit.tx_pa import TxPA, TxPAConfig
from e2e.viz import fig_dir, to_db


FIG_DIR = fig_dir(__file__)

_C_LIGHT = 299_792_458.0


# --------------------------------------------------------------------------------
# Oversample / de-alias: exact inverses of each other (see module docstring)
# --------------------------------------------------------------------------------
def _oversample_time(tx_freq, fft_size, cp_len, oversample):
    """Zero-pad `tx_freq` (fftshift-order, as `OFDMModem.modulate` produces it) to
    `fft_size * oversample` bins and IFFT, mirroring `modulate`'s own
    ifftshift-then-ifft convention but at a higher rate. Returns
    (time_with_cp [n_symbols, (fft_size+cp_len)*oversample], off) where `off` is the
    index of the original band's first bin in the padded grid (needed to invert).
    """
    n_symbols = tx_freq.shape[0]
    n_up = fft_size * oversample
    padded = torch.zeros(n_symbols, n_up, dtype=torch.complex64, device=tx_freq.device)
    off = (n_up - fft_size) // 2
    padded[:, off:off + fft_size] = tx_freq
    grid = torch.fft.ifftshift(padded, dim=-1)
    time = torch.fft.ifft(grid, dim=-1)
    cp_up = cp_len * oversample
    cp = time[:, -cp_up:] if cp_up > 0 else time[:, :0]
    return torch.cat([cp, time], dim=-1), off


def _undersample_freq(time_up, fft_size, cp_len, oversample, off):
    """Inverse of `_oversample_time`: strip the (oversampled) CP, FFT at the SAME
    padded length -- so any spectral regrowth a nonlinearity applied to `time_up`
    is resolved without aliasing -- then slice out the original fft_size-wide band.
    """
    cp_up = cp_len * oversample
    if cp_up > 0:
        time_up = time_up[:, cp_up:]
    grid = torch.fft.fftshift(torch.fft.fft(time_up, dim=-1), dim=-1)
    return grid[:, off:off + fft_size]


# --------------------------------------------------------------------------------
# PSD / ACPR (plain Bartlett periodogram -- no scipy dependency, matching the rest
# of e2e.comms which is numpy+torch only)
# --------------------------------------------------------------------------------
def _periodogram(time_data, fs_hz):
    """Bartlett-averaged periodogram: average |FFT|^2 across OFDM symbols to cut
    single-symbol PSD variance. `time_data`: [n_symbols, n_up], CP already stripped.
    Returns (freqs_hz [n_up], psd [n_up]), both ascending, DC-centered.
    """
    n_up = time_data.shape[-1]
    X = torch.fft.fftshift(torch.fft.fft(time_data, dim=-1), dim=-1)
    psd = torch.mean(torch.abs(X) ** 2, dim=0) / n_up
    faxis = torch.fft.fftshift(torch.fft.fftfreq(n_up, d=1.0 / fs_hz)).to(psd.device)
    return faxis, psd


def _band_power(faxis, psd, f_lo, f_hi):
    mask = (faxis >= f_lo) & (faxis < f_hi)
    if not bool(mask.any()):
        return 0.0
    return float(torch.sum(psd[mask]).item())


def _acpr(faxis, psd, occupied_bw_hz):
    """(acpr_lower_db, acpr_upper_db, main_band_power): adjacent-channel power ratio
    in the channel immediately below/above the occupied band, each the same width as
    the occupied band -- the standard ACPR definition."""
    half = occupied_bw_hz / 2.0
    p_main = _band_power(faxis, psd, -half, half)
    p_lo = _band_power(faxis, psd, -3 * half, -half)
    p_hi = _band_power(faxis, psd, half, 3 * half)
    eps = 1e-30
    return (10 * np.log10((p_lo + eps) / (p_main + eps)),
            10 * np.log10((p_hi + eps) / (p_main + eps)))


# --------------------------------------------------------------------------------
# Radar-side cost: peak-to-sidelobe ratio of a synthetic point-target range profile
# built from the effective TX spectrum (pre-channel), ideal vs non-ideal.
# --------------------------------------------------------------------------------
def _peak_sidelobe_db(power, guard_bins=3):
    power = np.asarray(power)
    peak_idx = int(np.argmax(power))
    lo, hi = max(0, peak_idx - guard_bins), min(len(power), peak_idx + guard_bins + 1)
    mask = np.ones_like(power, dtype=bool)
    mask[lo:hi] = False
    if not mask.any():
        return float("nan")
    sidelobe = float(power[mask].max())
    return 10 * np.log10(float(power[peak_idx]) / (sidelobe + 1e-30))


# --------------------------------------------------------------------------------
# One backoff point: build the driven waveform, run both arms through the SAME
# channel/noise/estimator/equalizer, return EVM plus the intermediate tensors the
# aggressive-backoff snapshot (constellation/PSD/radar) needs.
# --------------------------------------------------------------------------------
def _run_one_backoff(backoff_db, normalized_up, a_knee, tx_pa, H_ripple, off,
                      fft_size, cp_len, oversample, H_sc, snr_db, modem, n_symbols,
                      tx_data_ref, rng_seed):
    scale = a_knee * 10 ** (-backoff_db / 20.0)
    driven_up = normalized_up * scale

    ideal_eff = _undersample_freq(driven_up, fft_size, cp_len, oversample, off)

    distorted_up = tx_pa.apply(driven_up)
    nonideal_eff = _undersample_freq(distorted_up, fft_size, cp_len, oversample, off)
    nonideal_eff = nonideal_eff * H_ripple[None, :]

    rx_ideal, _ = ch.apply_channel(ideal_eff, H_sc, snr_db, rng_seed=rng_seed)
    rx_nonideal, _ = ch.apply_channel(nonideal_eff, H_sc, snr_db, rng_seed=rng_seed)

    tx_pilots = modem.pilot_grid(n_symbols)

    def _equalize(rx_freq):
        rx_pilots = modem.extract_pilots(rx_freq)
        H_est = ch.ls_estimate(rx_pilots, tx_pilots, modem.pilot_idx, modem.fft_size)
        return modem.extract_data(ch.zf_equalize(rx_freq, H_est))

    eq_ideal = _equalize(rx_ideal)
    eq_nonideal = _equalize(rx_nonideal)

    return {
        "evm_ideal": ch.evm(eq_ideal, tx_data_ref),
        "evm_nonideal": ch.evm(eq_nonideal, tx_data_ref),
        "eq_ideal": eq_ideal,
        "eq_nonideal": eq_nonideal,
        "ideal_eff": ideal_eff,
        "nonideal_eff": nonideal_eff,
        "driven_up": driven_up,
        "distorted_up": distorted_up,
    }


def main(backoff_db_list=None, aggressive_backoff_db=None, fft_size=64, cp_len=16,
         n_active=52, pilot_spacing=8, bits_per_symbol=6, n_symbols=48, oversample=8,
         snr_db=40.0, target_range_m=15.0, tx_pa_config=None, seed=0, show=False):
    """Run the ideal-vs-non-ideal TX A/B and return a results dict (numbers only, no
    matplotlib) so tests can assert on the physics without touching figures.

    `backoff_db_list` defaults to 0..12 dB in 2 dB steps (the range the task brief
    asks for); `aggressive_backoff_db` defaults to the SMALLEST (most compressed)
    backoff in that list.
    """
    if backoff_db_list is None:
        backoff_db_list = list(range(0, 13, 2))
    if aggressive_backoff_db is None:
        aggressive_backoff_db = min(backoff_db_list)

    scenario = munich_radar_scenario()
    freqs = scenario.frequency.linspace()
    carrier = scenario.frequency.carrier_hz
    subcarrier_spacing = 240e3   # narrow comm band, see main_comms_link's note

    modem = OFDMModem(fft_size=fft_size, cp_len=cp_len, n_active=n_active,
                       pilot_spacing=pilot_spacing, bits_per_symbol=bits_per_symbol)

    k = np.arange(fft_size)
    sc_freqs = carrier + (k - fft_size / 2.0) * subcarrier_spacing   # physical grid

    cfr_dense, source = ch.load_or_synthesize_cfr("munich", freqs, rng=np.random.default_rng(seed))
    print(f"[tx_nonideality] channel source: {source}")
    H_sc = ch.cfr_to_subcarriers(cfr_dense, freqs, modem.fft_size, carrier, subcarrier_spacing)

    tx_pa = TxPA(tx_pa_config if tx_pa_config is not None else TxPAConfig())
    cfg = tx_pa.config
    a_knee = cfg.a_sat / (10 ** (cfg.small_signal_gain_db / 20.0))
    H_ripple = tx_pa.frequency_response(sc_freqs).to(H_sc.device)

    tx_bits = random_bits(n_symbols * modem.data_bits_per_symbol_block, seed=seed + 1)
    _, tx_freq = modem.modulate(tx_bits, n_symbols)
    tx_data_ref = modem.extract_data(tx_freq)

    time_up, off = _oversample_time(tx_freq, fft_size, cp_len, oversample)
    cp_up = cp_len * oversample
    data_up = time_up[:, cp_up:]                       # data-only portion, CP stripped
    rms0 = torch.sqrt(torch.mean(torch.abs(data_up) ** 2)).item()
    normalized_up = time_up / rms0                       # unit average power

    # Measured PAPR of the OFDM waveform itself (independent of the PA/backoff),
    # from the OVERSAMPLED data-only samples -- a critically-sampled (fft_size-only)
    # peak estimate is known to understate the true continuous-time peak.
    papr_lin = torch.max(torch.abs(data_up) ** 2) / torch.mean(torch.abs(data_up) ** 2)
    papr_db = 10 * torch.log10(papr_lin).item()

    # ATTRIBUTION (a reviewer asked whether the EVM rise is lower gain -> worse
    # SNR, or nonlinearity): apply_channel references noise power to the RECEIVED
    # signal power (e2e/comms/channel.py), so PA gain compression cannot change
    # SNR. The noise+estimation floor is constant by construction; the entire EVM
    # rise with drive is nonlinear distortion, sqrt(EVM^2 - floor^2).
    # Fixed noise seed across EVERY backoff and BOTH arms: the only thing that
    # changes between arms/backoffs is the (deterministic) PA drive level, so any
    # EVM difference is attributable to the PA, not to a different noise draw.
    rng_seed = 4242

    sweep = {}
    for b in backoff_db_list:
        sweep[b] = _run_one_backoff(b, normalized_up, a_knee, tx_pa, H_ripple, off,
                                     fft_size, cp_len, oversample, H_sc, snr_db, modem,
                                     n_symbols, tx_data_ref, rng_seed)
    if aggressive_backoff_db in sweep:
        agg = sweep[aggressive_backoff_db]
    else:
        agg = _run_one_backoff(aggressive_backoff_db, normalized_up, a_knee, tx_pa,
                                H_ripple, off, fft_size, cp_len, oversample, H_sc,
                                snr_db, modem, n_symbols, tx_data_ref, rng_seed)

    evm_ideal = [sweep[b]["evm_ideal"] for b in backoff_db_list]
    evm_nonideal = [sweep[b]["evm_nonideal"] for b in backoff_db_list]

    # ---- PSD / ACPR at the aggressive backoff ----------------------------------
    fs_up = subcarrier_spacing * fft_size * oversample
    occupied_bw_hz = n_active * subcarrier_spacing
    ideal_data_up = agg["driven_up"][:, cp_up:]
    nonideal_data_up = agg["distorted_up"][:, cp_up:]
    faxis_i, psd_i = _periodogram(ideal_data_up, fs_up)
    faxis_n, psd_n = _periodogram(nonideal_data_up, fs_up)
    acpr_lo_ideal, acpr_hi_ideal = _acpr(faxis_i, psd_i, occupied_bw_hz)
    acpr_lo_nonideal, acpr_hi_nonideal = _acpr(faxis_n, psd_n, occupied_bw_hz)
    acpr_asymmetry_db = abs(acpr_hi_nonideal - acpr_lo_nonideal)

    # ---- ISAC / radar-side cost at the same operating point ---------------------
    # `ideal_eff`/`nonideal_eff` carry the TRANSMITTED DATA's own spectral shape, not
    # a bare channel -- so simulating "reflected off a target at tau0" by multiplying
    # by a pure delay phase gives received(f) = X(f) * H_target(f), not H_target(f)
    # alone. Range profiling that directly would IFFT the (data-dependent, non-flat)
    # X(f) convolved with the target delta, NOT a clean impulse at the target range.
    # Real (and ISAC-literature) sensing off a known TX waveform matched-filters
    # against the known transmitted reference first -- divide out `tx_freq` on the
    # ACTIVE subcarriers (the only ones populated; the rest are exact zeros and
    # would divide by zero) to recover the target/PA-residual response alone.
    tau0 = 2.0 * target_range_m / _C_LIGHT
    active = modem.active_idx.detach().cpu().numpy()
    sc_active = sc_freqs[active]
    steer = np.exp(-2j * np.pi * sc_active * tau0).astype(np.complex64)
    ref_active = tx_freq[0][active].detach().cpu().numpy()
    recv_ideal = agg["ideal_eff"][0][active].detach().cpu().numpy() * steer
    recv_nonideal = agg["nonideal_eff"][0][active].detach().cpu().numpy() * steer
    cfr_ideal = recv_ideal / ref_active
    cfr_nonideal = recv_nonideal / ref_active
    ranges_i, power_i = isac.range_profile(cfr_ideal, sc_active)
    ranges_n, power_n = isac.range_profile(cfr_nonideal, sc_active)
    peak_range_ideal = isac.peak_range(ranges_i, power_i)
    peak_range_nonideal = isac.peak_range(ranges_n, power_n)
    psl_ideal_db = _peak_sidelobe_db(power_i)
    psl_nonideal_db = _peak_sidelobe_db(power_n)

    # ---- printed summary ---------------------------------------------------------
    print(f"[tx_nonideality] OFDM {2 ** bits_per_symbol}-QAM, {n_active}/{fft_size} "
          f"active subcarriers -- measured PAPR = {papr_db:.2f} dB (oversample x{oversample})")
    print(f"[tx_nonideality] PA: gain={cfg.small_signal_gain_db:.1f} dB, "
          f"a_sat={cfg.a_sat:.2f}, a_knee(0 dB IBO)={a_knee:.4f}, "
          f"AM/PM@sat={cfg.am_pm_deg_at_sat:.1f} deg, Rapp p={cfg.rapp_p:.1f}")
    print("[tx_nonideality] backoff(dB)  EVM(ideal)   EVM(non-ideal)")
    for b, ei, en in zip(backoff_db_list, evm_ideal, evm_nonideal):
        print(f"                {b:5d}      {ei:.4f}       {en:.4f}")
    print(f"[tx_nonideality] ACPR @ {aggressive_backoff_db:.0f} dB IBO -- ideal: "
          f"lower={acpr_lo_ideal:.1f} dB upper={acpr_hi_ideal:.1f} dB | "
          f"non-ideal: lower={acpr_lo_nonideal:.1f} dB upper={acpr_hi_nonideal:.1f} dB")
    print(f"[tx_nonideality] non-ideal upper/lower ACPR asymmetry = "
          f"{acpr_asymmetry_db:.3f} dB -- near-zero BY CONSTRUCTION (TxPA is memoryless,"
          " see module docstring); a real PA's regrowth is generally asymmetric, so do "
          "NOT read this PSD as an ACPR/spectral-mask hardware prediction.")
    # Report the range RESOLUTION alongside the peak. Without it the line reads like an
    # error -- a 15 m target reported at 12 m -- when it is simply the nearest bin of a
    # coarse grid: this is a narrow COMMS band (52 x 240 kHz = 12.5 MHz), so
    # c/(2*B) is ~12 m, not the sub-metre resolution a wideband radar preset gives. The
    # sensing point of this plot is the SIDELOBE degradation, not range accuracy.
    range_res_m = _C_LIGHT / (2.0 * occupied_bw_hz)
    print(f"[tx_nonideality] sensing @ {aggressive_backoff_db:.0f} dB IBO, target="
          f"{target_range_m:.1f} m -- peak range: ideal={peak_range_ideal:.2f} m "
          f"(PSL={psl_ideal_db:.1f} dB), non-ideal={peak_range_nonideal:.2f} m "
          f"(PSL={psl_nonideal_db:.1f} dB)")
    print(f"[tx_nonideality]   range resolution = {range_res_m:.2f} m over "
          f"{occupied_bw_hz/1e6:.2f} MHz occupied -- the {target_range_m:.1f} m target's "
          f"NEAREST BIN is {round(target_range_m / range_res_m) * range_res_m:.2f} m, so "
          f"the peak above is correct, not a range error. Sidelobe level (PSL) is the "
          f"quantity to read here: the PA costs "
          f"{psl_ideal_db - psl_nonideal_db:.1f} dB of it.")

    results = {
        "source": source,
        "papr_db": papr_db,
        "backoff_db_list": list(backoff_db_list),
        "aggressive_backoff_db": aggressive_backoff_db,
        "evm_ideal": evm_ideal,
        "evm_nonideal": evm_nonideal,
        "acpr_ideal_db": (acpr_lo_ideal, acpr_hi_ideal),
        "acpr_nonideal_db": (acpr_lo_nonideal, acpr_hi_nonideal),
        "acpr_asymmetry_db": acpr_asymmetry_db,
        "peak_range_ideal_m": peak_range_ideal,
        "peak_range_nonideal_m": peak_range_nonideal,
        "psl_ideal_db": psl_ideal_db,
        "psl_nonideal_db": psl_nonideal_db,
    }

    if show:
        _make_figures(results, agg, backoff_db_list, faxis_i, psd_i, faxis_n, psd_n,
                      occupied_bw_hz, ranges_i, power_i, ranges_n, power_n,
                      target_range_m, bits_per_symbol)

    return results


def _make_figures(results, agg, backoff_db_list, faxis_i, psd_i, faxis_n, psd_n,
                   occupied_bw_hz, ranges_i, power_i, ranges_n, power_n,
                   target_range_m, bits_per_symbol):
    agg_b = results["aggressive_backoff_db"]

    # (1) headline: EVM vs backoff
    plt.figure()
    plt.plot(backoff_db_list, np.array(results["evm_ideal"]) * 100.0, "o-", label="ideal TX")
    plt.plot(backoff_db_list, np.array(results["evm_nonideal"]) * 100.0, "s-", label="non-ideal TX (PA)")
    plt.xlabel("input backoff (dB)")
    plt.ylabel("EVM (% RMS)")
    plt.title(f"EVM vs backoff -- {2 ** bits_per_symbol}-QAM OFDM, "
              f"measured PAPR = {results['papr_db']:.1f} dB")
    plt.grid(True)
    plt.legend()
    evm_path = os.path.join(FIG_DIR, "tx_nonideality_evm_vs_backoff.png")
    plt.savefig(evm_path, dpi=120, bbox_inches="tight")
    plt.close()

    # (2) constellation at the aggressive backoff
    eq_ideal = agg["eq_ideal"].reshape(-1).cpu().numpy()
    eq_nonideal = agg["eq_nonideal"].reshape(-1).cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].scatter(eq_ideal.real, eq_ideal.imag, s=5, alpha=0.4)
    axes[0].set_title(f"ideal TX ({agg_b:.0f} dB IBO)")
    axes[0].axis("equal"); axes[0].grid(True)
    axes[1].scatter(eq_nonideal.real, eq_nonideal.imag, s=5, alpha=0.4, color="tab:orange")
    axes[1].set_title(f"non-ideal TX ({agg_b:.0f} dB IBO)")
    axes[1].axis("equal"); axes[1].grid(True)
    fig.suptitle("RX constellation post-EQ: AM/AM compression + AM/PM rotation")
    const_path = os.path.join(FIG_DIR, "tx_nonideality_constellation.png")
    fig.savefig(const_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # (3) PSD / spectral regrowth
    plt.figure()
    f_i = faxis_i.cpu().numpy() / 1e6
    f_n = faxis_n.cpu().numpy() / 1e6
    # Cross-normalized (both arms relative to the SAME non-ideal-arm peak, not each
    # their own) -- e2e.viz.to_db always normalizes to the input's OWN peak, so this
    # deliberately different convention is kept inline; eps matches to_db's chosen
    # 1e-12 (see its docstring) for consistency, not a re-derivation of that constant.
    _eps = 1e-12
    p_i = 10 * np.log10(psd_i.cpu().numpy() / (psd_n.max().item() + _eps) + _eps)
    p_n = 10 * np.log10(psd_n.cpu().numpy() / (psd_n.max().item() + _eps) + _eps)
    plt.plot(f_i, p_i, label="ideal TX", alpha=0.8)
    plt.plot(f_n, p_n, label="non-ideal TX (PA)", alpha=0.8)
    half_mhz = occupied_bw_hz / 2e6
    plt.axvspan(-half_mhz, half_mhz, color="gray", alpha=0.1, label="occupied band")
    lo, hi = results["acpr_nonideal_db"]
    # Caveat text is WRAPPED and placed inside the axes: an un-wrapped single line ran
    # past the axes and stretched the saved canvas to twice the plot's width under
    # bbox_inches="tight", which is unusable on a slide.
    caveat = (f"non-ideal ACPR: lower={lo:.1f} dB, upper={hi:.1f} dB\n"
              f"|asymmetry| = {results['acpr_asymmetry_db']:.2f} dB -- structurally ~0.\n"
              "TxPA is MEMORYLESS, so this regrowth is symmetric BY\n"
              "CONSTRUCTION. A real PA's is generally asymmetric --\n"
              "do not read this as a hardware ACPR prediction.")
    plt.text(0.015, 0.03, caveat, transform=plt.gca().transAxes, fontsize=6.5,
             va="bottom", ha="left", linespacing=1.35,
             bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    plt.xlabel("frequency offset from carrier (MHz)")
    plt.ylabel("PSD (dB, normalized to non-ideal peak)")
    plt.title(f"Spectral regrowth @ {agg_b:.0f} dB IBO")
    plt.grid(True, alpha=0.4)
    plt.legend(loc="upper right")
    # Floor the y-axis at the noise rather than the 1e-30 epsilon, so the interesting
    # 80 dB of regrowth fills the panel instead of a decade of numerical underflow.
    plt.ylim(max(-100.0, float(np.min(p_n)) - 5.0), 5.0)
    plt.tight_layout()
    psd_path = os.path.join(FIG_DIR, "tx_nonideality_psd.png")
    plt.savefig(psd_path, dpi=120)
    plt.close()

    # (4) ISAC sensing-side cost
    plt.figure()
    pi_db = to_db(power_i, floor_db=None)
    pn_db = to_db(power_n, floor_db=None)
    plt.plot(ranges_i, pi_db, label=f"ideal TX (PSL={results['psl_ideal_db']:.1f} dB)")
    plt.plot(ranges_n, pn_db, label=f"non-ideal TX (PSL={results['psl_nonideal_db']:.1f} dB)")
    plt.axvline(target_range_m, color="k", linestyle="--", alpha=0.5, label="true target range")
    plt.xlim(0, target_range_m * 3)
    plt.xlabel("range (m)")
    plt.ylabel("normalized power (dB)")
    plt.title(f"ISAC sensing-side cost @ {agg_b:.0f} dB IBO (shared-waveform range profile)")
    plt.grid(True)
    plt.legend()
    range_path = os.path.join(FIG_DIR, "tx_nonideality_range_profile.png")
    plt.savefig(range_path, dpi=120, bbox_inches="tight")
    plt.close()

    print(f"[tx_nonideality] wrote {evm_path}")
    print(f"[tx_nonideality] wrote {const_path}")
    print(f"[tx_nonideality] wrote {psd_path}")
    print(f"[tx_nonideality] wrote {range_path}")


if __name__ == "__main__":
    main(show=True)
