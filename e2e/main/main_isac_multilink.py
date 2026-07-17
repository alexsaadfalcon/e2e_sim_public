"""
Multi-link ISAC demo: consumes the per-link ``.pkl`` dict exported by
``ScenarioRunner`` for a multi-link scenario (see ``e2e/environment/scenario_runner.py``
module docstring, "Links and output layout").

Where ``main_isac.py`` builds its radar/comm channels ad hoc (a single hardcoded
``munich.pkl`` frame, or a synthetic fallback), this example is the plumbing demo for
the *actual* multi-link export: it generates (or reuses) one ``.pkl`` holding BOTH the
radar link and the comm link of ``munich_isac_scenario()``, discovers the links with
``SionnaIterator.available_links()``, and drives each sub-problem straight off its own
link's frame stack.

Flow
----
1. Build ``munich_isac_scenario()`` and shrink it in-process (``num_frames`` /
   ``frequency.num_freqs``) so the demo runs in seconds -- the reference scenario's
   defaults (100 frames x 5000 freq bins) are sized for real generation, not a quick
   plumbing demo. ``Scenario`` / ``FrequencyPlan`` are plain dataclasses, so this is a
   direct attribute mutation on the returned instance.
2. If no cached multi-link ``.pkl`` exists yet at ``cache_path``, generate one via
   ``ScenarioRunner(..., dry_run=True)`` (no Sionna/GPU needed). If a ``.pkl`` is
   already present -- e.g. a REAL generation dropped at the same path by a machine
   with Sionna RT -- it is used as-is and the demo processes real frames automatically
   (the frequency grid is derived from the actual per-link array shape, not the
   shrunk demo scenario, so this works regardless of which one produced the file).
3. Discover links with ``SionnaIterator.available_links()`` and print them.
4. RADAR leg: pick the link whose name contains "radar", average the S-parameters
   over array elements per frame, and compute a range profile
   (``e2e.comms.isac.range_profile``) -> report the peak range per frame.
5. COMM leg: pick the link whose name contains "comm", take one representative
   frame's CFR, and run the shared OFDM 16-QAM link (``e2e.comms.ofdm`` /
   ``e2e.comms.channel``) at a few SNRs -> report BER/EVM per SNR.
6. Figures (matplotlib Agg, no display) to ``fig_dir``:
   isac_multilink_range_waterfall.png, isac_multilink_comm_ber.png.

IMPORTANT -- dry-run data caveat: the dry-run frames are independent complex Gaussian
noise per element, scaled by 1/distance (see ``ScenarioRunner._mock_frame``); they have
no multipath structure, so the radar leg's "range profile" will NOT show a physical
target and the comm leg's channel is not a realistic propagation channel. That is
expected and fine -- the point of this demo is exercising the multi-link .pkl
PLUMBING (link enumeration, per-link loading, feeding each link into its downstream
processing), not physically meaningful numbers. Dropping a REAL Sionna-generated
multi-link .pkl at the same cache path makes both legs physically meaningful with no
code changes.

Run:
    python -m e2e.main.main_isac_multilink

Outputs (e2e/main/figures/):
    isac_multilink_range_waterfall.png   radar range profile vs frame index
    isac_multilink_comm_ber.png          comm BER vs SNR
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless: write figures to files, no display
import matplotlib.pyplot as plt

from e2e.scenario import munich_isac_scenario
from e2e.environment.scenario_runner import ScenarioRunner, SIONNA_SIMS_DIR
from e2e.environment.sionna_iterator import SionnaIterator
from e2e.comms.ofdm import OFDMModem, qam_demod, random_bits
from e2e.comms import channel as ch
from e2e.comms import isac


FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _default_cache_path(scenario):
    """Cache path fingerprinted by the scenario content.

    The fingerprint makes a scenario change (e.g. munich_isac gaining tx_power_dbm)
    invalidate the cache automatically -- a bare filename check would silently reuse
    frames generated under an older, incompatible convention.
    """
    import hashlib
    fp = hashlib.sha1(scenario.to_json().encode("utf-8")).hexdigest()[:8]
    return os.path.join(SIONNA_SIMS_DIR, f"munich_isac_demo_{fp}.pkl")

# Demo-sized scenario knobs -- see module docstring step 1.
DEMO_NUM_FRAMES = 10
DEMO_NUM_FREQS = 512

SEED = 7
SNR_LIST_DB = [5.0, 15.0, 25.0]
N_OFDM_SYMBOLS = 32


def _demo_scenario(num_frames, num_freqs):
    """``munich_isac_scenario()`` shrunk for a fast demo (see module docstring)."""
    scenario = munich_isac_scenario()
    scenario.num_frames = num_frames
    scenario.frequency.num_freqs = num_freqs
    return scenario


def _ensure_cache(cache_path, num_frames, num_freqs, seed, verbose=True):
    """Generate the multi-link .pkl (dry-run) if it doesn't already exist.

    Returns the FULL-SIZE reference scenario (unshrunk frequency band), used only to
    recover the carrier / start / stop Hz -- the actual per-link frequency *count* is
    read back from the cached array shape, so this works whether the cache was just
    dry-run-generated here or is a real Sionna dump left at the same path.
    """
    if not os.path.isfile(cache_path):
        scenario = _demo_scenario(num_frames, num_freqs)
        runner = ScenarioRunner(scenario, dry_run=True, seed=seed)
        if verbose:
            print(runner.describe())
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        runner.run(out_path=cache_path, verbose=verbose)
    else:
        if verbose:
            print(f"[isac_multilink] reusing cached multi-link pkl: {cache_path}")
    return munich_isac_scenario()


def _pick_link(links, needle):
    for name in links:
        if needle in name.lower():
            return name
    raise ValueError(f"no link name containing {needle!r} among {links}")


def _radar_leg(cache_path, radar_link, freq_band):
    """Per-frame range profile off the radar link -> (ranges, profiles[F,B], peak_ranges[F])."""
    it = SionnaIterator(cache_path, link=radar_link)
    n_frames = len(it)

    profiles = []
    peak_ranges = []
    ranges = None
    freqs = None  # frequency grid is link-invariant; built once from the first frame
    for i in range(n_frames):
        frame = np.asarray(it[i], dtype=np.complex64)     # [n_rx_ant, n_tx_ant, n_time, n_freqs]
        if freqs is None:
            freqs = np.linspace(freq_band[0], freq_band[1], frame.shape[-1])
        # mean over array elements (and the single tx/time index) -> [n_freqs]
        cfr = frame[:, 0, 0, :].mean(axis=0)
        ranges, power = isac.range_profile(cfr, freqs)
        profiles.append(power)
        peak_ranges.append(isac.peak_range(ranges, power))

    return ranges, np.stack(profiles, axis=0), np.asarray(peak_ranges)


def _comm_leg(cache_path, comm_link, freq_band, snr_list, frame_idx=None):
    """Run the OFDM 16-QAM link over one representative comm-link frame at each SNR.

    Returns dict(source=link_name, snr_list=[...], ber=[...], evm_pct=[...], eq_snapshot).
    """
    it = SionnaIterator(cache_path, link=comm_link)
    n_frames = len(it)
    if frame_idx is None:
        frame_idx = n_frames // 2

    frame = np.asarray(it[frame_idx], dtype=np.complex64)  # [n_rx_ant, n_tx_ant, n_time, n_freqs]
    n_f = frame.shape[-1]
    freqs = np.linspace(freq_band[0], freq_band[1], n_f)
    # single spatial channel (element 0), matching channel.load_or_synthesize_cfr's
    # convention -- the OFDM link here is SISO, not spatially combined.
    cfr_dense = frame[0, 0, 0, :]
    carrier = float(freq_band[0] + freq_band[1]) / 2.0

    bits_per_symbol = 4  # 16-QAM
    modem = OFDMModem(fft_size=64, cp_len=16, n_active=52, pilot_spacing=8,
                      bits_per_symbol=bits_per_symbol)
    subcarrier_spacing = 240e3   # narrow comm band, see main_comms_link.py note

    H_sc = ch.cfr_to_subcarriers(cfr_dense, freqs, modem.fft_size, carrier, subcarrier_spacing)

    tx_bits = random_bits(N_OFDM_SYMBOLS * modem.data_bits_per_symbol_block, seed=11)
    _, tx_freq = modem.modulate(tx_bits, N_OFDM_SYMBOLS)

    ber_list, evm_list = [], []
    eq_snapshot = None
    for snr_db in snr_list:
        rx_freq, _ = ch.apply_channel(tx_freq, H_sc, snr_db, rng_seed=int(1000 + snr_db))

        rx_pilots = modem.extract_pilots(rx_freq)
        tx_pilots = modem.pilot_grid(N_OFDM_SYMBOLS)
        H_est = ch.mmse_estimate(rx_pilots, tx_pilots, modem.pilot_idx, modem.fft_size, snr_db)
        eq = modem.extract_data(ch.mmse_equalize(rx_freq, H_est, snr_db))
        rx_bits = qam_demod(eq.reshape(-1), bits_per_symbol, modem.const)

        ber_list.append(ch.ber(tx_bits, rx_bits))
        # EVM against the TRUE transmitted data symbols (not decision-directed --
        # nearest-point references bias EVM low once symbols cross decision boundaries).
        ref = modem.extract_data(tx_freq).reshape(-1)
        evm_list.append(ch.evm(eq.reshape(-1), ref) * 100.0)

        if snr_db == max(snr_list):
            eq_snapshot = eq.reshape(-1).cpu().numpy()

    return {
        "frame_idx": frame_idx,
        "snr_list": list(snr_list),
        "ber": ber_list,
        "evm_pct": evm_list,
        "eq_snapshot": eq_snapshot,
    }


def main(cache_path=None, fig_dir=None, seed=SEED,
        num_frames=DEMO_NUM_FRAMES, num_freqs=DEMO_NUM_FREQS, verbose=True):
    """Run the multi-link ISAC demo.

    Parameters (all optional, default to the repo's shared cache/figures locations)
    let tests redirect I/O into a tmp dir without touching the repo tree.
    """
    if cache_path is None:
        # Fingerprint the default cache by the DEMO scenario's content so a scenario
        # change regenerates instead of silently reusing incompatible frames.
        cache_path = _default_cache_path(_demo_scenario(num_frames, num_freqs))
    fig_dir = fig_dir or FIG_DIR
    os.makedirs(fig_dir, exist_ok=True)

    scenario_ref = _ensure_cache(cache_path, num_frames, num_freqs, seed, verbose=verbose)
    freq_band = (scenario_ref.frequency.start_hz, scenario_ref.frequency.stop_hz)

    links = SionnaIterator.available_links(cache_path)
    if verbose:
        print(f"[isac_multilink] links in {cache_path}: {links}")
    if links is None or len(links) < 2:
        raise RuntimeError(f"expected a multi-link pkl (>=2 links) at {cache_path}, got {links}")

    radar_link = _pick_link(links, "radar")
    comm_link = _pick_link(links, "comm")
    # Surface (don't hide) a size mismatch between an explicitly supplied cache and
    # the requested demo knobs: reuse is intentional (that's how a real Sionna dump
    # is consumed), but it should never be silent.
    n_cached = len(SionnaIterator(cache_path, link=radar_link))
    if verbose and n_cached != num_frames:
        print(f"[isac_multilink] note: cache holds {n_cached} frames "
              f"(requested {num_frames}); using the cache as-is.")
    if verbose:
        print(f"[isac_multilink] radar link: {radar_link!r}   comm link: {comm_link!r}")
        print("[isac_multilink] NOTE: dry-run frames are unstructured noise (no multipath) -- "
              "range/BER numbers below are plumbing checks, not physical results "
              "(see module docstring).")

    # ===== RADAR leg =====
    ranges, profiles, peak_ranges = _radar_leg(cache_path, radar_link, freq_band)
    if verbose:
        print(f"[isac_multilink] radar: {profiles.shape[0]} frames, "
              f"peak range per frame (m): {np.round(peak_ranges, 2)}")

    # ===== COMM leg =====
    comm_result = _comm_leg(cache_path, comm_link, freq_band, SNR_LIST_DB)
    if verbose:
        print(f"[isac_multilink] comm: link={comm_link}, frame={comm_result['frame_idx']}")
        print("[isac_multilink] comm SNR(dB)   BER         EVM(%)")
        for s, b, e in zip(comm_result["snr_list"], comm_result["ber"], comm_result["evm_pct"]):
            print(f"                  {s:6.1f}   {b:.3e}   {e:.2f}")

    # ===== figures =====
    plt.figure()
    db = 10 * np.log10(profiles / (profiles.max() + 1e-30) + 1e-12)
    plt.imshow(db, aspect="auto", origin="lower",
              extent=[ranges[0], ranges[-1], 0, profiles.shape[0]],
              cmap="viridis", vmin=-30, vmax=0)
    plt.colorbar(label="normalized power (dB)")
    plt.xlabel("range (m)")
    plt.ylabel("frame index")
    plt.title(f"ISAC multi-link radar range profile ({radar_link})")
    waterfall_path = os.path.join(fig_dir, "isac_multilink_range_waterfall.png")
    plt.savefig(waterfall_path, dpi=120, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.semilogy(comm_result["snr_list"], np.clip(comm_result["ber"], 1e-6, 1), "o-")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title(f"ISAC multi-link comm BER vs SNR ({comm_link})")
    plt.grid(True, which="both")
    ber_path = os.path.join(fig_dir, "isac_multilink_comm_ber.png")
    plt.savefig(ber_path, dpi=120, bbox_inches="tight")
    plt.close()

    if verbose:
        print(f"[isac_multilink] wrote {waterfall_path}")
        print(f"[isac_multilink] wrote {ber_path}")
        print("[isac_multilink] summary: "
              f"links={links}  radar_peak_range_mean={float(np.mean(peak_ranges)):.2f}m  "
              f"comm_ber@{comm_result['snr_list'][-1]:.0f}dB={comm_result['ber'][-1]:.3e}")

    return {
        "cache_path": cache_path,
        "links": links,
        "radar_link": radar_link,
        "comm_link": comm_link,
        "ranges": ranges,
        "profiles": profiles,
        "peak_ranges": peak_ranges,
        "comm": comm_result,
        "figures": [waterfall_path, ber_path],
    }


if __name__ == "__main__":
    main()
