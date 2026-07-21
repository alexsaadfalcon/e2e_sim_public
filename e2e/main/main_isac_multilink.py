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
3. Discover links with ``SionnaIterator.available_links()`` and print them. Links are
   selected by the v2 payload's ``meta["links"][name]["kind"]`` ("radar"/"comm") when
   present, falling back to a name-substring match ("radar"/"comm" in the link name)
   for legacy pkls with no meta. The frequency band likewise comes from
   ``SionnaIterator.freq_plan`` when present, falling back to the reference scenario's
   ``FrequencyPlan`` only for legacy caches. The demo prints which convention was used
   for each (see step 3's "kind detected via" / "frequency band source" lines).
4. RADAR leg: average the S-parameters over array elements per frame, and compute a
   range profile (``e2e.comms.isac.range_profile``) -> report the peak range per frame.
5. COMM leg: run the shared OFDM 16-QAM link (``e2e.comms.ofdm`` / ``e2e.comms.channel``)
   two ways -- (a) one representative frame's CFR at a few SNRs -> BER/EVM per SNR, and
   (b) EVERY frame's CFR at one fixed SNR -> BER vs frame index, which is the point of
   a *multi-link* demo (the comm channel evolves frame-to-frame just like the radar leg).
6. Figures (matplotlib Agg, no display) to ``fig_dir``:
   isac_multilink_range_waterfall.png, isac_multilink_comm_ber.png,
   isac_multilink_comm_ber_per_frame.png.

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
    isac_multilink_comm_ber.png          comm BER vs SNR (one representative frame)
    isac_multilink_comm_ber_per_frame.png  comm BER vs frame index (fixed SNR)
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
PER_FRAME_SNR_DB = 15.0  # fixed SNR for the per-frame BER-vs-frame-index sweep


def _demo_scenario(num_frames, num_freqs):
    """``munich_isac_scenario()`` shrunk for a fast demo (see module docstring)."""
    scenario = munich_isac_scenario()
    scenario.num_frames = num_frames
    scenario.frequency.num_freqs = num_freqs
    return scenario


def _ensure_cache(cache_path, num_frames, num_freqs, seed, verbose=True):
    """Generate the multi-link .pkl (dry-run) if it doesn't already exist."""
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


def _pick_link(links, needle):
    for name in links:
        if needle in name.lower():
            return name
    raise ValueError(f"no link name containing {needle!r} among {links}")


def _pick_link_by_kind(cache_path, links, kind, needle):
    """Select a link by its v2 ``meta["links"][name]["kind"]`` ("radar"/"comm") when
    available, falling back to the legacy name-substring match for pkls with no meta
    (or where meta doesn't cover every link).

    Returns (link_name, kind_source) where kind_source is "meta" or "name" -- surfaced
    so the demo can print which convention was actually used (see module docstring).
    """
    it = SionnaIterator(cache_path, link=links[0])
    if it.meta is not None:
        links_meta = it.meta.get("links", {})
        matches = [name for name in links if links_meta.get(name, {}).get("kind") == kind]
        if matches:
            return matches[0], "meta"
    return _pick_link(links, needle), "name"


def _freq_band(cache_path, link):
    """Frequency band (start_hz, stop_hz) from the iterator's ``freq_plan`` (v2 meta)
    when present, else None -- caller falls back to the reference scenario for legacy
    caches with no meta."""
    it = SionnaIterator(cache_path, link=link)
    plan = it.freq_plan
    if plan is None:
        return None
    return (plan["start_hz"], plan["stop_hz"])


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


_BITS_PER_SYMBOL = 4  # 16-QAM
_SUBCARRIER_SPACING = 240e3   # narrow comm band, see main_comms_link.py note


def _make_ofdm_tx():
    """Shared OFDM modem + one fixed tx bit/symbol stream, reused by both comm legs
    so the SNR sweep and the per-frame sweep transmit the identical payload."""
    modem = OFDMModem(fft_size=64, cp_len=16, n_active=52, pilot_spacing=8,
                      bits_per_symbol=_BITS_PER_SYMBOL)
    tx_bits = random_bits(N_OFDM_SYMBOLS * modem.data_bits_per_symbol_block, seed=11)
    _, tx_freq = modem.modulate(tx_bits, N_OFDM_SYMBOLS)
    return modem, tx_bits, tx_freq


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

    modem, tx_bits, tx_freq = _make_ofdm_tx()
    H_sc = ch.cfr_to_subcarriers(cfr_dense, freqs, modem.fft_size, carrier, _SUBCARRIER_SPACING)

    ber_list, evm_list = [], []
    eq_snapshot = None
    for snr_db in snr_list:
        rx_freq, _ = ch.apply_channel(tx_freq, H_sc, snr_db, rng_seed=int(1000 + snr_db))

        rx_pilots = modem.extract_pilots(rx_freq)
        tx_pilots = modem.pilot_grid(N_OFDM_SYMBOLS)
        H_est = ch.mmse_estimate(rx_pilots, tx_pilots, modem.pilot_idx, modem.fft_size, snr_db)
        eq = modem.extract_data(ch.mmse_equalize(rx_freq, H_est, snr_db))
        rx_bits = qam_demod(eq.reshape(-1), _BITS_PER_SYMBOL, modem.const)

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


def _comm_leg_per_frame(cache_path, comm_link, freq_band, snr_db=PER_FRAME_SNR_DB):
    """Run the OFDM 16-QAM link over EVERY comm-link frame at one fixed SNR.

    This is the actual multi-link-demo point: the per-frame channel evolution, not a
    single-frame SNR sweep (that stays in ``_comm_leg`` to keep runtime sane). Returns
    a ``ber_per_frame`` ndarray, one entry per frame in the comm link.
    """
    it = SionnaIterator(cache_path, link=comm_link)
    n_frames = len(it)
    carrier = float(freq_band[0] + freq_band[1]) / 2.0

    modem, tx_bits, tx_freq = _make_ofdm_tx()
    tx_pilots = modem.pilot_grid(N_OFDM_SYMBOLS)

    ber_per_frame = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        frame = np.asarray(it[i], dtype=np.complex64)
        freqs = np.linspace(freq_band[0], freq_band[1], frame.shape[-1])
        cfr_dense = frame[0, 0, 0, :]
        H_sc = ch.cfr_to_subcarriers(cfr_dense, freqs, modem.fft_size, carrier, _SUBCARRIER_SPACING)

        rx_freq, _ = ch.apply_channel(tx_freq, H_sc, snr_db, rng_seed=int(2000 + i))
        rx_pilots = modem.extract_pilots(rx_freq)
        H_est = ch.mmse_estimate(rx_pilots, tx_pilots, modem.pilot_idx, modem.fft_size, snr_db)
        eq = modem.extract_data(ch.mmse_equalize(rx_freq, H_est, snr_db))
        rx_bits = qam_demod(eq.reshape(-1), _BITS_PER_SYMBOL, modem.const)
        ber_per_frame[i] = ch.ber(tx_bits, rx_bits)

    return ber_per_frame


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

    _ensure_cache(cache_path, num_frames, num_freqs, seed, verbose=verbose)

    links = SionnaIterator.available_links(cache_path)
    if verbose:
        print(f"[isac_multilink] links in {cache_path}: {links}")
    if links is None or len(links) < 2:
        raise RuntimeError(f"expected a multi-link pkl (>=2 links) at {cache_path}, got {links}")

    radar_link, radar_link_src = _pick_link_by_kind(cache_path, links, "radar", "radar")
    comm_link, comm_link_src = _pick_link_by_kind(cache_path, links, "comm", "comm")

    # Frequency band: meta.freq_plan (v2 payload) when present, else the reference
    # scenario's band (legacy cache with no meta -- munich_isac_scenario's frequency
    # plan is the only place left to recover start/stop Hz for those).
    freq_band = _freq_band(cache_path, radar_link)
    if freq_band is not None:
        band_source = "meta.freq_plan"
    else:
        band_source = "reference scenario (legacy cache, no meta)"
        scenario_ref = munich_isac_scenario()
        freq_band = (scenario_ref.frequency.start_hz, scenario_ref.frequency.stop_hz)

    # Surface (don't hide) a size mismatch between an explicitly supplied cache and
    # the requested demo knobs: reuse is intentional (that's how a real Sionna dump
    # is consumed), but it should never be silent.
    n_cached = len(SionnaIterator(cache_path, link=radar_link))
    if verbose and n_cached != num_frames:
        print(f"[isac_multilink] note: cache holds {n_cached} frames "
              f"(requested {num_frames}); using the cache as-is.")
    if verbose:
        radar_meta = SionnaIterator(cache_path, link=radar_link)
        comm_meta = SionnaIterator(cache_path, link=comm_link)
        print(f"[isac_multilink] radar link: {radar_link!r} (kind detected via {radar_link_src}, "
              f"physical_scale={radar_meta.physical_scale})")
        print(f"[isac_multilink] comm link: {comm_link!r} (kind detected via {comm_link_src}, "
              f"physical_scale={comm_meta.physical_scale})")
        print(f"[isac_multilink] frequency band source: {band_source}  band={freq_band}")
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

    # Per-frame BER at a single fixed SNR -- the actual multi-link-demo point: the
    # comm channel evolves frame-to-frame just like the radar leg above (see task
    # docstring / module docstring), not just an SNR sweep on one snapshot.
    ber_per_frame = _comm_leg_per_frame(cache_path, comm_link, freq_band, snr_db=PER_FRAME_SNR_DB)
    if verbose:
        print(f"[isac_multilink] comm BER vs frame index @ {PER_FRAME_SNR_DB:.0f}dB: "
              f"{np.round(ber_per_frame, 4)}")

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

    plt.figure()
    plt.plot(np.arange(len(ber_per_frame)), np.clip(ber_per_frame, 1e-6, 1), "o-")
    plt.yscale("log")
    plt.xlabel("frame index")
    plt.ylabel("BER")
    plt.title(f"ISAC multi-link comm BER vs frame index ({comm_link}, {PER_FRAME_SNR_DB:.0f}dB)")
    plt.grid(True, which="both")
    ber_per_frame_path = os.path.join(fig_dir, "isac_multilink_comm_ber_per_frame.png")
    plt.savefig(ber_per_frame_path, dpi=120, bbox_inches="tight")
    plt.close()

    if verbose:
        print(f"[isac_multilink] wrote {waterfall_path}")
        print(f"[isac_multilink] wrote {ber_path}")
        print(f"[isac_multilink] wrote {ber_per_frame_path}")
        print("[isac_multilink] summary: "
              f"links={links}  radar_peak_range_mean={float(np.mean(peak_ranges)):.2f}m  "
              f"comm_ber@{comm_result['snr_list'][-1]:.0f}dB={comm_result['ber'][-1]:.3e}")

    return {
        "cache_path": cache_path,
        "links": links,
        "radar_link": radar_link,
        "comm_link": comm_link,
        "band_source": band_source,
        "freq_band": freq_band,
        "ranges": ranges,
        "profiles": profiles,
        "peak_ranges": peak_ranges,
        "comm": comm_result,
        "ber_per_frame": ber_per_frame,
        "figures": [waterfall_path, ber_path, ber_per_frame_path],
    }


if __name__ == "__main__":
    main()
