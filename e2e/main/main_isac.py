"""
Joint radar + communications (ISAC) example.

RE-POINTED (see notes on the change below) onto `e2e.comms.ofdm_isac.waveform_chain_spec`
("jsac"): ONE hybrid OFDM frame, symbol-divided, that yields BOTH a sensing cube
(`SymbolDivisionBlock` + the chain's own `RangeTransformBlock`, the actual one-chain
range FFT -- not a second, hand-rolled transform) AND a BER/EVM (`OFDMReceiveBlock` +
`BERBlock`) from the SAME channel. That is the shared-waveform idea made literal: one
frame, one mixing block, two products.

WHAT THIS REPLACES, and why it is a real change to what the script demonstrates, not
just an implementation swap: the previous version loaded `munich_isac_scenario()` (a
MULTI-NODE scene -- a car-mounted radar node AND a separate building->car comm link)
and ran two INDEPENDENT sub-problems, one per physical link, each through its own
hand-rolled transform in `e2e.comms.isac` (`range_angle_map`, a reimplementation of
`RangeAzBlock`/`RangeTransformBlock`). That demonstrated spatial-division ISAC (two
systems sharing a band). The JSAC waveform demonstrates a DIFFERENT architecture --
waveform-division ISAC, one system, one channel, one frame -- and does not have a
comm-link node to read a second channel from. This version therefore runs the JSAC
frame over the SENSING node's channel only; the scenario's separate comm-link node
(`split["comm_links"]`) is still reported (`isac.describe_split`) but no longer driven.
If the multi-node, two-channel demonstration is still wanted, it needs a separate
script -- forcing it into this one would silently misrepresent one architecture as
the other.

Run:
    python -m e2e.main.main_isac

Outputs (e2e/main/figures/):
    isac_range_angle.png   sensing range/angle map (preamble symbol, the sensing
                            reference under the default sensing_source="preamble")
    isac_constellation.png comm RX constellation (post-EQ, data symbols)
"""

import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from e2e.blocks import RangeAzBlock
from e2e.chain.receive import RangeTransformBlock
from e2e.comms import channel as ch
from e2e.comms import isac
from e2e.comms.blocks import BERBlock
from e2e.comms.constellation_viz import plot_constellation
from e2e.comms.ofdm_isac import waveform_chain_spec
from e2e.scenario import munich_isac_scenario
from e2e.viz import fig_dir, to_db, imshow_ra


FIG_DIR = fig_dir(__file__)

N_RX_X = N_RX_Y = 32


class _RadarCfg:
    """The minimal cfg the JSAC mixing block reads (`SymbolDivisionBlock`/
    `mimo_combine`): single TX, no transmit multiplexing to undo. Mirrors
    `tests/test_waveform_classes.py::_Cfg` and `e2e.simulation.Simulation._SingleTxCfg`."""
    mimo = "single"
    n_tx = 1
    n_chirps = 1


def _radar_s_pars(scenario, freqs, rng, src_band=None):
    """[N_RX, N_FREQS] S-parameters for the radar node, `(s_pars, source_str)`.

    UNCHANGED from the previous version (signature, return shape, and body) --
    `tests/test_comms_isac.py` calls this directly and asserts a 2-D
    `[n_rx, n_freqs]` return, so the reshape to the 4-D `[n_rx, n_tx, chirp,
    n_freqs]` layout `waveform_chain_spec`'s blocks expect happens at the ONE call
    site in `main()` instead of here. `scenario` is accepted but unused (kept for
    the same reason the original did: the caller has it in scope).

    Real munich frame if present, else a synthetic per-element multipath fallback so
    the aperture sees a coherent point target plus clutter.
    """
    n_rx = N_RX_X * N_RX_Y
    arr = None
    try:
        from e2e.environment import sionna_iterator as si
        it = si.SionnaMunichIterator()             # raises if .pkl missing
        arr = it[0]
    except Exception:
        arr = None                                 # .pkl missing -> synthetic below

    if arr is not None:
        arr_np = np.asarray(arr)
        # The aperture geometry (N_RX_X x N_RX_Y) is fixed by the range/angle map;
        # the loaded frame MUST provide exactly that many elements or the spatial
        # FFT would be fed uninitialised/garbage rows. Fail loudly (clear error)
        # rather than silently using zeros or falling back to synthetic.
        if arr_np.shape[0] != n_rx:
            raise ValueError(
                f"munich frame has {arr_np.shape[0]} array elements but the "
                f"range/angle map expects N_RX_X*N_RX_Y = {n_rx}; cannot map a "
                f"frame with a different antenna count onto this aperture "
                f"(set N_RX_X/N_RX_Y to match, or provide a matching frame).")
        out = ch.frame_to_cfr(arr_np, freqs, src_band=src_band, element=None)
        return out, "sionna:munich"

    # synthetic fallback (no .pkl): shared target delay (~30 m) + per-element
    # random taps so the aperture sees a coherent point target plus clutter.
    c = 299_792_458.0
    target_range = 30.0
    tau0 = 2 * target_range / c
    out = np.zeros((n_rx, len(freqs)), dtype=np.complex64)
    # tau0 is a scalar delay -> steering vector is just exp(-j2pi f tau0), [F].
    # (np.outer would yield a spurious [F,1] and break the per-row assignment.)
    steer = np.exp(-2j * np.pi * np.asarray(freqs) * tau0)         # [F]
    for i in range(n_rx):
        clutter = ch.synthetic_multipath_cfr(freqs, n_taps=4, rng=rng).cpu().numpy()
        phase = np.exp(1j * 2 * np.pi * (i % N_RX_X) * 0.01)       # small spatial taper
        out[i] = (0.7 * steer * phase + 0.3 * clutter).astype(np.complex64)
    return torch.from_numpy(out).to(ch.device), "synthetic"


def main():
    rng = np.random.default_rng(7)
    scenario = munich_isac_scenario()
    freqs = scenario.frequency.linspace()

    print(isac.describe_split(scenario))
    split = isac.split_scenario(scenario)
    assert split["is_isac"], "expected an ISAC scenario"
    if split["comm_links"]:
        tx_node, rx_node = split["comm_links"][0]
        print(f"[isac] scenario also names a comm link {tx_node.name} -> {rx_node.name}, "
              f"not driven here: the JSAC waveform is single-channel by design (see the "
              f"module docstring's 'WHAT THIS REPLACES').")

    # ===== ONE joint JSAC frame: sensing cube + BER from the SAME channel =====
    s_pars_2d, sense_src = _radar_s_pars(scenario, freqs, rng)
    print(f"[isac] channel source: {sense_src}")
    n_rx = N_RX_X * N_RX_Y
    h = s_pars_2d.view(n_rx, 1, 1, -1)   # [n_rx, n_freqs] -> [n_rx, n_tx, chirp, n_freqs]

    bits_per_symbol = 4          # 16-QAM, same as the previous comm sub-problem
    n_symbols = 4                # 1 all-pilot preamble (the sensing reference) + 3 data
    pilot_spacing = 8
    freq_plan = {"start_hz": float(freqs[0]), "stop_hz": float(freqs[-1]),
                "num_freqs": len(freqs)}
    spec = waveform_chain_spec("jsac", _RadarCfg(), freq_plan=freq_plan,
                               n_symbols=n_symbols, pilot_spacing=pilot_spacing,
                               bits_per_symbol=bits_per_symbol, combining="mrc")

    state = {"s_pars": h, "freq_plan": freq_plan}
    state.update(spec.channel_block.apply(state))          # H -> Y = H*X (the TX grid)

    # `OFDMReceiveBlock` injects NO noise of its own -- contract section 1.4, and
    # unlike `ModemBlock` it has no `add_noise` knob at all (see its docstring): a
    # comms head is a noiseless TAP by design. The one chain's actual noise source,
    # the front end, sits on the SAMPLED BEAT RECORD after the mixing block -- a
    # point neither this comm head (which taps BEFORE the mixing block) nor the
    # sensing image built from that same block's input can see. So, for the same
    # structural reason `main_comms_head.py` passes `ModemBlock(add_noise=True)`,
    # this example draws its OWN explicit, STATED noise directly on the received
    # grid `Y` -- shared by construction between the two products built from it
    # below (the SAME noisy `Y` drives both the comm head and the sensing mixing
    # block), which is the joint-channel comparison this script exists to show.
    added_snr_db = 20.0
    g = torch.Generator(device="cpu").manual_seed(42)
    sig_pow = float(torch.mean(torch.abs(state["s_pars"]) ** 2))
    noise_pow = sig_pow / (10 ** (added_snr_db / 10.0))
    noise = (torch.randn(state["s_pars"].shape, generator=g)
             + 1j * torch.randn(state["s_pars"].shape, generator=g))
    noise = noise.to(state["s_pars"].device) * float(np.sqrt(noise_pow / 2.0))
    state["s_pars"] = (state["s_pars"] + noise.to(state["s_pars"].dtype))

    # ---- comm product: OFDMReceiveBlock (the comms head) + BERBlock -------
    state.update(spec.receive_block.apply(state))
    comm_out = BERBlock().apply(state)
    bit_err, evm_frac = comm_out["ber"], comm_out["evm"]
    evm_pct = evm_frac * 100.0
    measured_snr_db = state["comm_snr_db"]   # measured from the pilot residual
    gain_db = state.get("comm_array_gain_db", float("nan"))
    print(f"[isac] comm (jsac head, MRC) @ added SNR={added_snr_db:.0f} dB "
          f"(measured post-combining SNR={measured_snr_db:.1f} dB): "
          f"BER={bit_err:.3e}, EVM={evm_pct:.2f}%, array gain={gain_db:.2f} dB")

    # ---- sensing product: the SAME frame's SymbolDivisionBlock + the chain's own
    # RangeTransformBlock + RangeAzBlock (identity-point transform, no window/DC
    # removal, matching Simulation._build_spine's imaging spine) -------------------
    state.update(spec.mixing_block.apply(state))            # Y -> adc (symbol division)
    state.update(RangeTransformBlock(None, window="none", dc_removal=False).apply(state))
    range_az = RangeAzBlock(bins=128, array_shape=(N_RX_X, N_RX_Y)).apply(state)["range_az"]
    ranges_m = state.get("range_axis")
    if ranges_m is None:
        ranges_m = np.arange(range_az.shape[-1])
    else:
        ranges_m = np.asarray(ranges_m.cpu().numpy() if torch.is_tensor(ranges_m)
                              else ranges_m)

    # Under the default sensing_source="preamble" only symbol 0 (the all-pilot
    # preamble) carries any sensing energy -- every other symbol's reference is
    # zeroed (see OFDMFrame.reference_grid), so its cube slice is exactly zero. The
    # preamble slab is therefore the one image to draw.
    ra_map = range_az[0].detach().cpu().numpy()              # [az_bins, n_range]
    rng_profile = ra_map.sum(axis=0)                         # collapse az -> range profile
    est_range = isac.peak_range(ranges_m, rng_profile)
    print(f"[isac] sensing: estimated target range = {est_range:.2f} m "
          f"(peak of range profile, preamble symbol)")

    # ===== plots =====
    plt.figure()
    # Zoom the range axis to where the energy actually lives (same 99%-cumulative-
    # energy heuristic as the previous version).
    cum = np.cumsum(rng_profile) / (rng_profile.sum() + 1e-30)
    k99 = int(np.searchsorted(cum, 0.99)) + 1
    disp_max_m = max(float(ranges_m[min(k99, len(ranges_m) - 1)]) * 1.5, 2.0)
    kmax = int(np.searchsorted(ranges_m, disp_max_m)) or len(ranges_m)
    ra_z = ra_map[:, :kmax]
    n_a = ra_map.shape[0]
    u = (np.arange(n_a) - n_a // 2) / (n_a // 2)
    ra_db = to_db(ra_z, floor_db=-30.0)
    im = imshow_ra(plt.gca(), ra_db, u, ranges_m[:kmax], cmap="viridis", vmin=-30, vmax=0)
    plt.colorbar(im, label="normalized power (dB)")
    plt.xlabel("azimuth  sin(θ)")
    plt.ylabel("range (m)")
    plt.title(f"ISAC sensing range/azimuth map ({sense_src}, jsac preamble symbol)")
    ra_path = os.path.join(FIG_DIR, "isac_range_angle.png")
    plt.savefig(ra_path, dpi=120, bbox_inches="tight")
    plt.close()

    eq_np = state["comm_data_eq"].reshape(-1).cpu().numpy()
    ref_np = state["comm_tx_data"].reshape(-1).cpu().numpy()   # ground-truth TX symbols
    fig, ax = plt.subplots()
    plot_constellation(ax, eq_np, state["comm_const"], tx_syms=ref_np,
                       title=f"ISAC comm RX constellation (measured SNR={measured_snr_db:.0f} dB, "
                             f"EVM={evm_pct:.1f}%)")
    const_path = os.path.join(FIG_DIR, "isac_constellation.png")
    fig.savefig(const_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"[isac] wrote {ra_path}")
    print(f"[isac] wrote {const_path}")


if __name__ == "__main__":
    main()
