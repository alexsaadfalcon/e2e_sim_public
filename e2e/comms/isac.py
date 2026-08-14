"""
ISAC (Integrated Sensing And Communications) utilities.

Two roles:

1. **Scenario splitting** -- given a multi-node `Scenario` (see `e2e/scenario.py`),
   pull out the radar sub-problem (sensing nodes) and the comm sub-problem
   (a comm_tx -> comm_rx link). This lets a single declarative scene drive both
   the radar examples and the communications examples.

2. **Shared-waveform processing** -- in ISAC the *same* transmitted waveform is
   used for both communication (recover bits) and sensing (estimate target
   range/angle). Here we provide light-weight sensing estimators that operate on
   the S-parameter / channel frequency response the simulator already produces:
     * `range_profile`    -- IFFT across frequency -> delay/range power profile.
     * `range_angle_map`  -- FFT across the array aperture + IFFT across frequency
                             -> a range/angle image (the comms-package analogue of
                             the radar RangeAz/RangeEl blocks).

Nothing here imports Sionna. The S-parameters can come either from a precomputed
`.pkl` frame or from the synthetic fallback in `channel.py`.
"""

import numpy as np
import torch

from e2e.scenario import NodeRole

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------------
# Scenario splitting
# --------------------------------------------------------------------------------
def split_scenario(scenario):
    """Split a scenario into its radar vs comm sub-problems.

    Returns a dict:
        {
          'radar_nodes': [Node, ...],          # sensing nodes
          'comm_links':  [(tx_node, rx_node), ...],
          'is_isac':     bool,
        }

    Comm links are formed by pairing each COMM_TX with each COMM_RX (the common
    case is exactly one of each, giving a single link).
    """
    radar_nodes = scenario.nodes_by_role(NodeRole.RADAR)
    tx_nodes = scenario.nodes_by_role(NodeRole.COMM_TX)
    rx_nodes = scenario.nodes_by_role(NodeRole.COMM_RX)

    comm_links = [(tx, rx) for tx in tx_nodes for rx in rx_nodes]

    return {
        "radar_nodes": radar_nodes,
        "comm_links": comm_links,
        "is_isac": scenario.is_isac,
    }


def describe_split(scenario):
    """Human-readable summary of `split_scenario`, handy for example logs."""
    s = split_scenario(scenario)
    lines = [f"scenario '{scenario.name}' (isac={s['is_isac']}):"]
    for n in s["radar_nodes"]:
        wf = n.params.get("waveform", "?")
        lines.append(f"  sensing : {n.name} @ {n.position}  waveform={wf} "
                     f"array={n.array.num_rows}x{n.array.num_cols}")
    for tx, rx in s["comm_links"]:
        wf = tx.params.get("waveform", "?")
        lines.append(f"  comm    : {tx.name} -> {rx.name}  waveform={wf}")
    if not s["radar_nodes"]:
        lines.append("  (no radar nodes)")
    if not s["comm_links"]:
        lines.append("  (no comm links)")
    return "\n".join(lines)


# --------------------------------------------------------------------------------
# Sensing estimators on a channel frequency response
# --------------------------------------------------------------------------------
def range_profile(cfr, freqs, n_bins=None):
    """Range (delay) power profile from a 1-D channel frequency response.

    Parameters
    ----------
    cfr : [n_freqs] complex channel frequency response over `freqs`.
    freqs : the frequency grid (Hz); used to convert delay bins -> range (m).
    n_bins : IFFT length (zero-padded); defaults to len(cfr).

    Returns (ranges_m, power) numpy arrays.
    """
    cfr = torch.as_tensor(cfr, dtype=torch.complex64, device=device).reshape(-1)
    freqs = np.asarray(freqs, dtype=np.float64)
    n_f = cfr.numel()
    if n_bins is None:
        n_bins = n_f

    # IFFT across frequency -> time/delay domain
    h_delay = torch.fft.ifft(cfr, n=n_bins)
    power = (torch.abs(h_delay) ** 2).cpu().numpy()

    # delay axis -> range: bin spacing dt = 1/(B), B ~ total bandwidth
    bw = freqs[-1] - freqs[0]
    df = bw / (n_f - 1) if n_f > 1 else bw
    fs = df * n_bins                       # effective sample rate of the delay axis
    dt = 1.0 / fs
    delays = np.arange(n_bins) * dt
    c = 299_792_458.0
    ranges = c * delays / 2.0              # two-way (monostatic) range
    return ranges, power


def range_angle_map(s_pars, freqs, n_rx_x=32, n_rx_y=32, angle_bins=256, range_bins=None,
                    axis="az"):
    """Range/angle image from S-parameters, analogous to the radar RangeAz/El blocks.

    Parameters
    ----------
    s_pars : channel frequency responses, shape [N_RX, N_FREQS] (already collapsed
             to a single TX / chirp) where N_RX = n_rx_x * n_rx_y.
    freqs  : frequency grid (Hz).
    axis   : 'az' integrates over rows (azimuth cut), 'el' over columns.

    Returns (ranges_m, range_angle_power[range_bins, angle_bins]).
    """
    s_pars = torch.as_tensor(s_pars, dtype=torch.complex64, device=device)
    n_rx, n_f = s_pars.shape
    assert n_rx == n_rx_x * n_rx_y, "N_RX must equal n_rx_x * n_rx_y"
    if range_bins is None:
        range_bins = n_f

    grid = s_pars.view(n_rx_x, n_rx_y, n_f)
    # Beamform (FFT) over the KEPT aperture axis and range-compress FIRST, then
    # integrate POWER (non-coherently) over the other aperture axis. The previous
    # behavior collapsed the other axis with a COHERENT sum before transforming,
    # letting the phase progression across it destructively interfere -- a target
    # off broadside in the collapsed dimension was near-nulled (measured: peak at
    # ~1.7% of its true power). Same fix as e2e.blocks.RangeAzBlock/RangeElBlock,
    # whose docstrings describe the identical defect; this sibling was missed.
    if axis == "az":
        angle = torch.fft.fftshift(torch.fft.fft(grid, angle_bins, dim=0), dim=0)
        rng = torch.fft.ifft(angle, range_bins, dim=2)      # [angle_bins, n_rx_y, range_bins]
        power_t = torch.mean(torch.abs(rng) ** 2, dim=1)    # non-coherent over elevation
    else:
        angle = torch.fft.fftshift(torch.fft.fft(grid, angle_bins, dim=1), dim=1)
        rng = torch.fft.ifft(angle, range_bins, dim=2)      # [n_rx_x, angle_bins, range_bins]
        power_t = torch.mean(torch.abs(rng) ** 2, dim=0)    # non-coherent over azimuth
    power = power_t.T.cpu().numpy()                          # [range_bins, angle_bins]

    bw = float(freqs[-1] - freqs[0])
    df = bw / (n_f - 1) if n_f > 1 else bw
    dt = 1.0 / (df * range_bins)
    c = 299_792_458.0
    ranges = c * np.arange(range_bins) * dt / 2.0
    return ranges, power


def peak_range(ranges, power):
    """Return the range (m) of the strongest bin in a 1-D range profile."""
    power = np.asarray(power)
    return float(ranges[int(np.argmax(power))])
