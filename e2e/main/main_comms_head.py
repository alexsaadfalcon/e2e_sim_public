"""
Headline example: swappable pipeline HEADS (radar OR comms) on the SAME chain.

The full receive chain -- environment -> RFFE -> interconnect -> AFE -> AdaOja
subspace tracking -- ends in a reconstructed aperture grid, the AFE's quantized
compress/reconstruct round-trip (``y = Aq @ V``, then ``Xt = pinv(Aq) @ y``
replaces ``s_pars``). Everything downstream of that point is a swappable "head":
``RangeAzBlock``/``RangeElBlock``/``FFTBlock`` turn the reconstruction into a
radar product, while ``ModemBlock``/``BERBlock`` turn the SAME reconstruction
into a communications link. ``ModemBlock``'s ``combining`` selects how the
array feeds the comms head: ``'element0'`` taps a single spatial channel (the
historical SISO shortcut, and the "no combining" baseline); ``'egc'``
coherently combines all 1024 elements with a naive phase-only beamformer
(equal-gain combining -- co-phase each element, no amplitude weighting);
``'mrc'`` coherently combines all 1024 elements with amplitude+phase
(matched-filter, SNR-optimal) weights (independent per-element noise injected
before combining in both ``'egc'`` and ``'mrc'``, so any measured array gain
is real signal-addition gain, not noise averaging); ``'subspace'`` instead
reuses the AdaOja tracker's dominant tracked direction (``state['U'][:, 0]``)
as a broadband beamformer weight, so the SAME online-tracked subspace serves
both heads.

Run:
    python -m e2e.main.main_comms_head

Outputs (e2e/main/figures/):
    comms_head_ber.png         per-frame BER for element0/egc/mrc/subspace (log-y)
    comms_head_evm_gain.png    mean EVM per mode, annotated with array gain (dB)
    comms_head_radar_map.png   range-azimuth map from the SAME pipeline run
"""

import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")            # headless: write figures to files, no display
import matplotlib.pyplot as plt

from e2e.simulation import Simulation
from e2e.blocks import (
    SionnaEnvironmentBlock,
    RFFEBlock,
    InterconnectBlock,
    AFEBlock,
    AdaOjaBlock,
    RangeAzBlock,
    device,
)
from e2e.comms.blocks import ModemBlock, BERBlock
from e2e.viz import fig_dir, to_db, imshow_ra


FIG_DIR = fig_dir(__file__)

N_RX_X = N_RX_Y = 32
N_RX = N_RX_X * N_RX_Y
N_TX = 1
START_HZ, STOP_HZ = 28.5e9, 31.5e9
DEFAULT_SYNTH_FREQS = 256      # small but wide enough for a 64-tone OFDM slice

MODES = ("element0", "egc", "mrc", "subspace")


class _SyntheticEnvBlock:
    """Tiny drop-in environment block (get_S_pars/step/reset/array_shape), used only
    when munich.pkl is absent (or synthetic is forced) so this example always runs.

    Serves a small stack of random complex64 frames shaped [N_RX, 1, 1, F] -- the
    per-frame layout the runtime pipeline expects (mirrors
    tests/conftest.py::make_env_block, reimplemented here since main/ scripts don't
    import test fixtures).
    """

    def __init__(self, n_frames, n_freqs, n_rx=N_RX, array_shape=(N_RX_X, N_RX_Y), seed=0):
        rng = np.random.default_rng(seed)
        frames = (rng.standard_normal((n_frames, n_rx, 1, 1, n_freqs))
                  + 1j * rng.standard_normal((n_frames, n_rx, 1, 1, n_freqs)))
        self._frames = frames.astype(np.complex64)
        self.frame_counter = 0
        # advertise geometry/metadata the same way SionnaEnvironmentBlock does, so
        # Simulation/RFFEBlock auto-resolution can't tell the difference.
        self.array_shape = array_shape
        self.physical_scale = False
        self.freq_plan = None

    def __len__(self):
        return len(self._frames)

    def step(self):
        self.frame_counter = (self.frame_counter + 1) % len(self._frames)

    def reset(self):
        self.frame_counter = 0

    def get_S_pars(self):
        arr = np.ascontiguousarray(self._frames[self.frame_counter])
        return torch.from_numpy(arr).to(device)


def _make_environment(n_frames, n_freqs, force_synthetic, seed):
    """Real munich frames if present (and not force_synthetic), else a synthetic
    fallback -- this example always runs, with or without the .pkl."""
    if not force_synthetic:
        try:
            env = SionnaEnvironmentBlock('munich')
            # freqs must match the .pkl's actual frequency-bin count -- read it off the
            # first frame rather than assuming a fixed N_FREQS (get_S_pars() does not
            # advance frame_counter, so this doesn't consume a step).
            n_freqs_actual = env.get_S_pars().shape[-1]
            freqs = np.linspace(START_HZ, STOP_HZ, n_freqs_actual)
            return env, freqs, "sionna:munich"
        except FileNotFoundError:
            pass   # no munich.pkl -> synthetic below
    nf = n_freqs or DEFAULT_SYNTH_FREQS
    env = _SyntheticEnvBlock(max(int(n_frames), 1), nf, seed=seed)
    freqs = np.linspace(START_HZ, STOP_HZ, nf)
    return env, freqs, "synthetic"


def _build_simulation(environment_block, combining, freqs, k, snr_db, n_symbols, seed):
    """A fresh Simulation (fresh RFFE/interconnect/AFE/subspace state) around the SAME
    environment_block (same frames), terminating in a radar head (RangeAzBlock) PLUS
    the requested comms head (ModemBlock(combining=...) -> BERBlock)."""
    circuit_block = RFFEBlock(
        n=N_RX * N_TX,
        # Mirror the webapp's auto scale-resolution (main_sionna_blocks does the same):
        # v2 pkls carry physical_scale metadata; legacy pkls / the synthetic fallback
        # expose it as None/False, which keeps the pre-existing renormalize behavior.
        physical_scale=bool(getattr(environment_block, "physical_scale", None)),
    )
    interconnect_block = InterconnectBlock(case="case3")
    afe_block = AFEBlock()
    subspace_block = AdaOjaBlock(N_RX, k)
    modem = ModemBlock(freqs, n_symbols=n_symbols, snr_db=snr_db, seed=seed,
                       combining=combining)
    downstream_blocks = [RangeAzBlock(), modem, BERBlock()]
    return Simulation(
        environment_block,
        downstream_blocks,
        k,
        circuit_block,
        interconnect_block,
        afe_block,
        subspace_block,
    )


def main(n_frames=5, show=False, force_synthetic=False, n_freqs=None,
         k=16, snr_db=5.0, n_symbols=16, seed=0):
    """Run the same full pipeline three times (fresh Simulation each time, same
    frames) -- once per comms `combining` mode -- print a summary table, and
    (if `show`) save the figures. Returns a dict `{mode: {"ber", "evm", "gain_db"}}`
    so callers (e.g. tests) can assert on the numbers without touching matplotlib.
    """
    environment_block, freqs, source = _make_environment(n_frames, n_freqs, force_synthetic, seed)
    print(f"[comms_head] environment source: {source}")

    results = {}
    last_range_az = None
    for mode in MODES:
        sim = _build_simulation(environment_block, mode, freqs, k, snr_db, n_symbols, seed)
        outputs = sim.run(n_steps=n_frames)
        ber = np.asarray(outputs["ber"], dtype=np.float64)
        evm = np.asarray(outputs["evm"], dtype=np.float64)
        gain = outputs.get("comm_array_gain_db")   # absent for element0 (no key emitted)
        gain_db = float(np.mean(gain)) if gain else None
        results[mode] = {"ber": ber, "evm": evm, "gain_db": gain_db}
        last_range_az = outputs["range_az"][-1]

    print(f"[comms_head] {n_frames} frames, comms SNR={snr_db:.1f} dB, subspace dim k={k}")
    print("[comms_head] mode        mean BER      mean EVM      mean array gain (dB)")
    for mode in MODES:
        r = results[mode]
        gain_str = f"{r['gain_db']:.2f}" if r["gain_db"] is not None else "-"
        print(f"            {mode:10s}  {r['ber'].mean():.3e}   {r['evm'].mean():.3e}   {gain_str}")

    if show:
        _make_figures(results, last_range_az, source, snr_db)

    return results


def _make_figures(results, range_az, source, snr_db):
    # (a) BER per frame, three modes, log-y
    plt.figure()
    for mode in MODES:
        plt.semilogy(np.clip(results[mode]["ber"], 1e-6, 1), "o-", label=mode)
    plt.xlabel("frame")
    plt.ylabel("BER")
    plt.title(f"Comms head BER per frame (SNR={snr_db:.0f} dB, channel: {source})")
    plt.grid(True, which="both")
    plt.legend()
    ber_path = os.path.join(FIG_DIR, "comms_head_ber.png")
    plt.savefig(ber_path, dpi=120, bbox_inches="tight")
    plt.close()

    # (b) mean EVM per mode, bar chart annotated with array gain
    plt.figure()
    means = [results[m]["evm"].mean() for m in MODES]
    bars = plt.bar(MODES, means)
    for bar, mode in zip(bars, MODES):
        gain = results[mode]["gain_db"]
        label = f"{gain:+.1f} dB gain" if gain is not None else "no combining"
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label,
                 ha="center", va="bottom", fontsize=9)
    plt.ylabel("mean EVM (RMS fraction)")
    plt.title("Comms head mean EVM per combining mode")
    evm_path = os.path.join(FIG_DIR, "comms_head_evm_gain.png")
    plt.savefig(evm_path, dpi=120, bbox_inches="tight")
    plt.close()

    # radar head, SAME pipeline run: range-azimuth map (RangeAzBlock -> [az, range]).
    # No physical az/range axes here (bin-indexed labels below), so imshow_ra gets no
    # sin_az_axis/range_axis_m and falls back to imshow's own pixel-index extent.
    ra_db = to_db(range_az, floor_db=-40.0)
    plt.figure()
    im = imshow_ra(plt.gca(), ra_db, cmap="viridis", vmin=-40, vmax=0)
    plt.colorbar(im, label="normalized power (dB)")
    plt.xlabel("azimuth bin")
    plt.ylabel("range bin")
    plt.title("Radar head: range-azimuth map (same pipeline, last frame)")
    map_path = os.path.join(FIG_DIR, "comms_head_radar_map.png")
    plt.savefig(map_path, dpi=120, bbox_inches="tight")
    plt.close()

    print(f"[comms_head] wrote {ber_path}")
    print(f"[comms_head] wrote {evm_path}")
    print(f"[comms_head] wrote {map_path}")


if __name__ == "__main__":
    main(show=True)
