"""
Intuitive "what the channel does to data" demo: transmit a standard test image
through the repo's own OFDM comms link and show what comes out the other end.

Bit mapping choice (read before changing)
-------------------------------------------
The image is transmitted as its RAW 8-bit grayscale pixel bytes (`np.unpackbits`
on the pixel array), NOT as a compressed file (PNG/JPEG) handed to the modem as
an opaque byte blob. A compressed bitstream has no redundancy left, so a single
bit error corrupts the entropy-coded stream at that point and typically makes
the rest of the file undecodable (garbage or a decoder exception) instead of a
locally visible artifact -- useless for an intuition-building demo. Raw pixel
bits instead degrade GRACEFULLY and LOCALLY: a bit error flips one bit of one
pixel byte (a small change for a low-order bit, a bright/dark speckle for a
high-order bit), so the reconstructed image visibly "salt-and-pepper" corrupts
in proportion to the channel's BER -- exactly the visual story this demo wants.

Link
----
Nothing here reimplements modulation, channel application, estimation or
equalization -- all of that is `e2e.comms.ofdm` / `e2e.comms.channel` verbatim,
mirroring `main_comms_link.py`'s structure (OFDM config, `load_or_synthesize_cfr`
channel sourcing with its Sionna-frame-if-present / synthetic-fallback so this
always runs without any `.pkl` assets, pilot LS estimate + ZF equalizer, `ber`).
Image pixel bits (padded to a whole number of OFDM symbols with random filler
bits, which are transmitted/received but discarded on reconstruction) are mapped
to 16-QAM symbols via `OFDMModem`/`qam_mod`/`qam_demod` exactly as in that
example.

One of the four operating points also drives the OFDM waveform through the
repo's TX power-amplifier nonideality model (`e2e.circuit.tx_pa.TxPA`) at 0 dB
input backoff, at high SNR so its effect isn't swamped by noise. Composing the
PA into this image path needed the SAME oversample/de-alias trick as
`main_tx_nonideality.py` (a memoryless nonlinearity generates spectral regrowth
outside the modem's own critically-sampled grid; sampling at that rate would
alias the regrowth straight back in-band) -- reproduced here in miniature
(`_oversample_time`/`_undersample_freq` below) rather than importing that
script's private helpers, to keep this module self-contained. It composed
cleanly: at 0 dB IBO the PA alone imposes an SNR-independent BER FLOOR (an
architecturally different failure mode than the noise-driven points -- more
signal power cannot buy it back), which is exactly the point of including it.

Run:
    python -m e2e.main.main_image_link
    python -m e2e.main.main_image_link --image path/to/grayscale.png

Outputs (e2e/main/figures/, not committed):
    image_link.png   top row: received images (leftmost = original); bottom
                      row: the corresponding RX constellation (leftmost =
                      ideal/noiseless TX reference); columns labeled with the
                      operating point and its measured BER.
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")            # headless: write figures to files, no display
import matplotlib.pyplot as plt

from e2e.scenario import munich_radar_scenario
from e2e.comms.ofdm import OFDMModem, random_bits, qam_demod
from e2e.comms import channel as ch
from e2e.circuit.tx_pa import TxPA, TxPAConfig


FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# --------------------------------------------------------------------------------
# Oversample / de-alias so a memoryless PA nonlinearity's spectral regrowth is
# resolved instead of aliased -- same trick as main_tx_nonideality.py, reproduced
# here (not imported) to keep this module self-contained. Exact inverses of each
# other when nothing is done to the signal in between.
# --------------------------------------------------------------------------------
def _oversample_time(tx_freq, fft_size, cp_len, oversample):
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
    cp_up = cp_len * oversample
    if cp_up > 0:
        time_up = time_up[:, cp_up:]
    grid = torch.fft.fftshift(torch.fft.fft(time_up, dim=-1), dim=-1)
    return grid[:, off:off + fft_size]


# --------------------------------------------------------------------------------
# Image <-> bits
# --------------------------------------------------------------------------------
def _default_image():
    """The standard test image: `scipy.datasets.ascent()`, 512x512 uint8 grayscale."""
    from scipy import datasets
    return datasets.ascent().astype(np.uint8)


def _load_image(path):
    """Load an arbitrary image file and convert to 8-bit grayscale."""
    from PIL import Image
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def _image_to_bits(img):
    """Raw pixel bytes -> a flat bit array (MSB first), see module docstring."""
    return np.unpackbits(img.reshape(-1).astype(np.uint8), bitorder="big")


def _bits_to_image(bits, shape):
    """Inverse of `_image_to_bits`: first `prod(shape)*8` bits -> uint8 image."""
    n = shape[0] * shape[1] * 8
    packed = np.packbits(np.asarray(bits, dtype=np.uint8)[:n], bitorder="big")
    return packed.reshape(shape)


# --------------------------------------------------------------------------------
# One operating point: modulate is shared (done once by the caller); this applies
# the point's TX path (ideal, or PA-driven), the channel, and equalization/demod.
# --------------------------------------------------------------------------------
def _run_operating_point(point, tx_freq, tx_pilots, modem, n_symbols, H_sc,
                          fft_size, cp_len, oversample, rng_seed):
    if point.get("pa", False):
        time_up, off = _oversample_time(tx_freq, fft_size, cp_len, oversample)
        cp_up = cp_len * oversample
        data_up = time_up[:, cp_up:]
        rms0 = torch.sqrt(torch.mean(torch.abs(data_up) ** 2)).item()
        normalized_up = time_up / rms0

        tx_pa = TxPA(TxPAConfig())
        cfg = tx_pa.config
        a_knee = cfg.a_sat / (10 ** (cfg.small_signal_gain_db / 20.0))
        scale = a_knee * 10 ** (-point["backoff_db"] / 20.0)
        driven_up = normalized_up * scale
        distorted_up = tx_pa.apply(driven_up)
        eff_freq = _undersample_freq(distorted_up, fft_size, cp_len, oversample, off)
    else:
        eff_freq = tx_freq

    rx_freq, _ = ch.apply_channel(eff_freq, H_sc, point["snr_db"], rng_seed=rng_seed)
    rx_pilots = modem.extract_pilots(rx_freq)
    H_est = ch.ls_estimate(rx_pilots, tx_pilots, modem.pilot_idx, modem.fft_size)
    eq = modem.extract_data(ch.zf_equalize(rx_freq, H_est))
    rx_bits = qam_demod(eq.reshape(-1), modem.bits_per_symbol, modem.const)
    # POST-equalization symbols for the constellation panel: after ZF the QAM
    # clusters are recognizable and their fuzz/merging IS the intuitive picture of
    # what each impairment does (pre-EQ symbols are channel-shaped blobs even on a
    # perfect link, which reads as "broken" to an audience).
    rx_syms = eq.reshape(-1)
    return rx_bits, rx_syms


# --------------------------------------------------------------------------------
# Operating points: high SNR (clean), mid SNR (speckle), low SNR (heavy noise),
# and a TX-nonideality point (PA at 0 dB input backoff, high SNR so the PA's own
# distortion -- not noise -- is what's on display). See module docstring.
# --------------------------------------------------------------------------------
OPERATING_POINTS = [
    {"name": "clean", "label": "Clean\n(SNR=25 dB)", "snr_db": 25.0},
    {"name": "mid", "label": "Mid SNR\n(speckle, 13 dB)", "snr_db": 13.0},
    {"name": "low", "label": "Low SNR\n(heavy noise, 6 dB)", "snr_db": 6.0},
    {"name": "pa", "label": "TX PA nonideality\n(0 dB IBO, SNR=30 dB)",
     "snr_db": 30.0, "pa": True, "backoff_db": 0.0},
]


def main(image=None, image_path=None, show=True, seed=0, force_synthetic=False,
         fft_size=64, cp_len=16, n_active=52, pilot_spacing=8, bits_per_symbol=4,
         oversample=8, operating_points=None):
    """Run the image-through-the-channel demo.

    `image` : optional pre-loaded uint8 grayscale array (mainly for tests, to
    avoid the `scipy.datasets.ascent()` download); if omitted, `image_path` is
    loaded via PIL, or (if that too is omitted) `scipy.datasets.ascent()` is used.

    Returns a dict `{"source": str, "results": {name: {"ber": float, "image":
    ndarray, "rx_syms": ndarray}}, "tx_syms": ndarray, "original": ndarray}` so
    tests can assert on the numbers without touching matplotlib.
    """
    if operating_points is None:
        operating_points = OPERATING_POINTS

    if image is not None:
        img = np.asarray(image, dtype=np.uint8)
    elif image_path is not None:
        img = _load_image(image_path)
    else:
        img = _default_image()

    rng = np.random.default_rng(seed)

    scenario = munich_radar_scenario()
    freqs = scenario.frequency.linspace()
    carrier = scenario.frequency.carrier_hz
    subcarrier_spacing = 240e3   # narrow comm band, see main_comms_link's note

    modem = OFDMModem(fft_size=fft_size, cp_len=cp_len, n_active=n_active,
                      pilot_spacing=pilot_spacing, bits_per_symbol=bits_per_symbol)

    cfr_dense, source = ch.load_or_synthesize_cfr("munich", freqs, rng=rng,
                                                   prefer_pkl=not force_synthetic)
    print(f"[image_link] channel source: {source}")
    H_sc = ch.cfr_to_subcarriers(cfr_dense, freqs, modem.fft_size, carrier, subcarrier_spacing)

    img_bits = _image_to_bits(img)
    n_img_bits = img_bits.size
    n_symbols = -(-n_img_bits // modem.data_bits_per_symbol_block)   # ceil div
    n_pad = n_symbols * modem.data_bits_per_symbol_block - n_img_bits
    pad_bits = random_bits(n_pad, seed=seed + 1).cpu().numpy() if n_pad else np.zeros(0, dtype=np.int64)
    tx_bits = np.concatenate([img_bits.astype(np.int64), pad_bits])

    _, tx_freq = modem.modulate(tx_bits, n_symbols)
    tx_data_syms = modem.extract_data(tx_freq).reshape(-1).cpu().numpy()
    tx_pilots = modem.pilot_grid(n_symbols)

    results = {}
    print(f"[image_link] image {img.shape[0]}x{img.shape[1]} -- {n_img_bits} bits, "
          f"{n_symbols} OFDM symbols ({modem.data_bits_per_symbol_block} bits/symbol)")
    print("[image_link] point                          BER")
    for i, point in enumerate(operating_points):
        rx_bits, rx_syms = _run_operating_point(
            point, tx_freq, tx_pilots, modem, n_symbols, H_sc,
            fft_size, cp_len, oversample, rng_seed=1000 + i)
        rx_bits_np = rx_bits.cpu().numpy()
        ber = ch.ber(tx_bits, rx_bits_np)
        rx_img = _bits_to_image(rx_bits_np, img.shape)
        results[point["name"]] = {"ber": ber, "image": rx_img,
                                  "rx_syms": rx_syms.cpu().numpy()}
        print(f"            {point['name']:>28s}   {ber:.4e}")

    out = {"source": source, "results": results, "tx_syms": tx_data_syms, "original": img}
    if show:
        _make_figure(img, tx_data_syms, results, operating_points, out["source"])
    return out


def _make_figure(original, tx_syms, results, operating_points, source):
    ncols = 1 + len(operating_points)
    fig, axes = plt.subplots(2, ncols, figsize=(3.2 * ncols, 6.6))

    # leftmost column: original image (top), ideal/noiseless TX reference constellation (bottom)
    axes[0, 0].imshow(original, cmap="gray", interpolation="nearest", vmin=0, vmax=255)
    axes[0, 0].set_title("Original", fontsize=11)
    axes[0, 0].axis("off")

    axes[1, 0].scatter(tx_syms.real, tx_syms.imag, s=3, alpha=0.35, color="black")
    axes[1, 0].set_title("TX (ideal)", fontsize=11)
    axes[1, 0].axis("equal")
    axes[1, 0].grid(True)
    axes[1, 0].tick_params(labelsize=8)

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, point in enumerate(operating_points):
        col = i + 1
        r = results[point["name"]]
        axes[0, col].imshow(r["image"], cmap="gray", interpolation="nearest", vmin=0, vmax=255)
        axes[0, col].set_title(f"{point['label']}\nBER={r['ber']:.2e}", fontsize=11)
        axes[0, col].axis("off")

        syms = r["rx_syms"]
        # Normalize each panel to unit RMS so the constellations are visually
        # comparable across operating points -- the PA point's raw output carries
        # the amplifier's absolute gain (|syms| ~ 1e2), which would dwarf every
        # other panel's axis scale and hide the SHAPE distortion that matters.
        rms = float(np.sqrt(np.mean(np.abs(syms) ** 2))) or 1.0
        syms = syms / rms
        axes[1, col].scatter(syms.real, syms.imag, s=3, alpha=0.35, color=colors[i % 10])
        axes[1, col].set_title(f"{point['name']} (unit RMS)", fontsize=11)
        axes[1, col].axis("equal")
        axes[1, col].grid(True)
        axes[1, col].tick_params(labelsize=8)

    fig.suptitle(f"Image through the OFDM comms link (channel: {source})", fontsize=13)
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "image_link.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[image_link] wrote {fig_path}")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=str, default=None,
                   help="path to a grayscale image (default: scipy.datasets.ascent())")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(image_path=args.image, show=True)
