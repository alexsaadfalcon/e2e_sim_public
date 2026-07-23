from pathlib import Path

import numpy as np
import torch

from e2e.environment.sionna_iterator import SionnaEtoileIterator, SionnaMunichIterator
from e2e.subspace.algorithms import Oja, gen_A_ada
from e2e.subspace.subspace_utils import subspace_dist_frob
from e2e.afe.afe_utils import quantizer_fp
from e2e.circuit.rffe_model import get_RX_config, circuit_model_batch
from e2e import frames


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SionnaEnvironmentBlock:
    def __init__(self, scenario_name, array_shape=None, link=None):
        valid_scenarios = {
            'etoile': SionnaEtoileIterator,
            'munich': SionnaMunichIterator,
        }
        if scenario_name not in valid_scenarios:
            raise ValueError(f'unknown scenario {scenario_name}')
        self.scenario_name = scenario_name
        # `link` selects which link of a multi-link pkl to consume; default None keeps the
        # existing behavior (first link, or the single array for legacy munich/etoile pkls).
        self.link = link
        self.sionna_iterator = valid_scenarios[scenario_name](link=link)
        self.frame_counter = 0
        # receive-array geometry (n_rx_x, n_rx_y); Simulation reads this to reshape frames.
        # array_shape=None (default) auto-derives: explicit arg wins, else the v2 pkl's
        # rx_array_shape metadata, else the legacy (32, 32) fallback for pkls with no meta.
        # The metadata stores [num_rows, num_cols] (ArrayConfig order), but Sionna's
        # PlanarArray numbers antennas column-first (row index varies FASTEST along the
        # frame's flat RX axis), so the row-major aperture view needs the slow axis
        # first: (num_cols, num_rows). Grid dim 0 is then columns (horizontal/azimuth)
        # and dim 1 rows (vertical/elevation) -- the convention RangeAz/RangeEl assume.
        # (Square legacy arrays are unaffected; see frames.to_aperture_grid.)
        meta_shape = self.sionna_iterator.rx_array_shape  # (num_rows, num_cols) or None
        auto_shape = (meta_shape[1], meta_shape[0]) if meta_shape else None
        self.array_shape = array_shape or auto_shape or (32, 32)
        # v2-only metadata pass-throughs for downstream consumers (e.g. the webapp); both
        # are None for legacy pkls.
        self.freq_plan = self.sionna_iterator.freq_plan
        self.physical_scale = self.sionna_iterator.physical_scale
    
    def step(self):
        self.frame_counter += 1
        if self.frame_counter >= len(self.sionna_iterator):
            self.frame_counter = 0

    def reset(self):
        self.frame_counter = 0

    def get_S_pars(self):
        s_pars = np.asarray(self.sionna_iterator[self.frame_counter], dtype=np.complex64)
        s_pars = torch.from_numpy(s_pars)
        s_pars = s_pars.to(device)
        return s_pars


# RF Frontend Block
class RFFEBlock:
    def __init__(self, n=None, freq_span_hz=3e9, signal_scaling=1e-5, if_filter=False,
                 physical_scale=False, chirp_dur=None):
        # chirp_dur: legacy kwarg accepted (and ignored) so existing call sites that
        # still pass it don't break.
        # fs (= freq_span_hz) is the complex-baseband buffer's true sample rate (the
        # default 28.5-31.5 GHz plan -> fs = 3e9, dt ~333 ps, record ~1.667 us). It
        # sizes the optional IF filter's boxcar width ONLY; the thermal-noise floor
        # is band-referenced to the receiver's IF bandwidth inside the circuit model
        # (stepped-frequency measurement semantics), not to fs.
        self.rx_config = get_RX_config(n).to(device)
        self.fs = freq_span_hz
        self.signal_scaling = signal_scaling
        self.if_filter = if_filter
        self.physical_scale = physical_scale
        self.n = n

    def apply_circuit(self, s_pars):
        # Chirp/pol dim size is inferred (-1) rather than hardcoded to 2: this makes
        # the reshape a no-op for whatever n_chirp the caller actually passes (1, the
        # single-chirp frames Simulation feeds via CircuitStage; or 2, as some direct
        # callers/tests still exercise) instead of assuming a pol pair.
        s_pars_shape = s_pars.shape
        s_pars = s_pars.view(self.n, 1, -1, s_pars.shape[-1])
        frame = torch.fft.ifft(s_pars, dim=-1)
        if not self.physical_scale:
            frame = frame * self.signal_scaling / torch.mean(torch.abs(frame))
        # physical_scale=True: skip the normalization above -- the frame is already
        # in volts at the LNA input (the generation layer produces volts via
        # sqrt(N*P_tx*Z0) scaling; see e2e/environment/scenario_runner.py).
        frame_dist, PRX = circuit_model_batch(self.rx_config, frame, self.fs,
                                              if_filter=self.if_filter)
        s_pars_dist = torch.fft.fft(frame_dist, dim=-1)
        s_pars_dist = s_pars_dist.view(s_pars_shape)
        return s_pars_dist, PRX


# Packaged interconnect transfer-function data (a Tessera TSV S21(f) sweep; the model
# that produced it is external and not vendored -- only this derived CSV ships). See
# e2e/data/interconnect/README.md.
TESSERA_INTERCONNECT_CSV = (
    Path(__file__).resolve().parent / "data" / "interconnect" / "tessera_tsv_s21.csv"
)


def load_interconnect_transfer(path):
    """Load an interconnect transfer function ``(freq_hz, s21)`` from a CSV.

    The CSV must have a header row with at least ``freq_hz``, ``s21_re`` and ``s21_im``
    columns (extra columns are ignored); ``#``-prefixed comment lines are skipped.
    Returns two numpy arrays sorted by ascending frequency: ``freq_hz`` (float) and
    ``s21`` (complex). See e2e/data/interconnect/ for the shipped file and its provenance.
    """
    freqs, s21 = [], []
    cols = None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if cols is None:  # first non-comment line is the header row
                names = [c.strip() for c in line.split(",")]
                cols = (names.index("freq_hz"), names.index("s21_re"), names.index("s21_im"))
                continue
            parts = line.split(",")
            freqs.append(float(parts[cols[0]]))
            s21.append(complex(float(parts[cols[1]]), float(parts[cols[2]])))
    freq = np.asarray(freqs, dtype=np.float64)
    s = np.asarray(s21, dtype=np.complex128)
    order = np.argsort(freq)
    return freq[order], s[order]


# RF Interconnect Model Block
class InterconnectBlock:
    """Interconnect filtering, applied multiplicatively across the frequency axis.

    Two modes:

    - **Placeholder (default):** a fixed 11-tap boxcar impulse response. `apply_interconnect`
      zero-pads an 11-tap all-ones window to the frame length and FFTs it, giving a
      sinc-like lowpass frequency response applied across the frequency axis. The tap
      count/shape is fixed and independent of the scenario's `FrequencyPlan`; it does not
      model any particular physical interconnect.

    - **Data-driven (`transfer_csv`):** a simulated/measured interconnect transfer function
      S21(f) loaded from a CSV (see e2e/data/interconnect/) and interpolated onto the
      frame's frequency grid, then applied as ``frame * S21(f)``. `band_hz=(f_start, f_stop)`
      is the physical span the frame's `n_freqs` samples cover (e.g. the FrequencyPlan's
      band); when omitted the CSV's own frequency span is mapped across the frame's samples
      (band-agnostic). Grid points outside the CSV's frequency range clamp to its endpoints.

    `case='case3'` is an identity pass-through in either mode.
    """

    def __init__(self, case=None, transfer_csv=None, band_hz=None):
        self.case = case
        self.transfer_csv = transfer_csv
        self.band_hz = band_hz
        self._csv_freq = None
        self._csv_s21 = None
        if transfer_csv is not None:
            self._csv_freq, self._csv_s21 = load_interconnect_transfer(transfer_csv)

    def _resampled_response(self, n_freqs, dev):
        """Interpolate the loaded S21(f) onto the frame's `n_freqs`-point grid."""
        if self.band_hz is not None:
            grid = np.linspace(self.band_hz[0], self.band_hz[1], n_freqs)
        else:
            grid = np.linspace(self._csv_freq[0], self._csv_freq[-1], n_freqs)
        # The transfer function is smooth and densely sampled, so linear interpolation of
        # the real and imaginary parts is faithful (and avoids phase-unwrap subtleties).
        re = np.interp(grid, self._csv_freq, self._csv_s21.real)
        im = np.interp(grid, self._csv_freq, self._csv_s21.imag)
        return torch.tensor(re + 1j * im, dtype=torch.complex64, device=dev)

    def apply_interconnect(self, frame):
        if self.case == 'case3':
            return frame
        if self.transfer_csv is not None:
            H = self._resampled_response(frame.shape[-1], frame.device)
            return frame * H.view(1, 1, 1, -1)
        window = torch.ones(11)
        window = window.to(device)
        window_padded = torch.nn.functional.pad(window, (0, frame.shape[-1] - window.shape[0]))
        window_padded = window_padded.to(device)
        # This is a forward FFT of the (zero-padded) impulse response, i.e. the
        # interconnect's frequency response -- despite the historical name, it is not
        # an inverse FFT. Do not "fix" the direction; that would change the filter.
        window_freq_response = torch.fft.fft(window_padded)
        frame = frame * window_freq_response.view(1, 1, 1, -1)
        return frame


# Adaptive Feature Extraction Block
class AFEBlock:
    def __init__(self, exp=5, mantissa=6):
        self.exp = exp
        self.mantissa = mantissa

    def apply_mat_mul(self, A, V):
        Aq_real = quantizer_fp(A.real, self.exp, self.mantissa)
        Aq_imag = quantizer_fp(A.imag, self.exp, self.mantissa)
        Aq = Aq_real + 1j * Aq_imag
        X = Aq @ V
        return Aq, X

    def reconstruct(self, Aq, X):
        return torch.linalg.pinv(Aq) @ X


# Adaptive Oja's Algorithm Block
class AdaOjaBlock:
    def __init__(self, n, d, eta=0.1):
        # eta is the (fixed) Oja step size. Default lowered from 1.0 to 0.1: with the
        # tracker now actually running across frames (the ground-truth reset was
        # removed from Simulation.feed_forward), eta=1.0 is too aggressive and the
        # basis diverges; ~0.1 tracks stably. Tune per scene / measurement regime.
        self.oja = Oja(n, d, eta=eta, fixed_step=True)

    def gen_A_ada(self, m=None):
        if m is None:
            m = self.oja.d * 2
        return gen_A_ada(self.oja.U, m)

    def update(self, X, A):
        # RMS-normalize X before tracking: the Oja gradient scales as |X|^2 with the
        # input amplitude, so feeding absolute-volts data (post physical_scale=True
        # RFFE) would silently retune eta. The tracked basis direction is
        # scale-invariant, so this decouples the tracker step size from the input's
        # absolute physical signal scale.
        xn = X / (torch.sqrt(torch.mean(torch.abs(X) ** 2)) + 1e-30)
        self.oja.add_data(xn, A)


# -------------------------------------------------------------- serial pipeline stages
# These implement the same protocol as downstream product blocks (`apply(state) ->
# dict of updates`), but they run in the FIRST loop of Simulation.feed_forward, in
# series, each one able to (and expected to) update 's_pars' -- they ARE the pipeline,
# not products of it. See e2e/simulation.py.

class CircuitStage:
    """RF front-end circuit distortion, applied directly to the single-chirp frame.

    Previously this duplicated the chirp dim (torch.cat([s, s], dim=2)) to feed
    RFFEBlock a vestigial "pol pair", ran the full circuit on both copies (2x
    compute, 2 independent noise draws), then discarded the second copy via
    s_pars[:, :, :1, :]. RFFEBlock.apply_circuit's reshape now infers the chirp dim
    instead of hardcoding 2, so it accepts the single-chirp frame natively -- one
    noise realization per element, no wasted compute.
    """

    def __init__(self, rffe_block):
        self.rffe_block = rffe_block

    def apply(self, state):
        s_pars, PRX = self.rffe_block.apply_circuit(state["s_pars"])
        return {"s_pars": s_pars, "PRX": PRX}


class GridStage:
    """Validates the frame (no MIMO, single chirp) and reshapes it onto the physical
    receive-array grid."""

    def __init__(self, array_shape):
        self.array_shape = array_shape

    def apply(self, state):
        s_pars = state["s_pars"]
        frames.require_no_mimo(s_pars, "Simulation.feed_forward")
        frames.require_single_chirp(s_pars, "Simulation.feed_forward")
        s_pars = frames.to_aperture_grid(s_pars, self.array_shape)
        return {"s_pars": s_pars}


class InterconnectStage:
    """Interconnect filtering on the aperture grid."""

    def __init__(self, interconnect_block):
        self.interconnect_block = interconnect_block

    def apply(self, state):
        s_pars = self.interconnect_block.apply_interconnect(state["s_pars"])
        return {"s_pars": s_pars}


class MeasurementStage:
    """Adaptive compression (optional AFE) + online subspace tracking.

    With an AFE block: quantized measurement matrix, quantized matmul, subspace
    update, then reconstruction -- the reconstructed grid replaces 's_pars'.
    Without an AFE block: the subspace tracker is fed the full-precision compressed
    measurements directly (same A-generation, no quantization); 's_pars' is left
    unchanged.
    """

    def __init__(self, afe_block, subspace_block):
        self.afe_block = afe_block
        self.subspace_block = subspace_block

    def apply(self, state):
        s_pars = state["s_pars"]
        if self.afe_block:
            V = s_pars.view(-1, s_pars.shape[-1])
            A = self.subspace_block.gen_A_ada()
            Aq, X = self.afe_block.apply_mat_mul(A, V)
            self.subspace_block.update(X, Aq)
            Xt = self.afe_block.reconstruct(Aq, X)
            s_pars = Xt.view(s_pars.shape)
            return {"s_pars": s_pars, "U": self.subspace_block.oja.U}
        else:
            # No AFE: feed the subspace tracker the full-precision compressed
            # measurements directly (same A-generation as the AFE branch, but
            # without quantization).
            V = s_pars.view(-1, s_pars.shape[-1])
            A = self.subspace_block.gen_A_ada()
            X = A @ V
            self.subspace_block.update(X, A)
            return {"U": self.subspace_block.oja.U}


def _aperture_window(kind, length, device):
    """Real 1-D aperture taper of `length` for sidelobe control, or None (uniform).

    Supported kinds: None / 'none' (no taper -- the default, bit-for-bit unchanged
    behavior), 'hann', 'hamming'. The taper multiplies an aperture (angle) axis
    before its FFT; it is never applied to the range/frequency axis.
    """
    if kind is None or kind == 'none':
        return None
    if kind == 'hann':
        w = torch.hann_window(length, periodic=False, device=device)
    elif kind == 'hamming':
        w = torch.hamming_window(length, periodic=False, device=device)
    else:
        raise ValueError(
            f"unknown aperture window {kind!r}; use None, 'hann', or 'hamming'"
        )
    return w.to(torch.float32)


def _power_bin(power, n_bins, dim):
    """Reduce a real `power` tensor along `dim` from length L to `n_bins` display
    gates by summing power within contiguous groups (zero-padded up to a multiple
    of n_bins).

    Energy- and extent-preserving: a point target's coherently range-compressed
    energy sits in ONE native range bin, which lands whole inside one display gate,
    so the gate captures the full integration gain (unlike truncating the frequency
    axis, which throws most of the band away). Identity when L == n_bins; a pass-
    through when L < n_bins (nothing to integrate down).
    """
    L = power.shape[dim]
    if L <= n_bins:
        return power
    per = -(-L // n_bins)              # ceil division
    pad = per * n_bins - L
    if pad:
        pad_shape = list(power.shape)
        pad_shape[dim] = pad
        power = torch.cat([power, power.new_zeros(pad_shape)], dim=dim)
    power = power.movedim(dim, -1)
    power = power.reshape(*power.shape[:-1], n_bins, per).sum(dim=-1)
    return power.movedim(-1, dim)


class FFTBlock:
    """Azimuth-elevation power map: coherent 2D aperture FFT + full-band range
    compression, with non-coherent (power) integration over range.

    ``bins`` sizes ONLY the azimuth/elevation (aperture) FFTs. The range transform
    always spans the FULL frequency band (all ``n_freqs`` samples). Earlier code
    passed ``bins`` as the range-FFT length too, which TRUNCATED the frequency axis
    to its first ``bins`` samples -- discarding most of the swept band (both range
    resolution and ~10*log10(n_freqs/bins) dB of coherent integration gain). Because
    range is integrated away in this product, we range-compress over the full band,
    aperture-FFT each range bin (chunked to bound memory), and power-sum over range,
    so a target at ANY range appears at full SNR.

    ``window`` optionally tapers the aperture (angle) axes for sidelobe control
    (None / 'hann' / 'hamming'); it never touches the range axis. The default None
    is the untapered map (numbers unchanged from the uniform-aperture case).
    """

    def __init__(self, bins=256, window=None):
        self.bins = bins
        self.window = window

    def apply(self, state_dict):
        frames.require_single_chirp(state_dict['s_pars'], "FFTBlock")
        data = frames.chirp0(state_dict['s_pars'])  # [az, el, n_freqs]
        n_az, n_el, n_freqs = data.shape
        # Full-band range compression (freq -> range); NOT truncated to `bins`.
        range_full = torch.fft.fft(data, dim=2)     # [az, el, n_freqs]
        w_az = _aperture_window(self.window, n_az, data.device)
        w_el = _aperture_window(self.window, n_el, data.device)
        if w_az is not None:
            range_full = range_full * w_az.view(n_az, 1, 1)
        if w_el is not None:
            range_full = range_full * w_el.view(1, n_el, 1)
        # Aperture FFT per range bin, power-integrated over range, chunked over the
        # range axis so we never materialize a [bins, bins, n_freqs] tensor.
        acc = torch.zeros(self.bins, self.bins, dtype=torch.float32, device=data.device)
        chunk = 256
        for r0 in range(0, n_freqs, chunk):
            blk = range_full[:, :, r0:r0 + chunk]
            ap = torch.fft.fft(torch.fft.fft(blk, self.bins, 0), self.bins, 1)
            ap = torch.fft.fftshift(torch.fft.fftshift(ap, 0), 1)
            acc = acc + torch.sum(torch.abs(ap) ** 2, dim=2)
        return {'fft': acc}


class RangeAzBlock:
    """Range-azimuth power map: coherent azimuth FFT + full-band range
    compression, non-coherent (power) integration across the collapsed
    elevation axis.

    As in FFTBlock, ``bins`` sizes only the azimuth (aperture) FFT and the range
    DISPLAY axis; the range compression spans the full frequency band (all
    ``n_freqs`` samples) rather than truncating to the first ``bins``. The full-band
    range profile is power-binned down to ``bins`` display gates (energy- and
    extent-preserving), and elevation is integrated non-coherently (power) so a
    target off broadside in elevation survives -- previously elevation was collapsed
    by a COHERENT sum (an implicit broadside beam) that nulled off-broadside targets.

    ``window`` optionally tapers the azimuth aperture (None / 'hann' / 'hamming').
    """

    def __init__(self, bins=256, window=None):
        self.bins = bins
        self.window = window

    def apply(self, state_dict):
        frames.require_single_chirp(state_dict['s_pars'], "RangeAzBlock")
        data = frames.chirp0(state_dict['s_pars'])  # [az, el, n_freqs]
        n_az, n_el, n_freqs = data.shape
        w_az = _aperture_window(self.window, n_az, data.device)
        # Accumulate power over the collapsed (elevation) axis one element at a
        # time -- bounds memory to a single [bins, n_freqs] slab regardless of
        # aperture size or band length.
        power = torch.zeros(self.bins, n_freqs, dtype=torch.float32, device=data.device)
        for e in range(n_el):
            col = data[:, e, :]                      # [az, n_freqs]
            if w_az is not None:
                col = col * w_az.view(n_az, 1)       # taper before the aperture FFT
            a = torch.fft.fftshift(torch.fft.fft(col, self.bins, 0), 0)   # [bins, n_freqs]
            r = torch.fft.fftshift(torch.fft.fft(a, dim=1), 1)            # full-band range
            power = power + torch.abs(r) ** 2
        range_az = _power_bin(power, self.bins, dim=1)   # n_freqs range bins -> bins gates
        return {'range_az': range_az}


class RangeElBlock:
    """Range-elevation power map: coherent elevation FFT + full-band range
    compression, non-coherent (power) integration across the collapsed azimuth
    axis. See RangeAzBlock for the rationale (azimuth/elevation swapped): ``bins``
    sizes only the elevation aperture FFT and the range display axis; range
    compression uses the full band and is power-binned to ``bins`` gates.

    ``window`` optionally tapers the elevation aperture (None / 'hann' / 'hamming').
    """

    def __init__(self, bins=256, window=None):
        self.bins = bins
        self.window = window

    def apply(self, state_dict):
        frames.require_single_chirp(state_dict['s_pars'], "RangeElBlock")
        data = frames.chirp0(state_dict['s_pars'])  # [az, el, n_freqs]
        n_az, n_el, n_freqs = data.shape
        w_el = _aperture_window(self.window, n_el, data.device)
        # Accumulate power over the collapsed (azimuth) axis one element at a time.
        power = torch.zeros(self.bins, n_freqs, dtype=torch.float32, device=data.device)
        for m in range(n_az):
            col = data[m, :, :]                      # [el, n_freqs]
            if w_el is not None:
                col = col * w_el.view(n_el, 1)       # taper before the aperture FFT
            a = torch.fft.fftshift(torch.fft.fft(col, self.bins, 0), 0)   # [bins, n_freqs]
            r = torch.fft.fftshift(torch.fft.fft(a, dim=1), 1)            # full-band range
            power = power + torch.abs(r) ** 2
        range_el = _power_bin(power, self.bins, dim=1)   # n_freqs range bins -> bins gates
        return {'range_el': range_el}


class SubspaceErrorBlock:
    def __init__(self):
        self.metric = subspace_dist_frob

    def apply(self, state_dict):
        U = state_dict['U_true']
        U_pred = state_dict['U']
        return {'subspace_err': self.metric(U, U_pred)}

