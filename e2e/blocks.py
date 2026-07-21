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


# RF Interconnect Model Block
class InterconnectBlock:
    """Placeholder interconnect filter: a fixed 11-tap boxcar impulse response.

    `apply_interconnect` zero-pads an 11-tap all-ones window to the frame length and
    takes its FFT, giving a sinc-like lowpass frequency response that is applied
    multiplicatively across the frequency axis. The tap count/shape is fixed and
    independent of the scenario's `FrequencyPlan` (bandwidth, center frequency, number
    of tones) -- it does not model any particular physical interconnect, and is
    placeholder-grade pending a frequency-plan-aware model (see ROADMAP.md).
    """

    def __init__(self, case=None):
        self.case = case

    def apply_interconnect(self, frame):
        if self.case == 'case3':
            return frame
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


class FFTBlock:
    """Azimuth-elevation power map: coherent 2D aperture FFT, non-coherent
    (power) integration over range.

    Previously the raw frequency samples were coherently summed (a bare
    ``torch.sum`` over the freq axis) before the aperture FFT, which is only
    coherent for a target sitting exactly at range 0 -- any other range
    collapses via destructive interference across the stepped-frequency
    sweep. Instead we range-transform first (freq -> range bins), then run
    the coherent aperture FFT per range sample, then power-sum over range so
    a target at ANY range still shows up in the az/el map.
    """

    def __init__(self, bins=256):
        self.bins = bins

    def apply(self, state_dict):
        frames.require_single_chirp(state_dict['s_pars'], "FFTBlock")
        data = frames.chirp0(state_dict['s_pars'])  # [az, el, n_freqs]
        # Range-FFT first (freq axis -> self.bins range bins): this shrinks the
        # freq axis (commonly n_freqs=5000) down to `bins` BEFORE the aperture
        # FFTs run, so we never materialize a [bins, bins, n_freqs] tensor.
        range_fft = torch.fft.fft(data, self.bins, 2)  # [az, el, range]
        data_fft = torch.fft.fft(torch.fft.fft(range_fft, self.bins, 0), self.bins, 1)
        data_fft = torch.fft.fftshift(torch.fft.fftshift(data_fft, 0), 1)
        # Non-coherent integration over range: sum power (not amplitude) across
        # the range axis, so a target's range doesn't have to be 0 to appear.
        az_el_power = torch.sum(torch.abs(data_fft) ** 2, dim=2)
        return {'fft': az_el_power}


class RangeAzBlock:
    """Range-azimuth power map: coherent in the displayed axes (azimuth,
    range), non-coherent (power) integration across the collapsed elevation
    axis.

    Previously elevation was collapsed by a COHERENT sum (an implicit
    un-steered broadside beam in elevation), which nulls any target off
    broadside in elevation. Here we run the coherent az/range FFTs per
    elevation element, then power-sum over elevation, so all elevation
    angles contribute instead of only broadside.
    """

    def __init__(self, bins=256):
        self.bins = bins

    def apply(self, state_dict):
        frames.require_single_chirp(state_dict['s_pars'], "RangeAzBlock")
        data = frames.chirp0(state_dict['s_pars'])  # [az, el, n_freqs]
        data_fft = torch.fft.fft(torch.fft.fft(data, self.bins, 0), self.bins, 2)
        data_fft = torch.fft.fftshift(torch.fft.fftshift(data_fft, 0), 2)
        # Non-coherent (power) integration over elevation. By Parseval, summing
        # power over the elevation ELEMENTS equals summing power over
        # elevation's own FFT bins (up to a constant factor), so no third FFT
        # over the collapsed axis is needed to integrate it away.
        range_az = torch.sum(torch.abs(data_fft) ** 2, dim=1)
        return {'range_az': range_az}


class RangeElBlock:
    """Range-elevation power map: coherent in the displayed axes (elevation,
    range), non-coherent (power) integration across the collapsed azimuth
    axis. See RangeAzBlock for the rationale (azimuth/elevation swapped)."""

    def __init__(self, bins=256):
        self.bins = bins

    def apply(self, state_dict):
        frames.require_single_chirp(state_dict['s_pars'], "RangeElBlock")
        data = frames.chirp0(state_dict['s_pars'])  # [az, el, n_freqs]
        data_fft = torch.fft.fft(torch.fft.fft(data, self.bins, 1), self.bins, 2)
        data_fft = torch.fft.fftshift(torch.fft.fftshift(data_fft, 1), 2)
        # Non-coherent (power) integration over azimuth (see RangeAzBlock).
        range_el = torch.sum(torch.abs(data_fft) ** 2, dim=0)
        return {'range_el': range_el}


class SubspaceErrorBlock:
    def __init__(self):
        self.metric = subspace_dist_frob

    def apply(self, state_dict):
        U = state_dict['U_true']
        U_pred = state_dict['U']
        return {'subspace_err': self.metric(U, U_pred)}

