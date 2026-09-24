"""OFDM-ISAC: the comms package folded ONTO the one chain.

This module is what makes `ofdm` and `jsac` waveform classes rather than labels. It
holds three things and nothing else:

1. **`OFDMFrame`** -- the transmitted grid `X [M, N]` (an all-pilot preamble symbol
   followed by `M-1` data symbols on a comb), built from the SOURCE's own frequency
   plan so the subcarrier grid and the stored channel's grid are the same grid by
   construction rather than by interpolation.
2. **`SymbolDivisionBlock`** -- THE mixing block for these two classes, the counterpart
   of `DechirpBlock`. It divides the received grid by the transmitted one and then runs
   *the same tail* `beat_from_cfr` runs -- imported from `e2e.chain.dechirp`, not
   reimplemented, because the bit-parity oracle (O2 below) is worth nothing if the two
   tails can drift apart.
3. **`OFDMReceiveBlock`** -- the comms head as a CONSUMER of the chain: it reads the
   received grid the chain produced, beamforms, estimates, equalises and demaps, and
   emits exactly the keys `BERBlock` already consumes. It injects NO noise of its own.

Design note: `notes/JSAC_WAVEFORM_2026-09-24.md` (literature map, the choice of
OFDM-ISAC, the numerology) plus its adversarial review. The review's binding change
list is implemented here: `ofdm` is its own class (not `jsac` with a tab hidden), the
equaliser's SNR is MEASURED rather than assumed or silently defaulted, the
resource-split knob is built ahead of the QAM-order knob, and the handedness question
has an oracle instead of an assurance.

THE THREE CLASSES, as the triple (source waveform, mixing mode, product set):

    fmcw   chirp      dechirp           sensing products only
    ofdm   OFDM grid  none (equalise)   comms products only
    jsac   OFDM grid  symbol_division   BOTH, from one frame, with a resource split

`ofdm` genuinely has no mixing block: a comms receiver equalises the grid, it does not
form a cube. That is the honest reading of "comms (OFDM)" and it is what makes the
difference between `ofdm` and `jsac` visible on a screen instead of assertible in
prose -- same waveform, same front end, one extra block, and a radar image appears.

WHY A FORWARD FFT AND NOT AN IFFT, when the literature says "IFFT over subcarriers":
because the conjugate is already in the tail. `FFT(conj(H)) = N * conj(IFFT(H))`, so the
forward FFT of the conjugated, divided grid has exactly the magnitude of the textbook
OFDM range profile -- which is what lets ONE `RangeTransformBlock` serve both waveforms,
and what makes the parity oracle exact rather than approximate.

THE ORACLES (`tests/test_waveform_classes.py`), and what each is worth:

* **O1, known delay** -- a one-tap CFR at delay tau peaks at cube bin
  `round(tau * N * df)`. Written against `N * df`, never a nominal `B`, so it cannot
  bake in the endpoint-inclusive off-by-one. This is the oracle that carries PHYSICS.
* **O2, FMCW parity** -- with `X == 1` (the all-pilot preamble) the JSAC cube is
  BIT-IDENTICAL to the FMCW cube on the same frame. Certifies the plumbing: conjugate
  present, antenna flip present, `mimo_combine` indexing right, FFT direction right,
  symbol axis mapped to the chirp axis right. It CANNOT catch a wrong `df`, a wrong
  frequency ordering, or anything in the data-bearing path -- both arms share those.
  It is the right gate for "is this one chain or two" and the wrong gate for "is the
  image correct". Do not loosen it; if it stops being exact, the tail has drifted.
* **O3, BER floor** -- a flat channel with no noise gives BER exactly 0.
* **O4, division noise** -- 16-QAM raises the cube's noise floor by
  `10*log10(E[1/|X|^2])` = +2.76 dB in the mean over QPSK. Closed form, so the test
  computes the constellation expectation rather than asserting a literal.
* **O5, handedness** -- the same one-tap CFR through the FMCW arm and the JSAC arm
  lands at the SAME range bin and the same azimuth bin. This one exists because the
  review measured the shipped v1.0 image path and the ADC path peaking at mirrored
  range bins, and an A/B screen showing two mirrored images is exactly the failure a
  side-by-side demo makes most visible.

WHAT THIS MODULE DELIBERATELY DOES NOT CLAIM
* **The Doppler axis is degenerate**, and that is a property of the CORPUS, not of
  JSAC: the stored munich channel is one CFR per frame and time-invariant within it, so
  all `M` symbols see the same `H` and the slow axis is a delta at bin 0. The FMCW arm
  on the same frames has exactly the same degenerate Doppler axis. Scene motion lives
  ACROSS frames for both. `RadarCubeBlock` refuses a symbol-slow cube by name rather
  than drawing it.
* **The cyclic prefix is decorative in this model.** `apply_ofdm_channel` multiplies per
  subcarrier, which IS a cyclic convolution by construction, so no CP length can produce
  inter-symbol interference and none can prevent it. The CP is charged honestly to the
  data rate and `cp_guard_ok` checks the premise against the measured PDP, but nothing
  in the chain ever convolves in time. Say that once, out loud, rather than letting a
  reader find it.
* **PAPR claims about the front-end clamp are not made here**, because the measurement
  runs the other way from the intuition: `RFFEBlock` normalises by the MEAN magnitude of
  `ifft(s_pars)`, and on this trace the LoS tap carries ~93% of the energy, so the
  SHIPPED FMCW preset is the peakier one at the LNA. `measure_lna_input_papr` is
  provided so a card quotes a measured number for the configuration it actually runs.
  Worse for the intuition, and measured on `munich_ka.pkl` frame 0 through THIS module
  2026-09-24: the ALL-PILOT PREAMBLE is 36.67 dB at the LNA -- bit-identically the bare
  CFR's own 36.67 dB, because `X == 1` makes `Y` the bare CFR -- while the data symbols
  are 19.34 dB. So the peakiest symbol in a JSAC frame is the one that is the FMCW
  preset. See `measure_lna_input_papr`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch

from e2e import frames
from e2e.chain.dechirp import beat_from_cfr, mimo_combine
from e2e.frames import FrameCapabilities

from .ofdm import OFDMModem, qam_constellation, random_bits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#: Where the sensing reference comes from. "preamble" uses symbol 0 (all-pilot, X == 1)
#: and is the parity point; "all_symbols" divides every symbol by its own transmitted
#: value, which is the textbook symbol-division method and pays the 1/|X|^2 noise
#: amplification O4 measures; "pilots_only" keeps the comb subcarriers of every symbol
#: and zeroes the data ones -- the resource split (see `OFDMFrame.pilot_spacing`).
SENSING_SOURCES = ("preamble", "all_symbols", "pilots_only")


# ================================================================================
# Numerology, derived -- never pasted
# ================================================================================
def subcarrier_spacing_hz(freq_plan) -> float:
    """The grid spacing an OFDM frame must use to sit ON a stored CFR's own grid.

    Endpoint-inclusive, `(stop - start) / (num_freqs - 1)`, because
    `sionna_simple_channel.build_frequencies` is an `np.linspace`. For the shipped
    munich Ka file that is 600 120.024 Hz, NOT 600 000 -- a 0.02% error is ~0.45 m at
    the far end of the window, which is why this is computed and never typed.
    (`e2e.chain.receive.delta_f_from_freq_plan` is the same arithmetic for the same
    reason; this wrapper exists so the comms side names it in its own vocabulary.)
    """
    from e2e.chain.receive import delta_f_from_freq_plan

    df = delta_f_from_freq_plan(freq_plan)
    if df is None:
        raise ValueError(
            "an OFDM/JSAC frame needs the SOURCE's frequency plan to place its "
            "subcarriers on the stored channel's own grid -- start_hz, stop_hz and "
            f"num_freqs; got {freq_plan!r}. Interpolating onto a foreign grid is "
            "the one thing this design exists to avoid."
        )
    return float(df)


def qam_division_noise_rise_db(bits_per_symbol: int):
    """`(mean, worst)` dB by which dividing by a QAM constellation raises the cube's
    noise floor, relative to a constant-modulus (QPSK) grid.

    `Z = Y/X` scales the noise on subcarrier `k` by `1/|X_k|`, so the floor rises by
    `10*log10(E[1/|X|^2])` in the mean and `10*log10(1/min|X|^2)` on the worst
    subcarrier. Computed from the repo's OWN `qam_constellation` (which sets the
    normalisation), so the number cannot drift away from the constellation actually
    transmitted. This is Xiong et al.'s deterministic-random tradeoff in closed form:
    a more informative waveform is a worse ranging waveform at the same power.

    QPSK 0.00 / 0.00 dB; 16-QAM +2.76 / +6.99; 64-QAM +4.29 / +13.22 (2026-09-24).
    """
    const = qam_constellation(int(bits_per_symbol)).to(torch.complex64)
    inv = 1.0 / (torch.abs(const) ** 2)
    mean_db = 10.0 * math.log10(float(inv.mean()))
    worst_db = 10.0 * math.log10(float(inv.max()))
    return mean_db, worst_db


def power_delay_profile(s_pars):
    """Element-averaged PDP of a CFR frame: `|ifft(H)|^2` averaged over elements.

    Returns a 1-D float tensor of length `n_freqs`, bin `n` at delay `n / (N * df)`.
    """
    h = torch.fft.ifft(torch.as_tensor(s_pars), dim=-1)
    return torch.mean(torch.abs(h.reshape(-1, h.shape[-1])) ** 2, dim=0)


def cp_guard_ok(s_pars, cp_len: int, *, energy_fraction: float = 0.999):
    """`(ok, bin_at_fraction)` -- does `cp_len` cover the channel's delay SPREAD?

    The premise `apply_ofdm_channel` rests on is that the cyclic prefix exceeds the
    channel's excess delay, which is what makes a per-subcarrier multiply the exact
    received grid. The criterion is the delay at which the cumulative PDP energy
    reaches `energy_fraction` -- an ENERGY measure, deliberately, and not the delay of
    some named path family: those are different quantities and only the energy one may
    size a CP.

    The 99.9% criterion is stable across the shipped munich Ka frames (bins 2270 /
    2256 / 2214 on frames 0 / 10 / 29, measured 2026-09-24); a 99.99% criterion is not
    (2915 / 3190 / 2932), which is why it is not the default.

    NOTE THE SCOPE: in THIS model the CP has no numerical effect whatsoever, because
    the channel is applied as a per-subcarrier multiply and nothing ever convolves in
    time. The guard checks the premise the multiply is justified BY; it cannot detect
    ISI, because the model cannot produce any. It costs data rate and buys honesty.
    """
    pdp = power_delay_profile(s_pars)
    cum = torch.cumsum(pdp, dim=0)
    total = cum[-1]
    if not float(total) > 0:
        return True, 0
    idx = int(torch.searchsorted(cum, total * float(energy_fraction)).item())
    return idx <= int(cp_len), idx


# ================================================================================
# The transmitted frame
# ================================================================================
@dataclass
class OFDMFrame:
    """The transmitted OFDM grid `X [M, N]` and everything derived from it.

    TWO MODEMS, not one, and this is forced rather than chosen: `OFDMModem` fixes its
    pilot/data split in `__init__`, so one instance cannot give symbol 0 a different
    layout from symbols 1..M-1. The preamble modem is all-pilot (`pilot_spacing=1`,
    `n_data == 0`) and the data modem carries the comb. `OFDMModem` is NOT extended --
    its round-trip tests are the only thing exercising `modulate`/`demodulate` in the
    repo, and this design depends on them staying honest.

    The preamble does TRIPLE duty and that is why it is not optional: it is the sensing
    reference (`X == 1`, so `Y/X == Y`), the per-element channel estimate the
    beamformer's weights come from (`Y[:, 0, 0, :]` IS `H` at full grid density, with no
    interpolation and no extra estimator), and the FMCW bit-parity point (O2).

    `pilot_spacing` is THE resource-split knob, and it is the one that makes "hybrid"
    mean something on a screen: with `sensing_source="pilots_only"` a comb of spacing P
    is a deterministic sensing waveform interleaved in frequency with the data
    subcarriers, so turning the knob moves BOTH products in opposite directions -- the
    image's unambiguous range window shrinks by P and the scene's multipath visibly
    aliases, while the data rate rises. `sensing_window_m` computes the first half of
    that trade and `data_rate_bps` the second.
    """

    n_subcarriers: int
    subcarrier_spacing_hz: float
    n_symbols: int = 4
    cp_len: int = 0
    bits_per_symbol: int = 2
    pilot_spacing: int = 8
    seed: int = 0
    sensing_source: str = "preamble"
    _preamble: OFDMModem = field(default=None, repr=False)
    _data: OFDMModem = field(default=None, repr=False)

    def __post_init__(self):
        if self.sensing_source not in SENSING_SOURCES:
            raise ValueError(
                f"unknown sensing_source {self.sensing_source!r}; expected one of "
                f"{SENSING_SOURCES}")
        if self.n_symbols < 1:
            raise ValueError(f"n_symbols must be >= 1, got {self.n_symbols}")
        if self.pilot_spacing < 1:
            raise ValueError(
                f"pilot_spacing must be >= 1 (1 = every subcarrier a pilot, the "
                f"all-pilot parity point), got {self.pilot_spacing}")
        n = int(self.n_subcarriers)
        self._preamble = OFDMModem(fft_size=n, cp_len=self.cp_len, n_active=n,
                                   pilot_spacing=1,
                                   bits_per_symbol=self.bits_per_symbol)
        self._data = OFDMModem(fft_size=n, cp_len=self.cp_len, n_active=n,
                               pilot_spacing=int(self.pilot_spacing),
                               bits_per_symbol=self.bits_per_symbol)
        self._build()

    # ------------------------------------------------------------------ construction
    def _build(self):
        n_data_symbols = self.n_symbols - 1
        pre_time, pre_freq = self._preamble.modulate(
            torch.zeros(0, dtype=torch.int64, device=device), 1)
        if n_data_symbols:
            n_bits = self._data.data_bits_per_symbol_block * n_data_symbols
            bits = random_bits(n_bits, seed=self.seed)
            dat_time, dat_freq = self._data.modulate(bits, n_data_symbols)
            grid = torch.cat([pre_freq, dat_freq], dim=0)
            time = torch.cat([pre_time, dat_time], dim=0)
        else:
            bits = torch.zeros(0, dtype=torch.int64, device=device)
            grid, time = pre_freq, pre_time
        self.tx_bits = bits
        self.tx_grid = grid.to(torch.complex64)                  # [M, N]
        #: `[n_tx=1, M, N + cp_len]` -- the `[n_tx, n_chirp, n_t]` layout `TxPABlock`
        #: and `ModulateBlock` consume. `modulate` returns 2-D; the unsqueeze is not
        #: cosmetic, those blocks index dim 0 as the TX axis.
        self.tx_wave = time.unsqueeze(0).to(torch.complex64)
        self.tx_data = (self._data.extract_data(grid[1:])
                        if n_data_symbols else grid[:0, :0])

    # ------------------------------------------------------------------ the reference
    def reference_grid(self):
        """`X_ref [M, N]`: what `SymbolDivisionBlock` divides by, per `sensing_source`.

        A zero entry means "this (symbol, subcarrier) carries no sensing information",
        and the block zeroes the quotient there rather than dividing -- which is the
        resource split, expressed as a mask on one grid rather than as a second chain.
        """
        X = self.tx_grid
        if self.sensing_source == "all_symbols":
            return X
        if self.sensing_source == "preamble":
            ref = torch.zeros_like(X)
            ref[0] = X[0]
            return ref
        # pilots_only: every symbol's comb subcarriers, data subcarriers masked out.
        ref = torch.zeros_like(X)
        ref[0] = X[0]                                  # the preamble is all pilots
        if X.shape[0] > 1:
            ref[1:, self._data.pilot_idx] = X[1:, self._data.pilot_idx]
        return ref

    # ------------------------------------------------------------------ the trade
    def sensing_window_m(self, convention="bistatic_path"):
        """Unambiguous window of the SENSING product, in metres.

        `pilots_only` samples the channel every `P`-th subcarrier, so its effective
        grid spacing is `P * df` and its unambiguous window is `c / (P * df)` -- the
        knob's first half. Every other sensing source uses the full grid.
        """
        from e2e.chain.receive import _CONVENTION_SCALE

        p = self.pilot_spacing if self.sensing_source == "pilots_only" else 1
        return _CONVENTION_SCALE[convention] / (p * self.subcarrier_spacing_hz)

    def symbol_duration_s(self):
        return (1.0 + self.cp_len / self.n_subcarriers) / self.subcarrier_spacing_hz

    def frame_duration_s(self):
        return self.n_symbols * self.symbol_duration_s()

    def data_rate_bps(self):
        """RAW UNCODED BURST rate, and all three words belong on any card that quotes
        it. The CP overhead is already charged (it is inside `symbol_duration_s`).

        It is a BURST rate: the system transmits one `frame_duration_s` burst per scene
        frame and the stored corpus defines no inter-frame interval, so an AVERAGE rate
        is undefined from the corpus. A card that wants one must state the frame rate it
        assumed -- see `average_rate_bps`.
        """
        n_data_symbols = max(self.n_symbols - 1, 0)
        bits = n_data_symbols * self._data.n_data * self.bits_per_symbol
        return bits / self.frame_duration_s()

    def average_rate_bps(self, frame_rate_hz):
        """The duty-cycled rate, which only exists once someone names a frame rate.
        Separated from `data_rate_bps` so the assumption cannot travel silently."""
        return self.data_rate_bps() * self.frame_duration_s() * float(frame_rate_hz)

    # ------------------------------------------------------------------ description
    def record_metadata(self):
        return {
            "cube_axes": dict(frames.CUBE_AXES_OFDM),
            "n_fast": int(self.n_subcarriers),
            "n_slow": int(self.n_symbols),
            "delta_f_hz": float(self.subcarrier_spacing_hz),
            "cp_len": int(self.cp_len),
            "bits_per_symbol": int(self.bits_per_symbol),
            "pilot_spacing": int(self.pilot_spacing),
            "sensing_source": self.sensing_source,
            "constant_envelope": False,
            "data_rate_bps": self.data_rate_bps(),
            "frame_duration_s": self.frame_duration_s(),
            "sample_rate_hz": self.n_subcarriers * self.subcarrier_spacing_hz,
        }


def frame_from_freq_plan(freq_plan, **kwargs) -> OFDMFrame:
    """The `OFDMFrame` whose subcarriers ARE the stored channel's frequency grid.

    Both numbers come from the source: `n_subcarriers = num_freqs` and the spacing is
    the plan's own endpoint-inclusive step. That equality is the premise of the whole
    design -- it is what makes `Y = H * X` an elementwise multiply with no
    interpolation anywhere -- so it is established by construction here rather than
    asserted downstream.
    """
    return OFDMFrame(n_subcarriers=int(freq_plan["num_freqs"]),
                     subcarrier_spacing_hz=subcarrier_spacing_hz(freq_plan),
                     **kwargs)


# ================================================================================
# The channel apply
# ================================================================================
def apply_ofdm_channel(s_pars, tx_grid):
    """`Y[r, t, m, k] = H[r, t, 0, k] * X[m, k]` -- the received grid.

    `s_pars [R, T, 1, N]` in, `[R, T, M, N]` out: THE EXISTING 4-D LAYOUT with the
    chirp axis reinterpreted as the SYMBOL axis, which is why `InterconnectBlock` and
    the front end need no change at all to carry an OFDM frame.

    Multiplying per subcarrier is exactly the CP-OFDM received grid PROVIDED the cyclic
    prefix exceeds the channel's excess delay (`cp_guard_ok`). Note what that means
    here: the multiply IS a cyclic convolution by construction, so the premise is
    built in rather than tested, and no CP length can create or prevent ISI in this
    model.

    `ModulateBlock`'s spectrum path CANNOT do this job -- verified 2026-09-24:
    `torch.fft.fft(x, n=k)` with `k < len(x)` truncates to the FIRST `k` samples, so on
    a CP-prefixed `tx_wave` of length `N + cp_len` it would transform
    `[CP | the first N - cp_len samples]` and silently drop the rest. Shape-correct,
    wrong, and it survives a smoke test.
    """
    s_pars = torch.as_tensor(s_pars)
    if s_pars.dim() != 4:
        raise frames.FrameContractError(
            f"apply_ofdm_channel expects s_pars [R, T, 1, N], got "
            f"{tuple(s_pars.shape)}")
    if s_pars.shape[2] != 1:
        raise frames.FrameContractError(
            f"apply_ofdm_channel expects ONE channel snapshot per frame (s_pars "
            f"[R, T, 1, N]); got {s_pars.shape[2]} on the chirp axis. The OFDM frame's "
            f"M symbols all see the same H -- the stored channel is time-invariant "
            f"within a frame -- so the symbol axis is created HERE, from one snapshot.")
    X = torch.as_tensor(tx_grid, dtype=torch.complex64, device=s_pars.device)
    if X.shape[-1] != s_pars.shape[-1]:
        raise frames.FrameContractError(
            f"the transmitted grid has {X.shape[-1]} subcarriers but the channel is "
            f"sampled at {s_pars.shape[-1]} points. The OFDM frame must be built from "
            f"the SOURCE's freq_plan (`frame_from_freq_plan`) so the two grids are the "
            f"same grid -- resampling one onto the other is not offered.")
    return (s_pars[:, :, :1, :] * X[None, None, :, :]).to(torch.complex64)


class OFDMChannelBlock:
    """`ChannelApply` for the OFDM/JSAC classes: `s_pars` (H) -> `s_pars` (Y = H*X).

    The frequency-domain counterpart of `ModulateBlock`'s spectrum multiply, for a
    waveform whose transmitted signal is a GRID rather than a sweep. It stays in
    `DOMAIN_CFR` because what it emits is still a frequency response times a spectrum;
    the domain crossing is the mixing block's job.

    `tx_pa` optionally re-derives the transmitted grid from the PA-distorted time
    record (`modem.demodulate`: strip CP, FFT, fftshift) so the amplifier's
    nonlinearity reaches both products. With no PA the grid is used directly, and that
    fast path is what makes the parity oracle BIT-exact rather than float-close.
    """

    frame_capabilities = FrameCapabilities(
        domain=frames.DOMAIN_CFR, accepts_mimo=True, chirps=frames.CHIRP_NATIVE)

    def __init__(self, ofdm_frame: OFDMFrame, *, use_pa_distorted_grid: bool = False):
        self.frame = ofdm_frame
        self.use_pa_distorted_grid = bool(use_pa_distorted_grid)

    def transmitted_grid(self, state):
        if not self.use_pa_distorted_grid or "tx_wave" not in state:
            return self.frame.tx_grid
        # Strip CP, FFT, fftshift -- `OFDMModem.demodulate` does exactly this, and the
        # fftshift is load-bearing: without it the grid comes back in natural DFT order
        # and multiplies the channel's negative half against its positive half.
        wave = state["tx_wave"]
        return self.frame._data.demodulate(wave.reshape(-1, wave.shape[-1]))

    def apply(self, state):
        X = self.transmitted_grid(state)
        return {"s_pars": apply_ofdm_channel(state["s_pars"], X),
                "tx_grid": X,
                "cube_axes": dict(frames.CUBE_AXES_OFDM)}


# ================================================================================
# The mixing block
# ================================================================================
class SymbolDivisionBlock:
    """THE mixing block for OFDM-ISAC: the received grid `Y` -> the sampled record
    `adc`, by dividing out the transmitted symbols.

    It sits exactly where `DechirpBlock` sits (contract section 1.2 block 6), before
    the floor and the ADC, and it SHARES ITS TAIL:

        Z = Y / X_ref                 # cancels the data EXACTLY -- no matched filter,
        Z = conj(Z)                   #   no code sidelobes; the sensing estimate is
        Z = flip(Z, dims=(0, 1))      #   independent of the transmitted bits
        adc = mimo_combine(cfg, Z)

    The last three lines are `beat_from_cfr` + `mimo_combine`, IMPORTED, not copied.
    That is the whole reason O2 is bit-exact: with `X_ref == 1` the division is the
    identity and the two waveforms' tails are literally the same function.

    `n_tx > 1` is refused: the TDM/DDMA per-chirp TX factor `mimo_combine` applies is
    meaningless on a symbol axis, and the failure would otherwise be a plausible
    picture rather than an error.

    Masked reference entries (the resource split's data subcarriers, and every
    non-preamble symbol under `sensing_source="preamble"`) yield an exactly ZERO
    quotient rather than a division by zero. Zeros in a range transform are a taper,
    not an artefact of arithmetic: a comb of spacing `P` sampled this way has its
    unambiguous window divided by `P`, which is the knob's visible half, and
    `OFDMFrame.sensing_window_m` is the number to put beside the picture.
    """

    frame_capabilities = FrameCapabilities(
        domain=frames.DOMAIN_CFR, emits_domain=frames.DOMAIN_RX_TIME,
        accepts_mimo=True, chirps=frames.CHIRP_NATIVE)

    class _SingleTxCfg:
        """The stand-in cfg for a chain that has no `RadarConfig`.

        `mimo_combine` reads `cfg.mimo` and `cfg.n_tx` and nothing else, so a
        single-TX OFDM chain needs no radar config at all -- and without this default
        `cfg=None` failed deep inside `mimo_combine` with
        `'NoneType' object has no attribute 'mimo'`, naming nothing. `Simulation`
        carries the same stand-in for `DechirpBlock`, for the same reason.
        """
        mimo = "single"
        n_tx = 1

    def __init__(self, cfg=None, ofdm_frame: OFDMFrame = None, *, x_ref=None):
        cfg = cfg if cfg is not None else self._SingleTxCfg()
        if ofdm_frame is None and x_ref is None:
            raise ValueError(
                "SymbolDivisionBlock needs the transmitted grid it divides by: pass "
                "the OFDMFrame that built it, or an explicit x_ref. Re-guessing X at "
                "the receiver is not the same receiver.")
        self.cfg = cfg
        self.frame = ofdm_frame
        self._x_ref = x_ref

    def reference_grid(self, state):
        if self._x_ref is not None:
            return torch.as_tensor(self._x_ref, dtype=torch.complex64)
        return self.frame.reference_grid()

    def apply(self, state):
        Y = state["s_pars"]
        n_tx = Y.shape[1]
        if n_tx > 1:
            raise frames.FrameContractError(
                f"SymbolDivisionBlock refuses a {n_tx}-TX frame: `mimo_combine`'s "
                f"per-chirp TX factor (TDM selection / DDMA code) is defined on a "
                f"CHIRP axis and is meaningless on a symbol axis. Combine the "
                f"transmit axis before the mixing block, or use one TX.")
        X = self.reference_grid(state).to(Y.device)
        if X.shape[0] != Y.shape[2] or X.shape[1] != Y.shape[3]:
            raise frames.FrameContractError(
                f"the sensing reference grid is {tuple(X.shape)} but the received "
                f"frame carries {Y.shape[2]} symbols x {Y.shape[3]} subcarriers")
        # Divide only where the reference carries sensing information; elsewhere the
        # quotient is exactly zero (see the class docstring -- this is the split, not
        # a guard against NaN). Written as a divide-by-a-safe-denominator then mask,
        # rather than as boolean indexing, because the indexing form materialises a
        # full expanded copy of both operands: at the munich Ka plan one frame is
        # [1024, 1, M, 5000] complex64 = 164 MB per symbol-set, and this runs inside a
        # web callback.
        live = (X != 0)
        safe = torch.where(live, X, torch.ones_like(X))
        Z = (Y / safe[None, None]) * live[None, None].to(Y.dtype)
        adc = mimo_combine(self.cfg, beat_from_cfr(Z))
        return {"adc": adc.to(torch.complex64),
                "signal_domain": frames.DOMAIN_RX_TIME,
                "cube_axes": dict(frames.CUBE_AXES_OFDM),
                "sensing_reference": self.frame.sensing_source if self.frame else None}


# ================================================================================
# The comms head, as a consumer of the chain
# ================================================================================
class OFDMReceiveBlock:
    """The comms head reading the CHAIN's received grid -- not a channel of its own.

    `ModemBlock` synthesises its own channel, its own noise and its own transmitted
    frame; on the one chain all three already exist upstream, and using its versions
    would mean the demo counts noise twice and demaps against a grid the mixer never
    saw. This block is the grid-consuming path: `s_pars` in, the same output keys
    `BERBlock` already reads out.

    WHERE THE PER-ELEMENT CHANNEL COMES FROM. Once `s_pars` carries `Y = H*X` rather
    than `H`, the beamformers cannot be handed `s_pars` -- they take a CHANNEL. The
    answer is free and already in the design: the all-pilot preamble has `X == 1`, so
    `Y[:, 0, 0, :]` IS `H` per element at full grid density, with no interpolation and
    no extra estimator. `element_channels` is the WRONG tool here; it interpolates onto
    a foreign subcarrier grid and ours is already native.

    `subspace` combining is NOT offered, and the reason is worth stating: it would read
    `state['U']` from the tracker, which sits DOWNSTREAM of the mixing block, so on this
    path the weights would come from the previous frame's basis or from nothing at all.

    NOISE: none is injected here, ever -- this block has no noise source and no
    `add_noise` knob. The chain's floor is the only one, which is contract section 1.4.

    THE SCOPE THAT MAKES THAT TRUE RATHER THAN EMPTY: for these waveform classes the
    front end belongs in the FREQUENCY domain, as `CircuitStage(RFFEBlock)` on the
    received grid -- and that is not a workaround, it is the physics. `ifft(s_pars)` of
    a received OFDM grid IS the received time-domain symbol, the signal a real
    amplifier sees, which is exactly what makes `RFFEBlock`'s round trip correct here
    and wrong on an FMCW CFR (F96). Placed there the front end runs BEFORE this block,
    stamps `noise_injected_by`, and its floor reaches both the comms products and the
    sensing cube -- one knob moving both, on one frame.

    A chain that instead puts a beat-placement `FrontEndBlock` after a mixing block
    gives this head a NOISELESS grid: `comm_noise_source` then reads `"none"` and the
    measured SNR is whatever float precision allows. That is reported, not hidden, and
    it is the signal that the front end is on the wrong side of the tap for this
    waveform.

    THE EQUALISER'S SNR is MEASURED, not assumed (`channel.estimate_snr_db`): once the
    front end is the only noise source, nothing in the pipeline knows the
    post-combining SNR, and `ModemBlock`'s constructor SNR would be a number typed into
    a preset masquerading as a measurement. `equalizer="zf"` needs no SNR at all and is
    the fallback when there are too few pilot observations to estimate one; unbiased
    MMSE and ZF make the same hard decision, so uncoded BER is identical either way and
    only EVM differs.
    """

    frame_capabilities = FrameCapabilities(
        domain=frames.DOMAIN_CFR, accepts_mimo=True, chirps=frames.CHIRP_NATIVE)

    COMBINING = ("element0", "egc", "mrc")

    def __init__(self, ofdm_frame: OFDMFrame, *, combining="mrc", equalizer="mmse",
                 estimator="ls"):
        if combining not in self.COMBINING:
            raise ValueError(
                f"unknown combining {combining!r}; expected one of {self.COMBINING}. "
                f"'subspace' is deliberately absent -- see this class's docstring.")
        self.frame = ofdm_frame
        self.combining = combining
        self.equalizer = equalizer
        self.estimator = estimator

    # ------------------------------------------------------------------ the channel
    def element_channel(self, Y):
        """`H [R, N]` from the all-pilot preamble symbol: `Y[:, 0, 0, :]` with `X == 1`.

        A SINGLE NOISY SNAPSHOT per subcarrier across R elements -- so MRC built on it
        loses array gain relative to ideal weights. `measure_array_gain_db` reports both
        so a card quotes the loss rather than assuming it away.
        """
        return Y[:, 0, 0, :]

    def weights(self, H):
        from . import beamforming as bf

        if self.combining == "egc":
            return bf.egc_weights(H)
        return bf.mrc_weights(H)

    def apply(self, state):
        from . import beamforming as bf
        from . import channel as ch
        from .ofdm import qam_demod

        frame = self.frame
        Y = state["s_pars"]
        if Y.shape[2] < 2:
            raise frames.FrameContractError(
                "OFDMReceiveBlock needs at least one DATA symbol beside the preamble "
                f"(got {Y.shape[2]} symbols). Build the frame with n_symbols >= 2.")
        H = self.element_channel(Y)                       # [R, N]
        if self.combining == "element0":
            rx_grid = Y[0, 0, 1:, :]                      # the historical SISO tap
            H_eff = H[0]
            extra = {}
        else:
            w = self.weights(H)
            # `combine` wants the ELEMENT axis second-to-last, with the symbol axis
            # leading; the chain carries elements first. Permute rather than reshape --
            # the two axes are not interchangeable and a reshape here would combine
            # across symbols.
            rx = Y[:, 0, 1:, :].permute(1, 0, 2).contiguous()   # [M-1, R, N]
            rx_grid = bf.combine(rx, w)                   # [M-1, N]
            H_eff = bf.combine(H, w)
            extra = {"comm_array_gain_db": measure_array_gain_db(H, w)}

        modem = frame._data
        rx_pilots = modem.extract_pilots(rx_grid)
        tx_pilots = modem.extract_pilots(frame.tx_grid[1:])
        snr_db = ch.estimate_snr_db(rx_pilots, tx_pilots)
        if self.estimator == "mmse" and snr_db is not None:
            H_est = ch.mmse_estimate(rx_pilots, tx_pilots, modem.pilot_idx,
                                     modem.fft_size, snr_db)
        else:
            H_est = ch.ls_estimate(rx_pilots, tx_pilots, modem.pilot_idx,
                                   modem.fft_size)
        if self.equalizer == "zf" or snr_db is None:
            eq = ch.zf_equalize(rx_grid, H_est)
        else:
            eq = ch.mmse_equalize(rx_grid, H_est, snr_db)
        data_eq = modem.extract_data(eq)
        rx_bits = qam_demod(data_eq.reshape(-1), modem.bits_per_symbol, modem.const)

        out = {
            "comm_tx_bits": frame.tx_bits,
            "comm_rx_bits": rx_bits,
            "comm_data_eq": data_eq,
            "comm_tx_data": frame.tx_data,
            "comm_H_est": H_est,
            "comm_H_true": H_eff,
            "comm_bits_per_symbol": modem.bits_per_symbol,
            "comm_const": modem.const,
            # MEASURED, not configured -- the number the equaliser actually used.
            "comm_snr_db": (float("nan") if snr_db is None else float(snr_db)),
            "comm_noise_source": state.get("noise_injected_by", "none"),
        }
        out.update(extra)
        return out


def measure_array_gain_db(H, w):
    """Post-combining power gain, in dB, of weights `w` on the per-element channel `H`.

    Reported so a card quotes the gain the demo REALISED (weights from one noisy
    preamble snapshot) rather than the ideal-weights figure. Compare against
    `measure_array_gain_db(H, mrc_weights(H_clean))` to bound the estimation loss.
    """
    from . import beamforming as bf

    elem = float(torch.mean(torch.abs(H) ** 2))
    comb = float(torch.mean(torch.abs(bf.combine(H, w)) ** 2))
    return 10.0 * math.log10(comb / elem) if elem > 0 else float("nan")


# ================================================================================
# The PAPR question the review left open -- answered by measurement, never by intuition
# ================================================================================
def measure_lna_input_papr(s_pars):
    """Mean per-element peak-to-average power ratio (dB) of `ifft(s_pars)` -- i.e. of
    the signal the RX front end's clamp actually sees under the v1.0 impulse placement.

    THIS IS THE MEASUREMENT THE CARD NEEDS, and it is not the transmitted waveform's
    PAPR. `RFFEBlock.apply_circuit` normalises by the MEAN magnitude of `ifft(s_pars)`,
    so whether the clamp engages is governed exactly by the peak-to-mean of THAT
    tensor. On the munich Ka trace the line-of-sight tap carries ~93% of the PDP energy,
    so `ifft(H)` is a spike. MEASURED 2026-09-24 on `munich_ka.pkl` frame 0, all 1024
    elements, through the frame this module builds (5000 subcarriers, `pilot_spacing=8`,
    QPSK, `M=4`):

    | what reaches the LNA | mean PAPR of `ifft(s_pars)` |
    |---|---|
    | FMCW as shipped (`s_pars = H`; the waveform/modulate blocks are off by default) | **36.67 dB** |
    | the JSAC frame's ALL-PILOT PREAMBLE symbol | **36.67 dB** |
    | the JSAC frame's DATA symbols | **19.34 dB** |
    | (an earlier note measured FMCW WITH the waveform+modulate blocks enabled at 3.95 dB) |  |

    Read the second row: `X == 1` makes `Y` the bare CFR, so the preamble symbol IS the
    shipped FMCW preset's tensor and has bit-identically its PAPR. **The peakiest symbol
    in a JSAC frame is the one that is the FMCW arm.**

    The data symbols measure 19.34 dB here and not the ~9.6 dB an earlier note recorded,
    and the difference is structural rather than a discrepancy: that measurement used a
    fully data-modulated grid, while this frame carries a constant-valued pilot every
    8th subcarrier. A periodic comb of equal values is an impulse TRAIN in time, so the
    comb adds ~10 dB of peak-to-mean on its own. A frame with a sparser comb or
    randomised pilot values measures differently -- which is the point of measuring per
    configuration rather than quoting a figure.

    So "the OFDM waveform is what makes the front end clip" is FALSE against the shipped
    FMCW preset -- it runs the other way, by 17.3 dB on the data symbols and by 0.0 dB
    on the preamble. Any A/B that turns a drive knob must name which FMCW configuration
    it is compared against and quote both numbers from this function at the preset's own
    operating point.
    """
    h = torch.fft.ifft(torch.as_tensor(s_pars), dim=-1)
    flat = h.reshape(-1, h.shape[-1])
    p = torch.abs(flat) ** 2
    papr = p.max(dim=-1).values / p.mean(dim=-1).clamp_min(1e-30)
    return float(10.0 * torch.log10(papr).mean())


def measure_papr_db(tx_wave, *, oversample=1):
    """Mean per-symbol PAPR (dB) of a transmitted time record, at a STATED oversampling
    factor -- both halves are card-bound and neither is optional.

    Measured 2026-09-24 on 400 random QPSK symbols, 5000-point grid with NO pilot comb:
    mean 9.55 dB at Nyquist, 10.03 dB at 4x oversampling (the standard convention),
    per-symbol max 11.75 / 12.21 dB. An earlier draft's "99.9th percentile 12.41 dB" was
    a percentile of pooled instantaneous sample power -- a different quantity from PAPR,
    above the observed per-symbol maximum -- and is WITHDRAWN. Quote the mean and the
    oversampling factor.

    ON THE FRAME THIS MODULE ACTUALLY BUILDS the figure is higher, and the difference is
    structural: `pilot_spacing=8` puts a CONSTANT value on every 8th subcarrier, and a
    periodic comb of equal values is an impulse train in time. Measured on the munich Ka
    plan: data symbols 19.62 dB at Nyquist (19.62 dB at 4x), and the all-pilot preamble
    **36.99 dB = 10*log10(5000) exactly**, because `X == 1` on every subcarrier IS a
    time-domain impulse. Quote the number for the frame on the screen, not the textbook
    one.
    """
    x = torch.as_tensor(tx_wave)
    x = x.reshape(-1, x.shape[-1])
    if oversample > 1:
        n = x.shape[-1]
        X = torch.fft.fft(x, dim=-1)
        Xp = torch.zeros(x.shape[0], n * oversample, dtype=X.dtype, device=X.device)
        half = n // 2
        Xp[:, :half] = X[:, :half]
        Xp[:, -half:] = X[:, -half:]
        x = torch.fft.ifft(Xp, dim=-1) * oversample
    p = torch.abs(x) ** 2
    papr = p.max(dim=-1).values / p.mean(dim=-1).clamp_min(1e-30)
    return float(10.0 * torch.log10(papr).mean())


# ================================================================================
# The three classes as ONE spine with one parameter changed
# ================================================================================
@dataclass
class ChainSpec:
    """What a waveform class contributes to the ONE spine, as data rather than as a
    branch in a builder.

    `mixing_block` is the stage that turns the received frame into the sampled record
    -- `DechirpBlock` for fmcw, `SymbolDivisionBlock` for jsac, `None` for ofdm (a
    comms receiver never forms a cube). `sensing` says whether the range transform and
    the sensing products are built at all; `comms` says whether the modem head is.

    Shard 3 reads this instead of writing a second `if kind == ...` ladder in the
    webapp: the registry's dropdown, the diagram's one branch point and the preset's
    product list all come from the same three rows, so "the dropdown IS the tradeoff"
    is enforced by construction rather than by care.
    """

    kind: str
    mixing_block: object
    sensing: bool
    comms: bool
    channel_block: object = None
    receive_block: object = None
    frame: Optional[OFDMFrame] = None
    #: Free-form, card-bound: sample rate, data rate, frame duration, the resource
    #: split. Every entry is COMPUTED from the frame, never typed into a preset.
    notes: dict = field(default_factory=dict)

    def tributary_stages(self):
        """The stages that go BEFORE the interconnect/front end: the waveform source
        and the channel apply. Empty for `fmcw` on a stored CFR, where the dechirp
        identity IS the modulation (contract section 1.2 row 3)."""
        return [s for s in (self.channel_block,) if s is not None]


def waveform_chain_spec(kind, cfg=None, *, freq_plan=None, ofdm_frame=None,
                        combining="mrc", equalizer="mmse", **frame_kwargs) -> ChainSpec:
    """THE table: `kind` -> (mixing block, sensing?, comms?), built for `cfg`.

        fmcw   DechirpBlock           sensing only
        ofdm   None                   comms only
        jsac   SymbolDivisionBlock    both, from one frame

    Everything else on the spine -- interconnect, front end, floor, impairments, ADC,
    the one range transform, the compressor, the products -- is IDENTICAL across the
    three and is not this function's business. That is the point: the waveform is one
    branch point inside one block, not a second pipeline.
    """
    from e2e.chain.dechirp import DechirpBlock

    kind = str(kind)
    if kind == "fmcw":
        return ChainSpec(kind="fmcw", mixing_block=DechirpBlock(cfg), sensing=True,
                         comms=False,
                         notes={"sample_rate_hz": getattr(cfg, "fs_hz", None)})
    if kind not in ("ofdm", "jsac"):
        raise ValueError(
            f"waveform_chain_spec: unknown kind {kind!r}; expected 'fmcw', 'ofdm' or "
            f"'jsac'")

    frame = ofdm_frame
    if frame is None:
        if not freq_plan:
            raise ValueError(
                f"the {kind!r} chain needs the SOURCE's freq_plan to place its "
                f"subcarriers on the stored channel's own grid -- pass freq_plan= or "
                f"a prebuilt ofdm_frame=.")
        frame = frame_from_freq_plan(freq_plan, **frame_kwargs)

    notes = dict(frame.record_metadata())
    notes["sensing_window_m"] = frame.sensing_window_m()
    notes["qam_noise_rise_db"] = qam_division_noise_rise_db(frame.bits_per_symbol)
    return ChainSpec(
        kind=kind,
        mixing_block=(SymbolDivisionBlock(cfg, frame) if kind == "jsac" else None),
        sensing=(kind == "jsac"),
        comms=True,
        channel_block=OFDMChannelBlock(frame),
        receive_block=OFDMReceiveBlock(frame, combining=combining,
                                       equalizer=equalizer),
        frame=frame,
        notes=notes,
    )
