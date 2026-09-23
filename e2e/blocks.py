from pathlib import Path

import warnings

import numpy as np
import torch

from e2e.environment.sionna_iterator import SionnaEtoileIterator, SionnaMunichIterator
from e2e.subspace.algorithms import Oja, gen_A_ada, orth
from e2e.subspace.subspace_utils import subspace_dist_frob
from e2e.afe.afe_utils import quantizer_fp
from e2e.circuit.rffe_model import get_RX_config, circuit_model_batch
from e2e import frames
from e2e.frames import FrameCapabilities


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Frame-contract shorthands used by the per-block `frame_capabilities` declarations
# below. Every block/stage that consumes 's_pars' declares one; Simulation validates the
# incoming frame against it before calling the component (see e2e/frames.py).
# ELEMENTWISE: acts per element/frequency, so extra TX/chirp axes just ride along.
_ELEMENTWISE = FrameCapabilities(accepts_mimo=True, chirps=frames.CHIRP_NATIVE)
# PER_CHIRP: single-chirp core, mapped over the chirp axis (results stacked, leading axis).
_PER_CHIRP = FrameCapabilities(accepts_mimo=False, chirps=frames.CHIRP_BROADCAST)
# SINGLE_CHIRP: the historical contract -- no MIMO, one chirp.
_SINGLE_CHIRP = FrameCapabilities(accepts_mimo=False, chirps=frames.CHIRP_SINGLE)


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
    # The circuit model is element-wise in (rx, tx, chirp) -- circuit_model_batch
    # flattens those three axes into its batch dim -- so MIMO and multi-chirp frames
    # pass through natively, one independent noise realization per trace.
    frame_capabilities = _ELEMENTWISE

    #: Column indices into `get_RX_config`'s `[nRx, 7]` table. The circuit model reads
    #: the table positionally (`RX_config[0]` is `Ibias_LNA`, ... `RX_config[6]` is the
    #: IF bandwidth), so these names are the only place the layout is spelled out.
    RX_CONFIG_IBIAS_LNA = 0     # A
    RX_CONFIG_IF_BW = 6         # Hz

    def __init__(self, n=None, freq_span_hz=3e9, signal_scaling=1e-5, if_filter=False,
                 physical_scale=False, chirp_dur=None, lna_bias_ma=None, if_bw_mhz=None,
                 seed=None):
        # chirp_dur: legacy kwarg accepted (and ignored) so existing call sites that
        # still pass it don't break.
        # fs (= freq_span_hz) is the complex-baseband buffer's true sample rate (the
        # default 28.5-31.5 GHz plan -> fs = 3e9, dt ~333 ps, record ~1.667 us). It
        # sizes the optional IF filter's boxcar width ONLY; the thermal-noise floor
        # is band-referenced to the receiver's IF bandwidth inside the circuit model
        # (stepped-frequency measurement semantics), not to fs.
        self.rx_config = get_RX_config(n).to(device)
        # The two circuit knobs the conference demo turns (Thrust 1). They override
        # single columns of the per-element config table -- the same override the
        # measurements in notes/DEMO_DEFENSE.md were taken with, previously done by
        # monkeypatching `get_RX_config` from a scratch script. None keeps the table's
        # own defaults (8 mA, 15 MHz), so every existing caller is bit-identical.
        # Both are validated because the circuit model divides by them: a zero bias
        # is 0/0 in the gm cascade, and a zero bandwidth zeroes the noise variance.
        if lna_bias_ma is not None:
            if not lna_bias_ma > 0:
                raise ValueError(f"lna_bias_ma must be > 0, got {lna_bias_ma!r}")
            self.rx_config[:, self.RX_CONFIG_IBIAS_LNA] = float(lna_bias_ma) * 1e-3
        if if_bw_mhz is not None:
            if not if_bw_mhz > 0:
                raise ValueError(f"if_bw_mhz must be > 0, got {if_bw_mhz!r}")
            self.rx_config[:, self.RX_CONFIG_IF_BW] = float(if_bw_mhz) * 1e6
        self.fs = freq_span_hz
        self.signal_scaling = signal_scaling
        self.if_filter = if_filter
        self.physical_scale = physical_scale
        self.n = n
        # None (default): the thermal-noise draw in circuit_model_bb_approx comes from
        # the global torch RNG, unseeded -- bit-for-bit unchanged from before this
        # option existed. An int seeds a per-instance torch.Generator, advanced by
        # frame index (seed + self._frame_idx) each call, in the same style as
        # ImpairmentBlock/ThermalNoiseBlock (e2e/chain/receive.py, link_budget.py):
        # same seed -> bit-identical noise across repeated runs; different seed/frame
        # -> a fresh draw.
        self.seed = seed
        self._frame_idx = 0

    def reset(self):
        """Rewind the per-frame counter (and hence the seed sequence) to frame 0."""
        self._frame_idx = 0

    def apply_circuit(self, s_pars):
        # Chirp/pol dim size is inferred (-1) rather than hardcoded to 2: this makes
        # the reshape a no-op for whatever n_chirp the caller actually passes (1, the
        # single-chirp frames Simulation feeds via CircuitStage; or 2, as some direct
        # callers/tests still exercise) instead of assuming a pol pair. A MIMO frame's
        # TX axis folds into the same batch dim (element (r, t, c) -> [r, 0, t*n_chirp+c])
        # -- correct because the circuit acts independently per trace -- and the final
        # view restores the caller's shape.
        s_pars_shape = s_pars.shape
        s_pars = s_pars.view(self.n, 1, -1, s_pars.shape[-1])
        frame = torch.fft.ifft(s_pars, dim=-1)
        if not self.physical_scale:
            frame = frame * self.signal_scaling / torch.mean(torch.abs(frame))
        # physical_scale=True: skip the normalization above -- the frame is already
        # in volts at the LNA input (the generation layer produces volts via
        # sqrt(N*P_tx*Z0) scaling; see e2e/environment/scenario_runner.py).
        if self.seed is None:
            frame_dist, PRX = circuit_model_batch(self.rx_config, frame, self.fs,
                                                  if_filter=self.if_filter)
        else:
            generator = torch.Generator(device=frame.device)
            generator.manual_seed(int(self.seed) + self._frame_idx)
            self._frame_idx += 1
            frame_dist, PRX = circuit_model_batch(self.rx_config, frame, self.fs,
                                                  if_filter=self.if_filter,
                                                  generator=generator)
        s_pars_dist = torch.fft.fft(frame_dist, dim=-1)
        s_pars_dist = s_pars_dist.view(s_pars_shape)
        return s_pars_dist, PRX


# Packaged interconnect transfer-function data (a Tessera TSV S21(f) sweep; the surrogate
# architecture is public (BSD-3, github.com/HiPerCAS/tessera) but not vendored here, and
# the specific finetuned checkpoint that produced this CSV was not released -- only this
# derived CSV ships). See e2e/data/interconnect/README.md.
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


# Case names that mean "apply no interconnect at all". Two spellings, deliberately:
# 'case3' is the LEGACY one and cannot be dropped -- e2e/main/main_sionna_blocks.py,
# main_comms_head.py, main_subspace_refine.py, main_interconnect.py and
# docs/FIRST_SCENARIO.md all pass it, and its published numbers depend on it. It is
# also a NAME COLLISION with the Tessera/UIC "Case3" transfer function in
# e2e/data/interconnect/, which is a real filter and the opposite of a pass-through
# (see webapp/pipeline_registry.py's note on the same dropdown). 'passthrough' is the
# honest name the UI offers; both must land here, or picking it silently runs the
# boxcar placeholder instead -- which is exactly what happened in 133f97f.
INTERCONNECT_PASSTHROUGH_CASES = frozenset({'passthrough', 'case3'})

# Case names that select the 11-tap boxcar PLACEHOLDER. `None` is the default spelling;
# 'synthetic' is an existing alias meaning "the synthetic placeholder filter" and is
# passed by e2e/main/main_sionna_blocks.py (pinned by tests/test_blocks.py). Note it is
# NOT related to a synthetic *transfer function* loaded via transfer_csv -- the
# placeholder models no particular physical interconnect at all.
INTERCONNECT_BOXCAR_CASES = frozenset({None, 'synthetic'})

INTERCONNECT_CASES = INTERCONNECT_PASSTHROUGH_CASES | INTERCONNECT_BOXCAR_CASES

# `source` values `InterconnectBlock` accepts, orthogonal to `case`/`transfer_csv`: the
# default `None` keeps the boxcar/CSV behaviour above; `'tessera'` evaluates the live
# Tessera TSV surrogate (e2e/interconnect_surrogate/) instead of reading a CSV.
INTERCONNECT_SOURCES = frozenset({None, 'tessera'})

# The five continuous design parameters the Tessera surrogate takes (see VALID_RANGES /
# SHIPPED_TSV_DESIGN in e2e/interconnect_surrogate/tessera.py). Kept local so validating
# them does not require importing that module.
TESSERA_DESIGN_PARAMS = ('radius_um', 'pitch_um', 'height_um', 'liner_um', 'temperature_k')

TESSERA_DEFAULT_ARRANGEMENT = 'ring3x3'


# Geometric parameters that a scale-model factor rescales; `temperature_k` is a material
# property, not a length, and is deliberately excluded (Maxwell's equations are
# invariant under length x s / frequency / s, conductivities excepted -- scaling a
# temperature has no such correspondence). ALL FOUR must scale together, never a
# subset: the scale-invariance oracle F91 (notes/ESTABLISHED_FACTS.md) found the
# surrogate's crosstalk is only preserved under scaling when height moves with the
# lateral geometry (radius/pitch/liner) -- scaling a subset is not a validated regime.
TESSERA_SCALED_PARAMS = ('radius_um', 'pitch_um', 'height_um', 'liner_um')

# Above this, the surrogate's crosstalk-vs-pitch trend is inverted (F89); `scale=None`
# (auto) picks the smallest integer factor that brings the block's frequency axis at or
# near this ceiling.
_TESSERA_SCALE_TARGET_HZ = 20e9


def _resolve_tessera_scale(scale, freq_hint):
    """Resolve `scale=None` into a concrete factor from a frequency hint (a `band_hz`
    pair or a frequency array); an explicit value always wins (validated `> 0`).

    Auto: 1.0 with no hint or a hint already at/under `_TESSERA_SCALE_TARGET_HZ`;
    otherwise the integer factor nearest `max(freq_hint) / _TESSERA_SCALE_TARGET_HZ`
    -- 2 for the pipeline's Ka band (28.5-31.5 GHz -> 14.25-15.75 GHz) and 4 for the ML
    corpora's 75-81 GHz band (-> ~19-20 GHz), both landing in F89's right-signed
    crosstalk regime.
    """
    if scale is not None:
        scale = float(scale)
        if scale <= 0:
            raise ValueError(f"scale must be > 0, got {scale!r}")
        return scale
    if freq_hint is None:
        return 1.0
    max_hz = float(np.max(freq_hint))
    if max_hz <= _TESSERA_SCALE_TARGET_HZ:
        return 1.0
    return float(round(max_hz / _TESSERA_SCALE_TARGET_HZ))


def _default_presented_tessera_params(scale):
    """The PRESENTED (user-facing) knob defaults: the wrapper's shipped geometry,
    divided by `scale` for the four length parameters (temperature untouched)."""
    from e2e.interconnect_surrogate import SHIPPED_TSV_DESIGN

    return {
        name: (value / scale if name in TESSERA_SCALED_PARAMS else value)
        for name, value in SHIPPED_TSV_DESIGN.items()
    }


def _presented_to_model_tessera_params(presented, scale):
    """PRESENTED knob values -> what is actually fed to the surrogate: geometry x scale,
    temperature unchanged (see `TESSERA_SCALED_PARAMS`)."""
    return {
        name: (value * scale if name in TESSERA_SCALED_PARAMS else value)
        for name, value in presented.items()
    }


def _validate_tessera_design(presented, model, arrangement, scale):
    """Raise ``ValueError`` naming the parameter, its PRESENTED and MODEL values, and
    the model's valid range -- or an unknown arrangement.

    Range validation applies to the MODEL geometry (what the surrogate actually sees):
    `presented x scale` must fall inside the training box. Imports
    `e2e.interconnect_surrogate` lazily (repo-local, torch-free -- no third-party
    `tessera`/`torch_geometric` import happens here). No silent extrapolation: an
    out-of-range value is a loud error, not a warning, even though the wrapper itself
    (built for sweeps) only warns.
    """
    from e2e.interconnect_surrogate import ARRANGEMENTS, VALID_RANGES

    unknown = set(presented) - set(TESSERA_DESIGN_PARAMS)
    if unknown:
        raise ValueError(
            f"unknown Tessera parameter(s) {sorted(unknown)}; expected a subset of "
            f"{TESSERA_DESIGN_PARAMS}"
        )
    for name, model_value in model.items():
        lo, hi = VALID_RANGES[name]
        if not (lo <= float(model_value) <= hi):
            presented_value = presented[name]
            raise ValueError(
                f"Tessera parameter {name}: presented value {float(presented_value):g} "
                f"(scale x{scale:g} -> model value {float(model_value):g}) is outside "
                f"the model's recovered training range [{lo:g}, {hi:g}]; no silent "
                f"extrapolation -- pass a presented value inside the range (see "
                f"VALID_RANGES in e2e/interconnect_surrogate/tessera.py)."
            )
    if arrangement not in ARRANGEMENTS:
        raise ValueError(
            f"unknown Tessera arrangement {arrangement!r}; known: {sorted(ARRANGEMENTS)}"
        )


def tessera_s21_for_axis(freqs_hz, params=None, arrangement=TESSERA_DEFAULT_ARRANGEMENT,
                          cache=None, scale=None):
    """Complex S21(f) from the Tessera TSV surrogate on an arbitrary frequency axis.

    Pure function (no torch): `params` overrides a subset of `TESSERA_DESIGN_PARAMS`,
    PRESENTED values, on top of the wrapper's shipped-geometry defaults (divided by
    `scale`). `scale` (see `_resolve_tessera_scale`; `None` auto-derives one from
    `freqs_hz`) runs a geometric scale model when `!= 1.0`: Maxwell's equations are
    invariant under length x s, frequency / s (conductivities excepted), so the
    surrogate is evaluated at `scale x` the presented geometry (ALL FOUR lengths
    together -- F91, see `TESSERA_SCALED_PARAMS`) and `freqs_hz / scale`, and the
    result is applied at the caller's real `freqs_hz`. `scale=1.0` reproduces the
    unscaled behaviour exactly.

    Validated against the recovered training ranges (`ValueError` naming the offending
    parameter's PRESENTED and MODEL value and the range on a miss, or an unknown
    `arrangement`). Goes through `e2e.interconnect_surrogate.SurrogateCache` -- pass one
    explicitly (e.g. a pre-warmed one) or leave `cache=None` for the default on-disk
    cache, so a repeated call at the same parameters costs a cache lookup, not a model
    forward pass. Raises `ImportError` (subclassed as `ModuleNotFoundError`, with the
    `pip install -r requirements-tessera.txt` line) when the optional third-party
    `tessera`/`torch_geometric` dependency is not usable here.

    This is what `InterconnectBlock(source='tessera')` calls internally; exposed so a
    caller (e.g. the webapp) can draw the response without constructing a pipeline block.
    """
    from e2e.interconnect_surrogate import TesseraTSV

    freqs = np.asarray(freqs_hz, dtype=np.float64)
    scale = _resolve_tessera_scale(scale, freqs)
    presented = _default_presented_tessera_params(scale)
    if params:
        presented.update(params)
    model = _presented_to_model_tessera_params(presented, scale)
    _validate_tessera_design(presented, model, arrangement, scale)
    # Ranges are already enforced above (louder than the wrapper's own warn-only
    # default), so silence its duplicate warning.
    tsv = TesseraTSV(cache=cache, warn_out_of_range=False)
    return tsv.s21(freqs / scale, grid=arrangement, **model)


# RF Interconnect Model Block
class InterconnectBlock:
    """Interconnect filtering, applied multiplicatively across the frequency axis.

    Three modes:

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

    - **Live surrogate (`source='tessera'`):** evaluates the public Tessera TSV_PhGNN
      surrogate (`e2e/interconnect_surrogate/`, an OPTIONAL dependency -- see
      `requirements-tessera.txt`) over the frame's frequency grid for `tessera_params`
      (a subset of `TESSERA_DESIGN_PARAMS`, PRESENTED/user-facing values, defaulting to
      the wrapper's shipped geometry) and `tessera_arrangement` (a key of
      `e2e.interconnect_surrogate.ARRANGEMENTS`), then applies the resulting S21(f) the
      same way `transfer_csv` does. `scale` runs a geometric scale model (`None`, the
      default, auto-derives a factor from `band_hz`; see `_resolve_tessera_scale`):
      the surrogate is evaluated at `scale x` ALL FOUR length parameters together
      (never a subset -- F91, notes/ESTABLISHED_FACTS.md, found crosstalk is only
      preserved when height scales with the lateral geometry) and `band_hz / scale`,
      with the result applied at the block's real frequency axis. Out-of-range
      (post-scale, MODEL-space) parameters or an unknown arrangement raise `ValueError`
      at construction time, naming both the PRESENTED and MODEL value (no silent
      extrapolation); if the surrogate itself is not importable, evaluating the
      response raises `ImportError` naming the install line. Repeated evaluation with
      the same parameters/frequency axis costs a `SurrogateCache` lookup, not a model
      forward pass -- see `tessera_s21_for_axis`, the pure function this delegates to.

    `case` names which of those to use, and only the names in `INTERCONNECT_CASES` are
    accepted:

    - `INTERCONNECT_PASSTHROUGH_CASES` -- `'passthrough'`, or its legacy alias
      `'case3'` -- return the frame untouched, in either mode.
    - `INTERCONNECT_BOXCAR_CASES` -- `None` (the default) or `'synthetic'` -- select
      the boxcar placeholder, or defer to a loaded `transfer_csv`.

    Anything else is a ValueError rather than a silent fall-through to the boxcar: an
    unrecognized name quietly becoming the placeholder filter is exactly the bug this
    guard exists to prevent (see the note on the case sets above).

    The filter multiplies along the frequency axis and broadcasts over every leading
    axis, so MIMO and multi-chirp frames pass through natively (declared below).
    """

    frame_capabilities = _ELEMENTWISE

    def __init__(self, case=None, transfer_csv=None, band_hz=None, normalize_gain=False,
                 source=None, tessera_params=None,
                 tessera_arrangement=TESSERA_DEFAULT_ARRANGEMENT, tessera_cache=None,
                 scale=None):
        if case not in INTERCONNECT_CASES:
            raise ValueError(
                f"case must be one of "
                f"{sorted(INTERCONNECT_CASES, key=lambda c: (c is not None, c))}, "
                f"got {case!r}. {sorted(INTERCONNECT_PASSTHROUGH_CASES)} pass the frame "
                f"through untouched; None/'synthetic' select the 11-tap boxcar "
                f"placeholder. To apply a real interconnect response pass "
                f"transfer_csv=<path> (see e2e/data/interconnect/), not a case name."
            )
        if source not in INTERCONNECT_SOURCES:
            raise ValueError(f"source must be one of {sorted(INTERCONNECT_SOURCES, key=str)}, "
                              f"got {source!r}")
        if source == 'tessera' and transfer_csv is not None:
            raise ValueError("InterconnectBlock: pass either transfer_csv or "
                              "source='tessera', not both")
        self.case = case
        self.transfer_csv = transfer_csv
        self.band_hz = band_hz
        # Scale the applied response so its peak magnitude is exactly 1 (0 dB). The
        # boxcar placeholder has a peak gain of 11 (+20.8 dB), which no passive
        # interconnect can have; on a peak-normalized display the flat gain divides
        # out anyway, but the absolute level reaches the AFE quantizer's exponent
        # range and the tracker's error metric, so the default stays False for
        # bit-compatibility and the demo opts in (Thrust 4, notes/DEMO_DEFENSE.md).
        # Applies to the CSV mode too, where it removes flat insertion loss and
        # leaves only the in-band shape. Passthrough is untouched by definition.
        self.normalize_gain = bool(normalize_gain)
        self._csv_freq = None
        self._csv_s21 = None
        if transfer_csv is not None:
            self._csv_freq, self._csv_s21 = load_interconnect_transfer(transfer_csv)
        self.source = source
        self.tessera_arrangement = tessera_arrangement
        self.tessera_cache = tessera_cache
        self.tessera_params = None
        self.scale = None
        if source == 'tessera':
            # Resolved and validated eagerly (cheap: no third-party import), so a bad
            # knob value -- or an out-of-box scaled geometry -- fails at construction,
            # not on the first frame through the pipeline. Derived from `band_hz` (not
            # the eventual frame axis) so it is known even before any frame is seen --
            # `None`/no band always resolves to 1.0, matching `_resolve_tessera_scale`.
            self.scale = _resolve_tessera_scale(scale, band_hz)
            presented = _default_presented_tessera_params(self.scale)
            if tessera_params:
                presented.update(tessera_params)
            model = _presented_to_model_tessera_params(presented, self.scale)
            _validate_tessera_design(presented, model, tessera_arrangement, self.scale)
            self.tessera_params = presented

    def describe(self):
        """A one-line human summary for GUI banners; only meaningful for `source='tessera'`."""
        if self.source != 'tessera':
            return f"Interconnect: case={self.case!r}" if self.transfer_csv is None \
                else f"Interconnect: transfer_csv={Path(self.transfer_csv).name}"
        p = self.tessera_params
        scale_note = ""
        if self.scale != 1.0:
            if self.band_hz is not None:
                lo_ghz = self.band_hz[0] / self.scale / 1e9
                hi_ghz = self.band_hz[1] / self.scale / 1e9
                freq_note = f"evaluated at {lo_ghz:g}-{hi_ghz:g} GHz"
            else:
                freq_note = f"evaluated at 1/{self.scale:g}x frequency"
            scale_note = (f", scale model x{self.scale:g} ({self.scale:g}x geometry, "
                          f"{freq_note})")
        return (f"Tessera TSV surrogate{scale_note}: radius {p['radius_um']:g} um, "
                f"pitch {p['pitch_um']:g} um, height {p['height_um']:g} um, "
                f"liner {p['liner_um']:g} um, {p['temperature_k']:g} K, "
                f"{self.tessera_arrangement} arrangement")

    def frequency_response(self, n_freqs, dev):
        """The complex response `H[f]` this block multiplies a frame by, `[n_freqs]`.

        Exposed so callers (tests, the interconnect tutorial, the demo captions) can
        state the filter's peak gain and ripple from the same array `apply_interconnect`
        uses, instead of re-deriving the boxcar's FFT by hand.
        """
        if self.case in INTERCONNECT_PASSTHROUGH_CASES:
            return torch.ones(n_freqs, dtype=torch.complex64, device=dev)
        if self.source == 'tessera':
            H = self._tessera_response(n_freqs, dev)
        elif self.transfer_csv is not None:
            H = self._resampled_response(n_freqs, dev)
        else:
            window = torch.ones(11, device=dev)
            window_padded = torch.nn.functional.pad(window, (0, n_freqs - window.shape[0]))
            # This is a forward FFT of the (zero-padded) impulse response, i.e. the
            # interconnect's frequency response -- despite the historical name, it is
            # not an inverse FFT. Do not "fix" the direction; that would change the filter.
            H = torch.fft.fft(window_padded)
        if self.normalize_gain:
            H = H / torch.abs(H).max().clamp_min(1e-30)
        return H

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

    def _tessera_response(self, n_freqs, dev):
        """Evaluate the live Tessera surrogate over `band_hz` (or its own valid
        frequency span if `band_hz` is unset, mirroring `_resampled_response`'s
        CSV-span fallback), going through the on-disk cache."""
        if self.band_hz is not None:
            freqs = np.linspace(self.band_hz[0], self.band_hz[1], n_freqs)
        else:
            from e2e.interconnect_surrogate import VALID_RANGES

            flo, fhi = VALID_RANGES["freq_hz"]
            freqs = np.linspace(flo, fhi, n_freqs)
        s21 = tessera_s21_for_axis(freqs, params=self.tessera_params,
                                    arrangement=self.tessera_arrangement,
                                    cache=self.tessera_cache, scale=self.scale)
        return torch.tensor(s21, dtype=torch.complex64, device=dev)

    def apply_interconnect(self, frame):
        if self.case in INTERCONNECT_PASSTHROUGH_CASES:
            return frame
        H = self.frequency_response(frame.shape[-1], frame.device)
        return frame * H.view(1, 1, 1, -1)


# Adaptive Feature Extraction Block
class AFEBlock:
    """Analog feature extraction: the ADAPTIVE variant of `e2e.chain.compress`.

    Same physical idea as `CompressBlock` -- a large analog receive aperture (this
    package's array is 32x32 = 1024 elements) is combined down to a far smaller number
    of digitized channels, so the converter count and every downstream data rate fall by
    N/M. The two differ in exactly two respects, and share everything else:

    * **How the matrix is chosen.** `CompressBlock` draws one static matrix and keeps it,
      modelling a fixed combining network. The AFE redraws `A` each frame from the
      subspace tracker's current estimate (`MeasurementStage` -> `gen_A_ada`), so the
      measurements steer onto the signal subspace -- the "adaptive" in the name.
    * **How the weights are quantized.** The AFE uses the low-precision FLOATING-POINT
      model (constant relative error, a compute-datapath format), which is why `exp`
      and `mantissa` rather than a bit count. `CompressBlock` uses the uniform model
      (constant absolute step) appropriate to analog control settings. See
      `e2e.chain.compress.quantize_weights` -- both live there now, named and
      contrasted, so the choice is visible instead of implied by which class you picked.

    The combining and reconstruction MATH is shared (`compress.combine` /
    `compress.reconstruct_aperture`); this class contributes the adaptive draw and the
    float weight model, not a second implementation of compression.

    A thin preset of `e2e.chain.compress`'s CONTROL x COMPUTE axis matrix (see that
    module's docstring): CONTROL_ADAPTIVE (a fresh `A` is handed in by `MeasurementStage`
    on every call -- this class never redraws or caches it itself) x
    COMPUTE_QUANTIZED(weight_model=WEIGHT_FLOAT). `apply_mat_mul`/`reconstruct` delegate
    to `realize_weights`/`combine_realized`/`reconstruct_aperture` rather than carrying a
    second implementation.
    """

    # The measurement matrix A is drawn for ONE frame's flattened aperture, so the
    # compressed measurement is defined for a single chirp of a single-TX frame.
    frame_capabilities = _SINGLE_CHIRP

    def __init__(self, exp=5, mantissa=6):
        self.exp = exp
        self.mantissa = mantissa

    def apply_mat_mul(self, A, V):
        """Quantize `A` (float model) and combine: returns `(Aq, Aq @ V)`."""
        from e2e.chain.compress import COMPUTE_QUANTIZED, WEIGHT_FLOAT, combine_realized, realize_weights

        Aq = realize_weights(A, COMPUTE_QUANTIZED, weight_model=WEIGHT_FLOAT, exp=self.exp,
                             mantissa=self.mantissa)
        return Aq, combine_realized(Aq, V, COMPUTE_QUANTIZED)

    def reconstruct(self, Aq, X):
        """Least-squares aperture from the compressed measurements. LOSSY for M < N --
        see `compress.reconstruct_aperture`."""
        from e2e.chain.compress import reconstruct_aperture

        return reconstruct_aperture(Aq, X)


# Adaptive subspace-tracking block feeding the AFE.
class AdaOjaBlock:
    """Online subspace tracker for the adaptive feature-extraction stage.

    ``method='reestimate'`` (default): a warm-started power-iteration step on the
    back-projected measurements ``Y = A^H X`` -- ``U <- orth(Y (Y^H U))`` -- which
    re-estimates the top-k subspace at Oja complexity O(d*n*k) per step (NO SVD, NO
    pinv). Iterated across MeasurementStage's refinement passes it FOLLOWS a fast-drifting
    scene subspace, reaching the k-truncated-SVD floor ~50x cheaper than a full-SVD
    re-estimate. Two things make it work (an adversarial panel established that neither
    alone suffices):

      * enough measurements to OBSERVE the drift -- pass a larger ``m`` (e.g. 512); the
        per-frame observability of the orthogonal-complement drift is ~(m-k)/(d-k);
      * track at the signal's SPECTRAL ELBOW (small ``k``, e.g. 8), where the top-k
        subspace is well defined; at large k (e.g. 16) there is no singular-value gap
        and the target itself is unstable, so nothing tracks it.

    ``method='oja'``: the legacy incremental gradient step (kept for reference/tests).
    It normalizes the gradient to unit norm and takes a fixed-size step carrying NO
    information about the drift magnitude, so it barely rotates the subspace and does
    not track fast drift. See ROADMAP.md.

    The tracker's state is a d-dimensional basis for ONE aperture snapshot, so it
    declares the single-chirp/no-MIMO contract: a multi-chirp or MIMO frame would
    silently change what `d` indexes.
    """

    frame_capabilities = _SINGLE_CHIRP

    _GAP_RESPONSES = ("none", "refine", "coast")
    #: Recognised `method` values -- anything else used to fall through to the legacy path.
    _METHODS = ("reestimate", "oja")

    def __init__(self, d, k, eta=0.1, m=None, method="reestimate", n_refine=1,
                 gap_response="none", gap_threshold=0.01, n_refine_hi=60):
        self.oja = Oja(d, k, eta=eta, fixed_step=True)
        # Validate `method` for the same reason `gap_response` is validated below: an
        # unrecognised string silently selects the `else` branch in `update()` -- the
        # legacy incremental Oja step, which this class's own docstring says barely
        # rotates the subspace and cannot follow fast drift. A typo ("Reestimate",
        # "re-estimate") therefore yields a running pipeline with believable but far
        # worse tracking error and no indication why. Found by a blind tier-benchmark
        # audit, 2026-08-16, which flagged the asymmetry with gap_response.
        if method not in self._METHODS:
            raise ValueError(f"method must be one of {self._METHODS}, got {method!r}")
        self.method = method
        # Measurement count for the adaptive sensing matrix (rows of A). Defaults to the
        # legacy 2*k; the 'reestimate' tracker needs many more to observe drift, so
        # callers that want accurate tracking pass an explicit m (see the demos).
        self.m = m if m is not None else self.oja.k * 2
        # m == k is a SILENT observability failure, not a slow one: gen_A_ada builds its
        # exploratory block B with (m - k) columns, so at m == k the sensing matrix is
        # nothing but the anchor rows U^H and no direction outside the current estimate
        # is ever measured. The estimate then freezes bit-identically no matter how much
        # real signal flows through -- demonstrated over 30 frames by the same audit.
        # (m < k already fails, but obscurely, inside a negative-dimension randn.)
        if self.m <= self.oja.k:
            raise ValueError(
                f"m must exceed k to observe drift outside the tracked subspace "
                f"(got m={self.m}, k={self.oja.k}); at m == k the sensing matrix is the "
                f"anchor rows alone and the estimate can never update"
            )
        # Refinement iterations per frame (MeasurementStage re-draws A from the updated
        # estimate and re-estimates n_refine times). The sensing matrix's anchor rows are
        # built from the PREVIOUS estimate; on a drifting scene that stale anchor caps the
        # error ~10x above the SVD floor, and each refinement re-aims the anchor at the
        # current subspace. n_refine≈5 reaches the floor; 1 = single-shot (legacy).
        self.n_refine = n_refine
        # Opt-in reaction to a near-degenerate singular-value cluster sitting right at
        # the k cutoff (see `effective_n_refine`). Off by default ("none"): every
        # existing caller keeps getting exactly `n_refine` every frame, unconditionally.
        if gap_response not in self._GAP_RESPONSES:
            raise ValueError(
                f"gap_response must be one of {self._GAP_RESPONSES}, got {gap_response!r}"
            )
        self.gap_response = gap_response
        # sv_gap_norm = (S[k-1]-S[k])/S[0] (see e2e.simulation.rank_diagnostic); below
        # this the top-k subspace identity is judged ill-conditioned at the cutoff.
        # 0.01 was calibrated on the 30-frame munich protocol: collapse frames 22-29
        # sit at <= 0.0022, every earlier frame at >= 0.013 -- the same trigger set as
        # the ratio gate this replaces, but the normalized ABSOLUTE gap also collapses
        # when the tail sinks into the noise floor (S[k-1], S[k] both << S[0]), where a
        # ratio of two noise-floor values can read "healthy" (munich frame 18: ratio
        # 2.34 while S[k-1]/S[0] had already fallen to 0.12).
        self.gap_threshold = gap_threshold
        # Refinement count used for gap_response="refine" while the gap is collapsed.
        self.n_refine_hi = n_refine_hi

    def effective_n_refine(self, sv_gap_norm=None):
        """The refinement count MeasurementStage should run THIS frame.

        `sv_gap_norm` is a SPECTRUM-only diagnostic -- the normalized absolute gap
        (S[k-1] - S[k]) / S[0], computed upstream in `Simulation.feed_forward` from the
        frame's singular-value spectrum alone (see `rank_diagnostic`). It never touches
        the ground-truth basis U_true, so reacting to it is not a peek at ground truth
        -- the same guarantee the tracker's own online update already has to hold.
        It IS, however, oracle-sourced: the simulator gets S[k] from the full-frame
        SVD it performs anyway for scoring, which a deployed receiver storing only a
        rank-k basis would not have. The equivalent signal is estimable from the
        m >> k measurement covariance (see the honesty note on `rank_diagnostic`),
        but that estimator is not implemented.
        The ABSOLUTE gap is what conditions the top-k subspace's identity (Davis-Kahan),
        so unlike the raw ratio S[k-1]/S[k] this signal also collapses when the tail is
        merely insignificant (both singular values near the noise floor), not only when
        the cutoff straddles a degenerate cluster.

        gap_response="none" (default): always `n_refine`, ignoring the signal entirely
        (bit-identical to every AdaOjaBlock built before this option existed).
        gap_response="refine": bump to `n_refine_hi` refinement passes while the gap is
        collapsed (sv_gap_norm < gap_threshold) -- spend more compute chasing the
        ambiguous cutoff. Productionized from the investigation's "reactive n_refine"
        finding (a near-degenerate SV cluster at the k cutoff on real munich frames
        makes the top-k subspace identity numerically arbitrary and spikes tracking
        error from frame ~22): spike-mean -66% / post -93% on the 30-frame munich
        protocol, re-measured under THIS shipped gate (sv_gap_norm, threshold 0.01).
        An earlier -46%/-94% figure circulated from the investigation's prototype
        ratio gate (S[k-1]/S[k] < 1.15) -- same mechanism, different gate signal;
        quote the re-measured numbers.
        gap_response="coast": drop to 0 refinement passes (freeze U, skip the update)
        while the gap is collapsed, resuming once it reopens -- "detect-and-coast": stop
        chasing a target whose own frame-to-frame identity is numerically arbitrary
        (ground truth's drift itself jumps 0.24 -> ~1.0 in the same window) rather than
        spend compute re-estimating in a direction likely to be reverted next frame.
        (Benchmarked WORSE than baseline -- kept as a documented negative result.)

        A missing or NaN sv_gap_norm (no diagnostic available -- e.g. the replay path,
        which never computes one) falls back to `n_refine`, matching the "off" behavior.
        """
        if self.gap_response == "none" or sv_gap_norm is None:
            return self.n_refine
        if sv_gap_norm != sv_gap_norm:  # NaN (k >= len(S)); no gap to react to
            return self.n_refine
        collapsed = sv_gap_norm < self.gap_threshold
        if not collapsed:
            return self.n_refine
        return self.n_refine_hi if self.gap_response == "refine" else 0

    def gen_A_ada(self, m=None):
        return gen_A_ada(self.oja.U, m if m is not None else self.m)

    def update(self, X, A):
        if self.method == "reestimate":
            # Cheap (Oja-complexity) subspace re-estimate: one warm-started power-iteration
            # step on the back-projected measurements Y = A^H X (a proxy for the aperture
            # signal). Started from the current U and iterated across MeasurementStage's
            # refinement passes, it converges to the top-k subspace at O(d*n*k) per step --
            # NO SVD and NO pinv, ~50x cheaper than a full-SVD re-estimate for the same
            # accuracy (it reaches the k-truncated-SVD floor). Row(A) still contains U's
            # anchor rows, so this recovers the signal's dominant directions.
            Y = A.conj().T @ X
            Z = Y @ (Y.conj().T @ self.oja.U)
            # Zero-norm guard, mirroring Oja.add_data's (see e2e/subspace/algorithms.py).
            # This path is WORSE than the legacy one without it: a degenerate frame makes
            # Z all-zero, and orth()/QR of a zero matrix returns a perfectly valid but
            # ARBITRARY orthonormal basis -- no NaN, no exception, no log. The tracker
            # would silently swap the scene's subspace for an unrelated one and carry on
            # looking healthy. An online tracker prefers skipping an uninformative frame
            # to corrupting its state. (Found by the compression/subspace audit,
            # 2026-08-16: the existing zero-frame regression test only ever exercised the
            # legacy Oja class, never this -- the shipped default -- path.)
            z_norm = torch.linalg.norm(Z)
            if not torch.isfinite(z_norm) or z_norm == 0:
                return
            self.oja.U = orth(Z)
        else:
            # Legacy incremental Oja step. RMS-normalize X first so the (scale-invariant)
            # tracked direction doesn't retune eta with the input's absolute volts.
            xn = X / (torch.sqrt(torch.mean(torch.abs(X) ** 2)) + 1e-30)
            self.oja.add_data(xn, A)


# -------------------------------------------------------------- serial pipeline stages
# These implement the same protocol as downstream product blocks (`apply(state) ->
# dict of updates`), but they run in the FIRST loop of Simulation.feed_forward, in
# series, each one able to (and expected to) update 's_pars' -- they ARE the pipeline,
# not products of it. See e2e/simulation.py.
#
# Each stage mirrors the `frame_capabilities` of the block it wraps (and names that
# block via `frame_contract_name`) so Simulation's per-component frame validation
# reports the block the user actually configured, not the wrapper.

class CircuitStage:
    """RF front-end circuit distortion, applied directly to the single-chirp frame.

    Previously this duplicated the chirp dim (torch.cat([s, s], dim=2)) to feed
    RFFEBlock a vestigial "pol pair", ran the full circuit on both copies (2x
    compute, 2 independent noise draws), then discarded the second copy via
    s_pars[:, :, :1, :]. RFFEBlock.apply_circuit's reshape now infers the chirp dim
    instead of hardcoding 2, so it accepts the single-chirp frame natively -- one
    noise realization per element, no wasted compute.
    """

    frame_capabilities = _ELEMENTWISE

    def __init__(self, rffe_block):
        self.rffe_block = rffe_block
        self.frame_capabilities = frames.capabilities_of(rffe_block)
        self.frame_contract_name = f"CircuitStage[{type(rffe_block).__name__}]"

    def reset(self):
        """Rewind a seeded RFFEBlock's per-frame noise counter so a reused Simulation
        replays the same noise sequence (D1, 2026-09-23); a no-op for unseeded blocks."""
        if hasattr(self.rffe_block, "reset"):
            self.rffe_block.reset()

    def apply(self, state):
        s_pars, PRX = self.rffe_block.apply_circuit(state["s_pars"])
        # Advertise WHICH amplitude scale the frame is now on. `RFFEBlock` with
        # physical_scale=False divides the frame by its own mean magnitude, erasing the
        # absolute level; anything downstream that installs an ABSOLUTE reference (the
        # kTBF floor in `ThermalNoiseBlock`) is then meaningless. That composition is
        # F63, and it survived every review because nothing in the chain could see it.
        # Blocks other than RFFEBlock are assumed absolute -- only the normalisation
        # above forfeits the scale.
        absolute = bool(getattr(self.rffe_block, "physical_scale", True))
        return {"s_pars": s_pars, "PRX": PRX,
                "amplitude_scale": "absolute" if absolute else "normalised"}


class GridStage:
    """Reshapes the frame onto the physical receive-array grid.

    The aperture view is receive-only, so this stage cannot express a TX axis (no
    MIMO); the chirp axis rides along untouched ([rx_x, rx_y, n_chirp, n_freqs]), and
    from here on dim 1 is ELEVATION, not TX -- which is why it also flips the state's
    frame layout to LAYOUT_APERTURE for the validation downstream of it.
    """

    frame_capabilities = FrameCapabilities(accepts_mimo=False, chirps=frames.CHIRP_NATIVE)

    def __init__(self, array_shape):
        self.array_shape = array_shape

    def apply(self, state):
        s_pars = state["s_pars"]
        # Redundant with Simulation's pre-call validation, but kept so direct callers
        # get the named contract error instead of a raw view() size mismatch.
        frames.require_no_mimo(s_pars, "GridStage")
        s_pars = frames.to_aperture_grid(s_pars, self.array_shape)
        return {"s_pars": s_pars, "frame_layout": frames.LAYOUT_APERTURE}


class InterconnectStage:
    """Interconnect filtering on the aperture grid."""

    frame_capabilities = _ELEMENTWISE

    def __init__(self, interconnect_block):
        self.interconnect_block = interconnect_block
        self.frame_capabilities = frames.capabilities_of(interconnect_block)
        self.frame_contract_name = f"InterconnectStage[{type(interconnect_block).__name__}]"

    def apply(self, state):
        s_pars = self.interconnect_block.apply_interconnect(state["s_pars"])
        return {"s_pars": s_pars}


class MeasurementStage:
    """Adaptive compression (optional AFE) + online subspace tracking.

    With an AFE block: quantized measurement matrix, quantized matmul, subspace
    update, then -- if `reconstruct=True` -- reconstruction, with the reconstructed grid
    replacing 's_pars'.
    Without an AFE block: the subspace tracker is fed the full-precision compressed
    measurements directly (same A-generation, no quantization); 's_pars' is left
    unchanged.

    `reconstruct` (default True, the historical behaviour) decides whether the chain
    continues in FULL or REDUCED dimension, which is the architectural choice this
    stage exists to express. Digitizing a 1024-element aperture needs 1024 converters;
    combining it down to 16-64 measurements first means that many, and everything
    downstream then works in the measurement space. Reconstructing immediately -- which
    this stage used to do unconditionally -- throws that away, pays a lossy pseudo-inverse
    for M < N, and hides the choice. Set `reconstruct=False` to keep the measurements:
    's_pars' stays `[M, 1, n_chirp, n_freqs]`, `state['signal_dimension']` becomes
    `DIMENSION_REDUCED`, and blocks that index physical antennas will be stopped by the
    frame contract rather than quietly imaging random projections. The subspace tracker
    itself is content either way -- estimating a subspace from projections is the entire
    premise of the AFE.
    """

    frame_capabilities = _SINGLE_CHIRP

    def __init__(self, afe_block, subspace_block, reconstruct: bool = True):
        self.afe_block = afe_block
        self.subspace_block = subspace_block
        self.reconstruct = bool(reconstruct)
        if afe_block is None and not self.reconstruct:
            # Without an AFE there is no compression to keep, so the flag would silently
            # do nothing. Refuse rather than accept a request we cannot honour.
            raise ValueError(
                "reconstruct=False requires an afe_block: without one, MeasurementStage "
                "feeds the tracker directly and never compresses `s_pars`, so there is "
                "no reduced dimension to stay in."
            )
        if afe_block is not None and not self.reconstruct:
            # Declared per-instance: the same class either preserves the dimension or
            # crosses it, depending on configuration, so the capability cannot be a
            # class attribute the way a fixed-behaviour block's can.
            self.frame_capabilities = frames.FrameCapabilities(
                accepts_mimo=_SINGLE_CHIRP.accepts_mimo,
                chirps=_SINGLE_CHIRP.chirps,
                domain=_SINGLE_CHIRP.domain,
                emits_dimension=frames.DIMENSION_REDUCED,
            )
        # The stage's own math (one A per frame, flatten the aperture to [d, n_freqs])
        # is single-chirp/no-MIMO regardless of what the blocks declare; name the
        # blocks it drives so the contract error points at what the user configured.
        wrapped = [b for b in (subspace_block, afe_block) if b is not None]
        if wrapped:
            names = "/".join(type(b).__name__ for b in wrapped)
            self.frame_contract_name = f"MeasurementStage[{names}]"

    def apply(self, state):
        s_pars = state["s_pars"]
        # Refine the subspace estimate n_refine times: each pass re-draws the sensing
        # matrix from the CURRENT estimate and re-estimates, so A's anchor rows converge
        # onto this frame's subspace (a stale anchor from the previous frame otherwise
        # caps accuracy ~10x above the SVD floor). n_refine=1 is the single-shot legacy path.
        # A subspace_block that declares `effective_n_refine` (AdaOjaBlock's opt-in
        # gap_response) gets to adjust the count from `state['sv_gap_norm']` -- a
        # spectrum-only diagnostic Simulation threads through, never the ground-truth
        # basis (see AdaOjaBlock.effective_n_refine). Anything else keeps the plain
        # `n_refine` attribute, unconditionally, as before.
        get_n_refine = getattr(self.subspace_block, "effective_n_refine", None)
        if callable(get_n_refine):
            n_refine = get_n_refine(state.get("sv_gap_norm"))
        else:
            n_refine = getattr(self.subspace_block, "n_refine", 1)
        if self.afe_block:
            V = s_pars.view(-1, s_pars.shape[-1])
            for _ in range(n_refine):
                A = self.subspace_block.gen_A_ada()
                Aq, X = self.afe_block.apply_mat_mul(A, V)
                self.subspace_block.update(X, Aq)
            # Final measurement with the converged basis.
            A = self.subspace_block.gen_A_ada()
            Aq, X = self.afe_block.apply_mat_mul(A, V)
            if not self.reconstruct:
                # Stay in the measurement space: dim 0 now counts MEASUREMENTS, not
                # antennas, and the contract says so for everything downstream.
                n_chirp, n_freqs = s_pars.shape[2], s_pars.shape[3]
                return {
                    "s_pars": X.view(X.shape[0], 1, n_chirp, n_freqs),
                    "U": self.subspace_block.oja.U,
                    "sensing_matrix": Aq,
                    "aperture_shape": (s_pars.shape[0], s_pars.shape[1]),
                    "signal_dimension": frames.DIMENSION_REDUCED,
                    "n_refine_used": n_refine,
                }
            Xt = self.afe_block.reconstruct(Aq, X)
            s_pars = Xt.view(s_pars.shape)
            return {"s_pars": s_pars, "U": self.subspace_block.oja.U,
                    "n_refine_used": n_refine}
        else:
            # No AFE: feed the subspace tracker the full-precision compressed
            # measurements directly (same A-generation as the AFE branch, but
            # without quantization).
            V = s_pars.view(-1, s_pars.shape[-1])
            for _ in range(n_refine):
                A = self.subspace_block.gen_A_ada()
                X = A @ V
                self.subspace_block.update(X, A)
            return {"U": self.subspace_block.oja.U, "n_refine_used": n_refine}


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
    if L < n_bins:
        # Pass-through, and SAY SO. Callers reasonably read the block docstrings'
        # "[bins, bins]" as a guarantee and slice on it; before this warning the map
        # came back [bins, L] silently and the mistake surfaced far from its cause.
        # We do not pad up to n_bins on purpose: zero-padding would manufacture range
        # gates -- and their sidelobes -- that the band never resolved.
        warnings.warn(
            f"range axis has only {L} samples but {n_bins} display gates were "
            f"requested, so the map is [{n_bins}, {L}], NOT [{n_bins}, {n_bins}]: a "
            f"short band is reported at the resolution it actually measured rather "
            f"than interpolated up. Pass bins<={L} to make this explicit.",
            stacklevel=3,
        )
        return power
    if L == n_bins:
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

    Multi-chirp frames are handled per chirp (CHIRP_BROADCAST): 'fft' is
    ``[bins, bins]`` for the single-chirp frames the pipeline has always produced, and
    ``[n_chirp, bins, bins]`` ONLY when n_chirp > 1.
    """

    frame_capabilities = _PER_CHIRP

    def __init__(self, bins=256, window=None):
        self.bins = bins
        self.window = window

    def apply(self, state_dict):
        return {'fft': frames.broadcast_over_chirps(state_dict['s_pars'], self._map)}

    def _map(self, data):
        # data: one chirp's aperture slab [az, el, n_freqs]
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
        return acc


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

    Multi-chirp frames are handled per chirp (CHIRP_BROADCAST): 'range_az' is
    ``[bins, n_range]`` for single-chirp frames and ``[n_chirp, bins, n_range]`` ONLY
    when n_chirp > 1, where ``n_range = min(bins, n_freqs)``.

    **``n_range`` is ``bins`` only when the band actually has that many samples.**
    The range axis is produced by power-BINNING the full-band profile DOWN to ``bins``
    display gates, which is only possible when ``n_freqs >= bins``; with a shorter band
    there is nothing to bin down and the block returns the ``n_freqs`` gates it really
    measured. It does NOT pad or interpolate up to ``bins``, and that is deliberate --
    zero-padding would manufacture range gates, and their sidelobes, that the band never
    resolved. This docstring previously promised ``[bins, bins]`` unconditionally, which
    was wrong for any short band and is reachable from a shipped scenario:
    ``e2e/environment/scenarios/canyon_radar.json`` sets ``num_freqs: 128`` while
    ``e2e/main/main_sionna_blocks.py`` constructs this block at the default ``bins=256``.
    A warning is emitted in that case, since a caller who slices assuming a square map
    would otherwise fail somewhere far away from the cause.
    """

    frame_capabilities = _PER_CHIRP

    def __init__(self, bins=256, window=None):
        self.bins = bins
        self.window = window

    def apply(self, state_dict):
        return {'range_az': frames.broadcast_over_chirps(state_dict['s_pars'], self._map)}

    def _map(self, data):
        # data: one chirp's aperture slab [az, el, n_freqs]
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
        return _power_bin(power, self.bins, dim=1)   # n_freqs range bins -> bins gates


class RangeElBlock:
    """Range-elevation power map: coherent elevation FFT + full-band range
    compression, non-coherent (power) integration across the collapsed azimuth
    axis. See RangeAzBlock for the rationale (azimuth/elevation swapped): ``bins``
    sizes only the elevation aperture FFT and the range display axis; range
    compression uses the full band and is power-binned to ``bins`` gates.

    ``window`` optionally tapers the elevation aperture (None / 'hann' / 'hamming').

    Multi-chirp frames are handled per chirp (CHIRP_BROADCAST): 'range_el' is
    ``[bins, n_range]`` for single-chirp frames and ``[n_chirp, bins, n_range]`` ONLY
    when n_chirp > 1, where ``n_range = min(bins, n_freqs)`` -- see ``RangeAzBlock`` for
    why a short band is reported at its own resolution rather than padded up to ``bins``.
    """

    frame_capabilities = _PER_CHIRP

    def __init__(self, bins=256, window=None):
        self.bins = bins
        self.window = window

    def apply(self, state_dict):
        return {'range_el': frames.broadcast_over_chirps(state_dict['s_pars'], self._map)}

    def _map(self, data):
        # data: one chirp's aperture slab [az, el, n_freqs]
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
        return _power_bin(power, self.bins, dim=1)   # n_freqs range bins -> bins gates


class RangeProfileBlock:
    """Per-measurement-channel range profile: a windowless FFT along the FREQUENCY
    axis only -- no aperture transform. The first COMPRESSED-DOMAIN downstream product.

    THE ASYMMETRY THIS BLOCK EXISTS TO SHOW. Compression measures ``y = A x``, and
    `A` mixes only the APERTURE axis -- each measurement is a linear combination of
    physical elements -- while leaving the FREQUENCY axis completely untouched
    (`e2e.chain.compress.combine` / `AFEBlock.apply_mat_mul` never index dim -1). So
    for measurement channel `m`, ``y[m, f] = sum_n A[m, n] x[n, f]``, and an FFT over
    `f` commutes straight through that sum: ``range(y[m, :]) = sum_n A[m, n] *
    range(x[n, :])`` -- a valid (if basis-mixed) full-resolution range profile per
    channel, with no reconstruction needed. An ANGLE FFT has no such luck: it needs
    one sample per physical element to steer a beam, and a random linear combination
    of elements is not a beam-steerable aperture at all -- which is why
    RangeAzBlock/RangeElBlock/FFTBlock consume the full aperture grid while THIS block,
    like `SubspaceErrorBlock`, declares `DIMENSION_ANY` (via `frame_capabilities`,
    exactly as `SubspaceErrorBlock` does): it runs unmodified whether `s_pars` is the
    full aperture or `MeasurementStage(reconstruct=False)`'s `M` measurements, because
    whether dim 0 counts elements or measurements is irrelevant to a per-channel range
    transform.

    Range compression matches RangeAzBlock/RangeElBlock's convention exactly: the
    full frequency band is FFT'd (never truncated to `bins`), fftshifted, and turned
    into power (`|.|**2`, non-coherent) -- no window on the range axis
    (`_aperture_window` tapers only aperture/angle axes and is never applied here),
    and no dB conversion (that is a display-layer choice made downstream, e.g.
    `webapp/pipeline_runner.py`'s dB helper). `bins` sizes only the DISPLAY range
    axis, via the same energy-preserving `_power_bin` RangeAzBlock/RangeElBlock use.

    Outputs `'range_profile'` -- per-channel power, `[n_channels, bins]` -- and
    `'range_profile_agg'` -- the non-coherent (power) MEAN over channels, `[bins]` --
    matching the repo's non-coherent combining convention for display products (see
    RangeAzBlock's elevation integration).
    """

    # Single-chirp/no-MIMO like the other range products, but DIMENSION_ANY: this
    # block only ever indexes the frequency axis, so whether the aperture has been
    # compressed is none of its business (see SubspaceErrorBlock for the same pattern).
    frame_capabilities = frames.FrameCapabilities(
        accepts_mimo=_SINGLE_CHIRP.accepts_mimo,
        chirps=_SINGLE_CHIRP.chirps,
        domain=_SINGLE_CHIRP.domain,
        dimension=frames.DIMENSION_ANY,
    )

    def __init__(self, bins=256):
        self.bins = bins

    def apply(self, state_dict):
        data = frames.chirp0(state_dict['s_pars'])    # [dim0, dim1, n_freqs]
        n_freqs = data.shape[-1]
        channels = data.reshape(-1, n_freqs)           # [n_channels, n_freqs]
        r = torch.fft.fftshift(torch.fft.fft(channels, dim=1), 1)   # full-band range
        power = torch.abs(r) ** 2
        power = _power_bin(power, self.bins, dim=1)    # n_freqs range bins -> bins gates
        agg = torch.mean(power, dim=0)                  # non-coherent combine over channels
        return {'range_profile': power, 'range_profile_agg': agg}


class SubspaceErrorBlock:
    # Reads the tracker's basis (state['U'] / state['U_true']), not the frame, but it is
    # only meaningful alongside the single-chirp/no-MIMO subspace path, so it declares
    # that contract for the frame AXES.
    #
    # DIMENSION_ANY, though, and deliberately: this block never touches `s_pars`, so
    # whether the aperture has been compressed is none of its business. Declaring
    # full-dimension (the default) made it reject the very configuration
    # `MeasurementStage(reconstruct=False)` exists to enable -- tracking a subspace from
    # compressed measurements and then measuring the error -- which would have forced a
    # pointless, lossy DecompressBlock in front of a block that reads no frame at all.
    frame_capabilities = frames.FrameCapabilities(
        accepts_mimo=_SINGLE_CHIRP.accepts_mimo,
        chirps=_SINGLE_CHIRP.chirps,
        domain=_SINGLE_CHIRP.domain,
        dimension=frames.DIMENSION_ANY,
    )

    def __init__(self):
        self.metric = subspace_dist_frob

    def apply(self, state_dict):
        U = state_dict['U_true']
        U_pred = state_dict['U']
        return {'subspace_err': self.metric(U, U_pred)}

