"""The radar link budget: what turns an arbitrary-scale cube into absolute power.

WHY THIS MODULE EXISTS -- read this before changing a constant in it.

`e2e.ml.rt_signal_chain` builds a dechirped cube whose per-target amplitude is

    amp = g_elem^2 * sqrt(sigma) * lambda / ((4*pi)^1.5 * d_r * d_t)

which is the radar range equation carrying antenna gain, RCS, wavelength and R^4 -- but
NOT transmit power, and with no noise term anywhere. The cube is therefore proportional to
physical amplitude but has no absolute scale, and the corpora have no thermal noise floor
at all (measured: `notes/ESTABLISHED_FACTS.md` F42 -- every RT corpus's background fails
both tests for a thermal distribution, while the analytic generator passes them almost
exactly).

That absence is what made F35 possible. With no absolute reference in the chain, every
"how strong is this impairment" question could only be answered RELATIVE to something else
in the cube -- and everything else in the cube scales with the target. Injected impairments
were calibrated against the cube's own peak, so target-to-clutter was exactly invariant to
target strength (0.00 dB across a 20 dB sweep) and no physics improvement could ever move
detection. The fix is not a better relative reference. It is an ABSOLUTE one.

    signal:  P_rx = P_tx * G^2 * sigma * lambda^2 / ((4*pi)^3 * R^4)
    noise:   N    = k * T * B * F

Everything on the signal side except `P_tx` is already in the cube. So the entire link
budget reduces to the three constants below, and they are now THE DIFFICULTY DIAL for
every generated corpus: they set the absolute SNR of every target at every range. Change
them and every detection number downstream moves. That is why they are here, named, in one
place, with their provenance written down -- rather than spread through the code as
plausible-looking magic numbers.

**These are datasheet-plausible values for a TI IWR1443-class 77 GHz automotive MMIC, not
measurements from a specific part.** If real numbers for the modelled hardware ever arrive,
this is the only place to change, and the whole corpus difficulty moves coherently with it.
Do not tune them to make a detector's curves look better; that is precisely the failure
mode this module was created to end.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

#: Boltzmann constant, J/K.
K_BOLTZMANN = 1.380649e-23

#: Standard noise reference temperature, K. 290 K is the IEEE convention for noise figure;
#: it is a DEFINITION, not the ambient temperature of a car bumper, and changing it would
#: make every quoted noise figure mean something different.
T0_KELVIN = 290.0

#: Transmit power per TX channel, dBm. IWR1443-class parts sit around 12 dBm of output
#: power per channel at 77 GHz.
DEFAULT_TX_POWER_DBM = 12.0

#: Receiver noise figure, dB, referenced at the antenna input. 15 dB is typical for an
#: integrated 77 GHz automotive receive chain (a discrete low-noise design would do better;
#: an integrated MMIC with the mixer and IF chain on die does not).
DEFAULT_NOISE_FIGURE_DB = 15.0


def _get(cfg, name: str, default: float) -> float:
    """Read a link-budget field off `cfg`, falling back to this module's default.

    Deliberately tolerant: `RadarConfig` gained these fields after several corpora were
    generated, and a config deserialized from an older manifest will not carry them. The
    fallback is the SAME constant the current default config uses, so an old config and a
    new one agree -- but the value is never silently invented from nothing.
    """
    value = getattr(cfg, name, None)
    return float(default if value is None else value)


def noise_bandwidth_hz(cfg) -> float:
    """The bandwidth the thermal floor is integrated over, Hz.

    `fs`, the ADC sample rate, and the reason is worth stating because it is the easiest
    thing in this module to get wrong by one large factor.

    After dechirp the receiver's IF chain is sampled at `fs`, and the anti-alias filter
    ahead of it passes the full Nyquist span. Every one of those `n_samples` samples
    therefore carries thermal noise of power `k*T*B*F` with `B = fs`. It is tempting to use
    the per-range-bin bandwidth `fs / n_samples` instead, on the grounds that a target
    occupies one bin -- but that would be double-counting the processing gain: the range
    FFT's coherent integration over `n_samples` is what concentrates the target while the
    noise adds incoherently, and that gain belongs in the PROCESSING, not in the floor.
    Put it in the floor as well and every reported SNR is optimistic by 10*log10(n_samples)
    -- 27 dB at n_samples = 512.
    """
    return float(cfg.fs_hz)


def thermal_noise_power_w(cfg, *, noise_figure_db: Optional[float] = None) -> float:
    """Absolute thermal noise power at the receiver input, watts: `N = k*T*B*F`.

    This is the per-sample noise power in the dechirped record, before any range or
    Doppler integration.
    """
    nf_db = _get(cfg, "noise_figure_db", DEFAULT_NOISE_FIGURE_DB) \
        if noise_figure_db is None else float(noise_figure_db)
    f_linear = 10.0 ** (nf_db / 10.0)
    return K_BOLTZMANN * T0_KELVIN * noise_bandwidth_hz(cfg) * f_linear


def tx_amplitude_scale(cfg, *, tx_power_dbm: Optional[float] = None) -> float:
    """The scalar that turns the chain's arbitrary-unit cube into volts: `sqrt(P_tx)`.

    `rt_signal_chain`'s amplitude term already carries antenna gain, RCS, wavelength and
    the R^4 spreading -- everything in the radar equation except the transmit power. So one
    multiply by `sqrt(P_tx)` makes the cube an absolute received voltage, and the thermal
    floor above is then directly comparable to it.
    """
    dbm = _get(cfg, "tx_power_dbm", DEFAULT_TX_POWER_DBM) \
        if tx_power_dbm is None else float(tx_power_dbm)
    return math.sqrt(10.0 ** ((dbm - 30.0) / 10.0))     # dBm -> W -> volts


def expected_target_snr_db(cfg, range_m: float, rcs_dbsm: float,
                           *, gain_dbi: float = 0.0) -> float:
    """Single-sample SNR, dB, for a point target -- the hand-checkable form of this budget.

    Written out as the textbook radar equation rather than measured off a cube, so it can
    be checked against a datasheet or a napkin independently of anything the simulator
    does. `tests/test_ml_link_budget.py` compares it against what the chain actually
    produces; the two agreeing is the oracle for this whole module.

    Post-integration SNR adds the coherent gain of the range and Doppler transforms,
    `10*log10(n_samples * n_chirps_per_tx)`; see `coherent_processing_gain_db`.
    """
    lam = 299_792_458.0 / float(cfg.f0_hz)
    p_tx_w = 10.0 ** ((_get(cfg, "tx_power_dbm", DEFAULT_TX_POWER_DBM) - 30.0) / 10.0)
    g = 10.0 ** (float(gain_dbi) / 10.0)
    sigma = 10.0 ** (float(rcs_dbsm) / 10.0)
    r = float(range_m)
    p_rx = p_tx_w * g * g * sigma * lam * lam / (((4.0 * math.pi) ** 3) * (r ** 4))
    return 10.0 * math.log10(p_rx / thermal_noise_power_w(cfg))


def coherent_processing_gain_db(cfg) -> float:
    """`10*log10(n_samples * n_chirps_per_tx)` -- the range+Doppler integration gain.

    Kept next to the floor it is deliberately NOT folded into (see `noise_bandwidth_hz`),
    so the one place someone might double-count it is the one place it is explained.
    """
    n_per_tx = getattr(cfg, "n_chirps_per_tx", None) or cfg.n_chirps
    return 10.0 * math.log10(float(cfg.n_samples) * float(n_per_tx))


def add_thermal_noise(adc: torch.Tensor, cfg, *, seed: int,
                      noise_figure_db: Optional[float] = None) -> torch.Tensor:
    """Add ABSOLUTE thermal noise to a cube that is already in volts.

    Contrast with `rt_signal_chain._add_awgn`, which derives its noise power from the
    PEAK SCATTERER in the scene (`sigma2 = a_max^2 * G / 10^(snr_db/10)`). That is a
    target-relative floor: a stronger target raises the noise with it, pinning target SNR
    at `snr_db` by construction no matter what the physics does. It is a legitimate knob
    for "synthesize a scene at a given SNR", and it is the wrong thing entirely for a
    corpus whose whole purpose is that detectability responds to the scene.

    This function ignores the cube's contents. The floor is `k*T*B*F`, full stop.
    """
    n_w = thermal_noise_power_w(cfg, noise_figure_db=noise_figure_db)
    gen = torch.Generator(device=adc.device)
    gen.manual_seed(int(seed))
    # Complex circular Gaussian: half the power in each quadrature.
    w = torch.randn(tuple(adc.shape) + (2,), generator=gen, device=adc.device,
                    dtype=torch.float32) * math.sqrt(n_w / 2.0)
    return adc + torch.view_as_complex(w.contiguous()).to(adc.dtype)


class ThermalNoiseBlock:
    """Chain stage: put the cube on an ABSOLUTE power scale and add the thermal floor.

    Sits between `DechirpBlock` (which produces `adc`) and `ImpairmentBlock` (which adds
    leakage/clutter/phase noise). That position is the whole point: impairments are
    specified relative to a reference, and until this block runs there is no absolute
    reference in the chain for them to be relative TO. Everything the corpora suffered
    from -- F35's ceiling, F42's missing floor -- traces to impairments being calibrated
    against the only thing available, which was the cube's own contents.

    Two operations, both scalar:
      * multiply by `sqrt(P_tx)`, which turns `rt_signal_chain`'s arbitrary-unit amplitude
        into an absolute received voltage (that amplitude already carries antenna gain,
        RCS, wavelength and R^4 -- transmit power is the only missing factor);
      * add complex Gaussian noise of power `k*T*B*F`.

    Deterministic from `seed`, per frame, in the same style as `ImpairmentBlock`: frame i
    uses `seed + i`, so two runs with the same seed reproduce bit-identically and a
    different seed does not.
    """

    def __init__(self, cfg, *, seed: int = 0, enabled: bool = True):
        from e2e.chain.receive import _RX_TIME

        self.frame_capabilities = _RX_TIME
        self.cfg = cfg
        self.seed = int(seed)
        self.enabled = bool(enabled)
        self._frame_idx = 0

    def reset(self):
        """Rewind the per-frame counter (and hence the seed sequence) to frame 0."""
        self._frame_idx = 0

    def apply(self, state):
        adc = state["adc"]
        if not self.enabled:
            return {"adc": adc}
        scaled = adc * tx_amplitude_scale(self.cfg)
        out = add_thermal_noise(scaled, self.cfg, seed=self.seed + self._frame_idx)
        self._frame_idx += 1
        # Recorded so a frame can always say what floor it was generated against -- the
        # number that makes every impairment dB value on it meaningful.
        return {"adc": out,
                "link_budget": {
                    "tx_power_dbm": _get(self.cfg, "tx_power_dbm", DEFAULT_TX_POWER_DBM),
                    "noise_figure_db": _get(self.cfg, "noise_figure_db",
                                            DEFAULT_NOISE_FIGURE_DB),
                    "noise_bandwidth_hz": noise_bandwidth_hz(self.cfg),
                    "thermal_noise_w": thermal_noise_power_w(self.cfg),
                    "seed": self.seed + self._frame_idx - 1,
                }}
