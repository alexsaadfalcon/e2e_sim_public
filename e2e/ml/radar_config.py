"""
FMCW MIMO radar configuration for the ML radar-dataset package.

Describes the chirp/frame timing of an FMCW MIMO radar and derives the standard
range/velocity resolution and limits from it. This is the **shared contract**
between the scene/scatterer generator, the range-Doppler synthesizer, and the
dataset transforms in this package (`e2e/ml/scatterers.py`, `rd_synth.py`,
`transforms.py`).

Nothing here imports torch or numpy -- like `e2e/scenario.py`, this module is
intentionally dependency-free (stdlib only) so it can be built/validated/
serialized anywhere, independent of the heavy synthesis code.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

# Speed of light, m/s.
C_MPS = 299_792_458.0

_VALID_MIMO = ("tdm", "ddma", "single")


@dataclass(frozen=True)
class RadarConfig:
    """FMCW MIMO radar chirp/frame timing.

    All fields describe one frame's worth of chirps. `mimo` selects how the
    `n_tx` transmitters share the array:
      * "tdm"    -- transmitters fire in round-robin (Time-Division Multiplexing);
                    `n_chirps` chirps are split evenly across `n_tx` TX antennas,
                    so the *per-TX* slow-time sampling rate is `n_tx` times
                    slower than the chirp rate -- see `max_velocity_mps`.
      * "ddma"   -- transmitters fire simultaneously with a per-TX Doppler-shift
                    encoding (Doppler-Division Multiplexing, e.g. RADIal); every
                    chirp illuminates with all TX at once. The full PRF is kept,
                    but the Doppler spectrum is code-divided into `n_tx` sub-
                    bands, so the *unambiguous* velocity span shrinks by `n_tx`
                    exactly as it does for TDM -- see `max_velocity_mps`.
      * "single" -- a single TX antenna (no MIMO); `n_tx` is 1 by convention.
    """

    name: str
    f0_hz: float            # chirp start frequency
    bandwidth_hz: float     # swept bandwidth over the sampled window
    n_tx: int
    n_rx: int
    n_chirps: int           # chirps per frame (total across TDM rounds)
    n_samples: int          # ADC samples per chirp
    fs_hz: float            # ADC sample rate
    chirp_period_s: float   # chirp-to-chirp period (ramp + idle)
    mimo: str = "tdm"       # "tdm" | "ddma" | "single"
    frame_rate_hz: float = 10.0

    # ---- link budget --------------------------------------------------------
    # These three turn an arbitrary-scale cube into absolute power, and they are
    # consequently THE DIFFICULTY DIAL for every generated corpus: they set the absolute
    # SNR of every target at every range. See `e2e.ml.link_budget` for the derivation and
    # for why the chain had no absolute reference at all before them
    # (notes/ESTABLISHED_FACTS.md F35/F42).
    #
    # Datasheet-plausible for a TI IWR1443-class 77 GHz automotive MMIC, reviewed by an
    # independent RF pass, NOT measured from a specific part. Do not tune them to make a
    # detector's curves look better -- that is the failure mode they exist to end.
    tx_power_dbm: float = 12.0        # per transmit channel
    noise_figure_db: float = 15.0     # receiver, referenced at the antenna input
    temperature_k: float = 290.0      # IEEE noise-figure reference, a definition

    def __post_init__(self):
        # Normalize the mimo tag once, at construction. rd_synth lower-cases its
        # local copy, but the derived properties below compare exact strings; a
        # case-mismatched tag (e.g. "TDM") would previously synthesize correctly
        # while silently mis-computing n_chirps_per_tx -- and with it the noise
        # coherent gain (a 10*log10(n_tx) dB SNR calibration error).
        object.__setattr__(self, "mimo", str(self.mimo).lower())

    # ---- derived, read-only ------------------------------------------------
    @property
    def n_virtual(self) -> int:
        """Virtual (TX, RX) MIMO array size."""
        return self.n_tx * self.n_rx

    @property
    def sweep_time_s(self) -> float:
        """Time spent sampling the ADC ramp (excludes idle time)."""
        return self.n_samples / self.fs_hz

    @property
    def ramp_slope_hzps(self) -> float:
        """Chirp slope, Hz/s: swept bandwidth over the sampled ramp time."""
        return self.bandwidth_hz / self.sweep_time_s

    @property
    def wavelength_m(self) -> float:
        """Wavelength at the chirp's center frequency (f0 + B/2)."""
        f_center = self.f0_hz + self.bandwidth_hz / 2.0
        return C_MPS / f_center

    @property
    def range_resolution_m(self) -> float:
        """Standard FMCW range resolution: c / (2 * swept bandwidth)."""
        return C_MPS / (2.0 * self.bandwidth_hz)

    @property
    def max_range_m(self) -> float:
        """Unambiguous max range from the ADC sample rate and ramp slope.

        max_range = fs * c / (2 * slope). Substituting slope = B*fs/n_samples
        (from `ramp_slope_hzps`) shows this is algebraically identical to
        `n_samples * range_resolution_m` -- fs cancels out, so (as expected)
        max range depends only on bandwidth and sample count, not on how fast
        the ADC runs. Both forms are verified equal in the test suite.
        """
        return self.fs_hz * C_MPS / (2.0 * self.ramp_slope_hzps)

    @property
    def n_chirps_per_tx(self) -> int:
        """Chirps illuminated by any single TX antenna.

        Only TDM divides the frame's chirps across TX antennas in round-robin;
        DDMA and single-TX illuminate every chirp with all (or the one) TX.
        """
        if self.mimo == "tdm":
            return self.n_chirps // self.n_tx
        return self.n_chirps

    @property
    def _n_tx_eff(self) -> int:
        """Number of TX antennas that divide up the unambiguous Doppler span.

        Both MIMO schemes pay this penalty, through different mechanisms:
        TDM slows the per-TX slow-time sample rate to 1/(n_tx*T_c); DDMA keeps
        the full PRF but code-multiplexes the TX replicas into n_tx equal
        Doppler sub-bands, so a target's Doppler is only unambiguous within
        1/n_tx of the full span (see rd_synth's DDMA comment: the replicas of
        a single target sit n_chirps/n_tx bins apart, and shifting the true
        Doppler by that spacing reproduces the identical replica set).
        Single-TX radars pay nothing.
        """
        return self.n_tx if self.mimo in ("tdm", "ddma") else 1

    @property
    def max_velocity_mps(self) -> float:
        """Unambiguous max radial velocity (+-v_max), standard Doppler limit.

        v_max = lambda / (4 * T_slow_eff), with T_slow_eff = _n_tx_eff *
        chirp_period_s. For TDM, a given TX only re-illuminates a target every
        `n_tx` chirps, so the per-TX PRF (and hence v_max) is divided by n_tx.
        For DDMA the mechanism differs -- the full PRF is sampled, but the
        spectrum is code-divided into n_tx replica sub-bands -- yet the
        unambiguous span is divided by n_tx all the same (a target's Doppler
        shifted by one sub-band width reproduces the identical replica set).
        Either way, MIMO buys n_tx times more virtual channels (angular
        resolution) at the cost of n_tx times less unambiguous velocity;
        only "single" escapes the trade.
        """
        t_slow = self._n_tx_eff * self.chirp_period_s
        return self.wavelength_m / (4.0 * t_slow)

    @property
    def velocity_resolution_mps(self) -> float:
        """Doppler (velocity) resolution: lambda / (2 * CPI length).

        The coherent processing interval (CPI) spans all `n_chirps` in the
        frame regardless of MIMO scheme -- TDM's PRF penalty shrinks the
        *unambiguous* velocity (see `max_velocity_mps`) but the Doppler FFT
        still integrates over the full frame, so resolution is unaffected by
        `mimo`.
        """
        return self.wavelength_m / (2.0 * self.n_chirps * self.chirp_period_s)

    # ---- validation ---------------------------------------------------------
    def validate(self) -> List[str]:
        """Return a list of human-readable problems; empty list == valid."""
        problems: List[str] = []
        if self.mimo not in _VALID_MIMO:
            problems.append(
                f"mimo must be one of {_VALID_MIMO}, got '{self.mimo}'"
            )
        if self.f0_hz <= 0:
            problems.append("f0_hz must be > 0")
        if self.bandwidth_hz <= 0:
            problems.append("bandwidth_hz must be > 0")
        if self.n_tx < 1:
            problems.append("n_tx must be >= 1")
        if self.n_rx < 1:
            problems.append("n_rx must be >= 1")
        if self.n_chirps < 1:
            problems.append("n_chirps must be >= 1")
        if self.n_samples < 1:
            problems.append("n_samples must be >= 1")
        if self.fs_hz <= 0:
            problems.append("fs_hz must be > 0")
        if self.chirp_period_s <= 0:
            problems.append("chirp_period_s must be > 0")
        if self.frame_rate_hz <= 0:
            problems.append("frame_rate_hz must be > 0")
        if self.mimo == "tdm" and self.n_tx >= 1 and self.n_chirps % self.n_tx != 0:
            problems.append(
                f"n_chirps ({self.n_chirps}) must be divisible by n_tx "
                f"({self.n_tx}) for TDM MIMO"
            )
        if self.fs_hz > 0 and self.sweep_time_s > self.chirp_period_s:
            problems.append(
                f"sweep_time_s ({self.sweep_time_s:.3g}s) exceeds "
                f"chirp_period_s ({self.chirp_period_s:.3g}s): the ramp does "
                f"not fit within one chirp period"
            )
        return problems

    # ---- (de)serialization ---------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RadarConfig":
        return cls(**d)


def answerability_problems(cfg: "RadarConfig", *, top_speed_mps: float,
                           max_sin_az_err: Optional[float] = None) -> List[str]:
    """Can a detection benchmark on `cfg` be ANSWERED, for scenes drawing speeds up to
    `top_speed_mps`? Returns human-readable problems (empty == answerable), in the same
    style as `RadarConfig.validate`.

    Both failure modes are F43, measured on shipped presets before this guard existed
    (see `BENCHMARK_V1`'s comment block): every detection number this project produced
    before 2026-08-18 sat on a config that failed one of them.

    * **Doppler**: a target's worst-case radial speed is its speed magnitude (even a
      sampler that clamps the radial component at frame 0 cannot hold the clamp over a
      multi-frame track -- the line of sight rotates). If `top_speed_mps` exceeds
      `cfg.max_velocity_mps`, targets alias in Doppler, landing anywhere on the axis
      including zero -- which destroys the zero-Doppler clutter discriminant. The
      check sits at the PHYSICAL ambiguity limit; samplers wanting margin against
      edge smearing should keep their own headroom (the analytic sampler clamps at
      0.8 * v_max -- `e2e.ml.scenes._RADIAL_VELOCITY_FRAC`), and all shipped
      (preset, tier) pairs that pass this guard do so with >15% headroom anyway.
    * **Azimuth**: if the scoring tolerance `max_sin_az_err` is finer than the array's
      Rayleigh limit `2 / n_virtual`, the metric demands better localization than the
      aperture physically resolves -- a perfect detector scores as a miss. The default
      tolerance is read from `e2e.ml.metrics.MatchCriterion` at call time (one source
      of truth, imported lazily to keep this module dependency-light).
    """
    if max_sin_az_err is None:
        from e2e.ml.metrics import MatchCriterion
        max_sin_az_err = MatchCriterion().max_sin_az_err
    problems: List[str] = []
    if float(top_speed_mps) > cfg.max_velocity_mps:
        problems.append(
            f"scene speeds up to {float(top_speed_mps):.2f} m/s exceed the unambiguous "
            f"v_max of +-{cfg.max_velocity_mps:.2f} m/s ({cfg.mimo} over "
            f"{cfg.n_tx} TX): targets will alias in Doppler (F43)"
        )
    rayleigh = 2.0 / float(cfg.n_virtual)
    if float(max_sin_az_err) < rayleigh:
        problems.append(
            f"match tolerance max_sin_az_err={float(max_sin_az_err):.4f} is finer than "
            f"the array's Rayleigh limit 2/n_virtual={rayleigh:.4f} "
            f"({cfg.n_virtual} virtual elements): the azimuth question is "
            f"unanswerable (F43)"
        )
    return problems


# --------------------------------------------------------------------------------
# Reference presets.
# --------------------------------------------------------------------------------

# TI IWR1443BOOST-like mid-range profile (76-81 GHz band, 3TX/4RX TDM-MIMO ->
# 12 virtual channels). Parameters chosen to be within the device's real
# operating envelope (IWR1443 supports up to ~4 GHz swept bandwidth, ~100
# MHz/us ramp slope, and multi-Msps ADC rates) AND to give a scene scale that
# fits vehicle/pedestrian training scenarios:
#   - bandwidth_hz=2e9 -> range_resolution_m = c/(2B) ~= 7.5 cm.
#     REVIEWED AND CONFIRMED 2026-08-17. The IWR1443 datasheet says "up to 4 GHz", and
#     the question of moving there was raised explicitly. It was measured and declined,
#     because bandwidth is not a free parameter here -- max_range = n_samples * c/(2B),
#     so at fixed n_samples doubling B HALVES the range window:
#         2 GHz, 512 samples  -> 7.5 cm res, 38.4 m range,  97.7 MHz/us, 12.8 m/s max vel
#         4 GHz, 512 samples  -> 3.7 cm res, 19.2 m range, 195.3 MHz/us, 12.7 m/s
#         4 GHz, 1024 samples -> 3.7 cm res, 38.4 m range,  97.7 MHz/us,  7.0 m/s
#     The middle row is unusable twice over: D1 targets sit at 15-32 m and would fall
#     outside a 19.2 m window, and 195 MHz/us is ~2x the device's ~100 MHz/us ramp class,
#     so it would no longer be an IWR1443-like profile at all -- which was the entire
#     reason for wanting 4 GHz. The bottom row keeps the range but halves the unambiguous
#     velocity (targets are clamped to 80% of it, i.e. 5.6 m/s radial -- slow for vehicle
#     scenes) and doubles every stored cube. 2 GHz is already inside the real device
#     envelope; "up to 4 GHz" is a maximum, not a requirement.
#   - n_samples=512 @ fs_hz=25e6 -> sweep_time_s = 20.48 us, inside
#     chirp_period_s=25e-6 (4.5 us idle time for TX/RX settling).
#   - ramp_slope = B/sweep_time ~= 97.7 MHz/us, at (but within) the device's
#     ~100 MHz/us ramp-slope class.
#   - max_range_m = n_samples * range_resolution_m ~= 38.4 m: room for
#     multi-vehicle urban scenes (the earlier 3.6 GHz/256-sample draft gave a
#     10.7 m max range -- too cramped for vehicle scenarios).
#   - TDM max velocity = lambda/(4*n_tx*T_c) ~= 12.8 m/s: covers pedestrians
#     and urban vehicle speeds; faster targets alias (documented behavior).
#   - n_chirps=192 chirps/frame, divided round-robin across 3 TX -> 64
#     chirps/TX; TDM slow-time period = n_tx*chirp_period_s = 75 us.
TI_IWR1443 = RadarConfig(
    name="ti_iwr1443",
    f0_hz=77e9,
    bandwidth_hz=2e9,
    n_tx=3,
    n_rx=4,
    n_chirps=192,
    n_samples=512,
    fs_hz=25e6,
    chirp_period_s=25e-6,
    mimo="tdm",
)

# RADIal-paper-like HD radar (Rebut et al., "Raw High-Definition Radar for
# Multi-Task Learning"): 12 TX / 16 RX simultaneous-DDMA MIMO (192 virtual
# channels), 512 range bins x 256 Doppler bins per frame. The paper (Table 5,
# Sec 5/Appendix A) states only the *resolution/FOV* targets, not RF chirp
# parameters (no carrier/bandwidth/ADC-rate numbers are given anywhere in the
# text) -- f0_hz/fs_hz/chirp_period_s below are chosen/solved to reproduce the
# paper's published numbers, not sourced from it:
#   - range_resolution_m = c/(2B) = 0.2 m -> bandwidth_hz = c/(2*0.2) ~= 750
#     MHz; we use the paper's own value bandwidth_hz=749.5e6 (given).
#   - max_range_m = n_samples*range_resolution_m = 512*0.2 ~= 102.4 m, matches
#     the paper's stated [0, 103] m FOV.
#   - fs_hz=10e6 is an arbitrary-but-realistic automotive ADC rate; max_range
#     is independent of fs_hz (it cancels algebraically, see
#     `max_range_m`'s docstring), so any fs giving sweep_time < chirp_period
#     is consistent with the paper's numbers. sweep_time = 512/10e6 = 51.2 us.
#   - n_chirps=252, NOT the paper's 256: 252 is divisible by n_tx=12, so the
#     DDMA replicas land 252/12 = 21 Doppler bins apart EXACTLY. With 256 the
#     spacing is fractional (256/12 = 21.33): the replicas smear across bins
#     (spectral leakage) and no uniformly-dilated demux (FFTRadNet's MIMO
#     pre-encoder included) can sample all 12 replicas correctly -- the last
#     tap misses by 4 full bins. The paper's radar presumably uses phase
#     increments tuned to its exact chirp count; for OUR synthesizer the
#     divisible count is the physically clean choice. Resolution impact is
#     ~1.6% (vel res 0.1012 vs 0.0996 m/s), still within 5% of the paper.
#   - chirp_period_s=76e-6 is solved so velocity_resolution_mps matches the
#     paper's stated 0.1 m/s to within ~2%: T = lambda / (2*n_chirps*0.1)
#     at f0=77 GHz (carrier assumed, not stated in the paper); sweep_time
#     (51.2 us) still comfortably fits inside this period.
#   - max_velocity_mps (DDMA, n_tx_eff=12) = lambda/(4*12*chirp_period_s)
#     ~= 1.06 m/s: the DDMA code-division shrinks the unambiguous span by
#     n_tx (the paper mentions Dmax but never gives a number; it likely
#     recovers a wider span with downstream Doppler unfolding, which this
#     package does not model -- training data must stay under THIS limit to
#     be alias-free).
RADIAL_LIKE = RadarConfig(
    name="radial_like",
    f0_hz=77e9,
    bandwidth_hz=749.5e6,
    n_tx=12,
    n_rx=16,
    n_chirps=252,
    n_samples=512,
    fs_hz=10e6,
    chirp_period_s=76e-6,
    mimo="ddma",
)

# --------------------------------------------------------------------------------
# BENCHMARK_V1 -- the first configuration on which a detection benchmark is VALID.
# --------------------------------------------------------------------------------
# Both older presets are broken for detection scoring, in exactly opposite ways, and every
# detection number produced by this project before 2026-08-18 sits on one of them:
#
#   ti_iwr1443:  12 virtual elements -> Rayleigh sin(az) = 0.1667, against a match
#                tolerance of 0.06. The metric demands 2.8x finer azimuth than the array
#                can physically resolve, so a PERFECT detector scores as a miss.
#   radial_like: 192 virtual elements resolves azimuth beautifully, but DDMA over 12 TX
#                shrinks the unambiguous velocity span to +-1.06 m/s while the scene
#                sampler draws targets at 0-8 m/s. Every moving target ALIASES in Doppler.
#                (RADIAL_LIKE's own comment above says "training data must stay under THIS
#                limit to be alias-free" -- the corpora violate it.) Aliasing lands targets
#                anywhere on the Doppler axis including zero, which destroys the strongest
#                clutter discriminant an automotive radar has: MEASURED, a zero-Doppler
#                notch removes the TARGETS (Pd 0.098 -> 0.000) rather than the clutter.
#
# This preset satisfies both constraints at once, and nothing else about it is exotic:
#
#   answerable azimuth : 4 x 16 = 64 virtual -> Rayleigh 2/64 = 0.0312, i.e. 1.9x finer
#                        than the 0.06 match tolerance, so the metric can be satisfied.
#   unaliased Doppler  : TDM over 4 TX at a 25 us chirp period gives a per-TX PRI of
#                        100 us, so v_max = lambda/(4*n_tx*T_c) = 9.69 m/s -- above the
#                        8 m/s the scene sampler draws.
#   physically feasible: sweep time = 512/25 MHz = 20.48 us, inside the 25 us chirp period.
#   R_max 102.4 m at 20 cm range resolution (B = 749.5 MHz, as radial_like).
#
# TDM rather than DDMA on purpose: DDMA divides the unambiguous Doppler span by n_tx, which
# is precisely what broke radial_like. With only 4 TX, TDM's cost (a 1/4 duty cycle per
# transmitter) is the cheaper trade.
BENCHMARK_V1 = RadarConfig(
    name="benchmark_v1",
    f0_hz=77e9,
    bandwidth_hz=749.5e6,
    n_tx=4,
    n_rx=16,
    n_chirps=256,
    n_samples=512,
    fs_hz=25e6,
    chirp_period_s=25e-6,
    mimo="tdm",
)

PRESETS = {"ti_iwr1443": TI_IWR1443, "radial_like": RADIAL_LIKE,
           "benchmark_v1": BENCHMARK_V1}


if __name__ == "__main__":
    for name, cfg in PRESETS.items():
        problems = cfg.validate()
        status = "OK" if not problems else f"PROBLEMS: {problems}"
        print(
            f"[{name}] n_virtual={cfg.n_virtual} range_res={cfg.range_resolution_m:.4f}m "
            f"max_range={cfg.max_range_m:.2f}m vel_res={cfg.velocity_resolution_mps:.4f}m/s "
            f"max_vel={cfg.max_velocity_mps:.2f}m/s -> {status}"
        )
        assert RadarConfig.from_dict(cfg.to_dict()) == cfg, "round-trip failed"
    print("radar_config round-trip OK")
