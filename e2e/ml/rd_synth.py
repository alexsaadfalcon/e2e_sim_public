"""
Analytic raw-ADC synthesizer for an FMCW MIMO radar.

Given a `RadarConfig` (waveform + array), a list of point `Scatterer`s and a
`RadarPose`, `synthesize_adc` returns the *dechirped* (beat) baseband signal that
an ADC behind each RX channel would record:

    adc[r, c, n]   r = RX channel, c = chirp index, n = fast-time sample

This is the classical stop-and-hop FMCW point-target model -- the same model that
underpins every range-Doppler textbook -- evaluated in closed form rather than by
ray tracing. It exists so the ML package can generate large, perfectly labelled
range-Doppler datasets in milliseconds; the Sionna path in `e2e/environment/`
remains the high-fidelity (but slow, offline) alternative.

Model summary (all equations are repeated at the point of use below)
--------------------------------------------------------------------
Transmitted ramp     s_t(t) = exp(j2pi (f0 t + S t^2 / 2)),  S = ramp slope [Hz/s]
Echo from range R    s_r(t) = s_t(t - tau),                  tau = 2R/c
Dechirp (mixer)      s_b(t) = s_t(t) conj(s_r(t))
                            = exp(j2pi [ f0 tau + S tau t - S tau^2 / 2 ])
so the beat tone sits at f_b = S tau = 2 R S / c and its *phase* f0 tau carries the
Doppler information from chirp to chirp. We adopt the **positive-exponent**
convention throughout: increasing range -> increasing beat phase.

Scope / explicitly out of scope
-------------------------------
* Far-field (plane-wave) array response: the per-element path-length difference is
  linearised as d sin(theta). Valid for R >> aperture^2 / lambda, which holds for
  the automotive-scale geometries this package targets.
* Isotropic elements: no antenna pattern, no mutual coupling, no occlusion and no
  multi-bounce. Field-of-view culling (e.g. dropping scatterers behind the array)
  is the caller's job.
* Amplitudes are *relative*: A = sqrt(sigma) / R^2 reproduces the radar-equation
  range law and RCS scaling, but absolute (calibrated) receive power -- Pt, Gt, Gr,
  lambda^2/(4pi)^3, receiver gain -- is out of scope. SNR is set explicitly instead.
  The amplitude is evaluated once at the chirp-0 range and held constant across the
  frame (the phase DOES track the per-chirp range walk): a fast, close target can
  change range by a few percent within one CPI (~5-10% amplitude error at 2 m and
  20 m/s), which is accepted as out of scope alongside the other amplitude idealisms.
* Single elevation cut: a ULA measures only the direction cosine along its axis, so
  an elevated target is indistinguishable from a coplanar one at the same cosine.

All tensors are torch complex64 on the shared `device` (cuda if available).
"""

import math

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

C_LIGHT = 299_792_458.0

# Scatterers are processed in chunks so a scene with thousands of points does not
# materialise a [K, n_chirps, n_samples] complex128 phase tensor all at once.
_SCATTERER_CHUNK = 64


# --------------------------------------------------------------------------------
# Scene frame / radar pose
# --------------------------------------------------------------------------------
# The canonical RadarPose lives in the torch-free `scatterers` module (the scene
# side of the package); it is re-exported here so synthesis-side callers can keep
# importing it from `rd_synth`. Scene-frame convention (right-handed, +z world up,
# ULA along u = normalise(z_up x boresight)) is documented on the class itself.
from e2e.ml.scatterers import RadarPose  # noqa: E402  (re-export)


def array_axis(pose):
    """Unit vector along the ULA axis for `pose` (see `RadarPose` for the convention).

    Degenerate case: if the boresight is (anti)parallel to world up there is no
    unique horizontal lateral axis, and we fall back to +x.
    """
    b = np.asarray(pose.boresight, dtype=np.float64).reshape(3)
    nb = np.linalg.norm(b)
    if nb < 1e-12:
        raise ValueError("RadarPose.boresight must be a non-zero vector")
    b = b / nb
    u = np.cross(np.array([0.0, 0.0, 1.0]), b)          # z_up x boresight
    nu = np.linalg.norm(u)
    if nu < 1e-9:                                       # boresight is vertical
        return np.array([1.0, 0.0, 0.0])
    return u / nu


# --------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------
def _resolve_device(dev):
    """`None` -> the library device; anything else -> `torch.device(dev)`."""
    return device if dev is None else torch.device(dev)


def _unpack_scatterers(scatterers):
    """Stack scatterer attributes into float64 arrays: (positions, velocities, rcs_dbsm)."""
    k = len(scatterers)
    if k == 0:
        return (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0,)))
    pos = np.stack([np.asarray(s.position, dtype=np.float64).reshape(3) for s in scatterers])
    vel = np.stack([np.asarray(getattr(s, "velocity", (0.0, 0.0, 0.0)),
                               dtype=np.float64).reshape(3) for s in scatterers])
    rcs = np.array([float(getattr(s, "rcs_dbsm", 0.0)) for s in scatterers], dtype=np.float64)
    return pos, vel, rcs


def _chirps_per_tx(cfg, mimo, n_chirps, n_tx):
    """Number of chirps a single TX contributes to the coherent Doppler FFT."""
    n = getattr(cfg, "n_chirps_per_tx", None)
    if n is not None:
        return int(n)
    return n_chirps // n_tx if mimo == "tdm" else n_chirps


# --------------------------------------------------------------------------------
# Raw-ADC synthesis
# --------------------------------------------------------------------------------
def synthesize_adc(cfg, scatterers, radar_pose=None, *, snr_db=30.0, seed=None,
                   device=None, random_phase=True, include_rvp=False):
    """Synthesize the raw (dechirped) ADC cube for one radar frame.

    Parameters
    ----------
    cfg : RadarConfig
        Waveform/array description. Fields used: `f0_hz`, `n_tx`, `n_rx`, `n_chirps`,
        `n_samples`, `fs_hz`, `chirp_period_s`, `mimo`, `wavelength_m`,
        `ramp_slope_hzps`, `n_chirps_per_tx`.
    scatterers : sequence of Scatterer
        Point targets with `.position` (m, scene frame), `.velocity` (m/s),
        `.rcs_dbsm`. An empty sequence gives a signal-free (noise-only) frame.
    radar_pose : RadarPose, optional
        Defaults to the origin looking along +x.
    snr_db : float or None
        Post-2D-FFT SNR of the *strongest* scatterer (see the noise section below).
        `None` disables noise entirely.
    seed : int or None
        Seeds the per-scatterer random phases and the AWGN. Determinism is
        per-device: CPU and CUDA generators produce different streams for the same
        seed (a torch property), but repeated calls on one device match exactly.
    device : torch device or None
        Defaults to the library device.
    random_phase : bool
        Give each scatterer an i.i.d. uniform reflection phase (the usual model for
        independent scattering centres). Constant over the frame, so it does not
        perturb range/Doppler/angle -- only the absolute phase.
    include_rvp : bool
        Include the residual video phase term -pi S tau^2 of the exact dechirp
        product. It is a per-chirp constant (no effect on the beat frequency) and is
        ~0.02 cycles at 10 m for a 78 THz/s ramp, so it is dropped by default; turn
        it on to model long-range phase bias exactly.

    Returns
    -------
    torch.Tensor, complex64, shape [n_rx, n_chirps, n_samples], on `device`.
    """
    dev = _resolve_device(device)

    n_rx = int(cfg.n_rx)
    n_tx = int(cfg.n_tx)
    n_chirps = int(cfg.n_chirps)
    n_samples = int(cfg.n_samples)
    fs = float(cfg.fs_hz)
    f0 = float(cfg.f0_hz)
    slope = float(cfg.ramp_slope_hzps)
    t_chirp = float(cfg.chirp_period_s)
    lam = float(cfg.wavelength_m)
    mimo = str(cfg.mimo).lower()

    if mimo not in ("tdm", "ddma", "single"):
        raise ValueError(f"unsupported mimo scheme {cfg.mimo!r}")
    if mimo == "single" and n_tx != 1:
        raise ValueError(f"mimo='single' requires n_tx == 1 (got {n_tx})")
    n_chirps_per_tx = _chirps_per_tx(cfg, mimo, n_chirps, n_tx)

    pose = radar_pose if radar_pose is not None else RadarPose()
    origin = np.asarray(pose.position, dtype=np.float64).reshape(3)
    u_ax = array_axis(pose)

    # ---------------------------------------------------------------- geometry
    pos, vel, rcs_dbsm = _unpack_scatterers(scatterers)
    los = pos - origin[None, :]                             # radar -> scatterer, [K,3]
    r0 = np.linalg.norm(los, axis=1)                        # range at chirp 0, [K]
    keep = r0 > 1e-6                                        # a target at the phase centre is meaningless
    los, vel, rcs_dbsm, r0 = los[keep], vel[keep], rcs_dbsm[keep], r0[keep]
    n_scat = r0.size

    if n_scat:
        e_los = los / r0[:, None]                           # unit LOS vectors, [K,3]
        # Radial velocity = projection of the velocity on the LOS. Positive = receding
        # (range increasing), which is the sign that drives the Doppler phase below.
        v_r = np.einsum("kd,kd->k", vel, e_los)
        # A ULA measures the direction cosine along its axis; for a target in the
        # array's horizontal plane this is exactly sin(azimuth from boresight).
        sin_th = e_los @ u_ax
        # Radar equation range law + RCS: A = sqrt(10^(rcs/10)) / R^2 (relative scale).
        amp = np.sqrt(10.0 ** (rcs_dbsm / 10.0)) / r0 ** 2
    else:
        v_r = np.zeros(0)
        sin_th = np.zeros(0)
        amp = np.zeros(0)

    # ---------------------------------------------------------------- RNG
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed) if seed is not None
                    else int(torch.randint(0, 2 ** 62, (1,)).item()))

    if n_scat and random_phase:
        phi0 = torch.rand(n_scat, generator=gen, device=dev, dtype=torch.float64) * (2 * math.pi)
    else:
        phi0 = torch.zeros(n_scat, device=dev, dtype=torch.float64)

    # ---------------------------------------------------------------- signal
    adc = torch.zeros((n_rx, n_chirps, n_samples), dtype=torch.complex64, device=dev)

    if n_scat:
        # Chirp c is emitted at t = c * chirp_period_s. Range advances within the
        # frame, R_k(c) = R_k0 + v_r,k * c * T_c; this linear range walk is what
        # turns into the chirp-to-chirp (Doppler) phase progression.
        c_idx = torch.arange(n_chirps, dtype=torch.float64, device=dev)
        n_idx = torch.arange(n_samples, dtype=torch.float64, device=dev)

        r0_t = torch.as_tensor(r0, dtype=torch.float64, device=dev)
        vr_t = torch.as_tensor(v_r, dtype=torch.float64, device=dev)
        sin_t = torch.as_tensor(sin_th, dtype=torch.float64, device=dev)
        amp_t = torch.as_tensor(amp, dtype=torch.float64, device=dev)

        rng_kc = r0_t[:, None] + vr_t[:, None] * (c_idx[None, :] * t_chirp)   # [K,C]
        tau_kc = 2.0 * rng_kc / C_LIGHT                                        # round-trip delay

        # Spatial phase. RX r sits at d_rx = r * lambda/2, TX t at d_tx = t * n_rx * lambda/2,
        # so the pair (t, r) has total offset (t*n_rx + r) * lambda/2 -- a uniform
        # lambda/2 virtual ULA with element index v = t*n_rx + r. The far-field two-way
        # phase is 2pi (d_tx + d_rx) sin(theta)/lambda = pi * v * sin(theta).
        rx_idx = torch.arange(n_rx, dtype=torch.float64, device=dev)
        rx_phase = math.pi * rx_idx[None, :] * sin_t[:, None]                  # [K,R]
        e_rx = torch.polar(torch.ones_like(rx_phase), rx_phase).to(torch.complex64)

        # Per-chirp TX factor.
        if mimo == "ddma":
            # Doppler-division: every TX transmits on every chirp, TX t carrying the
            # extra per-chirp phase 2pi t c / n_tx. Aliasing signature: TX t's echo is
            # shifted by t/n_tx of the Doppler PRF, so a single target appears as n_tx
            # Doppler peaks spaced n_chirps/n_tx bins apart (that is how the TXs are
            # demultiplexed), and the unambiguous Doppler span shrinks by n_tx.
            tx_idx = torch.arange(n_tx, dtype=torch.float64, device=dev)
            ph_tx = (math.pi * n_rx * tx_idx[None, :, None] * sin_t[:, None, None]
                     + 2 * math.pi * tx_idx[None, :, None] * c_idx[None, None, :] / n_tx)
            tx_factor = torch.polar(torch.ones_like(ph_tx), ph_tx).sum(dim=1)   # [K,C]
        else:
            # TDM (and the degenerate single-TX case): chirp c is transmitted by
            # TX (c mod n_tx) alone, so only that TX's spatial phase appears.
            tx_of_chirp = torch.arange(n_chirps, dtype=torch.float64, device=dev) % n_tx
            ph_tx = math.pi * n_rx * tx_of_chirp[None, :] * sin_t[:, None]      # [K,C]
            tx_factor = torch.polar(torch.ones_like(ph_tx), ph_tx)

        # Combine everything that does not depend on the fast-time sample index.
        gain_kc = (amp_t[:, None].to(torch.complex128)
                   * torch.polar(torch.ones_like(phi0), phi0)[:, None]
                   * tx_factor)                                                # [K,C]

        for lo in range(0, n_scat, _SCATTERER_CHUNK):
            hi = min(lo + _SCATTERER_CHUNK, n_scat)
            tau = tau_kc[lo:hi]                                                # [k,C]
            # Dechirped beat phase, phi(n,c) = 2pi [ S tau n/fs + f0 tau ] (+ RVP).
            # Kept in float64 until the wrap: 2pi f0 tau is O(1e5) rad, which float32
            # cannot represent finely enough to preserve the Doppler phase steps.
            ph = (2 * math.pi * slope / fs) * tau[:, :, None] * n_idx[None, None, :]
            ph = ph + (2 * math.pi * f0) * tau[:, :, None]
            if include_rvp:
                ph = ph - math.pi * slope * (tau[:, :, None] ** 2)
            ph = torch.remainder(ph, 2 * math.pi)

            sig = torch.polar(torch.ones_like(ph), ph) * gain_kc[lo:hi][:, :, None]
            # sum_k e_rx[k,r] * sig[k,c,n] -> [R,C,N]
            adc = adc + torch.einsum("kr,kcn->rcn", e_rx[lo:hi],
                                     sig.to(torch.complex64))

    # ---------------------------------------------------------------- noise
    # Convention: `snr_db` is the SNR of the strongest scatterer *at its peak* in an
    # unwindowed 2-D (fast-time x slow-time) FFT of a single RX channel, using the
    # chirps of one TX. That FFT has coherent gain G = n_samples * n_chirps_per_tx:
    #   peak signal power = (A_max G)^2 = A_max^2 G^2,  noise power/bin = sigma^2 G
    #   => SNR_peak = A_max^2 G / sigma^2  =>  sigma^2 = A_max^2 G / 10^(snr_db/10)
    # sigma^2 is the total (I+Q) per-sample noise power. No RX beamforming gain and
    # no window loss are included -- both are processing choices made downstream.
    if snr_db is not None:
        a_max = float(amp.max()) if n_scat else 0.0
        coh_gain = float(n_samples * n_chirps_per_tx)
        # An empty/zero scene has no reference amplitude; emit unit-variance noise so
        # the frame is still a usable "background only" sample.
        sigma2 = (a_max ** 2 * coh_gain / (10.0 ** (float(snr_db) / 10.0))) if a_max > 0 else 1.0
        w = torch.randn((n_rx, n_chirps, n_samples, 2), generator=gen,
                        device=dev, dtype=torch.float32) * math.sqrt(sigma2 / 2.0)
        adc = adc + torch.view_as_complex(w.contiguous())

    return adc.to(torch.complex64)
