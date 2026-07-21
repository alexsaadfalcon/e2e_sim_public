import numpy as np
import torch

# Single source of truth for the system reference impedance: the generation layer
# converts tx_power_dbm to volts against this same constant, so the circuit model's
# source impedance must not drift from it independently.
from e2e.scenario import SYSTEM_IMPEDANCE_OHMS


# kT at T=290K (standard noise-reference temperature); the thermal-noise voltage PSD
# is 4kT*R, not 4e-21*R (which is kT, off by 4x).
FOURKT = 4 * 1.380649e-23 * 290  # ~1.601e-20


'''
Todo: Convert the RX config to Json dict
'''
def get_RX_config(nRx):
    # // https://www.ti.com/lit/an/swra553a/swra553a.pdf?ts=1723503949605 Pg 7
    # Define other simulation parameters

    # Placeholder values for other configurations (config.)
    # Ibias_LNA = torch.full((nRx,), 1) # mA --> A from Datasheet
    #from the email ~500 uA – 10 mA.
    Ibias_LNA = torch.full((nRx,), 8e-3) # Retuned to 8 mA (within 500 uA - 10 mA range)

    # Vbias_LNA = torch.full((nRx,), 1.3) # V from datasheet
    #from the email 50 mV – 200 mV
    Vbias_LNA = torch.full((nRx,), 0.1) # V Let's take 100mV

    #considering 30mW
    Pdc_mix = torch.full((nRx,), 20e-3) ################  Mixer Power? (Estimated from passive mixers (published) with different nodes)
    
    '''
        Mixer Power
            1. What do we take as the correct values
    '''
    # Ibias_BB = torch.full((nRx,), 850e-3) # mA --> A  from datasheet
    #from email
    Ibias_BB = torch.full((nRx,),5e-3) # mA --> A Let's take 5 mA

    # Vbias_BB = torch.full((nRx,), 2) # V from datasheet
    # from email
    Vbias_BB = torch.full((nRx,), 0.2) # V from datasheet lets take 200mV


    # Total chain VOLTAGE gain: 24 dB -> linear 10^(24/20) ~= 15.85. The chain is
    # specified in voltage gains throughout (AvLNA = Gm*Ro, Avmix, and AvBB is
    # back-derived as Av/(AvLNA*Avmix)), so the dB value converts with /20.
    Av_dB = 24
    Av = torch.full((nRx,), 10 ** (Av_dB / 20))
    # Plain 15 MHz IF bandwidth. This column band-references the thermal-noise floor
    # (per-point noise = NBB*BW_IF; stepped-frequency measurement semantics -- see
    # the noise-injection comment in circuit_model_bb_approx) and sizes the optional
    # IF filter, which is gated by the explicit `if_filter` arg.
    BW_IF = 15e6 #IF BW is 15 Mhz
    BW = torch.full((nRx,), BW_IF)
    RX_config = torch.stack((Ibias_LNA, Vbias_LNA, Pdc_mix, Ibias_BB, Vbias_BB, Av, BW), dim=1)
    
    return RX_config


def circuit_model_bb_approx(RX_config, bb_IQ, fs, if_filter=False):
    '''
        Input values - V_bias_BB +- V1
        V1 --> 10^-5 t0 10^-6
        fs --> true complex sample rate of bb_IQ [Hz] (e.g. the span of the frequency
               plan the buffer's IFFT was built from). Used ONLY to size the optional
               IF filter's boxcar width; the noise floor is band-referenced to the
               BW config column (15 MHz IF), not to fs.
    '''

    '''
        What are the correct values for the IWR TI radar?
        Cannot control these params for the Radar
        Industry standards --> fixed from the design time.
    '''
    device = bb_IQ.device
    assert bb_IQ.device == RX_config.device, 'Device mismatch'
    # Preserved to reshape the output back at the end: for a single-vector caller
    # this is (nt,) (identity reshape); for the batched caller (circuit_model_batch)
    # it is (batch, nt) -- RX_config columns then broadcast as (batch, 1) against it.
    orig_shape = bb_IQ.shape
    Rs = SYSTEM_IMPEDANCE_OHMS  # ohms (shared with the generation-side voltage scaling)
    gammalna = 3 #efficiency LNA
    RoLNA = 50 #
    Plomax = 0.02
    Vodmax = 0.6
    Gsw0 = 0.06 
    Vsat = 0.5 #PDK Si Foundary
    Kn = 8 #In-phase passive mixers **
    gammabb = 1

    Ibias_LNA = RX_config[0]
    Vbias_LNA = RX_config[1]
    Plo = RX_config[2]
    Ibias_BB = RX_config[3]
    Vbias_BB = RX_config[4]
    Av = RX_config[5]
    BW = RX_config[6]

    GmLNA = 1.5 * Ibias_LNA / Vbias_LNA
    G3LNA = Ibias_LNA / Vbias_LNA**3 / 2
    AvLNA = GmLNA * RoLNA
    FLNA = 1 + gammalna / GmLNA / Rs + 1 / Av**2
    Pdclna = 2 * RoLNA * Ibias_LNA**2

    # Envelope clamp-at-cubic-peak (replaces the old per-rail I/Q clamp at +-Vbias_LNA,
    # which was only valid for a REAL per-rail cubic -- the LNA is a single complex
    # bandpass baseband signal, not two independent real rails; see the envelope
    # nonlinearity comment below). The compressive characteristic g(r) =
    # GmLNA*r - (3/4)*G3LNA*r^3 (r = envelope |v|, derived below) only compresses up
    # to its peak; past it the algebraic cubic turns over and would unphysically
    # EXPAND again. Peak: dg/dr = GmLNA - (9/4)*G3LNA*r^2 = 0
    #   => r* = sqrt(GmLNA / ((9/4)*G3LNA)).
    # Clamping the envelope (magnitude, phase preserved) at r* before the cubic pins
    # the output at the peak for any overdrive, instead of continuing past it.
    r_star_lna = torch.sqrt(GmLNA / (2.25 * G3LNA))
    v_mag = torch.abs(bb_IQ)
    clamp_scale = torch.clamp(r_star_lna / torch.clamp(v_mag, min=1e-30), max=1.0)
    bb_IQ = bb_IQ * clamp_scale

    # Bandpass cubic nonlinearity, baseband-equivalent (envelope) form. A real
    # bandpass cubic y = a1*u + a3*u^3 acting on u = Re{v e^{jwt}} produces, in the
    # fundamental zone, the complex envelope response a1*v + (3/4)*a3*|v|^2*v (the
    # standard third-order AM/AM baseband-equivalent identity) -- NOT the complex
    # cube v**3, which mixes the I/Q rails unphysically and makes the gain
    # phase-dependent. Here a1 = GmLNA, a3 = -G3LNA (the underlying real cubic being
    # Gm*u - G3*u^3), so the envelope form is GmLNA*v - (3/4)*G3LNA*|v|^2*v.
    Vlna = RoLNA * (GmLNA * bb_IQ - 0.75 * G3LNA * bb_IQ * torch.abs(bb_IQ) ** 2)
    Nlna = (Rs * FOURKT) * FLNA * AvLNA**2

    Vod = Vodmax * torch.sqrt(Plo / Plomax)
    Gsw = Gsw0 * Vod / Vodmax
    rho = 1 / (Gsw * RoLNA)
    a2 = -1 / 4 / Vod
    a3 = -1 / 2 / Vsat**2
    Avmix = rho * Kn / (1 + rho * (1 + Kn))
    Fmix = (1 + rho) * (1 + (rho + 1) / (rho * Kn))
    # Same envelope substitution as the LNA above: (Vlna*rho)**3 -> (3/4)*(Vlna*rho)*
    # |Vlna*rho|^2, so the mixer's cubic term doesn't mix I/Q or depend on phase.
    Imix = Vlna / RoLNA / (1 + rho) - 0.75 * (Vlna * rho) * torch.abs(Vlna * rho) ** 2 / RoLNA / (1 + rho)**5 * (2 * a2**2 - a3 * (1 + rho))
    Vmix_I = torch.real(Imix) * RoLNA * rho * Kn * (1 + rho) / (1 + rho * (1 + Kn))
    Vmix_Q = torch.imag(Imix) * RoLNA * rho * Kn * (1 + rho) / (1 + rho * (1 + Kn))
    Nmix = (Nlna + (RoLNA * FOURKT) * (Fmix - 1)) * Avmix**2

    # Optional IF anti-alias filter (boxcar of ~fs/BW_IF taps). Off by default; when
    # enabled, the width is floored at 1 tap (identity) and clamped to the record
    # length. Supports both the 1-D single-vector caller (Vmix_I shape (nt,)) and
    # the batched caller (shape (batch, nt), one row per (rx,tx,s) triple), whose
    # BW column may in principle differ per row -- rows are grouped by their
    # (typically-identical) resample width so each conv1d call stays batched.
    if if_filter:
        was_1d = Vmix_I.dim() == 1
        VI = Vmix_I.unsqueeze(0) if was_1d else Vmix_I
        VQ = Vmix_Q.unsqueeze(0) if was_1d else Vmix_Q
        nt = VI.shape[-1]
        BW_flat = BW.reshape(-1).expand(VI.shape[0]) if BW.numel() == 1 else BW.reshape(-1)
        resam_per_row = torch.clamp(torch.round(fs / BW_flat).long(), min=1, max=nt)

        def _boxcar(x):
            out = torch.empty_like(x)
            for width in torch.unique(resam_per_row):
                width = int(width.item())
                mask = resam_per_row == width
                kernel = torch.ones(1, 1, width, device=x.device, dtype=x.dtype)
                rows = x[mask].unsqueeze(1)  # [g, 1, nt]
                out[mask] = torch.nn.functional.conv1d(rows, kernel, padding='same').squeeze(1)
            return out

        VBB_I = _boxcar(VI)
        VBB_Q = _boxcar(VQ)
        if was_1d:
            VBB_I = VBB_I.squeeze(0)
            VBB_Q = VBB_Q.squeeze(0)
    else:
        VBB_I = Vmix_I
        VBB_Q = Vmix_Q

    GmBB = 1.5 * Ibias_BB / Vbias_BB
    G3BB = Ibias_BB / Vbias_BB**3 / 2
    AvBB = Av / (AvLNA * Avmix)
    RoBB = AvBB / GmBB
    FBB = 1 + gammabb / GmBB / RoLNA
    PdcBB = 2 * RoBB * Ibias_BB**2

    RBBI_id = torch.clamp(VBB_I, -Vbias_BB, Vbias_BB)
    RBBQ_id = torch.clamp(VBB_Q, -Vbias_BB, Vbias_BB)

    # Compressive (saturating), not expansive: AvBB*v - G3BB*RoBB*v^3. The old '+'
    # sign made the BB stage expand rather than saturate near clipping (author-
    # confirmed bug). With the minus sign, G3BB*RoBB = AvBB/(3*Vbias_BB^2) algebraically
    # (same GmBB/G3BB ~ Ibias_BB cancellation as the LNA), so the BB peak also lands
    # exactly at v* = Vbias_BB, matching the LNA's clamp-at-peak design.
    RBBI = AvBB * RBBI_id - G3BB * RoBB * RBBI_id**3
    RBBQ = AvBB * RBBQ_id - G3BB * RoBB * RBBQ_id**3
    NBB = (Nmix + (RoLNA * FOURKT) * (FBB - 1)) * AvBB**2
    # Per-sample noise variance = NBB * BW_IF. An S-parameter frame is a
    # stepped-frequency measurement: each frequency point is observed through the
    # receiver's IF bandwidth (BW column, 15 MHz), so the per-point noise power is
    # NBB*BW_IF; with the time-domain volts anchoring (mean|v|^2 = P_rx*Z0) that
    # maps to the same per-time-sample variance. Referencing the noise to the full
    # buffer sample rate fs (~3 GHz) instead would inflate the floor by fs/BW_IF
    # (~23 dB) -- fs is used only for the IF-filter width above, and the boxcar
    # does NOT act on this noise (it is injected at the BB output, after the
    # filter position), so the noise must be band-referenced here, not filtered.
    #
    # Two-seam bookkeeping with the caller's FFT round-trip (RFFEBlock.apply_circuit):
    # the signal seam takes ifft (1/N-normalized) into this time-domain buffer and
    # (unnormalized) fft back out to frequency bins. The noise seam is different: it
    # is injected HERE, in the time domain, then only goes through the unnormalized
    # forward fft. That fft is a linear combination of the `nt` (= orig_shape[-1], the
    # buffer's last-dim length) time samples, so it multiplies the per-sample
    # variance by nt (each output bin sums nt independent unit-magnitude-coefficient
    # contributions). Injecting NBB*BW per sample would therefore land nt*NBB*BW per
    # frequency bin instead of the intended NBB*BW -- so the per-sample variance
    # injected here must be pre-divided by nt.
    nt = orig_shape[-1]
    RBBI += torch.randn_like(RBBI) * torch.sqrt(NBB * BW / nt)
    RBBQ += torch.randn_like(RBBQ) * torch.sqrt(NBB * BW / nt)
    PRX = Pdclna + Plo + PdcBB
    RxBB = RBBI + 1j * RBBQ

    return RxBB.reshape(orig_shape), PRX

def circuit_model_batch(rx_config, input_signals, fs, if_filter=False):
    '''
        Vectorized replacement for the old nrx*ntx*ns Python loop: every (rx,tx,s)
        slice is elementwise-independent in circuit_model_bb_approx (the only
        non-elementwise op, the IF boxcar, is itself batched -- see above), so we
        flatten the whole [nrx, ntx, ns, nt] signal to a [batch, nt] matrix and each
        RX_config column to a broadcastable [batch, 1], and call
        circuit_model_bb_approx once for the entire batch.
    '''
    device = input_signals.device
    assert len(input_signals.shape) == 4, 'Input signals must have 4 dimensions'
    nrx, ntx, ns, nt = input_signals.shape
    n_cfg = rx_config.shape[1]
    batch = nrx * ntx * ns

    # rx_config is per-rx only; broadcast each row across its (ntx, ns) slices, then
    # transpose to [n_cfg, batch, 1] so RX_config[i] inside circuit_model_bb_approx is
    # a [batch, 1] column (matches the elementwise ops' broadcasting against [batch, nt]).
    cfg_batch = rx_config.unsqueeze(1).unsqueeze(1).expand(nrx, ntx, ns, n_cfg)
    cfg_batch = cfg_batch.reshape(batch, n_cfg).t().unsqueeze(-1)

    sig_flat = input_signals.reshape(batch, nt)
    out_flat, PRX_flat = circuit_model_bb_approx(cfg_batch, sig_flat, fs, if_filter=if_filter)
    input_signals_circuit = out_flat.reshape(nrx, ntx, ns, nt)

    # PRX (Pdclna + Plo + PdcBB) depends only on RX_config, not on the signal or s/t,
    # so it is identical across the ntx*ns slices for a given rx -- reducing to the
    # last one reproduces the old loop's PRX[t, r] (which always ended up holding
    # that same value; the pipeline asserts a single tx/chirp, i.e. ntx == 1).
    PRX = PRX_flat.reshape(nrx, ntx * ns)[:, -1].reshape(1, nrx)

    return input_signals_circuit, PRX
