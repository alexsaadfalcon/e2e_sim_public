import numpy as np
import torch


'''
Todo: Convert the RX config to Json dict
'''
def get_RX_config(nRx, chirp_duration):
    # // https://www.ti.com/lit/an/swra553a/swra553a.pdf?ts=1723503949605 Pg 7
    # Define other simulation parameters
    
    rf_fc = 78.5e9  # center frequency for the RF, typical for radar applications
    rf_bw = 5e9  # bandwidth
    # https://www.ti.com/lit/ds/symlink/iwr1443.pdf?ts=1723445927051 Pg 7
    if_freq = 15e6  # intermediate frequency

    # Calculation derived from defined parameters
    dt = 1 / if_freq  # Time step [s]
    chirp_prop = [chirp_duration, rf_fc - rf_bw/2, rf_fc + rf_bw/2]  # [Duration [s], flo [Hz], fhi [Hz]]

    # Placeholder values for other configurations (config.)
    # Ibias_LNA = torch.full((nRx,), 1) # mA --> A from Datasheet
    #from the email ~500 uA – 10 mA.
    Ibias_LNA = torch.full((nRx,), 1e-3) # Let's take 1e-3

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


    Av = torch.full((nRx,), 24) # dB
    BW_IF = 15e6 #IF BW is 15 Mhz
    BW_cons = BW_IF*chirp_duration ## using BW(x) = BW*T
    BW = torch.full((nRx,), BW_cons)##### multiple of 1/T where T is the chirp duration? #times
    RX_config = torch.stack((Ibias_LNA, Vbias_LNA, Pdc_mix, Ibias_BB, Vbias_BB, Av, BW), dim=1)
    
    return RX_config


def circuit_model_bb_approx(RX_config, bb_IQ, T):
    '''
        Input values - V_bias_BB +- V1
        V1 --> 10^-5 t0 10^-6
        T --> Chirp Duration
    '''

    '''
        What are the correct values for the IWR TI radar?
        Cannot control these params for the Radar
        Industry standards --> fixed from the design time.
    '''
    device = bb_IQ.device
    assert bb_IQ.device == RX_config.device, 'Device mismatch'
    Rs = 50 # ohms
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

    bb_approx_I = torch.real(bb_IQ)
    bb_approx_Q = torch.imag(bb_IQ)
    bb_approx_I = torch.clamp(bb_approx_I, -Vbias_LNA, Vbias_LNA)
    bb_approx_Q = torch.clamp(bb_approx_Q, -Vbias_LNA, Vbias_LNA)
    bb_IQ = bb_approx_I + 1j * bb_approx_Q

    Vlna = RoLNA * (GmLNA * bb_IQ - G3LNA * bb_IQ**3)
    Nlna = (Rs * 4e-21) * FLNA * AvLNA**2

    Vod = Vodmax * torch.sqrt(Plo / Plomax)
    Gsw = Gsw0 * Vod / Vodmax
    rho = 1 / (Gsw * RoLNA)
    a2 = -1 / 4 / Vod
    a3 = -1 / 2 / Vsat**2
    Avmix = rho * Kn / (1 + rho * (1 + Kn))
    Fmix = (1 + rho) * (1 + (rho + 1) / (rho * Kn))
    Imix = Vlna / RoLNA / (1 + rho) - (Vlna * rho)**3 / RoLNA / (1 + rho)**5 * (2 * a2**2 - a3 * (1 + rho))
    x3 = (AvLNA * rho)**3 / RoLNA / (1 + rho)**5 * (2 * a2**2 - a3 * (1 + rho)) / (AvLNA / RoLNA / (1 + rho))
    Vmix_I = torch.real(Imix) * RoLNA * rho * Kn * (1 + rho) / (1 + rho * (1 + Kn))
    Vmix_Q = torch.imag(Imix) * RoLNA * rho * Kn * (1 + rho) / (1 + rho * (1 + Kn))
    Nmix = (Nlna + (RoLNA * 4e-21) * (Fmix - 1)) * Avmix**2

    resam = round(Vmix_I.shape[0] / BW.item())
    if resam != 0:
        VBB_I = []
        VBB_Q = []
        for num_TX in range(Vmix_I.shape[-1]):
            VBB_I.append(torch.nn.functional.conv1d(Vmix_I[:, num_TX].unsqueeze(0).unsqueeze(0), 
                                                    torch.ones(1, 1, resam), padding='same').squeeze())
            VBB_Q.append(torch.nn.functional.conv1d(Vmix_Q[:, num_TX].unsqueeze(0).unsqueeze(0), 
                                                    torch.ones(1, 1, resam), padding='same').squeeze())
        VBB_I = torch.stack(VBB_I, dim=1)
        VBB_Q = torch.stack(VBB_Q, dim=1)
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

    RBBI = AvBB * RBBI_id + G3BB * RoBB * RBBI_id**3
    RBBQ = AvBB * RBBQ_id + G3BB * RoBB * RBBQ_id**3
    NBB = (Nmix + (RoLNA * 4e-21) * (FBB - 1)) * AvBB**2
    NBW = BW / T
    RBBI += torch.randn_like(RBBI) * torch.sqrt(NBB * NBW)
    RBBQ += torch.randn_like(RBBQ) * torch.sqrt(NBB * NBW)
    PRX = Pdclna + Plo + PdcBB
    RxBB = RBBI + 1j * RBBQ

    return RxBB.reshape(-1), PRX

def circuit_model_batch(rx_config, input_signals, chirp_dur):
    device = input_signals.device
    assert len(input_signals.shape) == 4, 'Input signals must have 4 dimensions'
    nrx, ntx, ns, nt = input_signals.shape
    input_signals_circuit = torch.zeros_like(input_signals, device=device)
    PRX = torch.zeros((1, nrx), device=device)

    for s in range(ns):
        for t in range(ntx):
            for r in range(nrx):
                input_signals_circuit[r, t, s, :], PRX[t, r] = circuit_model_bb_approx(
                    rx_config[r], input_signals[r, t, s, :], chirp_dur)

    return input_signals_circuit, PRX
