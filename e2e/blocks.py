import torch

from e2e.environment.sionna_iterator import SionnaEtoileIterator, SionnaMunichIterator
from e2e.subspace.algorithms import Oja, gen_A_ada
from e2e.subspace.subspace_utils import subspace_dist_frob
from e2e.afe.afe_utils import quantizer_fp
from e2e.circuit.rffe_model import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SionnaEnvironmentBlock:
    def __init__(self, scenario_name):
        valid_scenarios = {
            'etoile': SionnaEtoileIterator,
            'munich': SionnaMunichIterator,
        }
        if scenario_name not in valid_scenarios:
            raise ValueError(f'unknown scenario {scenario_name}')
        self.scenario_name = scenario_name
        self.sionna_iterator = valid_scenarios[scenario_name]()
        self.frame_counter = 0
    
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
    def __init__(self, n=None, chirp_dur=10e-9, signal_scaling=1e-5):
        rx_config = get_RX_config(n, chirp_dur)
        rx_config[:, 6] = 20001
        self.rx_config = rx_config.to(device)
        self.chirp_dur = chirp_dur
        self.signal_scaling = signal_scaling
        self.n = n

    def apply_circuit(self, s_pars):
        s_pars_shape = s_pars.shape
        s_pars = s_pars.view(self.n, 1, 2, s_pars.shape[-1])
        frame = torch.fft.ifft(s_pars, dim=-1)
        frame = frame * self.signal_scaling / torch.mean(torch.abs(frame))
        frame_dist, PRX = circuit_model_batch(self.rx_config, frame, self.chirp_dur)
        s_pars_dist = torch.fft.fft(frame_dist, dim=-1)
        s_pars_dist = s_pars_dist.view(s_pars_shape)
        return s_pars_dist, PRX


# RF Interconnect Model Block
class InterconnectBlock:
    def __init__(self, case=None):
        self.case = case
        
    def apply_interconnect(self, frame):
        if self.case == 'case3':
            return frame
        window = torch.ones(11)
        window = window.to(device)
        window_padded = torch.nn.functional.pad(window, (0, frame.shape[-1] - window.shape[0]))
        window_padded = window_padded.to(device)
        window_ifft = torch.fft.fft(window_padded)
        frame = frame * window_ifft.view(1, 1, 1, -1)
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
    def __init__(self, n, d):
        self.oja = Oja(n, d, eta=1e0, fixed_step=True)

    def gen_A_ada(self, m=None):
        if m is None:
            m = self.oja.d * 2
        return gen_A_ada(self.oja.U, m)

    def update(self, X, A):
        self.oja.add_data(X, A)


class FFTBlock:
    def __init__(self, bins=256):
        self.bins = bins

    def apply(self, state_dict):
        data = state_dict['s_pars'][:, :, 0, :]
        assert len(data.shape) == 3
        data_fft = torch.fft.fft(torch.fft.fft(torch.sum(data, dim=2), self.bins, 0), self.bins, 1)
        data_fft = torch.fft.fftshift(torch.fft.fftshift(data_fft, 0), 1)
        return {'fft': data_fft}


class RangeAzBlock:
    def __init__(self, bins=256):
        self.bins = bins

    def apply(self, state_dict):
        data = state_dict['s_pars'][:, :, 0, :]
        assert len(data.shape) == 3
        data_fft = torch.fft.fft(torch.fft.fft(torch.sum(data, dim=1), self.bins, 0), 1)
        data_fft = torch.fft.fftshift(torch.fft.fftshift(data_fft, 0), 1)
        return {'range_az': data_fft}


class RangeElBlock:
    def __init__(self, bins=256):
        self.bins = bins

    def apply(self, state_dict):
        data = state_dict['s_pars'][:, :, 0, :]
        assert len(data.shape) == 3
        data_fft = torch.fft.fft(torch.fft.fft(torch.sum(data, dim=0), self.bins, 0), 1)
        data_fft = torch.fft.fftshift(torch.fft.fftshift(data_fft, 0), 1)
        return {'range_el': data_fft}


class SubspaceErrorBlock:
    def __init__(self):
        self.metric = subspace_dist_frob

    def apply(self, state_dict):
        U = state_dict['U_true']
        U_pred = state_dict['U']
        return {'subspace_err': self.metric(U, U_pred)}

