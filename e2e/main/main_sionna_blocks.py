import torch
import numpy as np
import matplotlib.pyplot as plt

from e2e.environment.sionna_iterator import SionnaMunichIterator


from e2e.simulation import Simulation
from e2e.blocks import \
    SionnaEnvironmentBlock, \
    RFFEBlock, \
    InterconnectBlock, \
    AFEBlock, \
    AdaOjaBlock, \
    FFTBlock, \
    RangeAzBlock, \
    RangeElBlock, \
    SubspaceErrorBlock



N_RX_X = 32
N_RX_Y = 32
N_RX = N_RX_X * N_RX_Y
N_TX = 1
N_FREQS = 5000
freqs = np.linspace(28.5e9, 31.5e9, N_FREQS)

environment_block = SionnaEnvironmentBlock('munich')
downstream_blocks = [
    FFTBlock(),
    RangeAzBlock(),
    RangeElBlock(),
    SubspaceErrorBlock(),
]

circuit_block = RFFEBlock(n=N_RX * N_TX, chirp_dur=10e-9)
interconnect_block1 = InterconnectBlock(case='case3')
interconnect_block2 = InterconnectBlock(case='synthetic')

afe_block = AFEBlock()
d = 16
subspace_block = AdaOjaBlock(N_RX, d)

sim = Simulation(
    environment_block,
    downstream_blocks,
    d,
    circuit_block,
    interconnect_block1,
    afe_block,
    subspace_block,
)
outputs = sim.run(n_steps=2)

plt.figure()
plt.title('Subspace Error')
plt.plot(outputs['subspace_err'])

for _fft in outputs['fft']:
    _fft = _fft / torch.max(torch.abs(_fft))
    fft_energy = 20 * torch.log10(torch.abs(_fft)).T.cpu()
    plt.figure()
    plt.imshow(fft_energy)
    plt.colorbar()
    plt.clim([-40, 0])
    plt.show()
