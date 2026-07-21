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

# Mirror the webapp's auto scale-resolution (webapp/pipeline_runner.py
# _resolve_physical_scale): v2 pkls carry physical_scale=True metadata (frames
# already in volts at the LNA input); legacy pkls expose it as None, which keeps
# the pre-existing renormalize-by-signal_scaling behavior (physical_scale=False).
circuit_block = RFFEBlock(
    n=N_RX * N_TX,
    physical_scale=bool(getattr(environment_block, 'physical_scale', None)),
)
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

# 'fft' is now a real, non-negative az/el power map: coherent 2D aperture FFT,
# non-coherent (power) integration over range -- a target at any range shows up,
# not just one at range 0 (see e2e/blocks.py FFTBlock).
for _fft in outputs['fft']:
    _fft = _fft / torch.max(torch.abs(_fft))
    # Power quantity -> 10*log10 (matches the webapp's conversion).
    fft_energy = 10 * torch.log10(torch.abs(_fft)).T.cpu()
    plt.figure()
    plt.title('Azimuth-Elevation power (non-coherent over range)')
    plt.imshow(fft_energy)
    plt.colorbar()
    plt.clim([-40, 0])
    plt.show()
