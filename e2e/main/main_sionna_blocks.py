import os

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

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.figure()
plt.title('Subspace tracking error per frame')
# subspace_err entries are scalar tensors on the library device; float() copies each
# to the host so matplotlib never sees a CUDA tensor (plt would try .numpy() and fail).
plt.plot([float(x) for x in outputs['subspace_err']], marker='o')
plt.xlabel('frame')
plt.ylabel('subspace error')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(FIG_DIR, 'sionna_blocks_subspace_err.png'), bbox_inches='tight')

# 'fft' is now a real, non-negative az/el power map: coherent 2D aperture FFT,
# non-coherent (power) integration over range -- a target at any range shows up,
# not just one at range 0 (see e2e/blocks.py FFTBlock).
for i, _fft in enumerate(outputs['fft']):
    _fft = _fft / torch.max(torch.abs(_fft))
    # Power quantity -> 10*log10 (matches the webapp's conversion). .cpu().numpy()
    # so the frame's device (CPU or CUDA) doesn't matter for plotting.
    fft_energy = (10 * torch.log10(torch.abs(_fft))).T.cpu().numpy()
    plt.figure()
    plt.title('Azimuth-Elevation power (non-coherent over range)')
    plt.imshow(fft_energy)
    plt.colorbar()
    plt.clim([-40, 0])
    plt.savefig(os.path.join(FIG_DIR, f'sionna_blocks_az_el_frame{i}.png'),
                bbox_inches='tight')
    plt.show()
