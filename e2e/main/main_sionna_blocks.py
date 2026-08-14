import os

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless: write figures to files, no display
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
from e2e.viz import to_db



N_RX_X = 32
N_RX_Y = 32
N_RX = N_RX_X * N_RX_Y
N_TX = 1
N_FREQS = 5000
freqs = np.linspace(28.5e9, 31.5e9, N_FREQS)

# Hand-rolled rather than e2e.viz.fig_dir: main(show=False) must never touch disk
# (pinned by tests/test_main_sionna_blocks.py), and fig_dir() creates the directory
# eagerly. The makedirs happens inside the `if show:` branch instead.
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")


def main(scenario_name="munich", environment_block=None, n_steps=2, k=8, show=False):
    """Run the full radar pipeline (environment -> RFFE -> interconnect -> AFE ->
    AdaOja subspace tracking -> FFT/RangeAz/RangeEl/SubspaceError) and, if `show`,
    save the subspace-error and az/el-map figures to FIG_DIR.

    `environment_block`, if given, is used as-is (e.g. a synthetic drop-in for
    tests); otherwise a `SionnaEnvironmentBlock(scenario_name)` is constructed,
    with a friendly error (instead of a raw FileNotFoundError) if that scenario's
    precomputed frames haven't been generated yet -- mirrors the guard around the
    same constructor in webapp/pipeline_runner.py.

    Returns `outputs`, the raw dict from `Simulation.run()`.
    """
    if environment_block is None:
        try:
            environment_block = SionnaEnvironmentBlock(scenario_name)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"No precomputed frames found for scenario '{scenario_name}'. Generate "
                "them first: `python -m e2e.environment.sionna_simple_channel` (writes "
                "munich.pkl), or for a quick check with no Sionna/GPU needed, "
                "`python -m e2e.environment.scenario_runner --scenario munich_radar "
                f"--dry-run`. Missing file: {e}"
            ) from e

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
    # Track at the signal's spectral elbow (rank ~8, where the top-k subspace is well
    # defined) with enough measurements (m=512) to observe the scene's subspace drift; the
    # tracker then follows it via a per-frame SVD re-estimate. See AdaOjaBlock / ROADMAP.
    subspace_block = AdaOjaBlock(N_RX, k, m=512, n_refine=10)

    sim = Simulation(
        environment_block,
        downstream_blocks,
        k,
        circuit_block,
        interconnect_block1,
        afe_block,
        subspace_block,
    )
    outputs = sim.run(n_steps=n_steps)

    if show:
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
        plt.close()

        # 'fft' is now a real, non-negative az/el power map: coherent 2D aperture FFT,
        # non-coherent (power) integration over range -- a target at any range shows up,
        # not just one at range 0 (see e2e/blocks.py FFTBlock).
        for i, _fft in enumerate(outputs['fft']):
            # Power quantity -> peak-normalized dB via the shared helper (matches the
            # webapp's conversion). This is an az/el (not az/range) map --
            # e2e.viz.imshow_ra's transpose+extent helper is specific to az/range
            # display conventions, so only the dB-normalize half is consolidated here.
            # .cpu().numpy() so the frame's device doesn't matter for plotting.
            fft_energy = to_db(_fft, floor_db=-40.0).T.cpu().numpy()
            plt.figure()
            plt.title('Azimuth-Elevation power (non-coherent over range)')
            plt.imshow(fft_energy)
            plt.colorbar()
            plt.clim([-40, 0])
            plt.savefig(os.path.join(FIG_DIR, f'sionna_blocks_az_el_frame{i}.png'),
                        bbox_inches='tight')
            plt.close()

    return outputs


if __name__ == "__main__":
    main(show=True)
