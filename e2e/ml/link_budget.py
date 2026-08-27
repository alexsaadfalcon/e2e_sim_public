"""Deprecated shim: this module moved to `e2e.chain.link_budget` (C1 move,
`notes/C1_MOVE_PLAN.md`) -- it is a core receive-chain stage (`ThermalNoiseBlock`
lives here, wired into `e2e/chain/receive.py`'s chain), not ML-specific.

Re-exports the old module's public names so existing imports of `e2e.ml.link_budget`
keep working. Kept through v1.1; removal left to successors. Import
`e2e.chain.link_budget` directly in new code.
"""

import warnings

warnings.warn(
    "e2e.ml.link_budget has moved to e2e.chain.link_budget; import from there instead. "
    "This shim is kept through v1.1 and will be removed after.",
    DeprecationWarning,
    stacklevel=2,
)

from e2e.chain.link_budget import (  # noqa: F401,E402
    DEFAULT_NOISE_FIGURE_DB,
    DEFAULT_TX_POWER_DBM,
    K_BOLTZMANN,
    T0_KELVIN,
    ThermalNoiseBlock,
    add_thermal_noise,
    coherent_processing_gain_db,
    expected_target_snr_db,
    noise_bandwidth_hz,
    thermal_noise_power_w,
    tx_amplitude_scale,
)
