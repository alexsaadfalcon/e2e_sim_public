"""Deprecated shim: this module moved to `e2e.environment.rt_signal_chain` (C1 move,
`notes/C1_MOVE_PLAN.md`) -- it is core Sionna RT path-to-signal physics, not
ML-specific; `e2e.environment.rt_gen` (also moved) re-exports its public surface.

Re-exports the old module's public names so existing imports of
`e2e.ml.rt_signal_chain` keep working. Kept through v1.1; removal left to successors.
Import `e2e.environment.rt_signal_chain` directly in new code.
"""

import warnings

warnings.warn(
    "e2e.ml.rt_signal_chain has moved to e2e.environment.rt_signal_chain; import from "
    "there instead. This shim is kept through v1.1 and will be removed after.",
    DeprecationWarning,
    stacklevel=2,
)

from e2e.environment.rt_signal_chain import (  # noqa: F401,E402
    _ANTENNA_INDEX_REVERSED,
    _add_awgn,
    _beat_from_paths,
    _coherent_gain,
    _peak_reference_amplitude,
    _resolve_device,
    _snr_reference_chirps,
    _solve,
    _specular_point,
    beat_frequencies,
    cfr_from_paths,
    cfr_sum_over_paths,
    cfr_sum_over_paths_budgeted,
    coherent_target_cfr,
    doppler_validity,
    mimo_combine,
    rt_cfr_frame,
    rt_retrace_reference,
    rt_synthesize_adc,
    warn_if_doppler_invalid,
)
