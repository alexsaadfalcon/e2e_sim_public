"""Deprecated shim: this module moved to `e2e.environment.rt_doppler_study` (C1 move,
`notes/C1_MOVE_PLAN.md`) -- it is the core native-vs-re-trace Doppler experiment
harness, not ML-specific; `e2e.environment.rt_gen` (also moved) re-exports its public
surface.

Re-exports the old module's public names so existing imports of
`e2e.ml.rt_doppler_study` keep working. Kept through v1.1; removal left to successors.
Import `e2e.environment.rt_doppler_study` directly in new code.
"""

import warnings

warnings.warn(
    "e2e.ml.rt_doppler_study has moved to e2e.environment.rt_doppler_study; import "
    "from there instead. This shim is kept through v1.1 and will be removed after.",
    DeprecationWarning,
    stacklevel=2,
)

from e2e.environment.rt_doppler_study import (  # noqa: F401,E402
    _demo_scenario,
    _rd_peak_bin,
    build_arg_parser,
    doppler_error_study,
    format_error_study,
    main,
)
