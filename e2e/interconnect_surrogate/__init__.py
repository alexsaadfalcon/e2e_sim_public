"""Optional, lazily-imported bridge to the public TSV S-parameter surrogate.

This subpackage wraps the third-party **Tessera / TSV_PhGNN** physics-informed GNN
surrogate (BSD 3-Clause, github.com/HiPerCAS/tessera) so that a TSV interconnect's
S21(f) -- and its NEXT/FEXT crosstalk -- can be computed *live* from geometry +
temperature instead of being read from the frozen CSVs in ``e2e/data/interconnect/``.

Import-time contract
--------------------
Importing ``e2e.interconnect_surrogate`` must NOT import torch, torch-geometric or
the ``tessera`` package. Everything heavy happens inside :meth:`TesseraTSV.s21` and
friends; :func:`available` answers the "is it installed?" question using
``importlib.util.find_spec`` plus a filesystem check, so a webapp shell can ask
without paying the import. This mirrors the convention enforced by
``tests/test_webapp.py::_import_without_torch`` and is pinned by
``tests/test_interconnect_surrogate.py::test_import_does_not_import_torch``.

Install the extra with::

    pip install -r requirements-tessera.txt      # or: pip install -e ".[tessera]"

See ``NOTICE`` in this directory for the upstream BSD-3 notice and the TCAD
citation that its licence requires us to carry.
"""

from e2e.interconnect_surrogate.tessera import (
    ARRANGEMENTS,
    SHIPPED_TSV_DESIGN,
    VALID_RANGES,
    PassivityError,
    TesseraTSV,
    available,
    checkpoint_dir,
    ring_arrangement,
)
from e2e.interconnect_surrogate.cache import SurrogateCache, default_cache_dir

__all__ = [
    "TesseraTSV",
    "available",
    "checkpoint_dir",
    "PassivityError",
    "VALID_RANGES",
    "SHIPPED_TSV_DESIGN",
    "ARRANGEMENTS",
    "ring_arrangement",
    "SurrogateCache",
    "default_cache_dir",
]
