"""Array-processing end-to-end simulator.

This package bundles the full chain used by the simulator:

* `e2e.environment` -- Sionna RT channel generation and precomputed-frame iterators.
* `e2e.circuit` / `e2e.afe` -- RF front-end and analog front-end (quantization) models.
* `e2e.subspace` -- adaptive subspace tracking (Oja / Ada-Oja).
* `e2e.blocks` / `e2e.simulation` -- the composable pipeline blocks and the
  feed-forward `Simulation` that drives them.
* `e2e.comms` -- OFDM modem, channel estimation/equalization, and ISAC utilities.
* `e2e.scenario` -- declarative, JSON-serializable scenario specifications shared by
  the web UI, the offline generator, and the examples.

Import-time contract
---------------------
`e2e.scenario` and the web UI are designed to be importable WITHOUT torch. The
pipeline code in `e2e.blocks` / `e2e.simulation` does import torch, so those
re-exports are kept LAZY (PEP 562 ``module __getattr__``): plain ``import e2e`` and
``import e2e.scenario`` stay torch-free, while ``from e2e import Simulation`` (or any
block class) imports torch on demand. The scenario symbols are torch-free and are
re-exported eagerly.
"""

from e2e.scenario import Scenario, REFERENCE_SCENARIOS

__all__ = [
    # Eager, torch-free scenario layer.
    "Scenario",
    "REFERENCE_SCENARIOS",
    # Lazy (torch-backed) pipeline driver.
    "Simulation",
    # Lazy (torch-backed) pipeline blocks.
    "SionnaEnvironmentBlock",
    "RFFEBlock",
    "InterconnectBlock",
    "AFEBlock",
    "AdaOjaBlock",
    "FFTBlock",
    "RangeAzBlock",
    "RangeElBlock",
    "SubspaceErrorBlock",
]

# Map each lazily-exported name to the module it lives in. Resolved on first access
# via __getattr__ so importing `e2e` (or `e2e.scenario`) never pulls in torch.
_LAZY_EXPORTS = {
    "Simulation": "e2e.simulation",
    "SionnaEnvironmentBlock": "e2e.blocks",
    "RFFEBlock": "e2e.blocks",
    "InterconnectBlock": "e2e.blocks",
    "AFEBlock": "e2e.blocks",
    "AdaOjaBlock": "e2e.blocks",
    "FFTBlock": "e2e.blocks",
    "RangeAzBlock": "e2e.blocks",
    "RangeElBlock": "e2e.blocks",
    "SubspaceErrorBlock": "e2e.blocks",
}


def __getattr__(name):
    """PEP 562 lazy attribute access for the torch-backed re-exports."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value  # cache so subsequent lookups skip __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
