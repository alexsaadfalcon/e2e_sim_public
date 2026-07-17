"""
Communications layer for the end-to-end array-processing simulator.

This package complements the existing radar-only path with link-level
communications, pilot-based channel estimation, and joint radar/comms (ISAC)
utilities. It is intentionally self-contained: everything works with plain
torch/numpy and does NOT require Sionna or any precomputed `.pkl` frames.

Sub-modules
-----------
* `ofdm`    -- QAM mapping/demapping and an OFDM modem (IFFT/FFT + cyclic prefix,
               pilot insertion).
* `channel` -- apply a frequency-domain channel (from S-parameters) to an OFDM
               signal, LS / MMSE channel estimation from pilots, ZF / MMSE
               equalization, plus BER / EVM / MSE metrics. Also provides a small
               synthetic multipath channel generator used as a fallback when the
               Sionna `.pkl` frames are unavailable.
* `isac`    -- share one waveform between sensing and communication; helpers to
               split a multi-node `Scenario` into radar vs comm sub-problems.
* `blocks`  -- optional pipeline blocks (ModemBlock / BERBlock) following the same
               `apply(state_dict) -> dict` convention as the downstream blocks in
               the top-level `e2e/blocks.py`.

The example entry points live under `e2e/main/`:
    python -m e2e.main.main_comms_link
    python -m e2e.main.main_channel_estimation
    python -m e2e.main.main_isac
"""

from . import ofdm
from . import channel
from . import isac
from . import blocks

__all__ = ["ofdm", "channel", "isac", "blocks"]
