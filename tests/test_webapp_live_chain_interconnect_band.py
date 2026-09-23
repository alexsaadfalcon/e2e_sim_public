"""`webapp.pipeline_runner._corpus_live_interconnect_band_hz`: the live-chain replay
path's interconnect band, derived from the corpus's own RadarConfig carrier instead of
a re-typed literal (owner course-correction, 2026-09-23) -- see that function's
docstring and `e2e.ml.chain_generate._interconnect_band_hz`, which it delegates to.

A separate file from tests/test_webapp_live_chain.py (which owns the live-chain
correctness-gate tests) so this narrow addition does not touch another coder's
assertions there; both files may grow independently.
"""
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from webapp.pipeline_runner import (
    _LEGACY_77GHZ_INTERCONNECT_BAND_HZ,
    _corpus_live_interconnect_band_hz,
)


def test_legacy_77ghz_corpus_gets_the_exact_historical_literal():
    """Every corpus generated at f0=77 GHz before the Ka-band re-founding (b1_bench_v3,
    benchmark_v1_D2/D4, b1_demo_cfr) must resolve to bit-for-bit (75e9, 81e9), or the
    live-chain correctness gate (tests/test_webapp_live_chain.py) would report a
    spurious non-zero diff on every one of them."""
    cfg = SimpleNamespace(f0_hz=77e9)
    assert _corpus_live_interconnect_band_hz(cfg) == _LEGACY_77GHZ_INTERCONNECT_BAND_HZ
    assert _LEGACY_77GHZ_INTERCONNECT_BAND_HZ == (75e9, 81e9)


def test_a_different_carrier_derives_its_own_band_not_the_77ghz_literal():
    """A re-traced (e.g. Ka-band) corpus must NOT be silently mapped over the 77 GHz
    literal -- this is the whole point of the fix (a hardcoded band would apply the
    wrong physical window once corpora regenerate at a different carrier)."""
    cfg = SimpleNamespace(f0_hz=30e9)
    band = _corpus_live_interconnect_band_hz(cfg)
    assert band != _LEGACY_77GHZ_INTERCONNECT_BAND_HZ
    assert band[0] < 30e9 < band[1]


def test_matches_the_generators_own_function_bit_for_bit():
    """Delegates to (does not re-derive) `e2e.ml.chain_generate._interconnect_band_hz`
    -- the drift the prior literal risked is impossible if both call sites are the same
    function."""
    from e2e.ml.chain_generate import _interconnect_band_hz

    for f0_hz in (77e9, 30e9, 60e9):
        cfg = SimpleNamespace(f0_hz=f0_hz)
        assert _corpus_live_interconnect_band_hz(cfg) == _interconnect_band_hz(cfg)


def test_missing_or_broken_corpus_cfg_falls_back_to_the_literal():
    """`corpus_cfg=None` (should not happen on the live-chain path, but a provenance
    lookup must never take the run down) and an object `_interconnect_band_hz` cannot
    resolve (no `f0_hz` at all) both fall back to the historical literal rather than
    raising."""
    assert _corpus_live_interconnect_band_hz(None) == _LEGACY_77GHZ_INTERCONNECT_BAND_HZ
    assert (_corpus_live_interconnect_band_hz(SimpleNamespace())
            == _LEGACY_77GHZ_INTERCONNECT_BAND_HZ)
