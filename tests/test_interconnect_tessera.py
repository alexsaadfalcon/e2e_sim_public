"""Tests for `InterconnectBlock(source='tessera')` -- the live Tessera TSV surrogate
mode (e2e/interconnect_surrogate/), as opposed to the boxcar placeholder or a CSV.

Validation (range/arrangement checks, the not-importable path, the scale model) runs
unconditionally -- no third-party ``tessera``/``torch-geometric`` import (or a real
forward pass) is needed for those; the scale-model tests use a monkeypatched
`TesseraTSV.s21` spy instead. The handful of tests that need a real prediction
(passivity, caching) are guarded by ``requires_surrogate`` and skip cleanly, mirroring
tests/test_interconnect_surrogate.py.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from e2e.blocks import InterconnectBlock, TESSERA_DESIGN_PARAMS, device, tessera_s21_for_axis
from e2e.interconnect_surrogate import SurrogateCache, available
from e2e.interconnect_surrogate import tessera as tessera_mod

requires_surrogate = pytest.mark.skipif(
    not available(),
    reason="the Tessera surrogate (tessera-tsv + checkpoint) is not installed; "
           "see requirements-tessera.txt",
)

PIPELINE_BAND = (28.5e9, 31.5e9)      # Ka: munich frames, main_sionna_blocks, etc.
ML_CORPUS_BAND = (75e9, 81e9)         # 77 GHz: e2e/ml/chain_generate.py, webapp/pipeline_runner.py

# Midpoint-ish of every (widened) VALID_RANGES interval -- deliberately not
# SHIPPED_TSV_DESIGN's raw numbers, so tests here are independent of that point.
IN_RANGE_PARAMS = {
    "radius_um": 4.0,
    "pitch_um": 30.0,
    "height_um": 80.0,
    "liner_um": 2.0,
    "temperature_k": 450.0,
}

# Half of IN_RANGE_PARAMS' geometry (temperature unchanged) -- used as PRESENTED values
# at scale=2, so the resulting MODEL geometry is exactly IN_RANGE_PARAMS again.
HALF_GEOMETRY_PARAMS = {
    "radius_um": 2.0,
    "pitch_um": 15.0,
    "height_um": 40.0,
    "liner_um": 1.0,
    "temperature_k": 450.0,
}


def _ones_frame(n_freqs):
    return torch.ones(1, 1, 1, n_freqs, dtype=torch.complex64, device=device)


def _spy_s21(monkeypatch):
    """Monkeypatch `TesseraTSV.s21` to record every call instead of predicting; returns
    the list calls are appended to. No surrogate/checkpoint needed."""
    calls = []

    def fake_s21(self, freqs_hz, *, radius_um, pitch_um, height_um, liner_um,
                 temperature_k, grid=None, signal_index=0):
        freqs = np.atleast_1d(np.asarray(freqs_hz, dtype=np.float64))
        calls.append(dict(freqs=freqs, radius_um=radius_um, pitch_um=pitch_um,
                          height_um=height_um, liner_um=liner_um,
                          temperature_k=temperature_k, grid=grid))
        return np.full(freqs.shape, 0.5 + 0j, dtype=np.complex128)

    monkeypatch.setattr(tessera_mod.TesseraTSV, "s21", fake_s21)
    return calls


# -----------------------------------------------------------------------------------
# Validation -- no surrogate needed.
# -----------------------------------------------------------------------------------

def test_unknown_source_raises():
    with pytest.raises(ValueError, match="source must be one of"):
        InterconnectBlock(source="not-a-source")


def test_source_and_transfer_csv_are_mutually_exclusive():
    from e2e.blocks import TESSERA_INTERCONNECT_CSV

    with pytest.raises(ValueError, match="not both"):
        InterconnectBlock(source="tessera", transfer_csv=TESSERA_INTERCONNECT_CSV)


def test_default_shipped_geometry_validates_clean():
    """SHIPPED_TSV_DESIGN is upstream's OWN canonical demo point (README quickstart,
    examples/predict_smatrix.py, config.yaml optimization.fixed_params); VALID_RANGES
    is measured+widened to always include it (tessera.py), so plain `source='tessera'`
    with no override or band must NOT raise."""
    blk = InterconnectBlock(source="tessera")
    assert blk.tessera_params["pitch_um"] == pytest.approx(60.0)
    assert blk.tessera_params["liner_um"] == pytest.approx(0.5)
    assert blk.scale == pytest.approx(1.0)


def test_out_of_range_presented_geometry_names_both_numbers():
    """pitch=80 um is outside even the widened envelope (max 60 um) -- the error must
    name the PRESENTED value, the MODEL value (equal at scale=1), and the range."""
    with pytest.raises(ValueError, match=r"pitch_um.*presented value 80.*model value 80.*\[20\.7"):
        InterconnectBlock(source="tessera", tessera_params=dict(IN_RANGE_PARAMS, pitch_um=80.0))


def test_unknown_tessera_param_key_raises():
    with pytest.raises(ValueError, match="unknown Tessera parameter"):
        InterconnectBlock(source="tessera",
                          tessera_params=dict(IN_RANGE_PARAMS, bogus_um=1.0))


def test_unknown_arrangement_raises():
    with pytest.raises(ValueError, match="unknown Tessera arrangement"):
        InterconnectBlock(source="tessera", tessera_params=IN_RANGE_PARAMS,
                          tessera_arrangement="no-such-grid")


def test_valid_params_construct_and_describe():
    blk = InterconnectBlock(source="tessera", tessera_params=IN_RANGE_PARAMS)
    assert set(TESSERA_DESIGN_PARAMS) == set(blk.tessera_params)
    desc = blk.describe()
    assert "Tessera TSV surrogate" in desc and "radius 4" in desc


def test_not_importable_raises_import_error_with_install_line(monkeypatch):
    """Simulates the missing-package case (not the missing-checkpoint case this box
    actually has): find_spec('tessera') -> None, everything else present."""
    monkeypatch.setattr(
        tessera_mod.importlib.util, "find_spec",
        lambda name, *a, **k: None if name == "tessera" else object(),
    )
    blk = InterconnectBlock(source="tessera", tessera_params=IN_RANGE_PARAMS,
                            band_hz=PIPELINE_BAND, scale=1.0)
    with pytest.raises(ImportError, match="requirements-tessera.txt"):
        blk.apply_interconnect(_ones_frame(5))


def test_default_and_other_modes_unaffected():
    """source=None is unchanged: constructing without it never imports the surrogate
    subpackage's third-party dependency (nothing here should raise)."""
    frame = _ones_frame(16)
    boxcar = InterconnectBlock().apply_interconnect(frame)
    passthrough = InterconnectBlock(case="passthrough").apply_interconnect(frame)
    assert not torch.equal(boxcar, frame)
    assert torch.equal(passthrough, frame)


# -----------------------------------------------------------------------------------
# Scale model -- spy on TesseraTSV.s21, no surrogate/checkpoint needed.
# -----------------------------------------------------------------------------------

def test_scale_defaults_to_one_with_no_band():
    assert InterconnectBlock(source="tessera").scale == pytest.approx(1.0)


def test_scale_auto_derives_for_ka_band():
    """28.5-31.5 GHz -> 2 (F89: right-signed crosstalk trend below ~23 GHz)."""
    blk = InterconnectBlock(source="tessera", band_hz=PIPELINE_BAND)
    assert blk.scale == pytest.approx(2.0)


def test_scale_auto_derives_for_77ghz_corpus_band():
    """75-81 GHz -- e2e/ml/chain_generate.py / webapp/pipeline_runner.py -- -> 4."""
    blk = InterconnectBlock(source="tessera", band_hz=ML_CORPUS_BAND)
    assert blk.scale == pytest.approx(4.0)


def test_scale_one_reproduces_unscaled_behaviour_exactly(monkeypatch):
    calls = _spy_s21(monkeypatch)
    n_freqs = 4
    blk = InterconnectBlock(source="tessera", tessera_params=IN_RANGE_PARAMS,
                            band_hz=PIPELINE_BAND, scale=1.0)
    blk.apply_interconnect(_ones_frame(n_freqs))

    assert len(calls) == 1
    call = calls[0]
    expected_freqs = np.linspace(*PIPELINE_BAND, n_freqs)
    assert np.allclose(call["freqs"], expected_freqs)          # unscaled frequency axis
    for name, value in IN_RANGE_PARAMS.items():
        assert call[name] == pytest.approx(value)               # unscaled geometry/temp


def test_scale_two_evaluates_model_at_scaled_geometry_and_half_frequency(monkeypatch):
    calls = _spy_s21(monkeypatch)
    n_freqs = 4
    blk = InterconnectBlock(source="tessera", tessera_params=HALF_GEOMETRY_PARAMS,
                            band_hz=PIPELINE_BAND, scale=2.0)
    blk.apply_interconnect(_ones_frame(n_freqs))

    assert len(calls) == 1
    call = calls[0]
    expected_freqs = np.linspace(*PIPELINE_BAND, n_freqs) / 2.0
    assert np.allclose(call["freqs"], expected_freqs)
    for name in ("radius_um", "pitch_um", "height_um", "liner_um"):
        assert call[name] == pytest.approx(IN_RANGE_PARAMS[name])   # presented x2
    assert call["temperature_k"] == pytest.approx(HALF_GEOMETRY_PARAMS["temperature_k"])


# -----------------------------------------------------------------------------------
# Live prediction -- needs the surrogate installed + a checkpoint.
# -----------------------------------------------------------------------------------

# One small axis: each point is a separate surrogate forward pass (~5 ms uncached).
TINY_N_FREQS = 5


@requires_surrogate
def test_tessera_response_is_passive_at_in_range_geometry(tmp_path):
    cache = SurrogateCache(cache_dir=tmp_path)
    blk = InterconnectBlock(source="tessera", tessera_params=IN_RANGE_PARAMS,
                            band_hz=PIPELINE_BAND, tessera_cache=cache, scale=1.0)
    H = blk.apply_interconnect(_ones_frame(TINY_N_FREQS))[0, 0, 0, :]
    assert torch.all(torch.abs(H) <= 1.0 + 1e-6)


@requires_surrogate
def test_repeated_evaluation_hits_the_cache(tmp_path):
    cache = SurrogateCache(cache_dir=tmp_path)
    blk = InterconnectBlock(source="tessera", tessera_params=IN_RANGE_PARAMS,
                            band_hz=PIPELINE_BAND, tessera_cache=cache, scale=1.0)
    frame = _ones_frame(TINY_N_FREQS)
    H1 = blk.apply_interconnect(frame)
    assert cache.stats()["misses"] >= 1
    H2 = blk.apply_interconnect(frame)
    assert cache.stats()["hits"] >= 1
    assert torch.equal(H1, H2)


@requires_surrogate
def test_pure_function_matches_block(tmp_path):
    """`tessera_s21_for_axis` is what the block delegates to; pin they agree, since the
    webapp is expected to call the pure function directly to draw the response."""
    cache = SurrogateCache(cache_dir=tmp_path)
    freqs = np.linspace(*PIPELINE_BAND, TINY_N_FREQS)
    s21 = tessera_s21_for_axis(freqs, params=IN_RANGE_PARAMS, cache=cache, scale=1.0)
    blk = InterconnectBlock(source="tessera", tessera_params=IN_RANGE_PARAMS,
                            band_hz=PIPELINE_BAND, tessera_cache=cache, scale=1.0)
    H = blk.apply_interconnect(_ones_frame(TINY_N_FREQS))[0, 0, 0, :]
    assert np.allclose(H.cpu().numpy(), s21.astype(np.complex64), atol=1e-6)


@requires_surrogate
def test_scale_two_is_passive_at_a_real_checkpoint(tmp_path):
    """One real (small, cached) evaluation at scale=2, to confirm the scaled call
    reaches the actual checkpoint and still returns a physical response."""
    cache = SurrogateCache(cache_dir=tmp_path)
    blk = InterconnectBlock(source="tessera", tessera_params=HALF_GEOMETRY_PARAMS,
                            band_hz=PIPELINE_BAND, tessera_cache=cache, scale=2.0)
    H = blk.apply_interconnect(_ones_frame(TINY_N_FREQS))[0, 0, 0, :]
    assert torch.all(torch.abs(H) <= 1.0 + 1e-6)
