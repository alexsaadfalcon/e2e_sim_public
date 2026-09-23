"""Tests for ``e2e.interconnect_surrogate`` -- the optional Tessera TSV wrapper.

Everything here runs with the surrogate ABSENT: the availability logic, the cache,
the passivity policy and the import-time contract are all testable without
``tessera``, ``torch-geometric`` or the 1.6 MB checkpoint, and that is deliberate --
CI has none of them. The handful of tests that genuinely need a live prediction are
guarded by ``requires_surrogate`` and skip cleanly.

The one contract that would be silently broken by a careless edit, and so is pinned
first: importing this subpackage must not import torch. See
``tests/test_webapp.py::_import_without_torch`` for the same pattern and why it is
done out-of-process.
"""

import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from e2e.interconnect_surrogate import (
    ARRANGEMENTS,
    SHIPPED_TSV_DESIGN,
    VALID_RANGES,
    PassivityError,
    SurrogateCache,
    TesseraTSV,
    available,
    checkpoint_dir,
    ring_arrangement,
)
from e2e.interconnect_surrogate import cache as cache_mod
from e2e.interconnect_surrogate import fetch as fetch_mod
from e2e.interconnect_surrogate import tessera as tessera_mod

_REPO_ROOT = Path(__file__).resolve().parent.parent

requires_surrogate = pytest.mark.skipif(
    not available(),
    reason="the Tessera surrogate (tessera-tsv + checkpoint) is not installed; "
           "see requirements-tessera.txt",
)

# A 5-point grid across the pipeline band. Small on purpose: each point is a separate
# forward pass (~5 ms), so a 1000-point frame grid would make this suite slow.
TINY_GRID = np.linspace(28.5e9, 31.5e9, 5)


# =============================================================================
# Import-time contract
# =============================================================================

def test_import_does_not_import_torch():
    """``import e2e.interconnect_surrogate`` must not pull in torch.

    Run in a fresh subprocess: deleting/reloading torch in the live test process
    leaves it half-initialised and breaks every later torch test.
    """
    code = (
        "import importlib, sys; "
        "importlib.import_module('e2e.interconnect_surrogate'); "
        "sys.exit(0 if 'torch' not in sys.modules else 3)"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(_REPO_ROOT),
                          capture_output=True, text=True)
    assert proc.returncode == 0, (
        "importing e2e.interconnect_surrogate must succeed WITHOUT importing torch "
        f"(rc={proc.returncode}); stderr:\n{proc.stderr}"
    )


def test_import_does_not_import_tessera_or_pyg():
    """Nor the third-party surrogate itself -- that is the point of the wrapper."""
    code = (
        "import importlib, sys; "
        "importlib.import_module('e2e.interconnect_surrogate'); "
        "leaked = [m for m in ('tessera', 'torch_geometric') if m in sys.modules]; "
        "sys.exit(3 if leaked else 0)"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(_REPO_ROOT),
                          capture_output=True, text=True)
    assert proc.returncode == 0, (
        f"importing the wrapper leaked a heavy import (rc={proc.returncode}); "
        f"stderr:\n{proc.stderr}"
    )


def test_constructing_the_wrapper_is_lazy():
    """``TesseraTSV()`` must not load anything -- it lives in a module-level registry."""
    obj = TesseraTSV()
    assert obj._loaded is None
    assert isinstance(repr(obj), str)  # repr must work before loading, too


# =============================================================================
# available() / checkpoint_dir() -- the absent-package path
# =============================================================================

def test_available_false_when_package_missing(monkeypatch):
    """With ``tessera`` unimportable, ``available()`` is False and does not raise."""
    real_find_spec = tessera_mod.importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "tessera":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(tessera_mod.importlib.util, "find_spec", fake_find_spec)
    assert available() is False


def test_available_false_when_pyg_missing(monkeypatch):
    real_find_spec = tessera_mod.importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "torch_geometric":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(tessera_mod.importlib.util, "find_spec", fake_find_spec)
    assert available() is False


def test_available_false_and_silent_when_find_spec_explodes(monkeypatch):
    """A broken distribution makes ``find_spec`` raise; a UI callback must not."""
    def boom(name, *args, **kwargs):
        raise ValueError("broken distribution metadata")

    monkeypatch.setattr(tessera_mod.importlib.util, "find_spec", boom)
    assert available() is False


def test_available_false_when_checkpoint_missing(tmp_path, monkeypatch):
    """The pip-install-from-git case: package present, weights absent.

    ``pip install git+https://github.com/HiPerCAS/tessera.git`` ships the Python
    package but not ``models/``, so this is the realistic failure, not a corner case.
    """
    monkeypatch.setenv("E2E_TESSERA_MODELS_DIR", str(tmp_path))  # empty dir
    monkeypatch.delenv("TESSERA_REPO", raising=False)
    monkeypatch.setattr(tessera_mod.importlib.util, "find_spec",
                        lambda name, *a, **k: None if name == "tessera" else object())
    # Candidate 5 (the `fetch()` checkout) must not resolve here either, or this
    # "nothing is configured" scenario would pass on a box that happens to have
    # already run `python -m e2e.interconnect_surrogate.fetch`.
    monkeypatch.setattr(fetch_mod, "DEFAULT_CHECKOUT_DIR", tmp_path / "no_such_fetch_checkout")
    assert checkpoint_dir(tmp_path) is None
    assert available(tmp_path) is False


def test_checkpoint_dir_finds_explicit_directory(tmp_path, monkeypatch):
    """Both files must be present; one alone is not a checkpoint directory.

    The environment fallbacks are cleared so this tests the explicit argument alone
    (on a developer box with TESSERA_REPO set, a half-populated directory would
    otherwise correctly fall through to the real checkout and mask the assertion).
    """
    monkeypatch.delenv("E2E_TESSERA_MODELS_DIR", raising=False)
    monkeypatch.delenv("TESSERA_REPO", raising=False)
    monkeypatch.setattr(tessera_mod.importlib.util, "find_spec",
                        lambda name, *a, **k: None)
    monkeypatch.setattr(fetch_mod, "DEFAULT_CHECKOUT_DIR", tmp_path / "no_such_fetch_checkout")
    (tmp_path / "best_model.pth").write_bytes(b"not a real checkpoint")
    assert checkpoint_dir(tmp_path) is None
    (tmp_path / "input_scaler.pt").write_bytes(b"nor is this")
    assert checkpoint_dir(tmp_path) == tmp_path


def test_checkpoint_dir_honours_tessera_repo_env(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    (models / "best_model.pth").write_bytes(b"x")
    (models / "input_scaler.pt").write_bytes(b"x")
    monkeypatch.delenv("E2E_TESSERA_MODELS_DIR", raising=False)
    monkeypatch.setenv("TESSERA_REPO", str(tmp_path))
    assert checkpoint_dir() == models


def test_loading_without_the_package_raises_a_useful_error(monkeypatch):
    """The error must say what to install, not just fail an import deep inside."""
    monkeypatch.setattr(tessera_mod, "available", lambda models_dir=None: False)
    with pytest.raises(ModuleNotFoundError) as exc:
        TesseraTSV()._load()
    assert "requirements-tessera.txt" in str(exc.value) or "checkpoint" in str(exc.value)


# =============================================================================
# Declared parameter ranges (the GUI bounds its knobs to these)
# =============================================================================

def test_valid_ranges_are_ordered_and_complete():
    expected = {"radius_um", "pitch_um", "height_um", "liner_um",
                "temperature_k", "freq_hz"}
    assert set(VALID_RANGES) == expected
    for name, (lo, hi) in VALID_RANGES.items():
        assert lo < hi, f"{name} range is inverted"


def test_shipped_geometry_is_inside_valid_ranges():
    """RETRACTED 2026-09-23 (was `..._is_documented_as_partly_out_of_range`): the old
    VALID_RANGES was inferred from the input scaler under a uniform-sampling
    assumption, which placed pitch=60um/liner=0.5um outside the "valid" box. That was
    wrong -- SHIPPED_TSV_DESIGN is upstream's OWN canonical demo point (README
    quickstart, examples/predict_smatrix.py, config.yaml optimization.fixed_params) --
    so VALID_RANGES is now measured from upstream's 40-sample
    examples/arrangements_sample.csv and WIDENED to always include it (see tessera.py).
    Pinned as a test because a false "this needs extrapolation" belief is the failure
    mode most likely to recur if this ever regresses.
    """
    for name, value in SHIPPED_TSV_DESIGN.items():
        lo, hi = VALID_RANGES[name]
        assert lo <= value <= hi, f"{name}={value} outside widened VALID_RANGES {(lo, hi)}"


def test_our_band_is_inside_the_frequency_range():
    lo, hi = VALID_RANGES["freq_hz"]
    assert lo <= 28.5e9 and 31.5e9 <= hi


def test_out_of_range_parameters_warn():
    """pitch=80 um is outside VALID_RANGES even after the widening above (max 60 um);
    SHIPPED_TSV_DESIGN no longer is, so it can't be used to exercise this path."""
    obj = TesseraTSV()
    with pytest.warns(UserWarning, match="outside"):
        obj._check_ranges({"radius_um": 5.0, "pitch_um": 80.0, "height_um": 100.0,
                           "liner_um": 0.5, "temperature_k": 300.0}, TINY_GRID)


def test_out_of_range_warning_can_be_silenced():
    obj = TesseraTSV(warn_out_of_range=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes an error
        obj._check_ranges(dict(SHIPPED_TSV_DESIGN, pitch_um=80.0), TINY_GRID)


# =============================================================================
# Arrangements
# =============================================================================

def test_ring_arrangement_is_one_signal_in_a_ground_ring():
    arr = ring_arrangement(3)
    assert arr.shape == (3, 3)
    assert (arr == 1).sum() == 1
    assert arr[1, 1] == 1
    assert (arr == -1).sum() == 8


@pytest.mark.parametrize("bad_size", [2, 4, 1])
def test_ring_arrangement_rejects_even_or_tiny_sizes(bad_size):
    with pytest.raises(ValueError):
        ring_arrangement(bad_size)


def test_resolve_grid_accepts_name_int_and_array():
    assert TesseraTSV._resolve_grid("ring3x3").shape == (3, 3)
    assert TesseraTSV._resolve_grid(5).shape == (5, 5)
    explicit = np.array([[1, -1], [-1, 1]], dtype=np.int8)
    assert np.array_equal(TesseraTSV._resolve_grid(explicit), explicit)
    assert TesseraTSV._resolve_grid(None).shape == (3, 3)


def test_resolve_grid_rejects_unknown_name_and_signal_free_grid():
    with pytest.raises(ValueError, match="unknown arrangement"):
        TesseraTSV._resolve_grid("no-such-arrangement")
    with pytest.raises(ValueError, match="at least one signal"):
        TesseraTSV._resolve_grid(np.full((3, 3), -1, dtype=np.int8))


def test_named_arrangements_have_the_signal_counts_their_names_imply():
    assert (ARRANGEMENTS["ring3x3"] == 1).sum() == 1
    assert (ARRANGEMENTS["checker3x3"] == 1).sum() >= 2


# =============================================================================
# Passivity guard
# =============================================================================

def test_passivity_guard_raises_on_gain():
    """|S21| > 0 dB is gain from a passive structure -- refuse it by default."""
    obj = TesseraTSV(passivity="raise")
    with pytest.raises(PassivityError, match="gain"):
        obj._passivity_scale(7.16, "h=800 um test design")  # +17.1 dB, the real case


def test_passivity_guard_passes_a_lossy_response_untouched():
    obj = TesseraTSV(passivity="raise")
    assert obj._passivity_scale(0.9354, "lossy") == 1.0   # -0.58 dB
    assert obj._passivity_scale(1.0, "exactly lossless") == 1.0


def test_passivity_clamp_warns_loudly_and_normalises():
    obj = TesseraTSV(passivity="clamp")
    with pytest.warns(RuntimeWarning, match="NOT the model's output"):
        scale = obj._passivity_scale(2.0, "some design")
    assert scale == pytest.approx(0.5)


def test_passivity_ignore_returns_raw():
    obj = TesseraTSV(passivity="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert obj._passivity_scale(7.16, "raw") == 1.0


def test_unknown_passivity_policy_is_rejected_at_construction():
    with pytest.raises(ValueError, match="passivity"):
        TesseraTSV(passivity="clip")


# =============================================================================
# Cache
# =============================================================================

@pytest.fixture
def params():
    return dict(SHIPPED_TSV_DESIGN)


def test_cache_key_is_deterministic(params):
    a = cache_mod.cache_key(params, TINY_GRID, ARRANGEMENTS["ring3x3"], "fp")
    b = cache_mod.cache_key(dict(params), TINY_GRID.copy(),
                            ARRANGEMENTS["ring3x3"].copy(), "fp")
    assert a == b


def test_cache_key_absorbs_slider_float_noise(params):
    """A GUI slider emits 60.000000000000004; that must not miss the cache."""
    noisy = dict(params, pitch_um=60.0 + 4e-14, temperature_k=300.0 + 1e-13)
    assert (cache_mod.cache_key(params, TINY_GRID, ARRANGEMENTS["ring3x3"], "fp")
            == cache_mod.cache_key(noisy, TINY_GRID, ARRANGEMENTS["ring3x3"], "fp"))


@pytest.mark.parametrize("changed", [
    {"pitch_um": 25.0},
    {"temperature_k": 400.0},
    {"height_um": 80.0},
])
def test_cache_key_separates_different_designs(params, changed):
    assert (cache_mod.cache_key(params, TINY_GRID, ARRANGEMENTS["ring3x3"], "fp")
            != cache_mod.cache_key(dict(params, **changed), TINY_GRID,
                                   ARRANGEMENTS["ring3x3"], "fp"))


def test_cache_key_separates_grid_arrangement_and_fingerprint(params):
    base = cache_mod.cache_key(params, TINY_GRID, ARRANGEMENTS["ring3x3"], "fp")
    other_freqs = cache_mod.cache_key(params, np.linspace(28.5e9, 31.5e9, 7),
                                      ARRANGEMENTS["ring3x3"], "fp")
    other_arr = cache_mod.cache_key(params, TINY_GRID, ARRANGEMENTS["checker3x3"], "fp")
    other_fp = cache_mod.cache_key(params, TINY_GRID, ARRANGEMENTS["ring3x3"], "fp2")
    assert len({base, other_freqs, other_arr, other_fp}) == 4


def test_cache_miss_then_hit_roundtrips_complex_arrays(tmp_path, params):
    cache = SurrogateCache(cache_dir=tmp_path)
    key = cache_mod.cache_key(params, TINY_GRID, ARRANGEMENTS["ring3x3"], "fp")
    assert cache.get(key) is None
    assert (cache.hits, cache.misses) == (0, 1)

    value = np.array([0.9 + 0.1j, 0.8 - 0.2j], dtype=np.complex128)
    cache.put(key, {"s": value})
    hit = cache.get(key)
    assert hit is not None
    np.testing.assert_allclose(hit["s"], value)
    assert hit["s"].dtype == np.complex128
    assert (cache.hits, cache.misses) == (1, 1)


def test_cache_survives_a_new_instance_on_the_same_directory(tmp_path, params):
    """On-disk, not in-memory: a fresh process must serve the pre-computed knob."""
    key = cache_mod.cache_key(params, TINY_GRID, ARRANGEMENTS["ring3x3"], "fp")
    SurrogateCache(cache_dir=tmp_path).put(key, {"s": np.ones(3, dtype=np.complex128)})
    assert SurrogateCache(cache_dir=tmp_path).get(key) is not None


def test_disabled_cache_is_a_passthrough(tmp_path, params):
    cache = SurrogateCache(cache_dir=tmp_path, enabled=False)
    key = "anything"
    cache.put(key, {"s": np.ones(3)})
    assert cache.get(key) is None
    assert list(tmp_path.glob("*.npz")) == []


def test_corrupt_cache_entry_reads_as_a_miss_not_an_exception(tmp_path):
    """A broken cache must never break a prediction."""
    cache = SurrogateCache(cache_dir=tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "deadbeef.npz").write_bytes(b"this is not an npz file")
    assert cache.get("deadbeef") is None
    assert cache.errors == 1


def test_cache_put_failure_is_swallowed_and_counted(tmp_path, monkeypatch):
    cache = SurrogateCache(cache_dir=tmp_path)
    monkeypatch.setattr(cache_mod.np, "savez",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    cache.put("k", {"s": np.ones(2)})
    assert cache.errors == 1


def test_default_cache_dir_honours_the_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("E2E_INTERCONNECT_CACHE_DIR", str(tmp_path / "elsewhere"))
    assert cache_mod.default_cache_dir() == tmp_path / "elsewhere"


def test_default_cache_dir_is_outside_the_repo(monkeypatch):
    """Derived third-party artifacts must never land in the working tree."""
    monkeypatch.delenv("E2E_INTERCONNECT_CACHE_DIR", raising=False)
    assert _REPO_ROOT not in cache_mod.default_cache_dir().resolve().parents


# =============================================================================
# Live prediction (skipped unless the optional dependency is installed)
# =============================================================================

@requires_surrogate
def test_s21_shape_and_dtype_on_a_tiny_grid(tmp_path):
    obj = TesseraTSV(cache=SurrogateCache(cache_dir=tmp_path), warn_out_of_range=False)
    s21 = obj.s21(TINY_GRID, **SHIPPED_TSV_DESIGN)
    assert s21.shape == TINY_GRID.shape
    assert s21.dtype == np.complex128
    assert np.all(np.isfinite(s21))
    assert np.abs(s21).max() <= 1.0, "a passive TSV cannot have |S21| > 1"


@requires_surrogate
def test_s21_db_is_negative_insertion_loss(tmp_path):
    obj = TesseraTSV(cache=SurrogateCache(cache_dir=tmp_path), warn_out_of_range=False)
    db = obj.s21_db(TINY_GRID, **SHIPPED_TSV_DESIGN)
    assert db.shape == TINY_GRID.shape
    assert db.dtype == np.float64
    assert np.all(db <= 0.0)


@requires_surrogate
def test_s_matrix_shape_matches_the_arrangement(tmp_path):
    obj = TesseraTSV(cache=SurrogateCache(cache_dir=tmp_path), warn_out_of_range=False)
    grid = ARRANGEMENTS["checker3x3"]
    n_sig = int((grid == 1).sum())
    s = obj.s_matrix(TINY_GRID[:2], grid=grid, **SHIPPED_TSV_DESIGN)
    assert s.shape == (2, 2 * n_sig, 2 * n_sig)
    assert s.dtype == np.complex128


@requires_surrogate
def test_crosstalk_needs_more_than_one_signal_via(tmp_path):
    obj = TesseraTSV(cache=SurrogateCache(cache_dir=tmp_path), warn_out_of_range=False)
    with pytest.raises(ValueError, match="at least|>= 2"):
        obj.crosstalk_db(TINY_GRID[:1], grid="ring3x3", **SHIPPED_TSV_DESIGN)


@requires_surrogate
def test_crosstalk_returns_next_and_fext_per_frequency(tmp_path):
    obj = TesseraTSV(cache=SurrogateCache(cache_dir=tmp_path), warn_out_of_range=False)
    out = obj.crosstalk_db(TINY_GRID[:2], grid="checker3x3", **SHIPPED_TSV_DESIGN)
    assert out["next_db"].shape == (2,)
    assert out["fext_db"].shape == (2,)
    assert out["n_signals"] >= 2
    assert np.all(out["next_db"] < 0.0) and np.all(out["fext_db"] < 0.0)


@requires_surrogate
def test_cache_makes_the_second_identical_call_a_hit(tmp_path):
    cache = SurrogateCache(cache_dir=tmp_path)
    obj = TesseraTSV(cache=cache, warn_out_of_range=False)
    obj.s21(TINY_GRID[:2], **SHIPPED_TSV_DESIGN)
    assert (cache.hits, cache.misses) == (0, 1)
    obj.s21(TINY_GRID[:2], **SHIPPED_TSV_DESIGN)
    assert (cache.hits, cache.misses) == (1, 1)


@requires_surrogate
def test_extrapolated_height_trips_the_passivity_guard(tmp_path):
    """h = 800 um / 30 GHz returns about +17 dB from the public checkpoint
    (measured 2026-09-23). If this ever stops raising, the checkpoint changed --
    re-measure before relaxing the guard."""
    obj = TesseraTSV(cache=SurrogateCache(cache_dir=tmp_path),
                     warn_out_of_range=False, passivity="raise")
    with pytest.raises(PassivityError):
        obj.s21(np.array([30e9]), radius_um=5.0, pitch_um=60.0, height_um=800.0,
                liner_um=0.5, temperature_k=300.0)


@requires_surrogate
def test_public_checkpoint_does_not_reproduce_the_shipped_csv(tmp_path):
    """A pinned RETRACTION, not a feature test.

    e2e/data/interconnect/tessera_tsv_s21.csv says about -7.5 dB at 30 GHz; the
    PUBLIC checkpoint says about -0.6 dB. The file came from an HFSS-finetuned
    checkpoint that was not released. If this test ever fails, the public checkpoint
    has changed and the "not regenerable from the public code" note in that data
    README must be re-checked.
    """
    obj = TesseraTSV(cache=SurrogateCache(cache_dir=tmp_path), warn_out_of_range=False)
    db = float(obj.s21_db(np.array([30e9]), **SHIPPED_TSV_DESIGN)[0])
    assert -1.5 < db < 0.0, f"public checkpoint gave {db:.3f} dB at 30 GHz"
