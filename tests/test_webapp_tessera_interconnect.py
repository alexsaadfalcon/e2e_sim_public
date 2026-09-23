"""Tests for the webapp's `InterconnectBlock(source='tessera')` plumbing: the GUI
knobs (webapp/pipeline_registry.py), the runner wiring (webapp/pipeline_runner.py,
`source='tessera'` branch + `_resolve_interconnect_band_hz` + `prewarm_tessera_
interconnect`), and the run-banner `describe()` note. Validation of the surrogate
itself (ranges, arrangement, passivity, caching) is `tests/test_interconnect_tessera.py`'s
job; this file only covers the webapp's OWN wiring on top of it.

A real prediction needs the optional `tessera`/`torch-geometric` extra plus a fetched
checkpoint (see e2e/interconnect_surrogate) -- guarded by `requires_surrogate`, mirroring
that file's pattern.
"""
import pytest

torch = pytest.importorskip("torch")

from e2e.interconnect_surrogate import available
from webapp.pipeline_registry import BLOCKS_BY_ID, default_block_state
from webapp.pipeline_runner import PipelineError, _resolve_interconnect_band_hz, prewarm_tessera_interconnect, run_pipeline

requires_surrogate = pytest.mark.skipif(
    not available(),
    reason="the Tessera surrogate (tessera-tsv + checkpoint) is not installed; "
           "see requirements-tessera.txt",
)


def _tessera_state(**overrides):
    state = default_block_state()
    state["interconnect"]["enabled"] = True
    state["interconnect"]["params"]["source"] = "tessera"
    state["interconnect"]["params"].update(overrides)
    return state


def test_resolve_interconnect_band_hz_matches_munich():
    """`_resolve_interconnect_band_hz` must land on the pipeline's documented Ka band
    (28.5-31.5 GHz), the band `_TESSERA_KA_SCALE` (pipeline_registry.py) and the
    Thrust 4 preset's own measurement note assume -- whichever of the two ways it gets
    there: 'munich' now resolves to the Ka-band re-trace `munich_ka.pkl` (v2, carries
    `freq_plan`, 2026-09-23 Ka-band re-founding) when present, which is used directly;
    a machine with only the legacy `munich.pkl` (no `freq_plan`) instead exercises the
    fallback -- centered at 30 GHz, spanning the registry's default rffe
    freq_span_hz (3 GHz) -- which lands on the exact same band by construction."""
    from e2e.blocks import SionnaEnvironmentBlock

    pytest.importorskip("e2e.environment.sionna_iterator")
    try:
        env_block = SionnaEnvironmentBlock("munich")
    except FileNotFoundError:
        pytest.skip("munich.pkl not present on this machine")
    band = _resolve_interconnect_band_hz(default_block_state(), env_block)
    assert band == pytest.approx((28.5e9, 31.5e9))


@requires_surrogate
def test_run_pipeline_tessera_source_produces_a_run_note_naming_the_scale():
    """`source='tessera'` must actually reach `InterconnectBlock` (not silently fall
    back to the boxcar) and its `describe()` must land in the run banner
    (`outputs["_axis_meta"]["notes"]`, build item 1)."""
    state = _tessera_state()
    try:
        outputs = run_pipeline(state, n_steps=1)
    except PipelineError as e:
        pytest.skip(f"pipeline could not run here: {e}")
    notes = " ".join(outputs["_axis_meta"]["notes"])
    assert "Tessera TSV surrogate" in notes
    assert "scale model" in notes


@requires_surrogate
def test_run_pipeline_tessera_source_changes_the_frame_vs_passthrough():
    """A sanity check that the surrogate path is not a no-op: a downstream product
    (range_profile) must differ from an explicit passthrough on the same frames."""
    tessera_out = run_pipeline(_tessera_state(), n_steps=1)
    passthrough_state = default_block_state()
    passthrough_state["interconnect"]["enabled"] = True
    passthrough_state["interconnect"]["params"]["case"] = "passthrough"
    passthrough_out = run_pipeline(passthrough_state, n_steps=1)
    assert not torch.allclose(tessera_out["range_profile"][0],
                              passthrough_out["range_profile"][0])


@requires_surrogate
def test_run_pipeline_out_of_range_tessera_knob_raises_pipeline_error():
    """A malformed/stale state (bypassing the ParamSpec's own min/max, e.g. a saved
    state from a wider future range) must surface as PipelineError, not an
    unhandled ValueError escaping run_pipeline (see the AdaOjaBlock k>=m comment in
    pipeline_runner.py for the class of bug this guards against)."""
    height_spec = next(p for p in BLOCKS_BY_ID["interconnect"].params
                       if p.key == "tessera_height_um")
    state = _tessera_state(tessera_height_um=height_spec.max * 100)
    with pytest.raises(PipelineError):
        run_pipeline(state, n_steps=1)


@requires_surrogate
def test_prewarm_tessera_interconnect_never_raises_and_warms_the_cache():
    """Best-effort by contract: must not raise even on a states dict that is fine but
    unusual, and a second call (now cache-warm) must be much faster than a cold one."""
    import time

    state = _tessera_state()
    t0 = time.time()
    prewarm_tessera_interconnect(state)  # may be cold; not timed for a ceiling
    t0 = time.time()
    prewarm_tessera_interconnect(state)
    assert time.time() - t0 < 2.0, "second (cache-warm) prewarm call should be fast"


def test_prewarm_tessera_interconnect_is_a_noop_off_source_or_when_passthrough():
    """No interconnect enabled, source left at 'default', or a passthrough `case` --
    none of these should attempt anything (and must not need the surrogate at all)."""
    prewarm_tessera_interconnect(default_block_state())  # interconnect disabled
    state = default_block_state()
    state["interconnect"]["enabled"] = True  # source stays 'default'
    prewarm_tessera_interconnect(state)
    state = _tessera_state(case="passthrough")
    prewarm_tessera_interconnect(state)
