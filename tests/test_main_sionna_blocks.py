"""Tests for e2e.main.main_sionna_blocks -- the scripted radar pipeline example.

Fast: synthetic environment frames via the shared `make_env_block` fixture (no
Sionna, no .pkl on disk). Mirrors tests/test_main_tx_nonideality.py's pattern
(monkeypatch FIG_DIR, assert on returned outputs / written files, show=False by
default so most tests never touch disk).
"""

import pytest

torch = pytest.importorskip("torch")

import matplotlib
matplotlib.use("Agg")            # matches the module's own convention

import e2e.main.main_sionna_blocks as m


# --------------------------------------------------------------------------------
# end-to-end run on synthetic frames
# --------------------------------------------------------------------------------
def test_main_runs_end_to_end_and_returns_expected_keys(make_env_block):
    env = make_env_block(n_frames=2, n_freqs=32)
    outputs = m.main(environment_block=env, n_steps=2, show=False)
    expected_keys = {"fft", "range_az", "range_el", "subspace_err"}
    assert expected_keys.issubset(outputs.keys())
    assert len(outputs["subspace_err"]) == 2
    assert len(outputs["fft"]) == 2


def test_main_does_not_touch_disk_when_show_is_false(tmp_path, monkeypatch, make_env_block):
    monkeypatch.setattr(m, "FIG_DIR", str(tmp_path))
    env = make_env_block(n_frames=2, n_freqs=32)
    m.main(environment_block=env, n_steps=2, show=False)
    assert list(tmp_path.iterdir()) == []


def test_main_writes_expected_figures_when_show_is_true(tmp_path, monkeypatch, make_env_block):
    monkeypatch.setattr(m, "FIG_DIR", str(tmp_path))
    env = make_env_block(n_frames=2, n_freqs=32)
    outputs = m.main(environment_block=env, n_steps=2, show=True)

    files = {p.name for p in tmp_path.iterdir()}
    assert "sionna_blocks_subspace_err.png" in files
    for i in range(len(outputs["fft"])):
        assert f"sionna_blocks_az_el_frame{i}.png" in files


# --------------------------------------------------------------------------------
# missing-.pkl UX: friendly message, not a bare FileNotFoundError
# --------------------------------------------------------------------------------
def test_main_missing_pkl_raises_friendly_message(tmp_path, monkeypatch):
    import e2e.environment.sionna_iterator as si

    # Point the munich .pkl path at a file that doesn't exist (empty tmp dir).
    monkeypatch.setattr(si, "SIONNA_MUNICH_PATH", str(tmp_path / "munich.pkl"))

    with pytest.raises(FileNotFoundError) as exc_info:
        m.main(scenario_name="munich")

    msg = str(exc_info.value)
    # A bare FileNotFoundError from open() only ever mentions the path -- the
    # friendly message must point the user at how to fix it.
    assert "sionna_simple_channel" in msg
    assert "scenario_runner" in msg
    assert "No precomputed frames found for scenario 'munich'" in msg
