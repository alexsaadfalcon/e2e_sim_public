"""Tests for the data-driven InterconnectBlock transfer-function mode and the shipped
Tessera TSV S21(f) CSV (e2e/data/interconnect/tessera_tsv_s21.csv).

These use only the committed CSV + numpy/torch -- no external interconnect model.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from e2e.blocks import (
    InterconnectBlock,
    load_interconnect_transfer,
    TESSERA_INTERCONNECT_CSV,
    device,
)

PIPELINE_BAND = (28.5e9, 31.5e9)


def test_shipped_csv_loads_and_is_physical():
    freq, s21 = load_interconnect_transfer(TESSERA_INTERCONNECT_CSV)
    # ascending 1..40 GHz sweep
    assert np.all(np.diff(freq) > 0)
    assert freq[0] == pytest.approx(1e9)
    assert freq[-1] == pytest.approx(40e9)
    assert s21.dtype == np.complex128 or np.iscomplexobj(s21)
    # a passive interconnect line: |S21| <= 1 (no gain)
    assert np.all(np.abs(s21) <= 1.0 + 1e-6)


def _ones_frame(n_freqs):
    # [az, el, chirp, n_freqs] all-ones so `frame * H` returns H per element directly
    return torch.ones(2, 2, 1, n_freqs, dtype=torch.complex64, device=device)


def test_transfer_mode_applies_resampled_s21_over_band():
    n_freqs = 64
    blk = InterconnectBlock(transfer_csv=TESSERA_INTERCONNECT_CSV, band_hz=PIPELINE_BAND)
    out = blk.apply_interconnect(_ones_frame(n_freqs))
    assert out.shape == (2, 2, 1, n_freqs)
    H = out[0, 0, 0, :]
    # every element multiplied by the same 1-D response
    assert torch.allclose(out[1, 1, 0, :], H)
    Hdb = 20 * torch.log10(torch.abs(H) + 1e-12)
    # band edges land exactly on the CSV's 0.25 GHz grid (28.5 / 31.5 GHz), so the
    # interpolated response matches the datasheet values there.
    assert Hdb[0].item() == pytest.approx(-7.045, abs=0.05)   # 28.5 GHz
    assert Hdb[-1].item() == pytest.approx(-7.845, abs=0.05)  # 31.5 GHz
    # insertion loss rises monotonically across this band
    assert Hdb[-1].item() < Hdb[0].item()
    # passive over the band
    assert torch.all(torch.abs(H) <= 1.0 + 1e-6)


def test_transfer_mode_band_none_spans_full_csv():
    freq, s21 = load_interconnect_transfer(TESSERA_INTERCONNECT_CSV)
    n_freqs = 32
    blk = InterconnectBlock(transfer_csv=TESSERA_INTERCONNECT_CSV, band_hz=None)
    H = blk.apply_interconnect(_ones_frame(n_freqs))[0, 0, 0, :]
    # band_hz=None maps the CSV's own 1..40 GHz span across the frame samples, so the
    # endpoints equal the CSV endpoints.
    assert H[0].item() == pytest.approx(complex(s21[0]), abs=1e-3)
    assert H[-1].item() == pytest.approx(complex(s21[-1]), abs=1e-3)


def test_transfer_mode_differs_from_boxcar_default():
    n_freqs = 64
    frame = _ones_frame(n_freqs)
    boxcar = InterconnectBlock().apply_interconnect(frame)
    tessera = InterconnectBlock(
        transfer_csv=TESSERA_INTERCONNECT_CSV, band_hz=PIPELINE_BAND
    ).apply_interconnect(frame)
    assert not torch.allclose(boxcar, tessera)


def test_case3_identity_ignores_transfer_csv():
    n_freqs = 16
    frame = _ones_frame(n_freqs)
    blk = InterconnectBlock(case='case3', transfer_csv=TESSERA_INTERCONNECT_CSV,
                            band_hz=PIPELINE_BAND)
    assert torch.equal(blk.apply_interconnect(frame), frame)


def test_missing_transfer_csv_raises():
    with pytest.raises((OSError, IOError, ValueError)):
        InterconnectBlock(transfer_csv="does_not_exist_interconnect.csv")


def test_tutorial_smoke_runs_without_disk(tmp_path, monkeypatch):
    """The main_interconnect tutorial runs (show=False writes nothing) and reports a
    sane pipeline-band insertion loss."""
    import matplotlib
    matplotlib.use("Agg")
    import e2e.main.main_interconnect as mi
    monkeypatch.setattr(mi, "FIG_DIR", str(tmp_path))
    res = mi.main(show=False, n_freqs=64)
    lo, hi = res["band_loss_db"]
    assert -12.0 < lo <= hi < 0.0            # physical insertion loss over the band
    assert list(tmp_path.iterdir()) == []    # show=False -> no figure written
