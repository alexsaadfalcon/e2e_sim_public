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


# --------------------------------------------------------------------------------
# range_profile_comparison: interconnect models vs. radar range profile.
# --------------------------------------------------------------------------------


def test_case3_csv_loads_and_is_physical():
    import e2e.main.main_interconnect as mi

    freq, s21 = load_interconnect_transfer(mi.CASE3_INTERCONNECT_CSV)
    assert np.all(np.diff(freq) > 0)
    assert freq[0] == pytest.approx(70e9)
    assert np.all(np.abs(s21) <= 1.0 + 1e-6)   # passive


def test_range_profile_comparison_insertion_loss_and_ripple():
    """Re-derives the two headline numbers the owner already measured directly from
    the shipped CSVs (mean/peak-to-peak |S21| in dB over each pipeline band)."""
    import e2e.main.main_interconnect as mi

    res = mi.range_profile_comparison(show=False, n_freqs=64)
    il = res["insertion_loss_db"]
    ripple = res["ripple_db"]
    assert il["tessera_tsv"] == pytest.approx(-7.46, abs=0.05)
    assert ripple["tessera_tsv"] == pytest.approx(0.80, abs=0.05)
    assert il["tessera_case3"] == pytest.approx(-0.53, abs=0.05)
    assert ripple["tessera_case3"] == pytest.approx(0.03, abs=0.02)
    # Case3 is the most demanding of the six IN OUR BAND, not necessarily better than TSV --
    # this module must never rank them; just sanity-check both are passive (loss <= 0).
    assert il["tessera_tsv"] < 0.0 and il["tessera_case3"] < 0.0


def test_range_profile_comparison_mainlobe_metrics():
    """The physically-modelled interconnects (ideal/TSV/Case3) all resolve to the
    finest possible native-resolution mainlobe (1 bin); only the legacy placeholder
    smears it -- the figure's central, owner-specified finding."""
    import e2e.main.main_interconnect as mi

    res = mi.range_profile_comparison(show=False, n_freqs=512)
    m = res["metrics"]
    assert m["ideal"]["width_3db_bins"] == 1
    assert m["tessera_tsv"]["width_3db_bins"] == 1
    assert m["tessera_case3"]["width_3db_bins"] == 1
    assert m["legacy_boxcar"]["width_3db_bins"] == 11
    # Ideal has (numerically) no sidelobe at all at native resolution; the two
    # physically-modelled interconnects show a small but real sub-mainlobe skirt from
    # their in-band ripple, well below the mainlobe.
    assert m["ideal"]["peak_sidelobe_db"] < -100.0
    assert -60.0 < m["tessera_tsv"]["peak_sidelobe_db"] < -10.0
    assert -90.0 < m["tessera_case3"]["peak_sidelobe_db"] < -40.0


def test_range_profile_comparison_smoke_runs_without_disk(tmp_path, monkeypatch):
    """`show=False` computes everything but touches no disk (same contract as
    `main`); `show=True` writes exactly one figure to the (monkeypatched) path."""
    import matplotlib
    matplotlib.use("Agg")
    import e2e.main.main_interconnect as mi

    monkeypatch.setattr(mi, "RANGE_PROFILE_FIG_PATH", str(tmp_path / "sub" / "out.png"))
    res = mi.range_profile_comparison(show=False, n_freqs=64)
    assert "metrics" in res
    assert not (tmp_path / "sub").exists()   # show=False -> no directory/file created

    res2 = mi.range_profile_comparison(show=True, n_freqs=64)
    assert (tmp_path / "sub" / "out.png").is_file()
    assert res2["metrics"]["legacy_boxcar"]["width_3db_bins"] >= res2["metrics"]["ideal"]["width_3db_bins"]


# --------------------------------------------------------------------------------
# before_after_comparison: the README gallery's compact 4-arm figure.
# --------------------------------------------------------------------------------


def test_before_after_comparison_smoke_runs_without_disk(tmp_path, monkeypatch):
    """Same no-disk-on-`show=False` / writes-one-file-on-`show=True` contract as
    `range_profile_comparison`, and the same 4 arms (same metrics)."""
    import matplotlib
    matplotlib.use("Agg")
    import e2e.main.main_interconnect as mi

    monkeypatch.setattr(mi, "BEFORE_AFTER_FIG_PATH", str(tmp_path / "sub" / "out.png"))
    res = mi.before_after_comparison(show=False, n_freqs=64)
    assert not (tmp_path / "sub").exists()

    res2 = mi.before_after_comparison(show=True, n_freqs=64)
    assert (tmp_path / "sub" / "out.png").is_file()
    m = res2["metrics"]
    # ALL SIX 77 GHz cases, not just Case3: the whole point of this figure is how the
    # measured designs differ from EACH OTHER, and the derived CSVs for the other five
    # landed 2026-08-19.
    assert set(m) == ({"ideal", "tessera_tsv", "legacy_boxcar"}
                      | {f"tessera_case{n}" for n in mi.CASE_NUMBERS})
    assert m["legacy_boxcar"]["width_3db_bins"] > m["ideal"]["width_3db_bins"]

    # Case3 is the WORST of the six, which the project had been asserting on the
    # collaborator's word. With all six present it is checkable, so check it: the most demanding
    # in-band ripple shows up as the highest (least negative) sidelobe floor.
    sidelobes = {n: m[f"tessera_case{n}"]["peak_sidelobe_db"] for n in mi.CASE_NUMBERS}
    assert max(sidelobes, key=sidelobes.get) == 3, (
        f"Case3 should have the highest sidelobe floor of the six in our band: {sidelobes}")


def test_arm_styling_covers_every_arm_with_distinct_colors():
    """Every arm has a color and linestyle, and no two arms share a color -- the
    owner's distinguishability complaint, checked structurally."""
    import e2e.main.main_interconnect as mi

    arms = mi._interconnect_arms()
    assert set(mi.ARM_COLORS) == set(arms)
    assert set(mi.ARM_STYLES) == set(arms)
    assert len(set(mi.ARM_COLORS.values())) == len(mi.ARM_COLORS)
