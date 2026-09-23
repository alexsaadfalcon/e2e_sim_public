"""Regenerate the PUBLIC Tessera TSV S21(f) data from the public checkpoint.

Context (notes/ESTABLISHED_FACTS.md F89/F90, measured 2026-09-23): the legacy
``tessera_tsv_s21.csv`` is an exactly lossless closed-form two-port that no checkpoint
of this architecture produced; it is not reproducible from the public release. This
script instead evaluates the ACTUAL public checkpoint
(``e2e.interconnect_surrogate.tessera.TesseraTSV``, upstream's own canonical demo
geometry ``SHIPPED_TSV_DESIGN``) and writes two files that ARE reproducible from a
clean clone plus ``python -m e2e.interconnect_surrogate.fetch``:

* ``tessera_tsv_s21_public.csv`` -- the DIRECT evaluation: the shipped geometry, no
  scale model (``scale=1.0``), over a 1-90 GHz sweep. This is what
  ``InterconnectBlock(transfer_csv=...)`` (the CSV/interpolation path) reads.
* ``tessera_tsv_s21_public_ka_scaled.csv`` -- the SCALE-MODEL evaluation at the
  pipeline's Ka band (28.5-31.5 GHz): what ``InterconnectBlock(source='tessera')``
  actually applies there BY DEFAULT (``scale=None`` auto-resolves to x2 for this band,
  see ``e2e.blocks._resolve_tessera_scale`` / F91) -- the model is evaluated at 2x the
  shipped geometry, half frequency, and the result is reported at the real Ka
  frequencies. This is measurably different from the direct file's Ka-band slice (see
  the header of each file); both are kept because the two `InterconnectBlock` modes
  (``transfer_csv=`` vs. ``source='tessera'``) do not evaluate the same thing over this
  band, and only one of the two files is reproducing the block's own default.

Both files carry a header (see ``_write_csv``) that records the geometry, arrangement,
checkpoint commit + sha256, the wrapper's ``TesseraTSV.OUTPUT_VERSION``, the scale
factor used, the generation date, and the command line -- so nothing needs to be
inferred later; re-run this to check any of those with `--check`.

Run:  python -m e2e.data.interconnect.regenerate_tessera_tsv
"""
import argparse
import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from e2e.blocks import (
    _default_presented_tessera_params,
    _presented_to_model_tessera_params,
    _resolve_tessera_scale,
    tessera_s21_for_axis,
)
from e2e.interconnect_surrogate import SHIPPED_TSV_DESIGN, TesseraTSV, checkpoint_dir

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

#: 1-90 GHz in 0.25 GHz steps -- covers the whole training envelope's frequency span
#: (F90: 1.7-97.7 GHz) plus the pipeline's own Ka-band, at the same grid density as the
#: legacy CSV so the two are visually comparable.
FULL_SWEEP_HZ = np.arange(1e9, 90e9 + 1e-6, 0.25e9)
#: The pipeline's default FrequencyPlan band (e2e/main/main_interconnect.py BAND).
KA_BAND_HZ = np.linspace(28.5e9, 31.5e9, 301)
ARRANGEMENT = "ring3x3"

DIRECT_CSV = HERE / "tessera_tsv_s21_public.csv"
KA_SCALED_CSV = HERE / "tessera_tsv_s21_public_ka_scaled.csv"

_COLUMNS = "freq_hz,s21_re,s21_im,s21_abs_db,s21_phase_deg,s11_re,s11_im,s11_abs_db\n"


def _checkpoint_provenance():
    """(commit, sha256) of the resolved public checkpoint; raises if none is found."""
    models_dir = checkpoint_dir()
    if models_dir is None:
        raise RuntimeError(
            "no Tessera checkpoint found -- run `python -m "
            "e2e.interconnect_surrogate.fetch` first"
        )
    weights = models_dir / "best_model.pth"
    sha256 = hashlib.sha256(weights.read_bytes()).hexdigest()
    try:
        out = subprocess.run(
            ["git", "-C", str(models_dir.parent), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        commit = out.stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        commit = f"unknown (no git checkout at {models_dir.parent})"
    return commit, sha256


def _wrapper_commit():
    """This repo's own HEAD, for 'wrapper version' -- best-effort, never raises."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _write_csv(path, freq_hz, s21, s11, header_lines):
    s21 = np.asarray(s21)
    s11 = np.asarray(s11)
    s21_db = 20 * np.log10(np.abs(s21) + 1e-15)
    s21_phase = np.degrees(np.angle(s21))
    s11_db = 20 * np.log10(np.abs(s11) + 1e-15)
    with open(path, "w", newline="\n") as fh:
        for line in header_lines:
            fh.write(f"# {line}\n")
        fh.write(_COLUMNS)
        for f, s21v, s11v, db, ph, s11db in zip(freq_hz, s21, s11, s21_db, s21_phase, s11_db):
            fh.write(f"{f:.6e},{s21v.real:.8e},{s21v.imag:.8e},{db:.6f},{ph:.6f},"
                      f"{s11v.real:.8e},{s11v.imag:.8e},{s11db:.6f}\n")


def regenerate(direct_path=DIRECT_CSV, ka_scaled_path=KA_SCALED_CSV, cmdline=None):
    """Write both public CSVs; returns (direct_path, ka_scaled_path)."""
    commit, sha256 = _checkpoint_provenance()
    wrapper_commit = _wrapper_commit()
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cmdline = cmdline or "python -m e2e.data.interconnect.regenerate_tessera_tsv"
    tsv = TesseraTSV(passivity="raise", warn_out_of_range=False)

    # -- direct evaluation: scale=1.0, canonical geometry, 1-90 GHz --------------
    s_full = tsv.s_matrix(FULL_SWEEP_HZ, grid=ARRANGEMENT, **SHIPPED_TSV_DESIGN)
    geom = SHIPPED_TSV_DESIGN
    header_direct = [
        "Tessera TSV -- PUBLIC checkpoint, DIRECT evaluation (not the scale model).",
        f"geometry: radius {geom['radius_um']:g} um, pitch {geom['pitch_um']:g} um, "
        f"height {geom['height_um']:g} um, liner {geom['liner_um']:g} um, "
        f"{geom['temperature_k']:g} K -- SHIPPED_TSV_DESIGN, upstream's own canonical "
        "demo geometry (README quickstart / examples/predict_smatrix.py / "
        "config.yaml optimization.fixed_params; see F90 in notes/ESTABLISHED_FACTS.md)",
        f"arrangement: {ARRANGEMENT} (one center signal via, ground ring)",
        f"checkpoint: models/best_model.pth, commit {commit}, sha256 {sha256}",
        f"wrapper: e2e_sim_public commit {wrapper_commit}, "
        f"TesseraTSV.OUTPUT_VERSION={TesseraTSV.OUTPUT_VERSION}",
        "scale: 1.0 -- this is the DIRECT evaluation at the geometry above, NOT the "
        "scale model (see tessera_tsv_s21_public_ka_scaled.csv for that)",
        f"generated: {date}",
        f"command: {cmdline}",
        "phase is the surrogate's OWN phase (not a minimum-phase reconstruction).",
        "columns: " + _COLUMNS.strip(),
    ]
    _write_csv(direct_path, FULL_SWEEP_HZ, s_full[:, 0, 1], s_full[:, 0, 0], header_direct)

    # -- Ka-band scale-model evaluation: InterconnectBlock(source='tessera')'s own
    # default over this band (scale=None auto-resolves; see _resolve_tessera_scale).
    scale = _resolve_tessera_scale(None, KA_BAND_HZ)
    presented = _default_presented_tessera_params(scale)
    model_geom = _presented_to_model_tessera_params(presented, scale)
    s_ka = tsv.s_matrix(KA_BAND_HZ / scale, grid=ARRANGEMENT, **model_geom)
    s21_ka, s11_ka = s_ka[:, 0, 1], s_ka[:, 0, 0]
    # This must equal the block's own public entry point exactly -- it is the same
    # computation, just spelled out here for the header/provenance.
    s21_via_block = tessera_s21_for_axis(KA_BAND_HZ, arrangement=ARRANGEMENT, scale=None)
    if not np.allclose(s21_ka, s21_via_block, atol=0, rtol=1e-9):
        raise RuntimeError(
            "internal check failed: the scale-model evaluation here does not match "
            "e2e.blocks.tessera_s21_for_axis -- the two have drifted apart"
        )
    header_ka = [
        "Tessera TSV -- PUBLIC checkpoint, SCALE-MODEL evaluation at the pipeline's "
        "Ka band. This IS InterconnectBlock(source='tessera')'s DEFAULT over "
        "28.5-31.5 GHz (band_hz with scale=None), NOT a direct evaluation of the "
        "shipped geometry at these frequencies -- compare against the in-band slice "
        "of tessera_tsv_s21_public.csv, which differs measurably (see each file's "
        "in-band mean/ripple in e2e/data/interconnect/README.md).",
        f"presented geometry: radius {presented['radius_um']:g} um, pitch "
        f"{presented['pitch_um']:g} um, height {presented['height_um']:g} um, liner "
        f"{presented['liner_um']:g} um, {presented['temperature_k']:g} K",
        f"model geometry actually evaluated: radius {model_geom['radius_um']:g} um, "
        f"pitch {model_geom['pitch_um']:g} um, height {model_geom['height_um']:g} um, "
        f"liner {model_geom['liner_um']:g} um, {model_geom['temperature_k']:g} K -- "
        f"scale x{scale:g} applied to all four lengths together (F91), frequency "
        f"evaluated at /{scale:g} and reported at the real Ka frequencies",
        f"arrangement: {ARRANGEMENT}",
        f"checkpoint: models/best_model.pth, commit {commit}, sha256 {sha256}",
        f"wrapper: e2e_sim_public commit {wrapper_commit}, "
        f"TesseraTSV.OUTPUT_VERSION={TesseraTSV.OUTPUT_VERSION}",
        f"scale: {scale:g} -- auto-resolved by e2e.blocks._resolve_tessera_scale for "
        "this band (see F91, notes/ESTABLISHED_FACTS.md); NOT the direct evaluation",
        f"generated: {date}",
        f"command: {cmdline}",
        "phase is the surrogate's OWN phase (not a minimum-phase reconstruction).",
        "columns: " + _COLUMNS.strip(),
    ]
    _write_csv(ka_scaled_path, KA_BAND_HZ, s21_ka, s11_ka, header_ka)

    return direct_path, ka_scaled_path


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m e2e.data.interconnect.regenerate_tessera_tsv",
        description=__doc__.splitlines()[0],
    )
    p.parse_args(argv)
    d, k = regenerate(cmdline="python -m e2e.data.interconnect.regenerate_tessera_tsv")
    print(f"wrote {d}")
    print(f"wrote {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
