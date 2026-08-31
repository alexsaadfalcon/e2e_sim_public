"""Derive a public interconnect transfer-function CSV from a raw collaborator HFSS export.

WHY THIS EXISTS. `e2e/data/interconnect/tessera_case3_s21_77ghz.csv` shipped with the note
"generated 2026-08-10 by e2e maintainers" and no script -- the same lost-scratch-script
failure that cost us the generators for tier_ladder, scene_vs_ra, signal_journey and
fleet_side. When five more cases needed the same treatment there was nothing to run. This
is that script, written down.

WHAT IT DOES. The raw export carries MAGNITUDE ONLY, on a coarse grid (31 points over
0.1-100 GHz), with per-case port names (`Port_Bot_6_6` for Case3, `Port_Bot_11_11` for
Case1, ...). The public file needs complex S21 on the fine in-band grid the pipeline
interpolates against. So:

  1. find the THROUGH column generically -- `dB(S(Port_Top_*, Port_Bot_*))` -- rather than
     by a hard-coded name, because the port suffix differs per case;
  2. interpolate |S21| in dB onto a 70-90 GHz / 50 MHz grid (401 points), matching the
     existing Case3 file exactly so the two are directly comparable;
  3. reconstruct PHASE as minimum phase, `arg H = -Hilbert(ln|H|)`, because the export has
     none. This is an assumption, not a measurement, and the output header says so.

WHAT IT DELIBERATELY DOES NOT DO. It does not copy, import, or paraphrase any collaborator
SOURCE CODE. It reads a data export and writes data, taking the path to a raw export as
an argument.

WHAT YOU CAN AND CANNOT REPRODUCE WITH IT. The raw HFSS exports belong to the authors
(Mohamed Gharib and Prof. Inna Partin-Vaisband, University of Illinois Chicago) and are
NOT redistributed here, so running this end to end needs an export obtained from them.
What shipping this script does buy, and the reason it is public: the derivation becomes
auditable rather than asserted. The interpolation grid, the minimum-phase reconstruction
and the provenance header are all readable, so the shipped .csv files can be checked
against their stated method instead of taken on trust.

VALIDATION. `--check` re-derives Case3 and compares against the shipped
`tessera_case3_s21_77ghz.csv`. That is the oracle: if the method is right, it reproduces a
file made by a different person on a different day. Run it before trusting a new case.

    python notes/tools/derive_interconnect_csv.py --check
    python notes/tools/derive_interconnect_csv.py \
        --raw <private>/interconnect/data/Case1.csv --case Case1 \
        --out e2e/data/interconnect/tessera_case1_s21_77ghz.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import re
import sys
from pathlib import Path

import numpy as np

# This script sits IN the data directory it writes to, so the shipped files are its
# siblings. (It previously lived under the maintainers' private notes/tools/ and
# resolved the repo root two levels up; moving it made that path wrong.)
DATA_DIR = Path(__file__).resolve().parent
REPO_ROOT = DATA_DIR.parents[2]

#: The public grid. Matches the shipped Case3 file exactly (401 points, 70-90 GHz,
#: 50 MHz) so every case is directly comparable and the pipeline's interpolation sees
#: the same support for all of them.
F_START_HZ, F_STOP_HZ, N_POINTS = 70.0e9, 90.0e9, 401

#: The automotive band the numbers are quoted over. Narrower than the file's span on
#: purpose: the file carries margin so interpolation never clamps at the band edge.
BAND_HZ = (75.0e9, 81.0e9)

#: A THROUGH path is Top -> Bot on the SAME port, e.g. dB(S(Port_Top_6_6, Port_Bot_6_6)).
#: The suffix must match on both sides. Some exports (Case2, Case6) carry TWO independent
#: signal lines, which produces four Top/Bot columns: two through paths and two CROSS-
#: COUPLING terms (Top_13_13 <- Bot_6_6). A looser pattern matches all four and would
#: happily present a coupling term as an insertion loss.
_THROUGH_RE = re.compile(
    r"dB\(S\(\s*Port_Top_(?P<a>[0-9_]+?)\s*,\s*Port_Bot_(?P<b>[0-9_]+?)\s*\)\)", re.I)


def find_through_columns(header):
    """Indices of the through (transmission) columns: Top -> Bot on the SAME port.

    Returns a list, because an export may carry several independent lines. Raises if it
    finds none -- a quiet wrong answer here would be a reflection coefficient or a
    cross-coupling term presented as an insertion loss.
    """
    hits = []
    for i, name in enumerate(header):
        m = _THROUGH_RE.search(name or "")
        if m and m.group("a") == m.group("b"):
            hits.append(i)
    if not hits:
        raise SystemExit(
            "found no dB(S(Port_Top_X, Port_Bot_X)) through column; columns were:\n"
            + "\n".join(f"  [{i}] {h}" for i, h in enumerate(header)))
    return hits


def read_raw(path):
    """(freq_hz, s21_db) from a raw HFSS export, sorted by ascending frequency."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = [r for r in csv.reader(fh) if r and any(c.strip() for c in r)]
    header, body = rows[0], rows[1:]
    if "ghz" not in (header[0] or "").lower():
        raise SystemExit(f"first column is not a GHz frequency axis: {header[0]!r}")
    cols = find_through_columns(header)
    freq_ghz = np.array([float(r[0]) for r in body], dtype=float)
    order = np.argsort(freq_ghz)
    freq_hz = freq_ghz[order] * 1e9

    # When an export carries several independent lines, take the WORST in-band one and say
    # which. That matches the convention the project already states for Case3 ("the worst
    # of six measured designs, read it as a conservative bound") -- picking the best line
    # of a multi-line part and quoting it as the part would quietly invert that.
    in_band = (freq_hz >= BAND_HZ[0]) & (freq_hz <= BAND_HZ[1])
    if not in_band.any():          # coarse export may straddle the band with no sample in it
        in_band = np.ones_like(freq_hz, dtype=bool)
    per_col = []
    for c in cols:
        db = np.array([float(r[c]) for r in body], dtype=float)[order]
        per_col.append((float(np.median(db[in_band])), c, db))
    per_col.sort(key=lambda t: t[0])          # most negative (worst) first
    _, col, s21_db = per_col[0]
    others = [header[c].strip() for _, c, _ in per_col[1:]]
    return freq_hz, s21_db, header[col], others


def minimum_phase(mag):
    """Minimum-phase argument for a magnitude response: arg H = -Hilbert(ln|mag|).

    Written out rather than pulled from scipy.signal so the one assumption this file makes
    about the collaborator's data is visible in the repo that ships it.
    """
    log_mag = np.log(np.clip(mag, 1e-30, None))
    n = log_mag.size
    spec = np.fft.fft(log_mag)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(n + 1) // 2] = 2.0
    return -np.imag(np.fft.ifft(spec * h))


def derive(raw_path, case):
    freq_raw, db_raw, col_name, others = read_raw(raw_path)
    freq = np.linspace(F_START_HZ, F_STOP_HZ, N_POINTS)
    # Interpolate in dB, which is how the export is quoted and how a passive loss behaves
    # across a narrow band; endpoints clamp rather than extrapolate.
    db = np.interp(freq, freq_raw, db_raw)
    mag = 10.0 ** (db / 20.0)
    phase = minimum_phase(mag)
    s21 = mag * np.exp(1j * phase)
    in_band = (freq >= BAND_HZ[0]) & (freq <= BAND_HZ[1])
    return {
        "freq": freq, "s21": s21, "db": db, "phase": phase,
        "col_name": col_name, "others": others, "n_raw": freq_raw.size,
        "raw_span_ghz": (freq_raw.min() / 1e9, freq_raw.max() / 1e9),
        "band_db": (db[in_band].min(), db[in_band].max()),
        "case": case,
    }


def write_csv(d, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lo, hi = d["band_db"]
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Interconnect transfer function for the 77 GHz automotive band.\n")
        fh.write(f"# PROVENANCE: magnitude resampled from the Tessera collaborator's HFSS\n")
        fh.write(f"#   S-parameter export {d['case']!r} ({d['col_name'].strip()},\n")
        fh.write(f"#   {d['raw_span_ghz'][0]:g}-{d['raw_span_ghz'][1]:g} GHz, {d['n_raw']} pts).\n")
        fh.write("#   Phase is a MINIMUM-PHASE reconstruction from |S21| (Hilbert transform of\n")
        fh.write("#   ln|H|); the source export carries magnitude only. No collaborator source\n")
        fh.write("#   code is reproduced here or anywhere in this repository -- this file is data.\n")
        if d["others"]:
            fh.write("#   This export carries several independent signal lines; the WORST in-band\n")
            fh.write("#   one was taken (a conservative bound, matching this project's Case3\n")
            fh.write("#   convention). Not taken: " + "; ".join(d["others"]) + "\n")
        fh.write(f"# In-band {BAND_HZ[0]/1e9:g}-{BAND_HZ[1]/1e9:g} GHz: {lo:.3f} to {hi:.3f} dB.\n")
        fh.write(f"# generated {_dt.date.today().isoformat()} by notes/tools/derive_interconnect_csv.py\n")
        fh.write("freq_hz,s21_re,s21_im,s21_abs_db,s21_phase_deg\n")
        for f, s, db in zip(d["freq"], d["s21"], d["db"]):
            fh.write(f"{f:.6e},{s.real:.9f},{s.imag:.9f},{db:.6f},"
                     f"{np.degrees(np.angle(s)):.6f}\n")
    return out_path


def check(raw_case3):
    """Oracle: re-derive Case3 and compare against the file shipped on 2026-08-10."""
    shipped = DATA_DIR / "tessera_case3_s21_77ghz.csv"
    ref_f, ref_db = [], []
    for line in open(shipped, encoding="utf-8"):
        if line.startswith("#") or line.startswith("freq_hz"):
            continue
        p = line.split(",")
        ref_f.append(float(p[0])); ref_db.append(float(p[3]))
    ref_f, ref_db = np.array(ref_f), np.array(ref_db)
    d = derive(raw_case3, "Case3")
    ok = ref_f.size == d["freq"].size and np.allclose(ref_f, d["freq"])
    print("=" * 68)
    print("RESULT -- derivation checked against the shipped Case3 file")
    print("=" * 68)
    print(f"  grid matches            : {ok} ({ref_f.size} vs {d['freq'].size} points)")
    if ok:
        err = np.abs(ref_db - d["db"])
        print(f"  |S21| max abs deviation : {err.max():.2e} dB")
        print(f"  |S21| RMS deviation     : {np.sqrt((err**2).mean()):.2e} dB")
        print("  -> the method reproduces a file made independently on 2026-08-10."
              if err.max() < 1e-3 else
              "  -> DEVIATION: do not trust new cases until this is understood.")
    return 0 if ok else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--raw", help="path to the raw collaborator export (private repo)")
    p.add_argument("--case", help="case label for the provenance header, e.g. Case1")
    p.add_argument("--out", help="output CSV under e2e/data/interconnect/")
    p.add_argument("--force", action="store_true",
                   help="allow overwriting the Case3 oracle file (see main())")
    p.add_argument("--check", metavar="RAW_CASE3", nargs="?", const=True,
                   help="re-derive Case3 and diff against the shipped file")
    a = p.parse_args(argv)

    if a.check:
        if not isinstance(a.check, str):
            p.error("--check needs the path to a raw Case3 export: the raw HFSS "
                    "exports are the collaborators' and are not redistributed with "
                    "this repository, so no default path would work here.")
        return check(a.check)

    if not (a.raw and a.case and a.out):
        p.error("--raw, --case and --out are all required (or use --check)")
    # Case3's shipped file is the ORACLE this tool checks itself against. Overwriting it
    # with our own output makes `--check` compare the tool to itself -- it would pass
    # forever, including when the method is wrong. Refuse unless someone means it.
    if Path(a.out).name == "tessera_case3_s21_77ghz.csv" and not a.force:
        raise SystemExit(
            "refusing to overwrite tessera_case3_s21_77ghz.csv: it is the independent "
            "reference --check validates against. Pass --force if you really mean to "
            "replace the oracle, and expect --check to become vacuous afterwards.")
    d = derive(a.raw, a.case)
    out = write_csv(d, a.out)
    lo, hi = d["band_db"]
    print(f"wrote {out}  ({d['case']}: in-band {lo:.3f} to {hi:.3f} dB, "
          f"ripple {hi - lo:.3f} dB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
