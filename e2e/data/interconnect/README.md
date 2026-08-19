# Interconnect transfer-function data

Seven derived datasets ship here: one Ka-band TSV line, and six 77 GHz automotive designs.
All are DATA. No collaborator source code, model, or HFSS project is reproduced anywhere in
this repository.

## `tessera_case{1..6}_s21_77ghz.csv` — the six 77 GHz automotive designs

Six measured interconnect designs from the Tessera collaborator's HFSS export, cleared by
the owner for this repository on 2026-08-19. Each file is 401 rows spanning **70–90 GHz**
with columns `freq_hz, s21_re, s21_im, s21_abs_db, s21_phase_deg`, and each carries its own
provenance in `#` header comments.

| file | in-band loss, 75–81 GHz |
| ---- | ----------------------- |
| `tessera_case1_s21_77ghz.csv` | −0.292 … −0.289 dB |
| `tessera_case2_s21_77ghz.csv` | −0.288 … −0.279 dB |
| **`tessera_case3_s21_77ghz.csv`** | **−0.547 … −0.513 dB — the worst of the six** |
| `tessera_case4_s21_77ghz.csv` | −0.260 … −0.259 dB |
| `tessera_case5_s21_77ghz.csv` | −0.249 … −0.248 dB |
| `tessera_case6_s21_77ghz.csv` | −0.249 … −0.248 dB |

**Case3 is the one the README's headline figure plots**, deliberately: it is a conservative
bound, not a typical part. `python -m e2e.main.main_interconnect` draws all six against each
other in its own figure, which is where the designs' differences from EACH OTHER are visible.

**Phase is reconstructed, not measured.** The source export carries magnitude only, so phase
is a **minimum-phase** reconstruction from |S21| (Hilbert transform of ln|H|). That is the
physically correct choice for a passive, causal, minimum-phase structure, but it is a
reconstruction: do not read the phase column as measured data.

Load one with `InterconnectBlock(transfer_csv=..., band_hz=...)`, or by number via
`e2e.main.main_interconnect.case_csv(n)`.

## `tessera_tsv_s21.csv` — the Ka-band TSV line

A frequency-swept scattering-parameter transfer function for a
**single Through-Silicon-Via (TSV) interconnect line** (one center signal via surrounded by
a ground ring). `InterconnectBlock(transfer_csv=...)` in `e2e/blocks.py` loads it and
resamples it onto a frame's frequency grid, as a physically-grounded alternative to the
default placeholder boxcar response.

### TSV columns

| column | meaning |
| ------ | ------- |
| `freq_hz` | frequency (Hz), 1–40 GHz in 0.25 GHz steps |
| `s21_re`, `s21_im` | complex through / insertion transfer coefficient S21 |
| `s21_abs_db`, `s21_phase_deg` | S21 magnitude (dB) and phase (deg), for convenience |
| `s11_re`, `s11_im`, `s11_abs_db` | complex reflection S11 (magnitude in dB) |

`InterconnectBlock` uses `s21_re + 1j*s21_im` as the interconnect's frequency response.

### TSV geometry

Single signal line, ground ring; default TSV geometry: radius 5 µm, pitch 60 µm,
height 100 µm, liner 0.5 µm. Over the pipeline's 28.5–31.5 GHz band the insertion loss is
≈ −7.0 to −7.9 dB (rising with frequency); there is a reflection resonance near ~8 GHz.

### TSV provenance and caveats

- These numbers were produced by a collaborator's **physics-informed GNN surrogate for TSV
  networks** (an HFSS-finetuned model that predicts the S-matrix from array layout +
  geometry). **The model/source code is intentionally NOT included in this repository** —
  only this derived `.csv` result is committed, for use in tests and tutorials.
- **Frequency-validity caveat:** the surrogate's documented demo point is ~15 GHz. It
  produces smooth, well-behaved output across the full 1–40 GHz sweep (no extrapolation
  artifacts), but whether it is *validated* at the pipeline's 28.5–31.5 GHz band is a
  question for the model's authors. Treat the >~20 GHz region as indicative pending
  confirmation.
- **TODO (attribution):** confirm the citation / acknowledgement wording with the
  collaborator before any public release (the associated paper is under review).

### Regenerating the TSV

Not reproducible from this repository alone (the model is external). The collaborator's
model, run over a 1–40 GHz sweep of the geometry above, regenerates this file.
