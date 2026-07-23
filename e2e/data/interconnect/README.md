# Interconnect transfer-function data

`tessera_tsv_s21.csv` — a frequency-swept scattering-parameter transfer function for a
**single Through-Silicon-Via (TSV) interconnect line** (one center signal via surrounded by
a ground ring). `InterconnectBlock(transfer_csv=...)` in `e2e/blocks.py` loads it and
resamples it onto a frame's frequency grid, as a physically-grounded alternative to the
default placeholder boxcar response.

## Columns

| column | meaning |
| ------ | ------- |
| `freq_hz` | frequency (Hz), 1–40 GHz in 0.25 GHz steps |
| `s21_re`, `s21_im` | complex through / insertion transfer coefficient S21 |
| `s21_abs_db`, `s21_phase_deg` | S21 magnitude (dB) and phase (deg), for convenience |
| `s11_re`, `s11_im`, `s11_abs_db` | complex reflection S11 (magnitude in dB) |

`InterconnectBlock` uses `s21_re + 1j*s21_im` as the interconnect's frequency response.

## Geometry

Single signal line, ground ring; default TSV geometry: radius 5 µm, pitch 60 µm,
height 100 µm, liner 0.5 µm. Over the pipeline's 28.5–31.5 GHz band the insertion loss is
≈ −7.0 to −7.9 dB (rising with frequency); there is a reflection resonance near ~8 GHz.

## Provenance and caveats

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

## Regenerating

Not reproducible from this repository alone (the model is external). The collaborator's
model, run over a 1–40 GHz sweep of the geometry above, regenerates this file.
