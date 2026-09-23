# Interconnect transfer-function data

Nine derived datasets ship here: three Ka-band TSV files (one legacy, two regenerated from
the public Tessera checkpoint), and six 77 GHz automotive designs. All are DATA. No
collaborator source code, model, or HFSS project is reproduced anywhere in this repository.

## Attribution

The interconnects were **simulated by Mohamed Gharib, Leonid Popryho, and Prof. Inna
Partin-Vaisband (University of Illinois Chicago)**; see Gharib, Popryho & Partin-Vaisband,
"From Physics to Surrogate Intelligence: A Unified Electro-Thermo-Optimization Framework
for TSV Networks," IEEE TCAD 2026, doi 10.1109/TCAD.2026.3718807. These `.csv` files are
the output of their simulations — they are *simulated interconnects*, not laboratory
measurements, and nothing here should be described as measured data.

**The TSV surrogate that produced `tessera_tsv_s21.csv` is public**: the code is BSD
3-Clause at github.com/HiPerCAS/tessera, citing the paper above. **The six 77 GHz
automotive HFSS projects behind `tessera_case{1..6}_s21_77ghz.csv` are not in that
repository** and are available on request to the authors; that package computes
through-silicon-via arrays, not automotive interconnects, and cannot reproduce them.

### How the shipped `.csv` files were derived

Each file here was produced by the e2e maintainers from a raw HFSS export supplied by the
authors. The export carries **magnitude only**, on a coarse grid; the shipped file is that
magnitude interpolated onto a uniform 70–90 GHz / 50 MHz grid (401 points), with **phase
reconstructed as minimum phase** (`arg H = −Hilbert(ln|H|)`). The phase is therefore an
assumption, not data — see the caveat below.

The derivation script itself is **not distributed with this repository**, and neither are
the raw exports. Both are available from the maintainers on request. The method has been
validated against an independently produced file: re-deriving Case3 from its raw export
reproduces the `tessera_case3_s21_77ghz.csv` shipped here — made on a different day, by a
different person — to **5e-7 dB** maximum deviation.

## `tessera_case{1..6}_s21_77ghz.csv` — the six 77 GHz automotive designs

Six simulated interconnect designs from the Tessera collaborator's HFSS export, cleared by
the owner for this repository on 2026-08-19. Each file is 401 rows spanning **70–90 GHz**
with columns `freq_hz, s21_re, s21_im, s21_abs_db, s21_phase_deg`, and each carries its own
provenance in `#` header comments.

| file | in-band loss, 75–81 GHz |
| ---- | ----------------------- |
| `tessera_case1_s21_77ghz.csv` | −0.292 … −0.289 dB |
| `tessera_case2_s21_77ghz.csv` | −0.288 … −0.279 dB |
| **`tessera_case3_s21_77ghz.csv`** | **−0.547 … −0.513 dB — the most demanding of the six for this pipeline** |
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

## `tessera_tsv_s21.csv` — the Ka-band TSV line (LEGACY, kept for continuity)

> **Provenance (added 2026-09-23, F89/F90 in `notes/ESTABLISHED_FACTS.md`):** supplied by
> the authors in **August 2026**, ahead of the public release. A fresh-context
> investigation found it is an **exactly lossless, closed-form reciprocal two-port**
> (|S21|²+|S11|²=1 to 1.4e-9 at every row; arg S11 − arg S21 = −90.000000°) that **no
> checkpoint of this architecture could have produced** — four regressed head outputs
> cannot satisfy an identity at 1e-9. It is **not reproducible from the public release**
> (unit/convention/renormalization searches all rejected; best match 3.14 dB RMS, wrong
> shape). It is kept in this repository **only for continuity with v1.0 figures** that
> already cite it; new work should use `tessera_tsv_s21_public.csv` below, which *is*
> the public checkpoint's own output at its own canonical geometry.

A frequency-swept scattering-parameter transfer function for a
**single Through-Silicon-Via (TSV) interconnect line** (one center signal via surrounded by
a ground ring). `InterconnectBlock(transfer_csv=...)` in `e2e/blocks.py` loads it and
resamples it onto a frame's frequency grid, as a physically-grounded alternative to the
default placeholder boxcar response. It is still the **default** value of
`e2e.blocks.TESSERA_INTERCONNECT_CSV`; `e2e/main/main_interconnect.py` no longer plots it
(it plots `tessera_tsv_s21_public.csv` instead) but still loads it in the tests that pin
its own numbers.

### TSV columns

| column | meaning |
| ------ | ------- |
| `freq_hz` | frequency (Hz), 1–40 GHz in 0.25 GHz steps |
| `s21_re`, `s21_im` | complex through / insertion transfer coefficient S21 |
| `s21_abs_db`, `s21_phase_deg` | S21 magnitude (dB) and phase (deg), for convenience |
| `s11_re`, `s11_im`, `s11_abs_db` | complex reflection S11 (magnitude in dB) |

`InterconnectBlock` uses `s21_re + 1j*s21_im` as the interconnect's frequency response.

### TSV geometry (as documented; not independently verifiable — see provenance above)

Single signal line, ground ring; nominal TSV geometry: radius 5 µm, pitch 60 µm,
height 100 µm, liner 0.5 µm. Over the pipeline's 28.5–31.5 GHz band the insertion loss is
≈ −7.0 to −7.9 dB (rising with frequency); there is a reflection resonance near ~8 GHz.

### TSV provenance and caveats

- **Phase is NOT minimum-phase.** An earlier version of this README claimed the phase
  column was "reconstructed as minimum phase" (`arg H = −Hilbert(ln|H|)`); that is FALSE
  for this file (measured 2026-09-23: 32.1° rms residual against a minimum-phase
  reconstruction of the same |S21|, while the same estimator reproduces the Case3 77 GHz
  file's phase to 0.000°). The phase in this file is whatever the authors supplied; do not
  assume any reconstruction method for it. (Minimum-phase reconstruction genuinely is used
  for the six 77 GHz case files below — that claim is correct there, just not here.)
- These numbers were said to come from the authors' **physics-informed GNN surrogate for
  TSV networks** (an HFSS-finetuned model that predicts the S-matrix from array layout +
  geometry). The public checkpoint (`models/best_model.pth`,
  github.com/HiPerCAS/tessera) is a **different instrument** and does not reproduce this
  file — see "Regenerating the TSV" below.
- **Attribution:** settled 2026-08-29, updated 2026-09-23 — see the Attribution section at
  the top of this file. The paper is now published: doi 10.1109/TCAD.2026.3718807.

### Regenerating the TSV

**Not reproducible from the public repository** (see the provenance note above — an
exactly lossless closed-form two-port, not a checkpoint's output). The public checkpoint
(`models/best_model.pth` in github.com/HiPerCAS/tessera) gives about −0.57 dB at 30 GHz
where this file has −7.48 dB (measured 2026-09-23). Use `tessera_tsv_s21_public.csv`
below for a version of this file that *is* regenerable from a clean clone.

## `tessera_tsv_s21_public.csv` / `tessera_tsv_s21_public_ka_scaled.csv` — the public checkpoint

**Fully reproducible** from a clean clone: `python -m e2e.interconnect_surrogate.fetch`
(pulls the pinned public checkpoint) then
`python -m e2e.data.interconnect.regenerate_tessera_tsv` (writes both files below). Each
file's own `#`-comment header records the exact geometry, arrangement, checkpoint commit +
sha256, wrapper version, scale factor, generation date and command line — see
`e2e/data/interconnect/regenerate_tessera_tsv.py`. Same column layout as the legacy TSV
file above; **phase is the surrogate's own phase, not a minimum-phase reconstruction**, in
both files.

- **`tessera_tsv_s21_public.csv`** — the **direct** evaluation: upstream's own canonical
  demo geometry (radius 5 µm, pitch 60 µm, height 100 µm, liner 0.5 µm, 300 K, `ring3x3`
  arrangement — `SHIPPED_TSV_DESIGN`), `scale=1.0`, over a 1–90 GHz sweep (0.25 GHz step).
  This is what `InterconnectBlock(transfer_csv=...)` (the CSV/interpolation mode) reads,
  and what `e2e/main/main_interconnect.py` now plots as "Tessera TSV". Over the pipeline's
  28.5–31.5 GHz band: **−0.574 dB mean insertion loss, 0.0033 dB p-p ripple**
  (measured 2026-09-23, F90 in `notes/ESTABLISHED_FACTS.md`) — very different from the
  legacy file's −7.0 to −7.9 dB, because the legacy file is not this checkpoint's output.
- **`tessera_tsv_s21_public_ka_scaled.csv`** — the **scale-model** evaluation at the
  pipeline's Ka band: what `InterconnectBlock(source='tessera')` (the *live surrogate*
  mode) actually applies over 28.5–31.5 GHz **by default** (`scale=None` auto-resolves to
  x2 for this band — `e2e.blocks._resolve_tessera_scale`, F91 in
  `notes/ESTABLISHED_FACTS.md`): the same canonical geometry evaluated at 2x its lengths
  and half the frequency, reported back at the real Ka frequencies. This is **not** the
  same number as the direct file's in-band slice above — −0.551 dB mean, 0.0075 dB p-p
  ripple (measured 2026-09-23) — because the two `InterconnectBlock` modes do not evaluate
  the same thing over this band; neither is "more correct," they are different declared
  approximations and both are recorded for that reason.

**Caveats carried over from the legacy file:** these are simulated, not measured; cite the
same TCAD 2026 paper (Attribution, above); a passivity guard is enforced by
`TesseraTSV(passivity="raise")`, the default the regeneration script uses, so a future
re-run over an out-of-range geometry fails loudly rather than writing a fabricated curve.
