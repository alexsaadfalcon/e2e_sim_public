# Beautification pass — implementation report

Clone: `<scratchpad>/beauty` (never the main tree). Rendered with
`python -m webapp.rehearse --expand-details --out <scratchpad>/render_all`.

**One-line result.** The Thrust-1 range-azimuth map went from **12.4 % of its panel
under a 289 px band of wrapped prose** to **50.3 % of its panel with a 60 px HTML
header**, the text above the first panel went from **396 px / 8–12 lines** to
**149 px / 4 lines**, and no honesty clause was deleted — every one is pinned, by
product, against the per-arm Details disclosure in a new test.

---

## 1. Per spec item, what was done

### 2.1 Page grid
- Page container 1560 px, 20 px side padding → 1520 px of content (`.app-shell`,
  `webapp/assets/demo.css`). Two 746 px columns with a **28 px gutter** (was 13 px with
  no rule) plus a full-width 1 px `#dfe4ea` section rule between the arm headers and
  the panels. Row gap 16 px.
- **Single-arm rule** implemented: one product → one **1008 px** panel, left-aligned
  (`SINGLE_PANEL_WIDTH`); two or more → the same two-up 746 px grid as the A/B case.
  `cancel_results.png` now shows the map at the same shape and scale as every other
  screen instead of a 4.7 : 1 letterbox.

### 2.2 Panel geometry — the load-bearing part
- `PANEL_HEIGHT = {map: 540, table: 388, pr: 552}` in `webapp/pipeline_runner.py`;
  `FIGURE_HEIGHT` derives each figure's fixed height (header 60 + padding 16).
  Every figure asserts to its row's height in
  `tests/test_webapp_layout_acceptance.py::test_every_figure_height_is_its_rows_fixed_height`.
- `_heatmap_margin_t()` is now a **constant function** (it takes and ignores the old
  `title` argument). That removes the mechanism behind defects 1, 3 and 5 in one move.
- Measured map-row geometry: panel 746 × 540, plot **594 × 341 = 50.3 % of panel area,
  63 % of panel height**. Both arms' plot-top y differ by **0.0 px** on every screen.

### 2.3 Where the words go
- **Run-identity line** (18/600, one line) replaces the `Results` H3 and the run half of
  the banner: `Thrust 1 · RF circuit knobs · munich (Ka-band, 30 GHz) · 5 frames ·
  run #1 16:54:03`. Built entirely from the run; shortened **at clause boundaries**
  (never mid-phrase, never with a mark) by a deterministic ladder when it would not fit.
- **Arm chip** (20/700 + 8 px dot, arm colour): `A — LNA bias current (mA) 8 mA`.
  When the knob value is prose and the chip would wrap (Thrust 4), the VALUE moves to
  the front of that arm's caption rather than being clipped.
- **Arm caption** (16/400, one line): the knob value overflow, then the first clause of
  the first run note — which on Thrust 5 is the live-vs-stored ADC gate's own number,
  `live chain vs stored ADC over 5 frame(s): max |diff| = 0 of 4096 LSB (12-bit)`.
- **`▸ Details (provenance, band, clip)` per arm, closed by default**: the full banner,
  every run note **untruncated**, the screen note, and every panel's clauses grouped
  under that panel's title.
- **Screen note** moved to a single muted 15 px line at the page foot, and is also
  inside both arms' Details.

### 2.4 Block Diagram tab
Sticky 56 px control bar (preset · Load · divider · Frames · **Run pipeline** · Cancel ·
status, one line) → work row (diagram | 28 px | 460 px parameter editor) → one-line
legend → **`▸ Presenter notes (Thrust N)` disclosure, closed by default**. Parameter
editor puts the **control above its help**, help capped at 2 lines with `▸ more`.
Alt-path edges 1 px at 30 % opacity. Run pipeline is at **y ≈ 194** (was y ≈ 1935).

### 3 Typography
Full scale in `demo.css` (`:root` tokens). Measured on the rendered page: page text
15–26 px, in-figure text 17–26 px, **nothing under 15 px anywhere, nothing under 17 px
inside a figure.** Figures carry **no title and no subtitle** — pinned by two tests.

### 4 Figure rules
- **Colour bar**: vertical, thickness 14, `len` 0.90, `xpad` 0, ticks 18 px, **no
  title**. Units/clip/sharing are one caption sentence:
  `dB rel. peak · clipped at -67.1 dB · same colour scale on both arms`.
- **Shared limits stated once per row**: the sharing sentence is on **arm A's caption
  only**; the exact `zmin/zmax` pair (with its `(was X)` provenance) is in **both arms'
  Details**. Idempotent across re-renders — the first pass remembers the arm's own
  zmin on the panel, because a second pass would otherwise silently drop `(was X)`
  (a real bug this work introduced and a test caught).
- **Statistic strip**: a reserved 68 px band above the plot, `yref="y domain"`, y > 1,
  no background pill. Headline number at 26/600 in the arm's colour; above it, at
  17 px, the per-frame readouts (`brightest -13.8 dB @ 37 m · frame 4 of 5`). Both
  restep with the clock. It can no longer cover a return.
- **One transport per screen**: per-figure `sliders`/`updatemenus` removed everywhere;
  one HTML `⏸ ▶ / frame N of M / slider` in the run-identity row, driven by the same
  `assets/results_clock.js` clock. Scrubbing now parks **both** arms.
- **Detector map**: short `scoring ≤ 40 m` tag at the right end of the dashed line
  (17 px, `rgba(45,58,74,0.7)`), hit rule in the caption.
- **Scoreboard**: ≤ 8 rows, 17 px, fixed 312 px figure; the CI row, seed-spread row,
  OOD rows, 3rd-corpus row, connector row, the threshold subline and all four caveat
  sentences moved verbatim into Details.
- **PR panel**: two-column × three-row legend, highlighted arm width 5 / opacity 1.0,
  others width 2 / opacity 0.45, the null/chance-floor arm never dimmed.
- **Plot background**: `#E5ECF6` plot / `#ffffff` paper on every figure, maps included.

### 5 Colour and spacing
Tokens exactly as specified; `#0fb9b1` / `#8854d0` are reserved to the block diagram.
Spacing scale 4/8/12/16/28/40.

---

## 2. Acceptance checks — measured, not eyeballed

`webapp/rehearse.py` now dumps `<preset>_geometry.json` straight out of the DOM beside
each PNG (panel rects, `.nsewdrag` plot rects, every rendered font size, CSS-clipped
elements, transport counts, page height). `<scratchpad>/measure.py` runs the pixel
checks over all eight screens.

**93 of 96 pixel checks pass across 8 screens.** The three failures are the same check
on the three Thrust-5 screens.

| # | Check | Result | Measured |
|---|---|---|---|
| 1 | No text < 15 px on the page, < 17 px inside a figure | **PASS** ×8 | page 15–26, figure 17–26 |
| 2 | ≤ 4 lines and ≤ 150 px above the first panel | **PASS** ×8 | 149 px (was 396–512) |
| 3 | First panel row inside the first 1000 px | **PASS** ×8 | bottom y 843–871 |
| 4 | Equal row heights; arms' plot tops within 2 px | **PASS** ×8 | Δh 0, Δy 0.0 |
| 5 | Plot ≥ 50 % of panel area, ≥ 40 % of height | **PASS** ×8 | 50.3–55 % / 63 % (was 12.4 %) |
| 6 | In-figure text band ≤ 20 % of panel height | **PASS** | 68 px of 540 = 12.6 % |
| 7 | One-line titles, zero `<sup>` subtitles | **PASS** ×8 | 0 figure titles rendered |
| 8 | Statistic never over the data | **PASS** | `y domain` y = 1.02/1.13, both > 1 |
| 9 | A/B gutter ≥ 24 px | **PASS** ×8 | 28 px + a rule |
| 10 | Exactly one transport per screen | **PASS** ×8 | 1 button, 1 slider, 0 Plotly; **0 / 0** on Thrust 3, which animates nothing |
| 11 | Scoreboard ≤ 8 rows, ≤ 40 px rows, ≤ 40 px blank | **PASS** | 8 × 34 + 40 header = 312 |
| 12 | No truncated visible text | **PASS** ×8 | 0 clipped, 0 ellipses |
| 13 | Same plot background on every panel | **PASS** ×8 | one `#E5ECF6` |
| 14 | Page height ≤ 2200 px | **PASS ×5 / FAIL ×3** | T1 1000, T2 2084, T3 1000, T4 1548, cancel 1000; **T5 ×3 = 2480** |
| 15 | Details closed by default; nothing lost when opened | **PASS** ×8 | 0 of 2 open; see §3 |
| 16 | Diagram canvas **and** Run visible in the first 1000 px | **PASS** | Run at y ≈ 194, diagram y 300–930 |
| 17 | Diagram node labels ≥ 12 px of rendered ink | **FAIL** | ~10 px — see §4 |
| 18 | Operator card not open by default, label names the thrust | **PASS** | `▸ Presenter notes (Thrust 3)` |
| 19 | Every input control above its help text | **PASS** | `param_editor` reordered |
| 20 | Same product, same geometry on every screen | **PASS** | pinned by a unit test too |

### Check 14 cannot pass as specified — the arithmetic
A Thrust-5 A/B screen has **four** products. The spec's own §2.2 row heights give
540 + 540 + 388 + 552 = 2020 px, plus 3 × 16 px row gaps, plus 149 px of header, plus
154 px of page chrome and tabs, plus the page-foot note ≈ **2480 px**. §2.2 and §7.14
are mutually exclusive for a four-product screen; no combination of the spec's own row
heights fits 2200. It is still down from **2968–3084 px** (3.1 screens → 2.5 screens).
Getting under 2200 needs a composition decision that is not mine: rendering the
offline PR panel **once** for the row instead of per arm (it is identical on both arms
by construction, and says so) would give 1912 px.

---

## 3. Check 15 — nothing was deleted, only moved

This is the check the spec says must not be skipped, so it is pinned twice:

- `tests/test_webapp_layout_acceptance.py::OLD_SUBTITLE_CLAUSES` lists **every clause
  the pre-change panel subtitles carried, per product, verbatim**, and asserts each is
  in `panel_text(fig)` (= title + caption + the whole Details body). Parametrised over
  `range_az`, `range_el`, `range_profile`, `subspace_err`, `fft`.
- `rehearse.py --expand-details` opens every disclosure, writes
  `<preset>_results_details.png` and stores the expanded text in `summary.json`, so the
  claim is checkable on the rendered page and not only on a dict.
- The shared-limits clause, which the sharing pass adds at render time, has its own
  test — including that `(was -40.0)` survives a second render pass.

---

## 4. Check 17 — the one I could not satisfy, and why

The diagram's content is ~1656 layout units wide in a ~1000 px column, so the fit zoom
is ~0.60 and a 20 px CSS label renders ~10–12 px of ink. The first attempt pinned
`minZoom = 0.85`, which does give ≥ 12 px — **and crops the diagram**: the rendered
1600 × 1000 page showed the source node cut in half on the left and the whole ADC-cube
chain running off the right edge. A cut-off block diagram on stage is a worse defect
than a 10 px label, and it fails check 16, which matters more, so `minZoom` was removed
and the font raised from 12 → 20 px (node box height 60 → 76 px to absorb the wrap).

Two ways to actually get both, neither done tonight:
1. **The spec's own remedy** (§2.4): collapse the ADC-cube subgraph into a single node
   with a "show" affordance. That is the real fix and it is a `_POSITIONS` change.
2. Narrow the parameter editor 460 → 400 px, giving the diagram ~1060 px → zoom ~0.64 →
   ~13 px of ink. Cheap, but it partly undoes the spec's own reason for widening it.

---

## 5. Defects found and fixed during the pass (beyond the spec)

Each of these was found by **rendering and reading**, not by a dict test:

1. **The sharing pass lost its `(was X)` provenance on re-render.** `_apply_shared_z`
   read the arm's own zmin off the trace, which by the second pass already held the
   shared value — and `_render_results` re-runs on every tab switch. Now remembered on
   the panel (`own_zmin`). Caught by `test_sharing_is_idempotent`.
2. **The statistic strip's two lines overlapped** side by side (215 px + 356 px of text
   in a 545 px plot). Stacked them instead, and shortened the sub-line.
3. **The run-identity line was CSS-clipped** on four screens. `text-overflow: ellipsis`
   draws a mark that is **not in `innerText`**, so it passed an ellipsis-count check
   while visibly ending in "…". The geometry dump now measures `scrollWidth >
   clientWidth` for the chrome, and the line shortens at clause boundaries.
4. **A bottom legend costs ~50–65 px of PLOT height**, not 32 px of panel: Plotly takes
   it out of the axis domain. The detector-map and subspace legends were duplicating
   the statistic strip and the axis titles respectively, so they went; the PR panel got
   `margin autoexpand=False` and an 80 px strip.
5. **The PR legend was cutting off two of six arms**, including the null/chance-floor
   curve — the panel's honesty anchor. Strip widened, the highlighted entry's CI moved
   to the caption (it was overdrawing the entry beside it), null name shortened.
6. **A stale claim in `pipeline_runner.py`** citing a "20 px standing threshold" that
   the code no longer enforces: retracted in place, with a pointer to the test that is
   now the authority (CLAUDE.md's provenance rule).
7. `thrust4` could not run in the clone at all — the Tessera checkpoint lives in a
   gitignored `_models/` directory. Added a junction; no code change.

8. **Opening `Details` showed nothing.** `.arm-header { max-height: 96px; overflow:
   hidden }` enforced the 150 px budget by clipping the header — and the Details
   disclosure is *inside* the header, so the expanded body was cut to a 4 px sliver.
   The whole honesty mechanism was one click away and then invisible. The cap now sits
   on a `.arm-summary` wrapper (chip + caption) instead. **Nothing but rendering the
   page with `--expand-details` and looking at the PNG could have found this** — every
   dict-level test still passed.
9. **The PR caption was clipped** once it carried the AP, the CI and arm B's clause
   (~110 chars against the ~86 a 746 px column fits at 16 px). Shortened; the long
   forms stay in Details.
10. **The run-identity ladder dropped too much** on the Thrust 5 ML screen, leaving
    "Thrust 5 · 5 frames · run #1" — no preset, no corpus. It now gives up the frame
    count before the environment.
11. **A dead transport on Thrust 3.** That screen's only product is a line plot with no
    frames, so the play button and slider sat there doing nothing. The row now omits
    the transport unless some figure on the page carries frames.
12. **The Details body scrolled inside itself** (`max-height: 520px; overflow-y:
    auto`), so the tail of the Thrust 5 disclosure was below its own inner fold and a
    photograph of the *opened* screen — the thing the honesty text exists for — still
    missed clauses. It expands now; the opened Thrust 5 CFAR page is 3630 px and every
    line is in one capture.

---

## 6. Left for someone else

- **`webapp/demo_presets.py` (not mine to edit):** `cancel_results.png` still shows the
  two-arm Thrust-1 screen note ("the two maps share one colour scale … the 0.5 mA arm's
  background is visibly brighter") over a **one-arm** screen. Hostile round 10 §1.10.
  It is now at the page foot rather than at the top, but it is still wrong.
- **`docs/DEMO_RUNBOOK.md`:** regenerate after this change. The transport is now one
  control in the run-identity row, the per-panel sliders are gone, "dragging a slider
  parks one arm" (§3.3) no longer applies, and the honesty text is behind `▸ Details`.
- **Check 17 / check 14** as described above — both need a decision, not a patch.


---

## 6b. Test suite

`python -m pytest tests/ -q -p no:randomly` in the clone:

**2325 passed · 165 skipped · 2 xfailed · 1 FAILED.**

The single failure is `tests/test_runbook.py::test_panel_titles_match_the_rendered_source`
-- a file I was told not to edit, whose owner regenerates the runbook after this pass.
It scrapes panel titles out of `webapp/pipeline_runner.py`'s SOURCE and asserts each
appears verbatim; the titles are now `set_panel(..., title="Range profile")` rather
than the old `"Range profile (non-coherent over channels)"` (the qualifier is a caption
clause). The fix is one line in the runbook's extractor, or simply regenerating it.

The skips are the usual hardware/human gates (`sionna`, `slow`, `gui`, `browser`), not
anything this pass introduced.

Test files rewritten rather than deleted: `test_webapp_figures_wave2/3/4/7/8/9/11`,
`test_detector_scoreboard`, `test_webapp_ab`, `test_webapp_detector`, `test_webapp`,
`test_webapp_rehearsal`, `test_webapp_run_notes` -- every assertion that pinned a
figure title or subtitle string was repointed onto `panel_text()`, which returns title
+ caption + the whole Details body, so a test pins the CLAUSE rather than the place.
New: `tests/test_webapp_layout_acceptance.py` (29 tests).

### Commits (in the clone, never pushed)

```
256cb97 fix(webapp): the Details body expands instead of scrolling inside itself
bc7e658 test(webapp): extend the check-15 honesty pin to the corpus-replay panels
6024d2d fix(webapp): no transport on a screen with nothing to animate
4f60ad3 fix(webapp): the Details disclosure was being clipped shut, and four smaller render-found defects
2d78d4d fix(webapp): render-measured corrections to the fixed-geometry layout, and the tests repointed to where the words now live
fdd1961 feat(webapp): fixed-geometry results layout -- figures carry no prose, panels carry no surprise
fe7720d wip(webapp): layout redesign checkpoint (protects against a sub-agent stash)
```

`git diff --shortstat c33bb64..HEAD`: **22 files changed, 4054 insertions(+), 1903
deletions(-)**.

---

## 7. Where things are

- **Clone** (all work; the main tree was never touched):
  `C:/Users/asf3/AppData/Local/Temp/claude/C--Users-asf3-workspace-e2e-sim-public/56ce5296-5481-4d28-af02-421b8a5acf69/scratchpad/beauty`
- **Rendered screens** (8 `*_results.png`, 8 `*_results_details.png`, 7 `*_card.png`,
  8 `*_geometry.json`, `summary.json`):
  `.../scratchpad/render_all/`
- **Pixel-check runner**: `.../scratchpad/measure.py` (`python measure.py render_all -v`)
- **Unit acceptance checks**: `<clone>/tests/test_webapp_layout_acceptance.py`
- **Extra junction needed in a clone**: `e2e/interconnect_surrogate/_models` (the
  Tessera checkpoint), without which Thrust 4 cannot run at all.
