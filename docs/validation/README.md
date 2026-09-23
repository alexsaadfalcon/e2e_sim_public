# docs/validation — the ground-up validation campaign

Owner directive (2026-09-23): validate every module top-to-bottom, distributed, with the
knowledge organised on three levels — purpose/UX (L0), block contracts (L1), numerical
implementation (L2). This directory is where that knowledge lives. It is public (it ships
with the repo); the private ledger `notes/ESTABLISHED_FACTS.md` is referenced by F-number only.

## Files

| file | role |
|---|---|
| `TREE.md` | the map: every node with an ID, its claim, its inherited assumptions, its oracle, its status and evidence. Status/evidence columns are GENERATED from cards (below). |
| `CARD_TEMPLATE.md` | the fixed schema every validator fills, one card per node. |
| `cards/<NODE-ID>.md` | the cards. **Cards are the source of truth; the tree's status column is a view of them.** |
| `evidence/<NODE-ID>/` | small scripts (≤200 lines, no data) a card cites, each runnable from a clean clone. |
| `WAVE<n>.md` | a fan-out: the nodes validated in that wave, one neutral brief each. |

The wider standard is `notes/RIGOR_STANDARD.md` (the five checks and the scoring), the
configuration-constant register is `notes/PHYSICAL_ASSUMPTIONS.md`, and the per-block physics
verdicts are `notes/PHYSICS_JUSTIFICATION_AUDIT.md` / `docs/PHYSICS.md`. This directory does
not restate them; a card cites them.

## Rules in force

1. **Validators never edit code.** A validator writes only under `docs/validation/`. Fixes
   are coder shards; a validator who finds a bug hands it back, never patches it.
2. **The reference band is Ka-band, 28.5–31.5 GHz, 30 GHz carrier, everywhere.** A card that
   validates a node at another band says so in §11 and does not count as evidence for Ka.
3. **Claims carry provenance.** Every number on a card has a date, a `tree_sha`, and the
   conditions it holds under. A stored claim (docstring, README, ledger, memory) is a claim to
   re-verify, not an answer; if it is load-bearing, re-run it.
4. **`VERIFIED` needs a name.** No node is VERIFIED without a named test (`file::test`) or a
   ledger F-number in its card's `evidence:`; the generator refuses otherwise.
5. **Nothing is guessed.** `UNKNOWN`, with the reason, is a valid entry in every section.
6. **A retracted claim still alive anywhere is a finding** (S2+), with a disposition, every time.

## Tier rules

| level | validator | scope of one assignment | may run |
|---|---|---|---|
| L2 | Sonnet (`sim-reviewer`-style, read-only + write under `docs/validation/`) | ONE node; must end in a card | CPU tests, small scripts; GPU only if the brief says so |
| L1 | Opus | ONE block or cross-cutting contract; reads its L2 children's cards first, may re-run their commands | as above; RT solves only where `RUN_SIONNA=1` is stated in the brief |
| L0 | ONE Fable reviewer per wave | the purpose/UX nodes, read against the L1 cards; the "what would an expert ask at the poster" pass | reads; renders the app (`python -m webapp.rehearse`) and reads the PNGs |

Status composition: an L1 node may be `VERIFIED` only when every L2 child on the demo path
has a card at `PARTIAL` or better and none is `SUSPECT`; an L0 node inherits the worst status
of the L1 nodes its screen runs through. Any `SUSPECT` card triggers an Opus re-read in the
same wave before it reaches the owner (automatic adversarial review, memory 2026-08-24).

Briefs are NEUTRAL: they name the node, the deliverable, the scratch dir, the tier and the
stop condition, and never the suspected answer. The ledger is handed over as "claims to
re-verify", not as findings to confirm.

## How a finding flows

```
card §10 finding
  ├─ measured fact  ──► notes/ESTABLISHED_FACTS.md  F-number (evidence = the card + its command)
  ├─ code change    ──► coder shard (swarm skill): owned-file list, cites card + F-number,
  │                     lands with a test named after the card's adversarial question
  ├─ wrong stored claim ──► corrected WHERE FOUND (docstring / README / PHYSICS.md / memory),
  │                     retraction recorded, never deleted
  └─ after the fix lands ──► the validator re-runs the card's commands at the new tree_sha,
                            writes a new card (supersedes: old), generator regenerates TREE.md
```

The seat orchestrates this loop and reviews; it does not implement any step of it.

## The tree-status generator (spec only — not written in wave 0)

`python docs/validation/tools/gen_tree_status.py [--check]`, modelled on
`notes/tools/gen_facts_index.py`:

- Reads every `cards/*.md` front matter (`node`, `status`, `missing`, `evidence`, `date`,
  `tree_sha`, `supersedes`). The newest card per node (by `date`, then `supersedes` chain) wins.
- Rewrites, in `TREE.md`, the `Ev`/`St` cell of every row whose ID has a card, between
  `<!-- BEGIN GENERATED -->` / `<!-- END GENERATED -->` markers that wave 1 adds to each table;
  rows with no card keep the hand-written wave-0 status, prefixed `(w0)` so it is visibly
  provisional.
- Refuses (`--check` exit 1) when: a card says `VERIFIED` with an empty `evidence:`; an
  `evidence:` test name does not exist under `tests/` (checked by `pytest --collect-only -q`);
  a TREE row is `VER` with no card; a card's `node` is not in TREE; the generated block is
  stale relative to the cards; an L1 `VERIFIED` has a demo-path L2 child below `PARTIAL`.
- Emits a one-line summary per level: counts by status, demo-path counts, cards older than the
  current `tree_sha` for a file they cite (staleness list, not a failure).
- Never touches cards; never touches anything outside `docs/validation/`.

## Arrival rule for new artifacts

A new preset, scene, corpus, config, pkl or CSV does not land without stating, in its own
metadata AND in a one-line note in the PR/commit message:

- **band** — carrier, span (start/stop), number of points, and which of these the frame's
  axis actually carries (`freq_plan` meta in pkls, `f0_hz`/`band_hz` in npz meta, `FrequencyPlan`
  in presets);
- **materials** — every radio material and roughness used, with the ITU validity range at that
  band, or the extrapolation/stand-in policy applied;
- **provenance** — the command, commit SHA and seed that produced it (or "hand-made", author,
  date); for collaborator data, the attribution line and what was reconstructed vs measured.

It also adds or attaches to a `TREE.md` node and lands with a card stub at `UNVERIFIED`. A
new configuration constant gets a row in `notes/PHYSICAL_ASSUMPTIONS.md`. The proposed
`band_audit` release gate (register, last section) is what will enforce the band part.
