# Validation card — fixed schema

One card per node of `docs/validation/TREE.md`. Every section is mandatory; write `UNKNOWN`
(with why) rather than leave it blank or guess. Cards are comparable only because the schema
is fixed: do not add, rename or reorder sections. File: `docs/validation/cards/<NODE-ID>.md`
(e.g. `cards/L2-B14.2.md`). Small evidence scripts (≤200 lines, no data) go in
`docs/validation/evidence/<NODE-ID>/`; everything else stays in the validator's scratch dir.

Validators write ONLY under `docs/validation/`. A card never edits code, tests, the ledger
or the tree; findings flow out of it (see `README.md`).

The YAML front matter is machine-read by the tree-status generator; the sections below it
are read by people. Keep the front matter exactly this shape.

```yaml
---
node: L2-B14.2                      # TREE.md ID, verbatim
title: QuantizerBlock.apply          # node name as in TREE.md
tier: S                              # S | O | F  (who validated)
validator: <model> / <agent name>    # e.g. Sonnet / sim-reviewer
date: 2026-09-24
tree_sha: 88f6240                    # `git rev-parse --short HEAD` at validation time
dirty: [e2e/radar_config.py]        # `git status --short` files at that time, or []
status: PARTIAL                      # UNVERIFIED | PARTIAL | VERIFIED | SUSPECT | RETRACTED
missing: [cross-check]               # for PARTIAL: which of oracle|invariance|cross-check|loud-failure|adversarial
evidence: [tests/test_chain_receive.py::test_quantization_error_is_bounded_by_half_an_lsb, F80]
demo_path: [T5]                      # Thrust presets this node sits on, or []
supersedes: null                     # previous card sha/date if this is a re-validation
---
```

## 1. Claim

What the node claims, in two forms: (a) the TREE.md row, verbatim; (b) the code's OWN claim
(docstring / comment / README sentence) with `path:line`. If the two disagree, say so here —
that is already a finding.

## 2. Contract

Inputs and outputs (shape, dtype, units, device); invariants the node promises; assumptions it
INHERITS, each cited to a row of `notes/PHYSICAL_ASSUMPTIONS.md` or a `path:line`. State the
conditions under which the node is meant to hold (band, array, MIMO scheme, chirp count).

## 3. Oracle test

The input whose correct output is known analytically, the tolerance, the exact command that
runs it from a clean clone (plus any data it needs and where that data comes from), and the
measured result with its date. If an existing test is the oracle, name it `file::test` and
say whether you re-ran it. "The docstring says it was measured" is not a result.

## 4. Invariance / conservation test

The property that must hold for all inputs (energy, linearity, identity under no-op
parameters, determinism under a fixed seed, shift-equivariance), the command, the result.

## 5. Independent cross-check

A second path to the same answer that shares NO code with the node: a hand-rolled loop, a
closed form, an external library, a re-derivation from the physics. State the agreement as a
number with units. Agreement between two things that share code is not a cross-check
(`notes/RIGOR_STANDARD.md`, standing red flags).

## 6. Loud-failure check

The inputs the node must refuse (wrong shape, wrong units, NaN, empty, degenerate rank, wrong
band, wrong domain), the error expected, the error observed. A silent plausible output is a
finding of severity S1 or higher.

## 7. Adversarial read

"What is the most plausible way this node is silently wrong, and what test would catch it?"
Write the mechanism, then name the test that catches it — existing (`file::test`) or
proposed (a one-line spec). If proposed, that test is the card's first recommendation.

## 8. Verdict

One of `UNVERIFIED` / `PARTIAL (missing: …)` / `VERIFIED` / `SUSPECT` / `RETRACTED`, with one
sentence of justification. `VERIFIED` requires §3–§7 all answered with a named test or ledger
F-number. `SUSPECT` means a specific reason to believe the node is wrong: stop and hand back
immediately. `RETRACTED` means a stored claim about this node (docstring, README, ledger,
memory) was found false; §10 must say where it still lives.

## 9. Evidence links

Every test name, ledger F-number, script (`docs/validation/evidence/<NODE-ID>/…` with its
command line), figure and log the verdict rests on. Each with the date and the `tree_sha` it
was produced at.

## 10. Findings

| # | severity | file:line | finding (one line) | conditions it holds under | disposition |
|---|---|---|---|---|---|

Severity: `S0` a wrong answer reaches a user (screen, README number, corpus) · `S1` wrong
under the stated operating conditions · `S2` an undocumented or unconditional assumption ·
`S3` hygiene (naming, stale comment, missing loud failure). Disposition is a proposal, not a
decision: `ledger` (a measured fact for `notes/ESTABLISHED_FACTS.md`), `shard` (a coder task,
with the owned-file list), `doc` (a docstring/README correction), `none`. A retracted claim
still alive anywhere is always at least S2 and always gets a disposition.

## 11. What this card does NOT establish

The scope boundary, in plain words: bands, arrays, schemes, data or code paths this card did
not exercise; conclusions a reader might be tempted to draw from it that it does not support.
This section is what stops the card from being applied unconditionally later.

## 12. Questions for the owner (only if a decision is genuinely his)

Each as a two-option dichotomy with a recommendation. Leave the section with `none` otherwise.
