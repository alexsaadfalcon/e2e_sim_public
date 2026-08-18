"""
Pre-ship dedup / train-vs-eval leak gate for `e2e.ml` radar-ML corpora.

WHY THIS EXISTS: `e2e.ml.rt_scenes._stable_seed` used to hash only `(tier, frame_idx,
seed)` -- corpus identity never entered the draw, so two corpora sharing a `(tier,
seed)` (a common accident: several corpora were generated with the project's default
`seed=0`/reused seeds) drew IDENTICAL scenes at every `frame_idx`. An audit found the
worst case live on disk: `rt_ablation_txoff`'s ENTIRE val+test split (70 scenes) sat
inside BOTH `rt_corpus_v1/ti_iwr1443_D1`'s and `rt_no_interconnect/ti_iwr1443_D1`'s
train splits, silently train-contaminating any conclusion drawn across that pair.
`_stable_seed` (and its analytic-path sibling `e2e.ml.dataset._stable_scene_seed`) now
require a `corpus_tag`, which fixes it FOR NEW GENERATION -- this module is the
complementary check for CORPORA ALREADY ON DISK, so an old collision (or a future one
introduced by two runs that happen to pass the SAME `corpus_tag`, e.g. a copy-pasted
generation script) fails a build instead of silently shipping.

Some cross-corpus scene reuse is INTENTIONAL (a matched ablation pair sharing scenes
by design, or a domain-randomization sweep whose legs are the same scenes at different
impairment severities) -- the point of this gate is that reuse must be DECLARED on the
allowlist, not merely true by accident. See `DEFAULT_ALLOWLIST` / `--allowlist`.

WHAT IT CHECKS, per corpus (a "corpus" is any directory under `--root` holding its own
`manifest.json`, identified by its path relative to `--root`):

  (a) SELF-OVERLAP: a corpus's own train/val/test scene-hash sets must be pairwise
      disjoint (should be structurally impossible given `write_manifest`'s scene-level
      split, but a bug there -- or a hand-edited manifest -- would show up here).
  (b) CROSS-CORPUS LEAK: for every pair of DIFFERENTLY-NAMED corpora, corpus A's train
      set must not intersect corpus B's val-or-test set (checked both directions),
      UNLESS `(A, B)` is on the allowlist.

Scene identity is a hash of each frame's `meta["targets"]` (range_m, sin_az,
object_class -- the fields common to both target-tuple vintages on disk; a 4th
`surface_range_m` field exists on newer corpora but is derived from the first three,
not independent, so it adds no discriminating information here), rounded to absorb
float jitter -- see `frame_scene_key`'s docstring for why: two corpora built from the
SAME seed under DIFFERENT `rt_scenes.py` code versions were measured to draw
near-duplicate, not bit-identical, scenes (e.g. `rt_radial_v2` vs `gainfix_v1`/
`verify_fixed_v1`, all seed 7000: ranges differ by ~1.5 cm, sin(az) by ~2e-4 -- an
exact-match hash would miss this and call it "no leak").

CLI
---
    python -m e2e.ml.check_corpus_overlap [--root DIR] [--allowlist PATH]

Exit code 0 = no undeclared overlap found; 1 = at least one undeclared self- or
cross-corpus overlap (prints a readable report either way, to stdout).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

#: Gitignored on-disk corpus root (see e2e.ml.dataset.DATASETS_DIR's docstring);
#: computed relative to this file so it resolves regardless of the caller's cwd.
DEFAULT_ROOT = Path(__file__).resolve().parent / "datasets"

#: Declared-intentional cross-corpus reuse. A companion DATA file (not a corpus, not
#: gitignored -- lives next to this module), so "this reuse is on purpose" survives a
#: fresh checkout even though `DEFAULT_ROOT` itself does not.
DEFAULT_ALLOWLIST = Path(__file__).resolve().parent / "corpus_overlap_allowlist.json"

_SPLITS = ("train", "val", "test")

# Rounding for the scene-identity hash: coarse enough to absorb the ~1.5 cm / ~2e-4
# sin(az) drift MEASURED between same-seed draws from different rt_scenes.py code
# versions (see module docstring), fine enough that two GENUINELY different scenes in
# the same tier's placement envelope (~6-34 m range, several targets each) essentially
# never collide by chance.
_RANGE_ROUND_M = 1     # decimal places -> 0.1 m bins
_SIN_AZ_ROUND = 3      # decimal places -> 1e-3 bins


# --------------------------------------------------------------------------------
# Corpus discovery
# --------------------------------------------------------------------------------
def find_corpora(root: Path) -> List[Tuple[str, Path]]:
    """Every directory under `root` holding a `manifest.json`, as `(name, dataset_dir)`.

    `name` is the directory's path relative to `root`, POSIX-separated (`.as_posix()`)
    so the same corpus gets the same name on Windows and Linux/CI -- this name is both
    the report label and the allowlist key.
    """
    root = Path(root)
    out = []
    for manifest_path in sorted(root.rglob("manifest.json")):
        rel = manifest_path.parent.relative_to(root).as_posix()
        out.append((rel, manifest_path.parent))
    return out


# --------------------------------------------------------------------------------
# Scene-identity hashing
# --------------------------------------------------------------------------------
def _target_fields(t: Sequence, *, rounded: bool):
    """`(range_m, sin_az, object_class)` -- the fields common to BOTH on-disk
    target-tuple vintages (a 3-tuple `(range_m, sin_az, object_class)` and a 4-tuple with
    a trailing `surface_range_m`; see `e2e.ml.dataset`'s "Sample format" docstring).

    `rounded=False` keeps full float precision (the EXACT tier); `rounded=True` bins to
    `_RANGE_ROUND_M` / `_SIN_AZ_ROUND` (the NEAR tier). Returns None for anything that
    does not look like a target tuple rather than raising -- a long-running gate over many
    corpora should not crash on one malformed record.
    """
    if not t or len(t) < 3:
        return None
    try:
        r, sin_az, cls = float(t[0]), float(t[1]), str(t[2])
    except (TypeError, ValueError, IndexError):
        return None
    if rounded:
        return (round(r, _RANGE_ROUND_M), round(sin_az, _SIN_AZ_ROUND), cls)
    return (repr(r), repr(sin_az), cls)


def _scene_key(meta: dict, *, rounded: bool) -> Optional[str]:
    targets = meta.get("targets") or []
    fields = sorted(f for f in (_target_fields(t, rounded=rounded) for t in targets)
                    if f is not None)
    if not fields:
        return None
    return hashlib.sha256(repr(fields).encode()).hexdigest()[:16]


def frame_scene_key(meta: dict) -> Optional[str]:
    """EXACT scene identity: a hash of `meta["targets"]` at FULL float precision.

    TWO TIERS, and the distinction is what makes this gate worth running.

    An EXACT match means two frames hold bit-identical target geometry. Two independently
    drawn scenes do not do that -- the probability of matching to float repr precision is
    nil -- so an exact match is a genuine seed collision and is a hard FAIL.

    A NEAR match (`frame_scene_key_near`) bins to 0.1 m / 1e-3, which catches the OTHER
    failure mode: the same seed drawn under a different generator version, where scenes
    drift by ~1.5 cm and are duplicates in every way that matters for train/test leakage.
    But coarse bins also collide by coincidence -- two independent single-target D0 scenes
    26.892 m and 26.912 m apart land in the same 0.1 m bin, and with 1200 scenes over a
    28 m window that is a near-certainty, not bad luck. So NEAR is reported as a WARNING
    with its count, never as a failure.

    A gate that fails on coincidence gets switched off, and a gate that is switched off
    catches nothing. Hence: exact fails, near warns.

    Frames with zero targets are DELIBERATELY excluded from both tiers: every empty scene
    would otherwise hash identically to every other empty scene, and a tier that sometimes
    draws 0 in-FOV objects is ordinary variety, not a duplicate.
    """
    return _scene_key(meta, rounded=False)


def frame_scene_key_near(meta: dict) -> Optional[str]:
    """NEAR scene identity: the same content binned to 0.1 m / 1e-3. See
    `frame_scene_key` for why this tier warns rather than fails."""
    return _scene_key(meta, rounded=True)


def corpus_split_keys(dataset_dir: Path, manifest: dict) -> Dict[str, Dict[str, List[str]]]:
    """`{split: {scene_key: [relative filenames sharing that key]}}` for one corpus.

    Reads only each frame's `.npz` "meta" entry (matching
    `RadarFrameDataset.targets()`'s lazy-load pattern) so scanning a whole corpus root
    never decompresses the multi-MB "adc" arrays.
    """
    out: Dict[str, Dict[str, List[str]]] = {s: {} for s in _SPLITS}
    for split in _SPLITS:
        for fname in manifest.get("files", {}).get(split, []):
            path = dataset_dir / fname
            try:
                with np.load(path) as data:
                    meta = json.loads(str(data["meta"].item()))
            except Exception as exc:  # noqa: BLE001 - report and keep scanning
                print(f"  WARNING: could not read {path}: {exc}", file=sys.stderr)
                continue
            key = frame_scene_key(meta)
            if key is None:
                continue
            out[split].setdefault(key, []).append(fname)
            near = frame_scene_key_near(meta)
            if near is not None:
                out.setdefault("_near_" + split, {}).setdefault(near, []).append(fname)
    return out


# --------------------------------------------------------------------------------
# Allowlist
# --------------------------------------------------------------------------------
def load_allowlist(path: Optional[Path]) -> Set[frozenset]:
    """A JSON file of unordered `[corpus_a, corpus_b]` pairs -> a set of
    `frozenset({a, b})`. Accepts either a bare `[[a, b], ...]` list or `{"pairs": [[a,
    b], ...], ...}` (the shipped `corpus_overlap_allowlist.json` uses the latter so it
    can carry a `"_comment"` alongside the pairs). Missing file -> empty allowlist
    (nothing declared), not an error -- a fresh checkout with no corpora yet still runs
    cleanly."""
    if path is None:
        return set()
    path = Path(path)
    if not path.exists():
        return set()
    with open(path) as f:
        data = json.load(f)
    pairs = data.get("pairs", []) if isinstance(data, dict) else data
    return {frozenset(pair) for pair in pairs}


# --------------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------------
def check_root(root: Path, allowlist_path: Optional[Path] = DEFAULT_ALLOWLIST) -> Tuple[bool, str]:
    """Run the gate over every corpus under `root`. Returns `(ok, report_text)`;
    `ok=False` iff an undeclared self- or cross-corpus overlap was found."""
    root = Path(root)
    corpora = find_corpora(root)
    allow = load_allowlist(allowlist_path)

    lines = [f"scanned {len(corpora)} corpora under {root}"]
    ok = True
    per_corpus: Dict[str, Dict[str, Dict[str, List[str]]]] = {}

    for name, dataset_dir in corpora:
        with open(dataset_dir / "manifest.json") as f:
            manifest = json.load(f)
        keys = corpus_split_keys(dataset_dir, manifest)
        per_corpus[name] = keys
        counts = {s: len(keys[s]) for s in _SPLITS}
        lines.append(f"  {name}: distinct scene-hashes  "
                     f"train={counts['train']} val={counts['val']} test={counts['test']}")

    lines.append("")
    lines.append("-- (a) self-overlap: one corpus's own train/val/test hash sets --")
    self_fail = False
    for name, keys in per_corpus.items():
        sets = {s: set(keys[s]) for s in _SPLITS}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            inter = sets[a] & sets[b]
            if inter:
                self_fail = True
                ok = False
                lines.append(f"  FAIL {name}: {a} OVERLAP {b} shares {len(inter)} scene(s)")
    if not self_fail:
        lines.append("  (none)")

    lines.append("")
    lines.append("-- (b) cross-corpus leak: differently-named corpora, train vs val/test --")
    names = list(per_corpus)
    any_cross = False
    for i, a_name in enumerate(names):
        for b_name in names[i + 1:]:
            declared = frozenset((a_name, b_name)) in allow
            a_train = set(per_corpus[a_name]["train"])
            b_train = set(per_corpus[b_name]["train"])
            a_eval = set(per_corpus[a_name]["val"]) | set(per_corpus[a_name]["test"])
            b_eval = set(per_corpus[b_name]["val"]) | set(per_corpus[b_name]["test"])
            leak_b_into_a = a_train & b_eval   # b's val/test sits inside a's train
            leak_a_into_b = b_train & a_eval   # a's val/test sits inside b's train
            if leak_b_into_a or leak_a_into_b:
                any_cross = True
                tag = "ALLOWLISTED" if declared else "FAIL"
                if not declared:
                    ok = False
                if leak_b_into_a:
                    lines.append(f"  {tag} {b_name} val/test OVERLAP {a_name} train: "
                                 f"{len(leak_b_into_a)} scene(s)")
                if leak_a_into_b:
                    lines.append(f"  {tag} {a_name} val/test OVERLAP {b_name} train: "
                                 f"{len(leak_a_into_b)} scene(s)")
    if not any_cross:
        lines.append("  (none)")

    # ---- near-duplicate tier: WARN ONLY, never affects the exit status --------------
    # This catches the "same seed, different generator version" case, where scenes drift
    # by ~1.5 cm and are duplicates in every way that matters for leakage but are not
    # bit-identical. It is reported separately, and never fails, because coarse bins also
    # collide by coincidence -- see `frame_scene_key`'s docstring.
    lines.append("")
    lines.append("-- (c) NEAR-duplicates (rounded match, not bit-identical) -- WARNING ONLY --")
    any_near = False
    for a_name, b_name in itertools.combinations(sorted(per_corpus), 2):
        if frozenset((a_name, b_name)) in allow:
            continue
        a_keys, b_keys = per_corpus[a_name], per_corpus[b_name]
        a_tr = set(a_keys.get("_near_train", {}))
        b_tr = set(b_keys.get("_near_train", {}))
        a_ho = set(a_keys.get("_near_val", {})) | set(a_keys.get("_near_test", {}))
        b_ho = set(b_keys.get("_near_val", {})) | set(b_keys.get("_near_test", {}))
        # Only report what the EXACT tier did not already fail on, so the two tiers do
        # not double-count the same collision.
        a_ex = set(a_keys.get("val", {})) | set(a_keys.get("test", {}))
        b_ex = set(b_keys.get("val", {})) | set(b_keys.get("test", {}))
        extra_b = len(b_ho & a_tr) - len(b_ex & set(a_keys.get("train", {})))
        extra_a = len(a_ho & b_tr) - len(a_ex & set(b_keys.get("train", {})))
        if extra_b > 0:
            any_near = True
            lines.append(f"  WARN {b_name} val/test NEAR-matches {a_name} train: "
                         f"{extra_b} further scene(s) beyond the exact matches above")
        if extra_a > 0:
            any_near = True
            lines.append(f"  WARN {a_name} val/test NEAR-matches {b_name} train: "
                         f"{extra_a} further scene(s) beyond the exact matches above")
    if not any_near:
        lines.append("  (none)")

    lines.append("")
    lines.append(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return ok, "\n".join(lines)


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m e2e.ml.check_corpus_overlap",
        description="Fail if any e2e.ml radar corpus leaks scenes across its own "
                    "train/val/test split, or between two differently-named corpora's "
                    "train and val/test splits, unless the pair is on the allowlist.",
    )
    p.add_argument("--root", default=str(DEFAULT_ROOT),
                   help=f"corpus root to scan (default: {DEFAULT_ROOT})")
    p.add_argument("--allowlist", default=str(DEFAULT_ALLOWLIST),
                   help=f"JSON file of declared-intentional [[corpus_a, corpus_b], ...] "
                        f"pairs (default: {DEFAULT_ALLOWLIST}; missing file = no "
                        f"declared pairs)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    ok, report = check_root(Path(args.root), Path(args.allowlist))
    print(report)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
