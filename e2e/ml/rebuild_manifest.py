"""Rebuild `manifest.json` for a corpus whose generation run died before writing one.

Why this exists
---------------
`e2e.ml.chain_generate.generate_chain_corpus` writes the manifest in a tail step, AFTER
the last scene. A run that is interrupted -- killed, crashed, or hung and killed -- leaves
a directory of perfectly valid `.npz` frames that nothing can load, because every consumer
(`RadarFrameDataset`, `compare_detectors`, `export_ssm`) enters through the manifest.
Hours of GPU time are recoverable and were previously being thrown away.

This is a RECOVERY tool, not a second producer. It calls `e2e.ml.dataset.write_manifest`
-- the same function the generator calls -- so the output is byte-compatible with a
normally-produced manifest rather than a lookalike with its own schema drift.

What it must get right, and what it refuses to guess
-----------------------------------------------------
* **The scene-identity salt.** `chain_generate` salts scene sampling with
  `f"{parent.name}/{leaf.name}"` (`_corpus_identity_tag`), while `write_manifest`'s own
  default is the LEAF name only. Recording the leaf would produce a manifest whose salt
  never drew these scenes: anything that reconstructs them from it -- `export_ssm`'s
  ground-truth verification -- would silently rebuild a DIFFERENT corpus. This module
  always writes the two-component tag.
* **`generator_git_commit` is recorded as HEAD RIGHT NOW**, because git history cannot be
  consulted retroactively for "what was checked out during the run". If HEAD has moved
  since, the recorded commit is a lie about provenance, and `--expect-commit` exists so
  the caller can assert what it should be and get a hard failure instead of a plausible
  wrong answer.
* **Partial and non-contiguous scenes are refused, not patched.** A torn final scene
  (some of its frames) would let the scene-level split straddle a partial sequence; a hole
  in the scene indices means an index no longer identifies the scene the salt drew.

CLI
---
    python -m e2e.ml.rebuild_manifest --corpus <dir> --config benchmark_v1 --tier D2 \\
        --seed 20260829 [--expect-commit 3b00f9d] [--write]

Dry by default: it prints what it would write and exits without touching the directory.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from e2e.ml.dataset import _input_scale, write_manifest
from e2e.ml.labels import LabelGrid
from e2e.radar_config import PRESETS

#: `chain_generate` names frames `<tag>_scene<N>_frame_<T>.npz`; the tag itself may
#: contain underscores, so scene/frame are anchored from the right.
_FNAME = re.compile(r"^(?P<tag>.+)_scene(?P<scene>\d+)_frame_(?P<frame>\d+)\.npz$")


def group_frames(corpus_dir: Path) -> List[List[str]]:
    """`[[frame filenames...], ...]`, one inner list per scene in frame order.

    Raises rather than repairing on a torn or non-contiguous corpus -- see the module
    docstring for why each of those is unsafe to paper over.
    """
    names = sorted(p.name for p in corpus_dir.glob("*.npz"))
    if not names:
        raise ValueError(f"no .npz frames under {corpus_dir}")

    by_scene: Dict[int, List] = {}
    for name in names:
        m = _FNAME.match(name)
        if not m:
            raise ValueError(
                f"{name!r} does not match '<tag>_scene<N>_frame_<T>.npz'; this tool only "
                "rebuilds manifests for chain_generate corpora")
        by_scene.setdefault(int(m.group("scene")), []).append((int(m.group("frame")), name))

    scenes = sorted(by_scene)
    if scenes != list(range(len(scenes))):
        missing = sorted(set(range(scenes[-1] + 1)) - set(by_scene))
        raise ValueError(
            f"scene indices are not contiguous 0..{len(scenes) - 1}; missing {missing[:10]}"
            f"{' ...' if len(missing) > 10 else ''}. A hole means a scene index no longer "
            "identifies the scene the corpus salt drew -- delete the trailing frames and "
            "rebuild over a contiguous prefix instead.")

    counts = {len(v) for v in by_scene.values()}
    if len(counts) != 1:
        expected = max(counts)
        torn = [s for s in scenes if len(by_scene[s]) != expected]
        raise ValueError(
            f"scenes {torn[:5]} have a partial frame set (expected {expected} frames each)"
            " -- an interrupted run can tear its last scene. Delete those frames and "
            "re-run; a partial sequence would let the scene-level split straddle it.")

    return [[name for _f, name in sorted(by_scene[s])] for s in scenes]


def rebuild(corpus_dir, cfg_name: str, tier: str, seed: int, *,
            range_stride: int = 4, n_azimuth: int = 192,
            label_classes=("vehicle", "pedestrian"),
            expect_commit: Optional[str] = None, write: bool = False) -> Dict:
    corpus_dir = Path(corpus_dir)
    if (corpus_dir / "manifest.json").exists():
        raise FileExistsError(
            f"{corpus_dir / 'manifest.json'} already exists -- refusing to overwrite a "
            "manifest. This tool recovers a MISSING one; delete it first if you really "
            "mean to replace it.")

    cfg = PRESETS[cfg_name]
    sequences = group_frames(corpus_dir)
    grid = LabelGrid.for_config(cfg, range_stride=range_stride, n_azimuth=n_azimuth)
    # See the module docstring: the two-component tag is the salt the generator used.
    corpus_tag = f"{corpus_dir.parent.name}/{corpus_dir.name}"

    summary = {
        "corpus": str(corpus_dir),
        "scenes": len(sequences),
        "frames_per_scene": len(sequences[0]),
        "total_frames": sum(len(s) for s in sequences),
        "corpus_tag": corpus_tag,
        "input_scale": _input_scale(cfg),
    }
    if not write:
        summary["written"] = False
        return summary

    path = write_manifest(corpus_dir, cfg, tier, sequences, grid=grid, seed=seed,
                          snr_db=None, frames_per_scene=len(sequences[0]),
                          label_classes=label_classes, corpus_tag=corpus_tag,
                          input_scale=_input_scale(cfg))
    manifest = json.loads(Path(path).read_text())
    recorded = str(manifest.get("generator_git_commit", ""))
    if expect_commit and not recorded.startswith(expect_commit):
        raise ValueError(
            f"recorded generator_git_commit {recorded!r} does not match the expected "
            f"{expect_commit!r}. HEAD moved since the run, so this manifest would "
            f"misattribute the frames. The manifest was written -- delete it, check out "
            f"the generating commit, and re-run.")
    summary.update(written=True, manifest=str(path), generator_git_commit=recorded,
                   splits={k: len(v) for k, v in manifest["files"].items()})
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m e2e.ml.rebuild_manifest",
        description="Rebuild manifest.json for a chain_generate corpus whose run died "
                    "before writing one. Dry by default.")
    p.add_argument("--corpus", required=True, help="the <out>/<config>_<tier> directory")
    p.add_argument("--config", required=True, help=f"one of {sorted(PRESETS)}")
    p.add_argument("--tier", required=True)
    p.add_argument("--seed", type=int, required=True,
                   help="the base seed the run was launched with -- it is not recoverable "
                        "from the frames, and a wrong value silently mislabels the corpus")
    p.add_argument("--range-stride", type=int, default=4)
    p.add_argument("--n-azimuth", type=int, default=192)
    p.add_argument("--expect-commit", default=None,
                   help="assert the recorded generator commit starts with this; use it "
                        "whenever you know which commit generated the frames")
    p.add_argument("--write", action="store_true", help="actually write manifest.json")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = rebuild(args.corpus, args.config, args.tier, args.seed,
                      range_stride=args.range_stride, n_azimuth=args.n_azimuth,
                      expect_commit=args.expect_commit, write=args.write)
    for k, v in summary.items():
        print(f"{k:22s}: {v}")
    if not summary.get("written"):
        print("\ndry run -- pass --write to create manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
