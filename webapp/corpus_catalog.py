"""Which generated ML corpora exist on THIS machine, for the Corpus Replay source.

Torch-free and import-cheap by design (the registry imports it at app start, and the
app shell must come up without torch). Corpora are not tracked by git -- `e2e/ml/datasets/`
is generated locally and is ~10 GB per corpus -- so the list is discovered by scanning,
and an empty list on a clean clone is the expected, honest answer rather than an error.

The demo machine must have `benchmark_v1_D2` (the corpus every published detection
number was scored on); `DEFAULT_CORPUS` prefers it when present so a preset lands on the
right frames without the operator typing a path.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO_ROOT / "e2e" / "ml" / "datasets"

#: The corpus the detection results in notes/ are scored on. Relative to the repo root.
PREFERRED_CORPUS = "e2e/ml/datasets/b1_bench_v3/benchmark_v1_D2/manifest.json"


def discover_manifests(datasets_dir: Path = DATASETS_DIR) -> List[str]:
    """Repo-relative paths of every `manifest.json` under `datasets_dir`, sorted."""
    if not datasets_dir.is_dir():
        return []
    found = sorted(p for p in datasets_dir.glob("*/*/manifest.json") if p.is_file())
    return [p.relative_to(REPO_ROOT).as_posix() for p in found]


CORPUS_MANIFESTS: List[str] = discover_manifests()
DEFAULT_CORPUS: str = (PREFERRED_CORPUS if PREFERRED_CORPUS in CORPUS_MANIFESTS
                       else (CORPUS_MANIFESTS[0] if CORPUS_MANIFESTS else ""))
