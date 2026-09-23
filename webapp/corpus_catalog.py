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


# ---- precomputed Sionna frame files ---------------------------------------------------
#: The names `e2e.blocks.SionnaEnvironmentBlock` accepts, in the order the GUI lists them.
SIONNA_SCENARIO_NAMES = ("munich", "etoile")
SIONNA_SIMS_DIR = REPO_ROOT / "e2e" / "environment" / "sionna_sims"


def discover_sionna_scenarios(sims_dir: Path = SIONNA_SIMS_DIR) -> List[str]:
    """The precomputed scenarios whose `.pkl` is actually on THIS machine.

    The frame files are not tracked (munich.pkl alone is 0.8 GB), so the GUI must not
    offer a name it cannot load: on 2026-09-23 the dropdown listed `etoile` with no
    `etoile.pkl` present and selecting it raised a raw FileNotFoundError mid-demo-rehearsal.
    An empty list on a clean clone is the honest answer.
    """
    return [n for n in SIONNA_SCENARIO_NAMES if (sims_dir / f"{n}.pkl").is_file()]


SIONNA_SCENARIOS: List[str] = discover_sionna_scenarios()
DEFAULT_SIONNA_SCENARIO: str = SIONNA_SCENARIOS[0] if SIONNA_SCENARIOS else "munich"
