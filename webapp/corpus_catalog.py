"""Which generated ML corpora exist on THIS machine, for the Corpus Replay source.

Torch-free and import-cheap by design (the registry imports it at app start, and the
app shell must come up without torch). Corpora are not tracked by git -- `e2e/ml/datasets/`
is generated locally and is ~10 GB per corpus -- so the list is discovered by scanning,
and an empty list on a clean clone is the expected, honest answer rather than an error.
One exception to "cheap": discovering the munich Ka-band label reads that pkl's `meta`
via one `pickle.load` (~0.5 s for the current ~0.8 GB file, measured 2026-09-23) to show
its real carrier rather than a hand-typed one that could go stale -- still torch-free,
and paid once at import.

The demo machine must have `benchmark_v1_D2` (the corpus every published detection
number was scored on); `DEFAULT_CORPUS` prefers it when present so a preset lands on the
right frames without the operator typing a path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Torch-free (os/pickle/numpy only, see that module) -- gives us the FILE-selector
# link tokens (MUNICH_LEGACY_LINK, MUNICH_LOSWEEP_LINK)
# and a cheap way to read a v2 pkl's `meta` without duplicating the pickle-format
# knowledge here.
from e2e.environment.sionna_iterator import (MUNICH_LEGACY_LINK,
                                            MUNICH_LOSWEEP_LINK, SionnaIterator)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO_ROOT / "e2e" / "ml" / "datasets"

#: The corpus the detection results in notes/ are scored on. Relative to the repo root.
PREFERRED_CORPUS = "e2e/ml/datasets/b1_bench_v3/benchmark_v1_D2/manifest.json"


def discover_manifests(datasets_dir: Path = DATASETS_DIR) -> List[str]:
    """Repo-relative paths of every `manifest.json` under `datasets_dir`, sorted.

    TWO depths, because the corpora on disk have two shapes. `chain_generate` normally
    writes `<corpus>/<config>/manifest.json`, but the Ka regeneration wrote
    `b1_demo_cfr_ka/benchmark_v1_ka_D2/benchmark_v1_ka_D2/manifest.json` -- the config
    directory doubled -- and a two-level glob simply did not see it, so the Ka demo
    corpus was invisible to the Corpus Replay source and to every preset validated
    against this list. Scanning both depths is the fix that does not require renaming a
    directory the Ka scoring JSONs already point at by path.
    """
    if not datasets_dir.is_dir():
        return []
    found = sorted(set(list(datasets_dir.glob("*/*/manifest.json"))
                       + list(datasets_dir.glob("*/*/*/manifest.json"))))
    return [p.relative_to(REPO_ROOT).as_posix() for p in found if p.is_file()]


CORPUS_MANIFESTS: List[str] = discover_manifests()
DEFAULT_CORPUS: str = (PREFERRED_CORPUS if PREFERRED_CORPUS in CORPUS_MANIFESTS
                       else (CORPUS_MANIFESTS[0] if CORPUS_MANIFESTS else ""))


# ---- precomputed Sionna frame files ---------------------------------------------------
#: The names `e2e.blocks.SionnaEnvironmentBlock` accepts, in the order the GUI lists them
#: (munich's own two files -- Ka-band and the legacy 3.5 GHz trace, F93 -- are both
#: surfaced under this one name; see `_discover_sionna_scenario_specs`).
SIONNA_SCENARIO_NAMES = ("munich", "etoile")
SIONNA_SIMS_DIR = REPO_ROOT / "e2e" / "environment" / "sionna_sims"

#: Static label for the legacy 3.5 GHz trace: a bare ndarray with no metadata to read
#: (F93, notes/ESTABLISHED_FACTS.md), unlike the Ka file this text is fixed rather than
#: derived from the pkl.
MUNICH_LEGACY_LABEL = "munich legacy (3.5 GHz trace, v1.0; no metadata)"


def _munich_ka_label(path: Path) -> str:
    """'munich (Ka-band, N GHz)', N read from the pkl's own `freq_plan.carrier_hz` so a
    future re-trace at a different carrier updates the GUI label automatically, rather
    than a hand-typed '30 GHz' going stale next to the file it no longer describes.
    Reading `meta` costs one `pickle.load` of the whole file (~0.5 s for the current
    ~0.8 GB munich_ka.pkl, measured 2026-09-23) -- paid once, at discovery time, same as
    every other entry here. Falls back to a metadata-free label if the read fails for
    any reason (corrupt file, unexpected format): discovery must never raise.
    """
    try:
        freq_plan = SionnaIterator(str(path)).freq_plan
        if freq_plan and freq_plan.get("carrier_hz") is not None:
            ghz = float(freq_plan["carrier_hz"]) / 1e9
            return f"munich (Ka-band, {ghz:g} GHz)"
    except Exception:
        pass
    return "munich (Ka-band)"


def _munich_losweep_label(path: Path) -> str:
    """'munich (Ka-band, 30 GHz, LoS sweep -28.4 to +28.9 deg)' -- the sweep read from
    the file's OWN per-frame receipts, never typed.

    `meta["boresight_offset_deg"]` is 0.0 on this file and would be a trap: it is the
    parsed `--boresight-offset-deg`, which the sweep overrides. What was actually flown
    is `meta["los_az_deg_first"/"los_az_deg_last"]` (or the per-frame `meta["frames"]`
    entries), recorded by the generator for exactly this reason
    (notes/LOSWEEP_REPORT_2026-09-25.md section 4, item 4).
    """
    try:
        it = SionnaIterator(str(path))
        meta = getattr(it, "meta", None) or {}
        ghz = None
        plan = it.freq_plan
        if plan and plan.get("carrier_hz") is not None:
            ghz = float(plan["carrier_hz"]) / 1e9
        band = f"Ka-band, {ghz:g} GHz" if ghz else "Ka-band"
        first, last = meta.get("los_az_deg_first"), meta.get("los_az_deg_last")
        if first is None or last is None:
            frames = meta.get("frames") or []
            az = [f.get("los_az_deg") for f in frames if f.get("los_az_deg") is not None]
            if az:
                first, last = az[0], az[-1]
        if first is not None and last is not None:
            # THE PER-FRAME RATE, not the file's total span (hostile round 13, N12).
            # "LoS sweep 57 deg" beside "8 frames" in the same line read as 57 degrees
            # of sweep in this run; the run flies 8 of the file's 30 frames, i.e. about
            # 15 deg. The rate is the quantity that is true of the file AND of any run
            # from it, and it is what the Thrust 3 card already quotes -- computed here
            # from the file's own endpoints and frame count, never typed.
            #
            # Kept to the run-identity line's 100-character budget whole (webapp/app.py
            # `_run_identity_line`): at "-28.4 to +28.9 deg" the ladder fell through to a
            # bare "munich" on the one screen whose subject is the sweep.
            span = abs(float(last) - float(first))
            n_frames = len(meta.get("frames") or []) or None
            band_full = f"Ka-band, {ghz:g} GHz" if ghz else "Ka-band"
            if n_frames and n_frames > 1:
                per_frame = span / (n_frames - 1)
                return (f"munich ({band_full}, LoS sweep "
                        f"~{per_frame:.1f} deg/frame)")
            return f"munich ({band_full}, LoS sweep {span:.0f} deg total)"
        return f"munich ({band}, LoS sweep)"
    except Exception:
        return "munich (Ka-band, LoS sweep)"


def _discover_sionna_scenario_specs(
    sims_dir: Path,
) -> Tuple[List[str], Dict[str, Tuple[str, Optional[str]]]]:
    """(labels, index): `labels` is what the Scenario dropdown lists (Ka-band munich
    first -- it is the default and what every preset uses -- then the legacy munich
    trace, then any other scenario); `index` maps each label back to the
    `(scenario_name, link)` pair `SionnaEnvironmentBlock`/`SionnaIterator` need to load
    it. Built together, from one scan, so the label the GUI shows and the token the
    runner resolves can never drift apart.
    """
    labels: List[str] = []
    index: Dict[str, Tuple[str, Optional[str]]] = {}
    ka_path = sims_dir / "munich_ka.pkl"
    legacy_path = sims_dir / "munich.pkl"
    if ka_path.is_file():
        label = _munich_ka_label(ka_path)
        labels.append(label)
        index[label] = ("munich", None)
    losweep_path = sims_dir / "munich_ka_losweep.pkl"
    if losweep_path.is_file():
        # The swept-line-of-sight Ka trace (Thrust 3). Listed right after the static Ka
        # file and BEFORE the legacy one, because it is a current-band scenario a preset
        # actually selects; the legacy 3.5 GHz trace is a comparison artifact.
        label = _munich_losweep_label(losweep_path)
        labels.append(label)
        index[label] = ("munich", MUNICH_LOSWEEP_LINK)
    if legacy_path.is_file():
        labels.append(MUNICH_LEGACY_LABEL)
        index[MUNICH_LEGACY_LABEL] = ("munich", MUNICH_LEGACY_LINK)
    for name in SIONNA_SCENARIO_NAMES:
        if name == "munich":
            continue  # handled above -- munich is the only name with a legacy split
        if (sims_dir / f"{name}.pkl").is_file():
            labels.append(name)
            index[name] = (name, None)
    return labels, index


def sionna_label_for_link(link: str,
                          sims_dir: Path = SIONNA_SIMS_DIR):
    """The dropdown LABEL that resolves to `("munich", link)`, or None when this
    machine has no file for it.

    A preset cannot hardcode one of these labels: they are built from the pkl's own
    metadata (carrier, sweep endpoints), so the string changes when the file does. It
    also cannot hardcode the raw link token -- the param is a `choice` and the value
    has to be one the dropdown offers. So a preset asks for the FILE, by its stable
    link token, and gets back whatever label this machine's scan produced for it.
    """
    _labels, index = _discover_sionna_scenario_specs(sims_dir)
    for label, (name, lnk) in index.items():
        if lnk == link:
            return label
    return None


def discover_sionna_scenarios(sims_dir: Path = SIONNA_SIMS_DIR) -> List[str]:
    """The precomputed scenarios whose `.pkl` is actually on THIS machine, as the
    labels the Scenario dropdown shows (see `_discover_sionna_scenario_specs`).

    The frame files are not tracked (munich_ka.pkl alone is 0.8 GB), so the GUI must
    not offer a name it cannot load: on 2026-09-23 the dropdown listed `etoile` with no
    `etoile.pkl` present and selecting it raised a raw FileNotFoundError mid-demo-rehearsal.
    An empty list on a clean clone is the honest answer.
    """
    labels, _index = _discover_sionna_scenario_specs(sims_dir)
    return labels


def resolve_sionna_scenario(token: str) -> Tuple[str, Optional[str]]:
    """`(scenario_name, link)` to build `SionnaEnvironmentBlock` with for a `token`
    from `discover_sionna_scenarios` (i.e. the Scenario dropdown's stored value).

    Falls back to `(token, None)` for a token the current scan does not recognize --
    e.g. a scenario name typed directly, or a UI state saved before this split -- so an
    unrecognized value still reaches `SionnaEnvironmentBlock` unchanged rather than
    being rejected here.
    """
    return _SIONNA_SCENARIO_INDEX.get(token, (token, None))


SIONNA_SCENARIOS: List[str]
_SIONNA_SCENARIO_INDEX: Dict[str, Tuple[str, Optional[str]]]
SIONNA_SCENARIOS, _SIONNA_SCENARIO_INDEX = _discover_sionna_scenario_specs(SIONNA_SIMS_DIR)
DEFAULT_SIONNA_SCENARIO: str = SIONNA_SCENARIOS[0] if SIONNA_SCENARIOS else "munich"
