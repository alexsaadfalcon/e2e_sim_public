"""Pre-flight check for the live demo: verify the desktop-over-RDP path never has to
fall back to the PDF deck.

    python -m webapp.preflight [--quick]

Six checks, in order, each printing PASS/FAIL/WARN per item:

  1. Assets      -- every preset's corpus manifest / ML checkpoint exists, munich_ka.pkl
                     and the legacy munich.pkl exist, the Tessera public CSV and
                     surrogate checkpoint exist, and the sionna_sims scan matches the
                     registry's Scenario choices.
  2. Environment  -- torch imports, CUDA is visible, dash/plotly import, webapp.app imports.
  3. Port         -- the app's host:port (read from webapp.app) is free.
  4. Warm-up      -- runs the cheapest preset once (1 frame) so THIS process pays the
                     ~10s torch cold start now, not while the audience watches.
  5. Presets      -- every preset validates (apply_preset succeeds, numeric overrides sit
                     on their input's step grid -- see tests/test_webapp_rehearsal.py).
  6. Timing       -- runs every preset once at its own frame count (both arms, for an
                     A/B preset) and prints wall seconds, WARN above 15s.

``--quick`` skips 4 and 6 (the two checks that actually run the pipeline and therefore
need torch + frames + real wall time -- everything else is data/import checks, seconds).

Exit code is non-zero iff any check reports FAIL (WARN does not fail the build: it is a
"the demo will be slower than expected" flag, not a "the demo will not work" one).

Every check function returns a list of ``CheckResult`` (or one, for check_port) and
accepts its heavy dependencies (the preset list, a repo root, a `run_pipeline` callable,
a port prober) as optional parameters, defaulting to the real ones -- so
tests/test_webapp_preflight.py can exercise every branch (missing file, busy port,
missing checkpoint, a slow run) without GPU or real frames.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class CheckResult:
    """One line of preflight output. ``status`` is one of PASS/FAIL/WARN/SKIP."""
    name: str
    status: str
    message: str
    details: List[str] = field(default_factory=list)


#: Above this many wall-seconds a single preset run (both arms, for an A/B preset)
#: WARNs in the timing-budget check (6) -- a number, not a hard failure, since a slow
#: run still works on stage, it just eats into the time between cards.
WARN_SECONDS = 15.0


# =====================================================================================
# 1. Assets
# =====================================================================================

def _preset_asset_paths(presets) -> Dict[str, List[str]]:
    """{repo-relative-or-absolute path: [preset ids that reference it]} for every corpus
    manifest and ML checkpoint any preset's overrides name. Generic over the override
    dict rather than importing DEFAULT_CORPUS/ML_CHECKPOINT/RADDETNET_CHECKPOINT/
    BRIDGE_CORPUS_* by name, so a new preset's path is covered automatically instead of
    needing this function edited too (the ledger's "one authority per question" rule --
    the presets ARE the list of what is referenced)."""
    paths: Dict[str, List[str]] = {}
    for p in presets:
        for bid, ov in p.overrides.items():
            params = ov.get("params") or {}
            if bid == "corpus_environment" and params.get("manifest"):
                paths.setdefault(params["manifest"], []).append(p.id)
            if bid == "detector" and params.get("mode") == "ml" and params.get("checkpoint"):
                paths.setdefault(params["checkpoint"], []).append(p.id)
    return paths


def check_assets(repo_root: Optional[Path] = None, presets=None,
                 sims_dir: Optional[Path] = None,
                 registry_choices: Optional[List[str]] = None) -> List[CheckResult]:
    """Every path a preset or the registry names actually exists on this machine.

    All four parameters default to the real ones (webapp.corpus_catalog /
    webapp.demo_presets / webapp.pipeline_registry); tests override them with a tmp_path
    tree so a missing file can be exercised without touching the real repo.
    """
    from webapp.corpus_catalog import REPO_ROOT as _REPO_ROOT
    from webapp.corpus_catalog import SIONNA_SIMS_DIR as _SIMS_DIR
    from webapp.corpus_catalog import discover_sionna_scenarios

    repo_root = repo_root or _REPO_ROOT
    sims_dir = sims_dir if sims_dir is not None else _SIMS_DIR
    if presets is None:
        from webapp.demo_presets import PRESETS as presets
    if registry_choices is None:
        from webapp.pipeline_registry import BLOCKS_BY_ID
        registry_choices = next(
            ps.choices for ps in BLOCKS_BY_ID["environment"].params if ps.key == "scenario_name"
        )

    results: List[CheckResult] = []
    missing: List[str] = []

    # munich_ka.pkl is what every preset actually loads by default (F93,
    # notes/ESTABLISHED_FACTS.md); munich.pkl is the legacy 3.5 GHz trace, still shipped
    # and selectable via the 'munich_legacy_3p5ghz' link -- both must be present or the
    # GUI offers a Scenario entry it cannot load.
    for name, fname in (("munich_ka_pkl", "munich_ka.pkl"), ("munich_legacy_pkl", "munich.pkl")):
        path = sims_dir / fname
        if path.is_file():
            results.append(CheckResult(f"assets.{name}", "PASS", f"{path} present"))
        else:
            results.append(CheckResult(f"assets.{name}", "FAIL", f"{path} MISSING"))
            missing.append(str(path))

    # The public Tessera interconnect artifacts (e2e/blocks.py TESSERA_INTERCONNECT_CSV,
    # and the surrogate checkpoint `python -m e2e.interconnect_surrogate.fetch` installs):
    # both are gitignored/derived so a clean clone can be missing either, and Thrust 4's
    # 'source: tessera' preset needs both to run live.
    tessera_csv = repo_root / "e2e" / "data" / "interconnect" / "tessera_tsv_s21_public.csv"
    if tessera_csv.is_file():
        results.append(CheckResult("assets.tessera_public_csv", "PASS", f"{tessera_csv} present"))
    else:
        results.append(CheckResult("assets.tessera_public_csv", "FAIL", f"{tessera_csv} MISSING"))
        missing.append(str(tessera_csv))

    checkpoint_dir = (repo_root / "e2e" / "interconnect_surrogate" / "_models"
                     / "tessera_checkout" / "models")
    if (checkpoint_dir / "best_model.pth").is_file() and (checkpoint_dir / "input_scaler.pt").is_file():
        results.append(CheckResult(
            "assets.tessera_checkpoint", "PASS", f"{checkpoint_dir} present"))
    else:
        results.append(CheckResult(
            "assets.tessera_checkpoint", "FAIL",
            f"{checkpoint_dir}/{{best_model.pth,input_scaler.pt}} MISSING -- run "
            f"`python -m e2e.interconnect_surrogate.fetch`"))
        missing.append(str(checkpoint_dir))

    scanned = discover_sionna_scenarios(sims_dir)
    if scanned == list(registry_choices):
        results.append(CheckResult(
            "assets.sionna_registry_match", "PASS",
            f"sionna_sims scan {scanned} matches the registry's Scenario choices"))
    else:
        results.append(CheckResult(
            "assets.sionna_registry_match", "FAIL",
            f"sionna_sims scan {scanned} != registry Scenario choices {list(registry_choices)} "
            "-- the GUI would offer a scenario it cannot load, or hide one it can"))

    for rel, preset_ids in sorted(_preset_asset_paths(presets).items()):
        path = Path(rel)
        if not path.is_absolute():
            path = repo_root / rel
        used_by = ", ".join(preset_ids)
        if path.is_file():
            results.append(CheckResult(f"assets.{rel}", "PASS", f"present (used by {used_by})"))
        else:
            results.append(CheckResult(f"assets.{rel}", "FAIL", f"MISSING (used by {used_by})"))
            missing.append(rel)

    if missing:
        results.append(CheckResult(
            "assets.summary", "FAIL",
            f"{len(missing)} missing asset(s): {', '.join(missing)}"))
    else:
        results.append(CheckResult("assets.summary", "PASS", "every preset's assets are present"))
    return results


# =====================================================================================
# 2. Environment
# =====================================================================================

def check_environment(torch_module=None) -> List[CheckResult]:
    """torch/CUDA/dash/plotly/webapp.app all import cleanly. ``torch_module`` lets a
    test inject a stub (e.g. one whose ``cuda.is_available()`` returns False) without
    needing an actual GPU-less machine."""
    results: List[CheckResult] = []

    torch = torch_module
    if torch is None:
        try:
            import torch
        except Exception as e:
            results.append(CheckResult("env.torch_import", "FAIL", f"import torch failed: {e}"))
            torch = None
    if torch is not None:
        if torch_module is None:
            results.append(CheckResult("env.torch_import", "PASS", f"torch {torch.__version__}"))
        cuda_query_failed = False
        try:
            count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception as e:
            results.append(CheckResult("env.cuda", "FAIL", f"torch.cuda query failed: {e}"))
            cuda_query_failed = True
            count = 0
        if count >= 1:
            name = torch.cuda.get_device_name(0)
            msg = f"{count} device(s) visible; device 0: {name}"
            try:
                free_b, _total_b = torch.cuda.mem_get_info(0)
                msg += f"; {free_b / 1e9:.1f} GB free"
            except Exception:
                pass  # mem_get_info is best-effort; device count/name are the load-bearing facts
            results.append(CheckResult("env.cuda", "PASS", msg))
        elif not cuda_query_failed:
            results.append(CheckResult(
                "env.cuda", "FAIL",
                "no CUDA device visible (device count 0) -- the demo needs the library's "
                "own device, never hardcode CPU, but there is nothing to run on here"))

    try:
        import dash
        import plotly
        results.append(CheckResult(
            "env.dash_plotly", "PASS", f"dash {dash.__version__}, plotly {plotly.__version__}"))
    except Exception as e:
        results.append(CheckResult("env.dash_plotly", "FAIL", f"import failed: {e}"))

    try:
        import webapp.app  # noqa: F401
        results.append(CheckResult("env.app_import", "PASS", "webapp.app imports without error"))
    except Exception as e:
        results.append(CheckResult("env.app_import", "FAIL", f"webapp.app import failed: {e}"))

    return results


# =====================================================================================
# 3. Port
# =====================================================================================

def _default_prober(host: str, port: int) -> bool:
    """True iff something is already listening on host:port."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


def check_port(host: Optional[str] = None, port: Optional[int] = None,
              prober: Optional[Callable[[str, int], bool]] = None) -> CheckResult:
    """The app's own host:port (read from webapp.app unless overridden) is free.

    ``prober`` lets a test simulate a busy port without actually binding a socket.
    """
    if host is None or port is None:
        from webapp.app import HOST as _HOST
        from webapp.app import PORT as _PORT
        host = host if host is not None else _HOST
        port = port if port is not None else _PORT
    prober = prober or _default_prober
    if prober(host, port):
        return CheckResult(
            "port", "FAIL",
            f"{host}:{port} is already in use -- find and stop whatever is bound there "
            f"(a previous `python -m webapp.app` that did not exit, most likely) before "
            f"launching the demo")
    return CheckResult("port", "PASS", f"{host}:{port} is free")


# =====================================================================================
# 5. Presets (defined before 4/6 -- warm-up and timing both need it)
# =====================================================================================

def _step_grid_violations(preset, blocks_by_id) -> List[str]:
    """Every numeric override that would trip a browser stepMismatch on blur (see
    tests/test_webapp_rehearsal.py::test_every_numeric_override_is_on_its_step_grid,
    which this mirrors) -- an off-grid value silently becomes None in the browser and
    the runner falls back to the registry default while the field keeps showing the
    preset's value."""
    problems = []
    for bid, ov in preset.overrides.items():
        for key, val in (ov.get("params") or {}).items():
            spec = next((ps for ps in blocks_by_id[bid].params if ps.key == key), None)
            if spec is None or spec.kind not in ("number", "int") or spec.step in (None, "any"):
                continue
            base = spec.min if spec.min is not None else 0.0
            steps = (float(val) - float(base)) / float(spec.step)
            if abs(steps - round(steps)) >= 1e-6:
                problems.append(
                    f"{bid}.{key}={val} is off the step={spec.step} grid (base {base})")
    return problems


def check_presets(presets=None) -> List[CheckResult]:
    """Every preset applies (arm A, and arm B when it defines one) and every numeric
    override sits on its input's step grid."""
    from webapp.demo_presets import PresetError, apply_preset
    from webapp.pipeline_registry import BLOCKS_BY_ID

    if presets is None:
        from webapp.demo_presets import PRESETS as presets

    results: List[CheckResult] = []
    n_ok = 0
    for p in presets:
        try:
            apply_preset(p)
            if p.ab is not None:
                apply_preset(p, arm="b")
        except PresetError as e:
            results.append(CheckResult(f"preset.{p.id}", "FAIL", f"apply_preset: {e}"))
            continue
        problems = _step_grid_violations(p, BLOCKS_BY_ID)
        if problems:
            results.append(CheckResult(f"preset.{p.id}", "FAIL", "; ".join(problems)))
            continue
        n_ok += 1
        results.append(CheckResult(f"preset.{p.id}", "PASS", f"thrust {p.thrust}, n_steps={p.n_steps}"))

    order = " -> ".join(f"{p.id} (T{p.thrust})" for p in presets)
    results.append(CheckResult(
        "presets.summary", "PASS" if n_ok == len(presets) else "FAIL",
        f"{n_ok}/{len(presets)} presets valid. Stage order: {order}"))
    return results


# =====================================================================================
# 4. Warm-up
# =====================================================================================

def check_warmup(presets=None, run_pipeline_fn=None) -> CheckResult:
    """Run the cheapest preset once, at 1 frame, so THIS process pays the torch cold
    start (~10s, measured 2026-09-22, notes/DEMO_DEFENSE.md REHEARSAL PROTOCOL) now.
    ``run_pipeline_fn`` lets a test inject a fast fake instead of the real pipeline."""
    from webapp.demo_presets import PresetError, apply_preset

    if presets is None:
        from webapp.demo_presets import PRESETS as presets
    if run_pipeline_fn is None:
        from webapp.pipeline_runner import run_pipeline as run_pipeline_fn

    cheapest = min(presets, key=lambda p: p.n_steps)
    try:
        state = apply_preset(cheapest)
    except PresetError as e:
        return CheckResult("warmup", "FAIL", f"could not apply preset {cheapest.id!r}: {e}")

    t0 = time.time()
    try:
        run_pipeline_fn(state, n_steps=1)
    except Exception as e:
        wall = time.time() - t0
        return CheckResult(
            "warmup", "FAIL",
            f"warm-up run of {cheapest.id!r} (1 frame) failed after {wall:.1f}s: {e}")
    wall = time.time() - t0
    return CheckResult(
        "warmup", "PASS",
        f"warm-up run of {cheapest.id!r} (1 frame) took {wall:.1f}s -- the cold start is "
        f"paid now, not in front of the audience")


# =====================================================================================
# 6. Timing budget
# =====================================================================================

def check_timing_budget(presets=None, run_pipeline_fn=None) -> List[CheckResult]:
    """Run every preset once, at its own configured frame count (both arms, for an A/B
    preset -- one click on stage runs both, see webapp/app.py `_run_pipeline`), and
    report wall seconds. WARN above WARN_SECONDS; never a FAIL on time alone."""
    from webapp.demo_presets import PresetError, apply_preset

    if presets is None:
        from webapp.demo_presets import PRESETS as presets
    if run_pipeline_fn is None:
        from webapp.pipeline_runner import run_pipeline as run_pipeline_fn

    results: List[CheckResult] = []
    for p in presets:
        try:
            state_a = apply_preset(p)
        except PresetError as e:
            results.append(CheckResult(f"timing.{p.id}", "FAIL", f"apply_preset failed: {e}"))
            continue
        t0 = time.time()
        try:
            run_pipeline_fn(state_a, n_steps=p.n_steps)
            n_runs = 1
            if p.ab is not None:
                state_b = apply_preset(p, arm="b")
                run_pipeline_fn(state_b, n_steps=p.n_steps)
                n_runs = 2
        except Exception as e:
            wall = time.time() - t0
            results.append(CheckResult(
                f"timing.{p.id}", "FAIL", f"run failed after {wall:.1f}s: {e}"))
            continue
        wall = time.time() - t0
        status = "WARN" if wall > WARN_SECONDS else "PASS"
        arms = "2 arms (A/B)" if n_runs == 2 else "1 run"
        results.append(CheckResult(
            f"timing.{p.id}", status,
            f"{wall:.1f}s -- {arms}, {p.n_steps} frame(s){' (over the 15s budget)' if status == 'WARN' else ''}"))
    return results


# =====================================================================================
# CLI
# =====================================================================================

def _print_result(r: CheckResult) -> None:
    print(f"[{r.status}] {r.name}: {r.message}")
    for d in r.details:
        print(f"    {d}")


def _print_section(title: str, results: List[CheckResult]) -> None:
    print(f"\n=== {title} ===")
    for r in results:
        _print_result(r)


def run_all(quick: bool) -> List[CheckResult]:
    """Run every check in order and print as it goes; return the flat result list."""
    results: List[CheckResult] = []

    section = check_assets()
    _print_section("1. Assets", section)
    results += section

    section = check_environment()
    _print_section("2. Environment", section)
    results += section

    r = check_port()
    _print_section("3. Port", [r])
    results.append(r)

    if quick:
        r = CheckResult("warmup", "SKIP", "skipped (--quick)")
        _print_section("4. Warm-up", [r])
    else:
        r = check_warmup()
        _print_section("4. Warm-up", [r])
    results.append(r)

    section = check_presets()
    _print_section("5. Presets", section)
    results += section

    if quick:
        r = CheckResult("timing", "SKIP", "skipped (--quick)")
        _print_section("6. Timing budget", [r])
        results.append(r)
    else:
        section = check_timing_budget()
        _print_section("6. Timing budget", section)
        results += section

    return results


def _runbook(presets) -> str:
    order = "\n".join(f"    {i + 1}. {p.label}" for i, p in enumerate(presets))
    return f"""
{'=' * 78}
RUNBOOK -- read this before walking on stage
{'=' * 78}
Launch:  python -m webapp.app
Open:    http://127.0.0.1:8050

Before the audience arrives: load and Run ONE throwaway preset (any one) so the
session's first run pays the torch cold start now (see check 4, Warm-up) --
not while the room is watching.

Presenting over RDP: step frames with the Results-tab frame SLIDER, not the
Play button -- Play's ~350ms/frame animation stutters over the link
(notes/DEMO_DEFENSE.md, "Presenting over RDP").

Preset stage order:
{order}

PDF fallback deck (LAST RESORT ONLY): e2e/main/figures/rehearsal/, generated by
`python -m webapp.rehearse`, then the deck script. This preflight check exists
so the deck is never needed -- if every check above is PASS, present live.
"""


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true",
                    help="skip check 4 (warm-up) and check 6 (timing budget)")
    args = ap.parse_args(argv)

    results = run_all(args.quick)
    n_fail = sum(1 for r in results if r.status == "FAIL")
    n_warn = sum(1 for r in results if r.status == "WARN")
    n_pass = len(results) - n_fail - n_warn

    print(f"\n{'=' * 78}")
    print(f"{len(results)} checks: {n_pass} pass, {n_warn} warn, {n_fail} fail")

    from webapp.demo_presets import PRESETS
    print(_runbook(PRESETS))

    if n_fail:
        print(f"PREFLIGHT FAILED: {n_fail} check(s) failed. Fix them before presenting live.")
    else:
        print("PREFLIGHT PASSED. Clear to present live.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
