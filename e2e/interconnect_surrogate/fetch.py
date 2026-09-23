"""Fetch the Tessera TSV surrogate checkpoint that pip does not install.

``pip install -r requirements-tessera.txt`` ships the ``tessera`` Python package but
not ``models/best_model.pth`` / ``models/input_scaler.pt`` (see that file's "THE
CHECKPOINT IS NOT INSTALLED BY PIP" section) -- upstream's ``pyproject.toml`` declares
no package data. Until now the only way to supply them was to set
``$TESSERA_REPO``/``$E2E_TESSERA_MODELS_DIR`` by hand, which a demo box is not
guaranteed to have done.

Run as::

    python -m e2e.interconnect_surrogate.fetch

This shallow-clones the PINNED commit (parsed from ``requirements-tessera.txt``, so it
cannot silently drift out of sync with that pin) into a gitignored local checkout,
:data:`DEFAULT_CHECKOUT_DIR`, which :func:`e2e.interconnect_surrogate.tessera.checkpoint_dir`
already searches -- no environment variable required afterwards. Safe to re-run: a
checkout already at the pinned commit with both weight files present is left alone.

Depth-1 fetch of an exact commit (not a branch/tag) relies on the server allowing a
direct SHA want (``uploadpack.allowReachableSHA1InWant``); GitHub enables this for
public repositories, which is what let a fresh clone here (2026-09-23) pin the exact
commit ``requirements-tessera.txt`` names instead of "whatever main currently is".
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
_REQUIREMENTS_FILE = _REPO_ROOT / "requirements-tessera.txt"
_REMOTE_URL = "https://github.com/HiPerCAS/tessera.git"

#: Where `fetch()` puts the checkout by default. `checkpoint_dir()` looks in
#: `DEFAULT_CHECKOUT_DIR / "models"` as its last-resort candidate (see .gitignore --
#: this directory is derived, third-party content and must never be committed).
DEFAULT_CHECKOUT_DIR = _HERE / "_models" / "tessera_checkout"


def pinned_commit():
    """The commit ``requirements-tessera.txt`` pins.

    Parsed rather than duplicated as a constant here, so the two cannot drift apart
    (the "drifting values" failure mode: a hardcoded copy of a value that lives
    elsewhere silently goes stale the next time that file is repinned).
    """
    text = _REQUIREMENTS_FILE.read_text()
    m = re.search(r"tessera(?:-tsv)?\s*@\s*git\+[^\s@]+@([0-9a-fA-F]+)", text)
    if not m:
        raise RuntimeError(
            f"could not find a pinned 'tessera ... @git+...@<commit>' line in "
            f"{_REQUIREMENTS_FILE}"
        )
    return m.group(1)


def _current_commit(checkout_dir):
    try:
        out = subprocess.run(
            ["git", "-C", str(checkout_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def _has_checkpoint(models_dir):
    return (models_dir / "best_model.pth").is_file() and (models_dir / "input_scaler.pt").is_file()


def fetch(checkout_dir=None, commit=None, remote_url=_REMOTE_URL):
    """Shallow-clone `commit` into `checkout_dir`; returns the `models/` Path within it.

    Idempotent: if `checkout_dir` is already a checkout of `commit` with both weight
    files present, this does nothing but return that path.
    """
    checkout_dir = Path(checkout_dir) if checkout_dir is not None else DEFAULT_CHECKOUT_DIR
    commit = commit or pinned_commit()
    models_dir = checkout_dir / "models"

    current = _current_commit(checkout_dir)
    if current is not None and current.startswith(commit) and _has_checkpoint(models_dir):
        print(f"already at pinned commit {commit} ({checkout_dir})")
        return models_dir

    if checkout_dir.exists():
        shutil.rmtree(checkout_dir)
    checkout_dir.mkdir(parents=True, exist_ok=True)

    def run(*args):
        subprocess.run(args, cwd=str(checkout_dir), check=True)

    run("git", "init", "-q", ".")
    run("git", "remote", "add", "origin", remote_url)
    run("git", "fetch", "--depth", "1", "origin", commit)
    run("git", "checkout", "-q", "FETCH_HEAD")

    if not _has_checkpoint(models_dir):
        raise RuntimeError(
            f"cloned {remote_url}@{commit} into {checkout_dir} but models/ is missing "
            f"best_model.pth/input_scaler.pt -- did the upstream layout change?"
        )
    print(f"fetched {remote_url}@{commit} -> {models_dir}")
    return models_dir


def main(argv=None):
    try:
        fetch()
    except Exception as exc:
        print(f"fetch failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
