"""A tiny on-disk cache for surrogate predictions.

Why this exists
---------------
One S21(f) curve over the pipeline's 1000-point band costs ~5 ms per frequency
point on CPU, i.e. **~5 s per curve** (measured 2026-09-23, this Windows box, torch
2.1.2+cu121 CPU inference, 3x3 ring arrangement). A GUI slider that recomputes on
every drag event is therefore unusable, but the same slider is instant if repeated
visits to a parameter tuple are served from disk -- and a demo's A/B pair can be
pre-computed once and shipped warm. That is the whole job here.

Key design
----------
The key is ``sha256`` over a canonical JSON blob of

* the **rounded** physical parameters (see ``ROUNDING``) -- float sliders emit noisy
  tails (60.000000000000004 um) that would otherwise miss on every drag;
* the **frequency grid**, hashed from its raw float64 bytes (not rounded: a grid is
  either the same array or a different question);
* the **arrangement** (the signal/ground grid), as a nested int list;
* a caller-supplied ``fingerprint`` string identifying the checkpoint + wrapper
  output convention, so a new checkpoint or a changed return shape cannot silently
  serve stale entries.

Values are ``.npz`` files holding one or more complex/float arrays. The cache is
**advisory**: any failure to read or write is swallowed (and counted) rather than
raised, because a broken cache must never break a prediction.

This module is deliberately torch-free -- see the package docstring.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

#: Decimal places used when rounding each parameter into the cache key. Geometry is
#: in micrometres, temperature in kelvin. 1e-3 um = 1 nm and 0.1 K are both far below
#: any resolution a GUI knob or the surrogate itself resolves, so this loses nothing
#: while making slider drags hit.
ROUNDING = {
    "radius_um": 3,
    "pitch_um": 3,
    "height_um": 3,
    "liner_um": 4,
    "temperature_k": 1,
}

_ENV_CACHE_DIR = "E2E_INTERCONNECT_CACHE_DIR"


def default_cache_dir():
    """Where cached predictions live when no explicit directory is given.

    ``$E2E_INTERCONNECT_CACHE_DIR`` wins if set (that is the hook a demo machine uses
    to point at a pre-computed, shipped cache); otherwise a per-user directory under
    the system temp dir. Deliberately NOT inside the repo: these are derived
    artifacts of a third-party checkpoint and must never be committed.
    """
    env = os.environ.get(_ENV_CACHE_DIR)
    if env:
        return Path(env)
    return Path(tempfile.gettempdir()) / "e2e_interconnect_surrogate_cache"


def _canonical_params(params):
    """Round the physical parameters the way :data:`ROUNDING` prescribes."""
    out = {}
    for name, value in sorted(params.items()):
        digits = ROUNDING.get(name)
        out[name] = round(float(value), digits) if digits is not None else value
    return out


def cache_key(params, freqs_hz, arrangement, fingerprint):
    """The hex digest identifying one prediction. Pure function, no I/O.

    Exposed (rather than kept private) so a pre-computation script can write entries
    for a demo without instantiating the surrogate, and so tests can assert that two
    float-noisy parameter dicts collapse to the same key.
    """
    freqs = np.ascontiguousarray(np.asarray(freqs_hz, dtype=np.float64))
    arr = np.asarray(arrangement, dtype=np.int8)
    blob = json.dumps(
        {
            "params": _canonical_params(params),
            "freq_sha": hashlib.sha256(freqs.tobytes()).hexdigest(),
            "n_freqs": int(freqs.size),
            "arrangement": arr.tolist(),
            "fingerprint": str(fingerprint),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class SurrogateCache:
    """An advisory ``.npz`` cache of surrogate outputs, keyed by :func:`cache_key`.

    ``hits`` / ``misses`` / ``errors`` are counters a caller (or a test) can read to
    confirm the cache is actually doing its job; ``enabled=False`` turns the whole
    thing into a pass-through without the caller branching.
    """

    def __init__(self, cache_dir=None, enabled=True):
        self.dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
        self.enabled = bool(enabled)
        self.hits = 0
        self.misses = 0
        self.errors = 0

    def _path(self, key):
        return self.dir / f"{key}.npz"

    def get(self, key):
        """Return the cached dict of arrays, or ``None`` on a miss/unreadable entry."""
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.exists():
            self.misses += 1
            return None
        try:
            with np.load(path, allow_pickle=False) as npz:
                value = {name: npz[name] for name in npz.files}
        except Exception:  # a truncated/corrupt entry is a miss, never an error path
            self.errors += 1
            self.misses += 1
            return None
        self.hits += 1
        return value

    def put(self, key, arrays):
        """Store ``arrays`` (a ``{name: ndarray}`` dict) under ``key``.

        Written to a temporary file in the same directory and then replaced
        atomically, so a killed process (or two workers racing on the same knob
        value) cannot leave a half-written entry that a later read would treat as
        real data.
        """
        if not self.enabled:
            return
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            # The suffix must be exactly ".npz": np.savez APPENDS ".npz" to any target
            # that does not already end in it, so a ".tmp" suffix would silently write
            # a different file and leave the empty mkstemp placeholder to be promoted.
            fd, tmp = tempfile.mkstemp(dir=str(self.dir), suffix=".npz")
            os.close(fd)
            try:
                np.savez(tmp, **arrays)
                os.replace(tmp, self._path(key))
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except Exception:
            self.errors += 1

    def clear(self):
        """Delete every entry (used by tests; safe if the directory is absent)."""
        if not self.dir.exists():
            return
        for path in self.dir.glob("*.npz"):
            try:
                path.unlink()
            except OSError:
                self.errors += 1

    def stats(self):
        return {"hits": self.hits, "misses": self.misses, "errors": self.errors,
                "dir": str(self.dir), "enabled": self.enabled}

    def __repr__(self):
        return f"SurrogateCache(dir={str(self.dir)!r}, enabled={self.enabled}, " \
               f"hits={self.hits}, misses={self.misses})"
