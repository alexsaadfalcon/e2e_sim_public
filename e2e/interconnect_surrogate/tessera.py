"""Thin wrapper around the public Tessera / TSV_PhGNN S-parameter surrogate.

Upstream
--------
    Tessera (package ``tessera-tsv``), github.com/HiPerCAS/tessera
    Copyright (c) 2026, HiPerCAS Lab -- released under the BSD 3-Clause License.
    The full licence text and the notice its terms require us to carry are in
    ``NOTICE`` beside this file.

    Cite the work, not this wrapper:

        M. Gharib, L. Popryho, and I. Partin-Vaisband, "From Physics to Surrogate
        Intelligence: A Unified Electro-Thermo-Optimization Framework for TSV
        Networks," *IEEE Transactions on Computer-Aided Design of Integrated
        Circuits and Systems*, 2026, doi: 10.1109/TCAD.2026.3718807.

    BSD clause 3 forbids using HiPerCAS's or the authors' names to endorse or promote
    this simulator. We may state that we *use* Tessera and cite the paper; we may not
    present UIC/HiPerCAS as endorsing anything here.

What it gives us
----------------
A design spec (via radius / pitch / height / liner-oxide thickness, temperature, and a
signal/ground arrangement grid) maps to a full complex S-matrix at one frequency in
milliseconds. We use it two ways:

* :meth:`TesseraTSV.s21` -- one complex S21(f) curve on an arbitrary frequency grid,
  shaped exactly like what ``InterconnectBlock(transfer_csv=...)`` interpolates out of
  a CSV, so it can drive the same multiply along the frame's frequency axis.
* :meth:`TesseraTSV.crosstalk_db` -- worst-case NEXT/FEXT between signal vias, which
  the shipped CSVs never carried at all.

What it is NOT
--------------
It models a **through-silicon via array**, not the automotive interconnect behind
``e2e/data/interconnect/tessera_case{1..6}_s21_77ghz.csv``; those six are a direct
HFSS export of a different structure and are not producible by this package.

Provenance of the numbers below (measured 2026-09-23, public checkpoint
``models/best_model.pth`` at repo commit ``e53bb88``, CPU, torch 2.1.2+cu121)
--------------------------------------------------------------------------------
* The public checkpoint does **not** reproduce the shipped ``tessera_tsv_s21.csv``.
  At the shipped geometry it predicts about -0.57 dB at 30 GHz where that file says
  -7.48 dB. The CSV came from an HFSS-finetuned checkpoint that is not in the public
  release, so this wrapper is a *different instrument*, not a regeneration path. Do
  not use it to "refresh" that CSV without an explicit owner ruling.
* **Extrapolation is unguarded.** At height 800 um / 30 GHz the model returns
  S21 = +17.1 dB -- gain from a passive structure. Upstream's own electrothermal path
  SVD-clips the S-matrix for exactly this reason. Hence :class:`PassivityError` and
  the ``passivity=`` policy below: this wrapper refuses to hand a physically
  impossible response to the pipeline by default.
* ``pip install git+https://github.com/HiPerCAS/tessera.git`` installs the *package*
  but **not** ``models/`` or ``config.yaml`` (``pyproject.toml`` declares
  ``packages = ["tessera"]`` with no package data). Upstream ``tessera.load_model()``
  therefore raises ``FileNotFoundError`` on ``config.yaml`` after a plain pip install.
  We resolve the checkpoint ourselves (:func:`checkpoint_dir`) and build the model
  from the public ``TSVPhysicsGNN`` / ``InputScaler`` classes, which needs no
  ``config.yaml``; prediction still goes through upstream's public
  ``predict_s_matrix``.

Naming footgun: this module is ``e2e.interconnect_surrogate.tessera`` and the upstream
package is ``tessera``. Python 3 absolute imports keep them apart, but never run this
file as a script (``python e2e/interconnect_surrogate/tessera.py``) -- that would put
this directory on ``sys.path`` and shadow upstream.
"""

import importlib.util
import os
import warnings
from pathlib import Path

import numpy as np

from e2e.interconnect_surrogate.cache import SurrogateCache, cache_key

# ---------------------------------------------------------------------------
# Valid parameter ranges (what the GUI should bound its knobs to)
# ---------------------------------------------------------------------------
# Upstream's README states no training ranges. These are RECOVERED from the shipped
# input scaler (``models/input_scaler.pt``), which stores the per-feature mean and
# standard deviation of the training set. Assuming each geometric parameter was
# sampled uniformly over an interval, the interval is ``mean +- std*sqrt(3)``; five of
# the six features land on round engineering numbers under that assumption, which is
# the evidence that it holds:
#
#   feature      mean        std         -> implied uniform interval
#   radius     4.014 um    1.161 um         2.00 - 6.02 um     -> 2-6 um
#   pitch     30.008 um    5.820 um        19.93 - 40.09 um     -> 20-40 um
#   height    79.972 um   11.596 um        59.89 - 100.06 um    -> 60-100 um
#   liner      2.010 um    0.578 um         1.01 - 3.01 um      -> 1-3 um
#   temp      448.88 K    86.558 K        298.96 - 598.80 K     -> 300-600 K
#   freq       34.44 GHz  31.944 GHz      -20.9 - 89.8 GHz      -> NOT uniform
#
# Frequency alone does not fit (the implied lower bound is negative), so it was
# sampled some other way; the band below is the positive part of that interval, and is
# consistent with upstream running their own examples at 15 GHz and 100 GHz. Our
# pipeline band, 28.5-31.5 GHz, sits just under the training mean either way.
#
# Measured 2026-09-23 from checkpoint e53bb88; re-derive with
#   torch.load('models/input_scaler.pt', weights_only=True)['node_mean' / 'node_std']
# if the checkpoint ever changes.
VALID_RANGES = {
    "radius_um": (2.0, 6.0),
    "pitch_um": (20.0, 40.0),
    "height_um": (60.0, 100.0),
    "liner_um": (1.0, 3.0),
    "temperature_k": (300.0, 600.0),
    "freq_hz": (1.0e9, 89.8e9),
}

#: The geometry our shipped ``tessera_tsv_s21.csv`` documents (e2e/data/interconnect/
#: README.md, "TSV geometry"). NOTE: ``pitch_um`` and ``liner_um`` sit OUTSIDE
#: :data:`VALID_RANGES` -- so does upstream's own README quickstart and the
#: ``optimization.fixed_params`` block of their ``config.yaml``, which use the same
#: numbers. Treat predictions at this point as the authors' own extrapolation; the
#: passivity guard is what stops that becoming silent nonsense.
SHIPPED_TSV_DESIGN = {
    "radius_um": 5.0,
    "pitch_um": 60.0,
    "height_um": 100.0,
    "liner_um": 0.5,
    "temperature_k": 300.0,
}


def ring_arrangement(size=3):
    """A single centre **signal** via inside a solid **ground** ring, ``size x size``.

    This is the structure ``e2e/data/interconnect/tessera_tsv_s21.csv`` documents
    ("one center signal via surrounded by a ground ring"), so it is the default here.
    It has exactly one signal via, hence a 2x2 S-matrix and *no* crosstalk terms --
    use a multi-signal arrangement (:data:`ARRANGEMENTS`) for NEXT/FEXT.
    """
    if size < 3 or size % 2 == 0:
        raise ValueError(f"ring size must be odd and >= 3, got {size}")
    arr = -np.ones((size, size), dtype=np.int8)
    arr[size // 2, size // 2] = 1
    return arr


def _checkerboard(size):
    """``size x size`` alternating signal/ground -- upstream's quickstart pattern."""
    i, j = np.indices((size, size))
    return np.where((i + j) % 2 == 0, 1, -1).astype(np.int8)


#: Named arrangements a GUI dropdown can offer without the caller building int arrays.
ARRANGEMENTS = {
    "ring3x3": ring_arrangement(3),      # 1 signal  -> 2x2 S, no crosstalk
    "ring5x5": ring_arrangement(5),      # 1 signal  -> 2x2 S, no crosstalk
    "checker3x3": _checkerboard(3),      # 5 signals -> 10x10 S, crosstalk available
    "checker5x5": _checkerboard(5),      # 13 signals
}

_DEFAULT_ARRANGEMENT = "ring3x3"

_ENV_MODELS_DIR = "E2E_TESSERA_MODELS_DIR"
_ENV_REPO = "TESSERA_REPO"


class PassivityError(RuntimeError):
    """Raised when the surrogate returns |S21| > 0 dB, i.e. gain from a passive TSV.

    This is not a numerical nicety. Upstream does not clamp its raw output, and the
    failure mode is loud and physical: at height 800 um / 30 GHz the public checkpoint
    returns +17.1 dB (measured 2026-09-23). Feeding that into ``InterconnectBlock``
    would amplify the frame and quietly corrupt every downstream product. Almost every
    observed violation is an out-of-range geometry, so the message names the design.
    """


# ---------------------------------------------------------------------------
# Availability (must stay torch-free: a webapp shell calls this to grey out a knob)
# ---------------------------------------------------------------------------

def checkpoint_dir(models_dir=None):
    """Locate the directory holding ``best_model.pth`` + ``input_scaler.pt``.

    Returns a :class:`~pathlib.Path`, or ``None`` if no candidate has both files.
    Never imports ``tessera`` (it reads the spec's origin instead), so it is safe to
    call from torch-free code. Search order, first hit wins:

    1. the explicit ``models_dir`` argument;
    2. ``$E2E_TESSERA_MODELS_DIR``;
    3. ``$TESSERA_REPO/models`` -- a checkout of the upstream repo;
    4. ``<parent of the installed tessera package>/models`` -- the editable-install /
       ``pip install -e .`` case, where the repo root really is on the path.

    Case 4 does **not** fire for ``pip install git+...``: that wheel carries the
    Python package only, no weights (see the module docstring). Something must point
    at a checkout, which is why (2) and (3) exist.
    """
    candidates = []
    if models_dir is not None:
        candidates.append(Path(models_dir))
    env_models = os.environ.get(_ENV_MODELS_DIR)
    if env_models:
        candidates.append(Path(env_models))
    env_repo = os.environ.get(_ENV_REPO)
    if env_repo:
        candidates.append(Path(env_repo) / "models")
    try:
        spec = importlib.util.find_spec("tessera")
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.origin:
        candidates.append(Path(spec.origin).resolve().parents[1] / "models")

    for cand in candidates:
        try:
            if (cand / "best_model.pth").is_file() and (cand / "input_scaler.pt").is_file():
                return cand
        except OSError:
            continue
    return None


def available(models_dir=None):
    """``True`` when the surrogate can actually run here. Never raises.

    Checks three things without importing anything heavy: the ``tessera`` package is
    importable, ``torch_geometric`` (its one non-obvious dependency) is importable,
    and a checkpoint directory exists. All three matter -- a pip install from git
    satisfies the first two and fails the third.
    """
    try:
        for name in ("tessera", "torch_geometric", "torch"):
            if importlib.util.find_spec(name) is None:
                return False
        return checkpoint_dir(models_dir) is not None
    except Exception:
        # find_spec can raise on a broken/partially-removed distribution. "Is it
        # available?" must answer False in that case, not explode in a UI callback.
        return False


def _missing_reason(models_dir=None):
    """A human-readable explanation of why :func:`available` said False."""
    for name, hint in (
        ("torch", "pip install torch"),
        ("torch_geometric", "pip install torch-geometric"),
        ("tessera", "pip install -r requirements-tessera.txt"),
    ):
        try:
            found = importlib.util.find_spec(name) is not None
        except Exception:
            found = False
        if not found:
            return f"the {name!r} package is not installed ({hint})"
    return (
        "the Tessera checkpoint was not found. `pip install git+...` ships the python "
        f"package but NOT models/best_model.pth; point ${_ENV_MODELS_DIR} at a models/ "
        f"directory, or ${_ENV_REPO} at a checkout of github.com/HiPerCAS/tessera"
    )


# ---------------------------------------------------------------------------
# The wrapper
# ---------------------------------------------------------------------------

class TesseraTSV:
    """Live TSV S-parameters from the public Tessera surrogate.

    The model and scaler are loaded on the **first prediction**, not in ``__init__``,
    and then held for the object's lifetime -- so constructing one of these in a
    module-level registry costs nothing and imports nothing.

    Parameters
    ----------
    models_dir : path, optional
        Overrides the checkpoint search in :func:`checkpoint_dir`.
    device : str
        ``"cpu"`` (default) is the right answer: upstream states CPU is sufficient and
        a GPU only helps large batched searches, and our calls are one small graph at
        a time.
    cache : SurrogateCache or None
        Where to memoise predictions. ``None`` builds a default on-disk cache; pass
        ``SurrogateCache(enabled=False)`` to disable.
    passivity : {"raise", "clamp", "ignore"}
        What to do when |S21| > 0 dB for any frequency.

        * ``"raise"`` (default) -- raise :class:`PassivityError`. Chosen as the
          default because the alternative is handing the pipeline a fabricated
          response that *looks* like a physical one; a GUI can catch this and show
          "outside the validated range" next to the knob, which is the truth.
        * ``"clamp"`` -- scale the whole curve so its peak is exactly 0 dB and emit a
          ``RuntimeWarning`` naming the design and the excess. Offered for sweeps that
          want to keep going past the valid box; the result is NOT the model's output.
        * ``"ignore"`` -- return raw output untouched. For diagnosing the surrogate
          itself, never for feeding the pipeline.
    warn_out_of_range : bool
        Emit a ``UserWarning`` when a parameter leaves :data:`VALID_RANGES`. Default
        ``True``. This is a warning and not an error on purpose: our own shipped TSV
        geometry (pitch 60 um, liner 0.5 um) is outside the training box, as is
        upstream's own README quickstart.
    """

    #: Bumped when the wrapper's output convention changes, so old cache entries
    #: (which encode this in their key) are ignored rather than silently reused.
    OUTPUT_VERSION = 1

    def __init__(self, models_dir=None, device="cpu", cache=None,
                 passivity="raise", warn_out_of_range=True):
        if passivity not in ("raise", "clamp", "ignore"):
            raise ValueError(
                f"passivity must be 'raise', 'clamp' or 'ignore', got {passivity!r}")
        self.models_dir = models_dir
        self.device = device
        self.passivity = passivity
        self.warn_out_of_range = bool(warn_out_of_range)
        self.cache = SurrogateCache() if cache is None else cache
        self._loaded = None       # (model, scaler, torch_device), built on first use
        self._fingerprint = None

    # -- loading ------------------------------------------------------------

    def available(self):
        """Instance-level :func:`available`, honouring this object's ``models_dir``."""
        return available(self.models_dir)

    def _load(self):
        """Import torch + tessera and build the model. Idempotent; the slow path."""
        if self._loaded is not None:
            return self._loaded
        if not available(self.models_dir):
            raise ModuleNotFoundError(
                "the Tessera TSV surrogate is not usable here: "
                + _missing_reason(self.models_dir)
            )
        import torch  # local: importing this module must not import torch
        from tessera.model import TSVPhysicsGNN
        from tessera.scaler import InputScaler

        mdir = checkpoint_dir(self.models_dir)
        dev = torch.device(self.device)
        # Deliberately NOT tessera.load_model(): that calls load_config() first, and
        # config.yaml is absent after a pip install from git (module docstring). This
        # rebuilds the same object from upstream's own public classes.
        model = TSVPhysicsGNN().to(dev)
        model.load_state_dict(
            torch.load(str(mdir / "best_model.pth"), map_location=dev, weights_only=True))
        model.eval()
        scaler = InputScaler()
        scaler.load(str(mdir / "input_scaler.pt"))

        self._fingerprint = f"{(mdir / 'best_model.pth').stat().st_size}" \
                            f"-{(mdir / 'input_scaler.pt').stat().st_size}" \
                            f"-v{self.OUTPUT_VERSION}"
        self._loaded = (model, scaler, dev)
        return self._loaded

    # -- validation ---------------------------------------------------------

    def _check_ranges(self, params, freqs_hz):
        if not self.warn_out_of_range:
            return
        bad = []
        for name, value in params.items():
            lo, hi = VALID_RANGES[name]
            if not (lo <= float(value) <= hi):
                bad.append(f"{name}={float(value):g} outside [{lo:g}, {hi:g}]")
        flo, fhi = VALID_RANGES["freq_hz"]
        fmin, fmax = float(np.min(freqs_hz)), float(np.max(freqs_hz))
        if fmin < flo or fmax > fhi:
            bad.append(f"freq_hz spans [{fmin:.3g}, {fmax:.3g}] outside [{flo:g}, {fhi:g}]")
        if bad:
            warnings.warn(
                "Tessera surrogate queried outside its recovered training ranges: "
                + "; ".join(bad)
                + ". Output is an extrapolation (see e2e/interconnect_surrogate/"
                  "tessera.py VALID_RANGES); the passivity guard is the only backstop.",
                UserWarning,
                stacklevel=3,
            )

    def _passivity_scale(self, peak, design_repr):
        """Return the factor to multiply the S-matrix by so ``|S21| <= 1``.

        ``1.0`` when the prediction is already passive (or ``passivity='ignore'``);
        otherwise either raises or returns ``1/peak`` with a loud warning.
        """
        if peak <= 1.0 or self.passivity == "ignore":
            return 1.0
        excess_db = 20.0 * np.log10(peak)
        message = (
            f"Tessera surrogate returned |S21| = +{excess_db:.2f} dB (gain) for "
            f"{design_repr}. A passive TSV cannot have gain; this is unguarded "
            f"extrapolation outside the training box (VALID_RANGES)."
        )
        if self.passivity == "raise":
            raise PassivityError(message + " Bound the knob to VALID_RANGES, or "
                                           "construct with passivity='clamp'.")
        warnings.warn(message + " Clamping the curve to a 0 dB peak -- the returned "
                                "response is NOT the model's output.",
                      RuntimeWarning, stacklevel=3)
        return 1.0 / peak

    # -- prediction ---------------------------------------------------------

    def _design(self, freq_hz, params, arrangement):
        return {
            "radius": params["radius_um"] * 1e-6,
            "pitch": params["pitch_um"] * 1e-6,
            "height": params["height_um"] * 1e-6,
            "liner": params["liner_um"] * 1e-6,
            "temperature": float(params["temperature_k"]),
            "freq": float(freq_hz),
            "arrangement": arrangement,
        }

    @staticmethod
    def _resolve_grid(grid):
        """Accept a preset name, an odd int (ring size), or an explicit 2-D array."""
        if grid is None:
            grid = _DEFAULT_ARRANGEMENT
        if isinstance(grid, str):
            try:
                return ARRANGEMENTS[grid]
            except KeyError:
                raise ValueError(
                    f"unknown arrangement {grid!r}; known: {sorted(ARRANGEMENTS)}"
                ) from None
        if isinstance(grid, (int, np.integer)):
            return ring_arrangement(int(grid))
        arr = np.asarray(grid, dtype=np.int8)
        if arr.ndim != 2:
            raise ValueError(f"arrangement must be 2-D, got shape {arr.shape}")
        if not (arr == 1).any():
            raise ValueError("arrangement must contain at least one signal via (+1)")
        return arr

    def s_matrix(self, freqs_hz, *, radius_um, pitch_um, height_um, liner_um,
                 temperature_k, grid=None):
        """Full complex S-matrix per frequency, ``[n_freqs, 2*n_sig, 2*n_sig]``.

        Ports are two per signal via: ``2k`` is via ``k``'s input, ``2k+1`` its output,
        so ``S[:, 2k, 2k+1]`` is that via's S21, ``S[:, 2k, 2k]`` its S11,
        ``S[:, 2u, 2v]`` the NEXT between vias ``u`` and ``v`` and ``S[:, 2u, 2v+1]``
        their FEXT (upstream ``tessera/smatrix.py`` defines this mapping).

        ``dtype`` is ``complex128``, matching upstream. The passivity guard is applied
        to the S21 diagonal only; other terms are returned as predicted.
        """
        arrangement = self._resolve_grid(grid)
        params = {"radius_um": float(radius_um), "pitch_um": float(pitch_um),
                  "height_um": float(height_um), "liner_um": float(liner_um),
                  "temperature_k": float(temperature_k)}
        freqs = np.atleast_1d(np.asarray(freqs_hz, dtype=np.float64))
        self._check_ranges(params, freqs)

        model, scaler, dev = self._load()
        key = cache_key(params, freqs, arrangement, self._fingerprint + "-smatrix")
        hit = self.cache.get(key)
        if hit is not None and "s" in hit:
            s_all = hit["s"]
        else:
            from tessera import predict_s_matrix
            mats = [
                predict_s_matrix(self._design(f, params, arrangement),
                                 model=model, scaler=scaler, device=dev)
                for f in freqs
            ]
            s_all = np.stack(mats, axis=0).astype(np.complex128)
            self.cache.put(key, {"s": s_all})

        n_sig = int((arrangement == 1).sum())
        s21 = np.stack([s_all[:, 2 * k, 2 * k + 1] for k in range(n_sig)], axis=1)
        peak = float(np.abs(s21).max()) if s21.size else 0.0
        scale = self._passivity_scale(peak, self._repr(params, arrangement))
        # Scale the WHOLE matrix, not just S21: the crosstalk terms come from the same
        # unguarded prediction, so silently leaving them at their raw level next to a
        # rescaled S21 would report a coupling ratio that the model never produced.
        return s_all if scale == 1.0 else s_all * scale

    def s21(self, freqs_hz, *, radius_um, pitch_um, height_um, liner_um,
            temperature_k, grid=None, signal_index=0):
        """Complex S21 of one signal via on ``freqs_hz``, ``[n_freqs]`` complex128.

        This is the array ``InterconnectBlock`` wants: the same quantity its CSV path
        reads out of ``s21_re + 1j*s21_im`` and multiplies along the frame's frequency
        axis. ``signal_index`` selects which signal via when the arrangement has more
        than one (the default ``ring3x3`` has exactly one).
        """
        s = self.s_matrix(freqs_hz, radius_um=radius_um, pitch_um=pitch_um,
                          height_um=height_um, liner_um=liner_um,
                          temperature_k=temperature_k, grid=grid)
        k = int(signal_index)
        return np.ascontiguousarray(s[:, 2 * k, 2 * k + 1])

    def s21_db(self, freqs_hz, **kwargs):
        """``20*log10|S21|`` on ``freqs_hz``, ``[n_freqs]`` float64.

        Insertion loss is negative. See :meth:`s21` for the parameters.
        """
        return 20.0 * np.log10(np.abs(self.s21(freqs_hz, **kwargs)) + 1e-15)

    def crosstalk_db(self, freqs_hz, *, grid="checker3x3", **kwargs):
        """Worst-case NEXT and FEXT in dB over all signal pairs, per frequency.

        Returns ``{"next_db": [n_freqs], "fext_db": [n_freqs], "n_signals": int}``,
        each entry the loudest coupling seen at that frequency (the number a crosstalk
        budget cares about). The default ``grid`` is a multi-signal arrangement
        because the single-via ring has no pairs at all -- asking for crosstalk on a
        one-signal grid raises rather than returning an empty answer.
        """
        arrangement = self._resolve_grid(grid)
        n_sig = int((arrangement == 1).sum())
        if n_sig < 2:
            raise ValueError(
                f"crosstalk needs >= 2 signal vias, arrangement has {n_sig}. "
                f"Use e.g. grid='checker3x3'."
            )
        s = self.s_matrix(freqs_hz, grid=arrangement, **kwargs)
        pairs = [(u, v) for u in range(n_sig) for v in range(n_sig) if u != v]
        nxt = np.stack([np.abs(s[:, 2 * u, 2 * v]) for u, v in pairs], axis=1)
        fxt = np.stack([np.abs(s[:, 2 * u, 2 * v + 1]) for u, v in pairs], axis=1)
        return {
            "next_db": 20.0 * np.log10(nxt.max(axis=1) + 1e-15),
            "fext_db": 20.0 * np.log10(fxt.max(axis=1) + 1e-15),
            "n_signals": n_sig,
        }

    @staticmethod
    def _repr(params, arrangement):
        return (f"r={params['radius_um']:g} um, pitch={params['pitch_um']:g} um, "
                f"h={params['height_um']:g} um, liner={params['liner_um']:g} um, "
                f"T={params['temperature_k']:g} K, {arrangement.shape[0]}x"
                f"{arrangement.shape[1]} arrangement")

    def __repr__(self):
        state = "loaded" if self._loaded is not None else "not loaded"
        return (f"TesseraTSV(device={self.device!r}, passivity={self.passivity!r}, "
                f"{state}, available={self.available()})")
