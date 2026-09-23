"""
On-disk compression for ML radar-corpus payload arrays.

Shared by `e2e.ml.blocks.SinkBlock`/`SourceBlock` and `e2e.ml.dataset`'s
writer (`generate_dataset`) / reader (`RadarFrameDataset`) so both producers of the
`.npz`-per-frame on-disk schema (see `e2e/ml/dataset.py`'s "On-disk dataset layout"
docstring) get the same codec, and a reader never has to special-case which one wrote
a given file.

MEASURED (see the investigation behind this module -- not re-derived here) on
`ti_iwr1443`-preset ADC cubes (4 rx x 192 chirps x 512 samples, complex64, 3.145
MB/frame), BOTH clean and impaired (`e2e.chain.impairments.apply_all` -- a realistic
corpus is noisier, and therefore less compressible, than a clean one, so measuring
only clean cubes would flatter the result):

    codec                                | ratio (unquant / quantized-12bit) | error  | enc+dec ms/frame
    np.savez_compressed (raw complex64)  | ~1.08              / ~2.1         | exact  | ~150 + 15
    zstandard level 3 (raw bytes)        | ~1.08              / ~1.7         | exact  | ~7 + 6
    float16 real+imag                    | ~2.2               / ~2.3         | ~74 dB SNR (lossy) | ~80 + 18
    int16 codes @ quantizer's own scale  | n/a (needs a quantized cube)/~2.7 | EXACT  | ~300 + 15
    RD-domain top-K energy (impaired)    | top 5% of bins hold only ~45-65% of the energy -- not
                                          | concentrated enough to threshold losslessly; not implemented.

CHOSEN: int16 codes (`CODEC_INT16`). A cube that has already been through
`e2e.chain.receive.QuantizerBlock` (the real corpus path, see `e2e.ml.chain_generate`)
is, BY CONSTRUCTION, a finite set of `2**bits` values spaced `full_scale /
2**(bits-1)` apart. Representing each component as a signed 16-bit code at THAT SAME
step is exactly lossless and needs no knowledge of the original `bits` -- only
`full_scale` -- because 16 bits is always at least as fine as any realistic ADC's own
depth (8-16 bits), so every original code maps to an exact multiple on the finer
grid: `code_16 = code_orig * 2**(16 - bits_orig)`, an exact integer. This alone halves
the array bytes (2 vs 8 bytes/component); because the low bits of every code are then
structurally zero, the follow-on `np.savez_compressed` zlib pass finds substantially
more redundancy than it does on the raw floats, ~2.7x measured overall.

REJECTED: float16 is lossy and, on an already-quantized cube, strictly worse than
int16 codes on BOTH axes (smaller ratio, not exact) -- no reason to accept lossy when
a same-or-better lossless option exists. zstandard is already installed (no new
dependency pulled in for this) but measured worse than zlib on the quantized cubes
that matter, and adopting it would mean a bespoke binary container instead of the
`.npz` format the rest of this package reads/writes. RD-domain sparsity -- the
instinct that there is exploitable structure is right, but an impaired,
clutter-and-noise-filled cube spreads energy too broadly to threshold losslessly.

SAFETY: `encode_payload` VERIFIES the round trip byte-for-byte (`np.array_equal`)
before ever returning `CODEC_INT16`; any mismatch (data that is not, in fact,
uniformly quantized at the assumed scale -- e.g. `e2e.ml.dataset.generate_sample`'s
analytic-fallback floats, which never pass through `QuantizerBlock`) falls back to
`CODEC_RAW`, byte-identical to the pre-existing behavior. The codec actually used is
always recorded in the sample's `meta["codec"]` (see `write_sample_npz`), so a reader
never has to guess -- and a `.npz` written before this module existed (no "codec" key
in `meta`) is, by definition, `CODEC_RAW` with the payload stored under its own key
unchanged (see `read_payload`'s back-compat branch).

A sample may additionally carry SIDECAR FILES holding the RAY-TRACED CHANNEL it was
produced from, so the analog/digital chain can be re-run live instead of being frozen
at generation time: `<stem>.cfr.npy` (`write_cfr_sidecar` -- the dense channel frame,
uncompressed, fast to replay, 64 MiB/frame at `benchmark_v1`) and `<stem>.paths.npz`
(`write_paths_sidecar` -- the path list it was synthesised from, ~1 MB/frame, the
durable form). Both are separate files: none of the `.npz` keys above change, and a
corpus with neither reads exactly as it always did.

This module is intentionally torch-free (numpy only) -- it has no reason to need
torch, and staying dependency-light keeps it reusable outside the pipeline (e.g. a
plain corpus-inspection script).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

CODEC_RAW = "raw"
CODEC_INT16 = "int16"

# Storage bit depth for CODEC_INT16 -- fixed, not the original ADC's `bits` (see
# module docstring: 16 bits is fine enough to exactly represent any realistic ADC
# depth's codes, and `encode_payload`'s round-trip verification is what actually
# guarantees losslessness, not this constant).
_INT16_BITS = 16
_INT16_TOP = 2 ** (_INT16_BITS - 1) - 1          # 32767
_INT16_BOTTOM = -(2 ** (_INT16_BITS - 1))        # -32768


def _peak_abs(array: np.ndarray) -> float:
    """Largest |value| across an array's real (and, if complex, imaginary) parts.
    Used only as a FALLBACK scale guess when no `full_scale` hint is given (see
    `encode_payload`) -- `np.imag` of a real-dtype array is an all-zero array, not an
    error, so this works for both real and complex `array`."""
    if array.size == 0:
        return 0.0
    peak = float(np.max(np.abs(np.real(array))))
    if np.iscomplexobj(array):
        peak = max(peak, float(np.max(np.abs(np.imag(array)))))
    return peak


def _int16_encode(array: np.ndarray, scale: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    re = np.clip(np.round(np.real(array) / scale), _INT16_BOTTOM, _INT16_TOP).astype(np.int16)
    im = None
    if np.iscomplexobj(array):
        im = np.clip(np.round(np.imag(array) / scale), _INT16_BOTTOM, _INT16_TOP).astype(np.int16)
    return re, im


def _int16_decode(re: np.ndarray, im: Optional[np.ndarray], scale: float, dtype) -> np.ndarray:
    out_re = re.astype(np.float32) * np.float32(scale)
    if im is None:
        return out_re.astype(dtype)
    out_im = im.astype(np.float32) * np.float32(scale)
    return (out_re + 1j * out_im).astype(dtype)


# --------------------------------------------------------------------------------
# Low-level codec: one array <-> (codec name, arrays to write, small codec_meta dict)
# --------------------------------------------------------------------------------
def encode_payload(array: np.ndarray, *, full_scale: Optional[float] = None
                   ) -> Tuple[str, Dict[str, np.ndarray], Dict[str, Any]]:
    """Choose the best LOSSLESS codec for `array` (a real- or complex-floating numpy
    array) and return `(codec, arrays, codec_meta)`.

    `full_scale`, if given, should be the EXACT value the array was quantized against
    (e.g. `state["adc_full_scale"]` -- see `e2e.chain.receive.QuantizerBlock`, which
    returns exactly this). Without it, `array`'s own peak magnitude is tried as a
    heuristic guess -- it only succeeds when that peak happens to be an exact multiple
    relationship with the array's true quantization step (e.g. no AGC headroom
    margin was used), which the round-trip check below still verifies before
    accepting; a bad guess just falls back to `CODEC_RAW`, it never produces a wrong
    answer.

    Non-floating dtypes (already-integer arrays, bools, ...) go straight to
    `CODEC_RAW` -- the int16-code scheme only makes sense for a converter's floating
    output.
    """
    array = np.asarray(array)
    if not (np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.complexfloating)):
        return CODEC_RAW, {"payload": array}, {}

    fs = float(full_scale) if full_scale is not None else _peak_abs(array)
    if fs > 0.0 and array.size:
        scale = fs / (2 ** (_INT16_BITS - 1))
        re, im = _int16_encode(array, scale)
        back = _int16_decode(re, im, scale, array.dtype)
        if np.array_equal(back, array):    # SAFETY: only ever accept an exact round trip
            arrays: Dict[str, np.ndarray] = {"code_re": re}
            if im is not None:
                arrays["code_im"] = im
            return CODEC_INT16, arrays, {"scale": scale, "dtype": str(array.dtype)}

    return CODEC_RAW, {"payload": array}, {}


def decode_payload(codec: str, arrays: Dict[str, np.ndarray], codec_meta: Dict[str, Any]) -> np.ndarray:
    """Inverse of `encode_payload`. `arrays` holds exactly the keys `encode_payload`
    returned (`"payload"` for `CODEC_RAW`, `"code_re"`/`"code_im"` for `CODEC_INT16`)."""
    if codec == CODEC_RAW:
        return arrays["payload"]
    if codec == CODEC_INT16:
        dtype = np.dtype(codec_meta.get("dtype", "complex64"))
        return _int16_decode(arrays["code_re"], arrays.get("code_im"), codec_meta["scale"], dtype)
    raise ValueError(f"unknown codec {codec!r} (expected {CODEC_RAW!r} or {CODEC_INT16!r})")


# --------------------------------------------------------------------------------
# Sample-file level: the payload_key/meta convention SinkBlock and dataset.py share
# --------------------------------------------------------------------------------
def write_sample_npz(path, arrays: Dict[str, np.ndarray], meta: Dict[str, Any], *,
                     payload_key: str, full_scale: Optional[float] = None,
                     json_default=None) -> None:
    """Write one sample `.npz`. `arrays[payload_key]` (e.g. `"adc"`) is
    codec-compressed via `encode_payload`; every OTHER entry of `arrays` (e.g.
    `"labels"`) is stored exactly as given -- small float32 grids, not worth the
    int16 machinery, and this module has no opinion about them. `meta` is written
    verbatim except `meta["codec"]`/`meta["codec_meta"]` are set/overwritten from
    whatever `encode_payload` actually chose, so `read_payload` never has to guess
    (see module docstring). `json_default` is forwarded to `json.dumps` (callers pass
    their own dataclass/numpy/tensor fallback -- see `e2e.ml.blocks._json_default` /
    `e2e.ml.dataset._json_default`).
    """
    payload = arrays[payload_key]
    codec, payload_arrays, codec_meta = encode_payload(payload, full_scale=full_scale)

    meta = dict(meta)
    meta["codec"] = codec
    meta["codec_meta"] = codec_meta

    out_arrays = {k: v for k, v in arrays.items() if k != payload_key}
    if codec == CODEC_RAW:
        out_arrays[payload_key] = payload_arrays["payload"]
    else:
        out_arrays[f"{payload_key}_code_re"] = payload_arrays["code_re"]
        if "code_im" in payload_arrays:
            out_arrays[f"{payload_key}_code_im"] = payload_arrays["code_im"]

    np.savez_compressed(
        path, meta=np.array(json.dumps(meta, default=json_default)), **out_arrays,
    )


# --------------------------------------------------------------------------------
# CFR sidecar: the ray-traced channel, stored beside the sample it produced
# --------------------------------------------------------------------------------
#: Suffix of the sidecar holding a sample's pre-chain CFR frame (`s_pars`), written
#: next to that sample's `.npz` and named in its `meta["cfr_sidecar"]`.
CFR_SIDECAR_SUFFIX = ".cfr.npy"


def cfr_sidecar_path(npz_path):
    """Sidecar path for a sample `.npz` (`<stem>.cfr.npy`, same directory)."""
    return Path(npz_path).with_suffix(CFR_SIDECAR_SUFFIX)


def write_cfr_sidecar(npz_path, array: np.ndarray) -> str:
    """Store `array` (a sample's entering CFR frame) beside its `.npz`; return the
    sidecar's BASENAME, for `meta["cfr_sidecar"]`.

    Deliberately `np.save` -- uncompressed, and NOT folded into the `.npz` above.
    MEASURED 2026-09-23 on the boundary array this exists for (`benchmark_v1`
    `s_pars`, complex64 `[16, 4, 256, 512]` = 64 MiB/frame, the RT/chain boundary of
    the D2 corpus; see the storage scoping pilot):

        zlib (level 1) ratio        0.92-0.93  -- i.e. ~7% recovered
        np.savez_compressed cost    ~3.5 s/frame
        np.save (this)              ~0.35 s write, ~0.08 s read

    A ray-traced CFR is a sum of many complex exponentials: it is essentially
    full-entropy float noise, so general-purpose compression has nothing to find.
    Paying 10x the write time for 7% is the wrong trade at 64 MiB x N frames, and
    the int16 codec above does not apply either -- this array has never been through
    a quantizer, so it is not on a uniform code grid and `encode_payload` would fall
    straight back to `CODEC_RAW` after doing the round-trip check.

    Lossy float16 WAS measured (half the bytes, rel-RMSE 2e-3 / 5e-4 on the two
    pilot scenes) and is NOT used: the point of storing the CFR is that everything
    downstream can be re-run from it and compared against what was stored, and a
    lossy boundary would put a floor under every such comparison.
    """
    path = cfr_sidecar_path(npz_path)
    np.save(path, np.asarray(array))
    return path.name


def read_cfr_sidecar(npz_path, meta: Dict[str, Any]) -> np.ndarray:
    """Load the CFR sidecar `meta` names, resolved next to `npz_path`.

    Raises `FileNotFoundError` -- loudly, naming the flag -- when the sample records
    no sidecar or the file is gone, rather than silently falling back to the stored
    ADC cube in a different signal domain.
    """
    npz_path = Path(npz_path)
    name = meta.get("cfr_sidecar")
    if not name:
        raise FileNotFoundError(
            f"{npz_path.name} records no 'cfr_sidecar': it was generated without the "
            f"CFR store (chain_generate --store-cfr / SinkBlock(store_cfr=True)), so "
            f"there is no ray-traced frame to replay from."
        )
    path = npz_path.parent / str(name)
    if not path.exists():
        raise FileNotFoundError(
            f"{npz_path.name} names CFR sidecar {name!r}, which is missing from "
            f"{npz_path.parent} -- the corpus was copied without its sidecars."
        )
    return np.load(path)


# --------------------------------------------------------------------------------
# Paths sidecar: the ray-traced PATH LIST, for re-synthesising the channel later
# --------------------------------------------------------------------------------
#: Suffix of the sidecar holding a sample's ray-traced path list and the per-object
#: geometry the coherent-target term is built from.
PATHS_SIDECAR_SUFFIX = ".paths.npz"


def paths_sidecar_path(npz_path):
    """Paths-sidecar path for a sample `.npz` (`<stem>.paths.npz`, same directory)."""
    return Path(npz_path).with_suffix(PATHS_SIDECAR_SUFFIX)


def write_paths_sidecar(npz_path, arrays: Dict[str, np.ndarray]) -> str:
    """Store a frame's ray-traced path list beside its `.npz`; return the BASENAME,
    for `meta["paths_sidecar"]`.

    `arrays` is a flat `{name: ndarray}` mapping produced by the RT layer (which owns
    the Sionna `Paths` object; this module never imports Sionna and has no opinion
    about the key names beyond writing them verbatim). It is the INPUT SIDE of the
    channel: what `e2e.environment.rt_signal_chain.cfr_from_paths` and
    `coherent_target_cfr` consume, rather than the dense CFR they produce.

    WHY BOTH sidecars exist (owner preference, notes/TODO.md ~line 824, 2026-08-10):
    the dense CFR (`write_cfr_sidecar`) is a ~12x storage regression at corpus scale
    -- 64 MiB/frame at `benchmark_v1` against ~0.7-1.6 MB for the paths that generated
    it -- so the durable form is the path list plus enough scene geometry to
    re-synthesise the CFR without re-tracing. The dense sidecar is kept for the demo,
    where re-synthesis cost matters more than disk.

    `np.savez_compressed`, unlike the CFR sidecar: at ~1 MB the compression time is
    negligible, and a path list has exploitable structure (padded/invalid path slots,
    repeated object ids) that the dense CFR does not.

    This function does NOT define the re-synthesis; it only makes sure the inputs
    survive. See the shard report for the exact fields the RT layer must supply.
    """
    path = paths_sidecar_path(npz_path)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in arrays.items()})
    return path.name


def read_paths_sidecar(npz_path, meta: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Load the paths sidecar `meta` names, as a plain `{name: ndarray}` dict."""
    npz_path = Path(npz_path)
    name = meta.get("paths_sidecar")
    if not name:
        raise FileNotFoundError(
            f"{npz_path.name} records no 'paths_sidecar': it was generated without the "
            f"path store (chain_generate --store-paths / SinkBlock(store_paths=True))."
        )
    path = npz_path.parent / str(name)
    if not path.exists():
        raise FileNotFoundError(
            f"{npz_path.name} names paths sidecar {name!r}, which is missing from "
            f"{npz_path.parent} -- the corpus was copied without its sidecars."
        )
    with np.load(path, allow_pickle=False) as data:
        return {k: np.array(data[k]) for k in data.files}


def read_payload(npz_data, meta: Dict[str, Any], payload_key: str) -> np.ndarray:
    """`npz_data` is an already-open `np.load(...)` result (or any `Mapping[str,
    ndarray]`, e.g. a plain dict) -- this module never opens files itself, so a
    caller that only needs `meta` (e.g. `RadarFrameDataset.targets()`'s meta-only fast
    path) is free to skip loading the payload arrays at all. Returns the array at
    `payload_key`, decoded per `meta.get("codec", CODEC_RAW)`.

    BACK-COMPAT: a `.npz` written before this module existed has no `"codec"` key in
    `meta` at all -- `meta.get("codec", CODEC_RAW)` treats that, correctly, as
    `CODEC_RAW` with the payload stored directly under `payload_key`, unchanged.
    """
    codec = meta.get("codec", CODEC_RAW)
    if codec == CODEC_RAW:
        return np.asarray(npz_data[payload_key])
    arrays = {"code_re": np.asarray(npz_data[f"{payload_key}_code_re"])}
    im_key = f"{payload_key}_code_im"
    if im_key in npz_data:
        arrays["code_im"] = np.asarray(npz_data[im_key])
    return decode_payload(codec, arrays, meta["codec_meta"])
