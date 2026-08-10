"""
Persistence + neural-detector pipeline blocks for the radar-ML chain.

Follows the same convention as `e2e/comms/blocks.py` (a thin, domain-scoped
adapter alongside the code it wraps, not folded into the top-level
`e2e/blocks.py`): each class exposes `apply(state) -> dict` (see
`e2e/blocks.py`'s serial-stage / downstream-block protocol and
`e2e/frames.py`'s `FrameCapabilities`), and importing this module has no
side effects beyond defining the classes.

Three blocks, sitting at the END of the chain (see the chain-integration
design notes):

* `SinkBlock`   -- persists whatever the chain holds at the point it is
  inserted (an intermediate artifact, or a full training sample).
* `SourceBlock` -- the inverse: loads what a `SinkBlock` wrote and replays it,
  usable as `Simulation`'s `environment_block` so a chain can start mid-way
  and skip ray tracing entirely.
* `NeuralDetectorBlock` -- runs `FFTRadNet`/`SSMRadNet` as a pipeline product
  (`mode="infer"`) or trains one from an on-disk corpus (`mode="train"`,
  which REUSES `e2e.ml.train.train` wholesale rather than reimplementing a
  training loop -- see that method's docstring).

Dependency rule (enforced by review, see the chain-integration design notes):
this module may import FROM `e2e.ml.dataset` (and does, only for the shared
on-disk artifact conventions it documents), but `e2e/ml/dataset.py` must
NEVER import from this module or any other block layer -- it stays a pure
data producer. `e2e/ml/train.py` similarly does not import this module.

Artifact format (shared by `SinkBlock`/`SourceBlock`)
-------------------------------------------------------
One `.npz` per frame, named `<tag>_frame_?????.npz` under the sink/source's
directory -- `tag` NAMESPACES the file so multiple sinks (e.g. `tag="clean"`
right after a dechirp stage and `tag="sample"` at the end of the same chain)
never collide, and a directory can hold several tagged artifact streams at
once. Each file holds:

  * the frame's payload tensor, stored under `frames.DOMAIN_PAYLOAD_KEY` of
    whatever domain was live in `state['signal_domain']` at the point the
    sink was inserted (`'adc'` / `'s_pars'` / `'tx_wave'`) -- so a sink placed
    where `signal_domain == frames.DOMAIN_RX_TIME` writes an `'adc'` array,
    exactly the key name `e2e.ml.dataset`'s on-disk format already uses;
  * `'labels'` (only if `state['labels']` is present -- the "training sample"
    flavor; a sink with no labels in scope, e.g. an intermediate artifact,
    omits it);
  * `'meta'`, a 0-d unicode array holding a JSON string: `{"tag", "frame_idx",
    "domain", "payload_key", "shape", "dtype"}` plus any of
    `{"impairment_params", "targets", "meta"}` that were present in `state`
    (see `_EXTRA_META_KEYS`) -- enough for the artifact to describe itself
    without the writer's code.

A directory of `tag="sample"`/domain-`DOMAIN_RX_TIME` artifacts (`'adc'` +
`'labels'` + `'meta'`) is therefore struct-compatible with the per-frame npz
`e2e.ml.dataset.generate_dataset` writes (see that module's "On-disk dataset
layout" docstring) -- the SAME field names, so such a directory plus a
hand-written `manifest.json` loads via `e2e.ml.dataset.RadarFrameDataset`.
There is no single writer FUNCTION to call, though (`generate_dataset` inlines
its `np.savez_compressed` call in a loop; nothing is factored out for a
single frame) -- `SinkBlock` matches the on-disk SCHEMA that function
documents rather than sharing code with it, which is what "reuse" means here.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from e2e import frames
from e2e.frames import FrameCapabilities


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Extra per-frame state keys that, when present, are folded into an artifact's
# self-describing 'meta' JSON blob (see the module docstring). Kept to a small,
# documented allowlist rather than dumping the whole state dict: most state
# entries (U, PRX, another block's product tensor) are either large arrays or
# meaningless replayed out of the frame that produced them.
_EXTRA_META_KEYS = ("impairment_params", "targets", "meta")


def _json_default(obj):
    """`json.dump(..., default=_json_default)` fallback for numpy/torch scalars.

    Mirrors `e2e.ml.dataset._json_default` -- duplicated rather than imported
    (a private helper of a sibling module; see the module docstring's
    dependency rule -- `dataset.py` must never import from here, so sharing
    code the other direction, importing dataset's private helper, is the only
    option, and this is cheap enough not to bother).

    Dataclasses are handled first because `ImpairmentBlock` records this frame's
    impairment settings as dataclass INSTANCES, and those are exactly the provenance a
    written sample needs to be self-describing. Without this the very first frame of a
    chain run dies with a TypeError from the JSON encoder.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serializable")


def _artifact_path(out_dir, tag, frame_idx):
    return Path(out_dir) / f"{tag}_frame_{frame_idx:05d}.npz"


# --------------------------------------------------------------------------------
# SinkBlock
# --------------------------------------------------------------------------------
class SinkBlock:
    """Persists whatever the chain holds at the point it is inserted.

    Works as either a serial stage or a downstream block -- both call
    `apply(state) -> dict`, and `SinkBlock` never rewrites the frame, so its
    return is always `{}`. Several `SinkBlock`s may appear in ONE chain (e.g.
    `tag="clean"` right after a dechirp stage and `tag="sample"` at the very
    end); `tag` namespaces the artifact filenames (see the module docstring)
    so they never collide on disk, even in the same `out_dir`.

    `domain` (default `frames.DOMAIN_RX_TIME`, this shard's primary use case)
    pins the block's declared `frame_capabilities.domain` to whatever domain
    is live at the point this instance is wired into the chain -- the caller
    knows that when they place it (see the chain-integration design notes'
    usage examples). This is a real limitation of the current
    `FrameCapabilities` contract (it names exactly one domain per component;
    there is no "any domain" declaration to fall back on) -- flagged here,
    not fixed, since `e2e/frames.py` is outside this shard.  At RUN TIME the
    domain used to pick the payload key is read from
    `state.get('signal_domain', self.domain)`, so a `SinkBlock` used directly
    (bypassing `Simulation`'s frame-contract check, e.g. in tests) still
    self-describes correctly even if `domain` wasn't set to match.
    """

    def __init__(self, out_dir, tag: str = "sample", domain: str = frames.DOMAIN_RX_TIME):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.tag = tag
        self.domain = domain
        self.frame_capabilities = FrameCapabilities(
            accepts_mimo=True, chirps=frames.CHIRP_NATIVE, domain=domain,
        )
        self._frame_counter = 0

    def reset(self):
        """Rewind the per-instance frame counter (mirrors ModemBlock's
        `reset()` -- see e2e/comms/blocks.py -- so repeated `Simulation.run()`
        calls overwrite the same filenames instead of appending forever).
        NOTE: `Simulation.reset()` only calls `reset()` on `downstream_blocks`,
        not `serial_stages` (see e2e/simulation.py) -- a `SinkBlock` used as a
        serial stage (e.g. `tag="clean"` in the design notes' example) is NOT
        auto-reset between runs; this is a handoff item for `e2e/simulation.py`,
        outside this shard, not fixed here."""
        self._frame_counter = 0

    def apply(self, state: Dict[str, Any]) -> Dict[str, Any]:
        idx = self._frame_counter
        self._frame_counter += 1

        domain = state.get("signal_domain", self.domain)
        payload_key = frames.DOMAIN_PAYLOAD_KEY.get(domain)
        if payload_key is None or payload_key not in state:
            raise KeyError(
                f"SinkBlock(tag={self.tag!r}): domain {domain!r} expects "
                f"state[{payload_key!r}], which is not present -- is this Sink "
                f"wired at the right point in the chain?"
            )
        payload = state[payload_key]
        payload_t = torch.as_tensor(payload)

        meta: Dict[str, Any] = {
            "tag": self.tag,
            "frame_idx": idx,
            "domain": domain,
            "payload_key": payload_key,
            "shape": list(payload_t.shape),
            "dtype": str(payload_t.dtype).replace("torch.", ""),
        }
        for key in _EXTRA_META_KEYS:
            if key in state and state[key] is not None:
                meta[key] = state[key]

        arrays = {payload_key: payload_t.detach().cpu().numpy()}
        labels = state.get("labels", None)
        if labels is not None:
            labels_np = (labels.detach().cpu().numpy() if torch.is_tensor(labels)
                        else np.asarray(labels))
            arrays["labels"] = labels_np

        path = _artifact_path(self.out_dir, self.tag, idx)
        np.savez_compressed(
            path, meta=np.array(json.dumps(meta, default=_json_default)), **arrays,
        )
        return {}


# --------------------------------------------------------------------------------
# SourceBlock
# --------------------------------------------------------------------------------
class SourceBlock:
    """The inverse of `SinkBlock`: loads stored artifacts and replays them.

    Usable as `Simulation`'s `environment_block` -- it implements the
    interface `SionnaEnvironmentBlock` does (`get_S_pars()` / `step()` /
    `reset()`, see `e2e/blocks.py`) -- and additionally exposes
    `signal_domain` (restored from the artifact's own recorded domain, so a
    chain resuming from a `DOMAIN_RX_TIME` sink resumes IN that domain, not
    silently back in `DOMAIN_CFR`) and `get_state_updates()` for the extra
    per-frame entries a `SinkBlock` recorded (labels/targets/meta/
    impairment_params -- see `_EXTRA_META_KEYS`).

    `get_S_pars()` is named for interface parity with `SionnaEnvironmentBlock`
    even when the replayed payload is not an S-parameter frame (e.g. `adc`
    at `DOMAIN_RX_TIME`) -- check `self.signal_domain` to know what it
    actually returned.

    `Simulation` reads both of these: `signal_domain` decides which state key the
    replayed payload is seeded under and whether the frequency-domain machinery
    (the SVD and subspace ground truth) applies at all, and `get_state_updates()`
    supplies the stored metadata -- labels included -- so they travel with the
    frame. See `Simulation._environment_state_updates` and `_feed_forward_from`.
    """

    def __init__(self, in_dir, tag: str = "sample"):
        self.in_dir = Path(in_dir)
        self.tag = tag
        self._files = sorted(self.in_dir.glob(f"{tag}_frame_*.npz"))
        if not self._files:
            raise FileNotFoundError(
                f"no {tag!r}-tagged artifacts found in {self.in_dir} (expected "
                f"files matching '{tag}_frame_?????.npz' -- see SinkBlock)"
            )
        self.frame_counter = 0
        self._cache = None   # (frame_counter, payload, meta, extra) of the last load
        # Peek the first artifact so signal_domain is known before the first
        # get_S_pars() call.
        _, first_meta, _ = self._load(self._files[0])
        self.signal_domain = first_meta["domain"]
        self.payload_key = first_meta["payload_key"]

    def __len__(self):
        return len(self._files)

    def step(self):
        self.frame_counter += 1
        if self.frame_counter >= len(self._files):
            self.frame_counter = 0

    def reset(self):
        self.frame_counter = 0

    def _load(self, path):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"].item()))
            payload_key = meta["payload_key"]
            payload = torch.from_numpy(data[payload_key]).to(device)
            extra: Dict[str, Any] = {}
            if "labels" in data:
                extra["labels"] = torch.from_numpy(data["labels"]).to(device)
            for key in _EXTRA_META_KEYS:
                if key in meta:
                    extra[key] = meta[key]
        return payload, meta, extra

    def _current(self):
        idx = self.frame_counter
        if self._cache is not None and self._cache[0] == idx:
            return self._cache[1:]
        payload, meta, extra = self._load(self._files[idx])
        self._cache = (idx, payload, meta, extra)
        return payload, meta, extra

    def get_S_pars(self):
        """Current frame's payload tensor (see the class docstring -- named
        for interface parity, not necessarily an S-parameter frame)."""
        payload, meta, _extra = self._current()
        self.signal_domain = meta["domain"]
        return payload

    def get_state_updates(self) -> Dict[str, Any]:
        """Extra per-frame state entries recorded by the `SinkBlock` that
        wrote the current frame (see the class docstring's handoff note)."""
        _payload, _meta, extra = self._current()
        return dict(extra)


# --------------------------------------------------------------------------------
# NeuralDetectorBlock
# --------------------------------------------------------------------------------
class NeuralDetectorBlock:
    """Runs `FFTRadNet`/`SSMRadNet` (`e2e.ml.models`) as a pipeline product.

    `mode="infer"` (default): a product block, like `FFTBlock` -- it reads
    `state['adc']` (the `DOMAIN_RX_TIME` payload), derives the model's input
    tensor, runs a forward pass, and returns `{'ml_detection': [3,
    n_range_out, n_azimuth_out]}` (channel 0 sigmoid objectness, channels 1-2
    raw range/azimuth regression residuals, matching `e2e.ml.labels`'s
    convention) plus `{'ml_detections': [...]}` (decoded `(range_m,
    sin_azimuth, score)` tuples via `e2e.ml.labels.decode_detections`) IF a
    `grid` is available. `state['adc']` itself is left UNTOUCHED -- the
    returned dict never includes it.

    `model_or_ckpt` is either an already-built `nn.Module` (used as-is, e.g. a
    freshly constructed, untrained model in tests) or a checkpoint (a path, or
    an already-`torch.load`ed dict) written by `e2e.ml.train.train` -- in
    which case `e2e.ml.train.build_model` reconstructs the exact architecture
    from the checkpoint's `manifest_path` (REUSED, not reimplemented).

    `input_format` ("rd" default | "adc", matching `e2e.ml.train`'s /
    `RadarFrameDataset`'s convention) selects how `state['adc']` becomes the
    model's `[B, C, R, D]` input:
      * "adc" -- the raw physical-channel transform (transpose + stack
        real/imag), matching `RadarFrameDataset._derive_input`'s "adc" branch
        exactly -- no `cfg` needed.
      * "rd"  -- `e2e.ml.transforms.adc_to_rd` (+ `tdm_deinterleave` for
        `cfg.mimo == "tdm"`) + `rd_to_input`, the same chain
        `e2e.ml.dataset.generate_sample` uses -- needs `cfg` (a `RadarConfig`).
    Loading from a checkpoint infers `input_format` from the checkpoint (or
    its manifest) instead of the constructor argument, matching
    `e2e.ml.train.evaluate`'s own precedence.

    `mode="train"`: does NOT run per-`apply()` gradient steps -- `Simulation`'s
    frame-by-frame streaming interface has no concept of epochs, and
    `e2e.ml.train.train` already implements a manifest-driven training loop
    (DataLoader batching, Adam, per-epoch validation, checkpoint selection) --
    reimplementing a second one here was explicitly out of scope. `apply(state)`
    is therefore a harmless no-op counter, valid as a `downstream_block` on a
    short `Simulation(environment_block=SourceBlock(corpus_dir), ...)` graph
    (see the chain-integration design notes' "training is its own graph");
    `fit()` is the actual entry point, forwarding straight to
    `e2e.ml.train.train(self.manifest_path, self.model_name, ...)`.
    """

    # Reads state['adc'] (DOMAIN_RX_TIME); its own [n_rx,n_chirp,n_samples] ->
    # batched-input reshape is internal, not a chirp-count restriction.
    frame_capabilities = FrameCapabilities(
        accepts_mimo=True, chirps=frames.CHIRP_NATIVE, domain=frames.DOMAIN_RX_TIME,
    )

    def __init__(self, model_or_ckpt=None, mode: str = "infer", *, manifest_path=None,
                model_name: Optional[str] = None, grid=None, input_format: str = "rd",
                cfg=None, threshold: float = 0.5, device=None,
                train_kwargs: Optional[Dict[str, Any]] = None):
        if mode not in ("infer", "train"):
            raise ValueError(f"mode must be 'infer' or 'train', got {mode!r}")
        self.mode = mode
        self.manifest_path = Path(manifest_path) if manifest_path is not None else None
        self.grid = grid
        self.input_format = input_format
        self.cfg = cfg
        self.threshold = threshold
        self._device = device if device is not None else globals()["device"]
        self._frame_count = 0

        if mode == "infer":
            self.model = self._build_infer_model(model_or_ckpt)
            self.model.eval()
            self.model_name = model_name
        else:  # "train"
            if self.manifest_path is None:
                raise ValueError(
                    "NeuralDetectorBlock(mode='train') needs manifest_path -- "
                    "training is driven off the on-disk corpus (see fit() / "
                    "e2e.ml.train.train), not individual apply() calls"
                )
            if model_name is None and isinstance(model_or_ckpt, str):
                model_name = model_or_ckpt
            if model_name is None:
                raise ValueError(
                    "NeuralDetectorBlock(mode='train') needs model_name "
                    "('fftradnet' | 'ssmradnet')"
                )
            self.model_name = model_name
            self.train_kwargs = dict(train_kwargs) if train_kwargs else {}
            self.model = None

    def _build_infer_model(self, model_or_ckpt):
        if isinstance(model_or_ckpt, nn.Module):
            return model_or_ckpt.to(self._device)
        if model_or_ckpt is None:
            raise ValueError(
                "NeuralDetectorBlock(mode='infer') needs model_or_ckpt (an "
                "nn.Module, or a checkpoint path/dict written by e2e.ml.train.train)"
            )
        # Path or already-loaded checkpoint dict: reuse e2e.ml.train.build_model
        # to reconstruct the exact architecture (REASONED-ONLY beyond this
        # module's own tests, which guard anything needing a real checkpoint --
        # see test_ml_blocks.py).
        from e2e.ml.train import build_model  # local: train.py pulls in DataLoader etc.

        ckpt = (model_or_ckpt if isinstance(model_or_ckpt, dict)
                else torch.load(model_or_ckpt, map_location=self._device))
        manifest_path = self.manifest_path or ckpt.get("manifest")
        if manifest_path is None:
            raise ValueError(
                "NeuralDetectorBlock: loading a checkpoint needs manifest_path "
                "(or a checkpoint that recorded its own 'manifest' path) to "
                "reconstruct the model's input/output geometry"
            )
        with open(manifest_path) as f:
            manifest = json.load(f)
        input_format = ckpt.get("input_format", manifest.get("input_format", "rd"))
        self.input_format = input_format
        manifest_for_model = dict(manifest)
        manifest_for_model["input_format"] = input_format
        model = build_model(ckpt["model_name"], manifest_for_model, device=self._device)
        model.load_state_dict(ckpt["model_state"])
        if self.grid is None:
            from e2e.ml.labels import LabelGrid

            g = manifest["grid"]
            self.grid = LabelGrid(n_range=int(g["n_range"]), n_azimuth=int(g["n_azimuth"]),
                                  max_range_m=float(g["max_range_m"]))
        return model

    def _derive_input(self, adc):
        adc = torch.as_tensor(adc, dtype=torch.complex64)
        if self.input_format == "adc":
            # Matches RadarFrameDataset._derive_input's "adc" branch exactly --
            # no deinterleave (see that method's docstring for why).
            adc_rsd = adc.transpose(1, 2)
            return torch.cat([adc_rsd.real, adc_rsd.imag], dim=0).to(torch.float32)
        if self.cfg is None:
            raise ValueError(
                "NeuralDetectorBlock(input_format='rd') needs cfg (a RadarConfig) "
                "to run adc_to_rd/tdm_deinterleave the same way "
                "e2e.ml.dataset.generate_sample does -- pass cfg=, or use "
                "input_format='adc'"
            )
        from e2e.ml import transforms

        if self.cfg.mimo == "tdm":
            sub_cfg = dataclasses.replace(self.cfg, n_tx=1, mimo="single",
                                          n_chirps=self.cfg.n_chirps_per_tx)
            rd = transforms.adc_to_rd(sub_cfg, transforms.tdm_deinterleave(self.cfg, adc))
        else:
            rd = transforms.adc_to_rd(self.cfg, adc)
        return transforms.rd_to_input(rd)

    def apply(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if self.mode == "train":
            # See the class docstring: no gradient step happens here.
            self._frame_count += 1
            return {}

        adc = state[frames.DOMAIN_PAYLOAD_KEY[frames.DOMAIN_RX_TIME]]
        x = self._derive_input(adc).unsqueeze(0).to(self._device)
        with torch.no_grad():
            detection = self.model(x)["detection"][0].cpu()   # [3, n_range_out, n_azimuth_out]
        out: Dict[str, Any] = {"ml_detection": detection}
        if self.grid is not None:
            from e2e.ml.labels import decode_detections

            out["ml_detections"] = decode_detections(self.grid, detection, threshold=self.threshold)
        return out   # deliberately does not touch 'adc' -- a product block, like FFTBlock

    def fit(self, **override_kwargs):
        """`mode="train"` only. REUSES `e2e.ml.train.train` wholesale (see the
        class docstring) -- no training loop is reimplemented in this module.
        Returns the `history` dict `train()` returns; a checkpoint lands under
        `train()`'s own `out_dir` convention (`<manifest dir>/runs/<model>/`
        by default), ready for a subsequent `mode="infer"` `NeuralDetectorBlock`
        to load."""
        if self.mode != "train":
            raise ValueError("fit() is only valid for mode='train'")
        from e2e.ml.train import train as _train

        kwargs = dict(self.train_kwargs)
        kwargs.update(override_kwargs)
        self.history = _train(self.manifest_path, self.model_name, **kwargs)
        return self.history
