"""
Ground-truth label encoding for the FMCW radar detection head, FFTRadNet/RADIal-style.

Adapted (structure only, not code -- the reference is a numpy `encoder.py`, this is pure
torch) from the RADIal repo's `dataset/encoder.py` `ra_encoder.encode`/`.decode`. See the
"label encoding" scout notes for the reference behaviour this mirrors.

Conventions
-----------
* Output-map geometry (`LabelGrid`): a `(range, sin(azimuth))` grid, NOT a `(range,
  angle_degrees)` grid -- azimuth is stored as `sin(theta)` on a uniform `[-1, 1)` axis,
  matching the direction-cosine convention a ULA actually resolves (see
  `e2e.ml.rd_synth.array_axis`), rather than the reference's linear-degrees axis.
* Label map: float32 `[3, n_range, n_azimuth]`.
  - channel 0: objectness, `1.0` on a dense 3x3 footprint centred on each target's
    (range, azimuth) cell, `0.0` elsewhere. Footprints are clipped (not wrapped) at the
    grid boundary.
  - channels 1-2: range/azimuth regression residuals, defined **per footprint cell**
    (not just at the true centre cell): for footprint cell `(i, j)`, the residual is
    `(true_value - that_cell's_own_bin_centre) / that_cell's_bin_size`. This is RADIal's
    "linear offset-gradient" scheme -- every one of the 9 footprint cells independently
    encodes enough information to reconstruct the exact target position, so decoding does
    not depend on which footprint cell a downstream NMS happens to keep. Because a target
    can sit up to half a bin away from its own cell's centre, and a footprint cell can be
    up to one full bin away from the target's own cell, residuals span roughly
    `[-1.5, 1.5]` in bin units across the footprint (as opposed to `[-0.5, 0.5]` if they
    were only ever written at the centre cell).
    CAVEAT -- the per-cell-reconstruction guarantee holds for ISOLATED targets only.
    When two targets' footprints overlap (centres within Chebyshev distance <= 2 cells),
    the encode loop is last-writer-wins on the shared cells: their residuals belong to
    whichever scatterer appears LATER in the input sequence (deterministic for a given
    list order, but order-dependent -- callers must supply scatterers in a stable order).
    Decoding such clusters is fundamentally lossy: the greedy NMS (radius = footprint)
    can suppress a target sandwiched 1-2 cells between two others entirely. This is an
    inherited limitation of the RADIal-style coarse-cell representation; the dataset
    generator keeps labelled targets >= min_target_separation_m apart, which at these
    grid resolutions keeps their footprints disjoint.
* `encode_detection_labels` is tensor-only (matches the RADIal reference's return type);
  `targets_in_grid` is a separate helper for bookkeeping/eval code that wants the raw
  (range, sin_azimuth, class) tuples of in-grid targets without re-deriving the geometry.
* Out-of-grid scatterers (`r` outside `[0, max_range_m)` or `|sin_azimuth| >= 1`) are
  silently skipped, matching the reference encoder's handling of its own sentinel/OOB rows.
* `decode_detections` inverts the map: a 3x3 max-pool identifies local maxima of channel 0
  above `threshold` (this is necessary, and NOT present in the RADIal reference decoder,
  because our channel-0 footprint is a flat plateau of nine equal `1.0`s rather than a
  single hot pixel -- naive thresholding would otherwise emit up to 9 duplicate detections
  per target); a following greedy suppression pass (keep highest-score first, drop any
  later candidate within `nms_footprint` cells of an already-kept one) collapses each
  plateau to one detection. The kept cell's own regression channels then reconstruct
  `(range, sin_azimuth)` exactly (see above), so the regression step recovers sub-bin
  precision that raw cell quantization alone could not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from e2e.ml.rd_synth import array_axis, device

# Footprint written around each target's cell by `encode_detection_labels` (3x3, per the
# RADIal reference's `geometry.size == 3` configuration).
_FOOTPRINT = 3
_HALF = _FOOTPRINT // 2


@dataclass(frozen=True)
class LabelGrid:
    """Output-map geometry: range x sin(azimuth) grid the detector predicts on."""

    n_range: int          # output range bins
    n_azimuth: int        # output azimuth bins
    max_range_m: float    # grid covers [0, max_range_m)

    @property
    def range_bin_m(self) -> float:
        """Range-bin size, metres."""
        return self.max_range_m / self.n_range

    @property
    def az_bin(self) -> float:
        """Azimuth-bin size in sin(theta) units (the axis spans [-1, 1))."""
        return 2.0 / self.n_azimuth

    @classmethod
    def for_config(cls, cfg, range_stride: int = 4, n_azimuth: int = 192) -> "LabelGrid":
        """Derive an output grid from a `RadarConfig`-like object.

        `range_stride` mirrors the reference's backbone stride (raw ADC range bins ->
        output range bins is a 4x downsample there); `n_azimuth` is a free design choice
        (the reference's angle-axis width falls out of its backbone channel count, which
        we don't have here, so it is just a parameter).
        """
        return cls(
            n_range=int(cfg.n_samples) // int(range_stride),
            n_azimuth=int(n_azimuth),
            max_range_m=float(cfg.max_range_m),
        )


# --------------------------------------------------------------------------------
# Geometry helpers (shared by encode / targets_in_grid)
# --------------------------------------------------------------------------------
def _range_sin_az(scatterer, pose) -> Tuple[float, float]:
    """`(range_m, sin_azimuth)` of `scatterer` relative to `pose` (see `rd_synth.array_axis`)."""
    origin = np.asarray(pose.position, dtype=np.float64).reshape(3)
    los = np.asarray(scatterer.position, dtype=np.float64).reshape(3) - origin
    r = float(np.linalg.norm(los))
    if r < 1e-6:
        return r, 0.0
    sin_az = float((los / r) @ array_axis(pose))
    return r, sin_az


def _in_grid(grid: LabelGrid, r: float, sin_az: float) -> bool:
    return (0.0 <= r < grid.max_range_m) and (abs(sin_az) < 1.0)


def targets_in_grid(grid: LabelGrid, scatterers: Sequence, pose,
                    classes: Optional[Sequence[str]] = None) -> List[Tuple[float, float, str]]:
    """`(range_m, sin_azimuth, object_class)` for every scatterer that falls inside `grid`.

    Reuses the exact same geometry `encode_detection_labels` uses, for dataset/eval code
    that needs the raw target list (e.g. counting objects per frame) without re-deriving
    range/azimuth itself.

    `classes`: when given, only scatterers whose `object_class` is in it are returned.
    The dataset generator passes ("vehicle", "pedestrian") so that background clutter
    (object_class "scatterer") appears in the SIGNAL but never in the ground truth --
    clutter is something a detector must reject, not detect. `None` keeps every class.
    """
    keep = None if classes is None else set(classes)
    out: List[Tuple[float, float, str]] = []
    for sc in scatterers:
        if keep is not None and sc.object_class not in keep:
            continue
        r, sin_az = _range_sin_az(sc, pose)
        if _in_grid(grid, r, sin_az):
            out.append((r, sin_az, sc.object_class))
    return out


# --------------------------------------------------------------------------------
# Encode
# --------------------------------------------------------------------------------
def encode_detection_labels(grid: LabelGrid, scatterers: Sequence, pose,
                            classes: Optional[Sequence[str]] = None) -> torch.Tensor:
    """Ground-truth label map for `scatterers` at `pose`, float32 `[3, n_range, n_azimuth]`.

    See the module docstring for the channel layout and footprint/residual conventions.
    Placed on the library `device` (cuda if available, else cpu) since the map is a plain
    torch computation with no upstream tensor to inherit a device from.

    `classes` filters which object classes become ground truth (same semantics as
    `targets_in_grid`): the dataset generator passes ("vehicle", "pedestrian") so
    background clutter contributes signal but no labels. `None` labels every class.
    """
    keep = None if classes is None else set(classes)
    label = torch.zeros((3, grid.n_range, grid.n_azimuth), dtype=torch.float32, device=device)
    range_bin_m = grid.range_bin_m
    az_bin = grid.az_bin

    for sc in scatterers:
        if keep is not None and sc.object_class not in keep:
            continue
        r, sin_az = _range_sin_az(sc, pose)
        if not _in_grid(grid, r, sin_az):
            continue

        ci = min(int(r / range_bin_m), grid.n_range - 1)
        cj = min(int((sin_az + 1.0) / az_bin), grid.n_azimuth - 1)

        for i in range(max(ci - _HALF, 0), min(ci + _HALF + 1, grid.n_range)):
            r_center = (i + 0.5) * range_bin_m
            for j in range(max(cj - _HALF, 0), min(cj + _HALF + 1, grid.n_azimuth)):
                az_center = -1.0 + (j + 0.5) * az_bin
                label[0, i, j] = 1.0
                label[1, i, j] = (r - r_center) / range_bin_m
                label[2, i, j] = (sin_az - az_center) / az_bin

    return label


# --------------------------------------------------------------------------------
# Decode
# --------------------------------------------------------------------------------
def decode_detections(
    grid: LabelGrid,
    label_map: torch.Tensor,
    threshold: float = 0.5,
    nms_footprint: int = 3,
) -> List[Tuple[float, float, float]]:
    """Invert a label/prediction map into `(range_m, sin_azimuth, score)` detections.

    `label_map` is float `[3, n_range, n_azimuth]` (either the ground-truth map from
    `encode_detection_labels` or a trained model's output of the same shape/convention).
    Runs entirely on `label_map`'s own device; returns plain Python floats (there is
    nothing left to keep on-device once detections are individual scalars). Sorted by
    score, descending.
    """
    label_map = torch.as_tensor(label_map)
    if label_map.dim() != 3 or label_map.shape[0] != 3:
        raise ValueError(f"label_map must be [3, n_range, n_azimuth], got {tuple(label_map.shape)}")

    objectness = label_map[0]
    reg_range = label_map[1]
    reg_az = label_map[2]

    # 3x3 (or nms_footprint) local-maxima mask: a cell survives if nothing in its window
    # beats it. This alone is not enough to dedupe the encoder's flat 3x3 footprint
    # plateaus (every cell in a plateau ties its own local max) -- the greedy suppression
    # pass below handles that.
    pad = nms_footprint // 2
    pooled = F.max_pool2d(objectness[None, None], kernel_size=nms_footprint, stride=1, padding=pad)[0, 0]
    peak_mask = (objectness >= pooled) & (objectness > threshold)
    idx = peak_mask.nonzero(as_tuple=False)
    if idx.numel() == 0:
        return []

    scores = objectness[idx[:, 0], idx[:, 1]]
    order = torch.argsort(scores, descending=True).tolist()

    kept: List[Tuple[int, int, float]] = []
    for k in order:
        i, j = int(idx[k, 0]), int(idx[k, 1])
        if any(max(abs(i - ki), abs(j - kj)) < nms_footprint for ki, kj, _ in kept):
            continue
        kept.append((i, j, float(scores[k])))

    range_bin_m = grid.range_bin_m
    az_bin = grid.az_bin
    detections = []
    for i, j, score in kept:
        r_center = (i + 0.5) * range_bin_m
        az_center = -1.0 + (j + 0.5) * az_bin
        r = r_center + float(reg_range[i, j]) * range_bin_m
        sin_az = az_center + float(reg_az[i, j]) * az_bin
        detections.append((r, sin_az, score))

    detections.sort(key=lambda d: d[2], reverse=True)
    return detections
