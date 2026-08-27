"""
Ground-truth label encoding for the FMCW radar detection head, FFTRadNet/RADIal-style.

Adapted (structure only, not code -- the reference is a numpy `encoder.py`, this is pure
torch) from the RADIal repo's `dataset/encoder.py` `ra_encoder.encode`/`.decode`. See the
"label encoding" scout notes for the reference behaviour this mirrors.

Surface footprint, centre regression (CHANGED 2026-08-17)
---------------------------------------------------------
A radar return comes from a target's nearest reflecting SURFACE, not from its geometric
centre. MEASURED offsets between the two on this project's own meshes: 2.39 m for a car,
0.50 m pedestrian/sphere, 4.17 m bus, 6.14 m trolley, 7.81 m truck -- up to ~26 output
range bins at the `ti_iwr1443` preset. Until 2026-08-17 the objectness footprint sat on
the CENTRE, so the classification head was trained on a blind plateau (no energy there),
and `e2e.ml.metrics`' 2.0 m match tolerance turned the offset into a cliff rather than a
gradual penalty: with predictions placed at the true surface and labels at the centre,
oracle AP measured 1.00 at 1.75 m of offset, 0.21 at 2.00 m, and 0.00 from 2.20 m. Cars,
buses, trolleys and trucks were total losses; pedestrians and spheres were unaffected.

Both quantities are therefore emitted, and they mean different things:

* the **objectness footprint** goes on the nearest visible SURFACE point along the line
  of sight (`e2e.environment.geometry.nearest_surface_point`, the same model
  `e2e.ml.rt_signal_chain` places its coherent point scatterer on) -- that is where the
  energy is, so that is what a detector can learn and what matching is done on;
* the **regression target stays the object CENTRE**, encoded as a residual from the
  footprint cell exactly as before. It stays the centre because a downstream tracker
  integrates centre-of-mass kinematics: feed it surface range and a car turning broadside
  produces a ~1.2 m range jump, which a constant-velocity filter at 10 Hz reads as a
  phantom ~12 m/s radial transient.

The surface point lies ON the radar-to-centre line, so surface and centre share an
azimuth exactly; only the range differs. The azimuth channel is untouched by all of this.

An object with unknown geometry (`Scatterer.extent_m is None` -- every hand-built point
scatterer, and every object of a `scatterers.SYNTHETIC_BASE_SCENE` scenario, whose
objects really ARE points) has surface == centre, i.e. exactly the pre-2026-08-17
encoding. Where an extent IS known, the signal side moves with the label:
`rd_synth.synthesize_adc` puts its single point of return on the same surface point
(`rd_synth._scattering_point`), as the ray-traced chain already did.

Conventions
-----------
* Output-map geometry (`LabelGrid`): a `(range, sin(azimuth))` grid, NOT a `(range,
  angle_degrees)` grid -- azimuth is stored as `sin(theta)` on a uniform `[-1, 1)` axis,
  matching the direction-cosine convention a ULA actually resolves (see
  `e2e.chain.rd_synth.array_axis`), rather than the reference's linear-degrees axis.
* Label map: float32 `[3, n_range, n_azimuth]`.
  - channel 0: objectness, `1.0` on a dense 3x3 footprint centred on each target's
    (SURFACE range, azimuth) cell, `0.0` elsewhere. Footprints are clipped (not wrapped)
    at the grid boundary.
  - channels 1-2: range/azimuth regression residuals, defined **per footprint cell**
    (not just at the target's own cell): for footprint cell `(i, j)`, the residual is
    `(true_CENTRE_value - that_cell's_own_bin_centre) / that_cell's_bin_size`. This is
    RADIal's "linear offset-gradient" scheme -- every one of the 9 footprint cells
    independently encodes enough information to reconstruct the exact target position, so
    decoding does not depend on which footprint cell a downstream NMS happens to keep.
    Because a target can sit up to half a bin away from its own cell's centre, and a
    footprint cell can be up to one full bin away from that cell, residuals span roughly
    `[-1.5, 1.5]` in bin units across the footprint (as opposed to `[-0.5, 0.5]` if they
    were only ever written at the target's own cell).
    RANGE-RESIDUAL BOUND (widened 2026-08-17): the range residual is measured from a cell
    on the SURFACE to the CENTRE, so it is offset by `delta / range_bin_m` and spans
    roughly `[-1.5, 1.5 + delta/range_bin_m]` -- one-sided, always toward larger range.
    MEASURED worst cases over the shipped asset fleet: ~8 bins for a car and ~26 for a
    15.7 m semi at `ti_iwr1443` (0.3 m output bins), ~3 and ~10 at `radial_like` (0.8 m
    bins). Nothing clamps it: the models emit raw, unbounded regression channels (see
    `e2e.ml.models.fftradnet`) and `e2e.ml.losses` never assumes a range.
    LOSS SCALE, checked rather than assumed: the smooth-L1 regression term now sits in
    its LINEAR regime at initialization, so its VALUE grows (MEASURED at `reg_weight=100`
    with a zero-regression prediction and 3 targets: 0.76 for point targets -> 6.6 car ->
    17.6 for a 15.7 m semi). Its GRADIENT does not -- smooth-L1 caps at 1 per element, so
    the measured per-cell gradient is bit-identical and the summed regression gradient
    rises only 1.25x. No loss re-weighting was needed. A future switch to a
    non-saturating (L2) regression term would NOT be safe here; that is pinned by
    `tests/test_ml_labels.py::test_widened_residual_does_not_change_the_regression_gradient_scale`.
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
  `(centre_range, sin_azimuth, class, surface_range)` tuples of in-grid targets without
  re-deriving the geometry. The first three entries keep their pre-2026-08-17 meaning and
  values exactly; the surface range is APPENDED, so a consumer that only reads
  `t[0]`/`t[1]`/`t[2]` is unaffected, and `e2e.ml.metrics` (which matches on the surface)
  falls back to `t[0]` for a 3-tuple, i.e. treats it as a point target.
* Out-of-grid scatterers are silently skipped, matching the reference encoder's handling
  of its own sentinel/OOB rows. In-grid is tested on the SURFACE point (`r` in
  `[0, max_range_m)`, `|sin_azimuth| < 1`) -- the cell the footprint would be written to --
  by both `encode_detection_labels` and `targets_in_grid`, so the label map and the target
  list can never disagree about which objects are present.
* `decode_detections` returns `(range_m, sin_azimuth, score, surface_range_m)` tuples,
  the same append-only extension: `range_m` is still the regression channels' own
  reconstruction (now the object CENTRE), and `surface_range_m` -- the kept cell's own
  bin centre -- is what `e2e.ml.metrics` matches on, since a detector has no way to
  estimate an object's size and therefore cannot regress its surface. It is CELL-
  QUANTIZED: the residual channels refine the centre to sub-bin precision, but nothing
  refines the surface, and the cell a plateau's NMS keeps can be up to one cell off the
  target's own, so the worst case is +-1.5 range bins (0.45 m at `ti_iwr1443`, 1.2 m at
  `radial_like`) against the 2.0 m match tolerance. That is the price of matching where
  the energy is; the alternative -- matching a regressed centre against a centre label --
  is what put the whole vehicle fleet off the cliff.
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

from e2e.environment.geometry import nearest_surface_point
from e2e.chain.rd_synth import array_axis, device

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
def _range_sin_az(position, pose) -> Tuple[float, float]:
    """`(range_m, sin_azimuth)` of a world `position` relative to `pose`.

    See `rd_synth.array_axis` for the azimuth convention.
    """
    origin = np.asarray(pose.position, dtype=np.float64).reshape(3)
    los = np.asarray(position, dtype=np.float64).reshape(3) - origin
    r = float(np.linalg.norm(los))
    if r < 1e-6:
        return r, 0.0
    sin_az = float((los / r) @ array_axis(pose))
    return r, sin_az


def target_geometry(scatterer, pose) -> Tuple[float, float, float]:
    """`(surface_range_m, sin_azimuth, centre_range_m)` of `scatterer` seen from `pose`.

    The one place this package converts a scatterer into label geometry -- used by
    `encode_detection_labels` and `targets_in_grid` here, and by `e2e.ml.dataset`'s
    parallel bookkeeping, so no consumer can drift out of step with the encoder about
    which objects are in the grid or where they sit.

    `surface_range_m` is the range to the nearest point of the object's bounding
    ellipsoid along the line of sight (`e2e.environment.geometry.nearest_surface_point`), and
    equals `centre_range_m` for a scatterer with no known extent. The surface point is on
    the radar-to-centre line, so the single returned `sin_azimuth` is correct for both.
    """
    centre = np.asarray(scatterer.position, dtype=np.float64).reshape(3)
    r_centre, sin_az = _range_sin_az(centre, pose)
    extent = getattr(scatterer, "extent_m", None)
    if extent is None:
        return r_centre, sin_az, r_centre
    origin = np.asarray(pose.position, dtype=np.float64).reshape(3)
    half = 0.5 * np.asarray(extent, dtype=np.float64).reshape(3)
    surface = nearest_surface_point(centre, half, origin,
                                    yaw_rad=float(getattr(scatterer, "yaw_rad", 0.0)))
    return float(np.linalg.norm(surface - origin)), sin_az, r_centre


def _in_grid(grid: LabelGrid, r: float, sin_az: float) -> bool:
    return (0.0 <= r < grid.max_range_m) and (abs(sin_az) < 1.0)


def targets_in_grid(grid: LabelGrid, scatterers: Sequence, pose,
                    classes: Optional[Sequence[str]] = None
                    ) -> List[Tuple[float, float, str, float]]:
    """Ground-truth tuples for every scatterer whose SURFACE point falls inside `grid`.

    Each entry is `(centre_range_m, sin_azimuth, object_class, surface_range_m)`. The
    first three keep their pre-2026-08-17 meaning and value exactly (the object's own
    centre); the surface range is appended for the matcher (see the module docstring and
    `e2e.ml.metrics`). For a scatterer with no known extent the two ranges are equal.

    Reuses the exact same geometry `encode_detection_labels` uses, for dataset/eval code
    that needs the raw target list (e.g. counting objects per frame) without re-deriving
    range/azimuth itself.

    `classes`: when given, only scatterers whose `object_class` is in it are returned.
    The dataset generator passes ("vehicle", "pedestrian") so that background clutter
    (object_class "scatterer") appears in the SIGNAL but never in the ground truth --
    clutter is something a detector must reject, not detect. `None` keeps every class.
    """
    keep = None if classes is None else set(classes)
    out: List[Tuple[float, float, str, float]] = []
    for sc in scatterers:
        if keep is not None and sc.object_class not in keep:
            continue
        r_surface, sin_az, r_centre = target_geometry(sc, pose)
        if _in_grid(grid, r_surface, sin_az):
            out.append((r_centre, sin_az, sc.object_class, r_surface))
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
        r_surface, sin_az, r_centre = target_geometry(sc, pose)
        if not _in_grid(grid, r_surface, sin_az):
            continue

        # Footprint on the SURFACE cell (where the energy is); regression toward the
        # CENTRE (what a tracker needs). See the module docstring.
        ci = min(int(r_surface / range_bin_m), grid.n_range - 1)
        cj = min(int((sin_az + 1.0) / az_bin), grid.n_azimuth - 1)

        for i in range(max(ci - _HALF, 0), min(ci + _HALF + 1, grid.n_range)):
            r_center = (i + 0.5) * range_bin_m
            for j in range(max(cj - _HALF, 0), min(cj + _HALF + 1, grid.n_azimuth)):
                az_center = -1.0 + (j + 0.5) * az_bin
                label[0, i, j] = 1.0
                label[1, i, j] = (r_centre - r_center) / range_bin_m
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
) -> List[Tuple[float, float, float, float]]:
    """Invert a label/prediction map into detections.

    Each detection is `(range_m, sin_azimuth, score, surface_range_m)`:

    * `range_m`/`sin_azimuth` are the kept cell's regression channels reconstructed
      against its own bin centres -- i.e. the predicted object CENTRE, at sub-bin
      precision, which is what `range_rmse_m`/`sin_az_rmse` are scored on;
    * `surface_range_m` is the kept cell's own range-bin centre -- the position the
      detector actually fired at, cell-quantized (worst case +-1.5 range bins; see the
      module docstring). `e2e.ml.metrics` matches on it, because a detector cannot
      regress a surface it has no size estimate for.

    For a point target (or an untrained/zero regression channel) the two ranges differ
    only by the sub-bin regression, so this is a strict superset of the pre-2026-08-17
    return value.

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
        detections.append((r, sin_az, score, r_center))

    detections.sort(key=lambda d: d[2], reverse=True)
    return detections
