"""
Animated bird's-eye + radar-view GIFs of `e2e.ml.scenes` scenarios.

Purely a visualization/README-media tool: it reuses the existing scene/synthesis stack
(`e2e.environment.scatterers.frame_scatterers`/`radar_pose`, `e2e.ml.rd_synth.synthesize_adc`,
`e2e.ml.transforms.adc_to_rd`/`tdm_deinterleave`, `e2e.ml.labels.LabelGrid`/
`targets_in_grid`) end to end; nothing here re-derives geometry or re-implements
synthesis. Two panels, side by side, one frame per animation tick:

  LEFT  -- bird's-eye (x, y) scatter: radar marker + boresight ray + FOV wedge + max-
           range arc, vehicles/pedestrians/clutter as distinct markers, velocity arrows.
  RIGHT -- the radar's own view: a range x sin(azimuth) power (dB) map formed from the
           SAME frame's synthesized ADC, with ground-truth (vehicle/pedestrian) targets
           overlaid as markers.

Motion: `e2e.ml.scenes.sample_scene` accepts `n_frames` and can return a real per-frame
motion track (the CLI passes `--frames` through when the installed `sample_scene`
supports it). A `Scenario` built elsewhere may still be single-frame, so
`render_scene_gif` checks `scenario.num_frames` at call time and takes one of two paths:
  * `scenario.num_frames >= n_frames`: a true motion track exists -- resolve each
    animation frame directly via `frame_scatterers(scenario, k, dt=dt)` /
    `radar_pose(scenario, k)` (no re-derivation, no dead reckoning).
  * otherwise (today's `sample_scene` output): resolve frame 0 once via
    `frame_scatterers`/`radar_pose`, then dead-reckon each scatterer's position for
    animation frame `k` as `position0 + velocity * dt * k` (velocity/rcs/class held
    fixed) -- linear extrapolation of the exact same per-object velocity
    `frame_scatterers` already computed, not a re-derivation of it.
`dt = 1 / cfg.frame_rate_hz` either way, matching `e2e.ml.dataset.generate_sample`'s
frame-timing convention.

Range-azimuth map: `adc_to_rd` (per-TX deinterleaved for TDM) gives a complex
`[n_channel, range_bin, doppler_bin]` cube. We FFT over the *virtual-array* channel
axis per Doppler bin (this is the operation that actually needs per-channel phase, so
it must run before any Doppler collapse), then collapse Doppler non-coherently by
taking the max power over the Doppler axis -- the standard "range-azimuth from a
range-Doppler cube" recipe (angle-resolve coherently, then integrate detection energy
across Doppler non-coherently since a target's exact Doppler bin isn't known a priori).
The channel-axis phase step is lambda/2 (see `rd_synth.synthesize_adc`'s virtual-array
comment), so an N-point FFT bin `k` (centred via fftshift, k in [-N/2, N/2)) maps to
sin(azimuth) = 2k/N.

Agg backend is forced at import time (no display / no Tk needed); safe to import in a
headless test process or CI.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # noqa: E402 -- must precede pyplot import; headless/CI-safe

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.animation import PillowWriter  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import Arc, Wedge  # noqa: E402

from e2e.environment.scatterers import RadarPose, Scatterer, frame_scatterers, radar_pose  # noqa: E402
from e2e.ml.labels import LabelGrid, targets_in_grid  # noqa: E402
from e2e.chain.rd_synth import synthesize_adc  # noqa: E402
from e2e.chain.transforms import adc_to_rd, ddma_demux, tdm_deinterleave  # noqa: E402
from e2e.viz import imshow_ra  # noqa: E402

# Marker/color convention: loosely matches `webapp/scenario_editor.py`'s ROLE_COLORS /
# OBJECT_COLOR (radar red, generic object amber) without importing Dash/Plotly here.
_COLOR_RADAR = "#eb3b5a"
_COLOR_VEHICLE = "#2d98da"
_COLOR_PEDESTRIAN = "#f7b731"
_COLOR_CLUTTER = "#a5b1c2"
_MARKERS = {"vehicle": "s", "pedestrian": "o", "scatterer": "."}
_COLORS = {"vehicle": _COLOR_VEHICLE, "pedestrian": _COLOR_PEDESTRIAN, "scatterer": _COLOR_CLUTTER}

# Velocity-arrow visual scale (seconds of travel drawn as the arrow length) -- purely
# cosmetic, chosen so typical D2/D3 speeds (a few m/s to ~10 m/s) give visible but not
# overwhelming arrows on a scene tens of metres across.
_VELOCITY_ARROW_S = 1.5

# GT marker style on the range-azimuth panel. Clutter is drawn too (grey, matching the
# bird's-eye panel's scatterer colour) because without it a viewer reads every unmarked
# bright return as a missed detection, when most of them are labelled clutter doing
# exactly what clutter does. Owner-requested, 2026-08-16.
_GT_MARKERS = {"vehicle": ("s", _COLOR_VEHICLE), "pedestrian": ("o", _COLOR_PEDESTRIAN),
               "scatterer": (".", _COLOR_CLUTTER)}
#: Classes marked on the radar panel, in draw order (clutter first, so real targets
#: sit on top of it rather than behind).
_RADAR_MARKED_CLASSES = ("scatterer", "vehicle", "pedestrian")


# --------------------------------------------------------------------------------
# Per-frame scatterer/pose resolution (motion-track-aware, dead-reckoning fallback)
# --------------------------------------------------------------------------------
def _resolve_frames(scenario, cfg, n_frames: int, dt: float):
    """`([scatterers_frame0, ...], [pose_frame0, ...])`, length `n_frames` each.

    See the module docstring's "Motion" section for the two paths this takes.
    """
    if scenario.num_frames >= n_frames:
        scats = [frame_scatterers(scenario, k, dt=dt) for k in range(n_frames)]
        poses = [radar_pose(scenario, k) for k in range(n_frames)]
        return scats, poses

    base = frame_scatterers(scenario, 0, dt=dt)
    pose0 = radar_pose(scenario, 0)  # single-frame scenarios have a static radar
    scats = []
    for k in range(n_frames):
        scats.append([
            Scatterer(
                position=tuple(p + v * dt * k for p, v in zip(sc.position, sc.velocity)),
                velocity=sc.velocity,
                rcs_dbsm=sc.rcs_dbsm,
                object_class=sc.object_class,
            )
            for sc in base
        ])
    return scats, [pose0] * n_frames


# --------------------------------------------------------------------------------
# Range-azimuth map
# --------------------------------------------------------------------------------
def range_azimuth_power(cfg, adc: torch.Tensor, n_angle_fft: Optional[int] = None,
                        azimuth_window: Optional[str] = None):
    """Raw ADC `[n_rx, n_chirps, n_samples]` -> `(ra_power [n_angle, n_range], sin_az_axis)`.

    Linear power (angle-FFT-then-non-coherent-Doppler-collapse recipe, see the module
    docstring), UN-normalized -- callers pick the dB reference. This is the seam that
    lets `render_scene_gif` normalize a whole animation against ONE global peak instead
    of re-normalizing every frame to its own peak (which silently re-scales the color
    map frame-to-frame).

    `azimuth_window` optionally tapers the aperture before the angle FFT ('hann'; None
    = the historical rectangular window). DISPLAY-ONLY knob: with a rectangular window
    a 12-element virtual array puts its first sidelobe just 13 dB down and its skirts
    decay slowly, so over a wide dB display a strong near target smears a bright ridge
    across EVERY azimuth bin at its range -- which visually buries genuinely weaker
    targets (pedestrians) sitting at the same range. A Hann taper trades ~1.6x mainlobe
    width for far lower sidelobes and makes targets read as targets. Detection paths do
    NOT come through here (`e2e.ml.baseline` has its own power function), so this
    changes pictures, never metrics -- and it defaults off so every other caller
    (`detect_viz`, `export_ssm`) keeps showing exactly what its detector saw.
    """
    if cfg.mimo == "tdm":
        sub_cfg = dataclasses.replace(cfg, n_tx=1, mimo="single", n_chirps=cfg.n_chirps_per_tx)
        rd = adc_to_rd(sub_cfg, tdm_deinterleave(cfg, adc))
    elif cfg.mimo == "ddma":
        # DDMA separates transmitters in Doppler, so the demux has to happen AFTER the
        # Doppler FFT (unlike TDM, which is de-interleaved in the raw ADC). Without this
        # the angle FFT below runs over n_rx PHYSICAL receivers rather than the
        # n_tx*n_rx virtual elements -- 16 instead of 192 for `radial_like`, which is
        # what smeared targets into horizontal ridges across every DDMA picture.
        rd = ddma_demux(cfg, adc_to_rd(cfg, adc))
    else:
        rd = adc_to_rd(cfg, adc)
    n_channel = rd.shape[0]
    n_fft = int(n_angle_fft) if n_angle_fft is not None else max(64, n_channel)

    if azimuth_window is not None:
        if azimuth_window != "hann":
            raise ValueError(f"unsupported azimuth_window {azimuth_window!r}; expected 'hann' or None")
        taper = torch.hann_window(n_channel, periodic=False, dtype=torch.float32,
                                  device=rd.device)
        rd = rd * taper.reshape(-1, 1, 1).to(rd.dtype)

    angle_spec = torch.fft.fftshift(torch.fft.fft(rd, n=n_fft, dim=0), dim=0)  # [n_fft, R, D]
    power = angle_spec.abs() ** 2
    ra_power = power.max(dim=2).values  # non-coherent Doppler collapse, [n_fft, R]

    sin_az_axis = 2.0 * (torch.arange(n_fft, dtype=torch.float32) - n_fft // 2) / n_fft
    return ra_power.to(torch.float32).cpu(), sin_az_axis.numpy()


def range_azimuth_map(cfg, adc: torch.Tensor, n_angle_fft: Optional[int] = None,
                      norm_peak: Optional[float] = None,
                      azimuth_window: Optional[str] = None):
    """Raw ADC `[n_rx, n_chirps, n_samples]` -> `(ra_db [n_angle, n_range], sin_az_axis)`.

    `ra_db` is normalized so `norm_peak` (linear power) sits at 0 dB; with the default
    `norm_peak=None` the map's own peak bin is the reference, the historical behavior.
    `sin_az_axis` is the centre sin(azimuth) of each row, ascending. Range axis
    (columns) is implicit: `i * cfg.range_resolution_m` for column `i`.
    """
    ra_power, sin_az_axis = range_azimuth_power(cfg, adc, n_angle_fft=n_angle_fft,
                                                azimuth_window=azimuth_window)
    eps = torch.finfo(torch.float32).tiny
    peak = torch.as_tensor(float(norm_peak)) if norm_peak is not None else ra_power.max()
    peak = peak.clamp_min(eps)
    ra_db = 10.0 * torch.log10((ra_power / peak).clamp_min(eps))
    return ra_db.to(torch.float32), sin_az_axis


# --------------------------------------------------------------------------------
# Bird's-eye panel
# --------------------------------------------------------------------------------
def _draw_birdseye(ax, scatterers: Sequence[Scatterer], pose: RadarPose, cfg):
    ax.clear()
    max_range = float(cfg.max_range_m)
    boresight = np.asarray(pose.boresight, dtype=float)[:2]
    origin = np.asarray(pose.position, dtype=float)[:2]

    # FOV wedge + max-range arc: full unambiguous sin(az) in [-1, 1] == +-90 deg from
    # boresight (see LabelGrid's [-1, 1) azimuth axis).
    heading_deg = float(np.degrees(np.arctan2(boresight[1], boresight[0])))
    wedge = Wedge(origin, max_range, heading_deg - 90.0, heading_deg + 90.0,
                  facecolor="#eb3b5a", alpha=0.05, edgecolor="none")
    ax.add_patch(wedge)
    arc = Arc(origin, 2 * max_range, 2 * max_range, angle=0.0,
              theta1=heading_deg - 90.0, theta2=heading_deg + 90.0,
              color="#eb3b5a", alpha=0.4, linestyle="--", linewidth=1.0)
    ax.add_patch(arc)

    ax.plot(*origin, marker="^", markersize=12, color=_COLOR_RADAR, linestyle="none",
            label="radar")
    ax.plot([origin[0], origin[0] + boresight[0] * max_range],
            [origin[1], origin[1] + boresight[1] * max_range],
            color=_COLOR_RADAR, linestyle=":", linewidth=1.0)

    seen_labels = set()
    for sc in scatterers:
        pos = np.asarray(sc.position, dtype=float)[:2]
        vel = np.asarray(sc.velocity, dtype=float)[:2]
        marker = _MARKERS.get(sc.object_class, ".")
        color = _COLORS.get(sc.object_class, _COLOR_CLUTTER)
        label = sc.object_class if sc.object_class not in seen_labels else None
        seen_labels.add(sc.object_class)
        size = 4 if sc.object_class == "scatterer" else 9
        ax.plot(pos[0], pos[1], marker=marker, markersize=size, color=color,
                linestyle="none", label=label)
        if sc.object_class != "scatterer":  # clutter is static; skip velocity arrows
            ax.annotate("", xy=tuple(pos + vel * _VELOCITY_ARROW_S), xytext=tuple(pos),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, alpha=0.85))

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Bird's-eye scene")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.7)
    pad = 0.1 * max_range
    ax.set_xlim(origin[0] - pad, origin[0] + max_range + pad)
    ax.set_ylim(origin[1] - max_range - pad, origin[1] + max_range + pad)
    # adjustable="box" (not the default "datalim") so the explicit x/y limits above are
    # kept exactly and the axes box itself is shrunk to preserve a 1:1 aspect ratio.
    ax.set_aspect("equal", adjustable="box")


# --------------------------------------------------------------------------------
# Radar-view panel
# --------------------------------------------------------------------------------
def _draw_radar_view(ax, cfg, grid: LabelGrid, ra_db: torch.Tensor, sin_az_axis: np.ndarray,
                     scatterers: Sequence[Scatterer], pose: RadarPose,
                     title: str = "Radar view: range-azimuth power (dB)",
                     vmin: float = -40.0, vmax: float = 0.0):
    ax.clear()
    max_range = float(cfg.max_range_m)
    n_range = ra_db.shape[1]
    range_axis = np.arange(n_range) * float(cfg.range_resolution_m)

    # Orientation (transpose + azimuth-on-x/range-on-y) is owned by e2e.viz.imshow_ra
    # -- see its docstring for why the transpose is needed (markers below use `extent`
    # directly and are unaffected either way).
    imshow_ra(ax, ra_db, sin_az_axis, range_axis, cmap="inferno", vmin=vmin, vmax=vmax)

    seen_labels = set()
    marked = targets_in_grid(grid, scatterers, pose, classes=_RADAR_MARKED_CLASSES)
    # Clutter first so real targets draw on top of it.
    order = {c: i for i, c in enumerate(_RADAR_MARKED_CLASSES)}
    for tgt in sorted(marked, key=lambda t: order.get(t[2], 99)):
        # (range_m, sin_azimuth, object_class[, surface_range_m]) -- see
        # `e2e.ml.labels.targets_in_grid`; index rather than unpack.
        r, sin_az, cls = tgt[0], tgt[1], tgt[2]
        marker, color = _GT_MARKERS.get(cls, ("x", "white"))
        is_clutter = cls == "scatterer"
        label = (("clutter" if is_clutter else f"GT {cls}")
                 if cls not in seen_labels else None)
        seen_labels.add(cls)
        if is_clutter:
            # Small, grey, no halo: present so unmarked bright returns are not misread
            # as missed targets, but deliberately quieter than the real targets.
            ax.plot(sin_az, r, marker="o", markersize=5, markerfacecolor="none",
                    markeredgecolor=color, markeredgewidth=1.1, linestyle="none",
                    label=label, alpha=0.75, zorder=3)
            continue
        # Dark halo under the marker: the class colors are shared with the bird's-eye
        # panel (so the legend reads the same across panels), but amber-on-inferno has
        # almost no contrast wherever the map sits mid-scale -- which is exactly where
        # the receiver noise floor lands. The halo keeps those colors legible on any
        # background without inventing a second color language for the radar view.
        ax.plot(sin_az, r, marker=marker, markersize=11, markerfacecolor="none",
                markeredgecolor="black", markeredgewidth=3.2, linestyle="none",
                zorder=4)
        ax.plot(sin_az, r, marker=marker, markersize=9, markerfacecolor="none",
                markeredgecolor=color, markeredgewidth=1.6, linestyle="none",
                label=label, zorder=5)

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(0.0, max_range)
    ax.set_xlabel("sin(azimuth)")
    ax.set_ylabel("range (m)")
    ax.set_title(title)
    if seen_labels:
        ax.legend(loc="upper right", fontsize=7, framealpha=0.7)


# --------------------------------------------------------------------------------
# Top-level GIF renderer
# --------------------------------------------------------------------------------
def render_scene_gif(cfg, scenario, out_path, *, n_frames: int = 30, fps: int = 8, seed: int = 0,
                     snr_db: Optional[float] = 30.0, dpi: int = 90,
                     n_angle_fft: Optional[int] = None, ideal_panel: bool = True,
                     db_range: float = 80.0,
                     azimuth_window: Optional[str] = "hann") -> Path:
    """Render a bird's-eye + radar-view animated GIF for `scenario`.

    `cfg` (a `RadarConfig`) sets both the synthesis parameters and the frame timing
    (`dt = 1 / cfg.frame_rate_hz`, see the module docstring). `seed` seeds
    `synthesize_adc`'s noise/phase RNG per frame (`seed + frame_index`, so frames are
    reproducible but not identical). Returns `out_path` as a `Path`.

    `ideal_panel` (default True) inserts a THIRD panel between the bird's-eye view
    and the noisy radar view: the SAME frame synthesized with `snr_db=None` (receiver
    noise disabled entirely; the per-scatterer reflection phases stay). What remains
    in that panel is only the scene's own ray-traced-style content -- targets,
    auxiliary scatterers, clutter -- so the ideal-vs-real pair shows exactly what the
    non-ideal front end costs.

    Color scale: ONE deliberate scale for the whole animation, not per-frame
    autoscale. Every radar map is referenced to the single global peak over ALL frames
    and BOTH arms (so the ideal and non-ideal panels are directly comparable, and
    brightness genuinely evolves as targets move), windowed to `[-db_range, 0]` dB and
    annotated with an explicit colorbar.

    `db_range` defaults to 80 dB, which is wider than a display window usually wants
    (2-4 decades) for a MEASURED reason: on a D2 scene the two pedestrians sit at
    -38.6 and -47.5 dB relative to the animation peak (a near-range vehicle), and the
    receiver noise floor sits at -40.1 dB. A 40 dB window therefore clipped the far
    pedestrian away entirely AND cut off exactly at the noise floor, which rendered
    black -- making the ideal and non-ideal panels look identical and hiding the very
    contrast the pair exists to show. At 80 dB the ideal arm resolves pedestrian-scale
    returns and the non-ideal arm shows the noise floor that buries them. Narrow this
    only after re-measuring the scene's own dynamic range.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dt = 1.0 / float(cfg.frame_rate_hz)
    scats_per_frame, pose_per_frame = _resolve_frames(scenario, cfg, n_frames, dt)
    grid = LabelGrid.for_config(cfg)

    # Pass 1 -- synthesize every frame (both arms) and collect linear power maps, so
    # the dB reference below can be the one global peak. Maps are [n_fft, n_range]
    # float32; even a long animation is a few MB.
    power_real: List[torch.Tensor] = []
    power_ideal: List[torch.Tensor] = []
    sin_az_axis = None
    for k in range(n_frames):
        adc = synthesize_adc(cfg, scats_per_frame[k], pose_per_frame[k],
                             snr_db=snr_db, seed=seed + k)
        p, sin_az_axis = range_azimuth_power(cfg, adc, n_angle_fft=n_angle_fft,
                                            azimuth_window=azimuth_window)
        power_real.append(p)
        if ideal_panel:
            # Same frame, same seed (identical reflection phases), receiver noise off:
            # the only content is the scene itself.
            adc_ideal = synthesize_adc(cfg, scats_per_frame[k], pose_per_frame[k],
                                       snr_db=None, seed=seed + k)
            p_ideal, _ = range_azimuth_power(cfg, adc_ideal, n_angle_fft=n_angle_fft,
                                            azimuth_window=azimuth_window)
            power_ideal.append(p_ideal)

    eps = torch.finfo(torch.float32).tiny
    global_peak = max(float(p.max()) for p in power_real + power_ideal)
    global_peak = max(global_peak, float(eps))

    def _to_db(p: torch.Tensor) -> torch.Tensor:
        return 10.0 * torch.log10((p / global_peak).clamp_min(eps))

    vmin, vmax = -float(db_range), 0.0

    if ideal_panel:
        fig, (ax_bev, ax_ideal, ax_rv) = plt.subplots(1, 3, figsize=(15, 4.2), dpi=dpi)
    else:
        fig, (ax_bev, ax_rv) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=dpi)
        ax_ideal = None
    fig.suptitle(f"{scenario.name}  ({cfg.name})", fontsize=10)

    def _update(k: int):
        scatterers = scats_per_frame[k]
        pose = pose_per_frame[k]
        _draw_birdseye(ax_bev, scatterers, pose, cfg)
        if ax_ideal is not None:
            _draw_radar_view(ax_ideal, cfg, grid, _to_db(power_ideal[k]), sin_az_axis,
                             scatterers, pose, title="Ideal front end: scene only",
                             vmin=vmin, vmax=vmax)
            _draw_radar_view(ax_rv, cfg, grid, _to_db(power_real[k]), sin_az_axis,
                             scatterers, pose, title="Non-ideal front end",
                             vmin=vmin, vmax=vmax)
        else:
            _draw_radar_view(ax_rv, cfg, grid, _to_db(power_real[k]), sin_az_axis,
                             scatterers, pose, vmin=vmin, vmax=vmax)
        return ()

    # ONE layout pass, primed with frame 0, then frozen: tight_layout inside the
    # per-frame loop re-settled the axes as early frames' contents (legends, tick
    # extents) differed, so the subplot sizes visibly shifted over the GIF's first
    # frames. Redraws after this keep these axes positions exactly.
    _update(0)
    # Right margin at 0.96, not 1.0: the colorbar body steals its width from the
    # radar axes, but its tick labels and rotated axis label extend PAST its own
    # axes -- at a full-width layout they clip off the figure's right edge.
    fig.tight_layout(rect=(0, 0, 0.96, 0.94))
    # Colorbar AFTER the layout pass (it steals its space from the radar axes once,
    # then everything is frozen), from a standalone mappable so the per-frame
    # ax.clear() + re-imshow cycle can never orphan it.
    mappable = plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap="inferno")
    radar_axes = [ax for ax in (ax_ideal, ax_rv) if ax is not None]
    fig.colorbar(mappable, ax=radar_axes, fraction=0.03, pad=0.02,
                 label="power (dB, rel. animation peak)")
    # One throwaway draw: the colorbar's own aspect-driven width settles on the FIRST
    # canvas draw (measured 0.019 -> 0.009 figure-fraction), which would otherwise be
    # a visible one-frame width jump at the start of the GIF -- the same class of
    # early-frame jitter the single tight_layout pass above exists to prevent.
    fig.canvas.draw()

    writer = PillowWriter(fps=fps)
    with writer.saving(fig, str(out_path), dpi=dpi):
        for k in range(n_frames):
            _update(k)
            writer.grab_frame()
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------------
# RT tier camera renders (D0-D3, real ray-traced meshes -- for human review, not a
# training-data path). Sionna is imported lazily so this module still imports (and the
# GIF path above still runs) with no Sionna installed; only `--rt`/`render_rt_tier_png`
# need it.
# --------------------------------------------------------------------------------
# Radar marker/boresight-rod colour: bright amber, chosen to stand out against the
# per-class object colours (build_rt_scene's `_default_object_render_color` -- vehicles
# blue, pedestrians cyan, clutter boxes violet, spheres green), the gray ground, and
# Sionna's own (green) device icon -- so the radar reads as unmistakable at a glance.
_RADAR_MARKER_COLOR = (1.0, 0.85, 0.0)
# Raised from 0.8 m after review renders showed no visible marker at all: against a
# 34 m-deep scene containing a 16 m semi-trailer, a 1.6 m sphere is a few pixels. 2.5 m
# reads clearly and is still unmistakably smaller than any vehicle.
_RADAR_MARKER_RADIUS_M = 2.5
_RADAR_BORESIGHT_LEN_M = 10.0           # length of the amber "boresight rod" in front of it
_RENDER_FOV_DEG = 50.0                  # explicit (not Sionna's 45 deg default) horizontal FOV
# Flat per-object framing radius: cheap stand-in for "how big is this thing on screen"
# (a real car/pedestrian/clutter-box bbox half-extent, generously rounded up so even the
# occasional oversized local asset -- e.g. the ~16 m tractor-trailer, see rt_gen's
# LOCAL_ASSET_SPECS -- doesn't get clipped at the frame edge).
_OBJECT_FRAMING_RADIUS_M = 9.0
# Slack multiplied onto the fitted camera distance, so the outermost framed point sits
# just inside the frame rather than exactly on its edge. `render_rt_tier_png(zoom=...)`
# divides this; it is named rather than inlined so the two stay in one place.
_RENDER_CAMERA_MARGIN = 1.2

# Overhead ("top-down") camera direction: pass this as `camera_dir` for a true
# vertical view (see `_build_camera`'s docstring for why straight-down needs its own
# code path, not just a near-vertical `camera_dir` value).
TOP_DOWN_CAMERA_DIR = (0.0, 0.0, 1.0)
# Fixed margin ADDED to each object's real footprint radius (rt_scenes._footprint_
# radius) when framing render_rt_topdown_gif's camera -- covers mesh detail outside
# the placement footprint (mirrors, limbs) without reintroducing the flat, oversized
# _OBJECT_FRAMING_RADIUS_M that made pedestrians unreadable in a top-down view.
_TOPDOWN_FRAMING_MARGIN_M = 1.0
# Pedestrian "flag" marker: even with a tight per-object framing radius (above), a
# camera fit to a whole D2/D3-sized scene (targets spread over the tier's full 6-34 m
# range envelope, see rt_scenes._RANGE_M) necessarily shows a real ~0.8 m pedestrian
# mesh as only a few pixels -- there is no framing choice that fixes this without
# clipping every farther-out vehicle/clutter box out of frame. Same fix already used
# for the radar itself (see _RADAR_MARKER_COLOR's own "no visible marker at all"
# note): a small, unmistakable, render-only sphere floating just above the
# pedestrian's head, same colour as its legend swatch. An ordinary (non-scattering-
# tagged) SceneObject, exactly like the radar marker -- never touches a path solve.
_PEDESTRIAN_FLAG_RADIUS_M = 0.5
_PEDESTRIAN_FLAG_HEIGHT_M = 2.3         # just above a ~1.74 m pedestrian mesh's head
# `_fit_camera_position`'s own threshold for swapping its local `world_up` helper axis
# (avoiding ITS degenerate cross product) -- reused here as the trigger for routing
# camera construction through the explicit-orientation path instead of `Camera.look_at`
# (see `_build_camera`), so both "near vertical" checks in this module agree.
_NEAR_VERTICAL_DOT = 0.98
# World +y -> screen "up", world +x -> screen "right" for the explicit top-down camera
# orientation below -- matches `_draw_birdseye`'s (and every other bird's-eye panel's)
# plot convention (x right, y up), calibrated empirically against Sionna's own
# `Camera.orientation`/`.look_at()` (see tests/test_ml_render.py's calibration test),
# not guessed.
_TOP_DOWN_YAW_RAD = math.pi / 2.0


def _fit_camera_position(points: np.ndarray, radii: np.ndarray, *, camera_dir,
                         fov_deg: float, aspect: float,
                         margin: float = _RENDER_CAMERA_MARGIN):
    """`(camera_position, look_at)` that keeps every `points[i]` (inflated by
    `radii[i]`) inside a `fov_deg`-wide (horizontal, Mitsuba `fov_axis="x"` convention)
    camera looking along `-camera_dir` at the points' centroid, positioned back along
    `camera_dir` from it. Fixes the earlier bare "2.2x bounding-radius" heuristic, which
    could still leave an off-axis point (typically the radar, since objects cluster in
    the boresight direction while the radar sits at the FOV's edge of that cluster)
    right at -- or past -- the frame edge: this fits the actual horizontal AND vertical
    half-angle needed, not just an isotropic bounding-sphere distance.
    """
    centroid = points.mean(axis=0)
    forward = -np.asarray(camera_dir, dtype=float)
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(forward, world_up))) > _NEAR_VERTICAL_DOT:  # near-vertical view:
        world_up = np.array([0.0, 1.0, 0.0])              # avoid a degenerate cross product
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    rel = points - centroid
    half_h = np.abs(rel @ right) + radii
    half_v = np.abs(rel @ up) + radii

    fov_h = math.radians(float(fov_deg))
    fov_v = 2.0 * math.atan(math.tan(fov_h / 2.0) / float(aspect))
    dist_h = float(half_h.max()) / math.tan(fov_h / 2.0)
    dist_v = float(half_v.max()) / math.tan(fov_v / 2.0)
    dist = max(dist_h, dist_v, 1.0) * float(margin)

    cam_pos = centroid - forward * dist
    return cam_pos, centroid


def _build_camera(cam_pos: np.ndarray, look_at: np.ndarray, rt):
    """`rt.Camera` at `cam_pos` looking at `look_at`, handling the near-vertical case
    explicitly instead of leaning on Sionna's own `Camera.look_at()`.

    `Camera.look_at()` derives yaw as `atan2(target.y - cam.y, target.x - cam.x)`
    (Sionna's own docstring: "Given a point with spherical angles theta, phi, the
    orientation will be set to (phi, pi/2 - theta, 0)"), which is mathematically
    undefined once that horizontal offset is (near) zero -- i.e. exactly the top-down
    case this module needs for `TOP_DOWN_CAMERA_DIR`. Sionna's own fallback (see its
    source) only nudges the target by a fixed, ABSOLUTE `1e-3` in x, and only when
    position/target match EXACTLY in x/y; a `camera_dir` that is near- but not
    exactly-vertical skips that fallback entirely and instead derives yaw from a
    near-zero denominator -- an unpredictable function of floating-point noise, so the
    image's "up" direction (and hence the whole frame's rotation) is effectively
    undefined and can differ between otherwise-identical runs/platforms.

    For `|forward.z| > _NEAR_VERTICAL_DOT` this instead sets the camera's `orientation`
    (yaw, pitch, roll) Euler angles directly -- `Camera`'s OTHER public constructor path,
    which bypasses `look_at()`'s spherical-angle derivation entirely. Pitch is the exact
    (deterministic) value for straight-down/-up (+-90 deg); yaw is the explicit, chosen
    constant `_TOP_DOWN_YAW_RAD` rather than whatever Sionna's internal epsilon nudge
    happens to produce. These angle/axis conventions were calibrated empirically against
    Sionna's own `Camera.orientation`/`.look_at()` output (not guessed or reasoned from
    docs alone) -- see `tests/test_ml_render.py`'s calibration-derived tests.
    """
    cam_pos = np.asarray(cam_pos, dtype=float)
    look_at = np.asarray(look_at, dtype=float)
    delta = look_at - cam_pos
    norm = float(np.linalg.norm(delta))
    forward = delta / norm if norm > 0.0 else np.array([0.0, 0.0, -1.0])
    if abs(float(forward[2])) > _NEAR_VERTICAL_DOT:
        pitch = math.pi / 2.0 if forward[2] < 0.0 else -math.pi / 2.0
        return rt.Camera(position=cam_pos.tolist(),
                         orientation=(_TOP_DOWN_YAW_RAD, pitch, 0.0))
    return rt.Camera(position=cam_pos.tolist(), look_at=look_at.tolist())


def _add_pedestrian_flags(rt, rt_scene, scenario, positions) -> List[Any]:
    """Add a small, bright `_OBJECT_COLOR_PEDESTRIAN` sphere marker floating just
    above every `object_class="pedestrian"` object in `scenario.objects`, at the
    matching (x, y) from `positions` (parallel list -- callers pass either each
    object's own static `.position` for a single-frame render or a resolved
    `frame_scatterers(...)` position per animation frame), added to `rt_scene.scene`.
    Render-only visibility fix -- see `_PEDESTRIAN_FLAG_RADIUS_M`'s docstring for why a
    full-scene framing can't make a real pedestrian mesh legible on its own. Returns
    the added `SceneObject`s (unused by callers today).
    """
    from e2e.ml.rt_scene_build import _OBJECT_COLOR_PEDESTRIAN

    added = []
    for obj, pos in zip(scenario.objects, positions):
        if obj.object_class != "pedestrian":
            continue
        mat = rt.ITURadioMaterial(f"e2e-ped-flag-mat-{obj.name}", "metal", thickness=0.01,
                                  color=_OBJECT_COLOR_PEDESTRIAN)
        flag = rt.SceneObject(fname=rt.scene.sphere, name=f"e2e-ped-flag-{obj.name}",
                              radio_material=mat)
        rt_scene.scene.edit(add=[flag])
        flag.scaling = float(_PEDESTRIAN_FLAG_RADIUS_M)
        flag.position = [float(pos[0]), float(pos[1]), _PEDESTRIAN_FLAG_HEIGHT_M]
        added.append(flag)
    return added


def _build_rt_scene_for_render(scenario, cfg, *, frame_idx: int = 0,
                               material_policy: str = "extrapolated",
                               stand_in_material: str = "concrete"):
    """`build_rt_scene(scenario, cfg, frame_idx=frame_idx, ...)`, repairing city-scene
    materials for `cfg`'s centre frequency first if `scenario.base_scene` needs it (see
    `render_rt_tier_png`'s "City-scene materials" docstring section for why). Split out
    from `render_rt_tier_png` so the branch is unit-testable without Sionna -- see
    `tests/test_ml_render.py::test_build_rt_scene_for_render_*`.

    `frame_idx` selects a MOTION frame within `scenario` (its own `Scenario.num_frames`
    /`Motion` tracks), not a different scenario draw -- see `render_rt_topdown_gif`,
    which builds one `scenario` with `num_frames > 1` and calls this once per animation
    frame so objects actually move across the render, unlike `render_rt_tier_png`
    (single-frame scenario, always `frame_idx=0`).
    """
    from e2e.ml.rt_gen import build_rt_scene

    if scenario.base_scene in ("flat", "free"):
        # Synthetic scenes only ever use in-band materials (see
        # `e2e.ml.rt_gen._GROUND_MATERIAL`) -- unmodified, matching
        # `RTEnvironmentBlock.get_S_pars`'s same no-op branch.
        return build_rt_scene(scenario, cfg, base_scene=scenario.base_scene,
                              frame_idx=frame_idx)

    # A Sionna built-in city scene (e.g. D4's "munich"): repair its materials for this
    # radar's frequency one Python frame before `build_rt_scene` sets `scene.frequency`
    # -- see `city_scenes.patched_builtin_loader`'s own docstring for why this is a
    # monkeypatch rather than a direct call.
    from e2e.environment.city_scenes import patched_builtin_loader

    f_center_hz = float(cfg.f0_hz) + float(cfg.bandwidth_hz) / 2.0
    with patched_builtin_loader(f_center_hz, policy=material_policy,
                                stand_in_itu_type=stand_in_material):
        return build_rt_scene(scenario, cfg, base_scene=scenario.base_scene,
                              frame_idx=frame_idx)


#: Corpus tag used by every RENDER path in this module.
#:
#: `rt_scenes._stable_seed` salts the scene-determinism key with a corpus identity so two
#: corpora sharing a numeric seed cannot draw the same scenes. Renders are figures, not
#: corpora -- they illustrate what a tier looks like rather than reproducing a specific
#: dataset -- so they get one fixed, documented tag instead of inventing one per call.
#:
#: Consequence worth knowing: a render will NOT reproduce the scene of a corpus frame
#: unless you pass that corpus's own tag (`--corpus-tag`). If you want to picture a
#: specific stored frame, read its tag out of the corpus manifest.
RENDER_CORPUS_TAG = "render"


def render_rt_tier_png(tier, out_path, *, cfg=None, frame_idx: int = 0, seed: int = 0,
                       corpus_tag: str = RENDER_CORPUS_TAG,
                       resolution=(1280, 720), camera_dir=(-1.0, -1.0, 1.15),
                       num_samples: int = 128, use_local_assets: bool = True,
                       caption: bool = True, material_policy: str = "extrapolated",
                       stand_in_material: str = "concrete", zoom: float = 1.0):
    """Ray-trace `e2e.ml.rt_scenes` tier `tier` and save a camera render (array +
    objects) as a PNG at `out_path`. Needs Sionna RT; this is a plain geometry render
    (no path solve -- `Scene.render_to_file` needs no `PathSolver` output unless a
    `paths=` overlay is requested, which this does not do), so it is comparatively
    cheap. Purely for human-eyes review of the object meshes/environment, not part of
    any training-data pipeline.

    `cfg` (a `RadarConfig`, default `e2e.ml.radar_config.PRESETS["radial_like"]`) only
    sets the radar's array/frequency for scene construction (`rt_gen.build_rt_scene`);
    it plays no role in the image itself. The camera auto-frames the radar node, the
    radar's own amber marker/boresight-rod (see below) AND every object: `_fit_camera_
    position` fits the actual `fov_deg`-wide horizontal/vertical half-angles needed to
    keep every (radius-inflated) point in frame, looking along `camera_dir` (default
    behind-and-above), so tiers with different object counts/spreads (D0's single close
    sphere vs D3's dozen-plus spread-out objects) both stay fully in frame -- including
    the radar itself, which otherwise tends to sit at the edge of the object cluster's
    field of view -- without per-tier tuning. `resolution` is `(width, height)` pixels.
    Pass `camera_dir=TOP_DOWN_CAMERA_DIR` for a true overhead view -- `_build_camera`
    routes any near-vertical `camera_dir` through an explicit-orientation construction
    instead of `Camera.look_at()`, which is genuinely ill-conditioned in that regime
    (see that function's docstring).

    The radar position is marked TWICE, redundantly, so it can't be missed: Sionna's
    own tx/rx device icon (pinned to a fixed, legible `display_radius` -- its default
    auto-sizes from the scene's own bounding box, which for the 400 m "flat" ground
    plane makes it a 2 m sphere that still gets lost at review-camera distance) PLUS an
    explicit amber marker sphere + a thin amber "boresight rod" pointing along the
    radar's look direction, added as ordinary (non-scattering-tagged) `SceneObject`s
    purely for this render -- this function never re-uses `rt_scene` for a path solve,
    so decorating it has no effect on any simulated return.

    `use_local_assets` (default True, unlike `build_rt_tier_scenario`'s own default)
    draws cars/pedestrians from the expanded local-mesh pool when available on this
    machine (see `e2e.ml.rt_gen`'s module docstring); it degrades gracefully to the
    Sionna-bundled meshes elsewhere. `caption` overlays a title bar (tier, object
    counts, radar position) via Pillow if installed; silently skipped otherwise.

    City-scene materials: when the tier's `base_scene` is not `"flat"`/`"free"` (e.g.
    the D4 "munich" tier), the base scene is loaded through
    `e2e.environment.city_scenes.patched_builtin_loader` before `build_rt_scene` sets
    `scene.frequency` -- otherwise an out-of-band ITU material (munich's `marble`/
    `brick` above their tabulated range, e.g. at automotive/mmWave 77 GHz) hard-raises
    inside Sionna and the render never happens. Mirrors
    `e2e.environment.blocks.RTEnvironmentBlock.get_S_pars`'s same guard, so `"flat"`/
    `"free"` tiers (which only ever use in-band materials, see
    `e2e.ml.rt_gen._GROUND_MATERIAL`) are unaffected. `material_policy`/
    `stand_in_material` are that module's substitution knobs.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import sionna.rt as rt

    from e2e.radar_config import PRESETS
    from e2e.ml.rt_gen import _box_mesh_path
    from e2e.ml.rt_scenes import build_rt_tier_scenario, tier_summary

    if cfg is None:
        cfg = PRESETS["radial_like"]

    scenario = build_rt_tier_scenario(tier, corpus_tag=corpus_tag, frame_idx=frame_idx,
                                      seed=seed, num_frames=1,
                                      use_local_assets=use_local_assets)
    rt_scene = _build_rt_scene_for_render(scenario, cfg, material_policy=material_policy,
                                          stand_in_material=stand_in_material)

    rt_scene.tx.display_radius = 0.3
    rt_scene.rx.display_radius = 0.3

    radar_node = scenario.nodes[0]
    radar_pos = np.asarray(radar_node.position, dtype=float)
    look_at = (np.asarray(radar_node.look_at, dtype=float) if radar_node.look_at is not None
              else radar_pos + np.array([1.0, 0.0, 0.0]))
    boresight = look_at - radar_pos
    boresight = boresight / np.linalg.norm(boresight)

    marker_mat = rt.ITURadioMaterial("e2e-radar-marker-mat", "metal", thickness=0.01,
                                     color=_RADAR_MARKER_COLOR)
    marker = rt.SceneObject(fname=rt.scene.sphere, name="e2e-radar-marker", radio_material=marker_mat)
    rt_scene.scene.edit(add=[marker])
    marker.scaling = float(_RADAR_MARKER_RADIUS_M)   # rt.scene.sphere is a unit (r~1 m) sphere
    marker.position = radar_pos.tolist()

    # Boresight rod: Sionna's box mesh is a 10x10x5 m primitive (see rt_gen._box_mesh_path)
    # -- non-uniform scaling shrinks it to a thin (0.3 x 0.15 m cross-section) rod along
    # its OWN local x axis. This package's radar boresight is always world +x (see
    # rt_scenes' module docstring), which is exactly the box's unrotated local x, so no
    # orientation transform is needed here -- this is specific to this pipeline's
    # convention, not a general-purpose gizmo.
    rod_mat = rt.ITURadioMaterial("e2e-radar-boresight-mat", "metal", thickness=0.01,
                                  color=_RADAR_MARKER_COLOR)
    rod = rt.SceneObject(fname=_box_mesh_path(rt), name="e2e-radar-boresight", radio_material=rod_mat)
    rt_scene.scene.edit(add=[rod])
    rod.scaling = (_RADAR_BORESIGHT_LEN_M / 10.0, 0.08, 0.08)
    rod.position = (radar_pos + boresight * _RADAR_BORESIGHT_LEN_M / 2.0).tolist()

    _add_pedestrian_flags(rt, rt_scene, scenario, [o.position for o in scenario.objects])

    # Each framed point carries an approximate world-space RADIUS (not just a bare
    # position) so the fit accounts for how big things actually are on screen -- a
    # bare-centroid distance heuristic put the (small-icon) radar right at the frame
    # edge in earlier renders; this is the actual fix, the marker/rod above is what
    # makes it worth getting right.
    points = [radar_pos, radar_pos + boresight * _RADAR_BORESIGHT_LEN_M]
    radii = [float(_RADAR_MARKER_RADIUS_M) * 1.1, 0.3]
    for o in scenario.objects:
        points.append(np.asarray(o.position, dtype=float))
        radii.append(_OBJECT_FRAMING_RADIUS_M)
    # `zoom` > 1 pulls the camera in by shrinking the fit margin. The fit above frames
    # the radar marker AND every object, and the radar sits at the edge of the object
    # cluster rather than inside it, so the automatic framing is necessarily wide --
    # correct for an overview, too wide to read a pedestrian. Zoom trades guaranteed
    # "everything is in frame" for legibility, so it is opt-in and defaults to 1.0
    # (unchanged framing); at zoom > ~1.4 the radar marker starts leaving the frame.
    if not zoom > 0.0:
        raise ValueError(f"zoom must be > 0, got {zoom}")
    cam_pos, centroid = _fit_camera_position(np.asarray(points, dtype=float),
                                             np.asarray(radii, dtype=float),
                                             camera_dir=camera_dir, fov_deg=_RENDER_FOV_DEG,
                                             aspect=resolution[0] / resolution[1],
                                             margin=_RENDER_CAMERA_MARGIN / float(zoom))
    camera = _build_camera(cam_pos, centroid, rt)

    rt_scene.scene.render_to_file(camera=camera, filename=str(out_path),
                                  resolution=tuple(resolution), num_samples=num_samples,
                                  fov=_RENDER_FOV_DEG, show_devices=True)

    if caption:
        summary = tier_summary(scenario)
        _caption_render(
            out_path,
            [
                f"{scenario.name}   tier={summary['tier']}   base_scene={summary['base_scene']}",
                f"objects: spheres={summary['n_spheres']}  "
                f"vehicles={summary['n_cars']} {summary['n_vehicles_by_class']}  "
                f"pedestrians={summary['n_pedestrians']}  clutter_boxes={summary['n_clutter_boxes']}",
                f"radar @ ({radar_pos[0]:.1f}, {radar_pos[1]:.1f}, {radar_pos[2]:.1f}) m "
                "-- amber marker + boresight rod",
            ],
        )
    return out_path


def _caption_render(png_path: Path, lines: List[str]) -> None:
    """Overlay a semi-transparent title bar with `lines` of text on a saved PNG.
    Best-effort: silently does nothing if Pillow isn't installed (matplotlib does not
    hard-depend on it in every backend configuration)."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return

    img = Image.open(png_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    pad, line_h = 8, 18
    bar_h = 2 * pad + line_h * len(lines)
    draw.rectangle([0, 0, img.width, bar_h], fill=(0, 0, 0, 170))
    for i, line in enumerate(lines):
        draw.text((pad, pad + i * line_h), line, fill=(255, 255, 255, 255))
    img.save(png_path)


def _overlay_legend(img, entries: Sequence[Tuple[str, Tuple[float, float, float]]], *,
                    title: Optional[str] = None) -> None:
    """Draw a colour-key legend (`[(label, rgb_0_1), ...]`) as a semi-transparent box in
    the bottom-left corner of a PIL RGB `img`, IN PLACE, plus an optional one-line
    `title` bar across the top (frame counter / scene name) -- same visual language as
    `_caption_render`'s title bar. A hard `PIL` dependency for callers (unlike
    `_caption_render`'s best-effort try/except): a viewer cannot tell a pedestrian from
    clutter in an un-labelled render, so this is not optional for `render_rt_topdown_gif`.
    """
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img, "RGBA")
    if title:
        draw.rectangle([0, 0, img.width, 22], fill=(0, 0, 0, 170))
        draw.text((8, 4), title, fill=(255, 255, 255, 255))

    pad, line_h, swatch = 8, 20, 14
    box_w = 150
    box_h = 2 * pad + line_h * len(entries)
    x0, y0 = 8, img.height - box_h - 8
    draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=(0, 0, 0, 170))
    for i, (label, rgb01) in enumerate(entries):
        rgb = tuple(int(round(255 * float(c))) for c in rgb01)
        cy = y0 + pad + i * line_h
        draw.rectangle([x0 + pad, cy + 2, x0 + pad + swatch, cy + 2 + swatch], fill=rgb + (255,))
        draw.text((x0 + pad + swatch + 6, cy), label, fill=(255, 255, 255, 255))


def _render_rt_topdown_frames(scenario, cfg, *, n_frames: int, dt: float,
                              resolution=(640, 480), num_samples: int = 64,
                              use_local_assets: bool = True,
                              material_policy: str = "extrapolated",
                              stand_in_material: str = "concrete",
                              caption_frames: bool = True) -> List[Any]:
    """Ray-trace `n_frames` TOP-DOWN camera frames of an ALREADY-BUILT `scenario`, at
    successive motion steps `frame_idx=0..n_frames-1`, under ONE fixed camera fit to the
    union of every frame's object positions. This is `render_rt_topdown_gif`'s own
    per-frame rendering loop/camera-fit logic, split out so a caller that already owns a
    `scenario` (e.g. `render_scene_gif_2x2`, which must share ITS `scenario`/frame
    indices with the analytic bird's-eye/range-azimuth panels) can drive the same
    machinery without an independent scenario draw. `render_rt_topdown_gif` itself calls
    this too, unchanged in behaviour. Needs Sionna RT (imported lazily). Returns a list
    of `PIL.Image.Image` (RGB), NOT saved to a file. `caption_frames=False` still draws
    the colour-key legend but drops the per-frame "top-down frame k/n" title bar, for a
    caller that embeds each frame inside its own titled subplot.
    """
    import sionna.rt as rt
    from PIL import Image

    from e2e.ml.rt_gen import _box_mesh_path
    from e2e.ml.rt_scene_build import (_OBJECT_COLOR_CLUTTER_BOX, _OBJECT_COLOR_PEDESTRIAN,
                                       _OBJECT_COLOR_SPHERE, _OBJECT_COLOR_VEHICLE)
    from e2e.environment.scatterers import frame_scatterers
    from e2e.ml.rt_scenes import _footprint_radius, tier_summary
    from e2e.scenario import ObjectKind

    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")

    radar_node = scenario.nodes[0]
    radar_pos = np.asarray(radar_node.position, dtype=float)
    look_at = (np.asarray(radar_node.look_at, dtype=float) if radar_node.look_at is not None
              else radar_pos + np.array([1.0, 0.0, 0.0]))
    boresight = look_at - radar_pos
    boresight = boresight / np.linalg.norm(boresight)

    # Camera framing: union of EVERY animation frame's object positions (not just
    # frame 0), so the fixed camera never clips a target that has drifted by the time
    # the GIF ends -- see the docstring. Framing radius is each object's REAL footprint
    # (`rt_scenes._footprint_radius`, the same half-length used for placement/overlap
    # checks) plus a small fixed margin -- NOT `render_rt_tier_png`'s flat
    # `_OBJECT_FRAMING_RADIUS_M` (9 m, sized so an oblique 3D view never clips the
    # largest local-asset vehicle). A top-down view has no perspective foreshortening
    # to hide behind: a uniform 9 m radius per object zooms the camera out far enough
    # that a ~0.8 m pedestrian became a few-pixel dot in review renders -- exactly the
    # "can't really see pedestrians" feedback this GIF exists to fix.
    points = [radar_pos, radar_pos + boresight * _RADAR_BORESIGHT_LEN_M]
    radii = [float(_RADAR_MARKER_RADIUS_M) * 1.1, 0.3]
    for k in range(n_frames):
        for obj, sc in zip(scenario.objects, frame_scatterers(scenario, k, dt=dt)):
            points.append(np.asarray(sc.position, dtype=float))
            # D0's sphere stand-in targets are tagged `object_class="vehicle"` for
            # RCS/dataset purposes (see rt_scenes.build_rt_tier_scenario) but are
            # sphere-sized, not car-sized -- `build_rt_tier_scenario`'s OWN placement
            # code special-cases this the same way (`_footprint("sphere")`, not
            # `_footprint(obj.object_class)`, for its sphere loop); match that here so
            # a D0/D2 sphere doesn't inflate the frame to car scale.
            framing_class = "sphere" if obj.kind == ObjectKind.SPHERE else obj.object_class
            radii.append(_footprint_radius(framing_class, obj.asset)
                        + _TOPDOWN_FRAMING_MARGIN_M)
    cam_pos, centroid = _fit_camera_position(np.asarray(points, dtype=float),
                                             np.asarray(radii, dtype=float),
                                             camera_dir=TOP_DOWN_CAMERA_DIR,
                                             fov_deg=_RENDER_FOV_DEG,
                                             aspect=resolution[0] / resolution[1])
    camera = _build_camera(cam_pos, centroid, rt)

    summary = tier_summary(scenario)
    legend_entries: List[Tuple[str, Tuple[float, float, float]]] = []
    if summary["n_spheres"] > 0:
        legend_entries.append(("sphere", _OBJECT_COLOR_SPHERE))
    if summary["n_cars"] > 0:
        legend_entries.append(("vehicle", _OBJECT_COLOR_VEHICLE))
    if summary["n_pedestrians"] > 0:
        legend_entries.append(("pedestrian", _OBJECT_COLOR_PEDESTRIAN))
    if summary["n_clutter_boxes"] > 0:
        legend_entries.append(("clutter box", _OBJECT_COLOR_CLUTTER_BOX))
    legend_entries.append(("radar", _RADAR_MARKER_COLOR))

    frames: List["Image.Image"] = []
    with tempfile.TemporaryDirectory(prefix="e2e-rt-topdown-") as tmp_dir:
        for k in range(n_frames):
            # Rebuilt fresh every animation frame (build_rt_scene loads a brand-new
            # Sionna Scene each call -- see its own docstring), same as
            # render_rt_tier_png's single-frame case; the marker/rod below are ordinary
            # SceneObjects re-added to THIS frame's scene, never affecting a path solve.
            rt_scene = _build_rt_scene_for_render(scenario, cfg, frame_idx=k,
                                                  material_policy=material_policy,
                                                  stand_in_material=stand_in_material)
            rt_scene.tx.display_radius = 0.3
            rt_scene.rx.display_radius = 0.3

            marker_mat = rt.ITURadioMaterial("e2e-radar-marker-mat", "metal", thickness=0.01,
                                             color=_RADAR_MARKER_COLOR)
            marker = rt.SceneObject(fname=rt.scene.sphere, name="e2e-radar-marker",
                                    radio_material=marker_mat)
            rt_scene.scene.edit(add=[marker])
            marker.scaling = float(_RADAR_MARKER_RADIUS_M)
            marker.position = radar_pos.tolist()

            rod_mat = rt.ITURadioMaterial("e2e-radar-boresight-mat", "metal", thickness=0.01,
                                          color=_RADAR_MARKER_COLOR)
            rod = rt.SceneObject(fname=_box_mesh_path(rt), name="e2e-radar-boresight",
                                 radio_material=rod_mat)
            rt_scene.scene.edit(add=[rod])
            rod.scaling = (_RADAR_BORESIGHT_LEN_M / 10.0, 0.08, 0.08)
            rod.position = (radar_pos + boresight * _RADAR_BORESIGHT_LEN_M / 2.0).tolist()

            frame_scats = frame_scatterers(scenario, k, dt=dt)
            _add_pedestrian_flags(rt, rt_scene, scenario, [sc.position for sc in frame_scats])

            frame_path = os.path.join(tmp_dir, f"frame_{k:03d}.png")
            rt_scene.scene.render_to_file(camera=camera, filename=frame_path,
                                          resolution=tuple(resolution), num_samples=num_samples,
                                          fov=_RENDER_FOV_DEG, show_devices=True)
            img = Image.open(frame_path).convert("RGB")
            title = f"{scenario.name}  top-down  frame {k + 1}/{n_frames}" if caption_frames else None
            _overlay_legend(img, legend_entries, title=title)
            frames.append(img)

    return frames


def render_rt_topdown_gif(tier, out_path, *, cfg=None, n_frames: int = 20, fps: int = 8,
                          frame_idx: int = 0, seed: int = 0,
                          corpus_tag: str = RENDER_CORPUS_TAG, resolution=(640, 480),
                          num_samples: int = 64, use_local_assets: bool = True,
                          material_policy: str = "extrapolated",
                          stand_in_material: str = "concrete") -> Path:
    """Ray-trace a TOP-DOWN animated GIF of `e2e.ml.rt_scenes` tier `tier`: ONE scenario,
    objects moving over TIME. Needs Sionna RT; each animation frame is a plain geometry
    render (no path solve, same as `render_rt_tier_png`), so an `n_frames`-frame GIF
    costs about `n_frames` times one PNG render.

    Do not confuse this module's two different `frame_idx` meanings: here (as in
    `render_rt_tier_png`) `frame_idx`/`seed` select which scenario is DRAWN (see
    `build_rt_tier_scenario`'s "Determinism" section) -- a different concept from the
    per-animation-frame `frame_idx` that `_build_rt_scene_for_render`/`build_rt_scene`
    take internally, once per k in `range(n_frames)` (inside `_render_rt_topdown_frames`),
    to place objects at that motion step of the SAME drawn scenario.

    `n_frames` is resolved via `build_rt_tier_scenario(..., num_frames=n_frames,
    dt=1/cfg.frame_rate_hz)` -- passing `dt` here is load-bearing: `Motion.velocity` is
    stored as a per-frame displacement, so the function's own `dt=1.0` default would
    move every object `cfg.frame_rate_hz` times too far per animation frame (the exact
    bug fixed in commit c212076, "RT targets ... moved 10x too fast when they moved").

    The camera fit, per-frame render loop, and legend overlay are
    `_render_rt_topdown_frames` (see that function for the camera/legend details); this
    wrapper only draws the scenario and saves the returned frames to a GIF.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from e2e.radar_config import PRESETS
    from e2e.ml.rt_scenes import build_rt_tier_scenario

    if cfg is None:
        cfg = PRESETS["radial_like"]
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")

    dt = 1.0 / float(cfg.frame_rate_hz)
    scenario = build_rt_tier_scenario(tier, corpus_tag=corpus_tag, frame_idx=frame_idx,
                                      seed=seed,
                                      num_frames=n_frames, dt=dt,
                                      use_local_assets=use_local_assets)

    frames = _render_rt_topdown_frames(
        scenario, cfg, n_frames=n_frames, dt=dt, resolution=resolution,
        num_samples=num_samples, use_local_assets=use_local_assets,
        material_policy=material_policy, stand_in_material=stand_in_material,
    )

    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=int(round(1000.0 / fps)), loop=0, optimize=True)
    return out_path


# --------------------------------------------------------------------------------
# 2x2 combined GIF: analytic bird's-eye + REAL RT top-down render + ideal/non-ideal
# range-azimuth maps, all four panels driven from ONE scenario/frame index.
# --------------------------------------------------------------------------------
def render_scene_gif_2x2(cfg, scenario, out_path, *, n_frames: int = 10, fps: int = 8,
                         seed: int = 0, snr_db: Optional[float] = 30.0, dpi: int = 90,
                         n_angle_fft: Optional[int] = None, db_range: float = 80.0,
                         azimuth_window: Optional[str] = "hann",
                         rt_resolution=(640, 480), rt_num_samples: int = 64,
                         use_local_assets: bool = False,
                         material_policy: str = "extrapolated",
                         stand_in_material: str = "concrete") -> Path:
    """2x2 layout: bird's-eye (top-left) | RT top-down camera render (top-right) |
    ideal-front-end range-azimuth map (bottom-left) | non-ideal range-azimuth map
    (bottom-right). Needs Sionna RT (imported lazily, via `_render_rt_topdown_frames`)
    AND a `scenario` with REAL geometry -- i.e. NOT `e2e.ml.scenes.sample_scene`'s
    `base_scene="synthetic"` point-target scenes, which have no mesh for a camera to
    render (see `e2e.environment.scatterers.SYNTHETIC_BASE_SCENE`). Build `scenario` with
    `e2e.ml.rt_scenes.build_rt_tier_scenario(tier, num_frames=n_frames,
    dt=1/cfg.frame_rate_hz, ...)` instead -- passing `dt` there is load-bearing (see that
    function's own docstring: omitting it inflates every velocity `frame_rate_hz`-fold).

    SAME-SCENE GUARANTEE: every panel is driven from this one `scenario` object and the
    same per-animation-frame index `k` -- nothing here draws an independent scenario.
    The analytic panels use `_resolve_frames`/`frame_scatterers`/`radar_pose`, exactly
    `render_scene_gif`'s own machinery; the RT panel uses `_render_rt_topdown_frames`
    (`render_rt_topdown_gif`'s own per-frame camera-fit/render loop), called with this
    same `scenario` and `n_frames` rather than letting it draw one internally.

    Color scale: identical convention to `render_scene_gif` -- ONE global dB reference
    over every frame and BOTH radar panels (see that function's docstring for the
    measured 80 dB `db_range` rationale). The RT panel is a plain RGB camera image, so
    it has no dB scale of its own and is not part of that normalization.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dt = 1.0 / float(cfg.frame_rate_hz)

    # RT render pass first -- one heavy Sionna job, driven by the SAME scenario/frame
    # indices the analytic passes below use (see docstring's same-scene guarantee).
    rt_frames = _render_rt_topdown_frames(
        scenario, cfg, n_frames=n_frames, dt=dt, resolution=rt_resolution,
        num_samples=rt_num_samples, use_local_assets=use_local_assets,
        material_policy=material_policy, stand_in_material=stand_in_material,
        caption_frames=False,
    )
    rt_arrays = [np.asarray(im) for im in rt_frames]

    scats_per_frame, pose_per_frame = _resolve_frames(scenario, cfg, n_frames, dt)
    grid = LabelGrid.for_config(cfg)

    # Pass over every frame -- same ideal/non-ideal synthesis + global-peak dB reference
    # as render_scene_gif (see that function for why this is a global, not per-frame,
    # reference).
    power_real: List[torch.Tensor] = []
    power_ideal: List[torch.Tensor] = []
    sin_az_axis = None
    for k in range(n_frames):
        adc = synthesize_adc(cfg, scats_per_frame[k], pose_per_frame[k],
                             snr_db=snr_db, seed=seed + k)
        p, sin_az_axis = range_azimuth_power(cfg, adc, n_angle_fft=n_angle_fft,
                                            azimuth_window=azimuth_window)
        power_real.append(p)
        adc_ideal = synthesize_adc(cfg, scats_per_frame[k], pose_per_frame[k],
                                   snr_db=None, seed=seed + k)
        p_ideal, _ = range_azimuth_power(cfg, adc_ideal, n_angle_fft=n_angle_fft,
                                        azimuth_window=azimuth_window)
        power_ideal.append(p_ideal)

    eps = torch.finfo(torch.float32).tiny
    global_peak = max(float(p.max()) for p in power_real + power_ideal)
    global_peak = max(global_peak, float(eps))

    def _to_db(p: torch.Tensor) -> torch.Tensor:
        return 10.0 * torch.log10((p / global_peak).clamp_min(eps))

    vmin, vmax = -float(db_range), 0.0

    fig, ((ax_bev, ax_rt), (ax_ideal, ax_rv)) = plt.subplots(2, 2, figsize=(12, 9), dpi=dpi)
    fig.suptitle(f"{scenario.name}  ({cfg.name})", fontsize=10)

    def _update(k: int):
        scatterers = scats_per_frame[k]
        pose = pose_per_frame[k]
        _draw_birdseye(ax_bev, scatterers, pose, cfg)

        ax_rt.clear()
        # aspect="auto" fills the (fixed) subplot cell exactly -- avoids letterboxing
        # when the RT render's pixel aspect ratio doesn't exactly match the cell's.
        ax_rt.imshow(rt_arrays[k], aspect="auto")
        ax_rt.set_title("RT top-down render (real Sionna RT geometry)")
        ax_rt.set_xticks([])
        ax_rt.set_yticks([])

        _draw_radar_view(ax_ideal, cfg, grid, _to_db(power_ideal[k]), sin_az_axis,
                         scatterers, pose, title="Ideal front end: scene only",
                         vmin=vmin, vmax=vmax)
        _draw_radar_view(ax_rv, cfg, grid, _to_db(power_real[k]), sin_az_axis,
                         scatterers, pose, title="Non-ideal front end",
                         vmin=vmin, vmax=vmax)
        return ()

    # Same one-layout-pass-then-freeze trick as render_scene_gif (see its comment) --
    # keeps every grabbed frame's axes positions identical.
    _update(0)
    fig.tight_layout(rect=(0, 0, 0.96, 0.94))
    mappable = plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap="inferno")
    fig.colorbar(mappable, ax=[ax_ideal, ax_rv], fraction=0.03, pad=0.02,
                 label="power (dB, rel. animation peak)")
    fig.canvas.draw()

    writer = PillowWriter(fps=fps)
    with writer.saving(fig, str(out_path), dpi=dpi):
        for k in range(n_frames):
            _update(k)
            writer.grab_frame()
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m e2e.ml.render_scene",
        description="Render an animated bird's-eye + radar-view GIF of a sampled e2e.ml scene "
                    "(or, with --rt, a single ray-traced camera PNG; --rt-topdown-gif, a "
                    "ray-traced top-down GIF; --rt-panel, a 2x2 bird's-eye+RT+ideal+non-ideal "
                    "GIF -- all of an e2e.ml.rt_scenes tier).",
    )
    p.add_argument("--tier", required=True,
                   help="difficulty tier (e2e.ml.scenes.DIFFICULTY_TIERS, or "
                        "e2e.ml.rt_scenes.RT_DIFFICULTY_TIERS with --rt)")
    p.add_argument("--config", required=True, help="radar config preset name (see e2e.ml.radar_config.PRESETS)")
    p.add_argument("--out", required=True, help="output path (.gif, or .png with --rt)")
    p.add_argument("--frames", type=int, default=30, help="animation frame count (GIF mode only)")
    p.add_argument("--fps", type=int, default=8, help="GIF playback frame rate (GIF mode only)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed (scene sampling + per-frame synthesis noise)")
    p.add_argument("--snr-db", type=float, default=30.0, help="synthesis SNR in dB (GIF mode only)")
    p.add_argument("--dpi", type=int, default=90, help="figure DPI (GIF mode only)")
    p.add_argument("--no-ideal-panel", action="store_true",
                   help="GIF mode: drop the noise-free 'ideal front end' middle panel "
                        "(see render_scene_gif's ideal_panel)")
    p.add_argument("--rt", action="store_true",
                   help="ray-trace a single camera PNG of an RT tier instead of an analytic GIF "
                        "(needs Sionna RT; see render_rt_tier_png)")
    p.add_argument("--rt-topdown-gif", action="store_true",
                   help="ray-trace a TOP-DOWN animated GIF of an RT tier (objects moving over "
                        "time, one scenario draw) instead of a single camera PNG (needs Sionna "
                        "RT; see render_rt_topdown_gif). Uses --frames/--fps like the analytic "
                        "GIF mode; implies --rt's tier vocabulary")
    p.add_argument("--rt-panel", action="store_true",
                   help="2x2 GIF: bird's-eye | RT top-down camera render | ideal-front-end "
                        "range-azimuth | non-ideal range-azimuth, all four panels driven from ONE "
                        "e2e.ml.rt_scenes RT-tier scenario (needs Sionna RT; see "
                        "render_scene_gif_2x2). Implies --rt's tier vocabulary, not "
                        "e2e.ml.scenes.DIFFICULTY_TIERS -- an analytic sample_scene() scenario has "
                        "no mesh for a camera to render.")
    p.add_argument("--frame-idx", type=int, default=0, help="RT mode: tier sample index (see rt_scenes)")
    p.add_argument("--corpus-tag", default=RENDER_CORPUS_TAG,
                   help="corpus identity salted into the scene seed. Defaults to "
                        f"{RENDER_CORPUS_TAG!r}; pass a CORPUS's own tag (from its "
                        "manifest) to picture one of that corpus's stored frames."),
    p.add_argument("--no-local-assets", action="store_true",
                   help="RT mode: disable the local (unshipped) higher-fidelity mesh pool, "
                        "use only Sionna-bundled meshes (see render_rt_tier_png)")
    p.add_argument("--no-caption", action="store_true",
                   help="RT mode: skip the title-bar caption overlay")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    from e2e.radar_config import PRESETS

    if args.config not in PRESETS:
        print(f"unknown --config {args.config!r}; choices: {sorted(PRESETS)}", file=sys.stderr)
        return 2
    cfg = PRESETS[args.config]

    if args.rt_topdown_gif:
        from e2e.ml.rt_scenes import RT_DIFFICULTY_TIERS

        if args.tier not in RT_DIFFICULTY_TIERS:
            print(f"unknown --tier {args.tier!r}; choices: {sorted(RT_DIFFICULTY_TIERS)}", file=sys.stderr)
            return 2
        out_path = render_rt_topdown_gif(args.tier, args.out, cfg=cfg, n_frames=args.frames,
                                         fps=args.fps, frame_idx=args.frame_idx, seed=args.seed,
                                         use_local_assets=not args.no_local_assets)
        size_mb = out_path.stat().st_size / 1e6
        print(f"wrote {out_path} ({size_mb:.2f} MB, RT tier {args.tier} top-down, "
             f"{args.frames} frames @ {args.fps} fps)")
        return 0

    if args.rt_panel:
        from e2e.ml.rt_scenes import RT_DIFFICULTY_TIERS, build_rt_tier_scenario

        if args.tier not in RT_DIFFICULTY_TIERS:
            print(f"unknown --tier {args.tier!r}; choices: {sorted(RT_DIFFICULTY_TIERS)}", file=sys.stderr)
            return 2
        dt = 1.0 / float(cfg.frame_rate_hz)
        scenario = build_rt_tier_scenario(args.tier, corpus_tag=args.corpus_tag,
                                          frame_idx=args.frame_idx, seed=args.seed,
                                          num_frames=args.frames, dt=dt,
                                          use_local_assets=not args.no_local_assets)
        out_path = render_scene_gif_2x2(cfg, scenario, args.out, n_frames=args.frames,
                                        fps=args.fps, seed=args.seed, snr_db=args.snr_db,
                                        dpi=args.dpi, use_local_assets=not args.no_local_assets)
        size_mb = out_path.stat().st_size / 1e6
        print(f"wrote {out_path} ({size_mb:.2f} MB, RT tier {args.tier} 2x2, "
             f"{args.frames} frames @ {args.fps} fps)")
        return 0

    if args.rt:
        from e2e.ml.rt_scenes import RT_DIFFICULTY_TIERS

        if args.tier not in RT_DIFFICULTY_TIERS:
            print(f"unknown --tier {args.tier!r}; choices: {sorted(RT_DIFFICULTY_TIERS)}", file=sys.stderr)
            return 2
        out_path = render_rt_tier_png(args.tier, args.out, cfg=cfg, frame_idx=args.frame_idx,
                                      seed=args.seed, use_local_assets=not args.no_local_assets,
                                      caption=not args.no_caption)
        print(f"wrote {out_path} (RT tier {args.tier} camera render)")
        return 0

    from e2e.ml.scenes import DIFFICULTY_TIERS, sample_scene

    if args.tier not in DIFFICULTY_TIERS:
        print(f"unknown --tier {args.tier!r}; choices: {sorted(DIFFICULTY_TIERS)}", file=sys.stderr)
        return 2

    rng = np.random.default_rng(args.seed)
    # Pass n_frames through to sample_scene if/when a sibling shard adds real per-frame
    # motion tracks; today's sample_scene() takes no such argument, so this degrades to
    # the single-frame call and render_scene_gif dead-reckons (see module docstring).
    sample_scene_kwargs = {}
    if "n_frames" in inspect.signature(sample_scene).parameters:
        sample_scene_kwargs["n_frames"] = args.frames
    scenario = sample_scene(cfg, args.tier, rng, **sample_scene_kwargs)

    out_path = render_scene_gif(cfg, scenario, args.out, n_frames=args.frames, fps=args.fps,
                                seed=args.seed, snr_db=args.snr_db, dpi=args.dpi,
                                ideal_panel=not args.no_ideal_panel)
    size_mb = out_path.stat().st_size / 1e6
    print(f"wrote {out_path} ({size_mb:.2f} MB, {args.frames} frames @ {args.fps} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
