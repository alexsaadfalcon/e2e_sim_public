"""
Animated bird's-eye + radar-view GIFs of `e2e.ml.scenes` scenarios.

Purely a visualization/README-media tool: it reuses the existing scene/synthesis stack
(`e2e.ml.scatterers.frame_scatterers`/`radar_pose`, `e2e.ml.rd_synth.synthesize_adc`,
`e2e.ml.transforms.adc_to_rd`/`tdm_deinterleave`, `e2e.ml.labels.LabelGrid`/
`targets_in_grid`) end to end; nothing here re-derives geometry or re-implements
synthesis. Two panels, side by side, one frame per animation tick:

  LEFT  -- bird's-eye (x, y) scatter: radar marker + boresight ray + FOV wedge + max-
           range arc, vehicles/pedestrians/clutter as distinct markers, velocity arrows.
  RIGHT -- the radar's own view: a range x sin(azimuth) power (dB) map formed from the
           SAME frame's synthesized ADC, with ground-truth (vehicle/pedestrian) targets
           overlaid as markers.

Motion: `e2e.ml.scenes.sample_scene` currently returns a single-frame (`num_frames=1`)
`Scenario` -- there is no per-frame motion track to resolve yet. `render_scene_gif`
checks `scenario.num_frames` at call time and takes one of two paths so it keeps working
either way:
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
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # noqa: E402 -- must precede pyplot import; headless/CI-safe

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.animation import PillowWriter  # noqa: E402
from matplotlib.patches import Arc, Wedge  # noqa: E402

from e2e.ml.labels import LabelGrid, targets_in_grid  # noqa: E402
from e2e.ml.rd_synth import synthesize_adc  # noqa: E402
from e2e.ml.scatterers import RadarPose, Scatterer, frame_scatterers, radar_pose  # noqa: E402
from e2e.ml.transforms import adc_to_rd, tdm_deinterleave  # noqa: E402

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

# GT marker style on the range-azimuth panel.
_GT_MARKERS = {"vehicle": ("s", _COLOR_VEHICLE), "pedestrian": ("o", _COLOR_PEDESTRIAN)}


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
def range_azimuth_map(cfg, adc: torch.Tensor, n_angle_fft: Optional[int] = None):
    """Raw ADC `[n_rx, n_chirps, n_samples]` -> `(ra_db [n_angle, n_range], sin_az_axis)`.

    `ra_db` is normalized so its own peak bin is 0 dB (see the module docstring for the
    angle-FFT-then-non-coherent-Doppler-collapse recipe). `sin_az_axis` is the centre
    sin(azimuth) of each row, ascending. Range axis (columns) is implicit:
    `i * cfg.range_resolution_m` for column `i`.
    """
    if cfg.mimo == "tdm":
        sub_cfg = dataclasses.replace(cfg, n_tx=1, mimo="single", n_chirps=cfg.n_chirps_per_tx)
        rd = adc_to_rd(sub_cfg, tdm_deinterleave(cfg, adc))
    else:
        rd = adc_to_rd(cfg, adc)
    n_channel = rd.shape[0]
    n_fft = int(n_angle_fft) if n_angle_fft is not None else max(64, n_channel)

    angle_spec = torch.fft.fftshift(torch.fft.fft(rd, n=n_fft, dim=0), dim=0)  # [n_fft, R, D]
    power = angle_spec.abs() ** 2
    ra_power = power.max(dim=2).values  # non-coherent Doppler collapse, [n_fft, R]

    eps = torch.finfo(torch.float32).tiny
    peak = ra_power.max().clamp_min(eps)
    ra_db = 10.0 * torch.log10((ra_power / peak).clamp_min(eps))

    sin_az_axis = 2.0 * (torch.arange(n_fft, dtype=torch.float32) - n_fft // 2) / n_fft
    return ra_db.to(torch.float32).cpu(), sin_az_axis.numpy()


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
                     scatterers: Sequence[Scatterer], pose: RadarPose):
    ax.clear()
    max_range = float(cfg.max_range_m)
    n_range = ra_db.shape[1]
    range_axis = np.arange(n_range) * float(cfg.range_resolution_m)

    ax.imshow(
        ra_db.numpy(),
        extent=[sin_az_axis[0], sin_az_axis[-1], range_axis[0], range_axis[-1]],
        origin="lower", aspect="auto", cmap="inferno", vmin=-40.0, vmax=0.0,
    )

    seen_labels = set()
    for r, sin_az, cls in targets_in_grid(grid, scatterers, pose, classes=("vehicle", "pedestrian")):
        marker, color = _GT_MARKERS.get(cls, ("x", "white"))
        label = f"GT {cls}" if cls not in seen_labels else None
        seen_labels.add(cls)
        ax.plot(sin_az, r, marker=marker, markersize=9, markerfacecolor="none",
                markeredgecolor=color, markeredgewidth=1.6, linestyle="none", label=label)

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(0.0, max_range)
    ax.set_xlabel("sin(azimuth)")
    ax.set_ylabel("range (m)")
    ax.set_title("Radar view: range-azimuth power (dB)")
    if seen_labels:
        ax.legend(loc="upper right", fontsize=7, framealpha=0.7)


# --------------------------------------------------------------------------------
# Top-level GIF renderer
# --------------------------------------------------------------------------------
def render_scene_gif(cfg, scenario, out_path, *, n_frames: int = 30, fps: int = 8, seed: int = 0,
                     snr_db: Optional[float] = 30.0, dpi: int = 90,
                     n_angle_fft: Optional[int] = None) -> Path:
    """Render a two-panel (bird's-eye + radar-view) animated GIF for `scenario`.

    `cfg` (a `RadarConfig`) sets both the synthesis parameters and the frame timing
    (`dt = 1 / cfg.frame_rate_hz`, see the module docstring). `seed` seeds
    `synthesize_adc`'s noise/phase RNG per frame (`seed + frame_index`, so frames are
    reproducible but not identical). Returns `out_path` as a `Path`.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dt = 1.0 / float(cfg.frame_rate_hz)
    scats_per_frame, pose_per_frame = _resolve_frames(scenario, cfg, n_frames, dt)
    grid = LabelGrid.for_config(cfg)

    fig, (ax_bev, ax_rv) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=dpi)
    fig.suptitle(f"{scenario.name}  ({cfg.name})", fontsize=10)

    def _update(k: int):
        scatterers = scats_per_frame[k]
        pose = pose_per_frame[k]
        adc = synthesize_adc(cfg, scatterers, pose, snr_db=snr_db, seed=seed + k)
        ra_db, sin_az_axis = range_azimuth_map(cfg, adc, n_angle_fft=n_angle_fft)
        _draw_birdseye(ax_bev, scatterers, pose, cfg)
        _draw_radar_view(ax_rv, cfg, grid, ra_db, sin_az_axis, scatterers, pose)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        return ()

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
        description="Render an animated bird's-eye + radar-view GIF of a sampled e2e.ml scene.",
    )
    p.add_argument("--tier", required=True, help="difficulty tier (see e2e.ml.scenes.DIFFICULTY_TIERS)")
    p.add_argument("--config", required=True, help="radar config preset name (see e2e.ml.radar_config.PRESETS)")
    p.add_argument("--out", required=True, help="output .gif path")
    p.add_argument("--frames", type=int, default=30, help="animation frame count")
    p.add_argument("--fps", type=int, default=8, help="GIF playback frame rate")
    p.add_argument("--seed", type=int, default=0, help="RNG seed (scene sampling + per-frame synthesis noise)")
    p.add_argument("--snr-db", type=float, default=30.0, help="synthesis SNR in dB (see synthesize_adc)")
    p.add_argument("--dpi", type=int, default=90, help="figure DPI (lower -> smaller file)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    from e2e.ml.radar_config import PRESETS

    if args.config not in PRESETS:
        print(f"unknown --config {args.config!r}; choices: {sorted(PRESETS)}", file=sys.stderr)
        return 2
    cfg = PRESETS[args.config]

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
                                seed=args.seed, snr_db=args.snr_db, dpi=args.dpi)
    size_mb = out_path.stat().st_size / 1e6
    print(f"wrote {out_path} ({size_mb:.2f} MB, {args.frames} frames @ {args.fps} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
