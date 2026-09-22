"""Side-by-side detections: the same test frames through every detector the demo
compares, each at its OWN recall-0.5 operating point, in one PNG.

    python -m e2e.ml.detect_side_by_side --out docs/media/detect_side_by_side.png
    python -m e2e.ml.detect_side_by_side --frames 0 1 2 3 --arms "classical CFAR" raddetnet

Rows are detectors, columns are frames. Every number on the figure comes from ONE
authority, `e2e/ml/runs/beat_cfar.json` (`python -m e2e.ml.beat_cfar --skip-train`): the
checkpoint path per arm, its test AP, and the objectness threshold at which it first
reaches recall 0.5 -- so the crosses on screen ARE the false-alarm comparison the AP
table quotes, not three arbitrary thresholds. Decoding goes through
`e2e.ml.detect_viz` (the metric's own `decode_detections`), so a panel cannot disagree
with the scored numbers for the same frame.

The backdrop is display-tapered (`hann`) for legibility, as in `detect_viz`; the
classical CFAR thresholds its own unwindowed map, so its crosses are not pixel-aligned
with the backdrop beneath them. The range axis is cropped to the scoring crop (40 m by
default); labels stop there too.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

DEFAULT_BEAT_CFAR = Path("e2e/ml/runs/beat_cfar.json")
DEFAULT_ARMS = ("classical CFAR", "fftradnet_rd_b5", "raddetnet")
_LABELS = {"classical CFAR": "classical CA-CFAR", "fftradnet_rd_b5": "FFTRadNet (ported, rd)",
           "ssmradnet_rd_b5": "SSMRadNet (ported, rd)", "raddetnet": "RADDetNet (ours, rad)",
           "fftradnet_rad": "FFTRadNet (rad)"}


def load_arms(beat_cfar_path: Path, names: Sequence[str]) -> List[Dict]:
    """The arms the figure draws, with checkpoint, AP and recall-0.5 threshold from the
    authority file. Raises on an unknown arm name rather than guessing."""
    doc = json.loads(Path(beat_cfar_path).read_text())
    by_name = {a["name"]: a for a in doc["arms"]}
    out = []
    for n in names:
        if n not in by_name:
            raise SystemExit(f"arm {n!r} not in {beat_cfar_path}; have {sorted(by_name)}")
        a = by_name[n]
        out.append({
            "name": n, "label": _LABELS.get(n, n), "checkpoint": a.get("checkpoint"),
            "AP": float(a["AP"]),
            "threshold": float(a["operating_point"]["score_threshold"]),
            "fa_per_frame": float(a["operating_point"]["fp_per_frame"]),
        })
    return out, doc


def render(manifest: Path, split: str, frames: Sequence[int], arms: List[Dict], out: Path,
           *, max_range_m: float = 40.0, device=None, doc: Optional[Dict] = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from e2e.ml.detect_viz import (decode_classical_frame, decode_model_frame,
                                   frame_background_ra, plot_frame_detections)

    n_rows, n_cols = len(arms), len(frames)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 4.2 * n_rows),
                             squeeze=False)
    for j, fidx in enumerate(frames):
        ra_db, sin_az, range_m, _cfg = frame_background_ra(manifest, split, fidx)
        full_max = float(range_m[-1])
        for i, arm in enumerate(arms):
            thr = arm["threshold"]
            if arm["checkpoint"] is None:
                fd = decode_classical_frame(manifest, split, fidx, threshold=thr, device=device)
            else:
                fd = decode_model_frame(manifest, arm["checkpoint"], split, fidx,
                                        threshold=thr, device=device)
            n_det = len(fd.detections)
            n_gt = len(fd.targets)
            title = (f"{arm['label']}\nthreshold {thr:.2f} (its recall-0.5 point): "
                     f"{n_det} detections, {n_gt} GT")
            plot_frame_detections(axes[i][j], ra_db, sin_az, range_m, fd.targets,
                                  fd.detections, threshold=thr, title=title,
                                  max_range_m=max_range_m, full_max_range_m=full_max)
            if i == 0:
                axes[i][j].set_title(f"{split} frame {fidx}\n" + title, fontsize=9)
            else:
                axes[i][j].set_title(title, fontsize=9)
    ap_line = "  |  ".join(f"{a['label']}: AP {a['AP']:.3f}, {a['fa_per_frame']:.1f} FA/frame "
                           f"at recall 0.5" for a in arms)
    src = f"{Path(manifest).parent.name} {split} split"
    fig.suptitle(f"Same frames, every detector at its own recall-0.5 operating point -- {src}\n"
                 f"{ap_line}\n"
                 "GT: squares = vehicle, circles = pedestrian; '+' = detection (size ~ score). "
                 "Backdrop hann-tapered for display; CFAR thresholds its own unwindowed map. "
                 f"Range cropped to the {max_range_m:g} m scoring crop.",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--beat-cfar", default=str(DEFAULT_BEAT_CFAR),
                    help="the authority JSON with checkpoints, APs and operating points")
    ap.add_argument("--manifest", default=None,
                    help="manifest.json (default: the one recorded in --beat-cfar)")
    ap.add_argument("--split", default=None, help="default: the split recorded in --beat-cfar")
    ap.add_argument("--frames", type=int, nargs="+", default=[0, 1, 2, 3],
                    help="frame indices within the split (the demo presets replay 0..4)")
    ap.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    ap.add_argument("--max-range-m", type=float, default=None,
                    help="default: the crop recorded in --beat-cfar")
    ap.add_argument("--out", default="docs/media/detect_side_by_side.png")
    args = ap.parse_args(argv)

    arms, doc = load_arms(Path(args.beat_cfar), args.arms)
    manifest = Path(args.manifest or doc["manifest"])
    split = args.split or doc.get("split", "test")
    max_range = args.max_range_m if args.max_range_m is not None else float(
        doc["arms"][0].get("max_range_m") or 40.0)
    out = render(manifest, split, args.frames, arms, Path(args.out), max_range_m=max_range,
                 doc=doc)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
