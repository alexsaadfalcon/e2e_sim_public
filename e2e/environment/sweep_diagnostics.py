"""Does a stored frame file rotate its PRINCIPAL DIRECTION, and at what rank?

The two questions a "moving line of sight" file has to answer (owner directive
2026-09-24: "the line of sight path change angle so that even if the rank doesn't
change, the direction changes"), measured on the STORED CFR -- no Sionna, no solve:

* RANK: `e2e.simulation.rank_diagnostic`'s own `effective_rank` (count of singular
  values above 1 % of the largest, `_RANK_RTOL`), plus the top singular value's energy
  share and `sv2/sv1`. This is the number that must NOT move between a static file and
  a swept one.
* DIRECTION: the principal angle between consecutive frames' rank-1 left singular
  vectors (and between their top-k subspaces), plus WHERE that vector points in the
  array's own angle grid -- the peak of a NATIVE-resolution 2D aperture FFT (no zero
  padding: a padded FFT manufactures angle bins the aperture never resolved).
* A dynamic-range cross-check: peak minus median of the range-azimuth map, the same
  quantity the demo cards print (`webapp/pipeline_runner._peak_minus_median_db`).
* `--pipeline N`: what the LIVE chain does with the file -- N frames through the real
  `Simulation` (RFFE -> interconnect -> AFE -> AdaOja tracker), reporting `subspace_err`
  per frame, repeatable so the run-to-run spread is visible rather than assumed. The
  tracker's own error is the quantity a "the direction keeps moving" file is supposed to
  change, so it does not belong in a scratch script.

For a file written by `e2e.environment.sionna_simple_channel` the per-frame GEOMETRY
(rx/tx positions, the LoS azimuth in the array frame, the direct-path presence receipt)
is read straight out of `meta["frames"]` and printed beside the measured columns, so the
commanded sweep and its effect on the data sit in one table.

Run it (numbers in `notes/LOSWEEP_REPORT_2026-09-25.md` came from exactly this)::

    python -m e2e.environment.sweep_diagnostics \\
        e2e/environment/sionna_sims/munich_ka.pkl \\
        e2e/environment/sionna_sims/munich_ka_losweep.pkl

Cost: one full SVD per frame of a `[n_rx, n_freqs]` matrix (1024x5000 here, a few
seconds per frame on the library device) plus one `pickle.load` of the whole file.
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import torch

from e2e.simulation import rank_diagnostic


def load_frames(path, link=None):
    """`(frames, meta)` for any of the three stored payload shapes
    `e2e.environment.sionna_iterator.SionnaIterator` accepts -- reused here rather than
    constructing an iterator so a file that is not on a scenario-name path (this module's
    normal case: a candidate file beside the shipped one) needs no path monkeypatching."""
    from e2e.environment.sionna_iterator import SionnaIterator

    it = SionnaIterator(path, link=link)
    return it.all_s_pars, it.meta


def frame_matrix(frame, device=None):
    """One stored frame `[n_rx, n_tx, n_chirp, n_freqs]` as the `[n_rx, n_freqs]` matrix
    whose left singular vectors ARE the array-domain directions -- the same flattening
    `e2e.simulation._svd_frame` feeds the tracker's ground truth (first chirp, tx axis
    folded in)."""
    t = torch.as_tensor(np.asarray(frame), dtype=torch.complex64)
    if t.ndim == 4:
        t = t[:, :, 0, :]
    t = t.reshape(-1, t.shape[-1])
    return t.to(device) if device is not None else t


def principal_angle_deg(u, v) -> float:
    """Angle in degrees between two unit vectors, SIGN-INVARIANT (`acos |<u,v>|`): a
    singular vector's sign/phase is arbitrary, so the raw inner product's phase carries
    no directional information."""
    c = float(torch.abs(torch.vdot(u, v)).item())
    return float(np.degrees(np.arccos(min(1.0, c))))


def subspace_angle_deg(u_basis, v_basis) -> float:
    """LARGEST principal angle in degrees between two orthonormal bases `[n, k]` -- the
    worst-case direction a rank-k tracker would have to re-acquire between the two
    frames (`arccos` of the SMALLEST singular value of `U^H V`)."""
    s = torch.linalg.svdvals(u_basis.conj().transpose(-2, -1) @ v_basis)
    return float(np.degrees(np.arccos(float(np.clip(s.min().item(), -1.0, 1.0)))))


def angular_peak(u1, array_shape=(32, 32)):
    """`(u_az, u_el, peak_share)` for a `[n_rx]` array-domain vector: the NATIVE
    resolution 2D aperture FFT's peak, in direction cosines, plus the fraction of the
    angular power in that one bin.

    `u = (bin - N/2) / (N/2)`, the same direction-cosine axis the webapp's angle axis
    uses for a lambda/2 array. Axis order follows `SionnaEnvironmentBlock`: dim 0 of the
    aperture view is columns (azimuth), dim 1 is rows (elevation). No zero padding: bins
    are the ones the aperture actually resolves, so the peak's quantization is
    1/(N/2) = 0.0625 in u for a 32-element axis (about 3.6 deg near broadside)."""
    n_az, n_el = array_shape
    g = torch.as_tensor(u1).reshape(n_az, n_el)
    sp = torch.fft.fftshift(torch.fft.fft2(g))
    p = (sp.abs() ** 2)
    idx = int(torch.argmax(p.reshape(-1)).item())
    i, j = idx // n_el, idx % n_el
    total = float(p.sum().item())
    u_az = (i - n_az / 2) / (n_az / 2)
    u_el = (j - n_el / 2) / (n_el / 2)
    return float(u_az), float(u_el), (float(p[i, j].item() / total) if total > 0 else float("nan"))


def library_device():
    """The device the library is configured for (cuda when present, else cpu) -- the
    convention every module here follows; a 1024x5000 complex SVD per frame is minutes
    on a CPU and seconds on the GPU."""
    from e2e.blocks import device
    return device


def range_az_peak_minus_median_db(frame, array_shape=(32, 32), bins=256,
                                  device=None) -> float:
    """peak - median (dB) of the frame's range-azimuth power map, through the shipped
    `e2e.blocks.RangeAzBlock` and the peak-normalized-dB convention the demo cards print.

    This is the RAW STORED frame, i.e. no RFFE / interconnect / AFE in front of it --
    comparable across files, but not the same number a full pipeline run reports."""
    from e2e.blocks import RangeAzBlock
    from e2e.chain.dechirp import DechirpBlock
    from e2e.chain.receive import RangeTransformBlock

    class _SingleTxCfg:
        """The two fields `DechirpBlock` reads. The stored munich traces are single-TX,
        and a full `RadarConfig` here would be four numbers nobody reads."""

        mimo = "single"
        n_tx = 1

    t = torch.as_tensor(np.asarray(frame), dtype=torch.complex64)
    if device is not None:
        t = t.to(device)
    # THE SPINE'S OWN TWO STEPS, in its own order, at its own settings. Merged from the
    # `losweep` branch, which forked before the one-chain contract: `RangeAzBlock` used
    # to range-compress `s_pars` itself, and now consumes a `cube` that
    # `RangeTransformBlock` owns -- so this called it on a state dict with no cube and
    # got a KeyError (found on the merge, 2026-09-25). `window="none"` and
    # `dc_removal=False` are the imaging spine's settings (e2e/simulation.py
    # `_build_spine`), NOT the ML protocol's, so the number this returns is the one the
    # demo cards print; `array_shape=` is required of a direct caller because
    # `Simulation` is what normally seeds `state["aperture_shape"]`.
    state = {"s_pars": t}
    state.update(DechirpBlock(_SingleTxCfg()).apply(state))
    state.update(RangeTransformBlock(None, window="none",
                                     dc_removal=False).apply(state))
    ra = RangeAzBlock(bins=bins, array_shape=tuple(array_shape)).apply(state)["range_az"]
    ra = ra.detach().to(torch.float64).cpu()
    db = 10.0 * torch.log10(torch.clamp(ra, min=1e-300) / ra.max())
    return float((db.max() - db.median()).item())


def diagnose_file(path, n_frames=None, k=2, array_shape=None, link=None, device=None):
    """Per-frame rows + a summary for one stored file. Each row carries the MEASURED
    columns and, when the file has per-frame generator geometry (`meta["frames"]`), the
    commanded/geometric ones beside them."""
    frames, meta = load_frames(path, link=link)
    if device is None:
        device = library_device()
    n_total = frames.shape[0]
    n = n_total if n_frames is None else min(n_frames, n_total)
    if array_shape is None:
        link_meta = (meta or {}).get("links", {}).get("munich") if meta else None
        shape = (link_meta or {}).get("rx_array_shape") if link_meta else None
        # Stored as [num_rows, num_cols]; the aperture view wants (num_cols, num_rows).
        array_shape = (shape[1], shape[0]) if shape else (32, 32)
    geom = (meta or {}).get("frames")

    rows = []
    prev_u1 = None
    prev_basis = None
    for i in range(n):
        m = frame_matrix(frames[i], device=device)
        u, s, _ = torch.linalg.svd(m, full_matrices=False)
        diag = rank_diagnostic(s, k)
        u1 = u[:, 0]
        u_az, u_el, peak_share = angular_peak(u1.cpu(), array_shape)
        row = {
            "index": i,
            "effective_rank": diag["effective_rank"],
            "sv_gap_norm": diag["sv_gap_norm"],
            "energy_share_1": float((s[0] ** 2 / (s ** 2).sum()).item()),
            "sv2_over_sv1": float((s[1] / s[0]).item()),
            "u_az": u_az,
            "u_el": u_el,
            "angular_peak_share": peak_share,
            "angle_u1_prev_deg": (float("nan") if prev_u1 is None
                                  else principal_angle_deg(prev_u1, u1)),
            "angle_subspace_prev_deg": (float("nan") if prev_basis is None
                                        else subspace_angle_deg(prev_basis, u[:, :k])),
            "peak_minus_median_db": range_az_peak_minus_median_db(frames[i], array_shape,
                                                                  device=device),
        }
        if geom is not None and i < len(geom):
            g = geom[i]
            row.update({"los_az_deg": g["los_az_deg"], "sin_az": g["sin_az"],
                        "los_present": g["los_present"],
                        "los_power_share": g.get("los_power_share"),
                        "range_m": g["range_m"],
                        "commanded_offset_deg": g["boresight_offset_deg"]})
        rows.append(row)
        prev_u1, prev_basis = u1, u[:, :k]
    return rows, meta


def pipeline_subspace_err(path, n_steps=8, k=2, warm_start=False, m=512, n_refine=10,
                          gap_response="refine", interconnect_case="case3", repeats=1):
    """Run the REAL pipeline on a stored file and return the per-frame `subspace_err`
    lists (one per repeat).

    Defaults mirror the Thrust 3 screen's configuration: `k=2`, `warm_start=False` (the
    honest cold start -- `warm_start="cold"` in the UI maps to exactly this, see
    `webapp/pipeline_runner.py`), `AdaOjaBlock(..., m=512, n_refine=10,
    gap_response="refine")`. Every repeat rebuilds the blocks, so `repeats > 1` measures
    this pipeline's own nondeterminism instead of leaving it as an assumption.

    The file is selected by pointing `e2e.environment.sionna_iterator.SIONNA_MUNICH_PATH`
    at it -- the same module attribute the test suite monkeypatches, and the only
    authority `SionnaMunichIterator`'s default branch consults. That indirection exists
    because the runtime's scenario names resolve to two fixed munich paths; a new file
    has no name of its own until the webapp's corpus catalogue gets one.
    """
    import e2e.environment.sionna_iterator as sionna_iterator

    # THE SOURCE MUST CARRY A FREQUENCY PLAN. `Simulation`'s default ('full')
    # composition puts the front end on the BEAT RECORD and needs the beat sample rate
    # to reference its noise bandwidth to; with no `radar_cfg=` it derives one from the
    # source's own `freq_plan`, and refuses by name when there is neither. Every file
    # this function is pointed at in anger (munich_ka.pkl and the two swept traces) is a
    # v2 pkl with a plan; a hand-built fixture has to carry one too. Also merged-in
    # breakage: this module forked before that requirement (2026-09-25).
    saved = sionna_iterator.SIONNA_MUNICH_PATH
    sionna_iterator.SIONNA_MUNICH_PATH = path
    try:
        from e2e.blocks import (AdaOjaBlock, AFEBlock, FFTBlock, InterconnectBlock,
                                RangeAzBlock, RFFEBlock, SionnaEnvironmentBlock,
                                SubspaceErrorBlock)
        from e2e.simulation import Simulation

        out = []
        for _ in range(int(repeats)):
            env = SionnaEnvironmentBlock("munich")
            n_rx = env.array_shape[0] * env.array_shape[1]
            sim = Simulation(
                env,
                [FFTBlock(), RangeAzBlock(), SubspaceErrorBlock()],
                k,
                RFFEBlock(n=n_rx, physical_scale=bool(env.physical_scale)),
                InterconnectBlock(case=interconnect_case),
                AFEBlock(),
                AdaOjaBlock(n_rx, k, m=m, n_refine=n_refine, gap_response=gap_response),
                warm_start=warm_start,
            )
            res = sim.run(n_steps=n_steps)
            out.append([float(x) for x in res["subspace_err"]])
    finally:
        sionna_iterator.SIONNA_MUNICH_PATH = saved
    return out


def print_pipeline_report(path, runs, n_steps, k, warm_start):
    print(f"\n--- live pipeline on {path}")
    print(f"    Simulation(k={k}, warm_start={warm_start}, AdaOja m=512 n_refine=10 "
          f"gap_response=refine, RFFE + interconnect case3 + AFE), {n_steps} frames, "
          f"{len(runs)} repeat(s)")
    for i, err in enumerate(runs):
        print(f"    run {i}: " + ", ".join(f"{e:.4f}" for e in err))
    if runs:
        a = np.array(runs, dtype=float)
        print("    per-frame mean : " + ", ".join(f"{v:.4f}" for v in a.mean(axis=0)))
        if len(runs) > 1:
            print("    per-frame range: " + ", ".join(f"{v:.4f}" for v in
                                                      (a.max(axis=0) - a.min(axis=0))))
        tail = a[:, 1:] if a.shape[1] > 1 else a
        print(f"    frames 1..{n_steps - 1}: min {tail.min():.4f} max {tail.max():.4f} "
              f"mean {tail.mean():.4f}")


def _summary(rows, key):
    v = np.array([r.get(key, float("nan")) for r in rows], dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return None
    return float(v.min()), float(v.max()), float(v.mean())


def print_report(path, rows, meta, k):
    print(f"\n=== {path}")
    if meta:
        fp = meta.get("freq_plan") or {}
        print(f"    carrier {meta.get('carrier_hz')} Hz, band {fp.get('start_hz')}-"
              f"{fp.get('stop_hz')} Hz, {fp.get('num_freqs')} pts, "
              f"rx spacing {meta.get('rx_spacing_m')} m, aperture {meta.get('aperture_m')} m")
        print(f"    diffuse={meta.get('diffuse_reflection')} S={meta.get('scattering_coefficient')} "
              f"boresight_offset_deg={meta.get('boresight_offset_deg')} "
              f"los_sweep_deg={meta.get('los_sweep_deg')} tx_lateral_m={meta.get('tx_lateral_m')}")
        print(f"    git_head={meta.get('git_head')} generated_at={meta.get('generated_at')} "
              f"argv={meta.get('generator_argv')}")
    print(f"    frames measured: {len(rows)}; k={k}; rank = effective_rank (sv > 1% of sv1)")
    print("   f   los_az  los%   rank  sv2/sv1     E1    u_az   u_el  ang(u1)  ang(k={})  pk-med".format(k))
    for r in rows:
        az = r.get("los_az_deg", float("nan"))
        sh = r.get("los_power_share")
        print(f" {r['index']:3d} {az:8.2f} "
              f"{'  n/a' if sh is None else format(100 * sh, '5.1f')} "
              f"{r['effective_rank']:6d} {r['sv2_over_sv1']:8.4f} {r['energy_share_1']:6.4f} "
              f"{r['u_az']:+6.3f} {r['u_el']:+6.3f} {r['angle_u1_prev_deg']:8.2f} "
              f"{r['angle_subspace_prev_deg']:9.2f} {r['peak_minus_median_db']:7.2f}")
    for key, label in (("effective_rank", "effective rank (1%)"),
                       ("energy_share_1", "energy share sv1"),
                       ("sv2_over_sv1", "sv2/sv1"),
                       ("angle_u1_prev_deg", "consec angle u1 (deg)"),
                       ("angle_subspace_prev_deg", f"consec angle k={k} (deg)"),
                       ("peak_minus_median_db", "peak-median (dB)"),
                       ("u_az", "angular peak u_az"),
                       ("los_az_deg", "LoS azimuth (deg)")):
        st = _summary(rows, key)
        if st is not None:
            print(f"    {label:24s} min {st[0]:9.3f}  max {st[1]:9.3f}  mean {st[2]:9.3f}")
    if "los_az_deg" in rows[0]:
        az = np.array([r["los_az_deg"] for r in rows])
        d = np.diff(az)
        print(f"    LoS azimuth span {az.max() - az.min():.2f} deg, "
              f"monotonic={bool((d > 0).all() or (d < 0).all())}, "
              f"mean |step| {np.abs(d).mean():.3f} deg/frame, "
              f"frames without a direct path "
              f"{sum(1 for r in rows if not r.get('los_present', True))}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("paths", nargs="+", help="stored frame pkl(s)")
    p.add_argument("--frames", type=int, default=None, help="measure only the first N")
    p.add_argument("--k", type=int, default=2, help="subspace rank for the angle/gap columns")
    p.add_argument("--link", default=None, help="link name inside a multi-link pkl")
    p.add_argument("--pipeline", type=int, default=None, metavar="N",
                   help="also run N frames of the REAL pipeline (Simulation + AdaOja) on "
                        "each file and print subspace_err per frame")
    p.add_argument("--pipeline-repeats", type=int, default=1,
                   help="repeat the pipeline run this many times (shows its own spread)")
    p.add_argument("--pipeline-warm-start", action="store_true",
                   help="warm-start the tracker from a perturbed ground truth; the "
                        "default is the honest cold start the T3 screen uses")
    p.add_argument("--frames-only", action="store_true",
                   help="skip the per-frame SVD table (useful with --pipeline)")
    p.add_argument("--device", default=None,
                   help="torch device (default: the library device -- cuda if present)")
    p.add_argument("--pickle-out", default=None,
                   help="also dump the rows to this pickle (for plots/tables elsewhere)")
    args = p.parse_args(argv)
    all_rows = {}
    for path in args.paths:
        if not args.frames_only:
            rows, meta = diagnose_file(path, n_frames=args.frames, k=args.k,
                                       link=args.link, device=args.device)
            print_report(path, rows, meta, args.k)
            all_rows[path] = rows
        if args.pipeline:
            runs = pipeline_subspace_err(path, n_steps=args.pipeline, k=args.k,
                                         warm_start=args.pipeline_warm_start,
                                         repeats=args.pipeline_repeats)
            print_pipeline_report(path, runs, args.pipeline, args.k,
                                  args.pipeline_warm_start)
            all_rows.setdefault(path, [])
            all_rows[path] = {"frames": all_rows[path], "pipeline_subspace_err": runs}
    if args.pickle_out:
        with open(args.pickle_out, "wb") as f:
            pickle.dump(all_rows, f)
    return all_rows


if __name__ == "__main__":
    main()
