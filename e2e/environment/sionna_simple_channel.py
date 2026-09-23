"""Re-trace the Munich scenario at Ka-band and write a v2 `{"meta", "links"}` pkl.

Fixes the defect recorded in `notes/ESTABLISHED_FACTS.md` F93: the original script set
`scene.frequency` on a `simple_street_canyon` scene it then discarded (line 66 rebound
`scene` to a freshly loaded, still-3.5-GHz `munich`), so the shipped `munich.pkl` was
traced at Sionna's 3.5 GHz default (4.28 cm array spacing) while everything downstream
labelled it "30 GHz". This module sets `scene.frequency` on the scene it actually solves,
and does so BEFORE building `PlanarArray`s -- Sionna sizes element spacing in
wavelengths at construction time (see F92's note on upstream issue 470), so an array
built before the frequency assignment silently keeps the wrong physical spacing even if
`scene.frequency` is later corrected.

Usage::

    python -m e2e.environment.sionna_simple_channel [--carrier-hz 30e9]
        [--band-hz 28.5e9 31.5e9] [--num-freqs 1000] [--num-frames 100]
        [--out e2e/environment/sionna_sims/munich_ka.pkl] [--seed 41]

The output is the SAME v2 payload format `e2e.environment.scenario_runner` writes and
`e2e.environment.sionna_iterator.SionnaIterator` reads (`{"meta": {...}, "links":
{name: ndarray}}`), so the frames load through `SionnaEnvironmentBlock` unchanged --
this is a generated artifact (`sionna_sims/` is gitignored), not a committed one.
"""

from __future__ import annotations

import argparse
import datetime
import os
import pickle
import subprocess
import sys

import numpy as np

_C = 299_792_458.0  # m/s

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CARRIER_HZ = 30e9
DEFAULT_BAND_HZ = (28.5e9, 31.5e9)
DEFAULT_NUM_FREQS = 1000  # keeps the runtime frame shape (n_rx, 1, 1, 1000) legacy munich.pkl ships
DEFAULT_NUM_FRAMES = 100
DEFAULT_SEED = 41
DEFAULT_OUT = os.path.join(_THIS_DIR, "sionna_sims", "munich_ka.pkl")
LINK_NAME = "munich"


def _git_head():
    """Best-effort HEAD sha for provenance; None if git is unavailable (never fatal)."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_THIS_DIR,
                                       stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def build_frequencies(carrier_hz: float, band_hz, num_freqs: int) -> np.ndarray:
    """`num_freqs` points spanning `band_hz` (absolute Hz), relative to `carrier_hz`.

    Sionna's `paths.cfr(frequencies=...)` wants frequencies RELATIVE to the carrier
    Sionna's array/materials were built at (`scene.frequency`), not absolute Hz.
    """
    start_hz, stop_hz = band_hz
    return np.linspace(start_hz - carrier_hz, stop_hz - carrier_hz, num_freqs)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Re-trace the Sionna 'munich' scene at a given carrier/band.")
    p.add_argument("--carrier-hz", type=float, default=DEFAULT_CARRIER_HZ,
                   help="Carrier frequency in Hz; sets scene.frequency (default 30e9).")
    p.add_argument("--band-hz", type=float, nargs=2, default=list(DEFAULT_BAND_HZ),
                   metavar=("START_HZ", "STOP_HZ"),
                   help="Absolute band edges in Hz (default 28.5e9 31.5e9).")
    p.add_argument("--num-freqs", type=int, default=DEFAULT_NUM_FREQS)
    p.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args(argv)


def build_scene(carrier_hz: float):
    """Load munich, set `scene.frequency` on the scene that will actually be solved,
    THEN attach the tx/rx `PlanarArray`s and place tx/rx -- see the module docstring for
    why the order matters. Returns `(scene, tx, rx, wavelength, rx_spacing_m, aperture_m)`.

    Split out from `generate()` so a test can check the scene's own `.frequency` and its
    rx array spacing directly, not just the meta values `generate()` derives from them.
    """
    import sionna.rt
    from sionna.rt import Camera, PlanarArray, Receiver, Transmitter, load_scene

    scene = load_scene(sionna.rt.scene.munich, merge_shapes=True)  # merge -> faster solves
    # Set frequency on THE SCENE THAT IS ACTUALLY SOLVED, before building arrays (see
    # module docstring / F92 / F93).
    scene.frequency = float(carrier_hz)
    # scene.frequency is a DrJit array, not a python float/np.float32 -- .numpy() pulls
    # the scalar off the device before any float()/f-string formatting touches it.
    scene_frequency_hz = float(scene.frequency.numpy()[0])

    wavelength = _C / scene_frequency_hz
    rx_spacing_m = 0.5 * wavelength
    aperture_m = 31 * rx_spacing_m  # 32-element ULA per axis -> 31 inter-element gaps
    print(f"scene.frequency  = {scene_frequency_hz:.6e} Hz")
    print(f"wavelength       = {wavelength:.6e} m")
    print(f"rx element spacing = {rx_spacing_m:.6e} m")
    print(f"rx aperture (per axis, 32 elements) = {aperture_m:.6e} m")

    # Arrays built AFTER the frequency assignment above -- PlanarArray sizes its
    # `vertical_spacing`/`horizontal_spacing` (given in wavelengths) at construction time.
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                 horizontal_spacing=0.5, pattern="tr38901", polarization="V")
    scene.rx_array = PlanarArray(num_rows=32, num_cols=32, vertical_spacing=0.5,
                                 horizontal_spacing=0.5, pattern="iso", polarization="V")

    tx = Transmitter(name="tx", position=[8.5, 21, 27], display_radius=10)
    scene.add(tx)
    rx = Receiver(name="rx", position=[45, 90, 1.5], display_radius=10)
    scene.add(rx)
    tx.look_at(rx)
    rx.look_at(tx)

    # Kept for parity with the original interactive script (a camera angle used for
    # scene.render()/preview()); this module runs headless (no display), so it is
    # constructed but never rendered.
    Camera(position=[150, 275, 150], look_at=[30, 70, 28])

    return scene, tx, rx, wavelength, rx_spacing_m, aperture_m


def generate(args) -> tuple[np.ndarray, dict]:
    """Ray-trace `args.num_frames` and return `(stacked_s_pars, meta)`.

    Sionna is imported here (via `build_scene`, not at module scope) so `python -m
    e2e.environment.sionna_simple_channel --help` and unit tests that only exercise
    `build_frequencies`/`parse_args`/the pkl writer never need Sionna/DrJit installed.
    """
    import sionna.rt
    from sionna.rt import PathSolver

    scene, tx, rx, wavelength, rx_spacing_m, aperture_m = build_scene(args.carrier_hz)

    frequencies = build_frequencies(args.carrier_hz, args.band_hz, args.num_freqs)
    p_solver = PathSolver()

    # Physics decisions the validation campaign will revisit -- unchanged from the
    # original script, but now recorded in `meta` (see the module docstring) rather
    # than silently baked into the array.
    normalize = True
    normalize_delays = True

    all_s_pars = []
    for _ in range(args.num_frames):
        rx.position += [1, 0, 0]  # same per-frame motion as the original script
        paths = p_solver(scene=scene, max_depth=5, los=True, specular_reflection=True,
                         diffuse_reflection=False, refraction=True, synthetic_array=False,
                         seed=args.seed)
        cfr = paths.cfr(frequencies=frequencies, normalize=normalize,
                        normalize_delays=normalize_delays, out_type="numpy")
        # [num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, num_freqs] ->
        # [num_rx_ant, num_tx_ant, num_time_steps, num_freqs] (one rx, one tx node).
        s_pars = cfr[0, :, 0, :, :, :]
        all_s_pars.append(s_pars)

    all_s_pars = np.stack(all_s_pars, axis=0).astype(np.complex64)

    try:
        sionna_version = sionna.rt.__version__
    except AttributeError:
        sionna_version = getattr(sys.modules.get("sionna"), "__version__", None)

    meta = {
        "version": 2,
        "scenario_name": "munich",
        "scene": "munich",
        "carrier_hz": float(args.carrier_hz),
        "freq_plan": {
            "carrier_hz": float(args.carrier_hz),
            "start_hz": float(args.band_hz[0]),
            "stop_hz": float(args.band_hz[1]),
            "num_freqs": int(args.num_freqs),
        },
        "rx_spacing_m": float(rx_spacing_m),
        "aperture_m": float(aperture_m),
        "normalize": normalize,
        "normalize_delays": normalize_delays,
        "sionna_version": sionna_version,
        "git_head": _git_head(),
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "seed": int(args.seed),
        "links": {
            LINK_NAME: {
                "tx_node": "tx",
                "rx_node": "rx",
                "rx_array_shape": [32, 32],
                "n_tx_ant": 1,
                "kind": "radar",
                "tx_power_dbm": None,
                "physical_scale": False,
            },
        },
    }
    return all_s_pars, meta


def write_payload(all_s_pars: np.ndarray, meta: dict, out_path: str) -> None:
    """Write the `{"meta": ..., "links": {name: ndarray}}` v2 payload `SionnaIterator`
    understands (see `e2e.environment.sionna_iterator` / `scenario_runner`'s writer)."""
    payload = {"meta": meta, "links": {LINK_NAME: all_s_pars}}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    print(f"dumping to file {out_path}")
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print("done dumping")


def main(argv=None):
    args = parse_args(argv)
    all_s_pars, meta = generate(args)
    write_payload(all_s_pars, meta, args.out)
    return all_s_pars, meta


if __name__ == "__main__":
    main()
