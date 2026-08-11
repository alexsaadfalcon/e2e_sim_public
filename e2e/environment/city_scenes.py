"""Makes Sionna's built-in city scenes (`"munich"`, `"etoile"`, ...) loadable at
frequencies their bundled ITU materials don't cover -- e.g. automotive/mmWave 77 GHz.

THE MECHANISM (see `prepare_scene_for_frequency`): `sionna/rt/radio_materials/itu.py`
hard-raises when a material's ITU-R P.2040 curve fit (`eps_r = a f_GHz^b`,
`sigma = c f_GHz^d`) has no tabulated range covering the requested frequency. Munich
uses `marble` (valid 1-60 GHz) and `brick` (valid 1-40 GHz); other built-in scenes may
use others -- this module detects them from Sionna's OWN `ITU_MATERIALS_PROPERTIES`
table, not a hardcoded material list.

THE TRAP: `Scene.frequency`'s setter iterates *every material registered on the
scene*, not just the ones an object actually uses (`Scene.radio_materials`, not
`Scene.objects`). Reassigning `SceneObject.radio_material` on every user of a stale
material is therefore not enough on its own -- the stale material must also be
`scene.remove()`d, which Sionna only allows once `is_used` is `False` (see
`out_of_band_materials`/`prepare_scene_for_frequency`, and
`tests/test_city_scenes.py::test_removal_of_stale_material_is_required`, which pins
this down independent of this module's own code).

TWO SUBSTITUTION POLICIES (`policy=`), trade this off differently -- pick per corpus:
  * `EXTRAPOLATED` (default): keep the material's own curve fit, evaluated past its
    documented range, as CONSTANT `(eps_r, sigma)` on a plain `RadioMaterial` (no
    Sionna frequency callback, so nothing left to hard-raise). Defensible for `b == 0`
    materials like marble (`eps_r` is frequency-INDEPENDENT: 7.074 at 1 GHz, 60 GHz, or
    77 GHz alike) whose conductivity extrapolates smoothly and in-family with the
    materials ITU *does* tabulate to 100 GHz (concrete, glass): marble's sigma goes
    0.244 S/m @ 60 GHz -> 0.307 @ 77 GHz, no discontinuity. Less defensible for a
    material with strong `b`/`d` curvature far past its range -- there is no way to
    know from this module alone whether that holds, so callers picking `EXTRAPOLATED`
    for an unfamiliar material should sanity-check its `(a, b, c, d)` first (see
    `out_of_band_materials`' report).
  * `STAND_IN`: swap for `stand_in_itu_type` (default `"concrete"`, ITU-tabulated
    1-100 GHz -- the same choice `e2e.ml.rt_gen` makes for its own synthetic "flat"
    ground plane), keeping Sionna's normal ITU frequency dependence. More
    conservative (no extrapolation at all) but less faithful: a marble facade solved
    as concrete has different real electrical properties, not just a different label.

Both policies report what they did (`MaterialSubstitution`) so a generated corpus can
record what its city was actually made of -- this is provenance, not decoration.

Pure-stdlib module (no Sionna/torch import at module scope, only inside functions) --
importable without Sionna installed, matching the rest of `e2e.environment`.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Dict, Optional

EXTRAPOLATED = "extrapolated"
STAND_IN = "stand_in"
_POLICIES = (EXTRAPOLATED, STAND_IN)

# In-band (1-100 GHz) ITU material used by the STAND_IN policy's default swap -- same
# choice `e2e.ml.rt_gen._GROUND_MATERIAL` makes for the synthetic "flat" ground plane.
DEFAULT_STAND_IN_ITU_TYPE = "concrete"


@dataclass(frozen=True)
class MaterialSubstitution:
    """Provenance for one material swapped out at scene-load time (see module
    docstring): which ITU type it replaced, how many scene objects used it, which
    policy did the swap, and the replacement's resulting `(eps_r, sigma)`."""

    original_itu_type: str
    policy: str
    n_objects: int
    relative_permittivity: float
    conductivity: float
    replacement_name: str
    stand_in_itu_type: Optional[str] = None    # set only for `policy == STAND_IN`


def _scalar(x) -> float:
    """`mi.Float`/drjit scalar or plain python number -> python `float`."""
    try:
        return float(x[0])
    except (TypeError, IndexError):
        return float(x)


def _itu_table():
    # Sionna's own table (Section 2.1.4 of ITU-R P.2040) -- see module docstring:
    # detection must read this, not a hardcoded material list.
    from sionna.rt.radio_materials.itu import ITU_MATERIALS_PROPERTIES

    return ITU_MATERIALS_PROPERTIES


def _itu_in_band(itu_type: str, frequency_hz: float) -> bool:
    f_ghz = frequency_hz / 1e9
    return any(lo <= f_ghz <= hi for (lo, hi) in _itu_table()[itu_type])


def _itu_closest_params(itu_type: str, frequency_hz: float):
    """The `(a, b, c, d)` curve-fit of whichever tabulated range is closest to
    `frequency_hz` -- exact if `frequency_hz` is in-band, the nearest range's model
    otherwise (the `EXTRAPOLATED` policy)."""
    f_ghz = frequency_hz / 1e9
    ranges = _itu_table()[itu_type]

    def edge_distance(rng):
        lo, hi = rng
        if lo <= f_ghz <= hi:
            return 0.0
        return min(abs(f_ghz - lo), abs(f_ghz - hi))

    return ranges[min(ranges, key=edge_distance)]


def out_of_band_materials(scene, frequency_hz: float) -> Dict[str, object]:
    """`{material_name: material}` for every ITU radio material REGISTERED on `scene`
    (used or not) whose ITU-R P.2040 table doesn't cover `frequency_hz`.

    Iterates `scene.radio_materials` (ALL registered materials), not `scene.objects`
    (only the ones in use) -- `Scene.frequency`'s setter iterates the former, which is
    exactly the trap this module exists to route around (see module docstring). Only
    `ITURadioMaterial` instances are range-checked (detected via their `itu_type`
    attribute, not a name list); plain `RadioMaterial`s (no ITU table, e.g. the
    frequency-independent constants this module itself substitutes in) are skipped.
    """
    out = {}
    for name, mat in scene.radio_materials.items():
        itu_type = getattr(mat, "itu_type", None)
        if itu_type is not None and not _itu_in_band(itu_type, frequency_hz):
            out[name] = mat
    return out


def prepare_scene_for_frequency(scene, frequency_hz: float, *, policy: str = EXTRAPOLATED,
                                stand_in_itu_type: str = DEFAULT_STAND_IN_ITU_TYPE,
                                ) -> Dict[str, MaterialSubstitution]:
    """Make `scene.frequency = frequency_hz` safe to assign, swapping out any
    registered ITU material whose table doesn't cover `frequency_hz` (see
    `out_of_band_materials` and the module docstring for the two `policy` choices).
    Sets `scene.frequency` itself at the end (once every stale material is gone) and
    returns the substitution provenance, `{stale_material_name: MaterialSubstitution}`
    (empty if nothing needed swapping, e.g. `scene` only uses materials already
    tabulated at `frequency_hz`).
    """
    import sionna.rt as rt

    if policy not in _POLICIES:
        raise ValueError(f"policy must be one of {_POLICIES}, got {policy!r}")

    stale = out_of_band_materials(scene, frequency_hz)
    report: Dict[str, MaterialSubstitution] = {}
    for name, mat in stale.items():
        itu_type = mat.itu_type
        thickness = _scalar(mat.thickness)
        scattering_coefficient = _scalar(mat.scattering_coefficient)
        color = mat.color

        if policy == EXTRAPOLATED:
            a, b, c, d = _itu_closest_params(itu_type, frequency_hz)
            f_ghz = frequency_hz / 1e9
            eps_r = a * f_ghz ** b
            sigma = c * f_ghz ** d
            replacement = rt.RadioMaterial(
                f"e2e-city-{name}-extrap", thickness=thickness,
                relative_permittivity=eps_r, conductivity=sigma,
                scattering_coefficient=scattering_coefficient, color=color,
            )
            stand_in_used = None
        else:
            from sionna.rt.radio_materials.itu import itu_material

            eps_r, sigma = (_scalar(v) for v in itu_material(stand_in_itu_type,
                                                              float(frequency_hz)))
            replacement = rt.ITURadioMaterial(
                f"e2e-city-{name}-standin", stand_in_itu_type, thickness=thickness,
                scattering_coefficient=scattering_coefficient, color=color,
            )
            stand_in_used = stand_in_itu_type

        # Move every object off the stale material BEFORE removing it -- `scene.remove`
        # requires `is_used == False` (see module docstring).
        users = [obj for obj in scene.objects.values() if obj.radio_material.name == name]
        for obj in users:
            obj.radio_material = replacement
        scene.remove(name)

        report[name] = MaterialSubstitution(
            original_itu_type=itu_type, policy=policy, n_objects=len(users),
            relative_permittivity=float(eps_r), conductivity=float(sigma),
            replacement_name=replacement.name, stand_in_itu_type=stand_in_used,
        )

    scene.frequency = frequency_hz
    return report


def load_city_scene(base_scene: str, frequency_hz: float, *, policy: str = EXTRAPOLATED,
                    stand_in_itu_type: str = DEFAULT_STAND_IN_ITU_TYPE,
                    merge_shapes: bool = False):
    """Load a Sionna built-in scene (`"munich"`, `"etoile"`, ...) or a scene file path,
    and repair it for `frequency_hz` in one call (`prepare_scene_for_frequency`).
    Returns `(scene, report)`.
    """
    import sionna.rt as rt

    builtin = getattr(rt.scene, base_scene, None)
    scene = rt.load_scene(builtin if builtin is not None else base_scene,
                          merge_shapes=merge_shapes)
    report = prepare_scene_for_frequency(scene, frequency_hz, policy=policy,
                                         stand_in_itu_type=stand_in_itu_type)
    return scene, report


@contextlib.contextmanager
def patched_builtin_loader(frequency_hz: float, *, policy: str = EXTRAPOLATED,
                           stand_in_itu_type: str = DEFAULT_STAND_IN_ITU_TYPE,
                           report_sink: Optional[dict] = None):
    """Monkeypatches `sionna.rt.load_scene` for the duration of the `with` block so
    that whatever calls it gets back a scene already repaired for `frequency_hz`
    (`prepare_scene_for_frequency`).

    Why a monkeypatch rather than calling `load_city_scene` directly: the caller this
    is built for, `e2e.ml.rt_gen.build_rt_scene`, is owned by a different part of this
    codebase (not this module) and itself calls Sionna's `rt.load_scene` (via its
    private `_load_base_scene` helper) and then immediately assigns `scene.frequency`
    -- the exact assignment that hard-raises for an out-of-band ITU material (see
    module docstring). Patching Sionna's own `load_scene` repairs the scene one Python
    frame earlier than that assignment, without editing `rt_gen.py`; see
    `e2e.environment.blocks.RTEnvironmentBlock.get_S_pars` for the call site.

    Scoped and restored on exit (`finally`), and only wraps whatever `load_scene`
    call(s) happen inside the `with` block -- `build_rt_scene` makes exactly one, for
    the base scene; scenario objects are added afterwards via `scene.edit`, not
    `load_scene`, so they're unaffected.
    """
    import sionna.rt as rt

    original_load_scene = rt.load_scene

    def _loader(scene_or_path, **kwargs):
        scene = original_load_scene(scene_or_path, **kwargs)
        report = prepare_scene_for_frequency(scene, frequency_hz, policy=policy,
                                             stand_in_itu_type=stand_in_itu_type)
        if report_sink is not None:
            report_sink.update(report)
        return scene

    rt.load_scene = _loader
    try:
        yield
    finally:
        rt.load_scene = original_load_scene
