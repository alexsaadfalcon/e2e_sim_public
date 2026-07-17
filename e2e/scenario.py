"""
Scenario specification for the end-to-end array-processing simulator.

This is the **shared contract** between three subsystems:

* the web UI (block-diagram + scenario placement),
* the offline scenario generator (Sionna RT ray tracing -> .pkl frames),
* the radar / communications / ISAC examples.

A `Scenario` is a fully declarative, JSON-serializable description of a scene: which
base environment to load, which antenna nodes exist (and what role each plays --
radar, comm TX, comm RX, ...), which physical objects populate the scene, and how
everything moves over a sequence of frames.

Nothing here imports Sionna or torch. It is intentionally dependency-free so the UI and
examples can build, load, save, and validate scenarios on any machine, and only the
heavy generation step needs Sionna installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any


Vec3 = Tuple[float, float, float]

# System reference impedance in ohms. Matches Rs=50 in e2e/circuit/rffe_model.py; the
# generation layer uses this to convert a node's received power into an LNA input
# voltage. Kept as a plain module-level float so this file stays dependency-free.
SYSTEM_IMPEDANCE_OHMS = 50.0


class NodeRole(str, Enum):
    """Role an antenna node plays in the scenario."""
    RADAR = "radar"          # monostatic/bistatic sensing node (TX+RX co-located by default)
    COMM_TX = "comm_tx"      # communications transmitter
    COMM_RX = "comm_rx"      # communications receiver
    MONITOR = "monitor"      # passive receiver (e.g. for mapping / interference study)


class AntennaPattern(str, Enum):
    ISO = "iso"
    TR38901 = "tr38901"
    DIPOLE = "dipole"
    HW_DIPOLE = "hw_dipole"


class Polarization(str, Enum):
    V = "V"
    H = "H"
    VH = "VH"
    CROSS = "cross"


class ObjectKind(str, Enum):
    SPHERE = "sphere"        # built-in Sionna sphere primitive
    BOX = "box"
    MESH = "mesh"            # external mesh referenced by `asset`


@dataclass
class ArrayConfig:
    """Planar antenna array configuration (maps onto sionna.rt.PlanarArray)."""
    num_rows: int = 1
    num_cols: int = 1
    vertical_spacing: float = 0.5      # in wavelengths
    horizontal_spacing: float = 0.5    # in wavelengths
    pattern: AntennaPattern = AntennaPattern.ISO
    polarization: Polarization = Polarization.V

    @property
    def num_elements(self) -> int:
        return self.num_rows * self.num_cols


@dataclass
class Motion:
    """
    Per-frame motion applied to a node or object.

    `velocity` is a constant displacement (meters) added each frame.
    `waypoints`, if given, overrides velocity: positions are interpolated across the
    scenario's frames so the entity passes through each waypoint in order.
    `angular_velocity_deg` rotates the entity (degrees per frame) about +z; useful for
    the circular car motion the existing environment code produces.
    """
    velocity: Vec3 = (0.0, 0.0, 0.0)
    waypoints: List[Vec3] = field(default_factory=list)
    angular_velocity_deg: float = 0.0

    @property
    def is_static(self) -> bool:
        return (
            self.velocity == (0.0, 0.0, 0.0)
            and not self.waypoints
            and self.angular_velocity_deg == 0.0
        )


@dataclass
class Node:
    """An antenna node (radar / comm endpoint) placed in the scene."""
    name: str
    role: NodeRole = NodeRole.RADAR
    position: Vec3 = (0.0, 0.0, 0.0)
    look_at: Optional[Vec3] = None
    array: ArrayConfig = field(default_factory=ArrayConfig)
    motion: Motion = field(default_factory=Motion)
    # Transmit power in dBm, only meaningful for TX-capable roles (RADAR, COMM_TX).
    # None = legacy mode: generation keeps Sionna's unit-energy normalization and no
    # absolute power scale is applied. There is deliberately no separate element-gain
    # field here -- element gain is already carried by ArrayConfig.pattern (the Sionna
    # antenna pattern); a standalone scalar would double-count it.
    tx_power_dbm: Optional[float] = None
    # free-form per-node settings (e.g. tx power, waveform id) used by examples
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneObject:
    """A physical object (scatterer / target / clutter) in the scene."""
    name: str
    kind: ObjectKind = ObjectKind.SPHERE
    position: Vec3 = (0.0, 0.0, 0.0)
    scaling: float = 1.0
    material: str = "metal"                 # ITU material name
    color: Optional[Tuple[float, float, float]] = None
    asset: Optional[str] = None             # path to mesh when kind == MESH
    motion: Motion = field(default_factory=Motion)


@dataclass
class FrequencyPlan:
    """Carrier and the frequency grid over which S-parameters are sampled."""
    carrier_hz: float = 30.0e9
    start_hz: float = 28.5e9
    stop_hz: float = 31.5e9
    num_freqs: int = 5000

    def linspace(self):
        # local import keeps this module numpy-optional for pure UI use
        import numpy as np
        return np.linspace(self.start_hz, self.stop_hz, self.num_freqs)


@dataclass
class Scenario:
    """Top-level declarative scenario."""
    name: str = "untitled"
    base_scene: str = "munich"              # munich | etoile | <sionna scene name>
    frequency: FrequencyPlan = field(default_factory=FrequencyPlan)
    num_frames: int = 100
    nodes: List[Node] = field(default_factory=list)
    objects: List[SceneObject] = field(default_factory=list)
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ---- convenience accessors -------------------------------------------------
    def nodes_by_role(self, role: NodeRole) -> List[Node]:
        role = NodeRole(role)
        return [n for n in self.nodes if n.role == role]

    @property
    def is_isac(self) -> bool:
        """True when the scenario mixes sensing and communication nodes."""
        roles = {n.role for n in self.nodes}
        has_sense = NodeRole.RADAR in roles
        has_comm = bool({NodeRole.COMM_TX, NodeRole.COMM_RX} & roles)
        return has_sense and has_comm

    def validate(self) -> List[str]:
        """Return a list of human-readable problems; empty list == valid."""
        problems: List[str] = []
        if not self.nodes:
            problems.append("scenario has no nodes")
        names = [n.name for n in self.nodes]
        if len(names) != len(set(names)):
            problems.append("node names must be unique")
        onames = [o.name for o in self.objects]
        if len(onames) != len(set(onames)):
            problems.append("object names must be unique")
        if self.num_frames < 1:
            problems.append("num_frames must be >= 1")
        fp = self.frequency
        if fp.stop_hz <= fp.start_hz:
            problems.append("frequency stop must be greater than start")
        if fp.num_freqs < 1:
            problems.append("num_freqs must be >= 1")
        has_comm_tx = bool(self.nodes_by_role(NodeRole.COMM_TX))
        has_comm_rx = bool(self.nodes_by_role(NodeRole.COMM_RX))
        if has_comm_tx and not has_comm_rx:
            problems.append("comm_tx present but no comm_rx to receive")
        if has_comm_rx and not has_comm_tx:
            problems.append("comm_rx present but no comm_tx to transmit")
        for n in self.nodes:
            a = n.array
            if a.num_rows < 1 or a.num_cols < 1:
                problems.append(
                    f"node '{n.name}' array must have num_rows >= 1 and num_cols >= 1"
                )
            if a.vertical_spacing <= 0 or a.horizontal_spacing <= 0:
                problems.append(
                    f"node '{n.name}' array spacings must be > 0"
                )
            if n.tx_power_dbm is not None and n.role not in (NodeRole.RADAR, NodeRole.COMM_TX):
                problems.append(
                    f"node '{n.name}' has tx_power_dbm but role '{n.role.value}' does not transmit"
                )
        return problems

    # ---- (de)serialization -----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Scenario":
        freq = FrequencyPlan(**d.get("frequency", {}))

        def _array(a: Dict[str, Any]) -> ArrayConfig:
            a = dict(a)
            if "pattern" in a:
                a["pattern"] = AntennaPattern(a["pattern"])
            if "polarization" in a:
                a["polarization"] = Polarization(a["polarization"])
            return ArrayConfig(**a)

        def _motion(m: Dict[str, Any]) -> Motion:
            m = dict(m)
            m["velocity"] = tuple(m.get("velocity", (0.0, 0.0, 0.0)))
            m["waypoints"] = [tuple(w) for w in m.get("waypoints", [])]
            return Motion(**m)

        nodes = []
        for n in d.get("nodes", []):
            n = dict(n)
            n["role"] = NodeRole(n.get("role", "radar"))
            n["position"] = tuple(n.get("position", (0.0, 0.0, 0.0)))
            if n.get("look_at") is not None:
                n["look_at"] = tuple(n["look_at"])
            n["array"] = _array(n.get("array", {}))
            n["motion"] = _motion(n.get("motion", {}))
            nodes.append(Node(**n))

        objects = []
        for o in d.get("objects", []):
            o = dict(o)
            o["kind"] = ObjectKind(o.get("kind", "sphere"))
            o["position"] = tuple(o.get("position", (0.0, 0.0, 0.0)))
            if o.get("color") is not None:
                o["color"] = tuple(o["color"])
            o["motion"] = _motion(o.get("motion", {}))
            objects.append(SceneObject(**o))

        return cls(
            name=d.get("name", "untitled"),
            base_scene=d.get("base_scene", "munich"),
            frequency=freq,
            num_frames=d.get("num_frames", 100),
            nodes=nodes,
            objects=objects,
            description=d.get("description", ""),
            metadata=d.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, s: str) -> "Scenario":
        return cls.from_dict(json.loads(s))

    @classmethod
    def load(cls, path: str) -> "Scenario":
        with open(path) as f:
            return cls.from_json(f.read())


# --------------------------------------------------------------------------------
# Reference scenarios. These double as defaults for the UI and as the inputs the
# example scripts expect. They mirror the existing hardcoded munich radar setup and
# add the multi-node radar+comm (ISAC) case.
# --------------------------------------------------------------------------------

def munich_radar_scenario() -> Scenario:
    """Single 32x32 monostatic radar node sweeping across the Munich square.

    Reproduces the setup currently hardcoded in sionna_simple_channel.py.
    """
    return Scenario(
        name="munich_radar",
        base_scene="munich",
        frequency=FrequencyPlan(carrier_hz=30e9, start_hz=28.5e9, stop_hz=31.5e9, num_freqs=5000),
        num_frames=100,
        nodes=[
            Node(
                name="radar",
                role=NodeRole.RADAR,
                position=(45.0, 90.0, 1.5),
                array=ArrayConfig(num_rows=32, num_cols=32, pattern=AntennaPattern.ISO),
                motion=Motion(velocity=(1.0, 0.0, 0.0)),
                tx_power_dbm=12.0,
            ),
        ],
        description="Monostatic 32x32 radar translating across the Munich square.",
    )


def munich_isac_scenario() -> Scenario:
    """Joint radar + communications: a radar on a (moving) car and a comm link to a
    building-mounted array, operating in the same scene at the same time."""
    return Scenario(
        name="munich_isac",
        base_scene="munich",
        frequency=FrequencyPlan(carrier_hz=30e9, start_hz=28.5e9, stop_hz=31.5e9, num_freqs=5000),
        num_frames=100,
        nodes=[
            Node(
                name="car_radar",
                role=NodeRole.RADAR,
                position=(45.0, 90.0, 1.5),
                array=ArrayConfig(num_rows=32, num_cols=32, pattern=AntennaPattern.ISO),
                motion=Motion(velocity=(1.0, 0.0, 0.0)),
                params={"waveform": "fmcw"},
                tx_power_dbm=12.0,
            ),
            Node(
                name="building_comm_tx",
                role=NodeRole.COMM_TX,
                position=(8.5, 21.0, 27.0),
                look_at=(45.0, 90.0, 1.5),
                array=ArrayConfig(num_rows=1, num_cols=1, pattern=AntennaPattern.TR38901),
                params={"waveform": "ofdm"},
                tx_power_dbm=12.0,
            ),
            Node(
                name="car_comm_rx",
                role=NodeRole.COMM_RX,
                position=(45.0, 90.0, 1.5),
                array=ArrayConfig(num_rows=4, num_cols=4, pattern=AntennaPattern.ISO),
                motion=Motion(velocity=(1.0, 0.0, 0.0)),
            ),
        ],
        objects=[
            SceneObject(name=f"car-{i}", kind=ObjectKind.SPHERE, scaling=5.0,
                        position=(-127.0 + 20 * i, 37.0, 1.5), color=(0.8, 0.1, 0.1))
            for i in range(4)
        ],
        description="ISAC: car-mounted radar and a building->car communications link sharing the scene.",
    )


def etoile_radar_scenario() -> Scenario:
    """Single 32x32 monostatic radar node translating along an avenue toward the
    Arc de Triomphe roundabout in the Sionna 'etoile' scene.

    Generate with --out e2e/environment/sionna_sims/etoile.pkl to make the runtime
    'etoile' base scene loadable.
    """
    return Scenario(
        name="etoile_radar",
        base_scene="etoile",
        frequency=FrequencyPlan(carrier_hz=30e9, start_hz=28.5e9, stop_hz=31.5e9, num_freqs=5000),
        num_frames=100,
        nodes=[
            Node(
                name="radar",
                role=NodeRole.RADAR,
                position=(60.0, 0.0, 1.5),
                array=ArrayConfig(num_rows=32, num_cols=32, pattern=AntennaPattern.ISO),
                motion=Motion(velocity=(-0.5, 0.0, 0.0)),
                tx_power_dbm=12.0,
            ),
        ],
        description="Monostatic 32x32 radar translating along an avenue toward the "
                    "Etoile roundabout.",
    )


def munich_patrol_scenario() -> Scenario:
    """Motion-rich radar patrol through the Munich square.

    Exercises the full motion API: the radar follows an L-shaped multi-leg waypoint
    track (overriding velocity), while the scatterer cars separately exercise plain
    translation, pure rotation (orbiting the scene centroid) and combined
    translation+rotation (curving heading about their own origin).
    """
    return Scenario(
        name="munich_patrol",
        base_scene="munich",
        frequency=FrequencyPlan(carrier_hz=30e9, start_hz=28.5e9, stop_hz=31.5e9, num_freqs=5000),
        num_frames=100,
        nodes=[
            Node(
                name="patrol_radar",
                role=NodeRole.RADAR,
                position=(45.0, 90.0, 1.5),
                array=ArrayConfig(num_rows=32, num_cols=32, pattern=AntennaPattern.ISO),
                motion=Motion(waypoints=[
                    (45.0, 40.0, 1.5),
                    (90.0, 40.0, 1.5),
                    (90.0, 90.0, 1.5),
                ]),
                params={"waveform": "fmcw"},
                tx_power_dbm=12.0,
            ),
        ],
        objects=[
            SceneObject(name="car-0", kind=ObjectKind.SPHERE, scaling=5.0,
                        position=(-127.0, 37.0, 1.5), color=(0.8, 0.1, 0.1),
                        motion=Motion(velocity=(0.5, 0.0, 0.0))),
            SceneObject(name="car-1", kind=ObjectKind.SPHERE, scaling=5.0,
                        position=(-107.0, 37.0, 1.5), color=(0.8, 0.1, 0.1),
                        motion=Motion(velocity=(0.0, 0.4, 0.0))),
            SceneObject(name="car-2", kind=ObjectKind.SPHERE, scaling=5.0,
                        position=(-87.0, 37.0, 1.5), color=(0.8, 0.1, 0.1),
                        motion=Motion(angular_velocity_deg=6.0)),
            SceneObject(name="car-3", kind=ObjectKind.SPHERE, scaling=5.0,
                        position=(-67.0, 37.0, 1.5), color=(0.8, 0.1, 0.1),
                        motion=Motion(velocity=(0.3, 0.3, 0.0), angular_velocity_deg=3.0)),
        ],
        description="Radar patrol on an L-shaped waypoint track through the Munich "
                    "square, with cars exercising translation, rotation and combined motion.",
    )


REFERENCE_SCENARIOS = {
    "munich_radar": munich_radar_scenario,
    "munich_isac": munich_isac_scenario,
    "etoile_radar": etoile_radar_scenario,
    "munich_patrol": munich_patrol_scenario,
}


if __name__ == "__main__":
    for name, factory in REFERENCE_SCENARIOS.items():
        sc = factory()
        problems = sc.validate()
        status = "OK" if not problems else f"PROBLEMS: {problems}"
        print(f"[{name}] {len(sc.nodes)} nodes, {len(sc.objects)} objects, "
              f"isac={sc.is_isac}  -> {status}")
        # round-trip check
        assert Scenario.from_json(sc.to_json()).to_dict() == sc.to_dict(), "round-trip failed"
    print("scenario round-trip OK")
