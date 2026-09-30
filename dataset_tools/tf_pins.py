"""Static transforms the extractor pins to a constant instead of taking from the bag.

The Odin1 publishes its own lidar-to-IMU and lidar-to-camera extrinsics on
``/tf_static``, and it does not publish a calibration: it publishes an
estimate it keeps refining while it runs. Read straight out of the bags of the
2026-09-14 firmware, ``imu_link -> odin1_lidar`` differs by up to 3 cm per axis
and half a degree between boots, and inside one bag it can take half a dozen
values in the first minutes (``26-09-28 18_03_45`` starts with roll +0.95 deg
and ends at -0.35 deg). Two sensors in one housing have one extrinsic. Read
from the bag, the same ground stands 3.5 cm higher in one scene than in the
next for no physical reason, and every map, anchor and label inherits it.

So the extractor puts these edges at one value for every bag (`PINS`), and
records beside each scene what the bag itself said (`FILE`), so that a bag
whose estimate sat unusually far from the constant can be found. The
constants are the per-axis median of what the driver published over the 42
bags of the 2026-09-14 firmware recorded up to 2026-09-29 (the per-bag values
are in ``docs/odin_tf_static_2026-09-30.txt``): the driver's estimate with its
wandering taken out, not a measurement. Between the 10th and 90th percentile
of the bags the lidar-to-IMU translation spans 2.6, 3.0 and 3.4 cm in x, y and
z, the pitch 0.25 deg. The mount of the
unit on the robot, ``base_link -> odin1_base_link``, is the robot's own URDF
and is not touched.

A pin applies only to an edge the bag has. A bag without the Odin1 (the Livox
rig) has no ``odin1_lidar`` and nothing is added.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dataset_tools.transforms import euler_matrix, roll_pitch_from_matrix

#: The record written beside a scene.
FILE = "tf_pins.txt"


@dataclass(frozen=True)
class Pin:
    """The pose of ``child`` in ``parent``: translation in m, roll/pitch/yaw in degrees ('sxyz')."""

    parent: str
    child: str
    xyz: tuple[float, float, float]
    rpy_deg: tuple[float, float, float]

    def matrix(self) -> np.ndarray:
        """``M`` with ``p_parent = M @ p_child``, what a TF message with these values carries."""
        M = euler_matrix(*(math.radians(a) for a in self.rpy_deg))
        M[:3, 3] = self.xyz
        return M


#: Where the constants come from; the per-bag values are docs/odin_tf_static_2026-09-30.txt.
BASIS = "per-axis median of the driver's last published value in each of 42 bags, 2026-09-14 to 2026-09-29"

PINS: tuple[Pin, ...] = (
    Pin(
        "imu_link",
        "odin1_lidar",
        xyz=(0.011, 0.024, 0.006),
        rpy_deg=(-0.29, -0.53, 0.12),
    ),
    Pin(
        "odin1_lidar",
        "odin1_camera_link",
        xyz=(-0.010, 0.039, 0.014),
        rpy_deg=(0.40, 0.04, 0.06),
    ),
)

#: A bag whose own value sits further than this from the pin is worth a look.
WARN_TRANSLATION_M = 0.03
WARN_ROTATION_DEG = 1.0


@dataclass(frozen=True)
class Record:
    """One pinned edge: the pin, and what the bag published for it."""

    pin: Pin
    bag_xyz: tuple[float, float, float]
    bag_rpy_deg: tuple[float, float, float]

    @property
    def translation_m(self) -> float:
        return float(np.linalg.norm(np.subtract(self.bag_xyz, self.pin.xyz)))

    @property
    def rotation_deg(self) -> float:
        Rb = euler_matrix(*(math.radians(a) for a in self.bag_rpy_deg))[:3, :3]
        Rp = self.pin.matrix()[:3, :3]
        cos = (np.trace(Rp.T @ Rb) - 1) / 2
        return math.degrees(math.acos(max(-1.0, min(1.0, cos))))

    @property
    def far(self) -> bool:
        return self.translation_m > WARN_TRANSLATION_M or self.rotation_deg > WARN_ROTATION_DEG


def rpy_deg(M: np.ndarray) -> tuple[float, float, float]:
    """Roll, pitch, yaw in degrees of a matrix from `euler_matrix` ('sxyz')."""
    roll, pitch = roll_pitch_from_matrix(M)
    yaw = math.atan2(M[1, 0], M[0, 0])
    return tuple(math.degrees(a) for a in (roll, pitch, yaw))


def apply(G: dict[str, dict[str, np.ndarray]]) -> list[Record]:
    """Replace each pinned edge the graph has with its pin; return what the bag had.

    ``G`` is from ``extract_data.build_tf_graph``: ``G[parent][child]`` holds
    ``inv(M)`` and ``G[child][parent]`` holds ``M``, the pose of the child in
    the parent. Both directions are replaced.
    """
    records = []
    for pin in PINS:
        was = G.get(pin.child, {}).get(pin.parent)
        if was is None:
            continue
        records.append(Record(pin, tuple(float(v) for v in was[:3, 3]), rpy_deg(was)))
        M = pin.matrix()
        G[pin.parent][pin.child] = np.linalg.inv(M)
        G[pin.child][pin.parent] = M
    return records


def write_report(path: Path, records: list[Record]) -> None:
    """What was pinned and what the bag said, one edge per row."""
    lines = [
        "# Static transforms the extractor pinned to a constant (dataset_tools.tf_pins).",
        "# Pose of child in parent: x y z in m, roll pitch yaw in degrees ('sxyz').",
        "# 'pin' is what the scene was built with; 'bag' is what the bag's own /tf_static said.",
        f"# far = bag more than {WARN_TRANSLATION_M} m or {WARN_ROTATION_DEG} deg from the pin.",
        "# parent child which x y z roll pitch yaw",
    ]
    for r in records:
        for which, xyz, rpy in (("pin", r.pin.xyz, r.pin.rpy_deg), ("bag", r.bag_xyz, r.bag_rpy_deg)):
            lines.append(
                f"{r.pin.parent} {r.pin.child} {which} "
                + " ".join(f"{v:.4f}" for v in xyz)
                + " "
                + " ".join(f"{v:.3f}" for v in rpy)
            )
        lines.append(
            f"# {r.pin.parent} -> {r.pin.child}: bag {r.translation_m * 1000:.0f} mm, "
            f"{r.rotation_deg:.2f} deg from the pin{' (far)' if r.far else ''}; pin is the {BASIS}"
        )
    path.write_text("\n".join(lines) + "\n")
