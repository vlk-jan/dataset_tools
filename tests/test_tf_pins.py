"""dataset_tools.tf_pins: one constant for an extrinsic the sensor keeps re-estimating."""

import math

import numpy as np

from dataset_tools import tf_pins
from dataset_tools.extract_data import build_tf_graph, get_transform
from dataset_tools.transforms import euler_matrix


class _TF:
    def __init__(self, parent, child, xyz, rpy_deg=(0.0, 0.0, 0.0)):
        M = euler_matrix(*(math.radians(a) for a in rpy_deg))
        q = _quat(M)

        class _O:
            pass

        self.header, self.child_frame_id, self.transform = _O(), child, _O()
        self.header.frame_id = parent
        self.transform.translation, self.transform.rotation = _O(), _O()
        self.transform.translation.x, self.transform.translation.y, self.transform.translation.z = xyz
        self.transform.rotation.x, self.transform.rotation.y, self.transform.rotation.z, self.transform.rotation.w = q


def _quat(M):
    R = M[:3, :3]
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return x, y, z, w


def _odin_graph(lidar_xyz=(0.02, 0.0, 0.03), lidar_rpy=(0.0, -1.5, 0.0)):
    return build_tf_graph(
        [
            _TF("base_link", "odin1_base_link", (0.1, 0.0, 0.1)),
            _TF("odin1_base_link", "imu_link", (0.0, 0.0, 0.0)),
            _TF("imu_link", "odin1_lidar", lidar_xyz, lidar_rpy),
            _TF("odin1_lidar", "odin1_camera_link", (0.0, 0.04, 0.01), (0.4, 0.0, 0.0)),
        ]
    )


def test_pin_matrix_is_the_tf_pose():
    pin = tf_pins.Pin("a", "b", (1.0, 2.0, 3.0), (10.0, -20.0, 30.0), "test")
    M = pin.matrix()
    assert np.allclose(M[:3, 3], (1.0, 2.0, 3.0))
    assert np.allclose(tf_pins.rpy_deg(M), (10.0, -20.0, 30.0))


def test_apply_replaces_both_directions_and_records_the_bag():
    G = _odin_graph()
    records = tf_pins.apply(G)
    assert [r.pin.edge for r in records] == [p.edge for p in tf_pins.PINS]
    lidar = records[0]
    assert np.allclose(lidar.bag_xyz, (0.02, 0.0, 0.03))
    assert np.allclose(lidar.bag_rpy_deg, (0.0, -1.5, 0.0), atol=1e-9)
    pin = next(p for p in tf_pins.PINS if p.edge == ("imu_link", "odin1_lidar"))
    assert np.allclose(G["odin1_lidar"]["imu_link"]["transform"], pin.matrix())
    assert np.allclose(G["imu_link"]["odin1_lidar"]["transform"], np.linalg.inv(pin.matrix()))
    # The chain to base_link is the URDF mount composed with the pin, whatever the bag said.
    mount = np.eye(4)
    mount[:3, 3] = (0.1, 0.0, 0.1)
    assert np.allclose(get_transform(G, "odin1_lidar", "base_link"), mount @ pin.matrix())


def test_far_is_by_translation_or_rotation():
    pin = tf_pins.Pin("a", "b", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), "test")
    near = tf_pins.Record(pin, (0.01, 0.01, 0.01), (0.2, 0.2, 0.0))
    assert not near.far
    assert tf_pins.Record(pin, (0.0, 0.0, 0.04), (0.0, 0.0, 0.0)).far
    assert tf_pins.Record(pin, (0.0, 0.0, 0.0), (0.0, 1.5, 0.0)).far
    assert abs(tf_pins.Record(pin, (0.0, 0.0, 0.0), (0.0, 1.5, 0.0)).rotation_deg - 1.5) < 1e-6


def test_apply_touches_nothing_without_the_edges():
    G = build_tf_graph([_TF("base_link", "lidar_frame", (0.17, 0.0, 0.145))])
    before = {e: G[e[0]][e[1]]["transform"].copy() for e in G.edges}
    assert tf_pins.apply(G) == []
    assert all(np.array_equal(before[e], G[e[0]][e[1]]["transform"]) for e in G.edges)


def test_report_round_trips(tmp_path):
    records = tf_pins.apply(_odin_graph())
    path = tmp_path / tf_pins.FILE
    tf_pins.write_report(path, records)
    rows = [line.split() for line in path.read_text().splitlines() if not line.startswith("#")]
    assert len(rows) == 2 * len(records)
    for r, (pin_row, bag_row) in zip(records, zip(rows[::2], rows[1::2], strict=True), strict=True):
        assert pin_row[:3] == [r.pin.parent, r.pin.child, "pin"]
        assert bag_row[:3] == [r.pin.parent, r.pin.child, "bag"]
        assert np.allclose([float(v) for v in pin_row[3:6]], r.pin.xyz)
        assert np.allclose([float(v) for v in bag_row[3:6]], r.bag_xyz, atol=1e-4)
    assert "pin is the" in path.read_text()


def test_pins_are_the_documented_medians():
    """The constants match the survey they were taken from (docs/odin_tf_static_2026-09-30.txt)."""
    from pathlib import Path

    doc = Path(__file__).resolve().parents[1] / "docs" / "odin_tf_static_2026-09-30.txt"
    rows = [line.split() for line in doc.read_text().splitlines() if line and not line.startswith("#")]
    for pin in tf_pins.PINS:
        vals = np.array([[float(v) for v in r[2:8]] for r in rows if r[1] == f"{pin.parent}->{pin.child}"])
        assert len(vals) >= 40
        med = np.median(vals, axis=0)
        assert np.allclose(med[:3], pin.xyz, atol=5e-4)
        assert np.allclose(med[3:], pin.rpy_deg, atol=5e-3)
