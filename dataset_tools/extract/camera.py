"""Camera intrinsics and extrinsic, written once per scene."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from dataset_tools.extract.tf_graph import get_transform

LOG = logging.getLogger("helhest.extract")

#: What ``--cam`` may point at: pass 2 decodes the payload with OpenCV.
IMAGE_MSGTYPES = ("sensor_msgs/msg/CompressedImage",)


def get_first_frame(reader, conn):
    for _, _, raw in reader.messages([conn]):
        return reader.deserialize(raw, conn.msgtype).header.frame_id
    return None


def save_intrinsics(reader, conns, cam_topic: str, scene_dir: Path) -> bool:
    # e.g. /odin1/image/compressed     -> /odin1/camera_info
    #      /cam/image_raw/compressed   -> /cam/camera_info
    candidates = [
        cam_topic.replace("image_raw/compressed", "camera_info"),
        cam_topic.replace("image/compressed", "camera_info"),
        cam_topic.rsplit("/", 2)[0] + "/camera_info",
    ]
    candidates = [c for c in candidates if c != cam_topic]
    conn = next((conns[c] for c in candidates if c in conns), None)
    if conn is None:
        LOG.warning(
            "no camera_info topic for %s (tried %s); camera_intrinsics.txt not written",
            cam_topic,
            ", ".join(candidates),
        )
        return False
    for _, _, raw in reader.messages([conn]):
        msg = reader.deserialize(raw, conn.msgtype)
        np.savetxt(scene_dir / "camera_intrinsics.txt", np.array(msg.k).reshape(3, 3))
        # The full CameraInfo too: K alone loses the distortion.
        write_camera_info_yaml(msg, scene_dir / "camera_info.yaml")
        return True
    LOG.warning("camera_info topic %s carries no messages", conn.topic)
    return False


def write_camera_info_yaml(msg, path: Path) -> None:
    """Dump a sensor_msgs/CameraInfo as camera_info_manager-style YAML (no yaml dep)."""

    def mat(name, rows, cols, data):
        vals = ", ".join(f"{float(v):.10g}" for v in data)
        return f"{name}:\n  rows: {rows}\n  cols: {cols}\n  data: [{vals}]\n"

    d = list(msg.d)
    text = (
        f"image_width: {int(msg.width)}\n"
        f"image_height: {int(msg.height)}\n"
        f"frame_id: {msg.header.frame_id}\n"
        f"camera_name: {msg.header.frame_id}\n"
        f"distortion_model: {msg.distortion_model}\n"
        + mat("camera_matrix", 3, 3, msg.k)
        + mat("distortion_coefficients", 1, len(d), d)
        + mat("rectification_matrix", 3, 3, msg.r)
        + mat("projection_matrix", 3, 4, msg.p)
        + f"binning_x: {int(msg.binning_x)}\nbinning_y: {int(msg.binning_y)}\n"
    )
    path.write_text(text)


def save_camera_extrinsic(tf_graph, cam_frame, base_frame, scene_dir: Path) -> bool:
    """Write T_cam_to_base.txt: p_base = T @ p_cam_optical, from the static TF."""
    if not cam_frame:
        return False
    T = get_transform(tf_graph, cam_frame, base_frame)
    if T is None:
        LOG.warning("no TF path %s -> %s; T_cam_to_base.txt not written", cam_frame, base_frame)
        return False
    np.savetxt(scene_dir / "T_cam_to_base.txt", T)
    return True
