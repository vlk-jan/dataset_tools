"""Synthetic dataset roots for the viewer tests.

``make_root`` writes a scene with a controllable set of optional files;
``write_atlas`` adds the optional curated ``atlas/`` half. Shared by every
``test_viewer_*`` module; the fixtures built on them live in ``conftest.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _write_png(path: Path, size=(16, 10)) -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    Image.new("RGB", size, (40, 80, 120)).save(path)


def make_root(
    tmp_path: Path,
    scenes=("0000",),
    frames: int = 12,
    *,
    with_height_maps: bool = True,
    with_cmd_vel: bool = True,
    with_transforms: bool = True,
    with_gravity: bool = True,
    with_joints: bool = True,
    with_camera_info: bool = True,
    segments: bool = True,
) -> Path:
    """A synthetic dataset root with a controllable set of optional files."""
    root = tmp_path / "root"
    root.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)

    for scene in scenes:
        d = root / scene
        (d / "images").mkdir(parents=True)
        (d / "clouds").mkdir()
        ts = 1_700_000_000.0 + 0.1 * np.arange(frames)
        for i in range(frames):
            _write_png(d / "images" / f"{i:05d}.png")
            cloud = rng.normal(scale=2.0, size=(64, 4)).astype(np.float32)
            cloud[:, 0] = np.abs(cloud[:, 0])  # forward-only, like a front lidar
            np.save(d / "clouds" / f"{i:05d}.npy", cloud)
        np.savetxt(d / "data_ts.txt", ts, fmt="%.6f")
        np.savetxt(d / "pcd_ts.txt", ts - 0.005, fmt="%.6f")

        poses = []
        for i in range(frames):
            yaw = 0.05 * i
            c, s = np.cos(yaw), np.sin(yaw)
            T = np.eye(4)
            T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
            T[:3, 3] = [0.3 * i, 0.02 * i, 0.0]
            poses.append(T)
        with open(d / "odometry.txt", "w") as fh:
            fh.write("# Timestamp(sec) r11 ... 0 0 0 1\n")
            for t, T in zip(ts, poses, strict=False):
                fh.write(" ".join(f"{v:.9f}" for v in [t, *T.reshape(-1)]) + "\n")
        if segments:
            half = frames // 2
            with open(d / "odometry_segments.txt", "w") as fh:
                fh.write("# id start_ts end_ts n_poses reason\n")
                fh.write(f"0 {ts[0]:.6f} {ts[half - 1]:.6f} {half} filtered:pose_jump\n")
                fh.write(f"1 {ts[half]:.6f} {ts[-1]:.6f} {frames - half} filtered:end_of_run\n")

        if with_transforms:
            np.save(d / "transforms.npy", np.stack(poses))
        if with_gravity:
            np.save(d / "gravity_align.npy", np.stack([np.eye(3)] * frames))
        if with_joints:
            jt = np.linspace(ts[0], ts[-1], frames * 4)
            js = np.zeros((len(jt), 10))
            js[:, 0] = jt
            js[:, 1:4] = np.linspace(0, 10, len(jt))[:, None]  # positions
            js[:, 4:7] = 1.5  # velocities
            js[:, 7:10] = 0.5  # efforts
            np.save(d / "joint_states.npy", js)
        if with_cmd_vel:
            ct = np.linspace(ts[0], ts[-1], frames * 2)
            cv = np.zeros((len(ct), 7))
            cv[:, 0] = ct
            cv[:, 1] = 0.4
            cv[:, 6] = 0.1
            np.save(d / "cmd_vel.npy", cv)
        if with_height_maps:
            (d / "height_maps").mkdir()
            (d / "height_masks").mkdir()
            for i in range(frames):
                hm = np.zeros((128, 128), dtype=np.float64)
                hm[:] = 0.01 * np.arange(128)[:, None]
                np.savez_compressed(d / "height_maps" / f"{i:05d}.npz", height_map=hm)
                validity = np.full((128, 128), 2, dtype=np.uint8)
                validity[:64] = 0  # behind the robot: extrapolated
                np.savez_compressed(d / "height_masks" / f"{i:05d}.npz", validity=validity)
        np.savetxt(d / "camera_intrinsics.txt", np.eye(3) * 700.0)
        np.savetxt(d / "T_cam_to_base.txt", np.eye(4))
        if with_camera_info:
            (d / "camera_info.yaml").write_text(CAMERA_INFO)
    return root


#: A ROS camera_info as the extractor writes it: flat scalars, then blocks whose
#: only interesting line is ``data:``.
CAMERA_INFO = """image_width: 640
image_height: 400
frame_id: test_camera_optical_frame
camera_name: test_camera
distortion_model: plumb_bob
camera_matrix:
  rows: 3
  cols: 3
  data: [700.0, 0, 320.0, 0, 700.0, 200.0, 0, 0, 1]
distortion_coefficients:
  rows: 1
  cols: 5
  data: [0.1, -0.2, 0.001, 0.002, 0.0]
binning_x: 0
"""


#: The fixture's curated thumbnail frames. Deliberately NOT the 10/50/90 %
#: fractions of a 12-frame scene (1, 6, 10), so a test can tell which one the
#: viewer used.
CURATED_THUMBS = {"a": 2, "b": 7, "c": 11}


def write_atlas(
    root_path: Path,
    scenes=("0000",),
    *,
    with_thumbs: bool = True,
    legacy_thumbs: bool = False,
    dataset_version: str | None = None,
) -> Path:
    """The optional ``atlas/`` half of a root: descriptions, groups, thumbnails.

    ``legacy_thumbs`` writes the pre-generator spelling, a bare list of slots
    that names no frame -- roots in the wild still carry it.
    """
    atlas_dir = root_path / "atlas"
    (atlas_dir / "thumbs").mkdir(parents=True, exist_ok=True)
    slots = ["a", "b", "c"]
    thumbs = slots if legacy_thumbs else dict(CURATED_THUMBS)
    entries = {}
    for scene in scenes:
        if with_thumbs:
            for slot in slots:
                _write_png(atlas_dir / "thumbs" / f"{scene}{slot}.jpg", size=(64, 40))
        entries[scene] = {
            "description": f"A synthetic place for scene {scene}, written by hand.",
            "date": "2025-09-12 15:13",
            "dur_s": 653.6,
            "frames": 1962,
            "images": 1962,
            "clouds": 1962,
            "pts_mean": 6090,
            "poses": 4580,
            "poses_unf": 4776,
            "segments": 9,
            "seg_lens": [314.3, 93.9, 67.0],
            "seg_med_s": 18.7,
            "gt_span_s": 565.0,
            "cov": 0.867,
            "path_m": 439.5,
            "moving_frac": 0.763,
            "win_med": 0.1207,
            "win_p95": 0.2537,
            "flags": 24,
            "bag_gb": 25.1,
            "session": "synthetic_session",
            "thumbs": thumbs if with_thumbs else {},
        }
    (atlas_dir / "atlas.json").write_text(
        json.dumps(
            {
                "schema": "dataset_tools.viewer/atlas@2",
                "generated": "2026-09-02",
                "dataset_version": dataset_version,
                "title": "Synthetic Atlas",
                "kicker": "test fixture",
                "intro": "One paragraph of curated introduction.",
                "root_note": "A curated note about the root.",
                "footer": "A curated footer.",
                "totals": [{"label": "scenes", "value": str(len(scenes)), "sub": None}],
                "thumbs": {"dir": "thumbs", "slots": slots, "pattern": "{scene}{slot}.jpg"},
                "groups": [
                    {"title": "An outing", "range": "all of it", "blurb": "What happened here.", "scenes": list(scenes)}
                ],
                "scenes": entries,
            }
        )
    )
    return atlas_dir
