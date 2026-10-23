"""Synthetic dataset roots for the viewer tests (fixtures live in ``conftest.py``)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest


def viewer_client(root: Path, cache_dir: Path, **app_kw):
    """A TestClient over ``root``; skips when the viewer extra is absent."""
    for mod in ("simview", "torch", "fastapi"):
        pytest.importorskip(mod)
    from fastapi.testclient import TestClient

    from dataset_tools.viewer.app import create_app

    return TestClient(create_app(root, cache_dir=cache_dir, **app_kw))


def scene_access(tmp_path: Path, root_path: Path, scene: str = "0000"):
    """A ``SceneAccess`` on ``root_path``, its caches under ``tmp_path``."""
    from dataset_tools.viewer import probe
    from dataset_tools.viewer.build import SceneAccess

    return SceneAccess(probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache"), scene)


def card_of(page, title: str) -> dict | None:
    return next((c for c in page["cards"] if c["title"] == title), None)


def value_of(card, label: str):
    return next((s["value"] for s in card["stats"] if s["label"] == label), None)


def inline_blobs(client, base: str, node) -> None:
    """Replace a bundle document's ``/blob/<n>`` references with the bytes from
    ``<base>/blob/<n>``, base64-inlined as SimView's own loader reads them."""
    import base64

    items = node.items() if isinstance(node, dict) else enumerate(node)
    for key, value in list(items):
        if isinstance(value, str) and value.startswith("/blob/"):
            raw = client.get(f"{base}/blob/{value.rsplit('/', 1)[-1]}").content
            node[key] = "__b64__" + base64.b64encode(raw).decode()
        elif isinstance(value, dict | list):
            inline_blobs(client, base, value)


def driven_sidecar():
    """``(driven, max_sim)`` of a driven-similarity frame ([row=x, col=y]): the
    centre cell and a 3x3 patch 1 m ahead driven, similarity falling off with
    range from that patch."""
    rows, cols = np.mgrid[0:128, 0:128]
    driven = np.zeros((128, 128), dtype=bool)
    driven[63:65, 63:65] = True  # under the robot
    driven[72:75, 63:66] = True  # ~1 m ahead
    dist = np.hypot(rows - 73.0, cols - 64.0) * 0.1  # metres from the patch
    max_sim = np.clip(1.0 - dist / 3.0, 0.0, 1.0)
    max_sim[driven] = 1.0
    return driven, max_sim.astype(np.float16)


def write_image(path: Path, size=(16, 10), rgb=(40, 80, 120)) -> None:
    """A flat ``rgb`` image ``size = (width, height)`` px; the format from the suffix."""
    cv2.imwrite(str(path), np.full((size[1], size[0], 3), rgb[::-1], np.uint8))


def decode_rgb(data: bytes) -> np.ndarray:
    """An encoded image as an ``(height, width, 3)`` RGB array."""
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


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
    with_gps: bool = False,
    with_driven_similarity: bool = False,
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
            write_image(d / "images" / f"{i:05d}.png")
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
        if with_driven_similarity:
            (d / "driven_similarity").mkdir()
            driven, max_sim = driven_sidecar()
            for i in range(frames):
                np.savez_compressed(
                    d / "driven_similarity" / f"{i:05d}.npz",
                    driven=driven,
                    max_sim=max_sim,
                    threshold=np.float32(0.9),
                    res=np.float32(0.1),
                    R_align=np.eye(3, dtype=np.float32),
                    label_horizon=np.float32(4.0),
                    robot="helhest",
                    setprop_ckpt="test.pth",
                    wi_ckpt="wi.pth",
                    feature_dim=np.int32(960),
                    rig="test",
                    dataset_version="test",
                    odometry_frame="base_link",
                )
        if with_gps:
            # Four fixes per frame, walking ENE across ~100 m of Prague.
            gt = np.linspace(ts[0], ts[-1], frames * 4)
            gps = np.zeros((len(gt), 8))
            gps[:, 0] = gt
            gps[:, 1] = 50.109 + 3e-6 * np.arange(len(gt))
            gps[:, 2] = 14.418 + 8e-6 * np.arange(len(gt))
            gps[:, 3] = 227.0
            gps[:, 4] = 2  # RTK / GBAS
            gps[:, 5:8] = [0.09, 0.16, 0.25]
            np.save(d / "gps.npy", gps)
        # A plausible pinhole for a 1280x800 camera, so the cloud-on-image
        # projection has somewhere to land.
        np.savetxt(d / "camera_intrinsics.txt", [[700.0, 0.0, 640.0], [0.0, 700.0, 400.0], [0.0, 0.0, 1.0]])
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


#: Curated thumbnail frames; deliberately not the 10/50/90 % ones (1, 6, 10).
CURATED_THUMBS = {"a": 2, "b": 7, "c": 11}


def write_splits(
    root_path: Path,
    *,
    val=("0000",),
    train=None,
    excluded=None,
    name: str | None = "scene_strat",
    note: str | None = None,
    root: str | None = None,
    dataset_version: str | None = None,
    windows: dict | None = None,
) -> Path:
    """The root-level ``splits.json`` as DRIFT's ``dump_split.py`` writes it.

    ``train=None`` omits ``train_scenes`` (bare hold-out: unnamed scenes train).
    """
    data: dict = {"val_scenes": list(val)}
    if name is not None:
        data["split_mode"] = name
    if note is not None:
        data["note"] = note
    if train is not None:
        data["train_scenes"] = list(train)
    if excluded is not None:
        data["excluded_scenes"] = list(excluded)
    if root is not None:
        data["root"] = root
    if dataset_version is not None:
        data["dataset_version"] = dataset_version
    if windows is not None:
        data["windows_per_scene"] = {k: dict(v) for k, v in windows.items()}
    path = root_path / "splits.json"
    path.write_text(json.dumps(data))
    return path


def write_atlas(root_path: Path, scenes=("0000",), *, dataset_version: str | None = None) -> Path:
    """The optional ``atlas/`` half of a root: descriptions, groups, thumbnails."""
    atlas_dir = root_path / "atlas"
    (atlas_dir / "thumbs").mkdir(parents=True, exist_ok=True)
    slots = ["a", "b", "c"]
    entries = {}
    for scene in scenes:
        for slot in slots:
            write_image(atlas_dir / "thumbs" / f"{scene}{slot}.jpg", size=(64, 40))
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
            "thumbs": dict(CURATED_THUMBS),
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
