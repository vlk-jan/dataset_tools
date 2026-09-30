"""Shared fixtures. The tests run straight from a checkout, without installing
the package: pyproject.toml puts the repository root on sys.path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def write_scene(
    scene_dir: Path,
    frames: int = 4,
    points: int = 32,
    dtype=np.float32,
    start_time: float = 1_700_000_000.0,
    period: float = 0.1,
) -> Path:
    """Write a minimal but valid scene (no bags involved)."""
    scene_dir.mkdir(parents=True, exist_ok=True)
    (scene_dir / "images").mkdir(exist_ok=True)
    (scene_dir / "clouds").mkdir(exist_ok=True)

    rng = np.random.default_rng(0)
    for i in range(frames):
        # The validator only counts *.png files, so a stub keeps the test light.
        (scene_dir / "images" / f"{i:05d}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        cloud = rng.normal(size=(points, 4)).astype(dtype)
        np.save(scene_dir / "clouds" / f"{i:05d}.npy", cloud)

    ts = start_time + period * np.arange(frames)
    np.savetxt(scene_dir / "data_ts.txt", ts)
    np.savetxt(scene_dir / "pcd_ts.txt", ts - 0.005)

    transforms = np.tile(np.eye(4), (frames, 1, 1))
    transforms[:, 0, 3] = 0.2 * np.arange(frames)
    np.save(scene_dir / "transforms.npy", transforms)

    np.save(scene_dir / "gravity_align.npy", np.tile(np.eye(3), (frames, 1, 1)))
    np.savetxt(
        scene_dir / "camera_intrinsics.txt",
        np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]),
    )
    (scene_dir / "source_bag.txt").write_text("synthetic_bag\n")
    return scene_dir


@pytest.fixture
def scene(tmp_path: Path) -> Path:
    return write_scene(tmp_path / "0000")


# -- viewer ---------------------------------------------------------------
# The viewer's HTTP fixtures. They skip when its optional extra is absent.


@pytest.fixture
def client(tmp_path):
    from helpers_viewer import make_root, viewer_client

    root_path = make_root(tmp_path, scenes=("0000", "0001"), frames=12, with_cmd_vel=False)
    return viewer_client(root_path, tmp_path / "cache")


@pytest.fixture
def gps_client(tmp_path):
    """The same root, but with gps.npy -- the only thing that turns maps on."""
    from helpers_viewer import make_root, viewer_client

    root_path = make_root(tmp_path, scenes=("0000",), frames=12, with_gps=True)
    return viewer_client(root_path, tmp_path / "cache")


@pytest.fixture
def atlas_client(tmp_path):
    from helpers_viewer import make_root, viewer_client, write_atlas

    root_path = make_root(tmp_path, scenes=("0000", "0001"), frames=12)
    write_atlas(root_path, scenes=("0000", "0001"))
    return viewer_client(root_path, tmp_path / "cache")
