"""Shared fixtures. Adds the repository root to sys.path so the tests run
straight from a checkout, without installing the package."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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
    pytest.importorskip("simview")
    pytest.importorskip("torch")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from helpers_viewer import make_root

    from dataset_tools.viewer.app import create_app

    root_path = make_root(tmp_path, scenes=("0000", "0001"), frames=12, with_cmd_vel=False)
    app = create_app(root_path, cache_dir=tmp_path / "cache")
    return TestClient(app)


@pytest.fixture
def atlas_client(tmp_path):
    pytest.importorskip("simview")
    pytest.importorskip("torch")
    from fastapi.testclient import TestClient
    from helpers_viewer import make_root, write_atlas

    from dataset_tools.viewer.app import create_app

    root_path = make_root(tmp_path, scenes=("0000", "0001"), frames=12)
    write_atlas(root_path, scenes=("0000", "0001"))
    return TestClient(create_app(root_path, cache_dir=tmp_path / "cache"))
