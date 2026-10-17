import json

import numpy as np
import pytest

from dataset_tools import release
from dataset_tools.gt_io import read_trajectory, write_trajectory
from dataset_tools.transforms import euler_matrix


def _scene(root, sc, bag, n=40, tilted=()):
    d = root / sc
    for sub in ("clouds", "images", "height_maps", "height_masks"):
        (d / sub).mkdir(parents=True)
    t = 100.0 + np.arange(n) * 0.1
    for i in range(n):
        for sub, ext in (("clouds", "npy"), ("images", "jpg"), ("height_maps", "npz"), ("height_masks", "npz")):
            (d / sub / f"{i:05d}.{ext}").touch()
    np.savetxt(d / "data_ts.txt", t)
    rows = [
        np.r_[ti, np.eye(4)[:3].ravel() + np.r_[0, 0, 0, 0.1 * i, 0, 0, 0, 0, 0, 0, 0, 0], 0, 0, 0, 1]
        for i, ti in enumerate(t)
    ]
    np.savetxt(d / "odometry.txt", rows, header="t r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz 0 0 0 1")
    (d / "odometry_segments.txt").write_text(
        f"# segment start_ts end_ts n_poses reason\n1 {t[0]} {t[-1]} {n} end_of_run\n"
    )
    np.save(
        d / "gravity_align.npy",
        np.stack([euler_matrix(np.radians(90) if i in tilted else 0, 0, 0)[:3, :3] for i in range(n)]),
    )
    meta = {
        "tool": {"git_commit": "abc", "git_dirty": False},
        "argv": ["x", "a", "b"],
        "frames": {"guard": {"n_frames_trimmed": 1}},
    }
    (d / "extract_meta.json").write_text(json.dumps(meta))
    (d / "source_bag.txt").write_text(bag + "\n")


def test_a_release_from_its_spec_and_its_scenes(tmp_path):
    root = tmp_path / "root"
    _scene(root, "0000", "bag_a")
    _scene(root, "0001", "bag_b", tilted={30})
    (tmp_path / "bags.csv").write_text(
        "scene,session,bag_dir_on_subtdata,bag_size_gb,extract_args\n0000,s1,/d/s1/bag_a,1,\n0001,s1,/d/s1/bag_b,1,--trim_idle 60 10\n"
    )
    (tmp_path / "rig.json").write_text(json.dumps({"rig": "test_rig", "camera": "a camera"}))
    spec = {
        "inherit": "rig.json",
        "dataset_version": "9.9",
        "title": "Test",
        "bags": str(tmp_path / "bags.csv"),
        "held_out": {"scenes": ["0001"], "decision": "because"},
        "sessions": [{"date": "2026-01-01", "scenes": "0000-0001", "site": "here"}],
        "curation": {"title": "Test", "scenes": {"0000": {"description": "a"}}},
    }
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    release.build(root, tmp_path / "spec.json")

    m = json.loads((root / "manifest.json").read_text())
    assert m["rig"] == "test_rig" and m["camera"] == "a camera" and m["dataset_version"] == "9.9"
    s0, s1 = m["scenes"]["0000"], m["scenes"]["0001"]
    assert s0["frames"] == 40 and abs(s0["path_m"] - 3.9) < 1e-6 and s0["storage"] == "real files"
    assert s1["held_out"] and s1["extract_args"] == "--trim_idle 60 10" and not s0["held_out"]
    assert m["defects"]["tilted_past_60deg"]["0001"]["count"] == 1
    assert m["keyframes"]["files"] == {"keyframes.lst": 49, "keyframes_train.lst": 40, "keyframes_heldout.lst": 9}
    splits = json.loads((root / "splits.json").read_text())
    assert splits["val_scenes"] == ["0001"] and splits["train_scenes"] == ["0000"]
    assert json.loads((root / "atlas" / "curation.json").read_text())["title"] == "Test"
    assert "| 0001 | val | bag_b |" in (root / "README.md").read_text()
    assert (root / "scenes.csv").exists()


def test_a_repeated_odometry_stamp_is_no_speed(tmp_path):
    _scene(tmp_path, "0000", "bag")
    d = tmp_path / "0000"
    ts, poses = read_trajectory(d / "odometry.txt")
    ts[10], poses[10] = ts[9], poses[9]
    write_trajectory(d / "odometry.txt", ts, poses)
    s = release.survey_scene(d)
    assert s["max_speed_mps"] == 1.0
    with pytest.raises(ValueError):
        release._write_json(tmp_path / "x.json", {"x": float("nan")})
