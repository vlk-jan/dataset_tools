"""validate_scene on synthetic scenes written to tmp_path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dataset_tools import validate_scene as vs

from conftest import write_scene


def test_clean_scene_passes(scene: Path):
    report = vs.validate_scene(scene)
    assert report.errors == []
    assert report.warnings == []
    assert report.status() == "OK"
    assert report.stats["frames"] == 4
    assert report.stats["points_mean"] == 32
    assert report.stats["cloud_dtype"] == "float32"
    assert report.extras == ["source_bag.txt"]


def test_missing_required_file_is_an_error(scene: Path):
    (scene / "gravity_align.npy").unlink()
    report = vs.validate_scene(scene)
    assert any("gravity_align.npy is missing" in e for e in report.errors)
    assert report.status() == "FAIL"


def test_frame_count_mismatch_is_an_error(scene: Path):
    (scene / "images" / "00003.png").unlink()
    report = vs.validate_scene(scene)
    assert any("frame counts disagree" in e for e in report.errors)


def test_index_gap_is_an_error(scene: Path):
    (scene / "images" / "00001.png").rename(scene / "images" / "00009.png")
    report = vs.validate_scene(scene)
    assert any("contiguous" in e for e in report.errors)


def test_non_monotonic_timestamps_are_an_error(scene: Path):
    ts = np.loadtxt(scene / "data_ts.txt")
    ts[2], ts[3] = ts[3], ts[2]
    np.savetxt(scene / "data_ts.txt", ts)
    report = vs.validate_scene(scene)
    assert any("not monotonic" in e for e in report.errors)


def test_duplicate_timestamps_are_a_warning(scene: Path):
    ts = np.loadtxt(scene / "data_ts.txt")
    ts[2] = ts[1]
    np.savetxt(scene / "data_ts.txt", ts)
    report = vs.validate_scene(scene)
    assert report.errors == []
    assert any("duplicate timestamps" in w for w in report.warnings)


def test_bad_cloud_shape_is_an_error(scene: Path):
    np.save(scene / "clouds" / "00002.npy", np.zeros((10, 3), dtype=np.float32))
    report = vs.validate_scene(scene)
    assert any("expected (N, 4)" in e for e in report.errors)


def test_integer_cloud_dtype_is_an_error(scene: Path):
    np.save(scene / "clouds" / "00002.npy", np.zeros((10, 4), dtype=np.int32))
    report = vs.validate_scene(scene)
    assert any("float dtype" in e for e in report.errors)


def test_float64_clouds_are_accepted(tmp_path: Path):
    # The old HELHEST dataset stores float64 clouds.
    report = vs.validate_scene(write_scene(tmp_path / "0000", dtype=np.float64))
    assert report.status() == "OK"
    assert report.stats["cloud_dtype"] == "float64"


def test_nan_fraction_is_reported_and_warned(scene: Path):
    cloud = np.load(scene / "clouds" / "00000.npy")
    cloud[:16] = np.nan
    np.save(scene / "clouds" / "00000.npy", cloud)
    report = vs.validate_scene(scene)
    assert report.stats["nan_frac"] > 0.0
    assert any("non-finite" in w for w in report.warnings)


def test_odometry_jump_is_a_warning(scene: Path):
    transforms = np.load(scene / "transforms.npy")
    transforms[2, 0, 3] += 5.0
    np.save(scene / "transforms.npy", transforms)
    report = vs.validate_scene(scene, max_jump=1.0)
    assert report.errors == []
    assert any("jumps > 1.0 m" in w for w in report.warnings)
    assert report.stats["max_step_m"] == pytest.approx(5.2, abs=1e-6)


def test_non_orthonormal_transform_is_an_error(scene: Path):
    transforms = np.load(scene / "transforms.npy")
    transforms[1, :3, :3] *= 2.0
    np.save(scene / "transforms.npy", transforms)
    report = vs.validate_scene(scene)
    assert any("orthonormal" in e for e in report.errors)


def test_tilt_statistics_and_warning(scene: Path):
    from dataset_tools.extract_data import euler_matrix

    gravity = np.load(scene / "gravity_align.npy")
    gravity[1] = euler_matrix(np.radians(50.0), 0.0, 0.0)[:3, :3]
    np.save(scene / "gravity_align.npy", gravity)
    report = vs.validate_scene(scene, max_tilt_deg=45.0)
    assert report.stats["tilt_max_deg"] == pytest.approx(50.0, abs=1e-6)
    assert any("tilted > 45.0 deg" in w for w in report.warnings)
    assert vs.validate_scene(scene, max_tilt_deg=60.0).warnings == []


def test_sync_offset_warning(scene: Path):
    pcd_ts = np.loadtxt(scene / "pcd_ts.txt")
    pcd_ts -= 0.2  # data_ts is the midpoint, so this is a 400 ms offset
    np.savetxt(scene / "pcd_ts.txt", pcd_ts)
    report = vs.validate_scene(scene)
    assert any("sync offset" in w for w in report.warnings)


def test_old_layout_extras_are_optional_and_reported(scene: Path):
    (scene / "odometry.txt").write_text("# Timestamp(sec) r11\n")
    (scene / "trajectory.txt").write_text("# Timestamp(sec) r11\n")
    (scene / "T_cam_to_lidar.txt").write_text("1 0 0 0\n")
    (scene / "height_maps").mkdir()
    (scene / "height_masks").mkdir()
    (scene / f"{scene.name}_full.npz").write_bytes(b"")
    report = vs.validate_scene(scene)
    assert report.status() == "OK"
    for name in ("odometry.txt", "trajectory.txt", "height_maps/", "0000_full.npz"):
        assert name in report.extras


def test_strict_promotes_warnings_to_failures(scene: Path):
    ts = np.loadtxt(scene / "data_ts.txt")
    ts[2] = ts[1]
    np.savetxt(scene / "data_ts.txt", ts)
    report = vs.validate_scene(scene)
    assert report.status(strict=False) == "WARN"
    assert report.status(strict=True) == "FAIL"


def test_find_scenes_expands_a_dataset_root(tmp_path: Path):
    for name in ("0000", "0001", "0002"):
        write_scene(tmp_path / name, frames=2)
    (tmp_path / "not_a_scene").mkdir()
    (tmp_path / "scenes.csv").write_text("scene,bag,start_time,frames\n")

    scenes = vs.find_scenes([tmp_path])
    assert [s.name for s in scenes] == ["0000", "0001", "0002"]
    # An explicit scene directory is used as-is.
    assert vs.find_scenes([tmp_path / "0001"]) == [tmp_path / "0001"]
    with pytest.raises(FileNotFoundError):
        vs.find_scenes([tmp_path / "missing"])


def test_main_exit_codes_and_csv(tmp_path: Path, capsys):
    write_scene(tmp_path / "0000", frames=3)
    csv_path = tmp_path / "report.csv"
    assert vs.main([str(tmp_path), "--csv", str(csv_path)]) == 0
    header, row = csv_path.read_text().splitlines()
    assert header.split(",")[:3] == ["scene", "status", "frames"]
    assert row.split(",")[:3] == ["0000", "OK", "3"]

    (tmp_path / "0000" / "transforms.npy").unlink()
    assert vs.main([str(tmp_path)]) == 1
    assert "transforms.npy is missing" in capsys.readouterr().out


def test_main_markdown_table(tmp_path: Path, capsys):
    write_scene(tmp_path / "0000", frames=2)
    assert vs.main([str(tmp_path), "--csv", "-", "--markdown", "-q"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("| scene | status |")
    assert "| 0000 | OK | 2 |" in out.replace(" |  |", " |")
