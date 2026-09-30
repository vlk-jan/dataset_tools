"""Bag discovery, scene selection, skip logic and scenes.csv merging."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import write_scene

from dataset_tools import extract_data as ed


def make_bag(root: Path, name: str) -> Path:
    bag = root / name
    bag.mkdir(parents=True)
    (bag / "metadata.yaml").write_text("rosbag2_bagfile_information:\n")
    (bag / f"{name}_0.mcap").write_bytes(b"")
    return bag


# ------------------------------------------------------------------ parser
def test_a_bad_time_align_value_is_refused_by_the_parser(tmp_path: Path, capsys):
    """Refused before any bag is opened: raised inside a worker it would take
    the whole process pool down with a BrokenProcessPool nobody can read."""
    with pytest.raises(SystemExit) as exc:
        ed.build_parser().parse_args([str(tmp_path), str(tmp_path), "--time_align", "soon"])
    assert exc.value.code == 2
    assert "'soon'" in capsys.readouterr().err
    assert ed.time_align_mode(" Auto ") == "auto"
    assert ed.time_align_mode("-0.69") == "-0.69"


# --------------------------------------------------------------------- bags
def test_find_bags_requires_metadata_yaml(tmp_path: Path):
    make_bag(tmp_path, "bag_b")
    make_bag(tmp_path, "bag_a")
    (tmp_path / "notes").mkdir()  # no metadata.yaml -> ignored
    (tmp_path / "loose.mcap").write_bytes(b"")

    assert [b.name for b in ed.find_bags(tmp_path)] == ["bag_a", "bag_b"]


def test_find_bags_accepts_a_single_bag_directory(tmp_path: Path):
    bag = make_bag(tmp_path, "one_bag")
    assert ed.find_bags(bag) == [bag]


def test_find_bags_on_empty_directory(tmp_path: Path):
    assert ed.find_bags(tmp_path) == []


def test_find_bags_missing_path(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        ed.find_bags(tmp_path / "nope")


# ---------------------------------------------------------------- selection
def test_scene_selection_none_selects_everything():
    assert ed.parse_scene_selection(None) is None
    assert ed.parse_scene_selection([]) is None


@pytest.mark.parametrize(
    "tokens, expected",
    [
        (["3"], [3]),
        (["1", "4"], [1, 4]),
        (["1,4"], [1, 4]),
        (["2-5"], [2, 3, 4, 5]),
        (["0,7-9"], [0, 7, 8, 9]),
        (["5-3"], [3, 4, 5]),  # reversed range
    ],
)
def test_scene_selection_indices(tokens, expected):
    selected = ed.parse_scene_selection(tokens)
    assert [i for i in range(10) if selected(i, "bag")] == expected


def test_scene_selection_by_bag_name():
    selected = ed.parse_scene_selection(["26-08-21-stromovka-roads_0", "4"])
    assert selected(0, "26-08-21-stromovka-roads_0")
    assert selected(4, "anything")
    assert not selected(1, "other_bag")


# ------------------------------------------------------------- completeness
def test_scene_is_complete(tmp_path: Path):
    scene = write_scene(tmp_path / "0000", frames=3)
    assert ed.scene_is_complete(scene)

    (scene / ed.SENTINEL_FILE).unlink()  # crashed before finishing
    assert not ed.scene_is_complete(scene)


def test_scene_is_incomplete_on_count_mismatch(tmp_path: Path):
    scene = write_scene(tmp_path / "0000", frames=3)
    (scene / "clouds" / "00002.npy").unlink()
    assert not ed.scene_is_complete(scene)


def test_scene_is_incomplete_when_empty(tmp_path: Path):
    scene = tmp_path / "0000"
    (scene / "images").mkdir(parents=True)
    (scene / "clouds").mkdir()
    (scene / "data_ts.txt").write_text("")
    (scene / "pcd_ts.txt").write_text("")
    (scene / ed.SENTINEL_FILE).write_text("bag\n")
    assert not ed.scene_is_complete(scene)


def test_missing_scene_dir_is_incomplete(tmp_path: Path):
    assert not ed.scene_is_complete(tmp_path / "9999")


# ---------------------------------------------------------------- scenes.csv
def test_merge_scenes_csv_recovers_rows_of_skipped_scenes(tmp_path: Path):
    write_scene(tmp_path / "0000", frames=4, start_time=1_700_000_000.0)
    write_scene(tmp_path / "0001", frames=2, start_time=1_700_000_500.0)
    (tmp_path / "0002").mkdir()  # no source_bag.txt -> not a finished scene

    rows = ed.merge_scenes_csv(tmp_path).read_text().splitlines()
    assert len(rows) == 3
    assert rows[1] == "0000,synthetic_bag,1700000000.000000,4"
    assert rows[2] == "0001,synthetic_bag,1700000500.000000,2"


def test_merge_scenes_csv_ignores_non_scene_entries(tmp_path: Path):
    write_scene(tmp_path / "0000", frames=4, start_time=1_700_000_000.0)
    (tmp_path / "notes").mkdir()
    (tmp_path / "0001.txt").write_text("")
    rows = ed.merge_scenes_csv(tmp_path).read_text().splitlines()
    assert rows == ["scene,bag,start_time,frames", "0000,synthetic_bag,1700000000.000000,4"]


# -------------------------------------------------------------------- CLI
def test_cli_is_backwards_compatible():
    args = ed.build_parser().parse_args(
        [
            "in",
            "out",
            "--cam",
            "/luxonis/oak/rgb/image_raw/compressed",
            "--lidar",
            "/livox/lidar_front",
            "--odom",
            "/odom",
            "--imu",
            "/ekf/imu/data",
            "--base_frame",
            "base_link",
            "--lidar_extrinsic",
            "0.1",
            "0",
            "0.2",
            "0",
            "0",
            "0",
            "--jobs",
            "8",
        ]
    )
    assert (args.input_dir, args.output_dir) == ("in", "out")
    assert args.lidar == "/livox/lidar_front"
    assert args.lidar_extrinsic == [0.1, 0.0, 0.2, 0.0, 0.0, 0.0]
    assert args.jobs == 8
    # New flags default to the previous behaviour.
    assert args.overwrite is False and args.scenes is None and args.start_index == 0


def test_main_reports_missing_input(tmp_path: Path):
    assert ed.main([str(tmp_path / "nope"), str(tmp_path / "out")]) == 2


def test_main_reports_no_bags(tmp_path: Path):
    (tmp_path / "in").mkdir()
    assert ed.main([str(tmp_path / "in"), str(tmp_path / "out")]) == 2


def test_main_skips_complete_scenes_and_writes_index(tmp_path: Path):
    make_bag(tmp_path / "bags", "bag_a")
    out = tmp_path / "out"
    write_scene(out / "0000", frames=3)  # pretend it was extracted already

    assert ed.main([str(tmp_path / "bags"), str(out), "--jobs", "1"]) == 0
    rows = (out / ed.SCENES_CSV).read_text().splitlines()
    assert rows[1].startswith("0000,synthetic_bag,")
    # The skipped scene was never measured, so the disk figures pick it up.
    usage = json.loads((out / "disk_usage.json").read_text())
    assert list(usage["scenes"]) == ["0000"] and usage["scenes"]["0000"]["frames"] == 3


@pytest.mark.parametrize("jobs", ["1", "2"])
def test_main_reports_failures_with_nonzero_exit(tmp_path: Path, jobs: str):
    # A directory with a metadata.yaml but no readable bag makes the reader fail.
    make_bag(tmp_path / "bags", "broken_bag_a")
    make_bag(tmp_path / "bags", "broken_bag_b")
    out = tmp_path / "out"
    assert ed.main([str(tmp_path / "bags"), str(out), "--jobs", jobs]) == 1
    # ... and the index only lists scenes that actually finished.
    assert (out / ed.SCENES_CSV).read_text().splitlines() == ["scene,bag,start_time,frames"]


def test_worker_turns_exceptions_into_failed_results(tmp_path: Path):
    args = ed.build_parser().parse_args([str(tmp_path), str(tmp_path)])
    result = ed._worker(tmp_path / "missing_bag", tmp_path / "out", 3, args)
    assert result.status == ed.STATUS_FAILED
    assert result.index == 3 and result.message
    # A failed scene leaves no sentinel, so it cannot end up in scenes.csv.
    assert not (tmp_path / "out" / "0003" / ed.SENTINEL_FILE).exists()
