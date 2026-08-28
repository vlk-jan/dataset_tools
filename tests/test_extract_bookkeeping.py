"""Bag discovery, scene selection, skip logic and scenes.csv merging."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from dataset_tools import extract_data as ed

from conftest import write_scene


def make_bag(root: Path, name: str) -> Path:
    bag = root / name
    bag.mkdir(parents=True)
    (bag / "metadata.yaml").write_text("rosbag2_bagfile_information:\n")
    (bag / f"{name}_0.mcap").write_bytes(b"")
    return bag


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
def test_merge_scenes_csv_from_worker_parts(tmp_path: Path):
    parts = tmp_path / ed.SCENES_PARTS_DIR
    for idx, bag, frames in ((1, "bag_b", 20), (0, "bag_a", 10)):
        ed.write_scene_part(
            parts,
            ed.SceneResult(
                index=idx,
                bag=bag,
                status=ed.STATUS_OK,
                frames=frames,
                start_time=1_700_000_000.0 + idx,
            ),
        )
    csv_path = ed.merge_scenes_csv(tmp_path)

    lines = csv_path.read_text().splitlines()
    assert lines[0] == "scene,bag,start_time,frames"
    assert lines[1] == "0000,bag_a,1700000000.000000,10"
    assert lines[2] == "0001,bag_b,1700000001.000000,20"
    assert not parts.exists()  # merged parts are cleaned up


def test_merge_scenes_csv_recovers_rows_of_skipped_scenes(tmp_path: Path):
    write_scene(tmp_path / "0000", frames=4, start_time=1_700_000_000.0)
    write_scene(tmp_path / "0001", frames=2, start_time=1_700_000_500.0)
    (tmp_path / "0002").mkdir()  # no source_bag.txt -> not a finished scene

    rows = ed.merge_scenes_csv(tmp_path).read_text().splitlines()
    assert len(rows) == 3
    assert rows[1] == "0000,synthetic_bag,1700000000.000000,4"
    assert rows[2] == "0001,synthetic_bag,1700000500.000000,2"


def test_merge_scenes_csv_prefers_fresh_worker_rows(tmp_path: Path):
    write_scene(tmp_path / "0000", frames=4)
    ed.write_scene_part(
        tmp_path / ed.SCENES_PARTS_DIR,
        ed.SceneResult(
            index=0, bag="fresh_bag", status=ed.STATUS_OK, frames=99, start_time=1.0
        ),
    )
    rows = ed.merge_scenes_csv(tmp_path).read_text().splitlines()
    assert rows[1] == "0000,fresh_bag,1.000000,99"


def test_write_scene_part_handles_unknown_start_time(tmp_path: Path):
    parts = tmp_path / ed.SCENES_PARTS_DIR
    ed.write_scene_part(
        parts, ed.SceneResult(index=7, bag="bag", status=ed.STATUS_EMPTY)
    )
    assert (parts / "0007.csv").read_text().strip() == "0007,bag,,0"


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


@pytest.mark.parametrize("jobs", ["1", "2"])
def test_main_reports_failures_with_nonzero_exit(tmp_path: Path, jobs: str):
    # A directory with a metadata.yaml but no readable bag makes the reader fail.
    make_bag(tmp_path / "bags", "broken_bag_a")
    make_bag(tmp_path / "bags", "broken_bag_b")
    out = tmp_path / "out"
    assert ed.main([str(tmp_path / "bags"), str(out), "--jobs", jobs]) == 1
    # ... and the index only lists scenes that actually finished.
    assert (out / ed.SCENES_CSV).read_text().splitlines() == [
        "scene,bag,start_time,frames"
    ]


def test_worker_turns_exceptions_into_failed_results(tmp_path: Path):
    args = ed.build_parser().parse_args([str(tmp_path), str(tmp_path)])
    result = ed._worker(tmp_path / "missing_bag", tmp_path / "out", 3, args)
    assert result.status == ed.STATUS_FAILED
    assert result.index == 3 and result.message
    # Failed scenes must not end up in scenes.csv.
    assert not (tmp_path / "out" / ed.SCENES_PARTS_DIR / "0003.csv").exists()
