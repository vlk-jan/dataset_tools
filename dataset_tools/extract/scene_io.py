"""What a scene directory holds, whether it is finished, and the root's ``scenes.csv``."""

from __future__ import annotations

import csv
import logging
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from dataset_tools import image_format, tf_pins

LOG = logging.getLogger("helhest.extract")

#: Written as the very last step of a scene, so its presence marks completion.
SENTINEL_FILE = "source_bag.txt"
SCENES_CSV = "scenes.csv"
SCENES_CSV_HEADER = ["scene", "bag", "start_time", "frames"]

STATUS_OK = "ok"
STATUS_SKIPPED = "skipped"
STATUS_EMPTY = "empty"
STATUS_FAILED = "failed"

#: Everything one extraction run writes into a scene directory. A re-extraction
#: (``--overwrite``, or resuming a scene that never finished) removes these
#: first, so frames of a previous run cannot outlive it and skew the counts.
SCENE_OUTPUT_DIRS = ("images", "clouds")
#: The record of what the frame clock was corrected by, and on what evidence.
TIME_ALIGN_FILE = "time_align.txt"
#: The record of what the pose guard threw away, and why (gnss_guard).
GNSS_GUARD_FILE = "gnss_guard.txt"
#: What this run was, what the bag held, and what the run made of it.
META_FILE = "extract_meta.json"
#: The joint order of the joint_states.npy / joint_setpoints.npy columns.
JOINT_NAMES_FILE = "joint_names.txt"
SCENE_OUTPUT_FILES = (
    "data_ts.txt",
    "pcd_ts.txt",
    "transforms.npy",
    "gravity_align.npy",
    "camera_intrinsics.txt",
    "camera_info.yaml",
    "T_cam_to_base.txt",
    "joint_states.npy",
    "joint_setpoints.npy",
    "cmd_vel.npy",
    "gps.npy",
    TIME_ALIGN_FILE,
    GNSS_GUARD_FILE,
    tf_pins.FILE,
    META_FILE,
    JOINT_NAMES_FILE,
    SENTINEL_FILE,
)
#: Written only when the odometry source is its own ground truth (``--gt``).
GROUND_TRUTH_FILES = ("odometry.txt", "odometry_segments.txt")
#: Built after the extraction, one file per frame index (DRIFT and friends).
#: The extractor never writes them, so a re-extraction that changes the frame
#: clock leaves them describing frames that no longer exist; it moves them
#: aside instead (see ``scene._set_aside_stale``). ``regen/`` is not here: it is
#: keyed by time, not by frame index.
DERIVED_DIRS = ("height_maps", "height_masks", "proprio_labels", "driven_similarity")


@dataclass
class SceneResult:
    """Outcome of one bag -> scene extraction (returned by worker processes)."""

    index: int
    bag: str
    status: str
    frames: int = 0
    start_time: float = float("nan")
    message: str = ""


def is_bag_dir(path: Path) -> bool:
    """A rosbag2 directory is anything that carries a ``metadata.yaml``."""
    return path.is_dir() and (path / "metadata.yaml").is_file()


def find_bags(input_dir: Path) -> list[Path]:
    """Bags under ``input_dir``; ``input_dir`` itself may also be a single bag."""
    if not input_dir.exists():
        raise FileNotFoundError(f"input path does not exist: {input_dir}")
    if is_bag_dir(input_dir):
        return [input_dir]
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input path is not a directory: {input_dir}")

    subdirs = sorted(d for d in input_dir.iterdir() if d.is_dir())
    bags = [d for d in subdirs if is_bag_dir(d)]
    ignored = [d.name for d in subdirs if not is_bag_dir(d)]
    if ignored:
        LOG.warning(
            "ignoring %d directory/ies without metadata.yaml in %s: %s",
            len(ignored),
            input_dir,
            ", ".join(ignored[:10]) + (" ..." if len(ignored) > 10 else ""),
        )
    return bags


def parse_scene_selection(tokens: Sequence[str] | None):
    """Build a predicate from ``--scenes`` tokens.

    Tokens may be scene indices (``7``), inclusive index ranges (``3-9``) or
    bag directory names; they may also be comma-separated inside one argument.
    Returns ``None`` when nothing was requested (i.e. everything is selected).
    """
    if not tokens:
        return None

    indices: set[int] = set()
    names: set[str] = set()
    for raw in tokens:
        for token in str(raw).split(","):
            token = token.strip()
            if not token:
                continue
            match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", token)
            if match:
                lo, hi = int(match.group(1)), int(match.group(2))
                indices.update(range(min(lo, hi), max(lo, hi) + 1))
            elif token.isdigit():
                indices.add(int(token))
            else:
                names.add(token.rstrip("/"))

    def selected(index: int, bag_name: str) -> bool:
        return index in indices or bag_name in names

    return selected


def _count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _count_files(directory: Path, suffix: str) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for p in directory.iterdir() if p.suffix == suffix)


def _count_images(scene_dir: Path) -> int:
    """Image files of a scene, under whichever extension it holds (`image_format`)."""
    return sum(image_format.held(scene_dir).values())


def root_image_extension(scene_dir: Path) -> str | None:
    """What the root this scene belongs to holds already: what its manifest
    declares, else what another of its scenes has. None for a root with no
    images yet, which may take either format."""
    root = scene_dir.parent
    want = image_format.declared(root)
    if want is not None:
        return want
    for other in image_format.scenes(root):
        if other != scene_dir:
            have = image_format.held(other)
            if have:
                return next(iter(have))
    return None


def image_extension(payload, how: str, index: int) -> str:
    """The extension frame ``index`` is written under.

    ``camera`` keeps the camera's own bytes, so the payload has to say what it
    is; ``png`` decodes and writes the pixels out again, as every root before
    2026-09-29 was made.
    """
    if how == "png":
        return ".png"
    ext = image_format.sniff(payload)
    if ext is None:
        raise ValueError(
            f"image {index}: the payload is neither JPEG nor PNG (it starts {bytes(payload[:4]).hex()}); "
            "--image_format png decodes it and stores the pixels instead"
        )
    return ext


def scene_frame_counts(scene_dir: Path) -> dict[str, int]:
    """Per-modality frame counts of an extracted scene (missing -> absent key)."""
    counts: dict[str, int] = {}
    counts["images"] = _count_images(scene_dir)
    counts["clouds"] = _count_files(scene_dir / "clouds", ".npy")
    for name in ("data_ts.txt", "pcd_ts.txt"):
        if (scene_dir / name).is_file():
            counts[name] = _count_lines(scene_dir / name)
    return counts


def scene_is_complete(scene_dir: Path) -> bool:
    """True if the scene was extracted to the end and its counts line up."""
    if not (scene_dir / SENTINEL_FILE).is_file():
        return False
    counts = scene_frame_counts(scene_dir)
    if len(counts) < 4:  # a timestamp file is missing entirely
        return False
    values = set(counts.values())
    return len(values) == 1 and values != {0}


def source_bag(scene_dir: Path) -> str | None:
    """The bag a finished scene was extracted from, or None for an unfinished one."""
    sentinel = scene_dir / SENTINEL_FILE
    return sentinel.read_text().strip() if sentinel.is_file() else None


def _scene_row(scene_dir: Path) -> list[str] | None:
    """Reconstruct a ``scenes.csv`` row from an already extracted scene."""
    sentinel = scene_dir / SENTINEL_FILE
    if not sentinel.is_file():
        return None
    bag = sentinel.read_text().strip()
    frames = _count_images(scene_dir)
    start = ""
    data_ts = scene_dir / "data_ts.txt"
    if data_ts.is_file():
        with data_ts.open() as handle:
            first = handle.readline().strip()
        if first:
            start = f"{float(first):.6f}"
    return [scene_dir.name, bag, start, str(frames)]


def merge_scenes_csv(output_dir: Path) -> Path:
    """Rewrite ``<output_dir>/scenes.csv`` from the finished scenes on disk."""
    rows = [row for row in map(_scene_row, image_format.scenes(output_dir)) if row is not None]
    csv_path = output_dir / SCENES_CSV
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(SCENES_CSV_HEADER)
        writer.writerows(rows)
    return csv_path


def clear_scene_outputs(scene_dir: Path, ground_truth: bool) -> None:
    """Remove what a previous extraction of this scene wrote.

    Only the extractor's own outputs go; anything else in the directory (a
    ``regen/`` provenance folder, height maps, reviewer notes) is left alone.
    ``odometry.txt`` and its segment table are removed only when this run will
    write them itself: on the old rig they come from the FAST-LIO regeneration,
    which runs *before* the extraction and has to survive it.
    """
    if not scene_dir.is_dir():
        return
    for name in SCENE_OUTPUT_DIRS:
        shutil.rmtree(scene_dir / name, ignore_errors=True)
    files = SCENE_OUTPUT_FILES + (GROUND_TRUTH_FILES if ground_truth else ())
    for name in files:
        (scene_dir / name).unlink(missing_ok=True)
