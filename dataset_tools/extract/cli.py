"""``helhest-extract``: arguments, bag discovery and the worker pool."""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from rosbags.highlevel import AnyReader

from dataset_tools import (  # noqa: F401  (rosbags_compat patches rosbags)
    disk_usage,
    image_format,
    rosbags_compat,
    tf_pins,
)
from dataset_tools.extract.scene import process_scene
from dataset_tools.extract.scene_io import (
    STATUS_EMPTY,
    STATUS_FAILED,
    STATUS_OK,
    STATUS_SKIPPED,
    SceneResult,
    find_bags,
    merge_scenes_csv,
    parse_scene_selection,
    scene_is_complete,
    source_bag,
)

LOG = logging.getLogger("helhest.extract")


def _worker(bag_path: Path, output_dir: Path, idx: int, args) -> SceneResult:
    if not logging.getLogger().handlers:  # a worker started without fork inherits no logging
        setup_logging(False)
    scene_dir = output_dir / f"{idx:04d}"
    try:
        with AnyReader([bag_path]) as reader:
            result = process_scene(reader, bag_path, scene_dir, args)
        result.index = idx
    except Exception as exc:  # noqa: BLE001 - one bad bag must not kill the run
        LOG.error("failed to process %s:\n%s", bag_path, traceback.format_exc())
        result = SceneResult(
            index=idx,
            bag=bag_path.name,
            status=STATUS_FAILED,
            message=f"{type(exc).__name__}: {exc}",
        )
    return result


def time_align_mode(value: str) -> str:
    """Normalise a ``--time_align`` value: ``auto``, ``off`` or a number as text.

    Used as the parser's ``type`` so typos fail before the worker pool starts.
    """
    mode = value.strip().lower()
    if mode in ("auto", "off"):
        return mode
    try:
        float(mode)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected 'auto', 'off' or seconds, not {value!r}") from None
    return mode


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="helhest-extract",
        description="Extract HELHEST scenes from ROS 2 bags.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_dir", type=str, help="Directory of bags, or a single bag directory")
    parser.add_argument("output_dir", type=str, help="Dataset root to write scenes to")
    parser.add_argument(
        "--cam",
        type=str,
        default="/odin1/image/compressed",
        help="Compressed image topic",
    )
    parser.add_argument(
        "--lidar",
        type=str,
        default="/odin1/cloud_filtered/cloudini",
        help="PointCloud2 or CompressedPointCloud2 topic",
    )
    parser.add_argument(
        "--odom",
        type=str,
        default="/odin1/odometry",
        help="Odometry topic for odometry.txt and transforms.npy (the Odin1 odometry is the ground truth; "
        "/fixposition/odometry_enu is the alternative). Note: the Fixposition _smooth variant's "
        "FP_POISH frame has no static TF to base_link; plain odometry_enu (FP_POI) does.",
    )
    parser.add_argument(
        "--imu",
        type=str,
        default="/odin1/odometry",
        help="Orientation source for gravity alignment, sensor_msgs/Imu or nav_msgs/Odometry "
        "(e.g. /fixposition/odometry_enu).",
    )
    parser.add_argument(
        "--gps",
        type=str,
        default="auto",
        metavar="TOPIC",
        help="GNSS fix topic (sensor_msgs/NavSatFix) written to gps.npy. "
        "'auto' takes the NavSatFix topic that carries messages, preferring a "
        "fused solution (/fixposition/odometry_llh) over a single antenna's "
        "raw fix; 'off' writes no GPS.",
    )
    parser.add_argument(
        "--time_align",
        type=time_align_mode,
        default="auto",
        metavar="auto|off|SECONDS",
        help="Correct the frame clock for the sensor's publish latency. 'auto' measures it per scene "
        "against the written clouds (it varies with load); a number applies that constant; 'off' keeps "
        "the recorded stamps. Changes data_ts.txt, pcd_ts.txt and the poses sampled onto them, not the "
        "camera/lidar pairing. Recorded in time_align.txt",
    )
    parser.add_argument(
        "--gnss_guard",
        choices=("auto", "on", "off"),
        default="auto",
        help="Drop stretches where the GNSS/INS published a reset solution (e.g. after a roll-over), "
        "and frames the pose stream does not cover. 'auto' enables the full guard for Fixposition topics, "
        "the coverage-only part for /odin1/odometry, nothing for wheel odometry. Recorded in gnss_guard.txt; "
        "holes read as 'fix_lost' in odometry_segments.txt",
    )
    parser.add_argument(
        "--cam_time_offset",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Added to the camera stamp before synchronisation (unlike --time_align, this re-pairs "
        "frames). The Odin1 camera and lidar agree to ~4 ms",
    )
    parser.add_argument(
        "--deskew",
        choices=("auto", "off"),
        default="auto",
        help="auto: move each return to the robot's pose at data_ts, where the cloud has per-point times "
        "(Odin1: a 67 ms sweep) and poses cover it. off: write sweeps as recorded. "
        "Recorded in extract_meta.json (frames.deskew)",
    )
    parser.add_argument(
        "--deskew_imu",
        type=str,
        default="/odin1/imu",
        metavar="TOPIC",
        help="sensor_msgs/Imu whose gyro gives the in-sweep rotation when deskewing; needs a TF path to "
        "--base_frame. 'off' or a missing topic falls back to slerp; the gyro is used only where it agrees "
        "with the slerp to 1 deg",
    )
    parser.add_argument(
        "--check_odom",
        type=str,
        default="/fixposition/odometry_enu",
        metavar="TOPIC",
        help="nav_msgs/Odometry to check the scene's odometry against (jumps, displacement, clock lag, "
        "tilt), written to pose_check.txt and extract_meta.json only. 'off' skips the check",
    )
    parser.add_argument(
        "--image_format",
        choices=("camera", "png"),
        default="camera",
        help="camera: store the camera's own compressed bytes. png: re-encode as PNG (~4x the size)",
    )
    parser.add_argument("--base_frame", type=str, default="base_link")
    parser.add_argument(
        "--tf_pins",
        choices=("on", "off"),
        default="on",
        help="on: replace the Odin1's online-estimated extrinsics with constants, recording the bag's "
        f"values in {tf_pins.FILE}. off: use the bag's own values",
    )
    parser.add_argument(
        "--lidar_extrinsic",
        type=float,
        nargs=6,
        default=None,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help="Manual lidar->base transform (metres; roll/pitch/yaw in RADIANS), used when TF does not connect "
        "the lidar frame to the base frame (the case for the 2026-08 Odin1 bags)",
    )
    parser.add_argument(
        "--max_sync_dt",
        type=float,
        default=0.1,
        metavar="SECONDS",
        help="Largest camera/lidar stamp difference a frame may pair; further pairs are dropped and "
        "counted in extract_meta.json",
    )
    parser.add_argument(
        "--allow_frame_fallback",
        action="store_true",
        help="Extract a bag whose TF does not reach the base frame (lidar, odometry child or orientation "
        "source), leaving that stream in its own frame instead of failing. Recorded in extract_meta.json",
    )
    parser.add_argument(
        "--trim_idle",
        type=float,
        nargs=2,
        default=None,
        metavar=("MAX_S", "KEEP_S"),
        help="Cut every stop longer than MAX_S seconds (odometry under 0.15 m/s and 5 deg/s) to its first KEEP_S "
        "seconds of frames; the odometry is kept whole. Off by default. Recorded in extract_meta.json "
        "(frames.idle_trim)",
    )
    parser.add_argument("--jobs", type=int, default=4, help="Worker processes")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract scenes whose output directory is already complete",
    )
    parser.add_argument(
        "--gt",
        choices=["auto", "on", "off"],
        default="auto",
        help="Write odometry.txt + odometry_segments.txt, segmented at gaps and jumps. 'auto' enables it "
        "when --odom is the Odin1 odometry or a Fixposition topic.",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        nargs="+",
        default=None,
        metavar="SEL",
        help="Only process these scenes: indices (7), ranges (3-9) or bag "
        "directory names; comma-separated values are accepted too",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Index of the first scene directory (offsets the numbering)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.trim_idle and args.trim_idle[1] > args.trim_idle[0]:
        # Refused here: inside a worker it would fire after the scene's outputs were cleared.
        parser.error(
            f"--trim_idle: keep {args.trim_idle[1]:g} s is longer than the {args.trim_idle[0]:g} s a stop may last"
        )
    setup_logging(args.verbose)

    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    try:
        bags = find_bags(input_dir)
    except (FileNotFoundError, NotADirectoryError) as exc:
        LOG.error("%s", exc)
        return 2
    if not bags:
        LOG.error("no rosbag2 directories (with metadata.yaml) found under %s", input_dir)
        return 2

    jobs = [(args.start_index + i, bag) for i, bag in enumerate(bags)]
    selected = parse_scene_selection(args.scenes)
    if selected is not None:
        jobs = [(idx, bag) for idx, bag in jobs if selected(idx, bag.name)]
        if not jobs:
            LOG.error("--scenes %s selected none of the %d bags found", args.scenes, len(bags))
            return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    # A finished scene holding another bag means the bag list shifted; refuse.
    moved = []
    for idx, bag in jobs:
        held = source_bag(output_dir / f"{idx:04d}")
        if held is not None and held != bag.name:
            moved.append(f"{idx:04d} holds {held}, where this run would put {bag.name}")
    if moved:
        for line in moved:
            LOG.error("scene %s", line)
        LOG.error(
            "the bags no longer line up with the scenes in %s: extract into a new root, or pick the bags with "
            "--scenes / --start_index so each lands on its own scene",
            output_dir,
        )
        return 2

    todo = []
    results: list[SceneResult] = []
    for idx, bag in jobs:
        scene_dir = output_dir / f"{idx:04d}"
        if not args.overwrite and scene_is_complete(scene_dir):
            LOG.info("scene %04d already complete (%s); skipping", idx, bag.name)
            results.append(SceneResult(index=idx, bag=bag.name, status=STATUS_SKIPPED))
            continue
        todo.append((idx, bag))

    LOG.info(
        "%d bag(s) found, %d to extract, %d skipped -> %s",
        len(bags),
        len(todo),
        len(jobs) - len(todo),
        output_dir,
    )

    if todo and args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {executor.submit(_worker, bag, output_dir, idx, args): (idx, bag) for idx, bag in todo}
            for n, future in enumerate(as_completed(futures), 1):
                idx, bag = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001 - a worker killed (OOM, segfault) breaks the pool
                    LOG.error("worker for %s died: %s: %s", bag.name, type(exc).__name__, exc)
                    results.append(SceneResult(index=idx, bag=bag.name, status=STATUS_FAILED, message=str(exc)))
                LOG.info("%d/%d bags done (%s: %s)", n, len(todo), results[-1].bag, results[-1].status)
    else:
        for n, (idx, bag) in enumerate(todo, 1):
            results.append(_worker(bag, output_dir, idx, args))
            LOG.info("%d/%d bags done (%s: %s)", n, len(todo), results[-1].bag, results[-1].status)

    csv_path = merge_scenes_csv(output_dir)
    # Best effort: rescan every touched scene's disk usage, failed ones included.
    try:
        disk_usage.update(output_dir, {f"{idx:04d}" for idx, _ in todo})
    except (OSError, ValueError) as exc:
        LOG.warning("could not update %s: %s", disk_usage.NAME, exc)
    # Best effort; the validator reports disagreeing scenes.
    try:
        image_format.declare(output_dir)
    except (OSError, ValueError) as exc:
        LOG.warning("could not declare the image format in %s: %s", image_format.NAME, exc)

    failed = [r for r in results if r.status == STATUS_FAILED]
    empty = [r for r in results if r.status == STATUS_EMPTY]
    ok = [r for r in results if r.status == STATUS_OK]
    skipped = [r for r in results if r.status == STATUS_SKIPPED]

    LOG.info(
        "done: %d extracted (%d frames), %d skipped, %d empty, %d failed; index -> %s",
        len(ok),
        sum(r.frames for r in ok),
        len(skipped),
        len(empty),
        len(failed),
        csv_path,
    )
    for result in empty:
        LOG.warning(
            "scene %04d (%s) produced no frames: %s",
            result.index,
            result.bag,
            result.message,
        )
    for result in failed:
        LOG.error("scene %04d (%s) FAILED: %s", result.index, result.bag, result.message)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
