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

    Also the parser's ``type``, so a typo is refused with a usage message before
    any bag is opened rather than inside a worker process, where a ``SystemExit``
    would take the whole pool down.
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
        help="Compressed image topic (old bags: /luxonis/oak/rgb/image_raw/compressed)",
    )
    parser.add_argument(
        "--lidar",
        type=str,
        default="/odin1/cloud_filtered/cloudini",
        help="PointCloud2 or CompressedPointCloud2 topic (old bags: /livox/lidar_front)",
    )
    parser.add_argument(
        "--odom",
        type=str,
        default="/odin1/odometry",
        help="Odometry topic used for odometry.txt and transforms.npy (default: the Odin1's own "
        "lidar-inertial odometry, 14 Hz, 6-DoF, on the lidar's clock -- the ground truth from v2.3 on; "
        "v2 beta to v2.2 used /fixposition/odometry_enu, old bags /odom). It is checked against "
        "--check_odom. Note: the Fixposition _smooth variant's child frame FP_POISH is not connected "
        "to base_link via static TF, plain odometry_enu (FP_POI) is.",
    )
    parser.add_argument(
        "--imu",
        type=str,
        default="/odin1/odometry",
        help="Orientation source for gravity alignment; accepts sensor_msgs/Imu "
        "or nav_msgs/Odometry topics (default: the Odin1 odometry's attitude; v2 beta to v2.2 used "
        "/fixposition/odometry_enu, old bags /ekf/imu/data). pose_check.txt reports the roll/pitch "
        "difference against --check_odom.",
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
        help="How to put the frame clock on the same clock as the poses. A "
        "unit that stamps at publish rather than at capture hands out headers "
        "that are late by its own pipeline latency, which makes the poses run "
        "ahead of the imagery and the clouds; nothing in the bag records it, "
        "because recv-minus-header measures publish-to-receive. 'auto' (the "
        "default) measures it against the clouds this run just wrote, as a "
        "curve over the scene -- a latency moves with load, and on the 2026-09 "
        "Odin1 rig one scene spends 80 of its 290 s at 0.05 s and the rest at "
        "0.69 s. A number applies that constant instead; 'off' leaves the "
        "stamps as recorded. What it changes is data_ts.txt, pcd_ts.txt and "
        "every pose sampled onto them; the camera/lidar pairing and the clouds "
        "themselves are untouched. Recorded per scene in time_align.txt",
    )
    parser.add_argument(
        "--gnss_guard",
        choices=("auto", "on", "off"),
        default="auto",
        help="Drop stretches where the GNSS/INS published a reset solution "
        "instead of a pose. A Fixposition that has lost its fix -- which is "
        "what rolling the robot over does -- keeps publishing odometry at full "
        "rate, filled with a position hundreds of metres away, held perfectly "
        "still, with the attitude snapped to identity, and nothing in the "
        "recording flags it. Those samples are removed, and so are the frames "
        "they leave without a pose -- along with any frame the pose stream "
        "never covered, which would otherwise be exported carrying the nearest "
        "sample -- before any image or cloud is written. "
        "'auto' (the default) turns it on for an absolute source and off for "
        "wheel odometry, which legitimately holds still. Recorded per scene in "
        "gnss_guard.txt; the holes read as 'fix_lost' in odometry_segments.txt",
    )
    parser.add_argument(
        "--cam_time_offset",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Added to the camera header stamp before synchronisation, for a "
        "camera that stamps differently from the lidar beside it. On the "
        "2026-09 Odin1 rig the two agree to a median 4 ms, so this stays 0.0 "
        "there. Unlike --time_align this one re-pairs camera frames with "
        "sweeps, since it moves only one side of the match",
    )
    parser.add_argument(
        "--deskew",
        choices=("auto", "off"),
        default="auto",
        help="auto (default): where the cloud carries a per-point time (Odin1: offset_time, a 67 ms "
        "bottom-to-top raster behind one stamp) and the pose stream covers the sweep, every return is "
        "moved to where the robot was at the frame's instant, so the cloud is a snapshot taken at data_ts "
        "like the image beside it. off: the sweep is written as recorded, with its motion smear, as "
        "every root before 2026-10-01 was. Recorded per scene in extract_meta.json (frames.deskew)",
    )
    parser.add_argument(
        "--deskew_imu",
        type=str,
        default="/odin1/imu",
        metavar="TOPIC",
        help="sensor_msgs/Imu whose gyro gives the rotation inside each sweep when deskewing (default: the "
        "Odin1's, 400 Hz); its frame needs a TF path to --base_frame. 'off', or a missing topic, falls back "
        "to the pose track's slerp across the sweep. Per sweep the gyro is used only where it agrees with "
        "that slerp to 1 deg; extract_meta.json counts gyro vs slerp",
    )
    parser.add_argument(
        "--check_odom",
        type=str,
        default="/fixposition/odometry_enu",
        metavar="TOPIC",
        help="nav_msgs/Odometry the scene's odometry is checked against, at the base frame: jumps, "
        "displacement over 1/2/5 s, clock lag, tilt. Written to pose_check.txt and extract_meta.json; "
        "nothing in the scene is built from it. 'off' skips the check; the default is the Fixposition "
        "GNSS/INS, which stays the georeference (gps.npy) and the check from v2.3 on",
    )
    parser.add_argument(
        "--image_format",
        choices=("camera", "png"),
        default="camera",
        help="camera: store the camera's own compressed bytes unmodified, as .jpg or .png by what they are "
        "(default). png: decode and store the pixels as PNG, four times the size for the same pixels; "
        "only for extending a root that already holds PNG",
    )
    parser.add_argument("--base_frame", type=str, default="base_link")
    parser.add_argument(
        "--tf_pins",
        choices=("on", "off"),
        default="on",
        help="on (default): the static transforms the Odin1 estimates online (its lidar-to-IMU and "
        "lidar-to-camera extrinsics) are replaced by one constant for every bag, and what the bag said "
        f"is recorded in {tf_pins.FILE}. off: the bag's own values are used, as before 2026-09-30",
    )
    parser.add_argument(
        "--lidar_extrinsic",
        type=float,
        nargs=6,
        default=None,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help="Manual lidar->base transform, used when TF does not connect the "
        "lidar frame to the base frame (the case for the 2026-08 Odin1 bags)",
    )
    parser.add_argument(
        "--max_sync_dt",
        type=float,
        default=0.1,
        metavar="SECONDS",
        help="Largest camera/lidar stamp difference a frame may pair; mutual nearest neighbours further apart "
        "(an image across a camera stall) are dropped and counted in extract_meta.json. The Odin1 pairs to "
        "within 2 ms; a 10 Hz lidar never leaves an image more than 0.05 s from its nearest sweep",
    )
    parser.add_argument(
        "--allow_frame_fallback",
        action="store_true",
        help="Extract a bag whose TF does not reach the base frame from the lidar, the odometry child or the "
        "orientation source, leaving that stream in its own frame. Without it such a scene fails: clouds, poses "
        "or gravity in the wrong frame look plausible and are found late. Recorded in extract_meta.json",
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
        help="Write odometry.txt + odometry_segments.txt (v1.2 ground-truth "
        "contract) from the odometry stream, segmented at gaps and jumps "
        "(relocalisations, GNSS teleports). 'auto' enables it when --odom is "
        "the Odin1 odometry or a Fixposition topic.",
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
    args = build_parser().parse_args(argv)
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

    # A finished scene names its bag. Under another bag's index it means the
    # bag list moved (a bag added in the middle, one removed): skipping it would
    # leave the new bag unextracted, overwriting it would renumber the root.
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
            futures = [executor.submit(_worker, bag, output_dir, idx, args) for idx, bag in todo]
            for n, future in enumerate(as_completed(futures), 1):
                results.append(future.result())
                LOG.info("%d/%d bags done (%s: %s)", n, len(todo), results[-1].bag, results[-1].status)
    else:
        for n, (idx, bag) in enumerate(todo, 1):
            results.append(_worker(bag, output_dir, idx, args))
            LOG.info("%d/%d bags done (%s: %s)", n, len(todo), results[-1].bag, results[-1].status)

    csv_path = merge_scenes_csv(output_dir)
    # The viewer's disk figures. Every scene this run touched is rescanned,
    # failed ones too, since an overwrite may have left them smaller; a failure
    # here costs the figures, never the extraction.
    try:
        disk_usage.update(output_dir, {f"{idx:04d}" for idx, _ in todo})
    except (OSError, ValueError) as exc:
        LOG.warning("could not update %s: %s", disk_usage.NAME, exc)
    # What the images are, where a consumer looks for it. Never the reason an
    # extraction fails: a root whose scenes disagree is reported by the validator.
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
