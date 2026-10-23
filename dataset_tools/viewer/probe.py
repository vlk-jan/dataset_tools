"""Probing a dataset root: what scenes exist, and what each scene actually has.

A scene is any all-digit subdirectory; every file in it is optional. Probing is
cheap (listings, line counts, ``.npy`` headers) for network storage, and cached
on disk keyed by :func:`scene_fingerprint`.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .. import image_format
from . import remote
from .loaders import count_lines, read_segments, read_text
from .splits import DECLARED_FILES, NO_SPLITS, SceneSplits, read_splits
from .windows import span_fractions

logger = logging.getLogger("dataset_tools.viewer.probe")

#: Bump whenever ``SceneProbe``'s fields change, so older caches are discarded.
PROBE_CACHE_FORMAT = 6


def scene_fingerprint(scene_dir: Path) -> str:
    """A digest that moves whenever anything a page shows from this scene does.

    Stats the directory and every entry directly in it (not per-frame files
    rewritten in place). Raises ``OSError`` when the scene is gone. A mirrored
    scene (:mod:`.remote`) is synced first and digested by its listing instead.
    """
    remote.refresh(scene_dir)
    digest = remote.fingerprint(scene_dir)
    if digest is not None:
        return digest
    st = scene_dir.stat()
    h = hashlib.sha1(f"{st.st_ino}:{st.st_mtime_ns}".encode())
    with os.scandir(scene_dir) as entries:
        for entry in sorted(entries, key=lambda e: e.name):
            try:
                es = entry.stat()
            except OSError:
                continue  # removed while listing; the directory's mtime says so
            h.update(f"/{entry.name}:{es.st_ino}:{es.st_mtime_ns}:{es.st_size}".encode())
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class Artefact:
    key: str
    relpath: str
    kind: str
    label: str
    description: str
    suffix: str = ""

    @property
    def purpose(self) -> str:
        """What the file is *for*, in prose. Empty for one nothing is written for."""
        return ARTEFACT_PURPOSE.get(self.key, "")

    @property
    def count_unit(self) -> str:
        """What this artefact's count counts, so a number can be read aloud."""
        if self.kind == "dir":
            return "files"
        return "rows" if self.relpath.endswith(".npy") else "lines"


#: Everything the viewer knows how to show. Order is the order shown in the UI.
# fmt: off
ARTEFACTS: tuple[Artefact, ...] = (
    Artefact("images", "images", "dir", "Camera images", "RGB frame per sync'd frame", ".jpg"),
    Artefact("clouds", "clouds", "dir", "Point clouds", "(N, >=3) lidar scan in the robot base frame", ".npy"),
    Artefact("height_maps", "height_maps", "dir", "Height maps", "per-frame BEV height grid", ".npz"),
    Artefact("height_masks", "height_masks", "dir", "Height-map validity", "measured / interpolated / extrapolated / wheel-anchored per cell", ".npz"),
    Artefact("driven_similarity", "driven_similarity", "dir", "Driven & similar terrain", "cells the wheels rolled over, and each cell's feature similarity to them (DRIFT)", ".npz"),
    Artefact("data_ts", "data_ts.txt", "file", "Frame timestamps", "one timestamp per frame"),
    Artefact("pcd_ts", "pcd_ts.txt", "file", "Lidar timestamps", "one lidar timestamp per frame"),
    Artefact("odometry", "odometry.txt", "file", "Ground-truth odometry", "timestamped 4x4 poses"),
    Artefact("odometry_segments", "odometry_segments.txt", "file", "Odometry segments", "independently anchored stretches of GT"),
    Artefact("transforms", "transforms.npy", "file", "Frame odometry", "(F, 4, 4) per-frame pose, one track for the scene"),
    Artefact("gravity_align", "gravity_align.npy", "file", "Gravity alignment", "(F, 3, 3) roll/pitch-removing rotation"),
    Artefact("joint_states", "joint_states.npy", "file", "Joint states", "(T, 10) ts + wheel position/velocity/effort"),
    Artefact("joint_setpoints", "joint_setpoints.npy", "file", "Joint setpoints", "(T, 10) commanded wheel state"),
    Artefact("cmd_vel", "cmd_vel.npy", "file", "Command velocity", "(T, 7) ts + commanded twist"),
    Artefact("gps", "gps.npy", "file", "GNSS fixes", "(M, 8) ts + lat/lon/alt, fix status and covariance"),
    Artefact("camera_intrinsics", "camera_intrinsics.txt", "file", "Camera intrinsics", "3x3 K"),
    Artefact("camera_info", "camera_info.yaml", "file", "Camera info", "full camera_info message"),
    Artefact("T_cam_to_base", "T_cam_to_base.txt", "file", "Camera extrinsic", "4x4 camera-optical -> base"),
    Artefact("source_bag", "source_bag.txt", "file", "Source bag", "provenance of the recording"),
    Artefact("time_align", "time_align.txt", "file", "Frame-clock correction", "the latency taken off the frame clock, per window measured"),
    Artefact("gnss_guard", "gnss_guard.txt", "file", "GNSS guard", "fabricated pose stretches removed, and what they cost"),
    Artefact("tf_pins", "tf_pins.txt", "file", "Pinned transforms", "the static transforms pinned to a constant, and what the bag said"),
)
# fmt: on

#: What each artefact is *for*, shown on the scene page; keyed by ``Artefact.key``.
ARTEFACT_PURPOSE: dict[str, str] = {
    "images": (
        "One camera frame per synced frame, numbered by frame index. The viewer shows them as the "
        "scene's thumbnails, in the frame strip and as the image panel beside a 3D view; nothing "
        "else in the pipeline reads them."
    ),
    "clouds": (
        "The lidar scan belonging to each frame, already rotated into the robot's base frame. Drawn "
        "as points in the 3D view, and used to build the terrain under the robot when the scene has "
        "no height maps."
    ),
    "height_maps": (
        "A bird's-eye grid of ground height around the robot, one per frame, accumulated from the "
        "lidar in the gravity-levelled frame. The 3D view's terrain is built from these, and the "
        "learning target a model is trained on is read straight out of them."
    ),
    "height_masks": (
        "How each height-map cell got its value: measured by the lidar, interpolated between "
        "measurements, extrapolated beyond them, or anchored to a wheel contact. Only measured "
        "cells are a real observation, so both the viewer's colouring and any supervision drawn "
        "from a map are gated on this."
    ),
    "driven_similarity": (
        "Where the robot actually drove, and how far DRIFT trusts that experience around it. Per "
        "keyframe, `driven` marks the height-map cells a wheel-bottom contact of the ground-truth "
        "path landed in over the label horizon after the scan, and `max_sim` gives every cell the "
        "cosine similarity of its terrain feature to the most similar driven cell -- the trust "
        "region DRIFT extends the driven footprint to (cells above `threshold`). The similarity says "
        "where a prediction is believed, not what the terrain is: it is a feature match, not a "
        "class. Written by DRIFT for keyframes with at least one driven cell; other frames have no file."
    ),
    "data_ts": (
        "The frame clock: one timestamp per frame, in seconds. Everything per-frame lines up by "
        "index against this file, and the scene's start, end and duration are its first and last "
        "line."
    ),
    "pcd_ts": (
        "The instant each frame's lidar scan was taken, which is a little before the frame's own "
        "timestamp. Poses and height maps are sampled on this clock, so a frame's 3D content and "
        "its image can differ by a few tens of milliseconds."
    ),
    "odometry": (
        "The ground truth: timestamped 4x4 base-link poses from the offline lidar-inertial run. "
        "Every segment is re-anchored to the identity at its own first pose, so two poses only "
        "mean something together when they come from the same segment."
    ),
    "odometry_segments": (
        "Which stretches of odometry.txt were tracked continuously: number, start, end, pose count "
        "and why the stretch ended. It is what makes the ground truth usable -- the viewer never "
        "builds a trajectory, terrain or cloud across a boundary."
    ),
    "transforms": (
        "The odometry sampled at each frame, relative to frame 0, as one track for the whole scene: "
        "not split at segment breaks, so it is not ground truth. It stands in for a view where "
        "there is no GT."
    ),
    "gravity_align": (
        "A per-frame rotation that takes roll and pitch out of the base frame. Height maps are "
        "built in that levelled frame, and the frame page reads the robot's attitude back out of "
        "this array."
    ),
    "joint_states": (
        "What the wheels actually did: timestamp plus each wheel's position, velocity and effort, "
        "on their own clock rather than the frame clock. Wheel speed on the proprioception card "
        "comes from here, and wheel travel is how far the robot moved across a gap in the GT."
    ),
    "joint_setpoints": (
        "What the wheels were commanded to do, in the same layout as joint_states. The difference "
        "between the two is where slip and load show up."
    ),
    "cmd_vel": (
        "The twist commanded to the base -- the driver's intent, before the wheel controllers act "
        "on it. Commanded speed on the proprioception card is read from here. Published by the "
        "autonomy stack, so a scene where the robot was teleoperated has none."
    ),
    "gps": (
        "Raw GNSS fixes: timestamp, latitude, longitude, altitude, fix status and covariance. Used "
        "for the map on this page and as an independent check on the odometry; never ground truth, "
        "and in some scenes the fix is too poor to be worth reading."
    ),
    "camera_intrinsics": (
        "The 3x3 camera matrix K. It is what lets a point in the robot frame be projected into the "
        "image, so a cloud and a picture can be compared."
    ),
    "camera_info": (
        "The full camera_info message the camera published: image size, distortion model and "
        "coefficients, and the projection matrix. The calibration card reads it; K itself comes "
        "from camera_intrinsics.txt."
    ),
    "T_cam_to_base": (
        "The 4x4 that takes a point from the camera-optical frame to the robot base -- where the "
        "camera sat on the robot. Every overlay of image on geometry depends on it."
    ),
    "source_bag": (
        "Which recording this scene was extracted from. Provenance only, but it is the only link "
        "back to the raw bag the scene came out of."
    ),
    "time_align": (
        "What the frame clock was corrected by and on what evidence: the sensor unit stamps at "
        "publish, not at capture, so the extractor measures its latency against the clouds it "
        "wrote and takes it off data_ts.txt and pcd_ts.txt. One row per window measured; the "
        "Extraction card on this page sums it up. Absent when --time_align was off, or on a root "
        "extracted before the correction existed."
    ),
    "gnss_guard": (
        "The stretches of fabricated GNSS/INS pose the extractor removed -- a receiver that has "
        "lost its fix keeps publishing a reset position -- with what they cost in poses and "
        "frames, and how many further frames went for having no pose under them at all. Absent "
        "when the guard did not run (--gnss_guard off)."
    ),
    "tf_pins": (
        "The static transforms the extractor pinned to one constant for every bag, because the "
        "Odin1 re-estimates its own extrinsics as it runs: for each, the pin the scene was built "
        "with and what the bag's own /tf_static said, so a bag whose estimate sat far from the "
        "constant can be found. Absent when nothing was pinned (--tf_pins off)."
    ),
}


ARTEFACTS_BY_KEY = {a.key: a for a in ARTEFACTS}

#: Directories whose file count is worth knowing on the index page.
COUNTED_DIRS = ("images", "clouds", "height_maps", "height_masks", "driven_similarity")

#: Artefacts a complete scene may lack; their absence is not shown as missing.
OPTIONAL_ARTEFACTS = frozenset({"driven_similarity", "cmd_vel", "time_align", "gnss_guard", "tf_pins"})


@dataclass
class SceneProbe:
    """What one scene directory contains. Cheap to compute, safe to cache."""

    scene_id: str
    present: dict[str, bool] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    n_frames: int = 0
    start_ts: float | None = None
    end_ts: float | None = None
    duration: float | None = None
    n_segments: int = 0
    segment_reasons: dict[str, int] = field(default_factory=dict)
    segment_coverage: float | None = None
    #: Each segment as ``[start, end]`` fractions of the scene's time span.
    segment_spans: list[list[float]] = field(default_factory=list)
    bag: str | None = None
    warnings: list[str] = field(default_factory=list)
    #: :func:`scene_fingerprint` when probed; empty for a scene that was unreadable.
    fingerprint: str = ""

    @property
    def missing(self) -> list[str]:
        return [a.key for a in ARTEFACTS if not self.present.get(a.key)]

    def has(self, key: str) -> bool:
        return bool(self.present.get(key))

    def state(self, key: str) -> str:
        """``yes``, ``partial`` (fewer files than frames), ``no`` or ``optional`` (absent but normal)."""
        if not self.has(key):
            return "optional" if key in OPTIONAL_ARTEFACTS else "no"
        if key in COUNTED_DIRS and self.n_frames and self.counts.get(key, 0) < self.n_frames:
            return "partial"
        return "yes"


def _npy_shape(path: Path) -> tuple[int, ...] | None:
    """Shape of an ``.npy`` from its header only (memory-mapped, or a mirror's range read)."""
    try:
        head = remote.head(path)
        if head is not None:
            fh = io.BytesIO(head)
            version = np.lib.format.read_magic(fh)
            read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            return tuple(read(fh)[0])
        return tuple(np.load(path, mmap_mode="r").shape)
    except Exception:  # noqa: BLE001 - any unreadable array is just "unknown"
        return None


def probe_scene(scene_dir: Path, scene_id: str | None = None) -> SceneProbe:
    """Probe one scene directory. Never raises for a malformed scene."""
    scene_id = scene_id or scene_dir.name
    probe = SceneProbe(scene_id=scene_id)
    try:
        probe.fingerprint = scene_fingerprint(scene_dir)
    except OSError:
        probe.fingerprint = ""

    try:
        entries = set(os.listdir(scene_dir))
    except OSError as exc:
        probe.warnings.append(f"cannot list scene directory: {exc}")
        return probe

    for art in ARTEFACTS:
        path = scene_dir / art.relpath
        if art.kind == "dir":
            probe.present[art.key] = art.relpath in entries and path.is_dir()
        else:
            probe.present[art.key] = art.relpath in entries

    for key in COUNTED_DIRS:
        if probe.present.get(key):
            art = ARTEFACTS_BY_KEY[key]
            try:
                if key == "images":  # .jpg or .png, by what the root holds
                    probe.counts[key] = sum(image_format.held(scene_dir).values())
                else:
                    probe.counts[key] = sum(1 for n in os.listdir(scene_dir / art.relpath) if n.endswith(art.suffix))
            except OSError:
                probe.counts[key] = 0

    # Frame count: the timestamp file is authoritative; fall back to whatever
    # per-frame directory exists, then to the transforms.npy array's length.
    if probe.present.get("data_ts"):
        probe.counts["data_ts"] = count_lines(scene_dir / "data_ts.txt")
    if probe.present.get("pcd_ts"):
        probe.counts["pcd_ts"] = count_lines(scene_dir / "pcd_ts.txt")
    for key in ("transforms", "joint_states", "joint_setpoints", "cmd_vel", "gravity_align", "gps"):
        if probe.present.get(key):
            shape = _npy_shape(scene_dir / ARTEFACTS_BY_KEY[key].relpath)
            if shape:
                probe.counts[key] = int(shape[0])

    candidates = [
        probe.counts.get(k, 0) for k in ("data_ts", "images", "clouds", "pcd_ts", "transforms") if probe.counts.get(k)
    ]
    probe.n_frames = max(candidates) if candidates else 0

    # Flag disagreement rather than silently picking one.
    frame_like = {
        k: probe.counts[k]
        for k in ("data_ts", "pcd_ts", "images", "clouds", "transforms", "gravity_align")
        if probe.counts.get(k)
    }
    if len(set(frame_like.values())) > 1:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(frame_like.items()))
        probe.warnings.append(f"per-frame counts disagree ({detail})")

    segments = []
    if probe.present.get("odometry_segments"):
        segments = read_segments(scene_dir / "odometry_segments.txt")
        probe.n_segments = len(segments)
        probe.segment_reasons = dict(Counter(seg.reason or "unlabelled" for seg in segments))
        if segments:
            probe.start_ts = segments[0].start_ts
            probe.end_ts = segments[-1].end_ts

    if probe.present.get("data_ts"):
        # Two seeks, not a full read: first and last timestamp line.
        first_last = _first_last_float(scene_dir / "data_ts.txt")
        if first_last:
            probe.start_ts, probe.end_ts = first_last
            probe.duration = first_last[1] - first_last[0]
            if segments and probe.duration and probe.duration > 0:
                covered = sum(max(0.0, min(s.end_ts, first_last[1]) - max(s.start_ts, first_last[0])) for s in segments)
                probe.segment_coverage = min(1.0, covered / probe.duration)
    if probe.duration is None and probe.start_ts is not None and probe.end_ts is not None:
        probe.duration = probe.end_ts - probe.start_ts

    if segments and probe.start_ts is not None and probe.duration and probe.duration > 0:
        base, span = probe.start_ts, probe.duration
        for seg in segments:
            lo, hi = span_fractions(seg, base, span)
            if hi <= lo:
                continue
            probe.segment_spans.append([round(lo, 5), round(hi, 5)])

    if probe.present.get("source_bag"):
        text = read_text(scene_dir / "source_bag.txt", 4096)
        if text:
            probe.bag = text.strip().splitlines()[0] if text.strip() else None

    if not probe.present.get("images") and not probe.present.get("clouds"):
        probe.warnings.append("no images/ and no clouds/ -- nothing per-frame to show")
    return probe


def _first_last_float(path: Path) -> tuple[float, float] | None:
    """First and last parseable float in a text file, without reading it all.

    Blank and ``#`` lines are skipped, as in ``loaders.read_timestamps``.
    """

    def data_lines(chunk: bytes) -> list[str]:
        lines = chunk.decode("utf-8", "replace").splitlines()
        return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]

    try:
        size = path.stat().st_size
        with open(path, "rb") as fh:
            head = data_lines(fh.read(4096))
            fh.seek(max(0, size - 4096))
            tail = data_lines(fh.read())
        if not head:
            return None
        return float(head[0].split()[0]), float((tail or head)[-1].split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _read_json(path: Path) -> dict | None:
    """A JSON object from ``path``; ``None`` when absent, unparseable or not an object."""
    text = read_text(path, 4_000_000)
    try:
        data = json.loads(text) if text else None
    except json.JSONDecodeError as exc:
        logger.warning("cannot parse %s: %s", path, exc)
        return None
    return data if isinstance(data, dict) else None


def _scene_csv(path: Path) -> dict[str, dict[str, str]] | None:
    """A CSV's rows by their ``scene`` column; ``None`` when absent or unreadable."""
    if not path.is_file():
        return None
    rows: dict[str, dict[str, str]] = {}
    try:
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                key = (row.get("scene") or "").strip()
                if key:
                    rows[key] = row
    except (OSError, csv.Error) as exc:
        logger.warning("cannot read %s: %s", path.name, exc)
        return None
    return rows


class DatasetRoot:
    """A dataset root directory plus a cached probe of each scene in it."""

    def __init__(self, path: str | Path, cache_dir: Path | None = None, clear_cache: bool = False) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_dir():
            raise NotADirectoryError(f"dataset root does not exist: {self.path}")
        self.cache_dir = Path(cache_dir) if cache_dir else default_cache_dir(self.path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if clear_cache:
            files, freed = clear_cache_dir(self.cache_dir)
            logger.info("cleared %s: %d file(s), %.1f MB", self.cache_dir, files, freed / 1e6)
        self._probe_cache: dict[str, SceneProbe] = {}
        #: Root-level files, each with the (mtime_ns, size, inode) it was read at.
        self._root_files: dict[str, tuple[tuple[int, int, int] | None, object]] = {}
        self._scene_ids: list[str] | None = None
        #: Set by a fresh probe, cleared by a save.
        self._probe_dirty = False
        self._load_probe_cache()

    # -- scene discovery ---------------------------------------------------

    def scene_ids(self, refresh: bool = False) -> list[str]:
        """All scene directories (all-digit names), sorted."""
        if self._scene_ids is None or refresh:
            remote.refresh(self.path)
            try:
                names = [n for n in os.listdir(self.path) if n.isdigit() and (self.path / n).is_dir()]
            except OSError as exc:
                logger.error("cannot list dataset root %s: %s", self.path, exc)
                names = []
            self._scene_ids = sorted(names)
        return self._scene_ids

    def scene_dir(self, scene_id: str) -> Path:
        # Re-list once for a scene that appeared after the list was taken.
        known = scene_id.isdigit() and (scene_id in self.scene_ids() or scene_id in self.scene_ids(refresh=True))
        if not known:
            raise KeyError(f"unknown scene {scene_id!r}")
        return self.path / scene_id

    # -- root-level metadata ----------------------------------------------

    def manifest(self) -> dict | None:
        return _read_json(self.path / "manifest.json")

    def splits(self) -> SceneSplits:
        """The root's train/val split (`read_splits`), re-read when its files change."""
        return self._root_file(("splits.json", *DECLARED_FILES), lambda: read_splits(self.path) or NO_SPLITS)

    def keyframe_counts(self) -> dict[str, int]:
        """Lines of ``keyframes.lst`` per scene (its first path component); empty without the file."""

        def load() -> dict[str, int]:
            try:
                with open(self.path / "keyframes.lst") as f:
                    return dict(Counter(line.split("/", 1)[0] for line in f if line.strip()))
            except OSError:
                return {}

        return self._root_file("keyframes.lst", load)

    def _root_file(self, relpaths: str | tuple[str, ...], load):
        """``load()``, re-run only when one of ``relpaths`` (under the root) changed.

        These small files can be rewritten beside a running viewer.
        """
        if isinstance(relpaths, str):
            relpaths = (relpaths,)
        keys = self.root_stamp(relpaths)
        name = "+".join(relpaths)
        cached = self._root_files.get(name)
        if cached is None or cached[0] != keys:
            cached = (keys, load())
            self._root_files[name] = cached
        return cached[1]

    def root_stamp(self, relpaths: tuple[str, ...]) -> tuple:
        """``(mtime_ns, size, inode)`` of each file under the root, ``None`` when absent."""
        keys = []
        for relpath in relpaths:
            try:
                st = (self.path / relpath).stat()
                keys.append((st.st_mtime_ns, st.st_size, st.st_ino))
            except OSError:
                keys.append(None)
        return tuple(keys)

    @property
    def atlas_dir(self) -> Path:
        return self.path / "atlas"

    def atlas(self) -> dict | None:
        """``atlas/atlas.json``: curated per-scene content, or ``None``.

        Optional and editorial; re-read when it changes.
        """
        return self._root_file("atlas/atlas.json", lambda: _read_json(self.atlas_dir / "atlas.json"))

    def disk_usage(self) -> dict | None:
        """``disk_usage.json``: bytes per scene and modality, or ``None``.

        Written by ``helhest-disk-usage``; re-read when it changes.
        """
        return self._root_file("disk_usage.json", lambda: _read_json(self.path / "disk_usage.json"))

    def disk_usage_scene(self, scene_id: str) -> dict:
        entry = ((self.disk_usage() or {}).get("scenes") or {}).get(scene_id)
        return entry if isinstance(entry, dict) else {}

    def validation(self) -> dict[str, dict[str, str]] | None:
        """``validation.csv``, the report ``helhest-validate --csv`` writes, by scene.

        Status, numbers and error/warning *counts* (messages are in validation.json).
        ``None`` when absent or unreadable.
        """
        return self._root_file("validation.csv", self._read_validation)

    def _read_validation(self) -> dict[str, dict[str, str]] | None:
        rows = _scene_csv(self.path / "validation.csv")
        if rows is None:
            return None
        return {
            key: {k: (v or "").strip() for k, v in row.items() if k} for key, row in rows.items() if row.get("status")
        }

    def validation_scene(self, scene_id: str) -> dict[str, str]:
        return (self.validation() or {}).get(scene_id, {})

    def validation_messages(self, scene_id: str) -> list[dict] | None:
        """This scene's findings from ``validation.json``, or ``None`` without the file."""
        doc = self._root_file("validation.json", lambda: _read_json(self.path / "validation.json"))
        if not isinstance(doc, dict):
            return None
        entry = (doc.get("scenes") or {}).get(scene_id) or {}
        return [m for m in entry.get("messages") or [] if isinstance(m, dict)]

    def atlas_scene(self, scene_id: str) -> dict:
        atlas = self.atlas() or {}
        entry = (atlas.get("scenes") or {}).get(scene_id)
        return entry if isinstance(entry, dict) else {}

    def atlas_thumb(self, scene_id: str, slot: str) -> Path | None:
        """Path of a stored thumbnail, or ``None`` if this root has no such file.

        ``slot`` must be listed in ``atlas.json``, and the resolved file must lie
        under ``atlas/``, so neither a request nor the atlas can escape it.
        """
        entry = self.atlas_scene(scene_id)
        thumbs = entry.get("thumbs") or ()
        if slot not in (thumbs.keys() if isinstance(thumbs, dict) else thumbs):
            return None
        spec = (self.atlas() or {}).get("thumbs") or {}
        pattern = spec.get("pattern") or "{scene}{slot}.jpg"
        subdir = spec.get("dir") or "thumbs"
        path = self.atlas_dir / subdir / pattern.format(scene=scene_id, slot=slot)
        try:
            inside = path.resolve().is_relative_to(self.atlas_dir.resolve())
        except OSError:
            return None
        if not inside:
            logger.warning("atlas thumbnail %s points outside %s; ignored", path, self.atlas_dir)
            return None
        return path if path.is_file() else None

    def csv_rows(self) -> dict[str, dict]:
        """``scenes.csv`` indexed by scene id ({} when the file is absent)."""
        return self._root_file("scenes.csv", lambda: _scene_csv(self.path / "scenes.csv") or {})

    # -- probing -----------------------------------------------------------

    def probe(self, scene_id: str, persist: bool = True) -> SceneProbe:
        """Probe one scene, reusing the on-disk cache while its fingerprint holds.

        ``persist=False`` records the result without writing the cache file;
        the caller is then responsible for calling :meth:`save_probe_cache`.
        """
        scene_dir = self.scene_dir(scene_id)
        cached = self._probe_cache.get(scene_id)
        if cached is not None and cached.fingerprint:
            try:
                if scene_fingerprint(scene_dir) == cached.fingerprint:
                    return cached
            except OSError:
                pass  # the scene is gone: probe it again, and say so
        probe = probe_scene(scene_dir, scene_id)
        row = self.csv_rows().get(scene_id)
        if row:
            probe.bag = probe.bag or (row.get("bag") or None)
        self._probe_cache[scene_id] = probe
        self._probe_dirty = True
        if persist:
            self.save_probe_cache()
        return probe

    def probe_all(self) -> list[SceneProbe]:
        """Probe every scene (re-listing the root), writing the cache once."""
        probes = [self.probe(sid, persist=False) for sid in self.scene_ids(refresh=True)]
        self.save_probe_cache()
        return probes

    # -- probe cache persistence ------------------------------------------

    @property
    def _probe_cache_path(self) -> Path:
        return self.cache_dir / PROBE_CACHE_NAME

    def _load_probe_cache(self) -> None:
        try:
            data = json.loads(self._probe_cache_path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        if data.get("root") != str(self.path) or data.get("format") != PROBE_CACHE_FORMAT:
            return
        for sid, raw in (data.get("scenes") or {}).items():
            raw.pop("missing", None)  # written by caches before 2026-10-23
            try:
                self._probe_cache[sid] = SceneProbe(**raw)
            except TypeError:
                continue  # cache written by an older version: just re-probe

    def save_probe_cache(self) -> None:
        """Write the probe cache, unless nothing has been probed since the last one."""
        if not self._probe_dirty:
            return
        payload = {
            "root": str(self.path),
            "format": PROBE_CACHE_FORMAT,
            "saved": time.time(),
            # list(): a probe on another request thread may insert while this runs.
            "scenes": {sid: asdict(p) for sid, p in list(self._probe_cache.items())},
        }
        # pid: two viewers on one root must not write into each other's tmp file.
        tmp = self._probe_cache_path.with_suffix(f".{os.getpid()}.tmp")
        try:
            tmp.write_text(json.dumps(payload))
            tmp.replace(self._probe_cache_path)
            self._probe_dirty = False
        except OSError as exc:
            logger.debug("cannot persist probe cache: %s", exc)


#: Everything the viewer writes under a cache dir; :func:`clear_cache_dir` removes
#: only these, since ``--cache-dir`` can point anywhere.
PROBE_CACHE_NAME = "probe_cache.json"
BUNDLE_DIR_NAME = "bundles"
#: Default ceiling on the built 3D windows kept under ``BUNDLE_DIR_NAME``.
DEFAULT_CACHE_MAX_GB = 5.0


def clear_cache_dir(cache_dir: Path) -> tuple[int, int]:
    """Delete the viewer's own files under ``cache_dir``, leaving the rest.

    Returns ``(files, bytes)`` removed.
    """
    import shutil

    stem = Path(PROBE_CACHE_NAME).stem
    targets = [cache_dir / PROBE_CACHE_NAME, *cache_dir.glob(f"{stem}.*.tmp"), cache_dir / BUNDLE_DIR_NAME]
    files = freed = 0
    for target in targets:
        try:
            if target.is_dir():
                for path in target.rglob("*"):
                    if path.is_file():
                        files += 1
                        freed += path.stat().st_size
                shutil.rmtree(target)
            elif target.exists():
                files += 1
                freed += target.stat().st_size
                target.unlink()
        except OSError as exc:
            logger.warning("cannot clear %s: %s", target, exc)
    return files, freed


def default_cache_dir(root: Path) -> Path:
    """Per-root cache directory under ``$XDG_CACHE_HOME``."""
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    digest = hashlib.sha1(str(root).encode()).hexdigest()[:12]
    return base / "dataset_tools_viewer" / f"{root.name}-{digest}"
