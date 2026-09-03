"""Probing a dataset root: what scenes exist, and what each scene actually has.

Nothing here assumes a particular sensor rig. A scene is any ``NNNN``-shaped
subdirectory; every file below is *optional* except that a scene needs at least
one of ``images/`` or ``clouds/`` to be worth listing. The probe is deliberately
cheap -- directory listings, line counts and ``.npy`` headers only, never a
cloud or an image -- because dataset roots live on contended network storage.

Results are cached on disk (keyed by the scene directory's mtime) so the index
page costs one listing per scene per session at most.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .loaders import count_lines, read_segments, read_text

logger = logging.getLogger("dataset_tools.viewer.probe")

#: "not looked at yet", distinct from "looked, and there is none".
_UNSET = object()

#: Bumped whenever ``SceneProbe`` gains or changes a field, so a cache written
#: by an older viewer is discarded rather than read back with holes in it.
PROBE_CACHE_FORMAT = 2


@dataclass(frozen=True)
class Artefact:
    key: str
    relpath: str
    kind: str
    label: str
    description: str
    suffix: str = ""


#: Everything the viewer knows how to show. Order is the order shown in the UI.
ARTEFACTS: tuple[Artefact, ...] = (
    Artefact("images", "images", "dir", "Camera images", "RGB frame per sync'd frame", ".png"),
    Artefact("clouds", "clouds", "dir", "Point clouds", "(N, >=3) lidar scan in the robot base frame", ".npy"),
    Artefact("height_maps", "height_maps", "dir", "Height maps", "per-frame BEV height grid", ".npz"),
    Artefact("height_masks", "height_masks", "dir", "Height-map validity", "measured / interpolated / extrapolated per cell", ".npz"),
    Artefact("data_ts", "data_ts.txt", "file", "Frame timestamps", "one timestamp per frame"),
    Artefact("pcd_ts", "pcd_ts.txt", "file", "Lidar timestamps", "one lidar timestamp per frame"),
    Artefact("odometry", "odometry.txt", "file", "Ground-truth odometry", "timestamped 4x4 poses"),
    Artefact("odometry_segments", "odometry_segments.txt", "file", "Odometry segments", "independently anchored stretches of GT"),
    Artefact("transforms", "transforms.npy", "file", "Wheel odometry", "(F, 4, 4) continuous but drifty per-frame pose"),
    Artefact("gravity_align", "gravity_align.npy", "file", "Gravity alignment", "(F, 3, 3) roll/pitch-removing rotation"),
    Artefact("joint_states", "joint_states.npy", "file", "Joint states", "(T, 10) ts + wheel position/velocity/effort"),
    Artefact("joint_setpoints", "joint_setpoints.npy", "file", "Joint setpoints", "(T, 10) commanded wheel state"),
    Artefact("cmd_vel", "cmd_vel.npy", "file", "Command velocity", "(T, 7) ts + commanded twist"),
    Artefact("camera_intrinsics", "camera_intrinsics.txt", "file", "Camera intrinsics", "3x3 K"),
    Artefact("camera_info", "camera_info.yaml", "file", "Camera info", "full camera_info message"),
    Artefact("T_cam_to_base", "T_cam_to_base.txt", "file", "Camera extrinsic", "4x4 camera-optical -> base"),
    Artefact("T_cam_to_lidar", "T_cam_to_lidar.txt", "file", "Camera->lidar extrinsic", "4x4, older roots"),
    Artefact("source_bag", "source_bag.txt", "file", "Source bag", "provenance of the recording"),
)

ARTEFACTS_BY_KEY = {a.key: a for a in ARTEFACTS}

#: Directories whose file count is worth knowing on the index page.
COUNTED_DIRS = ("images", "clouds", "height_maps", "height_masks")


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
    #: Each segment as a ``[start, end]`` pair of fractions of the scene's own
    #: time span -- enough to draw a timeline without re-reading the file.
    segment_spans: list[list[float]] = field(default_factory=list)
    #: The subset of those spans whose segment reason marks *borrowed* ground
    #: truth (any reason containing "fill"). Drawn differently, because such a
    #: stretch did not come out of this root's own odometry run.
    fill_spans: list[list[float]] = field(default_factory=list)
    bag: str | None = None
    warnings: list[str] = field(default_factory=list)
    mtime: float = 0.0

    @property
    def missing(self) -> list[str]:
        return [a.key for a in ARTEFACTS if not self.present.get(a.key)]

    def has(self, key: str) -> bool:
        return bool(self.present.get(key))

    def to_json(self) -> dict:
        d = asdict(self)
        d["missing"] = self.missing
        return d


def _npy_shape(path: Path) -> tuple[int, ...] | None:
    """Shape of an ``.npy`` without transferring its data.

    ``mmap_mode="r"`` reads the header and maps the rest lazily, so this costs
    one small read even for a multi-hundred-megabyte array on network storage.
    """
    try:
        return tuple(np.load(path, mmap_mode="r").shape)
    except Exception:  # noqa: BLE001 - any unreadable array is just "unknown"
        return None


def probe_scene(scene_dir: Path, scene_id: str | None = None) -> SceneProbe:
    """Probe one scene directory. Never raises for a malformed scene."""
    scene_id = scene_id or scene_dir.name
    probe = SceneProbe(scene_id=scene_id)
    try:
        probe.mtime = scene_dir.stat().st_mtime
    except OSError:
        probe.mtime = 0.0

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
                probe.counts[key] = sum(
                    1 for n in os.listdir(scene_dir / art.relpath) if n.endswith(art.suffix)
                )
            except OSError:
                probe.counts[key] = 0

    # Frame count: the timestamp file is authoritative; fall back to whatever
    # per-frame directory exists, then to the wheel-odometry array's length.
    if probe.present.get("data_ts"):
        probe.counts["data_ts"] = count_lines(scene_dir / "data_ts.txt")
    if probe.present.get("pcd_ts"):
        probe.counts["pcd_ts"] = count_lines(scene_dir / "pcd_ts.txt")
    if probe.present.get("transforms"):
        shape = _npy_shape(scene_dir / "transforms.npy")
        if shape:
            probe.counts["transforms"] = int(shape[0])
    for key in ("joint_states", "joint_setpoints", "cmd_vel", "gravity_align"):
        if probe.present.get(key):
            shape = _npy_shape(scene_dir / ARTEFACTS_BY_KEY[key].relpath)
            if shape:
                probe.counts[key] = int(shape[0])

    candidates = [
        probe.counts.get(k, 0)
        for k in ("data_ts", "images", "clouds", "pcd_ts", "transforms")
        if probe.counts.get(k)
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
        for seg in segments:
            reason = seg.reason or "unlabelled"
            probe.segment_reasons[reason] = probe.segment_reasons.get(reason, 0) + 1
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
                covered = sum(
                    max(0.0, min(s.end_ts, first_last[1]) - max(s.start_ts, first_last[0]))
                    for s in segments
                )
                probe.segment_coverage = min(1.0, covered / probe.duration)
    if probe.duration is None and probe.start_ts is not None and probe.end_ts is not None:
        probe.duration = probe.end_ts - probe.start_ts

    if segments and probe.start_ts is not None and probe.duration and probe.duration > 0:
        base, span = probe.start_ts, probe.duration
        for seg in segments:
            lo = min(1.0, max(0.0, (seg.start_ts - base) / span))
            hi = min(1.0, max(0.0, (seg.end_ts - base) / span))
            if hi <= lo:
                continue
            pair = [round(lo, 5), round(hi, 5)]
            probe.segment_spans.append(pair)
            if "fill" in (seg.reason or "").lower():
                probe.fill_spans.append(pair)

    if probe.present.get("source_bag"):
        text = read_text(scene_dir / "source_bag.txt", 4096)
        if text:
            probe.bag = text.strip().splitlines()[0] if text.strip() else None

    if not probe.present.get("images") and not probe.present.get("clouds"):
        probe.warnings.append("no images/ and no clouds/ -- nothing per-frame to show")
    return probe


def _first_last_float(path: Path) -> tuple[float, float] | None:
    """First and last parseable float in a text file, without reading it all."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as fh:
            first = fh.readline().decode("utf-8", "replace").strip()
            tail_start = max(0, size - 4096)
            fh.seek(tail_start)
            tail = fh.read().decode("utf-8", "replace").strip().splitlines()
        last = tail[-1].strip() if tail else first
        return float(first.split()[0]), float(last.split()[0])
    except (OSError, ValueError, IndexError):
        return None


class DatasetRoot:
    """A dataset root directory plus a cached probe of each scene in it."""

    def __init__(self, path: str | Path, cache_dir: Path | None = None) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_dir():
            raise NotADirectoryError(f"dataset root does not exist: {self.path}")
        self.cache_dir = Path(cache_dir) if cache_dir else default_cache_dir(self.path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._probe_cache: dict[str, SceneProbe] = {}
        self._scene_ids: list[str] | None = None
        self._csv_rows: dict[str, dict] | None = None
        self._atlas: dict | None | object = _UNSET
        #: Set by every fresh probe, cleared by every save: what makes
        #: ``probe_all`` able to write the cache once instead of once a scene.
        self._probe_dirty = False
        self._load_probe_cache()

    # -- scene discovery ---------------------------------------------------

    def scene_ids(self, refresh: bool = False) -> list[str]:
        """All scene directories, by name. A scene is any directory whose name
        is all digits (``0000``); this is what makes the viewer rig-agnostic."""
        if self._scene_ids is None or refresh:
            try:
                names = [
                    n
                    for n in os.listdir(self.path)
                    if n.isdigit() and (self.path / n).is_dir()
                ]
            except OSError as exc:
                logger.error("cannot list dataset root %s: %s", self.path, exc)
                names = []
            self._scene_ids = sorted(names)
        return self._scene_ids

    def scene_dir(self, scene_id: str) -> Path:
        if not scene_id.isdigit() or scene_id not in self.scene_ids():
            raise KeyError(f"unknown scene {scene_id!r}")
        return self.path / scene_id

    # -- root-level metadata ----------------------------------------------

    def manifest(self) -> dict | None:
        text = read_text(self.path / "manifest.json")
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def readme(self) -> str | None:
        return read_text(self.path / "README.md", 40_000)

    @property
    def atlas_dir(self) -> Path:
        return self.path / "atlas"

    def atlas(self) -> dict | None:
        """``atlas/atlas.json``: curated per-scene content, or ``None``.

        Everything in it is editorial -- prose descriptions, groupings, numbers
        that took a full pass over the bags to produce -- so it is *stored* next
        to the data rather than derived. It is entirely optional: a root without
        an ``atlas/`` directory (a fresh extraction from a different rig, say)
        still renders, just without the curated half. Read once per process; the
        file is small and only changes when someone regenerates it by hand.
        """
        if self._atlas is _UNSET:
            text = read_text(self.atlas_dir / "atlas.json", 4_000_000)
            try:
                data = json.loads(text) if text else None
            except json.JSONDecodeError as exc:
                logger.warning("cannot parse atlas/atlas.json: %s", exc)
                data = None
            self._atlas = data if isinstance(data, dict) else None
        return self._atlas

    def atlas_scene(self, scene_id: str) -> dict:
        atlas = self.atlas() or {}
        entry = (atlas.get("scenes") or {}).get(scene_id)
        return entry if isinstance(entry, dict) else {}

    def atlas_thumb(self, scene_id: str, slot: str) -> Path | None:
        """Path of a stored thumbnail, or ``None`` if this root has no such file.

        ``slot`` is one short token (``a``/``b``/``c``); it is checked against
        the manifest in ``atlas.json`` rather than pasted into a path, so a
        request cannot walk out of the atlas directory.
        """
        entry = self.atlas_scene(scene_id)
        if slot not in (entry.get("thumbs") or []):
            return None
        spec = (self.atlas() or {}).get("thumbs") or {}
        pattern = spec.get("pattern") or "{scene}{slot}.jpg"
        subdir = spec.get("dir") or "thumbs"
        path = self.atlas_dir / subdir / pattern.format(scene=scene_id, slot=slot)
        return path if path.is_file() else None

    def csv_rows(self) -> dict[str, dict]:
        """``scenes.csv`` indexed by scene id ({} when the file is absent)."""
        if self._csv_rows is None:
            rows: dict[str, dict] = {}
            path = self.path / "scenes.csv"
            if path.is_file():
                try:
                    with open(path, newline="") as fh:
                        for row in csv.DictReader(fh):
                            key = (row.get("scene") or "").strip()
                            if key:
                                rows[key] = row
                except (OSError, csv.Error) as exc:
                    logger.warning("cannot read scenes.csv: %s", exc)
            self._csv_rows = rows
        return self._csv_rows

    # -- probing -----------------------------------------------------------

    def probe(self, scene_id: str, refresh: bool = False, persist: bool = True) -> SceneProbe:
        """Probe one scene, reusing the on-disk cache while its mtime holds.

        ``persist=False`` records the result without writing the cache file;
        the caller is then responsible for calling :meth:`save_probe_cache`.
        """
        scene_dir = self.scene_dir(scene_id)
        cached = self._probe_cache.get(scene_id)
        if cached is not None and not refresh:
            try:
                if abs(scene_dir.stat().st_mtime - cached.mtime) < 1e-6:
                    return cached
            except OSError:
                return cached
        probe = probe_scene(scene_dir, scene_id)
        row = self.csv_rows().get(scene_id)
        if row:
            probe.bag = probe.bag or (row.get("bag") or None)
        self._probe_cache[scene_id] = probe
        self._probe_dirty = True
        if persist:
            self.save_probe_cache()
        return probe

    def probe_all(self, refresh: bool = False) -> list[SceneProbe]:
        """Probe every scene, writing the cache at most once.

        Saving inside the loop rewrote the whole (growing) cache file once per
        scene, so the first index render of a 43-scene root serialised it 43
        times -- quadratic in the number of scenes, for one file that only has
        to be right at the end.
        """
        probes = [
            self.probe(sid, refresh=refresh, persist=False)
            for sid in self.scene_ids(refresh=refresh)
        ]
        self.save_probe_cache()
        return probes

    # -- probe cache persistence ------------------------------------------

    @property
    def _probe_cache_path(self) -> Path:
        return self.cache_dir / "probe_cache.json"

    def _load_probe_cache(self) -> None:
        try:
            data = json.loads(self._probe_cache_path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        if data.get("root") != str(self.path) or data.get("format") != PROBE_CACHE_FORMAT:
            # A cache written before a field existed would load with that field
            # silently defaulted; re-probe instead of showing a blank timeline.
            return
        for sid, raw in (data.get("scenes") or {}).items():
            raw.pop("missing", None)
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
            "scenes": {sid: p.to_json() for sid, p in self._probe_cache.items()},
        }
        tmp = self._probe_cache_path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(payload))
            tmp.replace(self._probe_cache_path)
            self._probe_dirty = False
        except OSError as exc:
            logger.debug("cannot persist probe cache: %s", exc)


def default_cache_dir(root: Path) -> Path:
    """Per-root cache directory under ``$XDG_CACHE_HOME``."""
    import hashlib

    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    digest = hashlib.sha1(str(root).encode()).hexdigest()[:12]
    return base / "dataset_tools_viewer" / f"{root.name}-{digest}"
