"""Building one SimView scene for one window of one dataset scene.

Everything here is *on demand and bounded*: a build touches only the requested
window's frames, reads at most a few dozen clouds and height maps out of it,
and never scans the rest of the dataset. That is what makes a 20k-frame scene
on network storage browsable.

Which dataset file becomes what in the built scene is tabulated in
docs/VIEWER.md ("What each file turns into").
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import astuple, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from .. import image_format
from ..transforms import interpolate_poses, mat_to_quat, poses_from_quat
from . import loaders as L
from .loaders import CHASSIS, CHASSIS_CENTER, WHEEL_RADIUS, WHEEL_WIDTH, WHEELS
from .probe import OPTIONAL_ARTEFACTS, DatasetRoot, SceneProbe
from .render import unit_colors
from .windows import (
    DEFAULT_MAX_FRAMES,
    DEFAULT_MAX_SECONDS,
    Window,
    frame_slack,
    parse_frame_window_key,
    segment_of_frame,
    single_frame_window,
    windows_for_scene,
)

logger = logging.getLogger("dataset_tools.viewer.build")

#: Height-map samples are weighted by ``exp(-(d / HM_BLEND_SIGMA)**2)``, ``d``
#: being how far the sample sat from the robot that measured it. A window splats
#: ~80 maps 12.8 m across along a path a few tens of metres long, so every cell
#: is covered by dozens of them; a plain mean blends observations made from
#: poses whose z disagrees by however much the odometry drifted between them,
#: which lifts the ground away from the robot standing on it. Weighting by
#: distance lets each cell be decided by the frames that drove over it.
HM_BLEND_SIGMA = 3.0
#: Ground (m) the terrain grid reaches beyond the window's path on every side.
TERRAIN_MARGIN = 7.0

WHEEL_LABELS = ("left", "rear", "right")

#: File suffix of each per-frame directory's entries.
#: Images are not here: their extension is the root's to say (`SceneAccess.frame_suffix`).
FRAME_SUFFIX = {"clouds": ".npy", "height_maps": ".npz", "height_masks": ".npz", "driven_similarity": ".npz"}

#: Threads reading a window's per-frame files. A root usually sits on network
#: storage, where a serial loop pays one round trip of latency per file before
#: any bytes flow; a build touches ~200 files, so overlapping those round trips
#: is what makes it fast. File reads release the GIL.
READ_WORKERS = 16


def _read_all(paths: Sequence, loader: Callable) -> list:
    """``[loader(p) for p in paths]``, with the reads overlapped in threads."""
    if len(paths) <= 1:
        return [loader(p) for p in paths]
    with ThreadPoolExecutor(max_workers=min(READ_WORKERS, len(paths))) as pool:
        return list(pool.map(loader, paths))


#: Bumped whenever :func:`build_scene`'s ``meta`` payload changes a field the
#: overlay reads, or the same window *key* comes to mean different frames.
#:
#: The bundle cache is keyed by what was *asked for*, which does not change when
#: the code that answers does; without a version in the file, an upgrade would
#: serve yesterday's ``meta.json`` for every window already opened.
#:
#: 4: the whole-scene fallback is bounded by ``data_ts.txt`` rather than the
#:    frame count, so ``raw.0`` can cover fewer frames than it used to.
#: 5: windows are gravity-levelled and terrain is distance-weighted, so every
#:    cached bundle holds poses and a height field that no longer match.
#: 6: ``meta`` carries the GT-vs-IMU attitude comparison the panel reads, and
#:    the attitude scalars are renamed to say which source they came from.
#: 7: the ``validity`` terrain layer is the weighted mean of the samples behind
#:    each cell rather than the best of them, so cached bundles hold a layer
#:    that no longer means what its legend says.
#: 8: wheel radius 0.36 -> 0.334 m, so cached bundles draw the old wheels.
#: 9: ``wheel_slip_*`` scalars (setpoint minus measured), so a cached bundle's
#:    plot list is short of them; and ``attitude`` carries its ``band`` (ok /
#:    warn / bad), the verdict the window tables show, so the panel and the
#:    tables cannot disagree.
#: 10: the ``driven`` and ``drivable`` terrain layers from ``driven_similarity/``,
#:    and ``meta.available`` lists that modality; a bundle built before knows
#:    neither.
#: 11: the episode label is the window's own ("Segment 1 (3/4)"), without the
#:    pose source after it.
#: 12: no episode at all. A window is one stretch of one segment, and the
#:    panel's title already names it; the timeline label only repeated it.
META_VERSION = 12

#: What each terrain colour mode means, in the viewer's own terms.
#:
#: SimView turns every named terrain property into a "Color Mode" entry with no
#: description anywhere -- the names are ours (see ``_finish_terrain`` and
#: ``aggregate_clouds``), so the explanations have to be ours too. They travel
#: in the scene metadata and are rendered by the overlay panel; a layer not
#: listed here still gets a legend row, just without prose.
TERRAIN_COLOR_MODE_HELP = {
    "height": "Surface height in metres -- the terrain's own z. SimView's built-in mode.",
    "coverage": (
        "How many height-map cells were splatted into this terrain cell. "
        "0 means nothing was ever observed here, so the surface is extended "
        "from the trend around it rather than measured."
    ),
    "validity": (
        "How measured this cell's height actually is: the mean validity of the "
        "height-map samples behind it, weighted exactly as the height was. 0 is "
        "extrapolated throughout, 1 interpolated (inside the hull, no return), "
        "2 a lidar return every time; dozens of samples land in a cell, so the "
        "values in between are real mixtures. Cells filled from the robot's own "
        "wheel contacts count as 0 here -- the lidar never saw them. The "
        "Validity tile shows the states unmixed, for one frame."
    ),
    "cloud_points": (
        "Number of aggregated raw lidar points that fell in the cell. Near "
        "zero in the sensor's blind wedge and far off the driven path."
    ),
    "cloud_intensity": "Mean lidar return intensity of those points.",
    "driven": (
        "1 where a wheel of the robot rolled over the cell, by DRIFT's own "
        "reckoning: the wheel-bottom contacts of the ground-truth path, "
        "rasterised per keyframe and united over the window. 0 everywhere "
        "else, including terrain the robot merely saw. The wheels are those "
        "of the model the checkpoint trained under (the sidecar's `robot`), "
        "not necessarily the rig's own radius."
    ),
    "drivable": (
        "How closely the cell's terrain feature matches something the robot "
        "drove over: the highest cosine similarity, over the window's "
        "keyframes, between the cell's pre-head SeTProP feature and any driven "
        "cell's. 1 on the driven path itself; DRIFT trusts its prediction "
        "above the threshold the sidecar records (0.9 as trained). A feature "
        "match, not a terrain class -- and these features are known to encode "
        "range and viewpoint as well as terrain, so read it with the Driven "
        "tile's per-frame view beside it."
    ),
    "cloud_spread": (
        "Vertical spread (max minus min z) of the lidar points in the cell: "
        "near zero on flat ground, large on vegetation, trunks and walls."
    ),
}


def terrain_color_modes(layer_names) -> list[dict]:
    """The colour modes this window's terrain actually offers, described.

    ``height`` always exists (it is SimView's built-in mode); every property we
    attach becomes a mode of the same name. Ordered exactly as SimView's own
    "Color Mode" dropdown lists them, so the overlay legend can be read
    alongside it.
    """
    return [{"name": name, "description": TERRAIN_COLOR_MODE_HELP.get(name, "")} for name in ["height", *layer_names]]


@dataclass
class BuildOptions:
    """Everything that trades visual richness against bytes read and shipped."""

    max_cloud_frames: int = 40
    max_cloud_points: int = 160_000
    max_hm_frames: int = 80
    max_terrain_cells: int = 500_000
    terrain_res: float = 0.1
    cloud_color: str = "height"  # "height" | "intensity"

    def sanitized(self) -> BuildOptions:
        return BuildOptions(
            max_cloud_frames=int(np.clip(self.max_cloud_frames, 0, 200)),
            max_cloud_points=int(np.clip(self.max_cloud_points, 0, 1_500_000)),
            max_hm_frames=int(np.clip(self.max_hm_frames, 0, 400)),
            max_terrain_cells=int(np.clip(self.max_terrain_cells, 1000, 2_000_000)),
            terrain_res=float(np.clip(self.terrain_res, 0.02, 2.0)),
            cloud_color=self.cloud_color if self.cloud_color in ("height", "intensity") else "height",
        )

    def cache_key(self) -> str:
        """The sanitized fields in declaration order, ``-``-joined: a bundle path segment."""
        return "-".join(map(str, astuple(self.sanitized())))


#: Window length the budgets in :class:`BuildOptions` were chosen against. A
#: window of this many frames gets exactly what the options ask for.
BUDGET_REFERENCE_FRAMES = 1500
#: Ceiling on that scaling. The budgets buy samples, and a window four times
#: the reference already reaches the sanitized ceilings on most of them.
MAX_BUDGET_SCALE = 4.0
#: The terrain budget scales more slowly than the sampling ones: it buys cells,
#: and cells are what the bundle is mostly made of (~36 bytes each across the
#: layers), so doubling it is the difference between a 9 MB and a 24 MB window.
MAX_TERRAIN_SCALE = 2.0


def budgets_for_window(opts: BuildOptions, n_frames: int) -> BuildOptions:
    """``opts`` with the sampling budgets scaled to a window's length.

    The budgets are fixed counts -- 80 height maps, 40 clouds, 160k points --
    so a window twice as long samples the same ground half as densely, and a
    long window would otherwise be visibly poorer than the parts it replaced.
    They are scaled with the frame count to hold that density roughly constant,
    and ``sanitized`` clips the result, so a very long window stops at the same
    ceilings the UI offers rather than at whatever the arithmetic produced.

    Short windows -- most windows, the median segment of v1.3 is 43 frames --
    are left exactly as asked for; this only ever scales up.
    """
    scale = min(max(1.0, n_frames / BUDGET_REFERENCE_FRAMES), MAX_BUDGET_SCALE)
    if scale <= 1.0:
        return opts
    terrain_scale = min(scale, MAX_TERRAIN_SCALE)
    return replace(
        opts,
        max_hm_frames=int(round(opts.max_hm_frames * scale)),
        max_cloud_frames=int(round(opts.max_cloud_frames * scale)),
        max_cloud_points=int(round(opts.max_cloud_points * scale)),
        max_terrain_cells=int(round(opts.max_terrain_cells * terrain_scale)),
    ).sanitized()


class SceneAccess:
    """Lazily-loaded, cached handle on one scene's *small* per-scene files.

    "Small" means anything whose whole-scene size is bounded by the frame count
    (poses, timestamps, proprioception) -- never clouds, images or height maps,
    which are read per frame by the caller.
    """

    def __init__(self, root: DatasetRoot, scene_id: str) -> None:
        self.root = root
        self.scene_id = scene_id
        self.dir = root.scene_dir(scene_id)
        self.probe: SceneProbe = root.probe(scene_id)
        self._cache: dict[str, object] = {}

    def cached(self, key: str, fn):
        """Memoise a value on this scene handle: a file's contents, or a
        summary assembled from several (the scene page's cards). They cost
        reads on network storage, and paging between the tabs of one scene
        must not pay for them twice.
        """
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def _optional(self, key: str, artefact: str, fn):
        """Load an optional artefact, returning None without complaining when
        the probe already says the scene does not have it."""
        if not self.probe.has(artefact):
            return None
        return self.cached(key, fn)

    # -- timestamps --------------------------------------------------------

    @property
    def frame_ts(self) -> np.ndarray | None:
        def load():
            for name in ("data_ts.txt", "pcd_ts.txt"):
                ts = L.read_timestamps(self.dir / name)
                if ts is not None:
                    self._cache["clock_source"] = name
                    return ts
            if self.probe.n_frames:
                # No timestamp file at all: synthesise a uniform 10 Hz clock so
                # the frames still have an ordering to play back.
                self._cache["clock_source"] = "synthetic"
                return np.arange(self.probe.n_frames, dtype=np.float64) * 0.1
            return None

        return self.cached("frame_ts", load)  # type: ignore[return-value]

    @property
    def clock_problem(self) -> str | None:
        """Why the frame times on a page are not the scene's own, or None.

        The fallbacks keep the pages working, which is also what hides them:
        segment membership, windows and every signal are placed on this clock.
        """
        _ = self.frame_ts  # loading it records which file it came from
        source = self._cache.get("clock_source")
        if source in (None, "data_ts.txt"):
            return None
        why = "unreadable" if self.probe.has("data_ts") else "missing"
        if source == "pcd_ts.txt":
            return f"data_ts.txt is {why}: frame times on this page are the lidar's (pcd_ts.txt)."
        return (
            f"data_ts.txt is {why} and so is pcd_ts.txt: frame times on this page are made up at 10 Hz, "
            "so which segment a frame is in, the windows and every signal are placed on a clock that is not real."
        )

    @property
    def pcd_ts(self) -> np.ndarray | None:
        return self.cached("pcd_ts", lambda: L.read_timestamps(self.dir / "pcd_ts.txt"))  # type: ignore[return-value]

    # -- poses -------------------------------------------------------------

    @property
    def odometry(self):
        return self._optional("odometry", "odometry", lambda: L.read_odometry(self.dir / "odometry.txt"))

    @property
    def segments(self) -> list[L.Segment]:
        return self._optional("segments", "odometry_segments", lambda: L.read_segments(self.dir / "odometry_segments.txt")) or []  # type: ignore[return-value]

    @property
    def transforms(self) -> np.ndarray | None:
        return self._optional("transforms", "transforms", lambda: L.load_npy(self.dir / "transforms.npy"))  # type: ignore[return-value]

    @property
    def gravity_align(self) -> np.ndarray | None:
        return self._optional("gravity_align", "gravity_align", lambda: L.load_npy(self.dir / "gravity_align.npy"))  # type: ignore[return-value]

    # -- proprioception ----------------------------------------------------

    @property
    def joint_states(self) -> np.ndarray | None:
        return self._optional("joint_states", "joint_states", lambda: L.load_npy(self.dir / "joint_states.npy"))  # type: ignore[return-value]

    @property
    def joint_setpoints(self) -> np.ndarray | None:
        return self._optional("joint_setpoints", "joint_setpoints", lambda: L.load_npy(self.dir / "joint_setpoints.npy"))  # type: ignore[return-value]

    @property
    def cmd_vel(self) -> np.ndarray | None:
        return self._optional("cmd_vel", "cmd_vel", lambda: L.load_npy(self.dir / "cmd_vel.npy"))  # type: ignore[return-value]

    # -- geolocation -------------------------------------------------------

    @property
    def gps(self) -> np.ndarray | None:
        """This scene's GNSS track, or None. Bounded by the recording length."""
        return self._optional("gps", "gps", lambda: L.read_gps(self.dir / "gps.npy"))  # type: ignore[return-value]

    # -- calibration -------------------------------------------------------

    @property
    def K(self) -> np.ndarray | None:
        return self._optional("K", "camera_intrinsics", lambda: L.read_matrix(self.dir / "camera_intrinsics.txt", (3, 3)))  # type: ignore[return-value]

    @property
    def camera_info(self) -> dict[str, Any] | None:
        return self._optional("camera_info", "camera_info", lambda: L.read_camera_info(self.dir / "camera_info.yaml"))  # type: ignore[return-value]

    @property
    def T_cam(self) -> np.ndarray | None:
        def load():
            for name in ("T_cam_to_base.txt", "T_cam_to_lidar.txt"):
                if (self.dir / name).is_file():
                    return L.read_matrix(self.dir / name, (4, 4))
            return None

        return self.cached("T_cam", load)  # type: ignore[return-value]

    # -- per-frame directories --------------------------------------------

    def available_frames(self, kind: str) -> set[int]:
        """Frame indices that actually have a file in ``kind``/ (cached).

        Directories are built in flight, so a scene routinely has height maps
        for only a prefix (or a subsample) of its frames.
        """
        suffix = self.frame_suffix(kind)
        return self.cached(f"avail_{kind}", lambda: set(L.frame_indices_dir(self.dir / kind, suffix)))  # type: ignore[return-value]

    def frame_suffix(self, kind: str) -> str:
        """The extension of a per-frame file. Fixed, except for the images: a
        root holds the camera's JPEG or, before 2026-09-29, a PNG of it."""
        if kind != "images":
            return FRAME_SUFFIX[kind]

        def held() -> str:
            try:
                return image_format.of_scene(self.dir)
            except image_format.ImageFormatError:
                # A scene the validator fails. Show what most of it is.
                have = image_format.held(self.dir)
                return max(have, key=have.get) if have else image_format.DEFAULT

        return self.cached("image_suffix", held)  # type: ignore[return-value]

    def frame_path(self, kind: str, index: int) -> Path:
        return self.dir / kind / f"{index:05d}{self.frame_suffix(kind)}"

    def nearest_frame(self, kind: str, index: int) -> int | None:
        """``index`` when it has a ``kind`` file, else the nearest frame that has
        one (a sparse directory), else None."""
        if self.frame_path(kind, index).is_file():
            return index
        available = sorted(self.available_frames(kind))
        return min(available, key=lambda i: abs(i - index)) if available else None

    # -- windows -----------------------------------------------------------

    def windows(self, max_frames: int, max_seconds: float = DEFAULT_MAX_SECONDS) -> list[Window]:
        return self.cached(
            f"windows_{max_frames}_{max_seconds}",
            lambda: windows_for_scene(
                self.frame_ts,
                self.segments,
                self.probe.n_frames,
                self.probe.has("transforms"),
                max_frames=max_frames,
                max_seconds=max_seconds,
            ),
        )  # type: ignore[return-value]

    def window(self, key: str, max_frames: int, max_seconds: float = DEFAULT_MAX_SECONDS) -> Window | None:
        for w in self.windows(max_frames, max_seconds):
            if w.key == key:
                return w
        # One-frame windows are addressable but not enumerated (see
        # ``single_frame_window``): a 20k-frame scene has 20k of them.
        index = parse_frame_window_key(key)
        if index is not None:
            return single_frame_window(self.frame_ts, self.segments, self.probe.n_frames, index)
        return None

    def segment_of_frame(self, index: int):
        return segment_of_frame(self.frame_ts, self.segments, index)


# --------------------------------------------------------------------------
# pose resolution
# --------------------------------------------------------------------------


@dataclass
class PoseTrack:
    positions: np.ndarray  # (T, 3)
    quats: np.ndarray  # (T, 4) [w,x,y,z]
    valid: np.ndarray  # (T,) bool
    source: str
    note: str = ""


def _hold_last(a: np.ndarray, n: int) -> np.ndarray:
    """``a`` padded to at least ``n`` rows by repeating its last row."""
    return np.vstack([a, np.repeat(a[-1:], n - len(a), axis=0)]) if len(a) < n else a


def _identity_track(n: int, source: str, note: str) -> PoseTrack:
    quats = np.zeros((n, 4))
    quats[:, 0] = 1.0
    return PoseTrack(np.zeros((n, 3)), quats, np.zeros(n, dtype=bool), source, note)


def _gravity_level(access: SceneAccess, frame_index: int) -> np.ndarray | None:
    """Rotation taking a window's anchor frame into a gravity-aligned world.

    Each segment is re-anchored to identity at its own first pose. That is right
    for position and yaw -- the window is its own frame -- but roll and pitch are
    absolute, measured against gravity, and zeroing them renders a robot that
    starts on a slope as level, and one that starts on its side as upright.
    ``gravity_align.npy`` holds the attitude the anchor discarded; rotating the
    whole window by the anchor frame's own value puts it back.
    """
    ga = access.gravity_align
    if ga is None or frame_index < 0 or frame_index >= len(ga):
        return None
    R = np.asarray(ga[frame_index], dtype=np.float64)
    return R if R.shape == (3, 3) else None


def _level_track(track: PoseTrack, R: np.ndarray | None) -> PoseTrack:
    """Rotate a whole pose track into the gravity-aligned world."""
    if R is None:
        return track
    q = mat_to_quat(R[None])
    return PoseTrack(
        track.positions @ R.T,
        _quat_mul(q, track.quats),
        track.valid,
        track.source,
        track.note,
    )


def _anchor_frame(access: SceneAccess, ts: float) -> int:
    """Index of the frame nearest ``ts`` -- where a segment's anchor pose sits."""
    frame_ts = access.frame_ts
    if frame_ts is None or not len(frame_ts):
        return 0
    return int(np.argmin(np.abs(np.asarray(frame_ts) - ts)))


def resolve_poses(access: SceneAccess, window: Window, times: np.ndarray) -> PoseTrack:
    """The window's robot pose, in the window's own reference frame.

    For a ``segment`` window this is ground truth interpolated inside that one
    segment -- never across its boundary, since each segment is anchored to its
    own identity. For a ``raw`` window it is the wheel-odometry pose
    re-anchored to the window's first frame, so the view starts at the origin
    and the drift accumulates visibly from there.
    """
    n = len(times)
    if window.kind == "frame":
        # A single frame is a static inspection: its absolute position inside
        # the identity-anchored segment carries no visual meaning, and using it
        # would drop the robot tens of metres from the origin, off the initial
        # camera (the black single-frame view). Anchored at the origin instead,
        # upright on its own height map. ``pose_source`` still records whether
        # ground truth exists.
        R = _gravity_level(access, window.frame_start)
        one = mat_to_quat(R[None])[0] if R is not None else np.array([1.0, 0.0, 0.0, 0.0])
        quats = np.tile(one, (n, 1))
        valid = np.ones(n, dtype=bool) if window.pose_source == "odometry" else np.zeros(n, dtype=bool)
        return PoseTrack(np.zeros((n, 3)), quats, valid, window.pose_source, "single frame, robot-centric")

    if window.pose_source == "odometry":
        odom = access.odometry
        if odom is None:
            return _identity_track(n, "none", "odometry.txt could not be read")
        ts, poses = odom
        seg = next((s for s in access.segments if s.index == window.segment_index), None)
        if seg is not None:
            lo = np.searchsorted(ts, seg.start_ts, side="left")
            hi = np.searchsorted(ts, seg.end_ts, side="right")
            ts, poses = ts[lo:hi], poses[lo:hi]
        if len(ts) < 2:
            return _identity_track(n, "none", "segment has fewer than two poses")
        pos, quat, valid = interpolate_poses(ts, poses, times, tol=frame_slack(access.frame_ts))
        track = PoseTrack(pos, quat, valid, "odometry", f"ground truth, segment {window.segment_index}")
        # Levelled about the *segment's* anchor, not the window's first frame:
        # a later chunk of a split segment is still expressed in that anchor.
        anchor = _anchor_frame(access, seg.start_ts) if seg is not None else window.frame_start
        return _level_track(track, _gravity_level(access, anchor))

    if window.pose_source == "transforms":
        tf = access.transforms
        if tf is None:
            return _identity_track(n, "none", "transforms.npy could not be read")
        idx = np.arange(window.frame_start, window.frame_end)
        idx = idx[idx < len(tf)]
        if len(idx) < 2:
            return _identity_track(n, "none", "transforms.npy too short")
        rel = np.linalg.inv(tf[idx[0]]) @ tf[idx]
        # Ragged array vs frame count: pad by holding the last pose, marked invalid.
        pos = _hold_last(rel[:, :3, 3], n)
        quat = _hold_last(mat_to_quat(rel[:, :3, :3]), n)
        valid = np.arange(len(pos)) < len(idx)
        track = PoseTrack(pos, quat, valid, "transforms", "wheel odometry, drifts")
        return _level_track(track, _gravity_level(access, int(idx[0])))

    return _identity_track(n, "none", "no pose data; robot held at the origin")


def _ghost_track(access: SceneAccess, window: Window, gt: PoseTrack) -> PoseTrack | None:
    """Wheel odometry expressed in the ground-truth window frame.

    Aligned to ground truth at the window's first valid frame, so the two
    bodies start coincident and separate by exactly the accumulated drift.
    """
    tf = access.transforms
    if tf is None or gt.source != "odometry" or not gt.valid.any():
        return None
    idx = np.arange(window.frame_start, min(window.frame_end, len(tf)))
    if len(idx) < 2:
        return None
    anchor = int(np.flatnonzero(gt.valid)[0])
    if anchor >= len(idx):
        return None
    gt_anchor = poses_from_quat(gt.positions[anchor : anchor + 1], gt.quats[anchor : anchor + 1])[0]
    rel = np.linalg.inv(tf[idx[anchor]]) @ tf[idx]
    world = gt_anchor @ rel
    pos, quat = world[:, :3, 3], mat_to_quat(world[:, :3, :3])
    n = len(gt.positions)
    return PoseTrack(
        _hold_last(pos, n)[:n], _hold_last(quat, n)[:n], np.ones(n, dtype=bool), "transforms", "wheel odometry"
    )


# --------------------------------------------------------------------------
# terrain
# --------------------------------------------------------------------------


@dataclass
class TerrainGrid:
    """A world-frame raster the window's terrain is accumulated into."""

    x0: float
    y0: float
    res: float
    nx: int
    ny: int
    #: Weighted height sum and the weights behind it; their ratio is the
    #: surface. ``count`` stays a plain observation tally for the coverage
    #: layer, which reports how much was seen, not how much it was trusted.
    sum_wz: np.ndarray = field(init=False)
    sum_w: np.ndarray = field(init=False)
    count: np.ndarray = field(init=False)
    #: Weighted validity sum and *its own* weights. A separate pair from the
    #: height's, because a frame whose validity mask is missing contributes a
    #: height but no validity, and dividing by the height's weights would read
    #: that absence as "extrapolated" rather than as "not said".
    sum_wv: np.ndarray = field(init=False)
    sum_w_v: np.ndarray = field(init=False)
    #: The same weighted height sum over *measured* samples only. A per-frame
    #: map is mostly invented (~70 % extrapolated on real data), and a tree or a
    #: person is measured by the few frames that saw it and filled in at ground
    #: level by the dozens that did not; one mean over all of them squashes it
    #: flat. Wherever any frame measured a cell, these decide its height.
    sum_wz_m: np.ndarray = field(init=False)
    sum_w_m: np.ndarray = field(init=False)
    layers: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sum_wz = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.sum_w = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.count = np.zeros((self.ny, self.nx), dtype=np.int64)
        self.sum_wv = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.sum_w_v = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.sum_wz_m = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.sum_w_m = np.zeros((self.ny, self.nx), dtype=np.float64)

    @property
    def x_lim(self) -> tuple[float, float]:
        return (self.x0, self.x0 + self.nx * self.res)

    @property
    def y_lim(self) -> tuple[float, float]:
        return (self.y0, self.y0 + self.ny * self.res)

    def cell_of(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """World (x, y) -> (row, col, inside) in SimView's [row=y, col=x] layout."""
        col = np.floor((x - self.x0) / self.res).astype(np.int64)
        row = np.floor((y - self.y0) / self.res).astype(np.int64)
        inside = (col >= 0) & (col < self.nx) & (row >= 0) & (row < self.ny)
        return row, col, inside

    def binned(self, row: np.ndarray, col: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        """Per-cell sum of ``weights`` (or a count) for samples at ``(row, col)``.

        ``np.bincount`` over the flattened cell index: ``np.add.at`` does the
        same job an order of magnitude slower, and a window splats over a
        million samples.
        """
        flat = np.bincount(row * self.nx + col, weights=weights, minlength=self.ny * self.nx)
        return flat.reshape(self.ny, self.nx)


def _make_grid(positions: np.ndarray, opts: BuildOptions) -> TerrainGrid:
    """A grid covering the window's path plus a margin, at a resolution that
    keeps the cell count under the budget (coarsening rather than clipping, so
    a long traverse is still shown whole)."""
    if len(positions) == 0:
        positions = np.zeros((1, 3))
    x_min, y_min = positions[:, 0].min() - TERRAIN_MARGIN, positions[:, 1].min() - TERRAIN_MARGIN
    x_max, y_max = positions[:, 0].max() + TERRAIN_MARGIN, positions[:, 1].max() + TERRAIN_MARGIN
    res = opts.terrain_res
    while True:
        nx = max(2, int(math.ceil((x_max - x_min) / res)))
        ny = max(2, int(math.ceil((y_max - y_min) / res)))
        if nx * ny <= opts.max_terrain_cells or res >= 2.0:
            break
        res *= 2.0
    return TerrainGrid(x0=float(x_min), y0=float(y_min), res=res, nx=nx, ny=ny)


def _pick(indices: np.ndarray, budget: int) -> np.ndarray:
    """Evenly-spaced subset of ``indices`` of at most ``budget`` entries."""
    if budget <= 0 or len(indices) == 0:
        return indices[:0]
    if len(indices) <= budget:
        return indices
    sel = np.linspace(0, len(indices) - 1, budget).round().astype(int)
    return indices[np.unique(sel)]


def _read_window(
    access: SceneAccess, kind: str, frames: np.ndarray, budget: int, loader: Callable
) -> tuple[np.ndarray, list, dict]:
    """``(chosen, loaded, stats)``: the positions in ``frames`` that have a ``kind``
    file, thinned to ``budget`` by :func:`_pick`, and their files read in threads."""
    have = access.available_frames(kind)
    usable = np.array([i for i, f in enumerate(frames) if int(f) in have], dtype=int)
    stats = {"requested": int(len(frames)), "available": int(len(usable)), "used": 0}
    chosen = _pick(usable, budget)
    return chosen, _read_all([access.frame_path(kind, int(frames[i])) for i in chosen], loader), stats


def _cell_xy(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Flattened local (x, y) of every cell of a robot-centred BEV map, row-major."""
    X, Y = np.meshgrid(*L.height_map_cell_centers(shape), indexing="ij")
    return X.ravel(), Y.ravel()


def _place(
    local: np.ndarray, pose: np.ndarray, ga: np.ndarray | None, frame: int, grid: TerrainGrid
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One robot-centred, gravity-aligned map's samples in the world grid.

    ``p_world = R_pose @ R_ga.T @ p_map + t_pose``: skipping the ``R_ga.T``
    would tilt every map by the robot's roll/pitch. Returns ``(world, row, col,
    inside)``, the last three as :meth:`TerrainGrid.cell_of` gives them.
    """
    R = pose[:3, :3]
    if ga is not None and frame < len(ga):
        R = R @ np.asarray(ga[frame], dtype=np.float64).T
    world = local @ R.T + pose[:3, 3]
    return (world, *grid.cell_of(world[:, 0], world[:, 1]))


def read_driven_similarity(path: Path) -> dict[str, Any] | None:
    """The layers of one ``driven_similarity`` sidecar, or None.

    ``driven`` and ``max_sim`` of one 2-D shape, ``threshold`` (0.9, as trained,
    when the file does not say) and ``max_sim_geom`` when the file has it.
    Tolerant the way ``loaders.load_npz_key`` is: a missing or malformed file is
    "nothing to show", not an error.
    """
    if not path.is_file():
        return None
    try:
        with np.load(path) as d:
            if "driven" not in d.files or "max_sim" not in d.files:
                return None
            out = {
                "driven": np.asarray(d["driven"], dtype=bool),
                "max_sim": np.asarray(d["max_sim"], dtype=np.float64),
                "threshold": float(d["threshold"]) if "threshold" in d.files else 0.9,
            }
            if "max_sim_geom" in d.files:
                out["max_sim_geom"] = np.asarray(d["max_sim_geom"], dtype=np.float64)
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("unreadable driven_similarity %s: %s", path, exc)
        return None
    if out["driven"].ndim != 2 or out["max_sim"].shape != out["driven"].shape:
        return None
    return out


def splat_height_maps(
    access: SceneAccess,
    grid: TerrainGrid,
    frames: np.ndarray,
    poses: np.ndarray,
    opts: BuildOptions,
) -> dict:
    """Accumulate per-frame BEV height maps into the world grid.

    Each map is robot-centred and *gravity-aligned*, so putting it in the world
    needs both the robot pose and the inverse of that frame's gravity rotation
    (:func:`_place`).

    Each sample carries a weight falling off with its distance from the robot
    that measured it (:data:`HM_BLEND_SIGMA`), so a cell is decided by the
    frames that drove over it rather than averaged flat across every frame that
    could see it.
    """
    chosen, maps, stats = _read_window(
        access, "height_maps", frames, opts.max_hm_frames, lambda p: L.load_npz_key(p, "height_map")
    )
    stats["with_masks"] = 0
    if not stats["available"]:
        return stats
    have_mask = access.available_frames("height_masks")
    ga = access.gravity_align
    masks = _read_all(
        [access.frame_path("height_masks", int(frames[i])) if int(frames[i]) in have_mask else None for i in chosen],
        lambda p: None if p is None else L.load_npz_key(p, "validity"),
    )

    for i, hm, mask in zip(chosen, maps, masks, strict=True):
        frame = int(frames[i])
        if hm is None or hm.ndim != 2:
            continue
        X, Y = _cell_xy(hm.shape)
        z = np.asarray(hm, dtype=np.float64).ravel()
        # A cell the generator left as NaN has to go before the grid ever sees
        # it. It would poison its grid cell for good -- the weighted sums go
        # NaN, ``covered`` stays true, and a NaN reaches the float32 heightfield
        # -- and the rotation below smears it across world x and y as well,
        # where the integer cast in ``cell_of`` is undefined. Clouds are
        # filtered the same way, in ``L.load_cloud``.
        finite = np.isfinite(z)
        if not finite.any():
            continue
        local = np.stack([X[finite], Y[finite], z[finite]], axis=1)

        world, row, col, inside = _place(local, poses[i], ga, frame, grid)
        if not inside.any():
            continue
        r, c = row[inside], col[inside]
        w = np.exp(-((np.hypot(local[inside, 0], local[inside, 1]) / HM_BLEND_SIGMA) ** 2))
        grid.sum_wz += grid.binned(r, c, world[inside, 2] * w)
        grid.sum_w += grid.binned(r, c, w)
        grid.count += grid.binned(r, c)
        stats["used"] += 1

        if mask is not None and mask.shape == hm.shape:
            # Rank, not the raw code. A root built with wheel-contact anchors
            # codes those cells 3, and averaging the codes would paint the one
            # region the lidar never saw as the best-measured in the scene.
            # The lookup is the identity on the three states every published
            # root ships (loaders.VALIDITY_RANK).
            code = np.clip(np.asarray(mask, dtype=np.intp), 0, len(L.VALIDITY_RANK) - 1)
            ranked = L.VALIDITY_RANK[code]
            flat = ranked.astype(np.float64).ravel()[finite][inside]
            grid.sum_wv += grid.binned(r, c, flat * w)
            grid.sum_w_v += grid.binned(r, c, w)
            wm = w * (flat == 2)
            grid.sum_wz_m += grid.binned(r, c, world[inside, 2] * wm)
            grid.sum_w_m += grid.binned(r, c, wm)
            stats["with_masks"] += 1
    return stats


def splat_driven_similarity(
    access: SceneAccess,
    grid: TerrainGrid,
    frames: np.ndarray,
    poses: np.ndarray,
    opts: BuildOptions,
) -> dict:
    """Accumulate the per-keyframe ``driven_similarity`` sidecars into two layers.

    Same grid, same layout and the same ``R_pose @ R_ga.T`` placement as
    :func:`splat_height_maps`, so a driven cell lands on the terrain cell whose
    height came from the same map. ``driven`` is the union over frames (a cell
    is driven once any wheel rolled over it); ``drivable`` keeps each cell's
    highest similarity to a driven cell, clipped below at 0 -- a max, not a
    weighted mean, because the question is "did anything the robot drove on ever
    look like this", and a distant frame's answer is as good as a near one's.
    Frames without a sidecar (no driven cell in their horizon) contribute
    nothing, which is what they have to say.
    """
    chosen, sidecars, stats = _read_window(
        access, "driven_similarity", frames, opts.max_hm_frames, read_driven_similarity
    )
    if not stats["available"]:
        return stats
    ga = access.gravity_align

    driven_layer = grid.layers.setdefault("driven", np.zeros((grid.ny, grid.nx), dtype=np.float64))
    drivable_layer = grid.layers.setdefault("drivable", np.zeros((grid.ny, grid.nx), dtype=np.float64))
    for i, sidecar in zip(chosen, sidecars, strict=True):
        if sidecar is None:
            continue
        driven, max_sim = sidecar["driven"], sidecar["max_sim"]
        frame = int(frames[i])
        X, Y = _cell_xy(driven.shape)
        sim = np.clip(np.nan_to_num(max_sim, nan=0.0), 0.0, 1.0).ravel()
        drv = driven.ravel()
        # Only cells that say something: the driven ones and any with a
        # positive similarity. The rest would splat zeros over other frames'
        # answers -- harmless under a max, but a million samples of nothing.
        keep = drv | (sim > 0.0)
        if not keep.any():
            continue
        local = np.stack([X[keep], Y[keep], np.zeros(int(keep.sum()))], axis=1)
        _, row, col, inside = _place(local, poses[i], ga, frame, grid)
        if not inside.any():
            continue
        r, c = row[inside], col[inside]
        np.maximum.at(driven_layer, (r, c), drv[keep][inside].astype(np.float64))
        np.maximum.at(drivable_layer, (r, c), sim[keep][inside])
        stats["used"] += 1
    return stats


def aggregate_clouds(
    access: SceneAccess,
    grid: TerrainGrid,
    frames: np.ndarray,
    poses: np.ndarray,
    opts: BuildOptions,
) -> tuple[np.ndarray | None, dict]:
    """Read a bounded subset of the window's clouds into one world-frame cloud.

    Returns the stacked ``(N, 4)`` array (x, y, z, intensity) and read stats.
    Also fills the grid's cloud-derived layers (point count, mean intensity,
    vertical spread), since the points are in hand anyway.
    """
    chosen, clouds, stats = _read_window(access, "clouds", frames, opts.max_cloud_frames, L.load_cloud)
    stats["points"] = 0
    if len(chosen) == 0:
        return None, stats
    per_frame = max(200, opts.max_cloud_points // max(1, len(chosen)))
    rng = np.random.default_rng(12345)

    sum_i = np.zeros((grid.ny, grid.nx), dtype=np.float64)
    cnt_i = np.zeros((grid.ny, grid.nx), dtype=np.int64)
    max_z = np.full((grid.ny, grid.nx), -np.inf)
    min_z = np.full((grid.ny, grid.nx), np.inf)

    parts: list[np.ndarray] = []
    for i, cloud in zip(chosen, clouds, strict=True):
        if cloud is None or len(cloud) == 0:
            continue
        R, t = poses[i][:3, :3], poses[i][:3, 3]
        world_xyz = cloud[:, :3].astype(np.float64) @ R.T + t
        intensity = cloud[:, 3].astype(np.float64) if cloud.shape[1] > 3 else np.zeros(len(cloud))

        row, col, inside = grid.cell_of(world_xyz[:, 0], world_xyz[:, 1])
        if inside.any():
            r, c = row[inside], col[inside]
            sum_i += grid.binned(r, c, intensity[inside])
            cnt_i += grid.binned(r, c)
            np.maximum.at(max_z, (r, c), world_xyz[inside, 2])
            np.minimum.at(min_z, (r, c), world_xyz[inside, 2])

        if len(cloud) > per_frame:
            keep = rng.choice(len(cloud), per_frame, replace=False)
            world_xyz, intensity = world_xyz[keep], intensity[keep]
        parts.append(np.column_stack([world_xyz, intensity]))
        stats["used"] += 1

    covered = cnt_i > 0
    pts = np.zeros((grid.ny, grid.nx))
    pts[covered] = cnt_i[covered]
    grid.layers["cloud_points"] = pts
    inten = np.zeros((grid.ny, grid.nx))
    inten[covered] = sum_i[covered] / cnt_i[covered]
    grid.layers["cloud_intensity"] = inten
    spread = np.zeros((grid.ny, grid.nx))
    spread[covered] = max_z[covered] - min_z[covered]
    grid.layers["cloud_spread"] = spread

    if not parts:
        return None, stats
    cloud = np.vstack(parts)
    stats["points"] = int(len(cloud))
    return cloud, stats


def _fill_coarse_nans(coarse: np.ndarray, max_passes: int = 400) -> np.ndarray:
    """Flood unobserved coarse cells outward from the observed ones.

    Each pass averages the not-yet-filled cells over whichever of their four
    neighbours already have a value, so the surface grows out of the observed
    corridor instead of jumping to a constant.
    """
    out = coarse.copy()
    for _ in range(max_passes):
        holes = np.isnan(out)
        if not holes.any():
            break
        # The four neighbours of every cell; the NaN border stands for "none".
        p = np.pad(out, 1, constant_values=np.nan)
        neighbours = np.stack([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
        valid = ~np.isnan(neighbours)
        total = np.where(valid, neighbours, 0.0).sum(axis=0)
        count = valid.sum(axis=0)
        fillable = holes & (count > 0)
        if not fillable.any():
            break
        out[fillable] = total[fillable] / count[fillable]
    if np.isnan(out).any():  # nothing observed at all
        out[np.isnan(out)] = 0.0
    return out


def _upsample_bilinear(coarse: np.ndarray, ny: int, nx: int, by: int, bx: int) -> np.ndarray:
    """Bilinearly expand a block-reduced grid back to full resolution."""
    cy, cx = coarse.shape
    y = (np.arange(ny) + 0.5) / by - 0.5
    x = (np.arange(nx) + 0.5) / bx - 0.5
    y0 = np.clip(np.floor(y), 0, cy - 1).astype(int)
    x0 = np.clip(np.floor(x), 0, cx - 1).astype(int)
    y1 = np.clip(y0 + 1, 0, cy - 1)
    x1 = np.clip(x0 + 1, 0, cx - 1)
    wy = np.clip(y - y0, 0.0, 1.0)[:, None]
    wx = np.clip(x - x0, 0.0, 1.0)[None, :]
    top = coarse[np.ix_(y0, x0)] * (1 - wx) + coarse[np.ix_(y0, x1)] * wx
    bottom = coarse[np.ix_(y1, x0)] * (1 - wx) + coarse[np.ix_(y1, x1)] * wx
    return top * (1 - wy) + bottom * wy


def _background_surface(height: np.ndarray, covered: np.ndarray, blocks: int = 56) -> np.ndarray:
    """A smooth trend surface through the observed cells.

    Unobserved cells have to be given *some* height -- the viewer's heightfield
    has no notion of a hole. Filling them with one global constant produces a
    huge flat plane cutting through a scene that climbs or descends, which
    reads as terrain that is not there. Extending the observed trend instead
    keeps the surroundings plausible, and the ``coverage`` colour layer still
    marks exactly which cells were actually observed.
    """
    ny, nx = height.shape
    by = max(1, int(math.ceil(ny / blocks)))
    bx = max(1, int(math.ceil(nx / blocks)))
    pad_y, pad_x = (-ny) % by, (-nx) % bx
    h = np.pad(height, ((0, pad_y), (0, pad_x)))
    c = np.pad(covered, ((0, pad_y), (0, pad_x)))
    gy, gx = h.shape[0] // by, h.shape[1] // bx
    blocks_h = h.reshape(gy, by, gx, bx)
    blocks_c = c.reshape(gy, by, gx, bx).astype(np.float64)
    total = (blocks_h * blocks_c).sum(axis=(1, 3))
    count = blocks_c.sum(axis=(1, 3))
    coarse = np.where(count > 0, total / np.maximum(count, 1.0), np.nan)
    coarse = _fill_coarse_nans(coarse)
    return _upsample_bilinear(coarse, ny, nx, by, bx)


def _finish_terrain(grid: TerrainGrid, cloud: np.ndarray | None) -> tuple[np.ndarray, dict, dict]:
    """Turn the accumulated grid into a height field plus its colour layers.

    Cells nothing was observed in are filled from the trend surface through
    the observed ones (:func:`_background_surface`) and reported through the
    ``coverage`` layer, so invented ground is visible in the viewer instead of
    being indistinguishable from measured ground.
    """
    covered = grid.sum_w > 0.0
    height = np.zeros((grid.ny, grid.nx), dtype=np.float64)
    if covered.any():
        height[covered] = grid.sum_wz[covered] / grid.sum_w[covered]
        measured = grid.sum_w_m > 0.0
        height[measured] = grid.sum_wz_m[measured] / grid.sum_w_m[measured]
        background = _background_surface(height, covered)
        height[~covered] = background[~covered]
    else:
        fill = float(np.median(cloud[:, 2])) if cloud is not None and len(cloud) else 0.0
        height[:] = fill

    validity = np.zeros((grid.ny, grid.nx), dtype=np.float64)
    said = grid.sum_w_v > 0.0
    validity[said] = grid.sum_wv[said] / grid.sum_w_v[said]
    # A measured height is what the cell shows, so that is what it reports.
    validity[grid.sum_w_m > 0.0] = 2.0

    layers = {
        "coverage": (grid.count > 0).astype(np.float64) * grid.count,
        "validity": validity,
    }
    layers.update(grid.layers)
    bounds = {"validity": (0.0, 2.0), "driven": (0.0, 1.0), "drivable": (0.0, 1.0)}
    # Drop layers that carry no information (all-zero) so the viewer's colour
    # mode list only offers modes that show something.
    layers = {k: v for k, v in layers.items() if np.any(v != v.flat[0])}
    bounds = {k: v for k, v in bounds.items() if k in layers}
    return height, layers, bounds


def _terrain_from_cloud(grid: TerrainGrid, cloud: np.ndarray) -> None:
    """Fill the grid's height from the aggregated cloud (mean z per cell).

    Used when a scene has no height maps at all -- the terrain is then the
    lidar's own ground estimate rather than the pipeline's BEV product.
    """
    row, col, inside = grid.cell_of(cloud[:, 0], cloud[:, 1])
    if not inside.any():
        return
    r, c, z = row[inside], col[inside], cloud[inside, 2]
    # Points, not a robot-centred raster: nothing here to weight by distance.
    grid.sum_wz += grid.binned(r, c, z)
    hits = grid.binned(r, c)
    grid.sum_w += hits
    grid.count += hits


# --------------------------------------------------------------------------
# scalars
# --------------------------------------------------------------------------


def build_scalars(
    access: SceneAccess, window: Window, times: np.ndarray, gt: PoseTrack, ghost: PoseTrack | None
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Every time series the scene can offer, resampled onto the frame clock.

    Channels whose source file is absent are simply not created -- the viewer's
    plot list is therefore itself a readout of what the scene contains.
    """
    scalars: dict[str, np.ndarray] = {}
    notes: list[str] = []
    dt = np.gradient(times) if len(times) > 1 else np.array([1.0])
    dt = np.where(dt > 1e-6, dt, 1e-6)

    if gt.source != "none":
        # A one-frame window has no motion to differentiate: the derived
        # channels are zero by construction rather than missing.
        if len(times) > 1:
            vel = np.gradient(gt.positions, axis=0) / dt[:, None]
            scalars["speed"] = np.linalg.norm(vel[:, :2], axis=1)
        else:
            scalars["speed"] = np.zeros(len(times))
        scalars["pose_z"] = gt.positions[:, 2]
        yaw = np.arctan2(
            2 * (gt.quats[:, 0] * gt.quats[:, 3] + gt.quats[:, 1] * gt.quats[:, 2]),
            1 - 2 * (gt.quats[:, 2] ** 2 + gt.quats[:, 3] ** 2),
        )
        scalars["yaw_rate"] = np.gradient(np.unwrap(yaw)) / dt if len(times) > 1 else np.zeros(len(times))
        if not gt.valid.all():
            # Outside the pose source's own time span: mark rather than invent.
            for key in ("speed", "pose_z", "yaw_rate"):
                scalars[key] = np.where(gt.valid, scalars[key], np.nan)
            notes.append(f"{int((~gt.valid).sum())} frames lie outside the pose track")

    if ghost is not None:
        scalars["wheel_odom_drift"] = np.linalg.norm(ghost.positions - gt.positions, axis=1)

    ga = access.gravity_align
    if ga is not None:
        idx = np.clip(np.arange(window.frame_start, window.frame_end), 0, len(ga) - 1)
        R_imu = np.asarray(ga)[idx][: len(times)]
        roll, pitch = L.roll_pitch_from_gravity_align(R_imu)
        scalars["roll_deg_imu"] = np.degrees(roll)
        scalars["pitch_deg_imu"] = np.degrees(pitch)
        if gt.source != "none":
            # The window is levelled, so the ground truth now carries an
            # absolute attitude too and the two are directly comparable. Row 2
            # of either rotation is the world's up axis in body coordinates,
            # which is what makes the same reader work on both.
            R_gt = poses_from_quat(gt.positions, gt.quats)[:, :3, :3]
            roll, pitch = L.roll_pitch_from_gravity_align(R_gt)
            scalars["roll_deg_gt"] = np.degrees(roll)
            scalars["pitch_deg_gt"] = np.degrees(pitch)
            scalars["attitude_error_deg"] = attitude_error_deg(R_gt, R_imu)

    js = access.joint_states
    if js is not None and js.ndim == 2 and js.shape[1] >= 10:
        res = L.resample_series(js[:, 0], js[:, 1:10], times)
        for k, label in enumerate(WHEEL_LABELS):
            scalars[f"wheel_vel_{label}"] = res[:, 3 + k]
            scalars[f"wheel_effort_{label}"] = res[:, 6 + k]
    elif js is not None:
        notes.append(f"joint_states.npy has unexpected shape {js.shape}; skipped")

    sp = access.joint_setpoints
    if sp is not None and sp.ndim == 2 and sp.shape[1] >= 10:
        res = L.resample_series(sp[:, 0], sp[:, 1:10], times)
        for k, label in enumerate(WHEEL_LABELS):
            scalars[f"wheel_setpoint_{label}"] = res[:, 3 + k]
            # Commanded minus measured: where slip and load show up, per wheel.
            # Both series are already on the frame clock, so the difference is
            # taken there rather than by resampling either again.
            if f"wheel_vel_{label}" in scalars:
                scalars[f"wheel_slip_{label}"] = res[:, 3 + k] - scalars[f"wheel_vel_{label}"]

    cv = access.cmd_vel
    if cv is not None and cv.ndim == 2 and cv.shape[1] >= 7:
        res = L.resample_series(cv[:, 0], cv[:, [1, 6]], times)
        scalars["cmd_linear_x"] = res[:, 0]
        scalars["cmd_angular_z"] = res[:, 1]
    elif cv is not None:
        notes.append(f"cmd_vel.npy has unexpected shape {cv.shape}; skipped")

    return scalars, notes


#: Tilt disagreement (degrees) at which the panel stops calling the ground
#: truth sound. Measured across this root: segments whose GT attitude is
#: healthy sit under 11 deg from the IMU even on sustained steep slopes, and a
#: segment whose GT has actually failed sits above 40.
ATTITUDE_BANDS = {"ok": 10.0, "warn": 20.0}


def attitude_error_deg(R_gt: np.ndarray, R_imu: np.ndarray) -> np.ndarray:
    """Angle (degrees) between the two rotations' up axes, per frame.

    The tilt disagreement itself, free of yaw and of the wrap a per-angle
    difference would hit near +-180 deg (a robot on its back). Row 2 of either
    rotation is the world's up axis in body coordinates, which is what makes
    the same reader work on the ground truth and on ``gravity_align.npy``.
    One function for the panel's per-frame plot and the window tables' verdict.
    """
    cos = np.clip(np.sum(R_gt[:, 2, :] * R_imu[:, 2, :], axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def attitude_band(median_deg: float) -> str:
    """``ok`` / ``warn`` / ``bad`` for a window's median tilt gap (:data:`ATTITUDE_BANDS`)."""
    if median_deg < ATTITUDE_BANDS["ok"]:
        return "ok"
    return "warn" if median_deg < ATTITUDE_BANDS["warn"] else "bad"


def window_attitude(access: SceneAccess, window: Window) -> dict | None:
    """The ground-truth-versus-IMU verdict for one window, without a build.

    The panel computes this inside the bundle, which exists only once someone
    has opened the window. The tables listing a scene's windows want it before
    that -- it is the one number that says which segments are worth opening --
    so the same comparison is run here on the window's own frames: the poses
    interpolated onto the frame clock and levelled exactly as a build does
    (:func:`resolve_poses`), against ``gravity_align.npy``. Numpy only, no
    torch, and cached on the scene handle. None for anything but a segment
    window, and for a scene with no gravity alignment to compare against.
    """
    if window.kind != "segment" or not access.probe.has("gravity_align"):
        return None

    def work() -> dict | None:
        ga = access.gravity_align
        frame_ts = access.frame_ts
        if ga is None or frame_ts is None or not len(frame_ts):
            return None
        frames = np.arange(window.frame_start, min(window.frame_end, len(frame_ts)))
        if len(frames) == 0:
            return None
        gt = resolve_poses(access, window, frame_ts[frames])
        if gt.source != "odometry":
            return None
        R_gt = poses_from_quat(gt.positions, gt.quats)[:, :3, :3]
        R_imu = np.asarray(ga)[np.clip(frames, 0, len(ga) - 1)]
        err = attitude_error_deg(R_gt, R_imu)
        median = float(np.median(err))
        return {
            "median": round(median, 2),
            "p95": round(float(np.percentile(err, 95)), 2),
            "band": attitude_band(median),
        }

    return access.cached(f"attitude_{window.key}", work)


def attitude_vs_imu(scalars: dict[str, np.ndarray]) -> dict | None:
    """The GT-vs-IMU attitude comparison the side panel reads, or None.

    Once a window is levelled both attitudes are absolute, so the gap between
    them says how far the ground truth's own orientation has wandered -- the one
    reading that separates a robot really on a slope from odometry that thinks
    it is. The IMU is independent of the odometry, which is what makes it a
    check rather than a restatement.
    """
    need = ("roll_deg_gt", "pitch_deg_gt", "roll_deg_imu", "pitch_deg_imu", "attitude_error_deg")
    if not all(k in scalars for k in need):
        return None
    err = np.asarray(scalars["attitude_error_deg"], dtype=np.float64)
    return {
        "gt_roll": np.round(scalars["roll_deg_gt"], 2).tolist(),
        "gt_pitch": np.round(scalars["pitch_deg_gt"], 2).tolist(),
        "imu_roll": np.round(scalars["roll_deg_imu"], 2).tolist(),
        "imu_pitch": np.round(scalars["pitch_deg_imu"], 2).tolist(),
        "error": np.round(err, 2).tolist(),
        "median_error": round(float(np.median(err)), 2),
        "p95_error": round(float(np.percentile(err, 95)), 2),
        "bands": ATTITUDE_BANDS,
        "band": attitude_band(float(np.median(err))),
    }


def wheel_angles(access: SceneAccess, times: np.ndarray) -> np.ndarray | None:
    """Per-frame wheel rotation angle (rad) for the three wheels, or None."""
    js = access.joint_states
    if js is None or js.ndim != 2 or js.shape[1] < 4:
        return None
    return L.resample_series(js[:, 0], js[:, 1:4], times)


# --------------------------------------------------------------------------
# the build
# --------------------------------------------------------------------------


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate one body-frame vector ``v`` (3,) by each [w,x,y,z] quaternion in ``q`` (T, 4)."""
    w, u = q[:, :1], q[:, 1:]
    uv = np.cross(u, v[None, :])
    return v[None, :] + 2.0 * w * uv + 2.0 * np.cross(u, uv)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of [w,x,y,z] quaternions, batched over the last axis."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def build_scene(
    root: DatasetRoot,
    scene_id: str,
    window_key: str,
    opts: BuildOptions,
    max_frames: int = DEFAULT_MAX_FRAMES,
    max_seconds: float = DEFAULT_MAX_SECONDS,
):
    """Build (but do not save) the SimView scene for one window.

    Returns ``(scene, meta)``; ``meta`` is the per-frame index the side panel
    needs plus a human-readable record of what went into the build.
    """
    from simview.scene import BodyShapeType, SimulationScene  # local: heavy import
    from simview.state import BodyTrajectory

    opts = opts.sanitized()
    access = SceneAccess(root, scene_id)
    window = access.window(window_key, max_frames, max_seconds)
    if window is None:
        raise KeyError(f"scene {scene_id} has no window {window_key!r}")

    frames = np.arange(window.frame_start, window.frame_end)
    frame_ts = access.frame_ts
    if frame_ts is None or len(frame_ts) == 0:
        raise ValueError(f"scene {scene_id} has no frame timestamps")
    frames = frames[frames < len(frame_ts)]
    if len(frames) == 0:
        # Only reachable for a window whose whole range is past the end of the
        # timestamp track; `windows` refuses to enumerate one, so this is the
        # guard for a hand-typed key rather than a path the UI can take.
        raise KeyError(
            f"scene {scene_id} window {window_key!r} covers frames "
            f"{window.frame_start}-{window.frame_end - 1}, none of which has a "
            f"timestamp ({len(frame_ts)} in data_ts.txt)"
        )
    abs_times = frame_ts[frames]
    times = abs_times - abs_times[0]
    # After the frame range is final: what the window costs to sample is set by
    # how much of it there is, not by what the key asked for.
    opts = budgets_for_window(opts, len(frames))

    gt = resolve_poses(access, window, abs_times)
    ghost = _ghost_track(access, window, gt)
    poses = poses_from_quat(gt.positions, gt.quats)

    # ---- terrain -------------------------------------------------------
    grid = _make_grid(gt.positions, opts)
    hm_stats = splat_height_maps(access, grid, frames, poses, opts)
    driven_stats = splat_driven_similarity(access, grid, frames, poses, opts)
    cloud, cloud_stats = aggregate_clouds(access, grid, frames, poses, opts)
    terrain_source = "height_maps"
    if hm_stats["used"] == 0:
        terrain_source = "clouds" if cloud is not None else "flat"
        if cloud is not None:
            _terrain_from_cloud(grid, cloud)
    height, layers, bounds = _finish_terrain(grid, cloud)

    scalars, notes = build_scalars(access, window, abs_times, gt, ghost)
    scalar_names = sorted(scalars)

    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.1
    metadata = {
        "dataset_root": str(root.path),
        "scene": scene_id,
        "window": window.to_json(),
        "pose_source": gt.source,
        "pose_note": gt.note,
        "terrain_source": terrain_source,
        "terrain_resolution_m": grid.res,
        "terrain_cells": [grid.ny, grid.nx],
        "terrain_color_modes": terrain_color_modes(layers.keys()),
        "height_maps": hm_stats,
        "driven_similarity": driven_stats,
        "clouds": cloud_stats,
        "frames": [int(frames[0]), int(frames[-1])],
        "notes": notes,
        # What the scene lacks that would make this view fuller; optional
        # extras (cmd_vel, the v1 camera file) are left out as noise.
        "missing_files": [k for k in access.probe.missing if k not in OPTIONAL_ARTEFACTS],
    }
    if access.K is not None:
        metadata["camera_intrinsics"] = np.asarray(access.K).round(3).tolist()

    scene = SimulationScene(batch_size=1, scalar_names=scalar_names, dt=max(dt, 1e-3), metadata=metadata)

    import torch

    scene.create_terrain(
        heightmap=torch.from_numpy(np.ascontiguousarray(height, dtype=np.float32)),
        x_lim=grid.x_lim,
        y_lim=grid.y_lim,
        grid_res=grid.res,
        properties={k: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)) for k, v in layers.items()} or None,
        property_bounds=bounds or None,
    )

    # ---- bodies --------------------------------------------------------
    scene.create_body("robot", BodyShapeType.BOX, **CHASSIS)
    # Cylinders point along +z locally (the viewer rotates the geometry), so a
    # wheel needs a -90 deg turn about x to put its axis along the body's y.
    for name in WHEELS:
        scene.create_body(name, BodyShapeType.CYLINDER, radius=WHEEL_RADIUS, height=WHEEL_WIDTH, parent="robot")
    if ghost is not None:
        # A deliberately small, flat marker: it shows where wheel odometry thinks
        # the robot is, without reading as a second robot.
        scene.create_body("wheel_odom_ghost", BodyShapeType.BOX, hx=0.2, hy=0.12, hz=0.03)
    if cloud is not None and len(cloud):
        values = cloud[:, 2] if opts.cloud_color == "height" else cloud[:, 3]
        colors = unit_colors(values)
        scene.create_pointcloud(
            "lidar_cloud",
            points=torch.from_numpy(np.ascontiguousarray(cloud[:, :3], dtype=np.float32)),
            color=torch.from_numpy(np.ascontiguousarray(colors, dtype=np.float32)),
        )

    # ---- trajectories --------------------------------------------------
    T = len(times)
    # The box is drawn about the body origin, so the body rides CHASSIS_CENTER
    # ahead of/behind base_link, rotated with the pose; the wheels below undo it.
    chassis_pos = gt.positions + _rotate(gt.quats, np.array(CHASSIS_CENTER, dtype=np.float64))
    trajectories = [BodyTrajectory("robot", positions=chassis_pos, orientations=gt.quats)]

    angles = wheel_angles(access, abs_times)
    for name, (wx, wy, wz, col, sign) in WHEELS.items():
        hub = np.array([wx, wy, wz], dtype=np.float64) - np.array(CHASSIS_CENTER, dtype=np.float64)
        offset = np.tile(hub, (T, 1))
        # +-90 deg about x (sign mirrored per side, as in DRIFT's helper).
        q_align = np.array([math.cos(sign * math.pi / 4), math.sin(sign * math.pi / 4), 0.0, 0.0])
        if angles is not None:
            theta = np.nan_to_num(angles[:, col], nan=0.0)
            spin = np.stack([np.cos(theta / 2), np.zeros(T), np.sin(theta / 2), np.zeros(T)], axis=1)
            quat = _quat_mul(spin, np.tile(q_align, (T, 1)))
        else:
            quat = np.tile(q_align, (T, 1))
        trajectories.append(BodyTrajectory(name, positions=offset, orientations=quat))

    if ghost is not None:
        trajectories.append(BodyTrajectory("wheel_odom_ghost", positions=ghost.positions, orientations=ghost.quats))
    # `lidar_cloud` deliberately gets *no* trajectory. It is the window's clouds
    # already accumulated in the world frame, so its transform is the identity
    # at every frame -- a per-frame pose that says nothing, costs a blob, and
    # (in SimView >= 4.3) is exactly what keeps a point-cloud body in the "Body
    # states" list, which is not a list it belongs in.

    scene.add_trajectory(
        times=times,
        trajectories=trajectories,
        scalar_values={k: v.astype(np.float32) for k, v in scalars.items()} or None,
    )

    meta = {
        "meta_version": META_VERSION,
        "scene": scene_id,
        "window": window.to_json(),
        "frames": frames.tolist(),
        "times": times.tolist(),
        "abs_times": abs_times.tolist(),
        "pose_source": gt.source,
        "pose_valid": gt.valid.tolist(),
        "attitude": attitude_vs_imu(scalars),
        "scalar_names": scalar_names,
        "build": metadata,
        "available": {
            kind: sorted(int(f) for f in access.available_frames(kind) & set(frames.tolist()))
            for kind in ("images", "clouds", "height_maps", "height_masks", "driven_similarity")
            if access.probe.has(kind)
        },
    }
    return scene, meta
