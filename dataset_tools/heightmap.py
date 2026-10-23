"""Height maps and validity masks from a scene's clouds (``height_maps/``, ``height_masks/``).

    python -m dataset_tools.heightmap ROOT [--scenes 0029,0030] [--workers 16]

One map per frame: the cloud levelled by ``gravity_align.npy``, binned onto a
128 x 128 grid of 0.1 m cells (+-6.4 m around base_link, ``[row = x forward,
col = y left]``), interpolated, filled and smoothed. The mask says where each
cell's value came from (``VALIDITY_*``). A frame with fewer than
``min_points`` returns left after the radius and canopy cuts gets no map.

The settings in :class:`Params` are the ones every HELHEST v2 root was built
with. Until 2026-10 this lived in DRIFT (``scripts/data/build_heightmaps.py``
and ``utils/pcd.py``, run by its ``slurm/build_heightmaps_v2_beta.batch``);
the code here reproduces those maps to within float rounding
(``tests/test_heightmap.py`` holds frames DRIFT built). Each scene records the
settings it was built with in ``height_maps.json``.

Why the settings are what they are
----------------------------------
- ``min_radius`` 0.65: the Odin1 sees the robot's own front wheels at
  r 0.5-0.6 and a near-range ramp of ground that reads 5-13 cm high.
- ``clip`` 1.0 m above base_link: canopy above that is not terrain.
- ``statistic`` p90: the median puts a trunk or a person at the middle of its
  visible extent (+0.7 m whatever its height); p90 reads +1.1 m there and costs
  ~1 cm on ground.
- ``max_edge`` 1.0: Delaunay triangles with a longer edge bridge the blind
  region under and behind the robot and invent a ~0.2 m wall in front of the
  rear wheel; their cells are dropped to the fill and marked extrapolated.
- ``anchors``: pseudo-returns under the three wheels (and a 0.45 m ring round
  each) -- the robot stands on that ground. The tilted contact plane is
  shifted by one offset onto the lidar ground measured in the
  ``anchor_ground_inner``-``anchor_ground_radius`` annulus, so it keeps the
  slope the attitude reports and meets the lidar ground, which reads ~7 cm
  above the wheel bottoms on this rig (an open calibration item; with the
  shift the wheel radius does not set any height). An anchor speaks only
  within ``anchor_radius``; beyond it the nearest return fills.
- ``fill`` ground: cells outside the hull of returns take the median measured
  height within ``fill_radius`` of the robot, not whatever stands at the
  field-of-view rim.
- ``tie_jitter`` (off in v2.0-v2.4): the cell centres are co-circular, so
  Delaunay breaks ties by input order and a 1 um anchor nudge changed ~28 cells
  a frame by over 1 cm (up to 0.77 m). A fixed seeded table of +-0.1 mm xy
  offsets, looked up by each point's cell, moves only the points the
  triangulation is built from; DRIFT's utils/pcd.py uses the same table.
"""

from __future__ import annotations

import argparse
import functools
import json
import multiprocessing as mp
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path

# One BLAS thread per pool worker: OpenBLAS otherwise starts one per core in
# each of them, and the busy-waiting threads halved the build rate (19 s -> 9 s
# a scene, maps identical). Only takes effect if numpy is not imported yet.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from scipy.interpolate import LinearNDInterpolator, griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import Delaunay, cKDTree
from scipy.stats import binned_statistic_2d

from dataset_tools.gt_io import _replace_atomically
from dataset_tools.time_align import usable_cpus

#: Outside the hull of returns: filled, not measured.
VALIDITY_EXTRAPOLATED = 0
#: No return in the cell, but inside the hull: interpolated from returns nearby.
VALIDITY_INTERPOLATED = 1
#: At least one return landed in the cell.
VALIDITY_MEASURED = 2
#: The value comes from a wheel-contact anchor, not from a return.
VALIDITY_ANCHORED = 3

#: Re-levelling and the ground fill need at least this many measured cells, else they keep the plain value.
ANCHOR_MIN_GROUND_CELLS = 8

STATISTICS = {"median": None, "max": 1.0, "p90": 0.9, "p75": 0.75}

SETTINGS_FILE = "height_maps.json"

#: The tie-break table HELHEST v2.5 is built with (``tie_jitter``); shared with DRIFT's builder.
TIE_JITTER_V1 = {"seed": 12345, "amplitude_m": 1e-4, "shape": [130, 130, 2], "rng": "numpy.default_rng.uniform"}


@dataclass(frozen=True)
class Params:
    """How a map is built. The defaults are HELHEST v2's (every v2.x root)."""

    resolution_m: float = 0.1
    extents_m: float = 6.4
    canopy_clip_m: float = 1.0
    min_radius_m: float = 0.65
    min_points: int = 250
    statistic: str = "p90"
    max_edge_m: float | None = 1.0
    fill: str = "ground"
    fill_radius_m: float = 2.0
    anchors: bool = True
    anchor_radius_m: float | None = 1.0
    #: "shift": the contact plane moved onto the measured ground; "kinematic": z = -wheel_radius as is.
    anchor_relevel: str = "shift"
    anchor_ground_inner_m: float = 0.65
    anchor_ground_radius_m: float = 2.5
    #: Only the "kinematic" relevel reads it as a height (the URDF's value; the rig rolls 0.334 m/rad).
    wheel_radius_m: float = 0.36
    #: Wheel centres in base_link, which is the front axle: left, right, rear.
    wheel_centres_m: tuple = field(default=((0.0, 0.36), (0.0, -0.36), (-0.697, 0.0)))
    #: None: off. Else a table like TIE_JITTER_V1, applied to the points the triangulation is built from.
    tie_jitter: dict | None = None

    def __post_init__(self):
        if self.statistic not in STATISTICS:
            raise ValueError(f"statistic must be one of {sorted(STATISTICS)}, got {self.statistic!r}")
        if self.fill not in ("nearest", "ground"):
            raise ValueError(f"fill must be 'nearest' or 'ground', got {self.fill!r}")
        if self.anchor_relevel not in ("shift", "kinematic"):
            raise ValueError(f"anchor_relevel must be 'shift' or 'kinematic', got {self.anchor_relevel!r}")
        if self.tie_jitter is not None:
            n = round(2 * self.extents_m / self.resolution_m) + 2  # one cell of margin each side
            if self.tie_jitter.get("rng") != TIE_JITTER_V1["rng"] or list(self.tie_jitter.get("shape", [])) != [
                n,
                n,
                2,
            ]:
                raise ValueError(
                    f"tie_jitter must be numpy.default_rng.uniform of shape [{n}, {n}, 2], got {self.tie_jitter!r}"
                )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["wheel_centres_m"] = [list(c) for c in self.wheel_centres_m]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Params:
        known = {f.name for f in fields(cls)}
        kw = {k: v for k, v in d.items() if k in known}
        if "wheel_centres_m" in kw:
            kw["wheel_centres_m"] = tuple(tuple(c) for c in kw["wheel_centres_m"])
        return cls(**kw)


def wheel_contact_anchors(wheel_centres, wheel_radius, R_align=None, ring=0.45, n_ring=8):
    """Pseudo-returns of the ground under the wheels: each contact at z = -r and a ring round it.

    One point per wheel does not stop the triangulation bridging the blind
    hole, hence the ring. ``R_align`` levels them like the cloud (``p @ R.T``).
    """
    c = np.asarray(wheel_centres, dtype=float).reshape(-1, 2)
    contacts = np.column_stack([c, np.full(len(c), -float(wheel_radius))])
    pts = [contacts]
    if n_ring:
        ang = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
        offsets = np.stack([ring * np.cos(ang), ring * np.sin(ang), np.zeros(n_ring)], axis=-1)
        pts.append((contacts[:, None, :] + offsets[None, :, :]).reshape(-1, 3))
    out = np.vstack(pts)
    if R_align is not None:
        out = out @ np.asarray(R_align, dtype=float).T
    return out


@functools.lru_cache(maxsize=4)
def _jitter_table(seed: int, amplitude: float, shape: tuple) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-amplitude, amplitude, size=shape)


def tie_jitter_table(spec: dict) -> np.ndarray:
    """The ``tie_jitter`` offsets, ``(n, n, 2)`` float64; DRIFT checks its copy by ``sha256(table.tobytes())``."""
    return _jitter_table(int(spec["seed"]), float(spec["amplitude_m"]), tuple(spec["shape"]))


def _jittered(xy, p: Params):
    """The triangulation's copy of the points: each moved by its cell's table entry (cell index + 1, clipped)."""
    if p.tie_jitter is None:
        return xy
    J = tie_jitter_table(p.tie_jitter)
    i = np.clip(np.floor((xy[:, 0] + p.extents_m) / p.resolution_m).astype(int) + 1, 0, J.shape[0] - 1)
    j = np.clip(np.floor((xy[:, 1] + p.extents_m) / p.resolution_m).astype(int) + 1, 0, J.shape[1] - 1)
    return xy + J[i, j]


def _binned_quantile(x, y, z, x_bins, y_bins, q):
    """Per-cell quantile of z (nearest rank, lower), NaN where a cell is empty; the grid of binned_statistic_2d."""
    nx, ny = len(x_bins) - 1, len(y_bins) - 1
    i = np.searchsorted(x_bins, x, side="right") - 1
    j = np.searchsorted(y_bins, y, side="right") - 1
    # binned_statistic_2d keeps the right edge inside the last bin.
    i[x == x_bins[-1]] = nx - 1
    j[y == y_bins[-1]] = ny - 1
    inside = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
    flat = (i * ny + j)[inside]
    zz = z[inside]
    order = np.lexsort((zz, flat))
    flat, zz = flat[order], zz[order]
    counts = np.bincount(flat, minlength=nx * ny)
    starts = np.cumsum(counts) - counts
    out = np.full(nx * ny, np.nan)
    filled = counts > 0
    rank = np.floor(q * (counts[filled] - 1)).astype(int)
    out[filled] = zz[starts[filled] + rank]
    return out.reshape(nx, ny)


def grid_height_map(points, p: Params, anchors=None):
    """``(height, validity)`` on the grid from levelled points (N, >=3), already cut to radius and clip;
    ``(None, None)`` with fewer than 4 measured cells."""
    lo, hi, res = -p.extents_m, p.extents_m, p.resolution_m
    x_bins = np.arange(lo, hi + res, res)
    y_bins = np.arange(lo, hi + res, res)
    try:
        height, x_edges, y_edges, _ = binned_statistic_2d(
            points[:, 0], points[:, 1], points[:, 2], statistic="median", bins=[x_bins, y_bins]
        )
        if STATISTICS[p.statistic] is not None:
            height = _binned_quantile(
                points[:, 0], points[:, 1], points[:, 2], x_edges, y_edges, STATISTICS[p.statistic]
            )
    except ValueError:
        return None, None

    mask = ~np.isnan(height)
    if np.sum(mask) < 4:  # too sparse for a map, not a flat floor
        return None, None

    xc = (x_edges[:-1] + x_edges[1:]) / 2
    yc = (y_edges[:-1] + y_edges[1:]) / 2
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    points_valid = np.stack([X[mask], Y[mask]], axis=-1)
    values_valid = height[mask]
    n_measured = len(values_valid)

    if anchors is not None and len(anchors):
        anchors = np.asarray(anchors, dtype=float).reshape(-1, 3)
        if p.anchor_relevel == "shift":
            r2 = points_valid[:, 0] ** 2 + points_valid[:, 1] ** 2
            near = (r2 <= p.anchor_ground_radius_m**2) & (r2 >= p.anchor_ground_inner_m**2)
            if near.sum() >= ANCHOR_MIN_GROUND_CELLS:
                # Shift the contact plane, do not flatten it: it carries the
                # robot's pitch and roll, the slope under it. The offset is the
                # median gap between the measured ground and the plane.
                A = np.column_stack([anchors[:, :2], np.ones(len(anchors))])
                coef, _, _, _ = np.linalg.lstsq(A, anchors[:, 2], rcond=None)
                plane = coef[0] * points_valid[near, 0] + coef[1] * points_valid[near, 1] + coef[2]
                anchors = anchors.copy()
                anchors[:, 2] += np.median(values_valid[near] - plane)
        points_valid = np.vstack([points_valid, anchors[:, :2]])
        values_valid = np.concatenate([values_valid, anchors[:, 2]])
    else:
        anchors = None

    anchor_radius = p.anchor_radius_m if p.anchor_radius_m is not None and p.anchor_radius_m > 0 else None
    if anchors is None and p.max_edge_m is None:
        height = griddata(_jittered(points_valid, p), values_valid, (X, Y), method="linear")
        from_anchor = np.zeros(height.shape, dtype=bool)
    else:
        # The same interpolant, with the triangulation kept to attribute cells.
        tri = Delaunay(_jittered(points_valid, p))
        height = LinearNDInterpolator(tri, values_valid)(X, Y)
        simplex = tri.find_simplex(np.stack([X.ravel(), Y.ravel()], axis=-1)).reshape(X.shape)
        # find_simplex gives -1 outside the hull; the appended False catches it.
        has_anchor = np.append((tri.simplices >= n_measured).any(axis=1), False)
        from_anchor = has_anchor[simplex]
        if anchors is not None and anchor_radius is not None:
            d_a, _ = cKDTree(anchors[:, :2]).query(np.stack([X.ravel(), Y.ravel()], axis=-1))
            from_anchor &= (d_a <= anchor_radius).reshape(X.shape)
        if p.max_edge_m is not None:
            verts = points_valid[tri.simplices]  # the unjittered points: edge lengths as measured
            longest = np.linalg.norm(verts - verts[:, [1, 2, 0]], axis=2).max(axis=1)
            bridged = np.append(longest > p.max_edge_m, False)[simplex] & ~mask
            height[bridged] = np.nan

    nan_mask = np.isnan(height)
    validity = np.full(height.shape, VALIDITY_INTERPOLATED, dtype=np.uint8)
    validity[from_anchor] = VALIDITY_ANCHORED
    validity[nan_mask] = VALIDITY_EXTRAPOLATED
    # MEASURED last and winning: anchors can move the hull so a measured edge
    # cell comes back NaN; its height is restored by the nearest pass below.
    validity[mask] = VALIDITY_MEASURED
    if nan_mask.any() and p.fill == "ground":
        # From the binned values, never from `height`: a measured hull-edge cell
        # can be NaN there, and one NaN in the median once wiped 1 % of a root.
        measured_xy = points_valid[:n_measured]
        near = measured_xy[:, 0] ** 2 + measured_xy[:, 1] ** 2 <= p.fill_radius_m**2
        if near.sum() >= ANCHOR_MIN_GROUND_CELLS:
            height[nan_mask & ~mask] = np.median(values_valid[:n_measured][near])
            nan_mask = np.isnan(height)
    if nan_mask.any():
        if anchors is None:
            height[nan_mask] = griddata(points_valid, values_valid, (X[nan_mask], Y[nan_mask]), method="nearest")
        else:
            # griddata's nearest query, explicit so the winning sample can be attributed.
            q = np.stack([X[nan_mask], Y[nan_mask]], axis=-1)
            _, idx = cKDTree(points_valid).query(q)
            won_by_anchor = idx >= n_measured
            if anchor_radius is not None and won_by_anchor.any():
                # Past the anchors' radius the nearest return fills, or anchors claim the whole rear.
                d_a, _ = cKDTree(anchors[:, :2]).query(q)
                too_far = won_by_anchor & (d_a > anchor_radius)
                if too_far.any():
                    _, ridx = cKDTree(points_valid[:n_measured]).query(q[too_far])
                    idx = idx.copy()
                    idx[too_far] = ridx
                    won_by_anchor &= ~too_far
            height[nan_mask] = values_valid[idx]
            anchored = np.zeros(height.shape, dtype=bool)
            anchored[nan_mask] = won_by_anchor
            validity[anchored & ~mask] = VALIDITY_ANCHORED

    # Smooth quantisation artefacts; the mask describes the data before smoothing.
    return gaussian_filter(height, sigma=0.5), validity


def build_frame(cloud: np.ndarray, R_align: np.ndarray, p: Params = Params()):
    """``(height_map, validity)`` for one cloud in base_link, or ``(None, None)`` when too sparse."""
    pts = cloud.astype(np.float64, copy=True)
    pts[:, :3] = pts[:, :3] @ R_align.T
    pts = pts[np.sum(pts[:, :2] ** 2, axis=1) > p.min_radius_m**2]
    pts = pts[pts[:, 2] < p.canopy_clip_m]
    if len(pts) < p.min_points:
        return None, None
    anchors = wheel_contact_anchors(p.wheel_centres_m, p.wheel_radius_m, R_align) if p.anchors else None
    return grid_height_map(pts, p, anchors)


def _write_npz(path: str, **arrays) -> None:
    # Not "*.npz": a crash mid-write must not leave a file the map listings count.
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, path)


def _one_frame(job):
    scene_dir, frame, overwrite, p = job
    hm_out = os.path.join(scene_dir, "height_maps", f"{frame:05d}.npz")
    mask_out = os.path.join(scene_dir, "height_masks", f"{frame:05d}.npz")
    if not overwrite and os.path.exists(hm_out) and os.path.exists(mask_out):
        return frame, "skip"
    try:
        cloud = np.load(os.path.join(scene_dir, "clouds", f"{frame:05d}.npy"))
        R_align = np.load(os.path.join(scene_dir, "gravity_align.npy"), mmap_mode="r")[frame]
        hm, val = build_frame(cloud, np.asarray(R_align), p)
        if hm is None:
            return frame, "sparse"
        _write_npz(hm_out, height_map=hm)
        _write_npz(mask_out, validity=val)
        return frame, "ok"
    except Exception as e:  # noqa: BLE001
        return frame, f"ERR {frame}: {type(e).__name__}: {e}"


def scene_params(scene_dir: Path) -> Params | None:
    """The settings a scene's maps were built with, from its ``height_maps.json``; None when it has none."""
    f = Path(scene_dir) / SETTINGS_FILE
    if not f.exists():
        return None
    return Params.from_dict(json.loads(f.read_text())["parameters"])


def build_scene(scene_dir: Path, p: Params = Params(), workers: int = 8, overwrite: bool = False) -> dict:
    """Build every missing map of a scene (all with ``overwrite``); returns the counts and records the settings."""
    scene_dir = Path(scene_dir)
    if not overwrite and (old := scene_params(scene_dir)) is not None and old != p:
        raise SystemExit(
            f"{scene_dir}: its maps were built with other settings ({SETTINGS_FILE}); "
            "--overwrite rebuilds all of them, mixing the two is refused"
        )
    for name in ("height_maps", "height_masks", SETTINGS_FILE):
        if (scene_dir / name).is_symlink():  # a derived root links its parent's: don't write through
            raise SystemExit(
                f"{scene_dir / name} -> {os.readlink(scene_dir / name)}: a symlink; build in the root that owns the files"
            )
    for sub in ("height_maps", "height_masks"):
        (scene_dir / sub).mkdir(exist_ok=True)
    frames = sorted(int(f[:-4]) for f in os.listdir(scene_dir / "clouds") if f.endswith(".npy"))
    counts = {"ok": 0, "skip": 0, "sparse": 0}
    errors = []
    jobs = [(str(scene_dir), f, overwrite, p) for f in frames]
    if workers > 1:
        with mp.Pool(workers) as pool:
            results = list(pool.imap_unordered(_one_frame, jobs, chunksize=32))
    else:
        results = [_one_frame(j) for j in jobs]
    for _, r in results:
        if r in counts:
            counts[r] += 1
        else:
            errors.append(r)
    if overwrite:
        # A frame that is too sparse now may have had a map from an earlier build.
        have = {int(f[:5]) for f in os.listdir(scene_dir / "height_maps") if f.endswith(".npz")}
        built = {f for f, r in results if r == "ok"}
        for stale in sorted(have - built):
            for sub in ("height_maps", "height_masks"):
                (scene_dir / sub / f"{stale:05d}.npz").unlink(missing_ok=True)
    n_maps = sum(1 for f in os.listdir(scene_dir / "height_maps") if f.endswith(".npz"))
    record = {
        "generator": "dataset_tools.heightmap",
        "built_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "parameters": p.to_dict(),
        "frames": len(frames),
        "maps": n_maps,
        "too_sparse": len(frames) - n_maps,
        "errors": len(errors),
    }
    _replace_atomically(scene_dir / SETTINGS_FILE, lambda fh: fh.write(json.dumps(record, indent=1) + "\n"))
    return {**counts, "errors": errors, "maps": n_maps, "frames": len(frames)}


def verify_against(ref_root: Path, p: Params, n_frames: int = 12, seed: int = 0) -> tuple[list[str], int]:
    """Rebuild a sample of an existing root's maps and compare (float rounding allowed).

    Returns ``(failures, frames checked)``. A scene with a ``height_maps.json``
    is rebuilt at its own settings, one without at ``p``.
    """
    rng = np.random.default_rng(seed)
    failures, checked = [], 0
    ref_root = Path(ref_root)
    scenes = [d for d in sorted(os.listdir(ref_root)) if d.isdigit() and (ref_root / d / "height_maps").is_dir()]
    for scene in rng.permutation(scenes):
        d = ref_root / scene
        files = sorted(f for f in os.listdir(d / "height_maps") if f.endswith(".npz"))
        if not files:
            continue
        sp = scene_params(d) or p
        ga = np.load(d / "gravity_align.npy", mmap_mode="r")
        for f in rng.choice(files, size=min(2, len(files)), replace=False):
            frame = int(f[:5])
            hm, val = build_frame(np.load(d / "clouds" / f"{frame:05d}.npy"), np.asarray(ga[frame]), sp)
            checked += 1
            ref_hm = np.load(d / "height_maps" / f)["height_map"]
            ref_val = np.load(d / "height_masks" / f)["validity"]
            if hm is None:
                failures.append(f"{scene}/{frame}: too sparse to rebuild")
            elif not np.allclose(hm, ref_hm, atol=1e-9, rtol=0, equal_nan=True):
                failures.append(f"{scene}/{frame}: max diff {np.nanmax(np.abs(hm - ref_hm)):.4g} m")
            elif not np.array_equal(val, ref_val):
                failures.append(f"{scene}/{frame}: {int(np.sum(val != ref_val))} mask cells differ")
            if checked >= n_frames:
                return failures, checked
    return failures, checked


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--scenes", default="", help="comma-separated; default every scene with clouds")
    ap.add_argument("--workers", type=int, default=usable_cpus(), help="default: the CPUs this process may use")
    ap.add_argument(
        "--overwrite", action="store_true", help="rebuild every map, dropping maps of frames now too sparse"
    )
    ap.add_argument(
        "--verify_against",
        type=Path,
        default=None,
        help="a root with maps: a sample is rebuilt first and must match, else nothing is written",
    )
    ap.add_argument("--verify_frames", type=int, default=12)
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="change one setting of Params (e.g. --set statistic=median, --set tie_jitter=v1 or a JSON table spec); recorded in height_maps.json",
    )
    a = ap.parse_args(argv)

    kw = {}
    defaults = Params().to_dict()
    for s in a.set:
        k, _, v = s.partition("=")
        if k not in defaults or k == "wheel_centres_m":
            ap.error(f"--set {s}: no such setting")
        d = defaults[k]
        if v.lower() == "none":
            kw[k] = None
        elif k == "tie_jitter":
            kw[k] = TIE_JITTER_V1 if v == "v1" else json.loads(v)
        elif isinstance(d, bool):
            kw[k] = v.lower() in ("1", "true", "yes", "on")
        else:
            kw[k] = type(d)(v)
    p = Params(**kw)

    if a.verify_against is not None:
        failures, checked = verify_against(a.verify_against, p, a.verify_frames)
        if not checked:
            print(f"KNOWN-ANSWER: no maps found in {a.verify_against}")
            return 1
        if failures:
            print(f"KNOWN-ANSWER FAILED on {len(failures)}/{checked} frames of {a.verify_against}:")
            for f in failures[:5]:
                print(f"  {f}")
            print("Refusing to write: every map would be wrong the same way.")
            return 1
        print(f"KNOWN-ANSWER: {checked}/{checked} frames of {a.verify_against} reproduce")

    root = a.root
    scenes = a.scenes.split(",") if a.scenes else sorted(d for d in os.listdir(root) if d.isdigit())
    print(f"BUILD {root}: {json.dumps(p.to_dict())}", flush=True)
    bad = 0
    for scene in scenes:
        d = root / scene
        if not (d / "clouds").is_dir():
            print(f"{scene}: no clouds, skipped")
            continue
        c = build_scene(d, p, a.workers, a.overwrite)
        print(
            f"{scene}: {c['ok']} built, {c['skip']} kept, {c['sparse']} too sparse, "
            f"{len(c['errors'])} errors; {c['maps']} maps for {c['frames']} frames",
            flush=True,
        )
        for e in c["errors"][:5]:
            print(f"  {e}")
        bad += bool(c["errors"])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
