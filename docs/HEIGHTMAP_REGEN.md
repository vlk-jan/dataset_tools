# Rebuilding the height maps without the rear-wheel wall

*Written 2026-09-07. Companion to `ODOMETRY_REGEN.md`, which covers the v1.2/v1.3
odometry rebuild. Nothing here touches odometry, clouds or images.*

## The defect

`height_maps/` carries an invented step under the robot's rear wheel. It is not
noise and it is not marked as invention.

The lidar is forward-only: **zero returns at x < −0.25 m in `base_link`** over
126 frames sampled from 14 scenes, and DRIFT's generator drops everything within
`min_radius = 0.5` of the origin on top of that. The sample set therefore has a
metres-wide hole under the robot, and the generator
(`DRIFT/utils/pcd.py::get_height_map`) fills every cell regardless:

* the linear pass triangulates only the cells that hold returns, so its
  Delaunay triangles **bridge the hole** — a cell behind the robot is a
  barycentric blend of two rim cells metres apart, one of them typically on
  vegetation. Those cells are inside the hull, so the mask calls them
  `interpolated` and a validity gate does not catch them;
* the nearest-neighbour pass floods the rest from one arbitrary rim cell.

The seam between the two lands at x ≈ −0.35…−0.55 m. The rear wheel is centred
at x = −0.697 with r = 0.36, so its footprint is x ∈ [−1.06, −0.34]: the seam is
under its leading edge.

Over 182 frames spanning all 43 scenes, peak rise in x ∈ [−1.6, 0], |y| < 1
against the local measured ground:

| | median | p90 | max |
|---|---|---|---|
| peak rise behind the robot | **+0.19 m** | +0.42 | +0.64 |
| rear-wheel footprint mean | +0.03 m | +0.22 | +0.35 |
| front-wheel footprint mean | −0.01 m | +0.09 | +0.24 |

70 % of frames carry more than 10 cm, 41 % more than 25 cm. This is the
"digging force" `DRIFT/scripts/analysis/check_start_seating.py` describes from
the other end: the simulator seats the robot on the invented ridge, and a
world-frame trajectory loss can only cancel the offset by excavating the terrain
under the robot.

## The repair

Two flags on `DRIFT/utils/pcd.py::get_height_map`, both off by default:

* `anchors` — pseudo-observations of the ground under the robot's own wheels
  (`wheel_contact_anchors`). `base_link` **is** the front axle and the wheels are
  r = 0.36 at (0, ±0.36) and (−0.697, 0), so the surface under them is known
  without measuring it. The measured ground beside the robot reads
  −0.335…−0.353 m against the kinematic −0.36, so the constant is right to a
  couple of centimetres; `anchor_ground_radius` re-levels onto the nearby
  measured ground to remove even that.
* `max_edge` — drop the linear value of any cell whose containing triangle has
  an edge longer than this. Exactly the triangles that span the hole.
* `anchor_radius` — how far from a wheel contact an anchor is allowed to speak.
  See "The anchors' authority" below; **this one is not optional.**

Measured on 58 frames from all 43 scenes:

| setting | wall p50 | wall p90 | footprint bias | grid reclassified |
|---|---|---|---|---|
| published | +0.211 | +0.455 | +0.060 | — |
| `max_edge=0.5` alone | +0.126 | +0.372 | −0.026 | 29.5 % |
| `anchors` alone | +0.091 | +0.283 | −0.010 | 0.0 % |
| `anchors --max_edge 0.5` | +0.020 | +0.238 | −0.010 | 29.5 % |
| **`anchors --max_edge 1.0`** | **+0.024** | **+0.208** | **−0.010** | **19.3 %** |
| `anchors --max_edge 1.5` | +0.054 | +0.221 | −0.010 | 13.4 % |

"reclassified" is the share of the grid moving `interpolated` → `extrapolated`.
Most of it is far field: past ~4 m the lidar's own sample spacing exceeds a few
decimetres, so a low `max_edge` fires on genuine (if wide) interpolation. **1.0
is the recommended setting** — the whole near-field gain for two thirds of the
reclassification.

`max_edge` alone is not a uniform improvement: it swaps a smooth bad blend for a
flood from one rim cell, and the worst frame in the sample got slightly worse
(+0.668 vs +0.643) even as the median improved. Use it with `anchors`.

## The anchors' authority (found 2026-09-07, in the viewer)

The first repaired build shipped the anchors unbounded, and it was wrong in a
way the BEV tile showed immediately: **a cone spreading astern of the robot,
widening to the map edge, labelled `anchored` — a boat's wake.**

The anchors are the rearmost samples in the interpolation set. The
nearest-neighbour pass gives every unfilled cell the value of its nearest
sample, and behind the robot there are no returns at all, so *every* cell
astern was nearest to a wheel contact. The anchors' Voronoi cell runs backwards
forever.

Under the wheels the label is earned — the robot is standing there, so the
ground is at the contact plane. Three metres astern it asserts a prior that
does not exist. Those cells are unknown and must say `extrapolated`.

`anchor_radius` bounds it. Beyond the radius an anchor neither wins the fill
(the nearest *return* does) nor labels the cell. Measured over 38 frames, with
the share of the grid still `anchored` astern of x = −1.6 m:

| `anchor_radius` | wall p50 | wall p90 | anchored | anchored astern of −1.6 m |
|---|---|---|---|---|
| unbounded | +0.042 | +0.237 | 15.7 % | **12.83 %** |
| **1.0 (default)** | **+0.042** | **+0.237** | **3.2 %** | **0.51 %** |
| 0.7 | +0.118 | +0.346 | 2.6 % | 0.13 % |
| 0.5 | +0.125 | +0.350 | 2.1 % | 0.01 % |

1.0 m is the union of discs around the three contact patches — the robot's own
footprint plus ~0.6 m of margin, i.e. ground continuous with a wheel bearing
load on it. It keeps the entire wall fix; below ~1.0 the near field goes back
to the nearest-return flood and the wall returns.

The 0.51 % residual is not wake. The rear contact is at x = −0.697 and the
anchor ring adds 0.45, so the authority disc legitimately reaches −2.15;
DRIFT's `tests/test_height_map_near_field.py` asserts no anchored cell exists past that.

**Do not ship an unbounded build.** If a BEV validity tile shows anything
`anchored` that is not sitting on the robot's own footprint, the radius is
wrong or absent.

## The new validity state

Repaired roots write a fourth state into `height_masks/validity`:

| code | meaning |
|---|---|
| 0 | `extrapolated` — outside the hull, nearest-neighbour invention |
| 1 | `interpolated` — no return in the cell, but inside the hull |
| 2 | `measured` — a lidar return landed here |
| **3** | **`anchored` — a wheel-contact anchor decided this cell** |

**3 is a provenance, not a better 2.** Rank codes through `VALIDITY_RANK`
(`DRIFT/utils/pcd.py`, mirrored in `dataset_tools/viewer/loaders.py`) or test
them with DRIFT's `founded_in_returns`; `validity >= 1` now silently *accepts* the
anchors, which is the opposite of what every gate written with that idiom meant.
Both helpers are the identity on the three states every published root ships.

Roughly 15–19 % of a repaired grid is anchored.

## Runbook

Everything below assumes a root that already has correct clouds, odometry and
`gravity_align.npy` — i.e. a v1.3 root, copied or hard-linked to a new version
directory. Only `height_maps/`, `height_masks/` and what derives from them
change.

The `scripts/…` commands below are DRIFT's, run from a DRIFT checkout. (Since
2026-10-12, `6bc7ff1`, this repository builds the maps itself, near-field repair
included: `helhest-heightmap`, or `slurm/heightmaps.batch` for a root already on
RCI; see `slurm/README.md`.)

1. **Build the maps and masks.** The known-answer gate runs first, with the
   repair *off*, against a root whose maps already exist; it must reproduce them
   bit-for-bit before anything is written.

   ```
   python3 scripts/data/build_heightmaps.py \
       --root <NEW_ROOT> --verify_against <V1_3_ROOT> \
       --clip 1.0 --verify_clip 1.25 \
       --anchors --max_edge 1.0 --anchor_radius 1.0 \
       --overwrite --workers 16
   ```

   Do **not** use `scripts/data/prep_dataset.py` or
   `scripts/data/apply_cloud_xfix.py`: neither supports the repair, and
   `prep_dataset.py` writes no validity masks at all — a repaired map without a
   mask is indistinguishable downstream from a measured one.

2. **Rebuild the keyframes** (they index map/mask presence):

   ```
   python3 scripts/data/write_keyframes.py --root <NEW_ROOT>
   ```

3. **Rebuild the proprioception labels.** They gate on `height_masks`, so labels
   built against the old maps score a different population of cells. Mixing the
   two is worse than either alone.

   ```
   python3 scripts/generate_proprio_labels.py \
       --rootdir <NEW_ROOT> --out_root <NEW_ROOT> --overwrite
   ```

   `--min_validity` now ranks state 3 with the extrapolation, so the default
   (`2`, measured only) is unaffected by the repair.

   **`--overwrite` rewrites, it does not prune.** A frame that produced a label
   under the old maps but does not now (a window clamped at a segment seam, say)
   keeps its stale file, and the root ends up with a mixed label set — the exact
   thing this step exists to prevent. Delete whatever the run did not touch,
   never by a hardcoded date:

   ```
   MARK=$(mktemp); touch $MARK; sleep 1
   python3 scripts/generate_proprio_labels.py --rootdir <NEW_ROOT> --overwrite
   find <NEW_ROOT> -path '*/proprio_labels/*.npz' ! -newer $MARK -delete
   ```

   In that order the root is never without labels. On the 2026-09-07 rebuild
   this pruned 1,170 files left over from 2026-09-04.

4. **Record it in `manifest.json`** — add `height_maps_near_field_repair` to
   `corrections`, and under `height_maps`:

   ```json
   "near_field_repair": {
     "anchors": "wheel_contact_anchors, base_link front axle, r 0.36,
                 centres (0,+-0.36) and (-0.697,0), ring 0.45 x 8",
     "max_edge_m": 1.0,
     "validity_states": "0 extrapolated, 1 interpolated, 2 measured, 3 anchored",
     "known_answer": "generator reproduces the v1.3 maps bit-exactly with the
                      repair off before building with it on"
   }
   ```

5. **Clear the viewer cache** for the root (`~/.cache/dataset_tools_viewer/`, or
   start the viewer once with `--clear-cache`). Cached bundles hold a splatted
   terrain and a validity layer built from the old maps.

## Verify the build before trusting it

An interrupted rebuild leaves a root that looks finished. Frames complete out
of order inside a scene, so a scene whose *newest* file is fresh can still hold
hundreds of frames from the previous build — that is exactly what happened on
2026-09-07, and a per-scene mtime check missed it. Check per FILE, and check
the invariant, not just the aggregate:

```bash
R=<ROOT>; SINCE='<the time the build started>'

# 1. No file predates the build. Zero, or the root is mixed.
find $R -path '*/height_maps/*.npz'  ! -newermt "$SINCE" | wc -l
find $R -path '*/height_masks/*.npz' ! -newermt "$SINCE" | wc -l

# 2. Counts match the clouds, minus the "too sparse" frames the log reports.
find $R -path '*/height_maps/*.npz' | wc -l
```

Then the wake invariant, which is what actually catches a bad or stale build —
**every ANCHORED cell must lie within reach of a wheel**, i.e. at most
`anchor_radius + ring` (1.45 m) from a contact, plus a little slack because
`gravity_align` rotates the anchors (measured max 1.58 m over 258 frames of all
43 scenes; nothing above 1.6 m):

```python
CONTACTS = np.array([[0, 0.36], [0, -0.36], [-0.697, 0.0]])
rows, cols = np.where(validity == VALIDITY_ANCHORED)
pts = np.stack([axis[rows], axis[cols]], -1)
d = np.linalg.norm(pts[:, None, :] - CONTACTS[None], axis=2).min(axis=1)
assert d.max() < 1.6, "anchored cells outside the wheels' reach -- wake, or a stale file"
```

A frame that fails this is either built unbounded or left over from an earlier
run. Distinguish them by rebuilding that one frame and comparing: if the rebuild
is clean, the file on disk is stale.

## What does not need changing

* **Clouds, odometry, images, `gravity_align.npy`** — untouched; the repair is
  entirely inside the map generator.
* **`META_VERSION` in `dataset_tools/viewer/build.py`** — the viewer now ranks
  validity codes before averaging them into the terrain layer, and that ranking
  is the identity on 0/1/2, so every bundle cached from an existing root still
  means exactly what its legend says.
* **Checkpoints trained on v1.3** — still valid against v1.3. They were trained
  on the walled maps, so a repaired root is a different input distribution;
  compare across roots only with that stated.
