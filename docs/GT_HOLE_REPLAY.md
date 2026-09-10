# Replaying ground-truth holes

Status: **run (v1.3 campaign 2026-09-09, merged as v1.4 / v1.5)**. This is the
record of that campaign and the home of the tools it used, so the next one --
almost certainly on the v2 rig -- does not start from nothing.

A *hole* is a stretch of frame time no odometry segment covers. The frames, the
clouds, the wheel encoders and `/odom` are all there; only the ground truth is
missing. On HELHEST v1.3, holes were **30 % of frame time** (coverage 0.698,
8,138 s of gaps over 1 s), and the worst scenes were almost unusable: 0001 at
0.19, 0021 at 0.17, 0005 at 0.32.

## The finding: two kinds of hole, only one worth replaying

`find_gt_holes.py` splits them by what FAST-LIO was doing:

| class | what happened | can a re-run fix it |
|---|---|---|
| `silent_stall` | FAST-LIO published **nothing** for over 20 s while frames, `/odom` and both Livox streams kept coming at full rate | **yes** |
| `restart_storm_or_gate` | short silences only -- the estimator was alive and either restarting or refusing to track (degenerate ground) | no; it refuses again |

On v1.3 that was 33 silent stalls totalling 5,308 s and 3,066 m of driving --
the biggest single one being 698 s in scene 0002. Bag metadata on subtdata shows
`/livox/lidar_front` at 9.8-10.0 Hz and `/livox/imu_front` at 193-199 Hz across
the whole of every affected bag, and the recorder writes every `/Odometry` it
receives with no drop path. So nothing was wrong with the *data*.

**Replaying at an offset reproduces no silence at all.** Scene 0021, silent from
t+194 in the original run, comes back gapless to t+325. The stalls were a
run-time condition -- eight scenes extracted in parallel, wall/scene ratio 1.5 to
4.6 at rate 1.0 -- not a property of the recording. That is the whole reason this
pipeline works.

## The pipeline

```
find_gt_holes.py <root> --csv holes.csv     # inventory + classify
  → hand-build the campaign CSV (scene, bag, first_frame_ts, hole_t0, hole_dur)
fetch_hole_bags.sh campaign.csv             # one bag at a time from subtdata
replay_holes.sh  campaign.csv               # FAST-LIO per hole, in the sif
  → run_scene_offset.sh inside the container does the actual replay
score_hole_replay.py <scene> <replay_dir> <t0> <t1>   # accept or reject, per hole
build_fill_root.py <src_root> <dst_root> <v11_root> --regen scene:dir:t0:t1 ...
```

Every appended run passes the **same** acceptance the root's own segments passed:
`filter_segments.bad_poses` (2 s windows against recorded `/odom`, drop over 1 m),
`gt_gaps` seams, `gt_map.window_offset` on measured cells (over 0.3 m across 4 s
is cut, 0.5 s margin), at least 20 poses, re-anchored at the identity. Provenance
is per segment: `regen_fill:t<t0>`, `v11_fill`, `<reason>+coverage_guard`.

The two drivers are deliberately a pair: one bag is on disk at a time. The v1.3
campaign moved 571 GB through a 14 GB working set on dgx07.

## What the v1.3 campaign yielded

31 holes in 14 scenes replayed from 13 bags -- the 29 in the campaign CSV plus
the two pilots that preceded it (0018 t+233, 0021 t+194); **2,768 s recovered of
5,253 s replayed (53 %)**. Frame-time coverage 0.698 → 0.846 (0020 0.49 → 0.95, 0021
0.17 → 0.60, 0002 0.65 → 0.89), segments 660 → 1,058, DRIFT train windows
88,973 → 103,283. Degenerate-ground holes yielded almost nothing, as predicted by
their class. The result was refined and merged as HELHEST v1.4, and v1.5 joined
its standstill seams; neither v1.2 nor v1.3 was modified.

Holes that stayed unrecovered: the restart-storm gaps (~1.5 ks) and 0030 t+983,
where FAST-LIO never initialised at that offset.

## Traps

* **The container's `run_scene.sh` must honour `START_OFFSET` / `PLAY_DURATION`.**
  That is `scripts/run_scene_offset.sh` here; the edit lives on the recorder
  repo's `odometry-regen-recorder` branch and was never committed. An older copy
  ignores both variables and silently replays the entire bag.
* **The campaign CSV's third column is the scene's first FRAME timestamp**, not
  the bag start. `replay_holes.sh` reads the bag start out of `metadata.yaml` and
  converts; get this wrong and every replay lands in the wrong place.
* **Never scan for frozen runs across a segment seam.** Segments are re-anchored
  at the identity, so a scan over a whole `odometry.txt` reports the anchor
  change as a fake multi-minute freeze (0020 seg 16 and 0021 seg 5 both do).
  `survey_gt_coverage.py` scans inside segments for exactly this reason.
* A replay overlapping ground truth the root already has is not automatically
  wrong, but it must be checked: `score_hole_replay.py` prints how many root
  poses fall inside the replay's span before the hole.

## Files

| file | was | what it is |
|---|---|---|
| `scripts/find_gt_holes.py` | `holes.py` | inventory + classify a root's holes; writes `docs/v13_gt_holes.csv` |
| `scripts/survey_gt_coverage.py` | `survey.py` | per-scene coverage, gaps with wheel distance, frozen runs |
| `scripts/fetch_hole_bags.sh` | `fetch_all.sh` | serial bag fetch from subtdata |
| `scripts/replay_holes.sh` | `replay_all.sh` | per-hole FAST-LIO replay driver |
| `scripts/run_scene_offset.sh` | (recorder repo) | the per-scene replay itself, with the offset window |
| `scripts/score_hole_replay.py` | `score_pilot.py` | accept/reject one replay against the gate |
| `scripts/build_fill_root.py` | -- | build the filled root as a symlink tree |
| `docs/gt_hole_campaign.csv` | `campaign_holes.csv` | the 29 holes the campaign drove, 13 scenes / 13 bags |
| `docs/v13_gt_holes.csv` | -- | the full v1.3 hole inventory, 129 holes |

Until 2026-09-10 the only copy of these lived in `HELHEST_v1_3_fill/tools/`
inside the dataset root they built.
