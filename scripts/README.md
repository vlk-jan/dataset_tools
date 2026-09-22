# Scripts

Command-line tools that work on a dataset root or a bag but are not part of the
installed package. They import `dataset_tools` from the checkout (each puts the
repository on `sys.path`), so run them from any directory with the repository's
virtualenv:

```bash
.venv/bin/python scripts/make_atlas.py /data/HELHEST_v2_beta --check
```

## Tools for any root

| Script | What it does |
| --- | --- |
| `make_atlas.py` | Generate `<root>/atlas/atlas.json` and its thumbnails from the root itself, merging the hand-written prose in `atlas/curation.json`; `--check` reports a stale atlas without writing. The viewer's index page renders it. |
| `measure_time_offset.py` | Audit an extracted scene's frame clock against its poses: was `--time_align` applied, is the scene still off, and by how much over its course. |
| `probe_clock_skew.py` | Every publisher's clock against the recorder's, straight out of a bag (`recv - header` per message), to see *when* a sensor's clock moved. |
| `survey_gt_coverage.py` | How much of a root's frame time has ground truth, every gap over a threshold with the wheel distance driven through it, and every frozen run inside a segment. |
| `find_gt_holes.py` | Inventory the ground-truth holes in a root and class each by what the estimator was doing -- the input to a replay campaign. |
| `split_segments_at_gaps.py` | Cut a root's segments at the holes `gt_gaps` finds and re-anchor each side, in place, keeping the input beside it. |
| `cut_segments_at_map_mismatch.py` | Cut a root's segments where the ground-truth path leaves the lidar's ground (`gt_map`), the same way. |
| `relabel_segments.py` | Retag segments whose `reason` column was inherited from the input segment rather than measured. |

## The v1 ground-truth regeneration (2026-09)

The odometry of the first-generation HELHEST scenes was re-run from the raw
bags through FAST-LIO and post-filtered; the holes the estimator left were then
replayed one at a time. These scripts are that campaign's tooling, kept so the
next one does not start from nothing. They assume the layout the campaign used
(`<scene>/regen/` provenance, the recorder branch, the RCI cluster); the record
of what was done and why is [`docs/ODOMETRY_REGEN.md`](../docs/ODOMETRY_REGEN.md)
and [`docs/GT_HOLE_REPLAY.md`](../docs/GT_HOLE_REPLAY.md).

| Script | Stage |
| --- | --- |
| `dgx07_regen.sh` | fan-out driver for the whole regeneration on one big machine (`slurm/regen_scene.batch` is its cluster twin) |
| `run_scene_offset.sh` | replay one bag (or one window of it) through FAST-LIO with the segment-restarting recorder; runs inside the container |
| `filter_segments.py` | post-hoc filter of a regenerated `odometry.txt` against the live `/odom`, the acceptance every later step reuses |
| `fetch_hole_bags.sh`, `replay_holes.sh` | the hole campaign: fetch one bag at a time, replay each hole with lead-in and run-out, delete the bag |
| `score_hole_replay.py` | score one replayed hole with the same acceptance before it reaches a root |
| `merge_v11_fill.py` | fill v1.2's gaps with quality-gated v1.1 odometry, tagged `v11_fill` |
| `build_fill_root.py` | build the v1.3 fill root: a symlink tree carrying the coverage guard, the hole replays and the v1.1 fill, each tagged in the segment table |
| `compare_v1_v12.py` | the paired comparison of the published v1 root against the regenerated v1.2 (`docs/regen_comparison.csv`) |

## Cluster helpers

| Script | What it does |
| --- | --- |
| `viewer_connect.sh` | open an ssh tunnel from a laptop to a viewer started by `slurm/viewer.batch` on an RCI compute node |
