# Scripts

Command-line tools that work on a dataset root or a bag but are not part of the
installed package. They import the installed `dataset_tools`, so run them with
the virtualenv the checkout is installed into (`pip install -e .`):

```bash
.venv/bin/python scripts/make_atlas.py /data/HELHEST_v2_beta --bags docs/v2_bags.csv --check
```

## Tools for any root

| Script | What it does |
| --- | --- |
| `make_atlas.py` | Generate `<root>/atlas/atlas.json` and its thumbnails from the root itself, merging the hand-written prose in `atlas/curation.json`; `--check` reports a stale atlas without writing. `--bags` names the bag list the sessions and bag sizes come from (`docs/scene_bags.csv` for v1, `docs/v2_<n>_bags.csv` for v2, as `slurm/finalize_v2.sh` does); without it those two fields are left out. There is no default because the v1 and v2 lists number their scenes the same way. The viewer's index page renders it. |
| `measure_time_offset.py` | Audit an extracted scene's frame clock against its poses: was `--time_align` applied, is the scene still off, and by how much over its course. |
| `probe_clock_skew.py` | Every publisher's clock against the recorder's, straight out of a bag (`recv - header` per message), to see *when* a sensor's clock moved. |
| `survey_v2_bag.py` | Survey one v2-rig bag without extracting it: `summary.json` (topics, `/tf_static`, how the Odin1 behaved, motion, GNSS, clouds), a 1 Hz `track.npy` and a few thumbnails, ~1 MB per bag to choose scenes from. Run per bag by `slurm/survey_v2.batch`. |
| `survey_gt_coverage.py` | How much of a root's frame time has ground truth, every gap over a threshold with the wheel distance driven through it, and every frozen run inside a segment. |
| `split_segments_at_gaps.py` | Cut a root's segments at the holes `gt_gaps` finds and re-anchor each side, in place, keeping the input beside it. |
| `cut_segments_at_map_mismatch.py` | Cut a root's segments where the ground-truth path leaves the lidar's ground (`gt_map`), the same way. |

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
| `filter_segments.py` | post-hoc filter of a regenerated `odometry.txt` against the live `/odom`, the acceptance every later step reuses |

The drivers (`dgx07_regen.sh`, `slurm/regen_scene.batch`), the hole-replay
scripts and their inventory (`find_gt_holes.py`, whose output is
`docs/v13_gt_holes.csv`), the v1.1 fill, the fill-root builder, the one-off
segment relabel (`relabel_segments.py`) and the v1 vs v1.2 comparison were
removed once the campaign finished; see git history.

## Cluster helpers

| Script | What it does |
| --- | --- |
| `viewer_connect.sh` | open an ssh tunnel from a laptop to a viewer started by `slurm/viewer.batch` on an RCI compute node |
