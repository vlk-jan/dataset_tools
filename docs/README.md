# Documentation

Two kinds of document live here. The first two describe things a user of the
tools or the dataset needs; the rest are the records of the campaigns that
produced the HELHEST releases -- what was decided, measured and fixed, dated,
and site-specific in places (the CTU RCI cluster, the group's storage, one
laptop). They are kept because the next campaign starts from them.

## For users

| Document | What it covers |
| --- | --- |
| [`DATA.md`](DATA.md) | The HELHEST dataset itself: platform and sensors, the scene layout and every file's format, sessions and sites, the per-scene table, known issues, the second sensor generation. |
| [`VIEWER.md`](VIEWER.md) | The dataset viewer: what each file turns into on the page, windows and segments, the map, reviewer notes, the atlas, the train/validation split display, and the command line. |

## Campaign records

| Document | Campaign |
| --- | --- |
| [`ODOMETRY_REGEN.md`](ODOMETRY_REGEN.md) | Regenerating the v1 ground-truth odometry from the raw bags with FAST-LIO (v1.2, 2026-09-02), and the post-filter that became v1.3. |
| [`GT_HOLE_REPLAY.md`](GT_HOLE_REPLAY.md) | Replaying the stretches where the estimator published nothing, one hole at a time (v1.4 / v1.5, 2026-09-09). |
| [`HEIGHTMAP_REGEN.md`](HEIGHTMAP_REGEN.md) | Rebuilding the height maps without the invented step under the rear wheel (2026-09-07). |
| [`NEW_RIG_READINESS.md`](NEW_RIG_READINESS.md) | What the second-generation rig (Odin1 + Fixposition) records, measured against the v1.3 scene contract before the first v2 extraction (2026-09-05). |
| [`V2_0_SELECTION.md`](V2_0_SELECTION.md) | The survey of every new-rig bag, which went into HELHEST v2.0 and why, and the v2.1 / v2.2 builds, the JPEG recovery and the pinned Odin1 extrinsics that followed (2026-09-25 to 09-30). |

## Tables

| File | What it is |
| --- | --- |
| `scene_bags.csv`, `v2_bags.csv`, `v2_<n>_bags.csv` | scene number to raw bag: the v1 list, the v2 beta list, and one cumulative list per v2 release; the slurm array jobs index into them and `scripts/make_atlas.py --bags` reads bag sizes and sessions from them |
| `v2_inventory.csv` | every new-rig bag on subtdata as `scripts/survey_v2_bag.py` (removed after the selection; commit 94955ed) saw it; the rows the v2 lists were chosen from (`V2_0_SELECTION.md`) |
| `odin_tf_static_2026-09-30.txt` | what 42 bags published for the Odin1's self-estimated extrinsics; the survey behind `dataset_tools/tf_pins.py` |
| `validation_v1.csv` | `helhest-validate` over the published v1 root |
| `regen_comparison.csv` | the paired v1 vs v1.2 comparison from `scripts/compare_v1_v12.py` (removed; see git history) |
| `v13_gt_holes.csv`, `gt_hole_campaign.csv` | the v1.3 hole inventory from `scripts/find_gt_holes.py` (removed; see git history), and the subset that was replayed |

`img/` holds the figures the documents reference.
