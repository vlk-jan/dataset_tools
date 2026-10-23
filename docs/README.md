# Documentation

The documentation site, <https://vlk-jan.github.io/dataset_tools/>, is built
from six pages here (`mkdocs.yml`): `index.md`, `installation.md`, `tools.md`,
`building.md`, `dataset.md` and `viewer.md`. The rest of this directory is
reference material the tools and the release specs use, kept in the
repository but not on the site.

## Documents

| Document | What it covers |
| --- | --- |
| [`VIEWER.md`](VIEWER.md) | The viewer in depth: what each file turns into on the page, windows and segments, the map, reviewer notes, the atlas, the split display, performance, routes, troubleshooting. |
| [`EXTRACTION_GUARDS.md`](EXTRACTION_GUARDS.md) | How `helhest-extract` decides the publish latency (`--time_align`), reset GNSS/INS solutions and frames with no pose under them (`--gnss_guard`). |

## Tables

| File | What it is |
| --- | --- |
| `v2_4_bags.csv` | scene number to raw bag for the current release (cumulative); the Slurm array jobs index into it and `scripts/make_atlas.py --bags` reads bag sizes and sessions from it |
| `releases/<release>.json` | a release's spec: what only a person can say about it (bag list, held-out scenes and why, sessions, defects, open items, atlas prose); `releases/helhest_v2.json` is the prose every release inherits. `helhest-release` turns a spec and the scenes into the root's `manifest.json`, `splits.json`, keyframe lists and `README.md` |
| `odin_tf_static_2026-09-30.txt` | what 42 bags published for the Odin1's self-estimated extrinsics; the survey behind `dataset_tools/tf_pins.py` |
