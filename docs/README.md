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
| [`V2_0_SELECTION.md`](V2_0_SELECTION.md) | Record of the survey of every Odin1 + Fixposition bag, which bags went into HELHEST v2.0 and why, and the v2.1 / v2.2 builds (2026-09-25 to 09-30). |

## Tables

| File | What it is |
| --- | --- |
| `v2_<n>_bags.csv` | scene number to raw bag, one cumulative list per release; the Slurm array jobs index into them and `scripts/make_atlas.py --bags` reads bag sizes and sessions from them. `v2_4_bags.csv` is the current release's |
| `releases/<release>.json` | a release's spec: what only a person can say about it (bag list, held-out scenes and why, sessions, defects, open items, atlas prose); `releases/helhest_v2.json` is the prose every release inherits. `helhest-release` turns a spec and the scenes into the root's `manifest.json`, `splits.json`, keyframe lists and `README.md` |
| `v2_inventory.csv` | every Odin1 + Fixposition bag on subtdata as the 2026-09 survey saw it; the rows the bag lists were chosen from (`V2_0_SELECTION.md`) |
| `odin_tf_static_2026-09-30.txt` | what 42 bags published for the Odin1's self-estimated extrinsics; the survey behind `dataset_tools/tf_pins.py` |
