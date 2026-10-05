# Dataset viewer

A browsable web view of an extracted dataset root: an index of every scene, a
per-scene inventory of what is actually on disk, a page per ground-truth
segment, and a 3D view of each odometry segment with the camera image, BEV
height map and raw cloud alongside it. Where a scene carries GNSS fixes, a map
shows where it happened.

```bash
pip install "dataset_tools[viewer]"                      # or, from a checkout: pip install -e ".[viewer]"
python -m dataset_tools.viewer /path/to/dataset_root     # or: helhest-viewer <root>
# then open http://127.0.0.1:8080/
```

The viewer needs Python 3.12+ and pulls in
[SimView](https://pypi.org/project/simview/) (`>=5.1.1`, with its `authoring`
extra: torch + einops), FastAPI, uvicorn, Jinja2 and Pillow. The 3.12 floor is
SimView's, not ours — the extraction tools still run on 3.10, so keep the viewer
in its own virtualenv when the extraction venv is older:

```bash
python3.12 -m venv .venv-viewer
.venv-viewer/bin/pip install -e ".[viewer]"
.venv-viewer/bin/python -m dataset_tools.viewer /path/to/dataset_root
```

### Command line

| Flag | Meaning |
| --- | --- |
| `root` | the dataset root: the directory holding `0000/`, `0001/`, ... |
| `--host` | bind address (default `127.0.0.1`; `0.0.0.0` to reach it from another machine -- see [Who can reach it](#who-can-reach-it)) |
| `--port` | port (default `8080`) |
| `--max-seconds S` | seconds per viewing window; longer segments are split into parts of equal duration (default `600`) |
| `--max-frames N` | frames per viewing window: a backstop under `--max-seconds`, which normally decides (default `12000`) |
| `--cache-dir DIR` | where built scenes are cached (default: a per-root directory under `$XDG_CACHE_HOME/dataset_tools_viewer/`) |
| `--clear-cache` | delete this root's cached probes and built scenes before starting, so everything is rebuilt |
| `--cache-max-gb GB` | keep at most this much of built 3D windows on disk, oldest build removed first (default `5`) |
| `--notes-dir DIR` | where reviewer notes are kept (default: `<root>/notes/`, so they travel with the dataset) |
| `--no-notes` | turn notes off entirely: none read, written or shown |
| `--author NAME` | who new notes are from (default: `$USER`) |
| `--no-prefetch` | do not build the next window in the background when one is opened (see [Performance](#performance)) |
| `--open` | open a browser at the index page |
| `-v` | debug logging |

Apart from notes (below), the dataset root is only ever **read** -- every byte
the viewer writes goes to the cache directory -- so it is safe to point at a
read-only or shared copy. Notes are the one thing a person adds, and they are
written into `<root>/notes/`; on a root that will not take them the viewer says
so and keeps composing switched off rather than failing at the save.

### Who can reach it

The server has **no authentication**. Anyone who can open the port can read
every scene, have windows built (which costs CPU and cache disk) and, with
notes on, write into `<root>/notes/`. It is meant to run on the machine you sit
at, or on a cluster node reached over an SSH tunnel (`slurm/viewer.batch` and
`scripts/viewer_connect.sh` do exactly that), and the default bind of
`127.0.0.1` keeps it there. A loopback bind also answers only to the `Host`
names `127.0.0.1` and `localhost`, so a web page elsewhere cannot reach it
through a hostname it arranged to resolve to your machine.

`--host 0.0.0.0` hands the viewer to everyone on the network. Do that only on
a network you would let write to the root, or pass `--no-notes` so nothing can
be written; the viewer logs a warning at startup when it is bound beyond
loopback with notes on.

Start at the index, click a scene id for its inventory and window list, then a
window to open the 3D view. Nothing else needs configuring.

A view opens with the camera aimed at the robot, wherever in the window it
starts -- which is not always the origin, since the later parts of a split
segment carry on from where the previous part ended. Opening a window from a
frame's own page (`?frame=N`) seeks the playhead to that frame rather than to
the window's start.

## It follows the file structure, not the rig

Nothing in the viewer is specific to a sensor generation. A scene is any
`NNNN/` directory; every file inside it is *probed* and the UI adapts:

- The index shows, per scene, which artefacts exist — and which are partial
  (`height_maps 1856/15434`, for a directory still being built).
- Absent files are struck through rather than hidden, so "this scene has no
  `cmd_vel.npy`" is visible rather than silently implied.
- A scene with no `gps.npy` has no map anywhere, and every page it could have
  been on looks exactly as it did before maps existed.
- A window builds from whatever it has: no height maps means the terrain comes
  from the aggregated cloud instead, no `joint_states.npy` means no wheel-velocity
  plots and no wheel spin, no `transforms.npy` means no drift comparison.
- Odometry is parsed from the columns, not from a rig's conventions. Segment
  `reason` tags (`filtered:pose_jump`, `v11_fill`, `fixposition:fix_lost`, …)
  are read out in words where they are shown (`windows.describe_reason`); a tag
  it does not recognise passes through verbatim.

## Windows, and why a view never spans two segments

`odometry_segments.txt` marks stretches of ground truth that were each
**re-anchored to identity at their own first pose**. Poses from two different
segments therefore share no reference frame, and a trajectory, an aggregated
cloud or a splatted terrain drawn across a boundary would be meaningless.

That anchoring zeroes the robot's roll and pitch as well as its position and
yaw, which is wrong: roll and pitch are absolute, measured against gravity. So
a window is **levelled** before it is drawn — rotated by the attitude
`gravity_align.npy` records at the segment's anchor frame, which is the attitude
the anchor threw away. Without it a segment that opens on a slope renders level
and stays level for its whole length, and one that opens with the robot on its
side renders it upright.

Segment edges are timestamps on the ground-truth pose clock; frames carry their
own. A frame landing within half a frame period of an edge counts as inside the
segment, so the frame nearest a segment's first pose is that segment's first
frame rather than a frame belonging to nothing. The slack never reaches past the
midpoint of the gap to a neighbouring segment, so no frame is ever claimed by
two.

So the unit of viewing is a *window*:

| window | pose source | what it is |
| --- | --- | --- |
| `segNNN` | `odometry.txt` | one ground-truth segment; the trustworthy view |
| `raw` | `transforms.npy` | the whole scene by wheel odometry — continuous, but it drifts |
| `raw` (no transforms) | none | frames only; the robot stays at the origin |

Segments longer than `--max-seconds` (default 600) are split into consecutive
parts, with `--max-frames` (default 12000) as a backstop for a root recorded
fast enough that 600 s is still too many frames. The parts are equal in
*duration*, not in frame count: the number of parts is chosen first, so a
segment 10% over the cap becomes two even halves rather than one full window
and a stub, and the boundaries are then placed at equal times. Only a
non-uniform frame clock tells the two apart -- but that is when it matters,
since frames dropped early in a segment would otherwise push the cut late and
hand the last part more seconds than the cap allows. Where cutting by time puts
more frames in one part than the ceiling permits, the part count is raised until
both caps hold for every part.

The cap counts seconds rather than frames because that is what a longer window
actually costs. Building one window reads a bounded number of per-frame files
whatever its length -- `hm_frames` and `cloud_frames` pick a fixed subset -- and
the bundle grows barely at all: on v1.3's longest segment (0019/0, 29 minutes),
twelve times the frames cost 1.9x the bytes and no extra build time. What does
scale is how thinly those fixed samples are spread over the ground driven, which
is a cost per second, and the budgets are scaled with the window to hold it
roughly constant (below). Counting frames also meant one cap with two meanings:
v1.3 holds scenes at 3.2 Hz and at 10 Hz, so the old 1500-frame cap left a 462 s
window whole in one scene while halving a 180 s segment in the next.

No window reaches past the last line of `data_ts.txt`. A scene's frame count is
the largest of several counts that can disagree — a directory built in flight
has more images on disk than the timestamp file has lines — and a frame with no
timestamp has no place on the playback clock, so there is nothing to build for
it. Such a frame still gets its own page (the image and the cloud are there);
it just does not offer a 3D view.

The window table's **GT poses** column is how many ground-truth poses the
segment behind a window actually carries, and at what rate. This matters more
than it looks: poses are interpolated onto the frame clock, so a segment
anchored at 0.5 Hz plays back exactly like one anchored at 10 Hz, with the
trajectory between its poses a straight line over a second or more of real
driving. Below 2 Hz the rate is flagged.

## What each file turns into

| file | in the viewer |
| --- | --- |
| `odometry.txt` | the robot body's animated pose |
| `odometry_segments.txt` | the window list; the episode marker on the playback bar |
| `transforms.npy` | a second "wheel odom" ghost body, plus the `wheel_odom_drift` plot |
| `gravity_align.npy` | the attitude comparison and its plots; places each height map level in the world; levels the window (above) |
| `height_maps/` | the terrain surface, and a per-frame tile in the side panel |
| `height_masks/` | the terrain's `validity` colour layer (weighted mean per cell), and its own panel tile |
| `driven_similarity/` | the terrain's `driven` and `drivable` colour layers, and the **Driven & drivable** tile (DRIFT's product; see below) |
| `clouds/` | an aggregated point-cloud body, three terrain colour layers, and a per-frame top-down tile |
| `images/` | the side panel's camera tile, served per frame |
| `joint_states.npy` | wheel spin animation + velocity/effort plots |
| `joint_setpoints.npy` | commanded wheel-velocity plots, and with `joint_states.npy` the `wheel_slip_*` plots (commanded minus measured, rad/s, per wheel) |
| `cmd_vel.npy` | `cmd_linear_x` / `cmd_angular_z` plots |
| `camera_intrinsics.txt`, `T_cam_to_base.txt` | shown on the scene page; a small camera body at the real extrinsic; and, on a frame page, the frame's cloud projected into its image (below) |
| `gps.npy` | the map, on the scene, segment and frame pages |
| `time_align.txt` | the scene page's **Extraction** card: whether the frame clock was corrected, by how much, and from how many windows |
| `gnss_guard.txt` | the same card: how many fabricated GNSS/INS stretches were removed and what they cost in poses and frames |
| `tf_pins.txt` | the same card: how far each pinned static transform sat from what the bag itself published |

## The cloud on the image: a check on the extrinsic

A frame page whose scene carries both `camera_intrinsics.txt` and
`T_cam_to_base.txt` shows a second copy of the camera frame under the first,
with every lidar return of that frame projected into it and coloured by depth
(`/api/scene/{scene}/frame/{n}/overlay.jpg`). It is a calibration check: a
road that lands beside the road, or a trunk that lands a metre from its
picture, is a wrong extrinsic -- the error that shifted every v1 cloud by
0.34 m, and the one the Odin1's own extrinsic estimate wanders by between
boots. Metres of error are pixels of misalignment at any range. Where
`camera_info.yaml` is present, its distortion model (e.g. `equidistant` fisheye
for Odin1 or `plumb_bob` / `rational_polynomial` for pinhole rigs) is applied;
otherwise, uncorrected pinhole projection is used as fallback. A root that has only
`T_cam_to_lidar.txt` (v1) gets no overlay: that matrix is the camera in the raw
lidar frame, not the base frame the clouds are in.

## The scene page's two strips

Both strips on a scene page are read the same way -- point at a thing, get the
thing itself -- because a tooltip is not an answer on a touch device and not a
place anyone looks on a desktop.

The **ground-truth timeline** draws one block per segment over the scene's own
time span, and each block is a link to that segment's page. Amber blocks are
filled ground truth, green is this root's own run; the tooltip names the
segment, how long it is, how many poses it carries and where it came from or
why it ended. A segment a second or two long is under two pixels wide on a
card-width track, so the block keeps its honest width and only its hit area is
widened, by two pixels either side; the tooltip says which one the pointer
found. A scene whose segment file cannot be read (a curated-only root) falls
back to the flat spans it always had, which name no segment and do not link.

The **file inventory** is a set of toggles. Opening one says what the file
holds and what the dataset does with it -- prose written per artefact in
`probe.ARTEFACT_PURPOSE`, not the one-line shape that used to live in the
`title` attribute -- along with whether this scene has it, how many rows,
lines or files, and where it sits on disk. One is open at a time, because the
card sits beside the map and stacking panels would push the map's height around
on every click. The server renders every panel expanded and `static/files.js`
collapses them, so with JavaScript off the card is a readable list of what each
artefact is rather than a dead strip.

Segment reasons are translated before they are shown anywhere -- in these
tooltips, in the window table, on the segment page. A reason accumulates tags
(`regen_fill:t194+refined`): the part before the first `+` says where the
segment came from or why it ended, each `+` tail says what was then done to its
poses, and both halves are turned into plain language by
`windows.describe_reason`. Nothing that reads as a tag the tooling invented for
itself should reach a page.

The **tabs** under the cards, and the pager of every frame strip, switch in
place: a click fetches the same page with the new query and swaps in only the
region the link lives in (`static/swap.js`), so the cards above and the scroll
position stay where they were, the address still updates and the back button
walks it. A frame tile opens the frame's own page as before. Without
JavaScript the links are what they always were, a full page each.

## The segment page

A segment is the unit the ground truth is actually anchored in, and until it
had a page of its own there was nowhere to *read* about one: the scene page
lists segments and the 3D view plays one. `/scene/{scene}/segment/{n}` is the
page in between, and it is reachable from everywhere a segment is named — the
window table, the ground-truth timeline above it, a frame's ground-truth card,
the 3D view's panel.

It carries three frames from inside the segment, what the segment is (when,
how long, how densely anchored, how many frames, the silence either side, why
it ended), what the robot did inside it (path, moving fraction, speed, net
displacement, z range, roll and pitch), the map of where it happened, the notes
whose frames touch it, its windows with their **Open 3D** buttons, and a strip
of its own frames.

The number in the URL is the segment's number in `odometry_segments.txt`, not
its position in the file — the same number the window keys carry (`seg003`),
so a link from a window and a link from a frame resolve to the same page even
after `scripts/filter_segments.py` has renumbered the table. Every figure on
the page stops at the segment's own boundaries: poses either side of one share
no reference frame, so nothing is ever accumulated across a boundary.

## Signals: the arrays over time

The cards reduce every array to a mean and a max, and the 3D view plots them only
per window, once built. The **Signals** card on the scene page (whole recording,
ground-truth segments shaded) and on the segment page (that segment only) plots
them over one shared time axis (`signals.signal_charts`):

| chart | from | what to look for |
| --- | --- | --- |
| ground speed, m/s | `odometry.txt` (differentiated inside each segment, never across one), `transforms.npy`, `cmd_vel.npy` | where GT and the robot's odometry part |
| wheel speed, rad/s -- one chart per wheel, one shared scale | `joint_states.npy`, dashed `joint_setpoints.npy` | a wheel leaving its dashed line: slip or load |
| wheel effort | `joint_states.npy`, in the file's units | slopes, mud, obstacles |
| roll & pitch, deg | `gravity_align.npy` | slopes, tip-overs |
| GNSS altitude, m and horizontal σ, m | `gps.npy` | a step in σ is a lost or downgraded fix |
| frame interval, ms | `data_ts.txt` (longest gap per column) | dropped frames |

One chart per unit, never two y-axes. Each is resampled onto 600 columns: the mean
of the samples in a column, interpolated where a column has none, and a gap where
the samples either side are more than 1.5 s apart. The y-range covers the central
99 % of values, so a spike is clipped at the edge rather than flattening the chart.
The lines are server-rendered SVG; `static/charts.js` adds a crosshair shared by
every chart, a per-chart readout, and a click (or arrow keys + Enter) that opens
the frame under the pointer. Each chart carries its own time axis and folds away
under the arrow before its name (a native `<details>`, open by default).

## The map: where the recording happened

A scene that carries `gps.npy` gets a map. On the scene page it sits beside the
file inventory, the two of them a pair; on a segment page the segment's own
stretch is highlighted with the rest of the outing dashed behind it; on a frame
page a marker shows where that frame was, interpolated between the fixes around
it (and left off when the nearest fix is more than a second away — a line drawn
across a receiver dropout would be a claim about where the robot went).

**A scene without `gps.npy` renders no map at all** — no card, no grid, no
Leaflet, no request. That is the same habit as everything else here, in the one
case where it is inverted: an absent artefact is normally struck through rather
than hidden, but a struck-through map is a hole in the page, and most roots
have no GNSS at all.

The fixes are the receiver's own position and are **not** ground truth. The
numbers beside the map are the track's: how many fixes and at what rate, the
distance along them, the extent of the box they cover, the median horizontal
sigma out of the covariance the receiver reported, and the share of the track
at each fix quality. On a segment page they describe that segment only.

Leaflet is **vendored** (`static/vendor/`, BSD-2-Clause, unmodified) rather than
loaded from a CDN: the viewer is routinely run on a cluster node or in a
container with no route to the internet, and a map whose library fails to load
is a blank box that explains nothing. The OpenStreetMap *tiles* do need the
network; without them the track is drawn on a plain backdrop and the card says
so. The tiles are inverted in CSS, since OSM ships one raster style and it is a
light one.

The track is thinned server-side (`?max_points=`, 2000 by default) so a long
scene ships tens of kilobytes rather than ten thousand pairs of doubles, and it
is served from its own endpoint rather than folded into a window bundle: a
bundle is cached per window under a content stamp, and a track belongs to the
scene, not to a view of it.

## The atlas: curated prose, generated numbers

A root may carry an `atlas/` directory the index renders instead of its own
probe output: a masthead, groupings into outings, a description per scene, and
three thumbnails per card. It is entirely optional — without it the same page
renders from the probe alone.

Only the prose in it is editorial. Everything else on those cards is a rendered
number, and rendered numbers go stale: HELHEST v1.3 shipped with `atlas/`
symlinked to v1.2's (v1.3 refines odometry in place, so almost every other file
is legitimately shared) and the index quietly showed v1.2's poses, segments,
coverage and path lengths under a v1.2 title. So the two halves live in two
files, and only one of them is written by hand:

| file | who writes it |
| --- | --- |
| `atlas/curation.json` | a person. Title, kicker, intro, root note, footer, the groups, one description per scene, and the frame index behind each thumbnail. |
| `atlas/atlas.json` | `scripts/make_atlas.py`. The curated half copied through, every number computed from the root, and the `dataset_version` it was generated against. |

```sh
scripts/make_atlas.py <root>            # regenerate atlas.json + thumbnails
scripts/make_atlas.py <root> --check    # recompute and diff; writes nothing
```

Numbers come from the same code the pages use (`SceneProbe`,
`scene_page.path_and_motion`), so the atlas cannot disagree with the page it
feeds; the rest comes from `<scene>/regen/` and, for the 2 s window error, from
that scene's `regen/odom_ref.txt`. Run it after anything that changes odometry,
frames or clouds — and when the atlas's `dataset_version` stops matching
`manifest.json`, the index says so above the masthead rather than showing the
old numbers silently.

**Thumbnails.** A slot names the *frame* it shows (`"thumbs": {"a": 196}`), and
the stored `atlas/thumbs/<scene><slot>.jpg` is a cache of that frame, not the
definition of it: delete it and `/api/scene/<id>/thumb/<slot>.jpg` renders the
same frame live at the same width (384 px). The index card, the scene page's
hero caption and the frame it links to all resolve through one
`index_page.thumb_index` call, so they cannot name different frames. A root
whose atlas predates this names no frames, and the slots fall back to 10/50/90 %
of the scene — which is where those files were cut.

## Disk usage

`helhest-disk-usage <root>` writes `<root>/disk_usage.json`: bytes per scene,
per modality directory, and per top-level file. `helhest-extract` refreshes it
for the scenes it just wrote, so run it by hand only on a root extracted before
that, or after changing files in place (name the scenes, or `--all`). A file
written under an older schema is rescanned whole.

When it is present the index adds an **on disk** total (with GB per hour of
recording), a bar of where the bytes go by modality, and an **on disk** cell on
each scene card. The index sum covers only scenes in both the root and the
file; any scene the file has not scanned yet is named under the bar. A scene
page gets a **Disk usage** card (total, per frame, the same bar) and each entry
in its file inventory shows its size. Like the atlas, the file is re-read when
it changes, so a rescan shows without a restart.

## Validation: what `helhest-validate` made of each scene

`helhest-validate <root> --csv <root>/validation.csv` (which is what
`slurm/finalize_v2.sh` runs) leaves the validator's table at the root. When it
is there, every index card gets a chip -- `validated ok`, `N warnings` or `N
errors`, with the row's numbers in its tooltip -- the masthead counts how the
scenes fared, and the scene page gets a **Validation** card with the status,
the counts and the numbers behind them (max step, max tilt, sync offset,
non-finite fraction). The file carries *counts* of findings, not their text.
The text comes from `validation.json` next to it (`helhest-validate <root>
--json <root>/validation.json`): the card then lists every finding, most
severe first, with its code, and the ones with a place in time -- a hole in the
frame clock, a gap in the joint states -- are blocks on a timeline and links
to the frame where they start. Without `validation.json` the card says how to
get the messages rather than pretending. Under the masthead, `Validation: all
scenes · ok · with warnings · failed` filters the index (`/?status=warn`). A
scene the file does not mention gets no chip, and a root without the file
looks exactly as before. Both files are re-read when they change.

## The train/val split: which scenes a model is scored on

A root that declares its hold-out itself is tagged without any export: from
HELHEST v2.1 the manifest names the held-out session (`held_out.scenes`), and
from v2.3 the root carries `keyframes_train.lst` beside `keyframes.lst`. The
viewer reads those as a `held_out` split (held out as validation, the training
list's scenes as training, a scene in neither unassigned) whenever there is no
`splits.json`; the file still wins when present, because it carries the window
counts a declaration cannot, and either is re-read when it changes on disk.

A root may carry a `splits.json` naming the scenes held out from training. When
it does, every index card gets a labelled band down its left edge — solid violet
`VAL` for held out, quiet slate `TRAIN` for trained on — and the scene page's
header a tag in the same colours. Under the masthead a panel names the fold,
counts each side in scenes and in windows (or labels), draws the window shares
as a bar, and filters the index to one side: *All scenes*, *Validation* or
*Training*. The filter is kept in the address (`/?split=val`), so a filtered
index can be linked; the fold's caveat note sits behind *What this split
measures*. Without the file nothing is coloured and the pages render exactly as
they did before; the viewer has no opinion of its own about which scenes are
validation data, because that is a modelling decision and not something a
directory listing can answer.

**Only a whole-scene split can be drawn.** DRIFT's default fold is a *stride*
over keyframes (`i % 4`), which cuts inside every scene: under it no scene is a
training scene and none is a validation scene, so there is nothing to colour.
`scene_strat` is the fold that holds out whole scenes, and it is the one this
reads.

**It is declared, never derived.** The lists are decided in DRIFT
(`datasets/helhest.py`, `SCENE_STRAT_VAL`) and *serialised* by its exporter —
never retyped here. A hand-copied duplicate is how DRIFT's own `train.py` came
to reject the mode name it had just gained, and mirroring the constant into this
repo would set up the same failure across two projects:

```sh
# in the DRIFT checkout
python3 scripts/data/dump_split.py --rootdir <root> --out <root>/splits.json
python3 scripts/data/dump_split.py --rootdir <root> --out <root>/splits.json --no_windows   # ~2 s, no window counts
```

| key | what the viewer does with it |
| --- | --- |
| `split_mode` | names the fold in the panel and in every band's and tag's tooltip |
| `root`, `dataset_version` | checked against this root — see below |
| `val_scenes`, `train_scenes` | the card band, the scene-page tag and the filter |
| `excluded_scenes` | a third, grey state: in neither split (empty today) |
| `labels_per_scene`, `windows_per_scene` | each side's share in the panel, and its bar |
| `source` | shown under *What this split measures*, so the page says where the lists came from |

A hand-written file may use the short key names `val`, `train`, `excluded` and
`name` instead, and may write scene ids as integers (`[0, 9, 15]`).

**Scene ids are only meaningful inside one tree.** They survive root *versions*
— v1.2 through v1.5 renumbered segments, never scenes — but not *trees*: the
ScaLR copy of HELHEST carries rear-camera scenes DRIFT's does not, and the two
part company from `0030` on. A file exported against one tree therefore labels
the other correctly up to that scene and wrongly after it, plausibly and with
nothing else on the page to reveal it. So the index says so above the masthead
when the file names a different root, when it says nothing about a scene this
root has, or when its `dataset_version` has gone stale. A scene the file does
not mention is left **unmarked** rather than assumed to be training data.

**The held-out share always names its unit.** The same fold is 23.7 % of the
proprio labels and 25.8 % of the loader windows, and one scene (`0019`, fallow
meadow) is 30 % of the held-out labels but 56 % of the held-out windows. A bare
percentage would be read as whichever the reader had in mind, so the panel
quotes windows when the file carries them, labels otherwise, and says which.

**What this fold can and cannot measure.** It tests "new patch, known site", not
unseen terrain: scenes repeat ground — `0018`/`0019` are the same meadow,
`0024`–`0027` the same drained reservoir, `0012`–`0015` the same farmyard,
`0020`/`0028`/`0029` the same gravel network, and all of `0030`–`0042` are one
forest on one day. With three sites no fold is both terrain-new and trainable.
Put that in the file's `note` and the panel will carry it, behind *What this
split measures*. The full reasoning is
in the comment block above `SCENE_STRAT_VAL` in DRIFT's `datasets/helhest.py`.

## Notes: what only a person can say

Everything else on a page is probed or computed. A note is the one thing that
cannot be: a stretch whose odometry is visibly wrong, a camera that dropped
out, something that happened in front of the robot. So notes are stored next to
the data, the way `atlas/` is, one file per scene:

```
<root>/notes/0007.json
```

A note is free text pinned to a **frame range**, with a topic (`odometry`,
`camera`, `event`, `other`) so a stretch can be found again by what is wrong
with it, and optionally a **verdict** -- `keep`, `drop` or `fix` -- when the
reviewer has decided what should happen to that range. Verdicts are what make a
review campaign trackable: each index card shows its note count and verdicts,
`/?verdict=drop` lists the scenes with a range marked drop, the `/segments`
table marks the windows a verdict touches, and `/api/notes.csv` exports every
note with its frame range *and* the same range as frame times (`t_start`,
`t_end` from `data_ts.txt`), so whatever applies the verdicts needs neither the
viewer nor the scene's clock. The range is the anchor because it is the most durable identifier a
scene has: segment indices are renumbered whenever `scripts/filter_segments.py`
or another cutting pass rewrites `odometry_segments.txt`, so a note
anchored to "segment 7" would afterwards point at different driving. Both ends
of the range are **inclusive** — "frames 1204–1388" means what a person writing
that means, which is deliberately not `windows.Window`'s convention, where
`frame_end` is exclusive.

```json
{
  "scene": "0007",
  "dataset_version": "HELHEST_v1_3",
  "notes": [
    {
      "id": "3f9a1c",
      "frames": [1204, 1388],
      "topic": "odometry",
      "text": "GT jumps ~2 m sideways around 1250 and never recovers.",
      "author": "jan",
      "created": "2026-09-05T10:42:11+00:00",
      "updated": null
    }
  ]
}
```

The file is a person's to edit. A malformed entry drops out and the rest of the
scene's notes still load; a topic the UI does not offer is kept verbatim,
exactly as an unrecognised segment `reason` tag is.

**Where they are written from.** The 3D view's panel, because that is where a
bad stretch is actually noticed — scrubbing a segment and watching the ground
truth walk off the terrain. The panel's **notes** section holds the range as
two frame-number fields: type either end, or press its *now* button to take the
frame showing, so marking a stretch is two clicks and a sentence. Ends typed
the other way round are ordered rather than refused. It lists the notes that
touch the open window, and clicking one seeks the playhead to its first frame.
Each of those carries two buttons down its right edge — an amber pen to edit
above a red bin to delete. Icons and not words, stacked and not in a row,
because the panel is only as wide as the sidebar and anything on the note's
head row is width taken off its range and topic. Editing opens the same
controls the note was written with, over that note: a range corrected by
scrubbing to the frame the bad stretch really starts on and pressing *now* is
the reason to fix it from here rather than from the scene page. A saved range that no longer
touches the open window takes the note off this list — it belongs to the
driving it names, not to the window it was typed in.

While a field there has focus it owns the keyboard. SimView binds space, the
arrows and `p` to `document` and `window` with no useful focus check —
`PlaybackControls` has none at all, and `Controls` tests for an `INPUT`, which
a textarea is not — so a space typed into a note used to toggle playback
instead. Both listen in the bubble phase, so the panel stops key events coming
from a field it holds before they get that far.

The scene page carries the same notes as a card, where they can equally be
written, edited and deleted; its window table marks every window a note's
frames land in, so a scene shows at a glance where the flagged stretches are.
An edit in either place reaches **every** field a note has — both ends of the
range, the topic and the text — because all four are typed by hand and so are
gettable wrong, and correcting one by deleting the note and writing it again
would throw away its author and its date, which are the parts nobody can
retype. The `id` and `created` never move; `updated` is stamped instead.

**When the root will not take them.** Roots are routinely read-only or shared
copies. The viewer probes the location once at startup — an actual write, since
network mounts report permissions they will not honour — and lands in one of
three states, which `/healthz` and the scene page both report:

| | what happens |
| --- | --- |
| the root is writable | notes go to `<root>/notes/`, and travel with the dataset |
| `--notes-dir PATH` | notes go there instead; they no longer travel with the data |
| `--no-notes` | off entirely: nothing read, written or shown |

A read-only root with no `--notes-dir` still **reads** its notes. Only
composing is refused, with the reason on the page rather than the form silently
missing — the same habit as an absent artefact, which is struck through rather
than hidden. The notes directory itself is created by the first note, not by
opening the root: looking at a dataset leaves it as it was found.

**Stale frame numbers.** A frame index only means the same thing while the
extraction does. Each notes file records the `dataset_version` it was written
against, and the scene page says so when that stops matching `manifest.json` —
the same check, for the same reason, as the atlas's.

**Reading them back.** `/notes` lists every note in the root on one page,
grouped by topic and ordered by scene and frame inside each, with `?topic=`
narrowing it to one topic: a review campaign's to-do list. Each entry links to
its scene, to its first frame and, where the frame sits in a window, to the 3D
view opened on it. A notes file whose scene is no longer in the root is listed
and marked as such rather than dropped: it is a person's work, and a scene
renumbered out from under it is exactly the case worth seeing. `/api/notes`
returns the same as JSON. With `--no-notes` the page says notes are off.

Notes are served from their own endpoint and are never folded into a window
bundle. A bundle URL carries a content stamp and its `meta.json` is cached per
window on disk, so a note stored in there would be invisible until the next
code change, or would rebuild every window it touched each time someone typed
one.

## Ground truth against the IMU

Levelling puts the ground truth's own attitude and the IMU's in the same frame,
so the two become directly comparable — and the IMU does not depend on the
odometry, which makes the gap between them a check rather than a restatement.
The side panel carries it: per frame, roll and pitch from each source and the
angle between the two up axes (`attitude_error_deg`, free of yaw and of the
wrap a per-angle difference hits near ±180°); per window, the median and 95th
percentile, with a verdict. `roll_deg_gt` / `roll_deg_imu` and their pitch
counterparts plot side by side.

Calibrated across this root: segments whose ground truth is sound sit under 11°
from the IMU even on sustained steep slopes, and a segment whose ground truth
has actually failed sits above 40°. The bands (`build.ATTITUDE_BANDS`) travel in
`meta.json` rather than living in the front-end. A single frame's gap can be
large honestly — a jolt the IMU catches and the interpolated ground truth smooths
over — so the verdict is taken from the window, not the frame.

The same verdict sits in the window tables of the scene page and the segment
page, as a **GT attitude** column: the median gap in degrees, green, amber or
red on the same bands. It is computed before any window is built, from the
same code (`build.window_attitude` interpolates and levels the poses exactly as
a build does, then `build.attitude_error_deg` reads the gap), so the number in
the table is the number the panel will show once the window is opened. A
scene with more than 200 windows on one page shows dashes instead, and a
window with no ground truth or no `gravity_align.npy` has nothing to compare.

A window splats ~80 height maps 12.8 m across along a path a few tens of metres
long, so every terrain cell is covered by dozens of them. Their contributions
are weighted by how far the sample sat from the robot that measured it
(`build.HM_BLEND_SIGMA`), so a cell is decided by the frames that drove over it.
Averaged flat instead, a cell blends observations made from poses whose z
disagrees by however much the odometry drifted between them, and the ground ends
up floating away from the robot standing on it.

The terrain offers several colour modes: `height`, plus `coverage` (how many
height maps actually observed each cell — cells nothing observed are filled from
a trend surface and show as zero coverage), `validity`, and `cloud_points` /
`cloud_intensity` / `cloud_spread` from the aggregated cloud. What each one
means travels with the window (`build.TERRAIN_COLOR_MODE_HELP` → `meta.json` →
the panel's **terrain colour modes** legend), so the names in SimView's own
"Color Mode" dropdown are explained next to it rather than left to be guessed.

A root DRIFT has written `driven_similarity/` into offers two more modes.
`driven` is 1 where a wheel rolled over the cell -- the wheel-bottom contacts of
the ground-truth path, rasterised per keyframe by DRIFT with the wheel geometry
of the model its checkpoint trained under (the sidecar's `robot`; on v2 roots
that is r 0.360 m where the rig measures 0.334) and united over the window --
and 0 everywhere else, terrain the robot merely saw included.
`drivable` is each cell's highest cosine similarity, over the window's
keyframes, between its pre-head SeTProP feature and any driven cell's: 1 on the
path itself, and above the threshold the sidecar records (0.9, as DRIFT trained)
where DRIFT trusts its prediction. It is a feature match and not a terrain
class, and the features are known to encode range and viewpoint as well as
terrain, so a band of high similarity around the robot is the sensor's
geometry, not grass. Measured on 2026-09-30 over 37 held-out v2.1 keyframes
(WI_v8 backbone, `v20_wi8` at epoch 39): the 0.9 gate admits 0.92 / 0.61 / 0.26
of ground-level cells at 0-2 / 2-4 / 4-6.4 m from the robot against 0.79 / 0.12
/ 0.03 of cells more than a metre tall, and 25 % of random cell pairs. Beyond
2 m it does separate obstacles from ground; within 2 m, where the wheels are
about to go, it admits most of both, and everywhere the number falls with
range. Read it as "near the path and seen by the lidar", not as a terrain
match (DRIFT `docs/METHOD.md`, issue 24). The same checkpoint's compression
prediction has the same reach: on held-out windows it tracks the target out to
about 1.5 m and decays to nothing by 3-4 m, so the trust bubble and the head's
own footprint coincide because both fade with range, not because they agree
about terrain. The **Driven & drivable** tile shows
one frame of it unmixed: the ember ramp is `max_sim`, a visible step marks the
threshold, and the driven cells are cyan.

`validity` is the mean validity of the samples behind a cell, weighted exactly
as its height was, so it reports how measured the surface you are looking at
actually is. Taking the best of them instead reads far more of the map as
measured than the height warrants: a distant frame's map edge does get real
lidar returns and does mark the cell measured, while the near frames that
decided the height were driving past it and extrapolating behind themselves — on
a straight traverse that describes roughly two covered cells in five. The
per-frame Validity tile still shows the three states unmixed.

### Paging through a scene's segments

The panel's header carries **&lsaquo;** and **&rsaquo;** beside the window's
name, and <kbd>[</kbd> / <kbd>]</kbd> do the same: they open the window before
or after this one, so a scene is reviewed segment by segment without going back
to the scene page between each. They are ordered by frame, so a single-frame
view (`frame00042`) pages back into the segment holding it, and stop at either
end of the scene rather than wrapping. The build options in the URL travel
along -- page to the next segment and it is built with the same cloud and
terrain settings -- while `?frame=` does not, since it names a frame of the
window you are leaving.

The two keys are SimView's own episode shortcuts, which do nothing here: a
dataset scene declares no episodes. A key pressed into a note field is left to
the field.

### The 3D view toggles

Two things in the scene are diagnostics rather than the robot, and both are off
until asked for, each with its explanation beside it in the panel:

- **show lidar cloud** — every cloud in the window, aggregated in the world
  frame. Turning it on also drops the terrain surface, which is drawn over the
  points and describes the same ground.
- **show wheel-odometry ghost** — an amber, translucent marker at the pose
  wheel odometry alone would give. The gap between it and the robot is the
  drift plotted as `wheel_odom_drift`; it grows without bound and routinely
  ends up a hundred metres off the path, which is why it is not shown unasked.

The robot stays a solid mesh throughout: the cloud is drawn independently of
SimView's global body-visualization mode rather than by switching the whole
scene into "points" (which has no robot to show).

### Working around SimView

`dataset_tools/viewer/static/simview_patch.js` holds the corrections that need
to reach into SimView's own objects rather than just its CSS: things only this
dataset can know, such as which bodies are bookkeeping and what the ghost is.
The stopgaps for SimView <= 4.2 bugs were removed with the move to SimView 5,
which fixes them upstream.

The `lidar_cloud` body deliberately carries no per-frame trajectory: it is
already accumulated in the window's frame, so a per-frame identity transform
would be a blob that says nothing. A stateless body is the supported way to
express a static prop.

## How it uses SimView

The viewer depends on the released `simview` package and **does not modify or
subclass it**. It builds a `SimulationScene` through the public authoring API,
then serves that scene through SimView's own documented static mode: setting
`window.__simviewStaticBase` redirects the front-end's fetches to flat
`model.json` / `states.json` / `blob/N` files, which this app generates on demand
and caches. The viewer page is SimView's own `index.html`, read from the
installed package at request time (so it tracks whatever version is installed),
with an overlay panel added around it. The panel follows playback by reading
`window.__debugSimView` — SimView's own debug handle — and never writes to it,
except through SimView's public GUI controllers for the view toggle.

## Performance

Dataset roots usually live on contended network storage, so:

- The index uses directory listings and `.npy` headers only — never a cloud or
  an image — and caches the result per scene, keyed by the scene's fingerprint:
  the directory and every entry directly in it (about thirty `stat` calls), so
  a file added to `height_maps/` or `odometry.txt` rewritten in place is seen.
- Scenes are built per window, on demand, and cached on disk under
  `$XDG_CACHE_HOME/dataset_tools_viewer/`.
- One window aggregates a bounded number of clouds and height maps -- chosen
  evenly across the window rather than from its start -- and the terrain is
  coarsened rather than clipped when a long traverse would exceed the cell
  budget. The defaults below are what a ~1500-frame window gets; a longer one
  scales them up with its frame count (samples up to 4x, terrain cells up to
  2x) so that it splats as densely per second driven, and stops at the same
  ceilings the query parameters do. Append a query parameter to the view URL to
  override a budget for that window:

  | Query parameter | Default | Accepted range |
  | --- | --- | --- |
  | `cloud_frames` | 40 | 0-200 |
  | `cloud_points` | 160000 | 0-1500000 |
  | `hm_frames` | 80 | 0-400 |
  | `terrain_cells` | 500000 | 1000-2000000 |
  | `cloud_color` | `height` | `height`, `intensity` |

  For example `/view/0000/seg001?cloud_frames=120&cloud_points=600000`. Values
  outside the range are clamped, not rejected, and each distinct combination is
  cached separately.
- Camera frames are served one at a time as downscaled JPEG, not shipped with
  the scene.
- Opening a window starts building the next one in the background, with the
  same build options, so paging with <kbd>]</kbd> does not wait. It takes the
  same per-window lock a request would, so paging early joins the build rather
  than starting a second one. `--no-prefetch` turns it off, for a root on
  storage slow enough that an unasked build competes with the one being looked
  at.

## URLs

| Path | What it serves |
| --- | --- |
| `/` | the scene index |
| `/scene/{scene}` | one scene's inventory, window list and calibration |
| `/scene/{scene}/segment/{n}` | one ground-truth segment: what it is, what happened inside it, where |
| `/scene/{scene}/frame/{n}` | one frame: its image, its derived tiles and its numbers |
| `/scene/{scene}/jump?index=N` | the frames tab's jump box: redirects to that frame, clamped to the scene |
| `/view/{scene}/{window}` | the 3D view: SimView's page plus the overlay panel |
| `/api/scene/{scene}/gps.json` | the GNSS track, thinned for drawing (`?segment=`, `?frame=`, `?max_points=`) |
| `/api/scene/{scene}/notes` | that scene's notes (`GET`), and where a new one is posted (`POST`, a JSON body `{"frames": [first, last], "text", "topic", "verdict"}`) |
| `/notes`, `/api/notes` | every note in the root by topic (`?topic=`), as a page and as JSON |
| `/api/notes.csv` | every note as one table: scene, frame range, frame times, topic, verdict, author, date, text |
| `/segments` | every ground-truth window of the root in one table, sortable (`?sort=scene\|att\|rate\|duration`), with the notes and verdicts touching each |
| `/?status=ok\|warn\|fail`, `/?verdict=keep\|drop\|fix` | the index, only the scenes `validation.csv` gave that status, or that carry a note with that verdict |
| `/api/scene/{scene}/notes/{id}` | edit (`PATCH`, the same body; a field left out stays as it is) or remove (`DELETE`) one note |
| `/api/scene/{scene}/frame/{n}/image.jpg` | that frame's camera image, downscaled (`?w=`); a root holding the camera's own JPEG is served the file as it is when no downscale is needed |
| `/api/scene/{scene}/frame/{n}/overlay.jpg` | that frame's cloud projected into its camera image, coloured by depth (`?w=`); `404` without `T_cam_to_base.txt` and `K` |
| `/api/scene/{scene}/thumb/{slot}.jpg` | one index thumbnail (`a`, `b` or `c`): the stored `atlas/thumbs/` file, else the frame rendered live at 384 px |
| `/api/scene/{scene}/frame/{n}/heightmap.png` | that frame's BEV height map |
| `/api/scene/{scene}/frame/{n}/validity.png` | that frame's validity mask |
| `/api/scene/{scene}/frame/{n}/driven.png` | that frame's driven cells and feature similarity (`driven_similarity/`) |
| `/api/scene/{scene}/frame/{n}/cloud.png` | that frame's cloud, top-down |
| `/api/data/{scene}/{window}/{opts}/...` | the SimView static bundle: `model.json`, `states.json`, `blob/N`, plus `meta.json` |
| `/healthz` | liveness, the root path and the scene count |

A per-frame artefact the scene does not have returns `404` rather than a
placeholder; one the scene has but that cannot be read (a truncated JPEG, a
corrupt `.npz`) returns `422` with a red "unreadable" tile, which the page
shows as it is -- the two used to look the same. A scene whose `data_ts.txt`
is missing or unreadable carries a red banner on its scene, segment and frame
pages: the viewer then places everything on `pcd_ts.txt`, or on a made-up
10 Hz clock, and says which. A mistyped scene or frame number gets a page with
a way back to the index rather than a bare JSON error. `meta.json` records what the window was actually built from --
`terrain_source`, how many clouds and height maps were read, and
`missing_files` -- which is the quickest way to find out why a view looks
sparser than expected.

## Troubleshooting

**`the viewer's dependencies are not installed`.** `simview` (or FastAPI, or
uvicorn) is not importable in the interpreter running the viewer. Install the
extra: `pip install "dataset_tools[viewer]"`, on Python 3.12+.

**The index lists no scenes.** A scene is a directory whose name is *all
digits*, so a root whose scenes are named `scene_01/` reads as empty.
`/healthz` reports the count the viewer found.

**A window is slow to open the first time.** It is being built; the overlay
waits for the build and loads the scene when it is ready. Builds are cached per
(scene, window, options), so reopening is immediate. On slow storage, lower
`cloud_frames` and `hm_frames`.

**The robot floats above the terrain, or sinks into it.** The window levels and
weights what it can, but it cannot correct the ground truth. A robot that keeps
a steady few centimetres of clearance and then lifts off over a stretch is
showing `odometry.txt` drifting in z: the terrain under it was built from the
height maps of frames whose poses had drifted differently. Check the `pose_z`
plot against `pitch_deg_imu` — a climb with no pitch behind it did not happen, and
the episode is an odometry failure rather than a viewer artefact.

**Every window rebuilds once after an upgrade.** The bundle cache is keyed by
what was *asked for*, which does not change when the builder does, so
`meta.json` carries `build.META_VERSION` and anything older is rebuilt rather
than served. Without that, a window anyone had already opened would go on
serving the old metadata forever. The same goes for the data: `meta.json`
records the scene's fingerprint it was built from (`build.data`), and a window
whose scene has changed since -- re-extracted, a regenerated `odometry.txt`,
new height maps -- is rebuilt on its next request.

**The terrain looks flat, or blockier than 0.1 m.** Check the panel's build
line. Without `height_maps/` the terrain is accumulated from the clouds, and
without clouds either it is flat; and a long traverse is coarsened to stay
inside the cell budget. Cells nothing ever observed are filled from a trend
surface -- the `coverage` colour mode shows which cells were really measured.

**The map is a grey box with the track on it.** No tiles could be fetched:
either the machine running the viewer has no route to the internet or
`tile.openstreetmap.org` is unreachable from it. The card says so, and
everything that comes out of the dataset — the track, the markers, the numbers
— is still there. Leaflet itself is vendored and never needs the network.

**A scene has GNSS but no map.** The viewer only reads `gps.npy`; a scene
extracted before `--gps` existed does not have one, and re-extracting that bag
writes it (`helhest-extract ... --overwrite`). `/api/scene/<id>/gps.json`
returning `404` is the quickest way to tell the two cases apart.

**A note will not save.**
 The scene page's notes card says why: most often the
root is read-only, in which case pass `--notes-dir PATH` to keep notes
somewhere else (they then no longer travel with the dataset) or `--no-notes` to
switch the feature off. `/healthz` reports the same state under `notes`.

**A scene changed on disk and the index still shows the old numbers.** The
probe cache is keyed by each scene's fingerprint and refreshes itself, and the
scene list is re-read on every index render, so a scene an array job finishes
while the viewer runs appears without a restart. `atlas/`, `splits.json`,
`disk_usage.json`, `validation.csv` and `scenes.csv` are re-read when their
mtime changes. Frame images, overlays, BEV tiles and thumbnails go to the
browser with `Cache-Control: no-cache` and an ETag built from the scene's
fingerprint, the atlas and the viewer's code: an unchanged picture costs a
304, a re-extracted or renumbered scene shows its new pictures on a plain
reload. (Until 2026-10-01 they were cached for a day, thumbnails for a year,
under URLs that name only a scene number and a frame index.) To force a full
rebuild, start once with `--clear-cache`. That removes only what
the viewer wrote -- `probe_cache.json` and `bundles/` -- so it is safe even
when `--cache-dir` points at a directory holding something else. To clear every
root at once, delete the whole default cache instead:

```bash
rm -rf "${XDG_CACHE_HOME:-$HOME/.cache}/dataset_tools_viewer"
```
