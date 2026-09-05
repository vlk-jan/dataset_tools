# Dataset viewer

A browsable web view of an extracted dataset root: an index of every scene, a
per-scene inventory of what is actually on disk, and a 3D view of each odometry
segment with the camera image, BEV height map and raw cloud alongside it.

```bash
pip install "dataset_tools[viewer]"                      # or, from a checkout: pip install -e ".[viewer]"
python -m dataset_tools.viewer /path/to/dataset_root     # or: helhest-viewer <root>
# then open http://127.0.0.1:8080/
```

The viewer needs Python 3.12+ and pulls in
[SimView](https://pypi.org/project/simview/) (`>=4.1`, with its `authoring`
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
| `--host` | bind address (default `127.0.0.1`; `0.0.0.0` to reach it from another machine) |
| `--port` | port (default `8080`) |
| `--max-frames N` | frames per viewing window; longer segments are split into equal parts (default `1500`) |
| `--cache-dir DIR` | where built scenes are cached (default: a per-root directory under `$XDG_CACHE_HOME/dataset_tools_viewer/`) |
| `--notes-dir DIR` | where reviewer notes are kept (default: `<root>/notes/`, so they travel with the dataset) |
| `--no-notes` | turn notes off entirely: none read, written or shown |
| `--author NAME` | who new notes are from (default: `$USER`) |
| `--open` | open a browser at the index page |
| `-v` | debug logging |

Apart from notes (below), the dataset root is only ever **read** -- every byte
the viewer writes goes to the cache directory -- so it is safe to point at a
read-only or shared copy. Notes are the one thing a person adds, and they are
written into `<root>/notes/`; on a root that will not take them the viewer says
so and keeps composing switched off rather than failing at the save.

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

Segments longer than `--max-frames` (default 1500) are split into consecutive
parts, so building one window reads a bounded number of files no matter how long
the scene is.

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
| `clouds/` | an aggregated point-cloud body, three terrain colour layers, and a per-frame top-down tile |
| `images/` | the side panel's camera tile, served per frame |
| `joint_states.npy` | wheel spin animation + velocity/effort plots |
| `joint_setpoints.npy` | commanded wheel-velocity plots |
| `cmd_vel.npy` | `cmd_linear_x` / `cmd_angular_z` plots |
| `camera_intrinsics.txt`, `T_cam_to_base.txt` | shown on the scene page; a small camera body at the real extrinsic |

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
scripts/make_atlas.py <root> --init     # seed curation.json from an existing atlas
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
with it. The range is the anchor because it is the most durable identifier a
scene has: segment indices are renumbered whenever `scripts/filter_segments.py`
or `scripts/relabel_segments.py` rewrites `odometry_segments.txt`, so a note
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

While a field there has focus it owns the keyboard. SimView binds space, the
arrows and `p` to `document` and `window` with no useful focus check —
`PlaybackControls` has none at all, and `Controls` tests for an `INPUT`, which
a textarea is not — so a space typed into a note used to toggle playback
instead. Both listen in the bubble phase, so the panel stops key events coming
from a field it holds before they get that far. The scene page carries the same notes as a card, where
they can also be written, edited and deleted; its window table marks every
window a note's frames land in, so a scene shows at a glance where the flagged
stretches are.

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

`validity` is the mean validity of the samples behind a cell, weighted exactly
as its height was, so it reports how measured the surface you are looking at
actually is. Taking the best of them instead reads far more of the map as
measured than the height warrants: a distant frame's map edge does get real
lidar returns and does mark the cell measured, while the near frames that
decided the height were driving past it and extrapolating behind themselves — on
a straight traverse that describes roughly two covered cells in five. The
per-frame Validity tile still shows the three states unmixed.

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
to reach into SimView's own objects rather than just its CSS. Each block says
whether it is permanent (something only this dataset can know: which bodies are
bookkeeping, what the ghost is) or a **stopgap** for an upstream bug. Every
stopgap is gated on the installed SimView version and stops firing by itself on
upgrade — a workaround left running over a fixed release would override the
real fix with a cruder one and hide whether it landed.

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
  an image — and caches the result per scene, keyed by the directory's mtime.
- Scenes are built per window, on demand, and cached on disk under
  `$XDG_CACHE_HOME/dataset_tools_viewer/`.
- One window aggregates a bounded number of clouds and height maps -- chosen
  evenly across the window rather than from its start -- and the terrain is
  coarsened rather than clipped when a long traverse would exceed the cell
  budget. Append a query parameter to the view URL to override a budget for
  that window:

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

## URLs

| Path | What it serves |
| --- | --- |
| `/` | the scene index |
| `/scene/{scene}` | one scene's inventory, window list and calibration |
| `/view/{scene}/{window}` | the 3D view: SimView's page plus the overlay panel |
| `/api/scenes` | the index as JSON |
| `/api/scene/{scene}` | one scene's probe, windows and details as JSON |
| `/api/scene/{scene}/notes` | that scene's notes (`GET`), and where a new one is posted (`POST`) |
| `/api/scene/{scene}/notes/{id}` | edit (`PATCH`) or remove (`DELETE`) one note |
| `/api/scene/{scene}/frame/{n}/image.jpg` | that frame's camera image, downscaled (`?w=`) |
| `/api/scene/{scene}/frame/{n}/heightmap.png` | that frame's BEV height map |
| `/api/scene/{scene}/frame/{n}/validity.png` | that frame's validity mask |
| `/api/scene/{scene}/frame/{n}/cloud.png` | that frame's cloud, top-down |
| `/api/data/{scene}/{window}/{opts}/...` | the SimView static bundle: `model.json`, `states.json`, `blob/N`, plus `meta.json` and `status` |
| `/healthz` | liveness, the root path and the scene count |

A per-frame artefact the scene does not have returns `404` rather than a
placeholder. `meta.json` records what the window was actually built from --
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
serving the old metadata forever.

**The robot is missing from the 3D view.** In SimView <= 4.2 an instanced
mesh's bounding sphere is computed once and cached, so a body is culled as soon
as its *starting* pose leaves the view frustum -- which reads as the robot
blinking out somewhere down a long traverse. The viewer turns culling off for
those bodies (`simview_patch.js`), gated on the SimView version; on a newer
package SimView handles it and the workaround does not run.

**The terrain looks flat, or blockier than 0.1 m.** Check the panel's build
line. Without `height_maps/` the terrain is accumulated from the clouds, and
without clouds either it is flat; and a long traverse is coarsened to stay
inside the cell budget. Cells nothing ever observed are filled from a trend
surface -- the `coverage` colour mode shows which cells were really measured.

**A note will not save.** The scene page's notes card says why: most often the
root is read-only, in which case pass `--notes-dir PATH` to keep notes
somewhere else (they then no longer travel with the dataset) or `--no-notes` to
switch the feature off. `/healthz` reports the same state under `notes`.

**A scene changed on disk and the index still shows the old numbers.** The
probe cache is keyed by each scene directory's mtime and refreshes itself; to
force a full rebuild, delete the cache directory (`--cache-dir`, or the default
under `$XDG_CACHE_HOME/dataset_tools_viewer/`).
