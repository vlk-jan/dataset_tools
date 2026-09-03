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
| `--open` | open a browser at the index page |
| `-v` | debug logging |

The dataset root is only ever **read** -- every byte the viewer writes goes to
the cache directory -- so it is safe to point at a read-only or shared copy.

Start at the index, click a scene id for its inventory and window list, then a
window to open the 3D view. Nothing else needs configuring.

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
| `gravity_align.npy` | `roll_deg` / `pitch_deg` plots; places each height map level in the world |
| `height_maps/` | the terrain surface, and a per-frame tile in the side panel |
| `height_masks/` | the terrain's `validity` colour layer, and its own panel tile |
| `clouds/` | an aggregated point-cloud body, three terrain colour layers, and a per-frame top-down tile |
| `images/` | the side panel's camera tile, served per frame |
| `joint_states.npy` | wheel spin animation + velocity/effort plots |
| `joint_setpoints.npy` | commanded wheel-velocity plots |
| `cmd_vel.npy` | `cmd_linear_x` / `cmd_angular_z` plots |
| `camera_intrinsics.txt`, `T_cam_to_base.txt` | shown on the scene page; a small camera body at the real extrinsic |

The terrain offers several colour modes: `height`, plus `coverage` (how many
height maps actually observed each cell — cells nothing observed are filled from
a trend surface and show as zero coverage), `validity`, and `cloud_points` /
`cloud_intensity` / `cloud_spread` from the aggregated cloud. What each one
means travels with the window (`build.TERRAIN_COLOR_MODE_HELP` → `meta.json` →
the panel's **terrain colour modes** legend), so the names in SimView's own
"Color Mode" dropdown are explained next to it rather than left to be guessed.

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

**A scene changed on disk and the index still shows the old numbers.** The
probe cache is keyed by each scene directory's mtime and refreshes itself; to
force a full rebuild, delete the cache directory (`--cache-dir`, or the default
under `$XDG_CACHE_HOME/dataset_tools_viewer/`).
