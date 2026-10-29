# Viewer

`helhest-viewer` serves a web view of a dataset root: an index of every scene
and its contents, a page per scene with its validation findings, trajectory
map, and signals over time, a per-frame inspection view, and an interactive 3D
view of each odometry segment showing the terrain, aggregated cloud, robot
poses, and camera. The 3D view is built on
[SimView](https://vlk-jan.github.io/simview/).

The viewer operates in read-only mode on scene data. The only files it writes
to a root are reviewer notes in `<root>/notes/`. It requires only the scene
directories to run: scenes lacking optional files will still open, with missing
files indicated.

Install the viewer as described under [Installation](installation.md#viewer).

## Running it

```bash
helhest-viewer /path/to/root --open                          # serves http://127.0.0.1:8080/
helhest-viewer http://subtdata.felk.cvut.cz/HELHEST/ --open  # a root published on a web server
```

When a root is provided as a URL, files are mirrored into the local cache as
you browse, and reviewer notes default to `<cache-dir>/notes`. The first index
lists every scene's per-frame directories (about 15 s for the 36-scene public
release); a root published with `listings/`, written by
`python -m dataset_tools.viewer.remote /path/to/root` just before publishing,
needs one request per scene for that instead.
`scripts/viewer_public.sh [viewer flags]` starts the viewer on the public
release and opens a browser; `VIEWER_URL` points it at another root.

| Option | Default | Meaning |
| --- | --- | --- |
| `--host`, `--port` | `127.0.0.1`, `8080` | address and port to listen on |
| `--uds PATH` | off | listen on a Unix socket instead, in a directory with restricted permissions; reach it with `ssh -L 8080:PATH host` (`--open` has no effect with it) |
| `--max-seconds`, `--max-frames` | `600`, `12000` | split segments longer than this into equal viewing windows |
| `--cache-dir` | `${XDG_CACHE_HOME:-~/.cache}/dataset_tools_viewer/<root-name>-<hash>` | directory for cached probes, built 3D windows and a URL root's mirror |
| `--cache-max-gb` | `5.0` | size limit for the built 3D windows in gigabytes; least recently used windows are evicted first |
| `--clear-cache` | | empty this root's cache (probes, 3D windows, URL mirror) before starting; reviewer notes are kept |
| `--no-prefetch` | | do not build subsequent windows in the background |
| `--notes-dir PATH` | `<root>/notes`; `<cache-dir>/notes` for a URL root | directory for reviewer notes |
| `--no-notes` | | do not read or write notes |
| `--author NAME` | your login name | author recorded on new notes |
| `--open`, `-v` | | open a browser automatically; enable debug logging |

!!! warning "No authentication"
    Anyone who can reach the port can read the root and write notes. Keep the
    viewer bound to `127.0.0.1` and connect through an SSH tunnel, or pass
    `--no-notes` before binding to `0.0.0.0`. On a machine shared with other
    users, use `--uds`, as local users can access loopback ports.

## On the CTU RCI cluster { #on-the-ctu-cluster }

The viewer can run as a Slurm job on a compute node next to the data, with an SSH
tunnel forwarding the interface to your laptop.

1. On the login node, from the repository root, with `.venv-viewer` built:

    ```bash
    mkdir -p slurm/slurm_out
    sbatch slurm/viewer.batch /mnt/personal/$USER/HELHEST_v2_4    # 8 h
    ```

2. On your laptop:

    ```bash
    scripts/viewer_connect.sh     # [local_port=8080] [job_id]; newest viewer job by default
    ```

    It needs ssh access to the login node and runs from a checkout on your
    laptop. It reads these environment variables:

    | Variable | Default | Meaning |
    | --- | --- | --- |
    | `VIEWER_USER` | `$USER` | your RCI account name |
    | `VIEWER_LOGIN` | `login3.rci.cvut.cz` | login node |
    | `VIEWER_REPO` | `/home/$VIEWER_USER/projects/dataset_tools` | the checkout on the cluster the job was submitted from |
    | `VIEWER_KNOWN_HOSTS` | `~/.ssh/known_hosts_rci_nodes` | known-hosts file for the compute nodes |
    | `VIEWER_COMPRESS` | `1` | compress the tunnel: a 3D window is ~37 MB raw, ~13 MB compressed; `0` is faster on wired gigabit |

    This locates the job's compute node and socket, keeping the tunnel open
    until you press Ctrl-C. The viewer runs until the Slurm job walltime
    expires.

3. Open <http://127.0.0.1:8080/>.

## Pages

| URL | Shows |
| --- | --- |
| `/` | one card per scene, filterable by validation status, note verdict, and split (filters are linkable: `?status=`, `?verdict=`, `?split=`) |
| `/segments` | every ground-truth segment with a viewable window in one table |
| `/notes` | all reviewer notes, downloadable as CSV |
| `/scene/<id>` | ground-truth timeline, file inventory, map, extraction/disk/validation cards, signals, and 3D windows |
| `/scene/<id>/segment/<n>` | one ground-truth segment: robot activity, map, notes, and frames |
| `/scene/<id>/frame/<n>` | image, height map, validity mask, a top-down cloud render, and lidar projection onto the camera image |
| `/view/<scene>/<window>` | interactive 3D view: Space plays and pauses; `[` and `]` step to the previous and next window |

A 3D view never spans two segments because each segment has its own coordinate
origin. The first time a window is opened, its 3D assets are built (2-4 s);
afterwards they load from the cache, and the next window is built in the
background while you look at this one.

## Optional root files

Each of these optional files adds information to the pages when present and is
re-read automatically when modified:

| File | Shows as | Made by |
| --- | --- | --- |
| `validation.csv`, `validation.json` | status chips on the index, scene page Validation card | `helhest-validate` |
| `disk_usage.json` | sizes and disk bars | `helhest-extract`, `helhest-disk-usage` |
| `manifest.json`, `splits.json`, `keyframes*.lst` | dataset version, train/validation split tags and filters | `helhest-release` |
| `atlas/` | index page's curated header and thumbnails | `scripts/make_atlas.py` |
| `listings/` | nothing: a faster first index when the root is opened by URL | `python -m dataset_tools.viewer.remote` |
| `notes/` | reviewer notes on scenes and frame ranges, with verdicts (`keep`, `drop`, `fix`) | the viewer |

The viewer automatically detects re-extracted scenes. The `--clear-cache` flag
is only needed if the cache becomes corrupted.
