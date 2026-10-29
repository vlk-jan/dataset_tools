# Installation

The tools require Python 3.12 or newer. Installation is split into two
environments: one for the extraction tools and one for the viewer. Keep them
separate because the viewer depends on PyTorch (`torch`), which the extraction
tools never use.

## Extraction tools

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

This installs every `helhest-*` command except the viewer. The base
dependencies are `rosbags`, `numpy`, `opencv-python-headless`, and `scipy`.

Optional extras:

| Extra | Adds | Needed for |
| --- | --- | --- |
| `build` | `open3d` | `helhest-odom-frame measure` (part of building a release root). The Linux wheel requires glibc 2.29 or newer. |
| `test` | `pytest`, `httpx` | Running the test suite |
| `viewer` | FastAPI, uvicorn, Jinja2, SimView, torch | `helhest-viewer` (see below) |

## Viewer

```bash
uv venv --python 3.12 .venv-viewer
uv pip install --python .venv-viewer -e ".[viewer]"
```

On Linux, `uv` installs `torch` from the CPU-only index configured in
`pyproject.toml`; on macOS it uses the regular PyPI wheel. Plain `pip` ignores
that configuration and on Linux downloads the CUDA build (about 3 GB); with
`pip`, install the CPU build first:

```bash
python3 -m venv .venv-viewer
.venv-viewer/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv-viewer/bin/pip install -e ".[viewer]"
```

The viewer requires SimView 6.4.0 or newer and refuses to start with earlier
versions.

## Container (CTU RCI cluster) { #container-ctu-cluster }

On the cluster, create the virtual environment using the container's Python
interpreter and run all commands through the container. Network data shares are
not mounted inside the container by default, so always specify `--bind`:

```bash
SIF=/mnt/data/vras/data/helhest-jazzy.sif
singularity exec --bind /mnt/personal --bind /mnt/data $SIF python3 -m venv .venv
singularity exec --bind /mnt/personal --bind /mnt/data $SIF .venv/bin/pip install -e ".[build]"
singularity exec --bind /mnt/personal --bind /mnt/data $SIF \
    .venv/bin/helhest-extract <bags> <root> --jobs 8
```

Add `--bind` for any other filesystem you read from or write to. The extraction,
height-map and finalize jobs in `slurm/` expect a virtual environment built this
way at `<repository>/.venv`; `slurm/viewer.batch` expects the viewer environment
at `<repository>/.venv-viewer`.

## The cloudini decoder

The Odin1 lidar publishes compressed clouds (cloudini). The decoder ships
prebuilt at `dataset_tools/libcloudini_shim.so` and was built on Ubuntu 22.04 and
needs glibc 2.32 or newer (`GLIBC_2.32`, `GLIBCXX_3.4.30`). On older systems,
rebuild it once:

```bash
cloudini_shim/build_shim.sh
```

This requires `cmake` and a C++20-capable `g++`. `build_shim.sh` clones the
cloudini sources from GitHub unless an existing checkout is passed as its first
argument (`cloudini_shim/build_shim.sh <cloudini-checkout>`), so it needs git
and network access otherwise; it uses system zstd/lz4 headers when present,
else builds vendored copies. All prerequisites are available in `helhest-jazzy.sif`,
so if in doubt, run the script inside the container.
