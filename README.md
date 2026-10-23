# dataset_tools

Tools for turning **HELHEST** robot recordings (ROS 2 bags) into a flat,
learning-friendly dataset: synchronised camera frames, deskewed lidar clouds in
the robot base frame, odometry, gravity alignment, height maps and
proprioception, with a validator and a web viewer for the result.

Bags are read with the pure-Python [`rosbags`](https://gitlab.com/ternaris/rosbags)
library, so **no ROS installation is required**.

**Documentation: <https://vlk-jan.github.io/dataset_tools/>**

| Page | What it covers |
| --- | --- |
| [Installation](https://vlk-jan.github.io/dataset_tools/installation/) | the extraction and viewer virtualenvs, the cluster container, the cloudini decoder |
| [Tools](https://vlk-jan.github.io/dataset_tools/tools/) | every `helhest-*` command and its options |
| [Building a release root](https://vlk-jan.github.io/dataset_tools/building/) | bag list and spec, the Slurm jobs, the same without Slurm |
| [Dataset layout](https://vlk-jan.github.io/dataset_tools/dataset/) | a root's scenes and files, frames, ground truth, clocks, the recordings |
| [Viewer](https://vlk-jan.github.io/dataset_tools/viewer/) | browsing a root locally, on the cluster, or from a web server |

The site is built from [`docs/`](docs/) (`mkdocs.yml`); [docs/README.md](docs/README.md)
lists the other documents kept there.

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -e .                  # extraction; Python >= 3.12
.venv/bin/helhest-extract  /path/to/bags /path/to/root --jobs 8      # one scene per bag: 0000/, 0001/, ...
.venv/bin/helhest-validate /path/to/root --csv /path/to/root/validation.csv

uv venv --python 3.12 .venv-viewer && uv pip install --python .venv-viewer -e ".[viewer]"
.venv-viewer/bin/helhest-viewer /path/to/root --open                 # http://127.0.0.1:8080/
```

That gives a bare root (clouds, images, poses). A release root also carries
the odometry-frame rotation, height maps, keyframe lists and a manifest; see
[Building a release root](https://vlk-jan.github.io/dataset_tools/building/).

## Commands

| Module | Entry point | What it does |
| --- | --- | --- |
| `dataset_tools/extract/` | `helhest-extract` | bags -> scene directories |
| `dataset_tools/odom_frame.py` | `helhest-odom-frame` | rotate the Odin1 odometry onto its clouds' frame (needs the `build` extra) |
| `dataset_tools/heightmap.py` | `helhest-heightmap` | height maps and validity masks from a scene's clouds |
| `dataset_tools/keyframes.py` | `helhest-keyframes` | the root's keyframe lists (all / train / held out) |
| `dataset_tools/release.py` | `helhest-release` | a release's root-level files from its spec in `docs/releases/` |
| `dataset_tools/validate_scene.py` | `helhest-validate` | check extracted scenes and whole roots |
| `dataset_tools/viewer/` | `helhest-viewer` | browse a root in a web browser |
| `dataset_tools/disk_usage.py` | `helhest-disk-usage` | bytes per scene and modality, for the viewer |
| `dataset_tools/image_format.py` | `helhest-image-format` | declare or check a root's image format |
| `dataset_tools/sort_bag.py` | `helhest-sort-bag` | rewrite a bag sorted by header stamp |

[`slurm/`](slurm/README.md) holds the CTU RCI cluster jobs the releases were
built with, [`scripts/`](scripts/README.md) the tools that are not part of the
package (the viewer's atlas, clock probes, viewer launchers).

Code comments refer to **DRIFT**: the downstream training code the roots are
extracted for (a separate repository). It adds its own per-frame products to a
root (`proprio_labels/`, `driven_similarity/`); everything else in a root is
this repository's.

## Tests

No bags are needed; everything runs on synthetic data.

```bash
.venv/bin/pip install -e ".[test]"
.venv/bin/python -m pytest tests
```

`.github/workflows/ci.yml` runs the lint (`black`, `ruff`, pinned), the
extraction tests and the viewer tests; `.github/workflows/docs.yml` builds and
publishes the documentation site. [CONTRIBUTING.md](CONTRIBUTING.md) covers the
two virtualenvs and the lint.

## License

BSD-3-Clause, see [LICENSE](LICENSE). Copyright (c) VRAS (Vision for Robotics
and Autonomous Systems), Czech Technical University in Prague. To cite the
software, see [CITATION.cff](CITATION.cff).
