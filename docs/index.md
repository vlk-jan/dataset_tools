# HELHEST dataset_tools

**dataset_tools** converts recordings from the HELHEST robot (ROS 2 bags) into
a flat, machine-learning-ready dataset. Each scene contains synchronised camera
frames, deskewed lidar clouds in the robot's base frame, odometry, gravity
alignment, height maps, and proprioception. The repository also builds
root-level release files and provides a web viewer for inspecting a finished
root.

Bags are read with the pure-Python [`rosbags`](https://gitlab.com/ternaris/rosbags)
library, so **you do not need a ROS installation**.

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -e .                  # Python >= 3.12
.venv/bin/helhest-extract  /path/to/bags /path/to/root --jobs 8     # one scene per bag: 0000/, 0001/, ...
.venv/bin/helhest-validate /path/to/root --csv /path/to/root/validation.csv

uv venv --python 3.12 .venv-viewer && uv pip install --python .venv-viewer -e ".[viewer]"
.venv-viewer/bin/helhest-viewer /path/to/root --open                # http://127.0.0.1:8080/
```

This workflow produces a bare root: scenes containing clouds, images, and
poses. A complete release root also includes height maps, keyframe lists, and a
manifest; see [Building a release root](building.md).

## Where to go next

| I want to... | Go to |
| --- | --- |
| Install the tools in a virtual environment or the CTU cluster container | [Installation](installation.md) |
| Learn what each command does and review its options | [Tools](tools.md) |
| Build a complete release root from a list of bags | [Building a release root](building.md) |
| Understand a root: scenes, files, frames and clocks | [Dataset layout](dataset.md) |
| Browse a root in the web viewer | [Viewer](viewer.md) |

dataset_tools is distributed under the
[BSD 3-Clause License](https://github.com/vlk-jan/dataset_tools/blob/main/LICENSE).
