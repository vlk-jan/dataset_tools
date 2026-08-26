# dataset_tools

Tools for extracting the HELHEST dataset from ROS 2 bag files.

## Extraction (`extract_data.py`)

Reads rosbag2 bags (sqlite3 or mcap) with the pure-Python `rosbags` library —
no ROS installation needed — and writes one scene directory per bag:

```
0000/
  images/00000.png ...     # synced camera frames
  clouds/00000.npy ...     # (N, 4) float32 [x y z intensity], base_link frame
  data_ts.txt  pcd_ts.txt  # synced timestamps
  transforms.npy           # base pose per frame, relative to first frame
  gravity_align.npy        # roll/pitch correction per cloud
  camera_intrinsics.txt
  joint_states.npy         # [t, positions..., velocities..., efforts...]
  joint_setpoints.npy      # same layout (if topic present)
  cmd_vel.npy              # [t, vx, vy, vz, wx, wy, wz] (if topic present)
```

### New bags (2026-08 Odin1 + Fixposition setup, e.g. 26-08-21-stromovka-roads)

Defaults target the new sensor stack directly:

```bash
python dataset_tools/extract_data.py <dir-with-bags> <output-dir> --jobs 4
```

which is equivalent to `--cam /odin1/image/compressed
--lidar /odin1/cloud_filtered/cloudini --odom /fixposition/odometry_enu
--imu /fixposition/odometry_enu`.

Notes on the new setup:

- The Odin1 point clouds are **cloudini-compressed**
  (`point_cloud_interfaces/msg/CompressedPointCloud2`). Decoding uses
  `libcloudini_shim.so`; build it once with `cloudini_shim/build_shim.sh`
  (works inside `helhest-jazzy.sif`, needs cmake/g++/libzstd-dev/liblz4-dev).
- The bags' TF tree does **not** connect `odin1_base_link` to `base_link`
  (the Odin1 mount extrinsic was not published). Pass it manually via
  `--lidar_extrinsic X Y Z ROLL PITCH YAW` once known, otherwise identity is
  used with a warning.
- Use `/fixposition/odometry_enu`, not `_smooth`: the smooth variant's child
  frame (`FP_POISH`) is not connected to `base_link` by static TF, so its
  poses cannot be re-expressed at the base.
- Odometry poses and the gravity-alignment orientation are automatically
  re-expressed at `--base_frame` (default `base_link`) via static TF when a
  path exists.
- Don't use `/imu/data` for gravity alignment: that IMU is mounted tilted
  (~5° roll, ~11° pitch) and its frame (`imu`) has no TF to `base_link`.
  The Fixposition orientation (default) gives clean base roll/pitch.
- Known dataset defect (2026-08-21 bags): `/joint_states` contains only
  efforts — positions and velocities are all zero. Commanded wheel velocities
  are in `/joint_setpoints` (saved as `joint_setpoints.npy`).

### Old bags (Livox + Luxonis OAK setup)

```bash
python dataset_tools/extract_data.py <dir-with-bags> <output-dir> \
    --cam /luxonis/oak/rgb/image_raw/compressed --lidar /livox/lidar_front \
    --odom /odom --imu /ekf/imu/data
```

Point cloud field layouts are now parsed dynamically from the PointCloud2
message, so any lidar works.

### Environment

A venv with the extraction deps lives at `.venv` (created with the container's
Python 3.12 — run it through the sif):

```bash
singularity exec /mnt/data/vras/data/helhest-jazzy.sif \
    .venv/bin/python dataset_tools/extract_data.py ...
```

Deps: `rosbags numpy opencv-python-headless networkx tqdm` (+ the cloudini shim).

## Other tools

- `sort_bag.py` — rewrite a bag with messages sorted by header timestamp.
- `odom_node.py` (`ros2 run dataset_tools odom_recorder`) — record an odometry
  topic to a trajectory text file.
