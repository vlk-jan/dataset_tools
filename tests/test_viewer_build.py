"""Viewer: the wire format and building a SimView scene (dataset_tools.viewer.wire, .build).

Every test builds a synthetic dataset root (``helpers_viewer``), so nothing
here touches a real dataset or the network. The end-to-end HTTP tests skip
cleanly when the viewer extra (SimView + torch) is not installed.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from helpers_viewer import make_root

from dataset_tools.transforms import poses_from_quat
from dataset_tools.viewer import loaders, windows, wire

# --------------------------------------------------------------------------
# wire format
# --------------------------------------------------------------------------


def test_write_static_bundle_lays_out_the_files_the_front_end_fetches(tmp_path):
    pytest.importorskip("simview")
    doc = {"model": {"x": "__b64__AAAA"}, "states": {"version": 4}}
    dest = tmp_path / "bundle"
    info = wire.write_static_bundle(doc, dest)
    assert info == {"blobs": 1, "bytes": 3}
    assert (dest / "model.json").is_file()
    assert (dest / "states.json").is_file()
    assert (dest / "blob" / "0").is_file()
    assert json.loads((dest / "model.json").read_text())["x"] == "/blob/0"
    # Moved into place whole: no staging directory is left beside it.
    assert [p.name for p in tmp_path.iterdir()] == ["bundle"]


def test_render_viewer_page_substitutes_urls_and_injects_the_static_base():
    pytest.importorskip("simview")
    page = wire.render_viewer_page("/simview-static", "/api/data/k", "<div id='x'></div>")
    assert 'window.__simviewStaticBase = "/api/data/k"' in page
    assert "/simview-static/js/main.js" in page
    assert '"./static/' not in page
    # SimView's modules resolve their own assets; the page needs no <base>.
    assert "<base" not in page
    assert "{{" not in page
    assert "<div id='x'></div>" in page


# --------------------------------------------------------------------------
# pose resolution and terrain
# --------------------------------------------------------------------------


def _roll_pitch(quat):
    """Roll and pitch, in degrees, of a [w, x, y, z] quaternion."""
    R = poses_from_quat(np.zeros((1, 3)), np.asarray(quat)[None])[0, :3, :3]
    return np.degrees(np.arctan2(R[2, 1], R[2, 2])), np.degrees(-np.arcsin(np.clip(R[2, 0], -1, 1)))


def _tilt_gravity(root: Path, scene: str, frames: int, roll_deg: float) -> None:
    """Rewrite gravity_align.npy so every frame carries a constant roll."""
    a = np.radians(roll_deg)
    R = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    np.save(root / scene / "gravity_align.npy", np.stack([R] * frames))


def test_a_window_is_levelled_by_the_attitude_its_segment_anchor_discarded(tmp_path):
    """A segment is re-anchored to identity at its own first pose, which zeroes
    the robot's roll and pitch there. Those are absolute, so a segment opening
    on a slope -- or with the robot on its side -- would otherwise render level
    and stay level for its whole length."""
    from dataset_tools.viewer import build
    from dataset_tools.viewer import probe as probe_mod

    root_path = make_root(tmp_path, frames=12)
    _tilt_gravity(root_path, "0000", 12, 30.0)
    access = build.SceneAccess(probe_mod.DatasetRoot(root_path), "0000")
    window = access.window("seg000", 1500)
    frames = np.arange(window.frame_start, window.frame_end)
    track = build.resolve_poses(access, window, access.frame_ts[frames])

    # The anchor pose is the identity; levelled, it stands at the real attitude.
    roll, _ = _roll_pitch(track.quats[0])
    assert roll == pytest.approx(30.0, abs=0.5)


def test_a_single_frame_stands_at_its_own_attitude(tmp_path):
    """A one-frame view is robot-centric, but centred is not the same as level."""
    from dataset_tools.viewer import build
    from dataset_tools.viewer import probe as probe_mod

    root_path = make_root(tmp_path, frames=12)
    _tilt_gravity(root_path, "0000", 12, -20.0)
    access = build.SceneAccess(probe_mod.DatasetRoot(root_path), "0000")
    window = access.window(windows.frame_window_key(3), 1500)
    track = build.resolve_poses(access, window, access.frame_ts[3:4])
    assert np.allclose(track.positions, 0.0)  # still at the origin
    roll, _ = _roll_pitch(track.quats[0])
    assert roll == pytest.approx(-20.0, abs=0.5)


def test_a_scene_without_gravity_alignment_is_left_unlevelled(tmp_path):
    from dataset_tools.viewer import build
    from dataset_tools.viewer import probe as probe_mod

    root_path = make_root(tmp_path, frames=12, with_gravity=False)
    access = build.SceneAccess(probe_mod.DatasetRoot(root_path), "0000")
    window = access.window("seg000", 1500)
    frames = np.arange(window.frame_start, window.frame_end)
    track = build.resolve_poses(access, window, access.frame_ts[frames])
    assert np.allclose(track.quats[0], [1.0, 0.0, 0.0, 0.0], atol=1e-6)


def test_terrain_under_the_robot_comes_from_the_frame_that_drove_over_it(tmp_path):
    """Every cell is covered by dozens of the ~80 maps a window splats. Averaged
    flat, a cell mixes observations made from poses whose z disagrees by however
    much the odometry drifted between them, and the ground drifts away from the
    robot standing on it."""
    from dataset_tools.viewer import build
    from dataset_tools.viewer import probe as probe_mod

    root_path = make_root(tmp_path, frames=2, with_height_maps=True)
    scene = root_path / "0000"
    ts = loaders.read_timestamps(scene / "data_ts.txt")
    # Two frames 6 m apart, each reporting flat ground at a different height.
    with open(scene / "odometry.txt", "w") as fh:
        for i, t in enumerate(ts):
            T = np.eye(4)
            T[0, 3] = 6.0 * i
            fh.write(" ".join(f"{v:.9f}" for v in [t, *T.reshape(-1)]) + "\n")
    with open(scene / "odometry_segments.txt", "w") as fh:
        fh.write(f"0 {ts[0]:.6f} {ts[-1]:.6f} 2 filtered:end_of_run\n")
    for i in range(2):
        np.savez_compressed(scene / "height_maps" / f"{i:05d}.npz", height_map=np.full((128, 128), float(i)))

    access = build.SceneAccess(probe_mod.DatasetRoot(root_path), "0000")
    window = access.window("seg000", 1500)
    frames = np.arange(window.frame_start, window.frame_end)
    track = build.resolve_poses(access, window, access.frame_ts[frames])
    poses = poses_from_quat(track.positions, track.quats)
    opts = build.BuildOptions().sanitized()
    grid = build._make_grid(track.positions, opts)
    build.splat_height_maps(access, grid, frames, poses, opts)
    height, _, _ = build._finish_terrain(grid, None)

    row, col, inside = grid.cell_of(track.positions[:, 0], track.positions[:, 1])
    assert inside.all()
    under = height[row[inside], col[inside]]
    # Both maps reach both robots, so a flat mean would put each on 0.5.
    assert under[0] == pytest.approx(0.0, abs=0.1)
    assert under[1] == pytest.approx(1.0, abs=0.1)


def test_validity_layer_reports_the_samples_that_decided_the_height(tmp_path):
    """A cell any frame measured takes its height from the measurements alone,
    however far off they were taken -- a near frame's extrapolation is invented
    ground and must not flatten a real return. The layer says so."""
    from dataset_tools.viewer import build
    from dataset_tools.viewer import probe as probe_mod

    root_path = make_root(tmp_path, frames=2, with_height_maps=True)
    scene = root_path / "0000"
    ts = loaders.read_timestamps(scene / "data_ts.txt")
    # Two frames 6 m apart. Each stands where the other only reaches with the
    # far edge of its map, worth ~2% of a sample directly underfoot.
    with open(scene / "odometry.txt", "w") as fh:
        for i, t in enumerate(ts):
            T = np.eye(4)
            T[0, 3] = 6.0 * i
            fh.write(" ".join(f"{v:.9f}" for v in [t, *T.reshape(-1)]) + "\n")
    with open(scene / "odometry_segments.txt", "w") as fh:
        fh.write(f"0 {ts[0]:.6f} {ts[-1]:.6f} 2 filtered:end_of_run\n")
    for i in range(2):
        np.savez_compressed(
            scene / "height_maps" / f"{i:05d}.npz", height_map=np.full((128, 128), 1.0 if i == 0 else 0.0)
        )
        # Frame 0 measured everything it saw; frame 1 extrapolated everything.
        np.savez_compressed(
            scene / "height_masks" / f"{i:05d}.npz",
            validity=np.full((128, 128), 2 if i == 0 else 0, dtype=np.uint8),
        )

    access = build.SceneAccess(probe_mod.DatasetRoot(root_path), "0000")
    window = access.window("seg000", 1500)
    frames = np.arange(window.frame_start, window.frame_end)
    track = build.resolve_poses(access, window, access.frame_ts[frames])
    poses = poses_from_quat(track.positions, track.quats)
    opts = build.BuildOptions().sanitized()
    grid = build._make_grid(track.positions, opts)
    build.splat_height_maps(access, grid, frames, poses, opts)
    height, layers, bounds = build._finish_terrain(grid, None)

    row, col, inside = grid.cell_of(track.positions[:, 0], track.positions[:, 1])
    under = layers["validity"][row[inside], col[inside]]
    assert bounds["validity"] == (0.0, 2.0)
    assert under[0] == pytest.approx(2.0, abs=0.1)
    # Only the distant frame measured it, and that measurement decides it.
    assert under[1] == pytest.approx(2.0, abs=0.1)
    # ... at full height: the near frame's invented ground would pull it to ~0.
    assert height[row[inside], col[inside]][1] == pytest.approx(1.0, abs=0.01)


def test_wheel_anchored_cells_do_not_read_as_measured(tmp_path):
    """A root built with DRIFT's --anchors codes wheel-contact cells 3. That is
    a provenance, not a better 2: the lidar never saw those cells. Averaging
    the raw code would paint the blind region under the robot as the
    best-measured terrain in the scene, and off the top of the legend's scale."""
    from dataset_tools.viewer import build
    from dataset_tools.viewer import probe as probe_mod

    root_path = make_root(tmp_path, frames=2, with_height_maps=True)
    scene = root_path / "0000"
    ts = loaders.read_timestamps(scene / "data_ts.txt")
    with open(scene / "odometry.txt", "w") as fh:
        for i, t in enumerate(ts):
            T = np.eye(4)
            T[0, 3] = 6.0 * i
            fh.write(" ".join(f"{v:.9f}" for v in [t, *T.reshape(-1)]) + "\n")
    with open(scene / "odometry_segments.txt", "w") as fh:
        fh.write(f"0 {ts[0]:.6f} {ts[-1]:.6f} 2 filtered:end_of_run\n")
    for i in range(2):
        np.savez_compressed(scene / "height_maps" / f"{i:05d}.npz", height_map=np.zeros((128, 128)))
        np.savez_compressed(
            scene / "height_masks" / f"{i:05d}.npz",
            validity=np.full((128, 128), 3, dtype=np.uint8),
        )

    access = build.SceneAccess(probe_mod.DatasetRoot(root_path), "0000")
    window = access.window("seg000", 1500)
    frames = np.arange(window.frame_start, window.frame_end)
    track = build.resolve_poses(access, window, access.frame_ts[frames])
    poses = poses_from_quat(track.positions, track.quats)
    opts = build.BuildOptions().sanitized()
    grid = build._make_grid(track.positions, opts)
    build.splat_height_maps(access, grid, frames, poses, opts)
    _, layers, _ = build._finish_terrain(grid, None)

    # An all-anchored scene carries no validity information at all, so the
    # layer is uniformly 0 -- and _finish_terrain drops uniform layers.
    assert "validity" not in layers or np.allclose(layers["validity"], 0.0)
    assert loaders.VALIDITY_RANK[3] == 0
    assert loaders.VALIDITY_LABELS[3] == "anchored"


def test_validity_rank_is_the_identity_on_every_published_root(tmp_path):
    """Existing masks only ever hold 0/1/2. Ranking must not move them."""
    assert list(loaders.VALIDITY_RANK[:3]) == [0, 1, 2]


def test_a_nan_in_a_height_map_does_not_poison_the_terrain(tmp_path):
    """The generator is free to leave a cell NaN. Splatted, it would sit in the
    weighted sums for good -- the cell counts as covered, so nothing fills it,
    and a NaN reaches the heightfield -- and it smears into world x and y too,
    where the cast to a cell index is undefined."""
    from dataset_tools.viewer import build
    from dataset_tools.viewer import probe as probe_mod

    root_path = make_root(tmp_path, frames=2, with_height_maps=True)
    scene = root_path / "0000"
    ts = loaders.read_timestamps(scene / "data_ts.txt")
    with open(scene / "odometry.txt", "w") as fh:
        for i, t in enumerate(ts):
            T = np.eye(4)
            T[0, 3] = 6.0 * i
            fh.write(" ".join(f"{v:.9f}" for v in [t, *T.reshape(-1)]) + "\n")
    with open(scene / "odometry_segments.txt", "w") as fh:
        fh.write(f"0 {ts[0]:.6f} {ts[-1]:.6f} 2 filtered:end_of_run\n")
    for i in range(2):
        hm = np.full((128, 128), float(i))
        hm[100:120, 100:120] = np.nan  # ahead and to the right of the robot
        np.savez_compressed(scene / "height_maps" / f"{i:05d}.npz", height_map=hm)

    access = build.SceneAccess(probe_mod.DatasetRoot(root_path), "0000")
    window = access.window("seg000", 1500)
    frames = np.arange(window.frame_start, window.frame_end)
    track = build.resolve_poses(access, window, access.frame_ts[frames])
    poses = poses_from_quat(track.positions, track.quats)
    opts = build.BuildOptions().sanitized()
    grid = build._make_grid(track.positions, opts)
    with warnings.catch_warnings():
        # The undefined cast announces itself; the guard has to run before it.
        warnings.simplefilter("error", RuntimeWarning)
        stats = build.splat_height_maps(access, grid, frames, poses, opts)
    height, layers, _ = build._finish_terrain(grid, None)

    assert stats["used"] == 2
    assert np.isfinite(height).all()
    # The NaN cells are simply unobserved, not observed-as-nothing: they fill
    # from the trend around them and read as zero coverage.
    assert layers["coverage"].min() == 0.0
    row, col, inside = grid.cell_of(track.positions[:, 0], track.positions[:, 1])
    under = height[row[inside], col[inside]]
    assert under[0] == pytest.approx(0.0, abs=0.1)
    assert under[1] == pytest.approx(1.0, abs=0.1)


def _rewrite_odometry(scene: Path, roll_step_deg: float) -> None:
    """Straight-line odometry whose roll grows by a fixed step per frame."""
    ts = loaders.read_timestamps(scene / "data_ts.txt")
    with open(scene / "odometry.txt", "w") as fh:
        for i, t in enumerate(ts):
            a = np.radians(roll_step_deg * i)
            T = np.eye(4)
            T[:3, :3] = [[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]
            T[0, 3] = 0.3 * i
            fh.write(" ".join(f"{v:.9f}" for v in [t, *T.reshape(-1)]) + "\n")


def _attitude_of(root_path: Path, scene: str = "0000") -> dict | None:
    from dataset_tools.viewer import build
    from dataset_tools.viewer import probe as probe_mod

    access = build.SceneAccess(probe_mod.DatasetRoot(root_path), scene)
    window = access.window("seg000", 1500)
    frames = np.arange(window.frame_start, window.frame_end)
    times = access.frame_ts[frames]
    gt = build.resolve_poses(access, window, times)
    scalars, _ = build.build_scalars(access, window, times, gt, None)
    return build.attitude_vs_imu(scalars)


def test_the_attitude_comparison_is_silent_when_the_ground_truth_agrees(tmp_path):
    """A levelled window puts both attitudes in the same frame, so ground truth
    that matches the IMU has to score zero -- otherwise the check cries wolf on
    every sound segment."""
    root_path = make_root(tmp_path, frames=12)
    _tilt_gravity(root_path, "0000", 12, 20.0)
    _rewrite_odometry(root_path / "0000", roll_step_deg=0.0)
    attitude = _attitude_of(root_path)
    assert attitude["median_error"] == pytest.approx(0.0, abs=0.2)
    assert attitude["gt_roll"][0] == pytest.approx(attitude["imu_roll"][0], abs=0.2)


def test_the_attitude_comparison_catches_ground_truth_that_rolls_away(tmp_path):
    """The failure this exists for: the odometry's own orientation wandering off
    while the IMU, which does not depend on it, holds still."""
    root_path = make_root(tmp_path, frames=12)
    _tilt_gravity(root_path, "0000", 12, 20.0)
    _rewrite_odometry(root_path / "0000", roll_step_deg=5.0)
    attitude = _attitude_of(root_path)
    assert attitude["median_error"] > 5.0
    assert attitude["error"][-1] > attitude["error"][0]
    # The IMU is unmoved throughout; only the ground truth walks.
    assert attitude["imu_roll"][0] == pytest.approx(attitude["imu_roll"][-1], abs=0.2)


def test_a_scene_without_gravity_alignment_has_nothing_to_compare_against(tmp_path):
    assert _attitude_of(make_root(tmp_path, frames=12, with_gravity=False)) is None


# --------------------------------------------------------------------------
# sampling budgets against window length
# --------------------------------------------------------------------------


def test_a_short_window_gets_exactly_the_budgets_it_asked_for():
    """Most windows are short -- the median segment of HELHEST v1.3 is 43
    frames -- and nothing about them should change."""
    from dataset_tools.viewer.build import BuildOptions, budgets_for_window

    opts = BuildOptions()
    assert budgets_for_window(opts, 43) is opts
    assert budgets_for_window(opts, 1500) is opts


def test_a_long_window_samples_it_as_densely_as_a_short_one():
    """The budgets are fixed counts, so a window twice as long would otherwise
    splat half as many height maps per second driven and show a visibly poorer
    terrain than the two windows it replaced."""
    from dataset_tools.viewer.build import BuildOptions, budgets_for_window

    opts = BuildOptions()
    long = budgets_for_window(opts, 3000)
    assert long.max_hm_frames == 2 * opts.max_hm_frames
    assert long.max_cloud_frames == 2 * opts.max_cloud_frames
    assert long.max_cloud_points == 2 * opts.max_cloud_points


def test_the_budgets_stop_at_the_ceilings_the_ui_offers():
    """A half-hour segment is twelve times the reference length; the scaling is
    bounded so its window stops where the option sliders do rather than at
    whatever the arithmetic produced."""
    from dataset_tools.viewer.build import BuildOptions, budgets_for_window

    huge = budgets_for_window(BuildOptions(), 17_443)
    assert huge == budgets_for_window(BuildOptions(), 6000)
    assert huge.sanitized() == huge
    # Cells are what the bundle is mostly made of, so they scale more slowly
    # than the samples do.
    assert huge.max_terrain_cells == 2 * BuildOptions().max_terrain_cells
    assert huge.max_hm_frames == 4 * BuildOptions().max_hm_frames


def test_a_long_window_keeps_the_terrain_resolution_its_parts_had(tmp_path):
    """The one thing merging two windows really costs: a longer traverse needs
    more cells at the same resolution, and without a bigger budget ``_make_grid``
    coarsens instead (0.1 m to 0.2 m across a 163 m drive of v1.3's 0020/0)."""
    from dataset_tools.viewer.build import BuildOptions, _make_grid, budgets_for_window

    # A 160 m x 30 m traverse: 766k cells at 0.1 m, over the default budget and
    # inside the scaled one, which is where v1.3's merged windows land.
    positions = np.stack(
        [np.linspace(0, 160, 400), np.linspace(0, 30, 400), np.zeros(400)],
        axis=1,
    )
    opts = BuildOptions()
    assert _make_grid(positions, opts.sanitized()).res > opts.terrain_res
    assert _make_grid(positions, budgets_for_window(opts, 3000)).res == opts.terrain_res


def test_wheel_slip_is_the_commanded_speed_minus_the_measured_one(tmp_path):
    """joint_setpoints beside joint_states: the difference is where slip and
    load show up, so it is a plot of its own rather than two to read against
    each other."""
    from dataset_tools.viewer import build
    from dataset_tools.viewer import probe as probe_mod

    root_path = make_root(tmp_path, frames=12)
    scene = root_path / "0000"
    setpoints = np.load(scene / "joint_states.npy").copy()
    setpoints[:, 4:7] = 2.0  # commanded, against the 1.5 rad/s the wheels did
    np.save(scene / "joint_setpoints.npy", setpoints)

    access = build.SceneAccess(probe_mod.DatasetRoot(root_path), "0000")
    window = access.window("seg000", 1500)
    times = access.frame_ts[np.arange(window.frame_start, window.frame_end)]
    gt = build.resolve_poses(access, window, times)
    scalars, _ = build.build_scalars(access, window, times, gt, None)
    for label in build.WHEEL_LABELS:
        assert scalars[f"wheel_slip_{label}"] == pytest.approx(0.5, abs=1e-6)


# --------------------------------------------------------------------------
# the window tables' verdict, without a build
# --------------------------------------------------------------------------


def _window_attitude_of(root_path: Path, key: str = "seg000") -> dict | None:
    from dataset_tools.viewer import build
    from dataset_tools.viewer import probe as probe_mod

    access = build.SceneAccess(probe_mod.DatasetRoot(root_path), "0000")
    return build.window_attitude(access, access.window(key, 1500))


def test_the_window_verdict_is_ok_when_the_ground_truth_tracks_the_imu(tmp_path):
    root_path = make_root(tmp_path, frames=12)
    _tilt_gravity(root_path, "0000", 12, 20.0)
    _rewrite_odometry(root_path / "0000", roll_step_deg=0.0)
    verdict = _window_attitude_of(root_path)
    assert verdict["median"] == pytest.approx(0.0, abs=0.2)
    assert verdict["band"] == "ok"


def test_the_window_verdict_is_bad_when_the_ground_truth_rolls_away(tmp_path):
    root_path = make_root(tmp_path, frames=12)
    _tilt_gravity(root_path, "0000", 12, 30.0)
    _rewrite_odometry(root_path / "0000", roll_step_deg=12.0)
    verdict = _window_attitude_of(root_path)
    assert verdict["median"] > 20.0
    assert verdict["band"] == "bad"


def test_the_window_verdict_is_the_panels_number(tmp_path):
    """One comparison, two places: the table must promise what the panel shows."""
    root_path = make_root(tmp_path, frames=12)
    _tilt_gravity(root_path, "0000", 12, 20.0)
    _rewrite_odometry(root_path / "0000", roll_step_deg=5.0)
    panel = _attitude_of(root_path)
    table = _window_attitude_of(root_path)
    assert table["median"] == pytest.approx(panel["median_error"], abs=1e-6)
    assert table["band"] == panel["band"]


def test_there_is_no_window_verdict_without_gravity_alignment_or_ground_truth(tmp_path):
    assert _window_attitude_of(make_root(tmp_path, frames=12, with_gravity=False)) is None
    (tmp_path / "raw").mkdir()
    assert _window_attitude_of(make_root(tmp_path / "raw", frames=12, segments=False), "raw") is None
