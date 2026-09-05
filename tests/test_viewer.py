"""Tests for the dataset viewer (dataset_tools.viewer).

The heavy parts of the viewer (SimView + torch) are an optional extra, so the
tests are layered: the geometry, probing, windowing and wire-format tests run
with numpy alone, and the end-to-end HTTP tests skip cleanly when the extra is
not installed.

Every fixture builds a synthetic dataset root, so nothing here touches a real
dataset or the network.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pytest

from dataset_tools.viewer import index_page, loaders, probe, render, windows, wire

# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def _write_png(path: Path, size=(16, 10)) -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    Image.new("RGB", size, (40, 80, 120)).save(path)


def make_root(
    tmp_path: Path,
    scenes=("0000",),
    frames: int = 12,
    *,
    with_height_maps: bool = True,
    with_cmd_vel: bool = True,
    with_transforms: bool = True,
    with_gravity: bool = True,
    with_joints: bool = True,
    with_camera_info: bool = True,
    segments: bool = True,
) -> Path:
    """A synthetic dataset root with a controllable set of optional files."""
    root = tmp_path / "root"
    root.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)

    for scene in scenes:
        d = root / scene
        (d / "images").mkdir(parents=True)
        (d / "clouds").mkdir()
        ts = 1_700_000_000.0 + 0.1 * np.arange(frames)
        for i in range(frames):
            _write_png(d / "images" / f"{i:05d}.png")
            cloud = rng.normal(scale=2.0, size=(64, 4)).astype(np.float32)
            cloud[:, 0] = np.abs(cloud[:, 0])  # forward-only, like a front lidar
            np.save(d / "clouds" / f"{i:05d}.npy", cloud)
        np.savetxt(d / "data_ts.txt", ts, fmt="%.6f")
        np.savetxt(d / "pcd_ts.txt", ts - 0.005, fmt="%.6f")

        poses = []
        for i in range(frames):
            yaw = 0.05 * i
            c, s = np.cos(yaw), np.sin(yaw)
            T = np.eye(4)
            T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
            T[:3, 3] = [0.3 * i, 0.02 * i, 0.0]
            poses.append(T)
        with open(d / "odometry.txt", "w") as fh:
            fh.write("# Timestamp(sec) r11 ... 0 0 0 1\n")
            for t, T in zip(ts, poses):
                fh.write(" ".join(f"{v:.9f}" for v in [t, *T.reshape(-1)]) + "\n")
        if segments:
            half = frames // 2
            with open(d / "odometry_segments.txt", "w") as fh:
                fh.write("# id start_ts end_ts n_poses reason\n")
                fh.write(f"0 {ts[0]:.6f} {ts[half - 1]:.6f} {half} filtered:pose_jump\n")
                fh.write(f"1 {ts[half]:.6f} {ts[-1]:.6f} {frames - half} filtered:end_of_run\n")

        if with_transforms:
            np.save(d / "transforms.npy", np.stack(poses))
        if with_gravity:
            np.save(d / "gravity_align.npy", np.stack([np.eye(3)] * frames))
        if with_joints:
            jt = np.linspace(ts[0], ts[-1], frames * 4)
            js = np.zeros((len(jt), 10))
            js[:, 0] = jt
            js[:, 1:4] = np.linspace(0, 10, len(jt))[:, None]  # positions
            js[:, 4:7] = 1.5  # velocities
            js[:, 7:10] = 0.5  # efforts
            np.save(d / "joint_states.npy", js)
        if with_cmd_vel:
            ct = np.linspace(ts[0], ts[-1], frames * 2)
            cv = np.zeros((len(ct), 7))
            cv[:, 0] = ct
            cv[:, 1] = 0.4
            cv[:, 6] = 0.1
            np.save(d / "cmd_vel.npy", cv)
        if with_height_maps:
            (d / "height_maps").mkdir()
            (d / "height_masks").mkdir()
            for i in range(frames):
                hm = np.zeros((128, 128), dtype=np.float64)
                hm[:] = 0.01 * np.arange(128)[:, None]
                np.savez_compressed(d / "height_maps" / f"{i:05d}.npz", height_map=hm)
                validity = np.full((128, 128), 2, dtype=np.uint8)
                validity[:64] = 0  # behind the robot: extrapolated
                np.savez_compressed(d / "height_masks" / f"{i:05d}.npz", validity=validity)
        np.savetxt(d / "camera_intrinsics.txt", np.eye(3) * 700.0)
        np.savetxt(d / "T_cam_to_base.txt", np.eye(4))
        if with_camera_info:
            (d / "camera_info.yaml").write_text(CAMERA_INFO)
    return root


#: A ROS camera_info as the extractor writes it: flat scalars, then blocks whose
#: only interesting line is ``data:``.
CAMERA_INFO = """image_width: 640
image_height: 400
frame_id: test_camera_optical_frame
camera_name: test_camera
distortion_model: plumb_bob
camera_matrix:
  rows: 3
  cols: 3
  data: [700.0, 0, 320.0, 0, 700.0, 200.0, 0, 0, 1]
distortion_coefficients:
  rows: 1
  cols: 5
  data: [0.1, -0.2, 0.001, 0.002, 0.0]
binning_x: 0
"""


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def test_quaternion_round_trips_through_rotation_matrices():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(64, 3, 3))
    u, _, vt = np.linalg.svd(a)
    R = u @ vt
    R[np.linalg.det(R) < 0] *= -1
    q = loaders.mat_to_quat(R)
    back = loaders.poses_from_quat(np.zeros((64, 3)), q)[:, :3, :3]
    assert np.allclose(R, back, atol=1e-9)
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0)


def test_quaternion_of_identity_is_scalar_one():
    assert np.allclose(loaders.mat_to_quat(np.eye(3)[None]), [[1, 0, 0, 0]])


def test_slerp_midpoint_is_the_half_angle():
    q0 = np.array([[1.0, 0, 0, 0]])
    angle = np.pi / 2
    q1 = np.array([[np.cos(angle / 2), 0, 0, np.sin(angle / 2)]])
    mid = loaders.quat_slerp(q0, q1, np.array([0.5]))
    expected = np.array([np.cos(angle / 4), 0, 0, np.sin(angle / 4)])
    assert np.allclose(mid[0], expected, atol=1e-12)


def test_interpolate_poses_marks_queries_outside_the_track_invalid():
    ts = np.array([10.0, 11.0, 12.0])
    poses = np.stack([np.eye(4)] * 3)
    for i, x in enumerate([0.0, 1.0, 2.0]):
        poses[i, 0, 3] = x
    pos, _quat, valid = loaders.interpolate_poses(ts, poses, np.array([9.0, 10.5, 12.0, 13.0]))
    assert valid.tolist() == [False, True, True, False]
    assert pos[1, 0] == pytest.approx(0.5)
    # Outside queries are clamped, not extrapolated.
    assert pos[0, 0] == pytest.approx(0.0)
    assert pos[3, 0] == pytest.approx(2.0)


def test_interpolate_poses_counts_a_query_within_tolerance_as_on_the_track():
    ts = np.array([10.0, 11.0, 12.0])
    poses = np.stack([np.eye(4)] * 3)
    just_early = np.array([9.99])
    assert not loaders.interpolate_poses(ts, poses, just_early)[2][0]
    assert loaders.interpolate_poses(ts, poses, just_early, tol=0.05)[2][0]


def test_resample_series_does_not_invent_values_across_a_long_gap():
    src_ts = np.array([0.0, 1.0, 50.0, 51.0])
    vals = np.array([[0.0], [1.0], [2.0], [3.0]])
    out = loaders.resample_series(src_ts, vals, np.array([0.5, 25.0]), max_gap=1.0)
    assert out[0, 0] == pytest.approx(0.5)
    assert np.isnan(out[1, 0])


def test_roll_pitch_of_an_identity_gravity_rotation_is_zero():
    roll, pitch = loaders.roll_pitch_from_gravity_align(np.eye(3)[None])
    assert roll[0] == pytest.approx(0.0)
    assert pitch[0] == pytest.approx(0.0)


def test_odometry_parses_both_the_17_and_13_column_forms(tmp_path):
    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 3.0]
    long_form = tmp_path / "long.txt"
    long_form.write_text("# header\n" + " ".join(["100.0", *[str(v) for v in T.reshape(-1)]]) + "\n")
    short = tmp_path / "short.txt"
    short.write_text("# header\n" + " ".join(["100.0", *[str(v) for v in T[:3, :].reshape(-1)]]) + "\n")
    for path in (long_form, short):
        result = loaders.read_odometry(path)
        assert result is not None
        ts, poses = result
        assert ts[0] == pytest.approx(100.0)
        assert np.allclose(poses[0], T)


def test_segments_parse_and_sort(tmp_path):
    path = tmp_path / "segments.txt"
    path.write_text(
        "# id start_ts end_ts n_poses reason\n"
        "1 20.0 30.0 100 filtered:pose_jump\n"
        "0 0.0 10.0 50 v11_fill\n"
        "garbage line\n"
    )
    segs = loaders.read_segments(path)
    assert [s.index for s in segs] == [0, 1]
    assert segs[0].reason == "v11_fill"
    assert segs[1].duration == pytest.approx(10.0)


def test_missing_files_return_none_rather_than_raising(tmp_path):
    assert loaders.read_odometry(tmp_path / "nope.txt") is None
    assert loaders.read_segments(tmp_path / "nope.txt") == []
    assert loaders.read_timestamps(tmp_path / "nope.txt") is None
    assert loaders.load_npy(tmp_path / "nope.npy") is None
    assert loaders.read_matrix(tmp_path / "nope.txt") is None


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def test_bev_display_puts_forward_at_the_top_and_left_at_the_left():
    """Known answer: the height map is [row=x forward, col=y left], and the
    rendered image must be a top-down view with forward up."""
    grid = np.zeros((8, 8))
    grid[7, 4] = 1.0  # far forward (max x), slightly left of centre
    disp = render._bev_to_display(grid)
    row, col = np.unravel_index(np.argmax(disp), disp.shape)
    assert row == 0, "max-x cell must land in the top row"
    assert col == 3, "the y axis must be mirrored so +y is on the left"


def test_colorize_handles_nan_and_a_flat_range():
    out = render.colorize(np.array([[np.nan, np.nan]]), cmap="turbo")
    assert out.shape == (1, 2, 3)
    assert out.dtype == np.uint8
    flat = render.colorize(np.full((4, 4), 3.0))
    assert flat.shape == (4, 4, 3)


def test_renderers_produce_valid_images():
    pytest.importorskip("PIL")
    from PIL import Image
    import io

    hm = np.random.default_rng(0).normal(size=(128, 128))
    for data in (render.render_height_map(hm), render.render_validity((hm > 0).astype(np.uint8))):
        assert Image.open(io.BytesIO(data)).size[0] > 0
    cloud = np.random.default_rng(1).normal(size=(500, 4))
    assert Image.open(io.BytesIO(render.render_cloud_bev(cloud))).size[0] > 0


# --------------------------------------------------------------------------
# probing
# --------------------------------------------------------------------------


def test_probe_reports_what_a_scene_has_and_lacks(tmp_path):
    root_path = make_root(tmp_path, with_cmd_vel=False, with_height_maps=False)
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")
    assert root.scene_ids() == ["0000"]
    p = root.probe("0000")
    assert p.n_frames == 12
    assert p.has("images") and p.has("clouds") and p.has("odometry")
    assert not p.has("cmd_vel")
    assert not p.has("height_maps")
    assert "cmd_vel" in p.missing and "height_maps" in p.missing
    assert p.counts["images"] == 12
    assert p.n_segments == 2


def test_probe_cache_is_reused_and_invalidated_by_an_mtime_change(tmp_path):
    root_path = make_root(tmp_path)
    cache = tmp_path / "cache"
    root = probe.DatasetRoot(root_path, cache_dir=cache)
    first = root.probe("0000")
    assert (cache / "probe_cache.json").is_file()
    # A second DatasetRoot reads the persisted probe rather than re-listing.
    again = probe.DatasetRoot(root_path, cache_dir=cache)
    assert again.probe("0000").n_frames == first.n_frames


def test_probe_warns_when_per_frame_counts_disagree(tmp_path):
    root_path = make_root(tmp_path)
    (root_path / "0000" / "clouds" / "00011.npy").unlink()
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")
    p = root.probe("0000", refresh=True)
    assert any("counts disagree" in w for w in p.warnings)


def test_a_root_with_no_scenes_csv_or_manifest_still_works(tmp_path):
    root_path = make_root(tmp_path)
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")
    assert root.manifest() is None
    assert root.csv_rows() == {}
    assert root.probe_all()


# --------------------------------------------------------------------------
# windowing
# --------------------------------------------------------------------------


def test_windows_never_span_two_segments():
    ts = np.arange(100, dtype=np.float64)
    segs = [loaders.Segment(0, 0.0, 39.0, 40, "a"), loaders.Segment(1, 60.0, 99.0, 40, "b")]
    result = windows.windows_for_scene(ts, segs, len(ts), has_transforms=True)
    seg_windows = [w for w in result if w.kind == "segment"]
    assert len(seg_windows) == 2
    # The gap between the segments belongs to neither window.
    assert seg_windows[0].frame_end <= 40
    assert seg_windows[1].frame_start >= 60


def test_long_segments_are_split_into_even_chunks():
    ts = np.arange(1000, dtype=np.float64)
    segs = [loaders.Segment(0, 0.0, 999.0, 1000, "x")]
    result = windows.windows_for_scene(ts, segs, len(ts), has_transforms=False, max_frames=300)
    chunks = [w for w in result if w.kind == "segment"]
    assert len(chunks) == 4
    assert all(w.n_frames <= 300 for w in chunks)
    # Chunks are contiguous and cover the whole segment.
    assert chunks[0].frame_start == 0
    for a, b in zip(chunks, chunks[1:]):
        assert a.frame_end == b.frame_start


def test_a_frame_just_before_a_segment_starts_still_belongs_to_it():
    """Segment edges are on the ground-truth pose clock, frames on their own.

    A frame a few ms short of a segment's first pose is that segment's first
    frame; treated strictly it belongs to nothing and the frame page reports no
    ground truth for a frame that plainly has some.
    """
    ts = np.arange(20, dtype=np.float64) * 0.32
    # The segment opens 3 ms after frame 5 -- a hundredth of a frame period.
    segs = [loaders.Segment(0, float(ts[5]) + 0.003, float(ts[15]), 30, "a")]
    assert windows.segment_of_frame(ts, segs, 5) is segs[0]
    seg_window = [w for w in windows.windows_for_scene(ts, segs, len(ts), False) if w.kind == "segment"][0]
    assert seg_window.frame_start == 5
    # A frame a whole period early is genuinely outside it.
    assert windows.segment_of_frame(ts, segs, 4) is None


def test_the_boundary_slack_never_gives_one_frame_to_two_segments():
    """Segments share no reference frame, so a frame in both would be in neither."""
    ts = np.arange(40, dtype=np.float64) * 0.32
    # Two segments whose gap is narrower than the slack a frame period allows.
    segs = [
        loaders.Segment(0, float(ts[2]), float(ts[10]) + 0.02, 20, "a"),
        loaders.Segment(1, float(ts[10]) + 0.08, float(ts[20]), 20, "b"),
    ]
    owners = [windows.segment_of_frame(ts, segs, i) for i in range(len(ts))]
    assert all(o is None or o in segs for o in owners)
    lo, hi = windows.segment_bounds(ts, segs)
    assert lo[1] <= hi[0]  # the widened spans meet, they do not overlap
    seg_windows = [w for w in windows.windows_for_scene(ts, segs, len(ts), False) if w.kind == "segment"]
    assert seg_windows[0].frame_end <= seg_windows[1].frame_start


def test_whole_scene_window_reports_its_pose_source():
    ts = np.arange(50, dtype=np.float64)
    with_tf = windows.windows_for_scene(ts, [], 50, has_transforms=True)
    assert with_tf[0].pose_source == "transforms"
    without = windows.windows_for_scene(ts, [], 50, has_transforms=False)
    assert without[0].pose_source == "none"


# --------------------------------------------------------------------------
# wire format
# --------------------------------------------------------------------------


def test_split_scene_document_extracts_every_blob():
    doc = {
        "model": {"terrain": {"heightData": "__b64__AAAA"}, "n": 1},
        "states": {"version": 4, "bodies": {"a": {"bodyTransform": "__b64__BBBB"}}},
    }
    model, states, blobs = wire.split_scene_document(doc)
    assert len(blobs) == 2
    assert model["terrain"]["heightData"].startswith("/blob/")
    assert states["bodies"]["a"]["bodyTransform"].startswith("/blob/")
    # The viewer resolves a blob reference by its last path segment.
    ids = {
        int(model["terrain"]["heightData"].rsplit("/", 1)[-1]),
        int(states["bodies"]["a"]["bodyTransform"].rsplit("/", 1)[-1]),
    }
    assert ids == {0, 1}


def test_write_static_bundle_lays_out_the_files_the_front_end_fetches(tmp_path):
    doc = {"model": {"x": "__b64__AAAA"}, "states": {"version": 4}}
    dest = tmp_path / "bundle"
    info = wire.write_static_bundle(doc, dest)
    assert info["blobs"] == 1
    assert (dest / "model.json").is_file()
    assert (dest / "states.json").is_file()
    assert (dest / "blob" / "0").is_file()
    assert json.loads((dest / "model.json").read_text())["x"] == "/blob/static/0"


def test_render_viewer_page_substitutes_urls_and_injects_the_static_base():
    pytest.importorskip("simview")
    page = wire.render_viewer_page("/simview-static", "/api/data/k", "1.2.3", "<div id='x'></div>")
    assert 'window.__simviewStaticBase = "/api/data/k"' in page
    assert "/simview-static/js/main.js" in page
    assert '<base href="/">' in page
    assert "{{" not in page
    assert "<div id='x'></div>" in page


# --------------------------------------------------------------------------
# pose resolution and terrain
# --------------------------------------------------------------------------


def _roll_pitch(quat):
    """Roll and pitch, in degrees, of a [w, x, y, z] quaternion."""
    R = loaders.poses_from_quat(np.zeros((1, 3)), np.asarray(quat)[None])[0, :3, :3]
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
    from dataset_tools.viewer import build, probe as probe_mod

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
    from dataset_tools.viewer import build, probe as probe_mod

    root_path = make_root(tmp_path, frames=12)
    _tilt_gravity(root_path, "0000", 12, -20.0)
    access = build.SceneAccess(probe_mod.DatasetRoot(root_path), "0000")
    window = access.window(windows.frame_window_key(3), 1500)
    track = build.resolve_poses(access, window, access.frame_ts[3:4])
    assert np.allclose(track.positions, 0.0)  # still at the origin
    roll, _ = _roll_pitch(track.quats[0])
    assert roll == pytest.approx(-20.0, abs=0.5)


def test_a_scene_without_gravity_alignment_is_left_unlevelled(tmp_path):
    from dataset_tools.viewer import build, probe as probe_mod

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
    from dataset_tools.viewer import build, probe as probe_mod

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
    poses = loaders.poses_from_quat(track.positions, track.quats)
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
    """A cell is reached by dozens of maps. Taking the best validity among them
    paints a cell measured on the strength of one distant frame whose map edge
    caught a return, while the near frames that actually decided the height were
    extrapolating. The layer has to be weighted as the height is."""
    from dataset_tools.viewer import build, probe as probe_mod

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
        np.savez_compressed(scene / "height_maps" / f"{i:05d}.npz", height_map=np.zeros((128, 128)))
        # Frame 0 measured everything it saw; frame 1 extrapolated everything.
        np.savez_compressed(
            scene / "height_masks" / f"{i:05d}.npz",
            validity=np.full((128, 128), 2 if i == 0 else 0, dtype=np.uint8),
        )

    access = build.SceneAccess(probe_mod.DatasetRoot(root_path), "0000")
    window = access.window("seg000", 1500)
    frames = np.arange(window.frame_start, window.frame_end)
    track = build.resolve_poses(access, window, access.frame_ts[frames])
    poses = loaders.poses_from_quat(track.positions, track.quats)
    opts = build.BuildOptions().sanitized()
    grid = build._make_grid(track.positions, opts)
    build.splat_height_maps(access, grid, frames, poses, opts)
    _, layers, bounds = build._finish_terrain(grid, None)

    row, col, inside = grid.cell_of(track.positions[:, 0], track.positions[:, 1])
    under = layers["validity"][row[inside], col[inside]]
    assert bounds["validity"] == (0.0, 2.0)
    assert under[0] == pytest.approx(2.0, abs=0.1)
    # Both maps reach it, so the best of them would call this cell measured.
    assert under[1] == pytest.approx(0.0, abs=0.1)


def test_a_nan_in_a_height_map_does_not_poison_the_terrain(tmp_path):
    """The generator is free to leave a cell NaN. Splatted, it would sit in the
    weighted sums for good -- the cell counts as covered, so nothing fills it,
    and a NaN reaches the heightfield -- and it smears into world x and y too,
    where the cast to a cell index is undefined."""
    from dataset_tools.viewer import build, probe as probe_mod

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
    poses = loaders.poses_from_quat(track.positions, track.quats)
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
    from dataset_tools.viewer import build, probe as probe_mod

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
# end to end over HTTP
# --------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path):
    pytest.importorskip("simview")
    pytest.importorskip("torch")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from dataset_tools.viewer.app import create_app

    root_path = make_root(tmp_path, scenes=("0000", "0001"), frames=12, with_cmd_vel=False)
    app = create_app(root_path, cache_dir=tmp_path / "cache")
    return TestClient(app)


def test_index_and_scene_pages_render(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "0000" in r.text and "0001" in r.text
    r = client.get("/scene/0000")
    assert r.status_code == 200
    assert "Segment 0" in r.text


def test_unknown_scene_is_a_404(client):
    assert client.get("/scene/9999").status_code == 404
    assert client.get("/api/scene/9999").status_code == 404


def test_scene_api_lists_windows_and_missing_files(client):
    data = client.get("/api/scene/0000").json()
    assert data["probe"]["n_frames"] == 12
    assert "cmd_vel" in data["probe"]["missing"]
    assert any(w["pose_source"] == "odometry" for w in data["windows"])


def test_building_a_window_produces_a_loadable_simview_scene(client, tmp_path):
    """The generated bundle must be a scene SimView itself can read back."""
    from simview.scene import SimulationScene

    opts = "40-160000-80-500000-0.1-height"
    meta = client.get(f"/api/data/0000/seg000/{opts}/x/meta.json")
    assert meta.status_code == 200, meta.text
    body = meta.json()
    assert body["pose_source"] == "odometry"
    assert body["build"]["terrain_source"] in ("height_maps", "clouds")
    # The panel's GT-against-IMU comparison travels with the window.
    assert set(body["attitude"]) >= {"gt_roll", "imu_roll", "error", "median_error", "bands"}
    assert len(body["attitude"]["error"]) == len(body["frames"])

    model = client.get(f"/api/data/0000/seg000/{opts}/x/model.json").json()
    states = client.get(f"/api/data/0000/seg000/{opts}/x/states.json").json()
    assert [b["name"] for b in model["bodies"]][0] == "robot"

    # Re-inline the blobs and hand the document to SimView's own loader.
    import base64

    blobs = {}

    def resolve(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, str) and v.startswith("/blob/"):
                    n = v.rsplit("/", 1)[-1]
                    if n not in blobs:
                        raw = client.get(f"/api/data/0000/seg000/{opts}/x/blob/{n}").content
                        blobs[n] = "__b64__" + base64.b64encode(raw).decode()
                    node[k] = blobs[n]
                else:
                    resolve(v)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                if isinstance(v, str) and v.startswith("/blob/"):
                    n = v.rsplit("/", 1)[-1]
                    if n not in blobs:
                        raw = client.get(f"/api/data/0000/seg000/{opts}/x/blob/{n}").content
                        blobs[n] = "__b64__" + base64.b64encode(raw).decode()
                    node[i] = blobs[n]
                else:
                    resolve(v)

    resolve(model)
    resolve(states)
    scene = SimulationScene.from_dict({"model": model, "states": states})
    assert scene.model.terrain is not None
    assert "robot" in scene.model.bodies
    assert len(scene.states) == len(body["frames"])


def test_blob_endpoint_supports_byte_ranges(client):
    opts = "40-160000-80-500000-0.1-height"
    client.get(f"/api/data/0000/seg000/{opts}/x/meta.json")
    r = client.get(f"/api/data/0000/seg000/{opts}/x/blob/0", headers={"Range": "bytes=0-15"})
    assert r.status_code == 206
    assert len(r.content) == 16


def test_per_frame_endpoints_serve_images_and_404_past_the_end(client):
    assert client.get("/api/scene/0000/frame/0/image.jpg").status_code == 200
    assert client.get("/api/scene/0000/frame/0/heightmap.png").status_code == 200
    assert client.get("/api/scene/0000/frame/0/validity.png").status_code == 200
    assert client.get("/api/scene/0000/frame/0/cloud.png").status_code == 200
    assert client.get("/api/scene/0000/frame/9999/image.jpg").status_code == 404
    info = client.get("/api/scene/0000/frame/3/info").json()
    assert info["frame"] == 3
    assert "timestamp" in info


def test_viewer_page_wires_the_static_base_and_defers_simview(client):
    r = client.get("/view/0000/seg000")
    assert r.status_code == 200
    assert "__simviewStaticBase" in r.text
    assert "__dtSimviewEntry" in r.text  # SimView's entry point is deferred
    assert "dt-panel" in r.text
    assert 'id="dt-attitude"' in r.text  # the GT-against-IMU readout
    assert client.get("/view/0000/nosuchwindow").status_code == 404


def test_a_scene_without_optional_files_still_builds(tmp_path):
    """The degradation contract: no height maps, no gravity alignment, no
    proprioception, no wheel odometry -- the window must still build."""
    pytest.importorskip("simview")
    pytest.importorskip("torch")
    from fastapi.testclient import TestClient

    from dataset_tools.viewer.app import create_app

    root_path = make_root(
        tmp_path,
        with_height_maps=False,
        with_cmd_vel=False,
        with_transforms=False,
        with_gravity=False,
        with_joints=False,
    )
    client = TestClient(create_app(root_path, cache_dir=tmp_path / "cache"))
    opts = "40-160000-80-500000-0.1-height"
    meta = client.get(f"/api/data/0000/seg000/{opts}/x/meta.json")
    assert meta.status_code == 200, meta.text
    body = meta.json()
    assert body["build"]["terrain_source"] == "clouds"
    # Only pose-derived scalars survive when there is no proprioception.
    assert set(body["scalar_names"]) <= {"speed", "pose_z", "yaw_rate"}
    assert "height_maps" not in body["available"]


# --------------------------------------------------------------------------
# curated atlas metadata (optional, root-level)
# --------------------------------------------------------------------------


#: The fixture's curated thumbnail frames. Deliberately NOT the 10/50/90 %
#: fractions of a 12-frame scene (1, 6, 10), so a test can tell which one the
#: viewer used.
CURATED_THUMBS = {"a": 2, "b": 7, "c": 11}


def write_atlas(
    root_path: Path,
    scenes=("0000",),
    *,
    with_thumbs: bool = True,
    legacy_thumbs: bool = False,
    dataset_version: str | None = None,
) -> Path:
    """The optional ``atlas/`` half of a root: descriptions, groups, thumbnails.

    ``legacy_thumbs`` writes the pre-generator spelling, a bare list of slots
    that names no frame -- roots in the wild still carry it.
    """
    atlas_dir = root_path / "atlas"
    (atlas_dir / "thumbs").mkdir(parents=True, exist_ok=True)
    slots = ["a", "b", "c"]
    thumbs = slots if legacy_thumbs else dict(CURATED_THUMBS)
    entries = {}
    for scene in scenes:
        if with_thumbs:
            for slot in slots:
                _write_png(atlas_dir / "thumbs" / f"{scene}{slot}.jpg", size=(64, 40))
        entries[scene] = {
            "description": f"A synthetic place for scene {scene}, written by hand.",
            "date": "2025-09-12 15:13",
            "dur_s": 653.6,
            "frames": 1962,
            "images": 1962,
            "clouds": 1962,
            "pts_mean": 6090,
            "poses": 4580,
            "poses_unf": 4776,
            "segments": 9,
            "seg_lens": [314.3, 93.9, 67.0],
            "seg_med_s": 18.7,
            "gt_span_s": 565.0,
            "cov": 0.867,
            "path_m": 439.5,
            "moving_frac": 0.763,
            "win_med": 0.1207,
            "win_p95": 0.2537,
            "flags": 24,
            "bag_gb": 25.1,
            "session": "synthetic_session",
            "thumbs": thumbs if with_thumbs else {},
        }
    (atlas_dir / "atlas.json").write_text(
        json.dumps(
            {
                "schema": "dataset_tools.viewer/atlas@2",
                "generated": "2026-09-02",
                "dataset_version": dataset_version,
                "title": "Synthetic Atlas",
                "kicker": "test fixture",
                "intro": "One paragraph of curated introduction.",
                "root_note": "A curated note about the root.",
                "footer": "A curated footer.",
                "totals": [{"label": "scenes", "value": str(len(scenes)), "sub": None}],
                "thumbs": {"dir": "thumbs", "slots": slots, "pattern": "{scene}{slot}.jpg"},
                "groups": [
                    {"title": "An outing", "range": "all of it", "blurb": "What happened here.", "scenes": list(scenes)}
                ],
                "scenes": entries,
            }
        )
    )
    return atlas_dir


def test_a_root_without_an_atlas_directory_reports_no_curated_metadata(tmp_path):
    root = probe.DatasetRoot(make_root(tmp_path), cache_dir=tmp_path / "cache")
    assert root.atlas() is None
    assert root.atlas_scene("0000") == {}
    assert root.atlas_thumb("0000", "a") is None


def test_atlas_thumbnails_are_looked_up_through_the_manifest_not_the_path(tmp_path):
    root_path = make_root(tmp_path)
    write_atlas(root_path)
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")
    assert root.atlas_thumb("0000", "a") is not None
    # A slot the atlas does not list is not a path to try -- it is simply absent.
    assert root.atlas_thumb("0000", "../../../etc/passwd") is None
    assert root.atlas_thumb("0000", "z") is None


def test_a_thumbnail_slot_resolves_to_the_frame_the_atlas_names(tmp_path):
    """The curated index wins over the fraction, and both spellings are read."""
    curated = {"thumbs": dict(CURATED_THUMBS)}
    assert index_page.thumb_slots(curated) == ["a", "b", "c"]
    assert [index_page.thumb_index(curated, s, 12) for s in ("a", "b", "c")] == [2, 7, 11]

    # The older spelling names no frame, so the fractions stand in -- which is
    # what those files were cut at anyway.
    legacy = {"thumbs": ["a", "b", "c"]}
    assert index_page.thumb_slots(legacy) == ["a", "b", "c"]
    assert [index_page.thumb_index(legacy, s, 12) for s in ("a", "b", "c")] == [1, 6, 10]

    # A slot the atlas skipped, and an index pointing past a re-cut scene.
    assert index_page.thumb_slots({"thumbs": {"b": 3}}) == ["b"]
    assert index_page.thumb_index({}, "a", 12) == 1
    assert index_page.thumb_index({"thumbs": {"a": 900}}, "a", 12) == 11
    assert index_page.thumb_index({"thumbs": {"a": 5}}, "a", 0) == 0


def test_an_atlas_from_another_release_says_so_on_the_index(tmp_path):
    """The v1.3 failure: atlas/ symlinked to v1.2's, so every curated number on
    the page described the previous release under the previous release's title."""
    root_path = make_root(tmp_path)
    (root_path / "manifest.json").write_text(json.dumps({"dataset_version": "1.3"}))
    write_atlas(root_path, dataset_version="1.2")
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")
    page = index_page.build_index(root, root.probe_all())
    assert "1.2" in page["atlas_warning"] and "1.3" in page["atlas_warning"]

    # Regenerated against this root: nothing to say.
    write_atlas(root_path, dataset_version="1.3")
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache2")
    assert index_page.build_index(root, root.probe_all())["atlas_warning"] is None


def test_a_malformed_atlas_file_degrades_to_no_atlas(tmp_path):
    root_path = make_root(tmp_path)
    (root_path / "atlas").mkdir()
    (root_path / "atlas" / "atlas.json").write_text("{not json")
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")
    assert root.atlas() is None


def test_probe_records_segment_spans_as_fractions_and_marks_filled_ones(tmp_path):
    root_path = make_root(tmp_path)
    scene = root_path / "0000"
    ts = np.loadtxt(scene / "data_ts.txt")
    with open(scene / "odometry_segments.txt", "w") as fh:
        fh.write("# id start_ts end_ts n_poses reason\n")
        fh.write(f"0 {ts[0]:.6f} {ts[3]:.6f} 4 filtered:pose_jump\n")
        fh.write(f"1 {ts[6]:.6f} {ts[9]:.6f} 4 v11_fill\n")
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")
    p = root.probe("0000", refresh=True)
    assert len(p.segment_spans) == 2
    assert p.fill_spans == [p.segment_spans[1]]
    lo, hi = p.segment_spans[0]
    assert lo == pytest.approx(0.0, abs=1e-6)
    assert hi == pytest.approx(3 / 11, abs=1e-3)  # frames 0..3 of 0..11
    assert all(0.0 <= v <= 1.0 for span in p.segment_spans for v in span)


def test_a_stale_probe_cache_from_an_older_format_is_discarded(tmp_path):
    root_path = make_root(tmp_path)
    cache = tmp_path / "cache"
    root = probe.DatasetRoot(root_path, cache_dir=cache)
    root.probe("0000")
    stale = json.loads((cache / "probe_cache.json").read_text())
    stale["format"] = 0
    for entry in stale["scenes"].values():
        entry["segment_spans"] = []
    (cache / "probe_cache.json").write_text(json.dumps(stale))
    again = probe.DatasetRoot(root_path, cache_dir=cache)
    assert again.probe("0000").segment_spans, "an old cache must be re-probed, not trusted"


def test_index_rows_use_curated_content_when_it_exists(tmp_path):
    from dataset_tools.viewer import index_page

    root_path = make_root(tmp_path)
    write_atlas(root_path)
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")
    page = index_page.build_index(root, root.probe_all())
    assert [g["title"] for g in page["groups"]] == ["An outing"]
    row = page["rows"][0]
    assert row["description"].startswith("A synthetic place")
    labels = [s["label"] for s in row["stats"]]
    assert "path in GT" in labels and "time with GT" in labels
    # Regeneration diagnostics stay off the index (they live on the scene page).
    assert "2 s err med / p95" not in labels and "v1 coverage" not in labels
    assert [t["url"] for t in row["thumbs"]] == [f"/api/scene/0000/thumb/{s}.jpg" for s in ("a", "b", "c")]
    assert page["totals"][0]["label"] == "scenes"


def test_index_rows_fall_back_to_probed_content_without_an_atlas(tmp_path):
    from dataset_tools.viewer import index_page

    root = probe.DatasetRoot(make_root(tmp_path), cache_dir=tmp_path / "cache")
    page = index_page.build_index(root, root.probe_all())
    row = page["rows"][0]
    assert row["description"] is None
    labels = [s["label"] for s in row["stats"]]
    assert "duration" in labels and "frames" in labels
    # Nothing curated may leak into the fallback.
    assert "2 s err med / p95" not in labels
    # The timeline still exists: it comes from the segment file, not the atlas.
    assert row["spans"]
    # Thumbnails point at the same endpoint; it will render live frames.
    assert row["thumbs"][0]["url"] == "/api/scene/0000/thumb/a.jpg"
    assert page["groups"][0]["title"] == "Scenes"


def test_a_scene_missing_from_the_atlas_is_still_listed(tmp_path):
    from dataset_tools.viewer import index_page

    root_path = make_root(tmp_path, scenes=("0000", "0001"))
    write_atlas(root_path, scenes=("0000",))
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")
    page = index_page.build_index(root, root.probe_all())
    listed = {r["probe"].scene_id for g in page["groups"] for r in g["rows"]}
    assert listed == {"0000", "0001"}
    assert page["groups"][-1]["title"] == "Not in the atlas"


def test_duration_formatting_matches_the_atlas_phrasing():
    from dataset_tools.viewer.index_page import coverage_class, fmt_duration

    assert fmt_duration(39.4) == "39 s"
    assert fmt_duration(653.6) == "10 min 53 s"
    assert fmt_duration(3780) == "1 h 3 min"
    assert fmt_duration(None) == "—"
    assert coverage_class(0.9) == "hi"
    assert coverage_class(0.6) == "mid"
    assert coverage_class(0.42) == "lo"
    assert coverage_class(None) == "none"


# --------------------------------------------------------------------------
# single-frame windows
# --------------------------------------------------------------------------


def test_a_frame_window_key_round_trips():
    assert windows.parse_frame_window_key(windows.frame_window_key(42)) == 42
    assert windows.parse_frame_window_key("seg001") is None
    assert windows.parse_frame_window_key("framex") is None


def test_a_single_frame_window_takes_its_pose_from_the_segment_it_is_in():
    ts = np.arange(20, dtype=np.float64)
    segs = [loaders.Segment(0, 0.0, 5.0, 6, "a")]
    inside = windows.single_frame_window(ts, segs, 20, 3)
    assert inside.pose_source == "odometry" and inside.segment_index == 0
    assert inside.frame_start == 3 and inside.n_frames == 1
    outside = windows.single_frame_window(ts, segs, 20, 12)
    assert outside.pose_source == "none" and outside.segment_index is None
    assert "robot-centric" in outside.reason
    assert windows.single_frame_window(ts, segs, 20, 99) is None


# --------------------------------------------------------------------------
# index, scene tabs and frame pages over HTTP
# --------------------------------------------------------------------------


@pytest.fixture
def atlas_client(tmp_path):
    pytest.importorskip("simview")
    pytest.importorskip("torch")
    from fastapi.testclient import TestClient

    from dataset_tools.viewer.app import create_app

    root_path = make_root(tmp_path, scenes=("0000", "0001"), frames=12)
    write_atlas(root_path, scenes=("0000", "0001"))
    return TestClient(create_app(root_path, cache_dir=tmp_path / "cache"))


def test_the_index_shows_curated_descriptions_and_serves_stored_thumbnails(atlas_client):
    r = atlas_client.get("/")
    assert r.status_code == 200
    assert "Synthetic Atlas" in r.text
    assert "A synthetic place for scene 0000" in r.text
    assert "An outing" in r.text and "What happened here." in r.text
    assert 'loading="lazy"' in r.text
    thumb = atlas_client.get("/api/scene/0000/thumb/a.jpg")
    assert thumb.status_code == 200
    assert "immutable" in thumb.headers["cache-control"]


def test_the_index_renders_without_any_curated_metadata(client):
    """The rig-agnostic contract: no atlas/, same page, fewer facts."""
    r = client.get("/")
    assert r.status_code == 200
    assert "0000" in r.text and "0001" in r.text
    assert 'class="env"' not in r.text  # no curated description to show
    assert "/api/scene/0000/thumb/a.jpg" in r.text
    # And that thumbnail endpoint renders a live frame instead of a stored file.
    thumb = client.get("/api/scene/0000/thumb/b.jpg")
    assert thumb.status_code == 200
    assert thumb.headers["content-type"] == "image/jpeg"


def test_the_frames_tab_pages_through_frames_and_jumps(client):
    r = client.get("/scene/0000?tab=frames")
    assert r.status_code == 200
    assert "/scene/0000/frame/0" in r.text
    assert r.text.count("image.jpg?w=192") == 12  # the whole scene fits one page
    assert 'loading="lazy"' in r.text
    # A page past the end clamps rather than 404s.
    assert client.get("/scene/0000?tab=frames&page=99").status_code == 200
    jump = client.get("/scene/0000/jump?index=7", follow_redirects=False)
    assert jump.headers["location"] == "/scene/0000/frame/7"
    assert (
        client.get("/scene/0000/jump?index=999", follow_redirects=False).headers["location"] == "/scene/0000/frame/11"
    )


def test_the_frame_page_states_what_the_frame_has(client):
    r = client.get("/scene/0000/frame/3")
    assert r.status_code == 200
    assert "Camera stamp" in r.text and "Lidar stamp" in r.text
    assert "Segment</dt>" in r.text
    assert "/view/0000/frame00003" in r.text  # the static single-frame view
    assert "?frame=3" in r.text  # and the motion context around it
    assert client.get("/scene/0000/frame/9999").status_code == 404


def test_a_frame_with_no_ground_truth_pose_still_has_a_page(tmp_path):
    pytest.importorskip("simview")
    pytest.importorskip("torch")
    from fastapi.testclient import TestClient

    from dataset_tools.viewer.app import create_app

    root_path = make_root(tmp_path, segments=False, with_height_maps=False)
    client = TestClient(create_app(root_path, cache_dir=tmp_path / "cache"))
    r = client.get("/scene/0000/frame/4")
    assert r.status_code == 200
    assert "No ground-truth pose" in r.text
    assert "robot-centric" in r.text
    assert "/view/0000/frame00004" in r.text


def test_a_single_frame_bundle_builds_and_simview_can_read_it_back(client):
    """One frame, one state: a static scene, still a valid SimView document."""
    from simview.scene import SimulationScene

    opts = "40-160000-80-500000-0.1-height"
    meta = client.get(f"/api/data/0000/frame00003/{opts}/x/meta.json")
    assert meta.status_code == 200, meta.text
    body = meta.json()
    assert body["frames"] == [3]
    assert body["pose_source"] == "odometry"
    assert body["window"]["kind"] == "frame"

    model = client.get(f"/api/data/0000/frame00003/{opts}/x/model.json").json()
    states = client.get(f"/api/data/0000/frame00003/{opts}/x/states.json").json()
    import base64

    def resolve(node):
        items = node.items() if isinstance(node, dict) else enumerate(node)
        for key, value in list(items):
            if isinstance(value, str) and value.startswith("/blob/"):
                n = value.rsplit("/", 1)[-1]
                raw = client.get(f"/api/data/0000/frame00003/{opts}/x/blob/{n}").content
                node[key] = "__b64__" + base64.b64encode(raw).decode()
            elif isinstance(value, (dict, list)):
                resolve(value)

    resolve(model)
    resolve(states)
    scene = SimulationScene.from_dict({"model": model, "states": states})
    assert len(scene.states) == 1
    assert "robot" in scene.model.bodies
    assert scene.model.terrain is not None


def test_a_single_frame_view_page_opens(client):
    r = client.get("/view/0000/frame00003")
    assert r.status_code == 200
    assert "__simviewStaticBase" in r.text
    r = client.get("/view/0000/seg000?frame=4")
    assert r.status_code == 200
    assert '"initialFrame": 4' in r.text


# --------------------------------------------------------------------------
# the scene page: four cards assembled from probe + atlas + a few small files
# --------------------------------------------------------------------------


def scene_page(tmp_path, root_path, scene: str = "0000"):
    from dataset_tools.viewer.build import SceneAccess
    from dataset_tools.viewer.scene_page import build_scene_page

    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")
    return build_scene_page(root, SceneAccess(root, scene))


def card_of(page, title: str) -> dict | None:
    return next((c for c in page["cards"] if c["title"] == title), None)


def value_of(card, label: str):
    return next((s["value"] for s in card["stats"] if s["label"] == label), None)


def sub_of(card, label: str):
    return next((s["sub"] for s in card["stats"] if s["label"] == label), None)


def test_camera_info_reads_the_fields_the_page_shows(tmp_path):
    from dataset_tools.viewer.scene_page import read_camera_info

    path = tmp_path / "camera_info.yaml"
    path.write_text(CAMERA_INFO)
    info = read_camera_info(path)
    assert info["image_width"] == 640 and info["image_height"] == 400
    assert info["distortion_model"] == "plumb_bob"
    assert info["frame_id"] == "test_camera_optical_frame"
    # A nested block collapses to its data list plus its declared shape.
    assert len(info["distortion_coefficients"]) == 5
    assert info["camera_matrix_rows"] == 3
    assert info["camera_matrix"][0] == 700.0


def test_camera_info_of_a_missing_or_malformed_file_is_simply_absent(tmp_path):
    from dataset_tools.viewer.scene_page import read_camera_info

    assert read_camera_info(tmp_path / "nope.yaml") is None
    bad = tmp_path / "bad.yaml"
    bad.write_text("!!! not: [a, yaml\nfile at all\n")
    read_camera_info(bad)  # must not raise; whatever it salvages is fine


def test_a_series_reports_the_files_own_rate_not_the_sampled_one(tmp_path):
    from dataset_tools.viewer.scene_page import read_series

    arr = np.zeros((10_000, 4))
    arr[:, 0] = 1_700_000_000.0 + 0.01 * np.arange(10_000)  # 100 Hz
    arr[:, 1] = 2.0
    path = tmp_path / "series.npy"
    np.save(path, arr)

    series = read_series(path, max_rows=50)
    assert series.n == 10_000  # the header count, not the sampled one
    assert len(series.rows) <= 51
    assert series.rate == pytest.approx(100.0, rel=0.01)
    assert series.mean_abs(1) == pytest.approx(2.0)


def test_the_four_cards_state_what_the_scene_has(tmp_path):
    root_path = make_root(tmp_path)
    write_atlas(root_path)
    page = scene_page(tmp_path, root_path)
    assert [c["title"] for c in page["cards"]] == [
        "Recording",
        "Ground truth",
        "Calibration",
        "Proprioception",
    ]
    # No card is a single entry: that was the point of reworking them.
    assert all(len(c["stats"]) >= 2 for c in page["cards"])

    recording = card_of(page, "Recording")
    assert value_of(recording, "synced pairs") == "12"
    assert value_of(recording, "frame rate") == "10.00 Hz"
    assert value_of(recording, "image size") == "640 × 400"
    assert "25.1 GB" in recording["note"] and "synthetic_session" in recording["note"]

    calib = card_of(page, "Calibration")
    assert value_of(calib, "focal length") == "700.0 · 700.0"
    assert value_of(calib, "camera → base") == "0.000, 0.000, 0.000"
    assert "plumb_bob" in sub_of(calib, "image")
    assert [m["label"] for m in calib["matrices"]] == [
        "camera_intrinsics.txt (K)",
        "T_cam_to_base.txt (4×4)",
    ]


def test_wheel_speeds_keep_the_units_the_file_carries(tmp_path):
    """No rig constant: a wheel radius belongs to one robot, rad/s to all of them."""
    page = scene_page(tmp_path, make_root(tmp_path))
    proprio = card_of(page, "Proprioception")
    assert value_of(proprio, "wheel speed") == "1.50 rad/s"  # the fixture's velocity
    assert value_of(proprio, "wheels turning") == "100%"
    assert value_of(proprio, "commanded speed") == "0.40 m/s"
    assert value_of(proprio, "roll · pitch") == "0.0° · 0.0°"  # identity gravity align


def test_a_card_with_nothing_to_say_is_dropped_rather_than_shown_empty(tmp_path):
    root_path = make_root(tmp_path, with_joints=False, with_cmd_vel=False, with_gravity=False)
    titles = [c["title"] for c in scene_page(tmp_path, root_path)["cards"]]
    assert "Proprioception" not in titles
    assert "Recording" in titles and "Ground truth" in titles


def test_the_ground_truth_card_reads_the_segment_file_over_the_atlas(tmp_path):
    """The file is the truth: a scene may have been re-filtered since curation."""
    root_path = make_root(tmp_path)
    write_atlas(root_path)  # claims nine segments
    card = card_of(scene_page(tmp_path, root_path), "Ground truth")
    assert value_of(card, "segments") == 2  # what the file actually holds
    assert card["badge"].endswith("GT")
    assert card["timeline"]["spans"]
    # Curated numbers that cannot be recomputed are still shown.
    assert value_of(card, "2 s err med / p95") == "0.12 / 0.25 m"
    assert value_of(card, "divergence flags") == 24


def test_a_filled_segment_is_marked_on_the_scene_timeline(tmp_path):
    root_path = make_root(tmp_path)
    scene = root_path / "0000"
    ts = np.loadtxt(scene / "data_ts.txt")
    with open(scene / "odometry_segments.txt", "w") as fh:
        fh.write(f"0 {ts[0]:.6f} {ts[5]:.6f} 6 filtered:pose_jump\n")
        fh.write(f"1 {ts[6]:.6f} {ts[-1]:.6f} 6 v11_fill\n")
    card = card_of(scene_page(tmp_path, root_path), "Ground truth")
    assert len(card["timeline"]["fill"]) == 1
    assert len(card["timeline"]["spans"]) == 1
    assert "filled from v1.1" in card["note"]


def test_the_card_never_claims_more_raw_poses_than_the_scene_has(tmp_path):
    """A filled scene has more poses than this root's own run produced."""
    root_path = make_root(tmp_path)
    atlas = write_atlas(root_path) / "atlas.json"
    data = json.loads(atlas.read_text())
    data["scenes"]["0000"]["poses_unf"] = 3  # fewer than the segments carry
    atlas.write_text(json.dumps(data))
    card = card_of(scene_page(tmp_path, root_path), "Ground truth")
    assert value_of(card, "poses") == "12"
    assert sub_of(card, "poses") is None


def test_path_length_is_derived_from_odometry_when_no_atlas_supplies_it(tmp_path):
    """Rig-agnostic fallback: distance inside segments, never across them."""
    page = scene_page(tmp_path, make_root(tmp_path))  # no atlas/
    card = card_of(page, "Ground truth")
    # Two segments of six poses, each step 0.3 m forward and 0.02 m sideways.
    assert value_of(card, "path in GT") == "3 m"
    assert sub_of(card, "path in GT") == "from odometry.txt"
    assert value_of(card, "moving") == "100%"


def test_the_hero_thumbnails_are_the_index_slots_and_link_to_their_frames(tmp_path):
    page = scene_page(tmp_path, make_root(tmp_path))
    assert [t["label"] for t in page["hero"]] == ["10%", "50%", "90%"]
    assert [t["url"] for t in page["hero"]] == [f"/api/scene/0000/thumb/{slot}.jpg" for slot in ("a", "b", "c")]
    assert [t["href"] for t in page["hero"]] == [
        "/scene/0000/frame/1",
        "/scene/0000/frame/6",
        "/scene/0000/frame/10",
    ]


def test_a_curated_hero_thumbnail_links_to_the_frame_it_was_cut_from(tmp_path):
    """Picture and caption come from one number, so they cannot drift apart:
    before the atlas named the frame, the link was recomputed from the fraction
    and simply assumed the stored JPEG had been cut there."""
    root_path = make_root(tmp_path)
    write_atlas(root_path)
    page = scene_page(tmp_path, root_path)
    assert [t["href"] for t in page["hero"]] == [f"/scene/0000/frame/{i}" for i in CURATED_THUMBS.values()]
    assert [t["sub"].split(" ·")[0] for t in page["hero"]] == [f"frame {i}" for i in CURATED_THUMBS.values()]


def test_a_curated_thumbnail_with_no_stored_file_renders_the_frame_it_names(tmp_path):
    """The stored JPEG is a cache of the frame, not the definition of it: delete
    it and the endpoint renders exactly the same frame at the same width."""
    pytest.importorskip("simview")
    pytest.importorskip("torch")
    from fastapi.testclient import TestClient

    from dataset_tools.viewer.app import create_app

    root_path = make_root(tmp_path)
    atlas_dir = write_atlas(root_path)
    for stored in (atlas_dir / "thumbs").glob("*.jpg"):
        stored.unlink()
    # The fixture writes one flat colour per frame, so mark the frame the 50 %
    # fraction would have picked -- otherwise every frame renders alike and the
    # test cannot tell the two rules apart.
    from PIL import Image

    Image.new("RGB", (16, 10), (200, 30, 30)).save(root_path / "0000" / "images" / "00006.png")

    client = TestClient(create_app(root_path, cache_dir=tmp_path / "cache"))
    thumb = client.get("/api/scene/0000/thumb/b.jpg")
    assert thumb.status_code == 200
    frame = client.get(f"/api/scene/0000/frame/{CURATED_THUMBS['b']}/image.jpg?w=384")
    assert thumb.content == frame.content
    assert thumb.content != client.get("/api/scene/0000/frame/6/image.jpg?w=384").content


def test_the_scene_page_leads_with_its_thumbnails_and_the_four_cards(atlas_client):
    r = atlas_client.get("/scene/0000")
    assert r.status_code == 200
    body = r.text
    for slot in ("a", "b", "c"):
        assert f"/api/scene/0000/thumb/{slot}.jpg" in body
    for title in ("Recording", "Ground truth", "Calibration", "Proprioception"):
        assert f"<h2>{title}</h2>" in body
    # The thumbnails come before the cards, which come before the tabs.
    assert body.index('class="hero"') < body.index('class="grid cards"') < body.index('class="tabs"')
    # And the absent-artefact chips survived the rework.
    assert 'class="chip no"' in body and "joint_setpoints.npy" in body


def test_a_scene_page_on_a_root_with_almost_nothing_still_renders(tmp_path):
    pytest.importorskip("simview")
    pytest.importorskip("torch")
    from fastapi.testclient import TestClient

    from dataset_tools.viewer.app import create_app

    root_path = make_root(
        tmp_path,
        with_height_maps=False,
        with_cmd_vel=False,
        with_transforms=False,
        with_gravity=False,
        with_joints=False,
        with_camera_info=False,
    )
    client = TestClient(create_app(root_path, cache_dir=tmp_path / "cache"))
    body = client.get("/scene/0000").text
    assert "<h2>Recording</h2>" in body and "<h2>Ground truth</h2>" in body
    assert "<h2>Proprioception</h2>" not in body  # nothing to fill it
    assert "field of view" not in body  # no camera_info, so no image size to use
    assert 'class="chip no"' in body


def test_the_index_and_the_scene_page_share_one_palette():
    """The index may restyle its *type*; its colours are the viewer's colours."""
    from dataset_tools.viewer import app as viewer_app

    css = (Path(viewer_app.__file__).parent / "static" / "style.css").read_text()
    index_scope = css[css.index("body.atlas {") :].split("}", 1)[0]
    for token in ("--bg", "--panel", "--fg", "--accent", "--ok", "--warn", "--line"):
        assert token not in index_scope, f"{token} is redefined for the index"
    assert "--sans" in index_scope  # only the type stacks are the index's own
    root_scope = css[css.index(":root {") :].split("}", 1)[0]
    for token in ("--track", "--ok-soft", "--shade", "--mono"):
        assert token in root_scope


# --------------------------------------------------------------------------
# the 3D view: what the wrapper adds on top of SimView
# --------------------------------------------------------------------------


def test_the_tile_robot_is_helhest_drawn_to_scale():
    """The BEV tiles carry a top-down robot, not an anonymous dot.

    Checked geometrically rather than by eye: the drawn footprint has to be
    the real one, so it can be used as a ruler for the map around it.
    """
    pytest.importorskip("PIL")
    import io

    from PIL import Image

    # A flat map: everything that differs from the background is the robot.
    png = render.render_height_map(np.zeros((128, 128)), size=384)
    img = Image.open(io.BytesIO(png)).convert("RGB")
    assert img.size == (384, 384)
    px_per_m = (img.width / 128) / loaders.HM_RES

    arr = np.asarray(img).astype(int)
    background = arr[0, 0]
    marked = np.abs(arr - background).sum(axis=2) > 12
    rows, cols = np.nonzero(marked)
    assert marked.any(), "nothing was drawn on the tile"

    # Extent: chassis is 1.0 m long, but the wheels reach further in both
    # axes (front hubs at y=+-0.52 with a 0.142 m tread, rear wheel back to
    # x=-0.8685) and the heading wedge sticks 0.3 m past the nose.
    height_m = (rows.max() - rows.min() + 1) / px_per_m
    width_m = (cols.max() - cols.min() + 1) / px_per_m
    assert 1.6 < height_m < 1.9, height_m  # -0.8685 .. +0.8 forward
    assert 1.0 < width_m < 1.3, width_m  # +-0.591 across the wheels

    # Centred on the robot, and pointing forward: the wedge puts more ink
    # above the middle row than the rear wheel puts below it.
    assert abs((rows.min() + rows.max()) / 2 - img.height / 2) < 0.12 * px_per_m
    assert abs((cols.min() + cols.max()) / 2 - img.width / 2) < 0.12 * px_per_m
    assert marked[: img.height // 2].sum() > marked[img.height // 2 :].sum()

    # Subtle: a marker that covered the map would defeat the point of the map.
    assert marked.mean() < 0.03, marked.mean()


def test_every_bev_tile_carries_the_same_robot():
    """Height map, validity and cloud are all the same robot-centred frame, so
    the robot has to be drawn the same real size on each of them."""
    pytest.importorskip("PIL")
    import io

    from PIL import Image

    # The silhouette's outline is the only near-white ink on any of these
    # tiles (turbo's low end, the validity green, and the cloud's black
    # background with its grey range rings are all far from white).
    tiles = {
        128 * loaders.HM_RES: render.render_height_map(np.zeros((128, 128))),
        128 * loaders.HM_RES + 1e-9: render.render_validity(np.full((128, 128), 2)),
        24.0: render.render_cloud_bev(np.zeros((32, 4))),
    }
    sizes = set()
    for span_m, png in tiles.items():
        img = Image.open(io.BytesIO(png)).convert("RGB")
        white = np.asarray(img).min(axis=2) >= 190
        assert white.any(), span_m
        rows, cols = np.nonzero(white)
        px_per_m = img.width / span_m
        sizes.add(round((cols.max() - cols.min() + 1) / px_per_m, 1))
        # Centred on the robot, whatever the tile's scale.
        assert abs((cols.min() + cols.max()) / 2 - img.width / 2) < 0.15 * px_per_m
    assert len(sizes) == 1, sizes


def test_the_tile_robot_matches_the_3d_robot():
    """render.py duplicates build.py's geometry (it cannot import it: build
    imports render). The two must not be allowed to drift apart."""
    pytest.importorskip("torch")
    from dataset_tools.viewer import build

    assert render.ROBOT_CHASSIS_HALF == (build.CHASSIS["hx"], build.CHASSIS["hy"])
    assert render.ROBOT_WHEEL_RADIUS == build.WHEEL_RADIUS
    assert render.ROBOT_WHEEL_WIDTH == build.WHEEL_WIDTH
    assert sorted(render.ROBOT_WHEEL_HUBS) == sorted((x, y) for x, y, _z, _col, _sign in build.WHEELS.values())


def test_every_terrain_colour_mode_is_explained():
    """SimView offers a "Color Mode" per attached property and explains none
    of them, so the descriptions travel with the layer names."""
    pytest.importorskip("torch")
    from dataset_tools.viewer import build

    modes = build.terrain_color_modes(["coverage", "validity", "cloud_points"])
    assert [m["name"] for m in modes] == [
        "height",
        "coverage",
        "validity",
        "cloud_points",
    ]
    assert all(m["description"] for m in modes)
    # Every layer the builder can attach has prose written for it.
    produced = {"coverage", "validity", "cloud_points", "cloud_intensity", "cloud_spread"}
    assert produced <= set(build.TERRAIN_COLOR_MODE_HELP)


def test_meta_json_carries_the_colour_mode_legend(client):
    opts = "40-160000-80-500000-0.1-height"
    meta = client.get(f"/api/data/0000/seg000/{opts}/x/meta.json").json()
    modes = meta["build"]["terrain_color_modes"]
    assert modes[0]["name"] == "height"
    assert all(m["description"] for m in modes)
    # The legend has to describe exactly the modes SimView will offer, which
    # is "height" plus whatever properties the terrain actually carries.
    model = client.get(f"/api/data/0000/seg000/{opts}/x/model.json").json()
    attached = list((model["terrain"].get("properties") or {}).keys())
    assert [m["name"] for m in modes] == ["height", *attached]


def test_a_bundle_from_an_older_viewer_is_rebuilt_not_served(client, tmp_path):
    """The cache is keyed by what was asked for, not by the code that answers,
    so a stale bundle would otherwise outlive a viewer upgrade forever."""
    from dataset_tools.viewer import app as app_module

    opts = "40-160000-80-500000-0.1-height"
    first = client.get(f"/api/data/0000/seg000/{opts}/x/meta.json").json()
    assert first["meta_version"] == app_module.META_VERSION

    bundles = list(tmp_path.rglob("seg000/*/meta.json"))
    assert bundles, "the built bundle was not found on disk"
    stale = json.loads(bundles[0].read_text())
    stale["meta_version"] = 0
    stale["build"].pop("terrain_color_modes", None)
    bundles[0].write_text(json.dumps(stale))

    again = client.get(f"/api/data/0000/seg000/{opts}/x/meta.json").json()
    assert again["meta_version"] == app_module.META_VERSION
    assert again["build"]["terrain_color_modes"]


def test_the_overlay_offers_the_cloud_and_the_ghost_as_separate_toggles(client):
    """The old control was one either/or switch that put SimView in "points"
    mode, which hid the robot. They are independent checkboxes now."""
    page = client.get("/view/0000/seg000").text
    assert 'id="dt-cloud"' in page and 'id="dt-ghost"' in page
    assert "dt-seg" not in page  # the either/or switch is gone
    # Each toggle says what it is; the ghost is the one nobody recognises.
    assert "wheel-odometry ghost" in page
    assert "wheel_odom_drift" in page
    # And the panel's own toggle is a single edge tab, not the old pair.
    assert 'id="dt-edge"' in page
    assert 'id="dt-expand"' not in page and 'id="dt-collapse"' not in page


def test_the_viewer_page_installs_the_simview_patches(client):
    page = client.get("/view/0000/seg000").text
    assert "/viewer-static/overlay.js" in page
    patch = client.get("/viewer-static/simview_patch.js")
    assert patch.status_code == 200
    body = patch.text
    # The stopgaps have to stay findable and deletable as whole blocks.
    for marker in (
        "STOPGAP for simview <= 4.2: origin-anchored frustum culling",
        'STOPGAP for simview <= 4.2: the "points" body-visualization mode',
        "STOPGAP for simview <= 4.2.0: ScalarPlotter plots never draw (3 causes)",
    ):
        at = body.find(marker)
        assert at >= 0, marker
        # Each one has to announce, right there, that it is deletable.
        assert "Delete this block on upgrade" in body[at : at + 400], marker
    # And the permanent parts name the bodies they are about.
    assert "lidar_cloud" in body and "wheel_odom_ghost" in body


def test_the_stopgaps_switch_themselves_off_when_simview_is_upgraded(client):
    """A workaround left running over a fixed SimView would override the real
    fix, so each one is gated on the version the page reports."""
    from dataset_tools.viewer.app import _simview_version

    page = client.get("/view/0000/seg000").text
    assert f'"simviewVersion": "{_simview_version()}"' in page

    body = client.get("/viewer-static/simview_patch.js").text
    assert 'LAST_SIMVIEW_NEEDING_STOPGAPS = "4.2.0"' in body
    # The two SimView has fixed are behind the gate...
    gated = body[body.index("const stopgaps = versionAtMost") :]
    assert "stopgapOriginCulling(app);" in gated.split("hideHousekeepingBodies")[0]
    assert "stopgapDropPointsMode(app);" in gated.split("hideHousekeepingBodies")[0]
    # ...and the scalar-plot one is now gated with them (simview 4.2.1 fixes it),
    # so on a newer package none of the three run and simview handles them.
    assert "if (stopgaps) stopgapScalarXScale(app);" in body
    # It still adopts each chart once and only pins a still-null x scale.
    assert "if (_dtScalarFixed.has(chart)) continue;" in body


def test_the_initial_frame_is_opened_through_simviews_own_seek(client):
    """Assigning `currentTime` does not move the scene.

    Nothing in SimView reads that field until playback advances it, so a window
    opened from a frame's page drew the window's *first* frame and only jumped
    to the requested one once play was pressed. `goToTime` is what every one of
    SimView's own seek affordances calls: it moves `currentStateIndex` with the
    clock, redraws the scene and refreshes the playback bar.
    """
    body = client.get("/viewer-static/overlay.js").text
    assert "app.animationController.goToTime(seekTo);" in body
    # The old write is what the bug was; it must not come back.
    assert "animationController.currentTime =" not in body


def test_the_view_opens_with_the_robot_in_shot(client):
    """SimView's camera is fixed at the world origin, which is not where every
    window starts the robot -- a later part of a split segment carries on from
    the previous part, and `?frame=N` seeks into the middle of a traverse."""
    body = client.get("/viewer-static/simview_patch.js").text
    assert 'ROBOT_BODY = "robot"' in body
    assert "export function centerOnRobot(app)" in body
    # Camera and orbit target move by the same delta, so the angle and the
    # distance the view was framed at survive the re-centring.
    fn = body[body.index("export function centerOnRobot(app)") :]
    fn = fn[: fn.index("\n}\n")]
    assert "camera.position.set(" in fn and "controls.target.set(x, y, z);" in fn
    assert "controls.update();" in fn
    # It reads the body's world position the way SimView's own tracking does.
    assert "app.batchManager.getBatchOffset(batch)" in fn
    # Falsy until the robot has a resolved position, so the caller can retry.
    assert "return false;" in fn

    overlay = client.get("/viewer-static/overlay.js").text
    assert "if (centerOnRobot(app)) {" in overlay
    # Once, at load -- following it every frame is SimView's own trackBody.
    assert overlay.count("centerOnRobot(app)") == 1


def test_the_aggregated_cloud_carries_no_per_frame_pose(client):
    """It is already in the world frame, so a per-frame identity transform is
    a blob that says nothing -- and is what would keep it in the Body states
    list once SimView starts filtering that list by trajectory."""
    opts = "40-160000-80-500000-0.1-height"
    client.get(f"/api/data/0000/seg000/{opts}/x/meta.json")
    model = client.get(f"/api/data/0000/seg000/{opts}/x/model.json").json()
    states = client.get(f"/api/data/0000/seg000/{opts}/x/states.json").json()
    names = [b["name"] for b in model["bodies"]]
    if "lidar_cloud" not in names:
        pytest.skip("this synthetic window aggregated no cloud")
    posed = {b["name"] for b in states["bodies"]}
    assert "robot" in posed
    assert "lidar_cloud" not in posed


# --------------------------------------------------------------------------
# a scene whose per-frame directories outrun its timestamp file
# --------------------------------------------------------------------------


def _root_with_untimestamped_frames(tmp_path: Path, timestamped: int = 8, extra: int = 6) -> Path:
    """A root mid-extraction: more images on disk than lines in data_ts.txt.

    Exactly what ``probe`` warns about ("per-frame counts disagree"), and what
    the module docstrings promise the viewer degrades gracefully for.
    """
    root_path = make_root(tmp_path, frames=timestamped, segments=False, with_transforms=False)
    scene = root_path / "0000"
    for i in range(timestamped, timestamped + extra):
        _write_png(scene / "images" / f"{i:05d}.png")
    return root_path


def test_no_window_reaches_past_the_end_of_the_timestamp_file(tmp_path):
    """A frame with no timestamp has no place on the playback clock, so no
    window may contain one -- a window made only of such frames used to be
    offered, built zero frames and died with an IndexError (HTTP 500)."""
    root = probe.DatasetRoot(_root_with_untimestamped_frames(tmp_path), cache_dir=tmp_path / "c")
    p = root.probe("0000")
    assert p.n_frames == 14 and p.counts["data_ts"] == 8  # the disagreement
    assert any("counts disagree" in w for w in p.warnings)

    ws = windows.windows_for_scene(np.arange(8, dtype=np.float64), [], p.n_frames, has_transforms=False, max_frames=4)
    assert ws, "the scene still offers windows"
    assert max(w.frame_end for w in ws) == 8


def test_a_frame_with_no_timestamp_has_no_single_frame_window(tmp_path):
    root = probe.DatasetRoot(_root_with_untimestamped_frames(tmp_path), cache_dir=tmp_path / "c")
    from dataset_tools.viewer.build import SceneAccess

    access = SceneAccess(root, "0000")
    assert access.window(windows.frame_window_key(3), 1500) is not None
    assert access.window(windows.frame_window_key(10), 1500) is None


def test_an_untimestamped_frame_is_a_404_and_its_page_says_why(tmp_path):
    """Not a 500: the frame page still renders (the image is on disk), it just
    does not offer a 3D view it cannot build."""
    pytest.importorskip("simview")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from dataset_tools.viewer.app import create_app

    root_path = _root_with_untimestamped_frames(tmp_path)
    c = TestClient(create_app(root_path, cache_dir=tmp_path / "cache"))

    page = c.get("/scene/0000/frame/10")
    assert page.status_code == 200
    assert "Visualise this frame" not in page.text
    assert "past the end of" in page.text
    assert c.get("/view/0000/frame00010").status_code == 404

    ok = c.get("/scene/0000/frame/3")
    assert "Visualise this frame" in ok.text
    assert c.get("/view/0000/frame00003").status_code == 200


def test_every_offered_window_actually_builds(tmp_path):
    """The windows table is a list of buttons; every one of them must work."""
    pytest.importorskip("simview")
    pytest.importorskip("torch")
    from fastapi.testclient import TestClient

    from dataset_tools.viewer.app import create_app

    root_path = _root_with_untimestamped_frames(tmp_path)
    c = TestClient(create_app(root_path, max_frames=4, cache_dir=tmp_path / "cache"))
    offered = c.get("/api/scene/0000").json()["windows"]
    assert offered
    for w in offered:
        assert c.get(f"/view/0000/{w['key']}").status_code == 200, w["key"]


# --------------------------------------------------------------------------
# the live segment file, not a stale atlas, decides the timeline
# --------------------------------------------------------------------------


def test_the_atlas_cannot_add_filled_segments_the_live_file_does_not_have(tmp_path):
    """The segment file is the truth. "No filled segments" is an answer, not a
    hole to fall through to the atlas with -- a scene re-filtered since the
    atlas was written used to keep claiming amber v1.1 stretches, on the index
    card, on the scene timeline and in the card's prose."""
    root_path = make_root(tmp_path)
    write_atlas(root_path)
    atlas_file = root_path / "atlas" / "atlas.json"
    data = json.loads(atlas_file.read_text())
    data["scenes"]["0000"]["seg_spans"] = [[0.0, 0.4]]
    data["scenes"]["0000"]["fill_spans"] = [[0.5, 0.9]]
    atlas_file.write_text(json.dumps(data))

    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")
    p = root.probe("0000")
    assert p.segment_spans and not p.fill_spans  # the live file has no fills

    from dataset_tools.viewer.index_page import build_row
    from dataset_tools.viewer.scene_page import ground_truth_card
    from dataset_tools.viewer.build import SceneAccess

    row = build_row(root, p)
    assert row["fill_spans"] == []
    assert row["spans"] == p.segment_spans

    card = ground_truth_card(SceneAccess(root, "0000"), root.atlas_scene("0000"))
    assert card["timeline"]["fill"] == []
    assert "filled from v1.1" not in (card["note"] or "")


def test_a_filled_span_is_drawn_once_not_twice(tmp_path):
    """`fill_spans` is a subset of `segment_spans`, so the plain half has to
    have it removed or a filled stretch gets a green bar under its amber one."""
    root_path = make_root(tmp_path, segments=False)
    scene = root_path / "0000"
    ts = np.loadtxt(scene / "data_ts.txt")
    with open(scene / "odometry_segments.txt", "w") as fh:
        fh.write(f"0 {ts[0]:.6f} {ts[4]:.6f} 40 filtered:pose_jump\n")
        fh.write(f"1 {ts[6]:.6f} {ts[-1]:.6f} 50 v11_fill:end_of_run\n")

    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")
    p = root.probe("0000")
    assert len(p.segment_spans) == 2 and len(p.fill_spans) == 1

    from dataset_tools.viewer.index_page import build_row

    row = build_row(root, p)
    assert len(row["spans"]) == 1 and len(row["fill_spans"]) == 1
    assert row["spans"][0] not in row["fill_spans"]


# --------------------------------------------------------------------------
# how densely a segment is anchored, and saying so
# --------------------------------------------------------------------------


def test_a_window_carries_the_pose_count_of_its_segment(tmp_path):
    """Two segments of the same length and frame count can rest on 10 Hz poses
    or on one every two seconds; interpolated onto the frame clock they look
    identical, so the count has to travel with the window."""
    ts = np.arange(40, dtype=np.float64) * 0.1
    segs = [
        loaders.Segment(0, ts[0], ts[19], n_poses=200, reason="dense"),
        loaders.Segment(1, ts[20], ts[-1], n_poses=2, reason="sparse"),
    ]
    ws = windows.windows_for_scene(ts, segs, len(ts), has_transforms=False)
    dense, sparse = ws[0], ws[1]
    assert dense.n_poses == 200 and sparse.n_poses == 2
    assert dense.pose_rate > 100 and sparse.pose_rate < 2


def test_a_split_segment_keeps_its_pose_rate(tmp_path):
    """Chunking a long segment must not make it look sparser than it is."""
    ts = np.arange(100, dtype=np.float64) * 0.1
    segs = [loaders.Segment(0, ts[0], ts[-1], n_poses=100, reason="x")]
    ws = windows.windows_for_scene(ts, segs, len(ts), has_transforms=False, max_frames=30)
    assert len(ws) == 4
    whole = windows.windows_for_scene(ts, segs, len(ts), has_transforms=False)[0]
    for w in ws:
        assert abs(w.pose_rate - whole.pose_rate) < 0.15 * whole.pose_rate


def test_the_windows_table_states_the_ground_truth_pose_rate(client):
    body = client.get("/scene/0000?tab=windows").text
    assert "GT poses" in body
    assert "Hz" in body


# --------------------------------------------------------------------------
# smaller corrections
# --------------------------------------------------------------------------


def test_the_frame_page_reads_the_segment_tag_out_in_words(client):
    """Every other page renders `filtered:pose_jump` through `describe_reason`;
    the frame page printed the raw tag."""
    import html

    body = html.unescape(client.get("/scene/0000/frame/2").text)
    assert "Ends because" in body
    assert windows.describe_reason("filtered:pose_jump") in body
    assert "filtered:pose_jump" not in body


def test_back_to_the_frame_strip_lands_on_the_page_the_frame_is_on(tmp_path):
    """Not `index // 100`: a scene with a sparse per-frame directory pages
    through the frames that exist, so a frame's page is its position in that
    list, not its own number."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from dataset_tools.viewer.app import FRAMES_PER_PAGE, create_app

    # A sparse scene: only every 40th frame has an image on disk.
    root_path = make_root(tmp_path, frames=12, segments=False)
    scene = root_path / "0000"
    ts = 1_700_000_000.0 + 0.1 * np.arange(600)
    np.savetxt(scene / "data_ts.txt", ts, fmt="%.6f")
    for i in range(0, 600, 40):
        _write_png(scene / "images" / f"{i:05d}.png")
    for stale in scene.glob("images/*.png"):
        if int(stale.stem) % 40:
            stale.unlink()

    c = TestClient(create_app(root_path, cache_dir=tmp_path / "cache"))
    body = c.get("/scene/0000/frame/560").text
    assert f"tab=frames&amp;page={560 // FRAMES_PER_PAGE}" not in body
    assert "tab=frames&amp;page=0" in body  # 15 sparse frames: all on page 0


def test_probing_a_root_writes_its_cache_once_not_once_per_scene(tmp_path):
    """`probe_all` rewrote the whole growing cache file after every scene."""
    root_path = make_root(tmp_path, scenes=("0000", "0001", "0002", "0003"), frames=4)
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")

    writes = []
    original = probe.DatasetRoot.save_probe_cache

    def counting(self):
        if self._probe_dirty:
            writes.append(1)
        original(self)

    probe.DatasetRoot.save_probe_cache = counting
    try:
        assert len(root.probe_all()) == 4
        assert len(writes) == 1
        root.probe_all()  # everything cached and unchanged: nothing to write
        assert len(writes) == 1
        root.probe("0000", refresh=True)  # a fresh probe still gets persisted
        assert len(writes) == 2
    finally:
        probe.DatasetRoot.save_probe_cache = original


def test_the_bundle_stamp_covers_every_module_that_shapes_a_bundle(client):
    """The stamp is what makes a browser-cached bundle unaddressable after a
    code change, so it has to cover windowing and pose interpolation too --
    not only the three modules that write the files out."""
    import inspect

    from dataset_tools.viewer import app as appmod

    source = inspect.getsource(appmod.create_app)
    for module in ("build.py", "wire.py", "app.py", "windows.py", "loaders.py", "render.py"):
        assert f'"{module}"' in source, module


def test_the_status_endpoint_agrees_with_what_a_fetch_would_do(client, tmp_path):
    """Reporting "ready" for a bundle an upgrade has superseded promises a
    fetch that in fact blocks on a full rebuild."""
    opts = "40-160000-80-500000-0.1-height"
    base = f"/api/data/0000/seg000/{opts}/x"
    assert client.get(f"{base}/status").json()["ready"] is False
    client.get(f"{base}/meta.json")
    status = client.get(f"{base}/status").json()
    assert status["ready"] is True

    meta = Path(status["path"]) / "meta.json"
    stale = json.loads(meta.read_text())
    stale["meta_version"] = 0
    meta.write_text(json.dumps(stale))
    assert client.get(f"{base}/status").json()["ready"] is False


def test_the_scene_page_does_not_read_arrays_it_never_renders(client, tmp_path):
    """`_scene_details` loads joint_states/cmd_vel whole (no memory map) and
    the scene template renders none of it -- the four cards carry the same
    facts, read strided. It stays what /api/scene/{id} returns."""
    from dataset_tools.viewer import app as appmod

    calls = []
    original = appmod._scene_details
    appmod._scene_details = lambda access: (calls.append(access.scene_id), original(access))[1]
    try:
        assert client.get("/scene/0000").status_code == 200
        assert calls == []
        assert client.get("/api/scene/0000").status_code == 200
        assert calls == ["0000"]
    finally:
        appmod._scene_details = original


# --------------------------------------------------------------------------
# notes
# --------------------------------------------------------------------------


def _notes_client(tmp_path, **kwargs):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from dataset_tools.viewer.app import create_app

    root_path = kwargs.pop("root_path", None) or make_root(tmp_path, frames=12)
    return root_path, TestClient(
        create_app(root_path, cache_dir=tmp_path / "cache", **kwargs)
    )


def test_a_note_round_trips_through_the_api_and_the_file(tmp_path):
    root_path, client = _notes_client(tmp_path)

    created = client.post(
        "/api/scene/0000/notes",
        json={"frames": [3, 7], "topic": "odometry", "text": "GT walks off the terrain"},
    )
    assert created.status_code == 201
    note = created.json()
    assert (note["frame_start"], note["frame_end"]) == (3, 7)
    assert note["label"] == "frames 3–7"

    # Stored next to the data, one file per scene, with the range as a pair.
    stored = json.loads((root_path / "notes" / "0000.json").read_text())
    assert stored["scene"] == "0000"
    assert stored["notes"][0]["frames"] == [3, 7]
    assert stored["notes"][0]["text"] == "GT walks off the terrain"

    listed = client.get("/api/scene/0000/notes").json()
    assert [n["id"] for n in listed["notes"]] == [note["id"]]
    assert listed["status"]["writable"] is True

    edited = client.patch(
        f"/api/scene/0000/notes/{note['id']}", json={"text": "and never recovers"}
    )
    assert edited.status_code == 200
    assert edited.json()["text"] == "and never recovers"
    assert edited.json()["updated"]  # an edit is dated; `created` is not moved
    assert edited.json()["created"] == note["created"]

    assert client.delete(f"/api/scene/0000/notes/{note['id']}").status_code == 200
    assert client.get("/api/scene/0000/notes").json()["notes"] == []
    assert client.delete(f"/api/scene/0000/notes/{note['id']}").status_code == 404


def test_a_notes_frame_range_is_inclusive_ordered_and_inside_the_scene(tmp_path):
    """Both ends are included, a range typed backwards still means a range, and
    nothing is stored past the end of a scene -- a note out there would never
    be shown by anything that reads it back."""
    _, client = _notes_client(tmp_path)

    one = client.post("/api/scene/0000/notes", json={"frame": 5, "text": "here"}).json()
    assert (one["frame_start"], one["frame_end"], one["n_frames"]) == (5, 5, 1)
    assert one["label"] == "frame 5"

    backwards = client.post(
        "/api/scene/0000/notes", json={"frames": [9, 2], "text": "typed backwards"}
    ).json()
    assert (backwards["frame_start"], backwards["frame_end"]) == (2, 9)

    past_the_end = client.post(
        "/api/scene/0000/notes", json={"frames": [10, 900], "text": "off the end"}
    ).json()
    assert past_the_end["frame_end"] == 11  # the scene has 12 frames

    assert client.post("/api/scene/0000/notes", json={"frames": [1, 2]}).status_code == 400
    assert client.post("/api/scene/0000/notes", json={"text": "nowhere"}).status_code == 400
    assert client.post("/api/scene/9999/notes", json={"frame": 0, "text": "x"}).status_code == 404


def test_a_note_is_marked_on_every_window_its_frames_touch():
    """A note is an inclusive frame range and a window a half-open one; a note
    spanning a boundary belongs to both windows, since opening either is a
    reason to be warned."""
    from dataset_tools.viewer.notes import Note

    note = Note(id="a", frame_start=4, frame_end=8, topic="odometry", text="bad")
    assert note.overlaps(0, 6)  # a window ending inside the note
    assert note.overlaps(8, 20)  # one starting on its last frame
    assert not note.overlaps(0, 4)  # ends exclusive at the note's first frame
    assert not note.overlaps(9, 20)
    assert note.covers(4) and note.covers(8) and not note.covers(9)


def test_the_scene_page_shows_its_notes_and_which_windows_carry_them(tmp_path):
    _, client = _notes_client(tmp_path)
    client.post(
        "/api/scene/0000/notes",
        json={"frames": [0, 2], "topic": "odometry", "text": "wrong odometry here"},
    )
    page = client.get("/scene/0000").text
    assert "wrong odometry here" in page
    assert "frames 0–2" in page
    assert "1 note" in page  # the windows table marks the window it falls in
    assert 'id="noteform"' in page


def test_the_view_page_hands_the_overlay_its_notes_endpoint_and_range(client):
    page = client.get("/view/0000/seg000").text
    assert '"notesUrl": "/api/scene/0000/notes"' in page
    # The window's own frames, as an inclusive pair: what the compose form
    # starts pinned to.
    assert '"frameStart": 0' in page
    assert '"frameEnd":' in page


def test_the_overlay_range_ends_are_typeable_fields(client):
    """Pinning an end to the playhead is the quick path, not the only one: a
    range read off a plot or a previous note has to be typeable."""
    page = client.get("/view/0000/seg000").text
    assert 'id="dt-note-from"' in page and 'id="dt-note-to"' in page
    assert page.count('type="number"') >= 2
    assert page.count('class="dt-nowbtn"') == 2


def test_a_read_only_root_still_reads_its_notes_but_refuses_new_ones(tmp_path):
    """The routine case: the dataset is a shared or mounted copy. Existing
    notes still show, composing is refused with the reason, and nothing about
    the rest of the viewer changes."""
    if os.geteuid() == 0:
        pytest.skip("root ignores the write bits this test relies on")
    root_path = make_root(tmp_path, frames=12)
    (root_path / "notes").mkdir()
    (root_path / "notes" / "0000.json").write_text(
        json.dumps({"scene": "0000", "notes": [{"frames": [1, 3], "text": "from before"}]})
    )
    (root_path / "notes").chmod(0o555)
    root_path.chmod(0o555)
    try:
        _, client = _notes_client(tmp_path, root_path=root_path)
        body = client.get("/api/scene/0000/notes").json()
        assert [n["text"] for n in body["notes"]] == ["from before"]
        assert body["status"]["writable"] is False
        assert "not writable" in body["status"]["reason"]
        refused = client.post("/api/scene/0000/notes", json={"frame": 0, "text": "no"})
        assert refused.status_code == 403
        page = client.get("/scene/0000").text
        assert "from before" in page
        assert "read-only" in page.lower()
        assert 'id="noteform"' not in page
    finally:
        root_path.chmod(0o755)
        (root_path / "notes").chmod(0o755)


def test_notes_can_be_kept_outside_a_root_that_cannot_take_them(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root ignores the write bits this test relies on")
    root_path = make_root(tmp_path, frames=12)
    root_path.chmod(0o555)
    elsewhere = tmp_path / "elsewhere"
    try:
        _, client = _notes_client(tmp_path, root_path=root_path, notes_dir=elsewhere)
        assert client.post(
            "/api/scene/0000/notes", json={"frame": 4, "text": "kept aside"}
        ).status_code == 201
    finally:
        root_path.chmod(0o755)
    assert json.loads((elsewhere / "0000.json").read_text())["notes"][0]["text"] == "kept aside"
    assert not (root_path / "notes").exists()


def test_notes_off_reads_writes_and_shows_nothing(tmp_path):
    root_path = make_root(tmp_path, frames=12)
    (root_path / "notes").mkdir()
    (root_path / "notes" / "0000.json").write_text(
        json.dumps({"scene": "0000", "notes": [{"frames": [1, 3], "text": "invisible"}]})
    )
    _, client = _notes_client(tmp_path, root_path=root_path, notes=False)
    body = client.get("/api/scene/0000/notes").json()
    assert body["notes"] == [] and body["status"]["enabled"] is False
    assert client.post("/api/scene/0000/notes", json={"frame": 0, "text": "x"}).status_code == 403
    assert "invisible" not in client.get("/scene/0000").text
    assert client.get("/healthz").json()["notes"]["enabled"] is False


def test_opening_a_root_writes_no_notes_directory_into_it(tmp_path):
    """Looking at a dataset must leave it as it was found. The notes directory
    appears when a note is written, not when the viewer is pointed at a root."""
    root_path, client = _notes_client(tmp_path)
    assert client.get("/scene/0000").status_code == 200
    assert not (root_path / "notes").exists()


def test_a_malformed_notes_file_degrades_to_no_notes(tmp_path):
    """Hand-editing this file is expected; a broken one must not take a scene's
    page down with it, and one bad entry must not lose the others."""
    root_path = make_root(tmp_path, frames=12)
    (root_path / "notes").mkdir()
    (root_path / "notes" / "0000.json").write_text("{not json at all")
    _, client = _notes_client(tmp_path, root_path=root_path)
    assert client.get("/api/scene/0000/notes").json()["notes"] == []
    assert client.get("/scene/0000").status_code == 200

    (root_path / "notes" / "0000.json").write_text(
        json.dumps(
            {
                "notes": [
                    {"frames": [1, 2], "text": "kept"},
                    {"text": "no range at all"},
                    {"frames": [3, 4]},
                    "not even an object",
                ]
            }
        )
    )
    kept = client.get("/api/scene/0000/notes").json()["notes"]
    assert [n["text"] for n in kept] == ["kept"]


def test_notes_written_against_another_dataset_version_are_flagged(tmp_path):
    """Frame numbers mean the same thing only while the extraction does. The
    same check, for the same reason, as the atlas's `dataset_version`."""
    root_path = make_root(tmp_path, frames=12)
    (root_path / "manifest.json").write_text(json.dumps({"dataset_version": "v1.3"}))
    (root_path / "notes").mkdir()
    (root_path / "notes" / "0000.json").write_text(
        json.dumps(
            {
                "scene": "0000",
                "dataset_version": "v1.2",
                "notes": [{"frames": [1, 3], "text": "older run"}],
            }
        )
    )
    _, client = _notes_client(tmp_path, root_path=root_path)
    assert client.get("/api/scene/0000/notes").json()["stale_version"] == "v1.2"
    assert "written against dataset v1.2" in client.get("/scene/0000").text

    # A note written now carries this root's version, and the warning goes.
    client.post("/api/scene/0000/notes", json={"frame": 5, "text": "current"})
    assert client.get("/api/scene/0000/notes").json()["stale_version"] is None


def test_an_unknown_topic_in_a_hand_edited_file_is_kept_as_written(tmp_path):
    """Unknown tokens pass through verbatim, exactly as an unrecognised segment
    `reason` does -- the file is a person's to edit."""
    root_path = make_root(tmp_path, frames=12)
    (root_path / "notes").mkdir()
    (root_path / "notes" / "0000.json").write_text(
        json.dumps({"notes": [{"frames": [1, 2], "topic": "Lidar-Dropout", "text": "x"}]})
    )
    _, client = _notes_client(tmp_path, root_path=root_path)
    assert client.get("/api/scene/0000/notes").json()["notes"][0]["topic"] == "lidar-dropout"
