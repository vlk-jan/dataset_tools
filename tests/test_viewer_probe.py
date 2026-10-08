"""Viewer: probing a root and its scenes (dataset_tools.viewer.probe)."""

from __future__ import annotations

import os

from helpers_viewer import (
    make_root,
)

from dataset_tools.viewer import probe

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


def test_every_artefact_says_what_it_is_for(tmp_path):
    """The scene page shows this prose in place of a tooltip, so none may be blank."""
    for art in probe.ARTEFACTS:
        assert art.purpose, f"{art.relpath} has no purpose written for it"
        assert art.purpose[0].isupper() and art.purpose.rstrip().endswith(".")
        assert art.count_unit in ("files", "rows", "lines")


def test_gps_is_probed_like_any_other_optional_file(tmp_path):
    """Absent on most roots, counted like the rest of them where it exists."""
    root_path = make_root(tmp_path)
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")
    assert not root.probe("0000").has("gps")
    assert "gps" in root.probe("0000").missing

    (tmp_path / "gps").mkdir()
    with_gps = probe.DatasetRoot(make_root(tmp_path / "gps", with_gps=True), cache_dir=tmp_path / "cache2")
    p = with_gps.probe("0000")
    assert p.has("gps") and p.counts["gps"] == 48
    # The frame count is unaffected: gps.npy is not frame-aligned.
    assert p.n_frames == 12


def test_probe_cache_is_reused_and_invalidated_by_an_mtime_change(tmp_path):
    root_path = make_root(tmp_path)
    cache = tmp_path / "cache"
    root = probe.DatasetRoot(root_path, cache_dir=cache)
    first = root.probe("0000")
    assert (cache / "probe_cache.json").is_file()
    # A second DatasetRoot reads the persisted probe rather than re-listing.
    again = probe.DatasetRoot(root_path, cache_dir=cache)
    assert again.probe("0000").n_frames == first.n_frames


def test_clear_cache_removes_the_probe_and_bundles_but_nothing_else(tmp_path):
    root_path = make_root(tmp_path)
    cache = tmp_path / "cache"
    root = probe.DatasetRoot(root_path, cache_dir=cache)
    root.probe("0000")
    (cache / "bundles" / "0000").mkdir(parents=True)
    (cache / "bundles" / "0000" / "states.json").write_text("{}")
    keep = cache / "notes.txt"  # --cache-dir can point at a directory of one's own
    keep.write_text("mine")

    files, freed = probe.clear_cache_dir(cache)
    assert files == 2 and freed > 0
    assert not (cache / "probe_cache.json").exists()
    assert not (cache / "bundles").exists()
    assert keep.read_text() == "mine"


def test_clear_cache_on_a_root_forces_a_fresh_probe(tmp_path):
    root_path = make_root(tmp_path)
    cache = tmp_path / "cache"
    probe.DatasetRoot(root_path, cache_dir=cache).probe("0000")
    again = probe.DatasetRoot(root_path, cache_dir=cache, clear_cache=True)
    assert again._probe_cache == {}
    assert not (cache / "probe_cache.json").exists()
    assert again.probe("0000").n_frames  # re-probed from disk


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


def test_a_root_of_camera_jpegs_is_counted_and_found(tmp_path):
    """The images are .jpg in a root made after 2026-09-29; the viewer reads
    the extension off the root instead of assuming one."""
    from dataset_tools.viewer import build

    root_path = make_root(tmp_path)
    for p in (root_path / "0000" / "images").iterdir():
        p.rename(p.with_suffix(".jpg"))
    (root_path / "manifest.json").write_text('{"images": {"extension": "jpg"}}')
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")
    p = root.probe("0000")
    assert p.has("images") and p.counts["images"] == 12 and p.n_frames == 12
    access = build.SceneAccess(root, "0000")
    assert access.frame_path("images", 3).name == "00003.jpg" and access.frame_path("images", 3).is_file()
    assert access.frame_path("clouds", 3).name == "00003.npy"
    assert len(access.available_frames("images")) == 12


def test_a_header_comment_in_data_ts_does_not_cost_the_scene_its_duration(tmp_path):
    root_path = make_root(tmp_path, frames=12)
    ts_file = root_path / "0000" / "data_ts.txt"
    ts_file.write_text("# seconds, Unix epoch\n\n" + ts_file.read_text())
    p = probe.probe_scene(root_path / "0000")
    assert p.n_frames == 12
    assert p.duration is not None and abs(p.duration - 1.1) < 1e-6


def test_a_scene_added_after_startup_is_found_without_a_restart(tmp_path):
    root_path = make_root(tmp_path, scenes=("0000",))
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")
    assert [p.scene_id for p in root.probe_all()] == ["0000"]
    # An array job finishes another scene while the viewer is up.
    make_root(tmp_path, scenes=("0001",))
    assert [p.scene_id for p in root.probe_all()] == ["0000", "0001"]
    # And a direct request for it resolves without the index having been visited first.
    again = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache2")
    make_root(tmp_path, scenes=("0002",))
    assert again.scene_dir("0002") == root_path / "0002"


def test_files_added_inside_a_scene_reach_the_probe_even_across_a_restart(tmp_path):
    """The persisted probe notices changes inside per-frame dirs, not just the scene dir."""
    from helpers_viewer import make_root

    root_path = make_root(tmp_path, scenes=("0000",), frames=6)
    cache = tmp_path / "cache"
    maps = root_path / "0000" / "height_maps"
    # Backdate first: mtimes tick coarsely, so the unlink's update would often be invisible.
    os.utime(maps, ns=(0, maps.stat().st_mtime_ns - 10_000_000_000))
    before = probe.DatasetRoot(root_path, cache_dir=cache).probe("0000")
    victim = sorted(maps.iterdir())[0]
    victim.unlink()

    after = probe.DatasetRoot(root_path, cache_dir=cache).probe("0000")  # reads the persisted cache
    assert after.counts["height_maps"] == before.counts["height_maps"] - 1
    assert after.fingerprint != before.fingerprint


def test_clear_cache_removes_another_viewers_probe_tmp_file(tmp_path):
    """The tmp name carries a pid, so two viewers on one root do not clobber each other's."""
    root_path = make_root(tmp_path)
    cache = tmp_path / "cache"
    probe.DatasetRoot(root_path, cache_dir=cache).probe("0000")
    stale = cache / f"probe_cache.{os.getpid() + 1}.tmp"
    stale.write_text("{}")
    probe.clear_cache_dir(cache)
    assert not stale.exists() and not (cache / "probe_cache.json").exists()
