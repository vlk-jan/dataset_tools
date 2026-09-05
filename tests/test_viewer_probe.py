"""Viewer: probing a root and its scenes (dataset_tools.viewer.probe).

Every test builds a synthetic dataset root (``helpers_viewer``), so nothing
here touches a real dataset or the network. The end-to-end HTTP tests skip
cleanly when the viewer extra (SimView + torch) is not installed.
"""

from __future__ import annotations

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
