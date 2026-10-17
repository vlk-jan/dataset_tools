import json
from pathlib import Path

import numpy as np
import pytest

from dataset_tools import heightmap as hmap

FIXTURE = Path(__file__).parent / "data" / "heightmap_parity.npz"


def test_maps_match_drifts_builder():
    """Three HELHEST v2.4 frames (meadow, paved path, forest; a third of each cloud) as DRIFT's
    scripts/data/build_heightmaps.py built them at the v2 settings, 2026-10-12."""
    d = np.load(FIXTURE)
    for k in range(3):
        hm, val = hmap.build_frame(d[f"cloud_{k}"], d[f"R_align_{k}"])
        np.testing.assert_allclose(hm, d[f"height_map_{k}"], atol=1e-9, rtol=0)
        np.testing.assert_array_equal(val, d[f"validity_{k}"])
        assert set(np.unique(val)) <= {0, 1, 2, 3} and (val == hmap.VALIDITY_ANCHORED).any()


def test_a_sparse_cloud_has_no_map():
    cloud = np.zeros((100, 4), np.float32)
    cloud[:, 0] = np.linspace(1, 5, 100)
    assert hmap.build_frame(cloud, np.eye(3)) == (None, None)


def _scene(tmp_path, n=3):
    d = tmp_path / "0000"
    (d / "clouds").mkdir(parents=True)
    fx = np.load(FIXTURE)
    for i in range(n):
        np.save(d / "clouds" / f"{i:05d}.npy", fx[f"cloud_{i % 3}"])
    np.save(d / "gravity_align.npy", np.stack([fx[f"R_align_{i % 3}"] for i in range(n)]))
    return d


def test_build_scene_writes_maps_and_records_the_settings(tmp_path):
    d = _scene(tmp_path)
    c = hmap.build_scene(d, workers=1)
    assert c["ok"] == 3 and c["maps"] == 3 and not c["errors"]
    rec = json.loads((d / hmap.SETTINGS_FILE).read_text())
    assert hmap.Params.from_dict(rec["parameters"]) == hmap.Params()
    assert hmap.build_scene(d, workers=1)["skip"] == 3
    fails, checked = hmap.verify_against(tmp_path, hmap.Params(), n_frames=2)
    assert checked == 2 and not fails


def test_other_settings_need_overwrite_and_drop_maps_gone_sparse(tmp_path):
    d = _scene(tmp_path)
    hmap.build_scene(d, workers=1)
    other = hmap.Params(statistic="median")
    with pytest.raises(SystemExit):
        hmap.build_scene(d, other, workers=1)
    sparse = hmap.Params(min_points=10**6)
    c = hmap.build_scene(d, sparse, workers=1, overwrite=True)
    assert c["sparse"] == 3 and c["maps"] == 0
    assert not list((d / "height_masks").iterdir())


def test_building_through_a_linked_scene_is_refused(tmp_path):
    owner = _scene(tmp_path / "v2", n=1)
    hmap.build_scene(owner, workers=1)
    d = tmp_path / "v3" / "0000"
    d.mkdir(parents=True)
    for name in ("clouds", "gravity_align.npy", "height_maps", "height_masks", hmap.SETTINGS_FILE):
        (d / name).symlink_to(Path("../../v2/0000") / name)
    with pytest.raises(SystemExit, match="v2/0000/height_maps"):
        hmap.build_scene(d, workers=1, overwrite=True)
    assert (owner / "height_maps" / "00000.npz").exists() and (d / hmap.SETTINGS_FILE).is_symlink()


def test_a_crashed_map_write_leaves_nothing_that_counts_as_a_map(tmp_path, monkeypatch):
    def boom(*_):
        raise OSError("disk full")

    monkeypatch.setattr(hmap.os, "replace", boom)
    with pytest.raises(OSError):
        hmap._write_npz(str(tmp_path / "00000.npz"), height_map=np.zeros(2))
    assert not [f for f in tmp_path.iterdir() if f.name.endswith(".npz")]


def test_verify_against_catches_a_changed_map(tmp_path):
    d = _scene(tmp_path, n=1)
    hmap.build_scene(d, workers=1)
    f = d / "height_maps" / "00000.npz"
    hm = np.load(f)["height_map"]
    np.savez_compressed(f, height_map=hm + 0.01)
    fails, checked = hmap.verify_against(tmp_path, hmap.Params(), n_frames=1)
    assert checked == 1 and "max diff" in fails[0]
