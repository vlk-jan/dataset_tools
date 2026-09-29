import json

from dataset_tools import disk_usage


def make_scene(root, sid, n_images, img_bytes=10):
    d = root / sid
    (d / "images").mkdir(parents=True)
    for i in range(n_images):
        (d / "images" / f"{i:05d}.png").write_bytes(b"x" * img_bytes)
    (d / "data_ts.txt").write_text("100.0\n102.5\n")


def test_incremental_update_keeps_hand_edits(tmp_path):
    make_scene(tmp_path, "0000", 3)
    doc = disk_usage.update(tmp_path)
    s = doc["scenes"]["0000"]
    assert s["modalities"]["images"] == {"bytes": 30, "files": 3}
    assert s["modalities"]["other"] == {"bytes": 12, "files": 1}
    assert s["files"] == {"data_ts.txt": 12}
    assert s["frames"] == 3 and s["span_s"] == 2.5 and s["bytes"] == 42

    # hand edit, new scene, then a plain run: 0000 is not rescanned, 0001 is added
    doc["scenes"]["0000"]["note"] = "keep me"
    (tmp_path / "disk_usage.json").write_text(json.dumps(doc))
    make_scene(tmp_path, "0001", 2)
    (tmp_path / "0000" / "images" / "00009.png").write_bytes(b"x" * 10)
    doc = disk_usage.update(tmp_path)
    assert doc["scenes"]["0000"]["frames"] == 3
    assert doc["totals"]["frames"] == 5 and doc["modalities"][0] == "images"

    # rescan picks up the change and keeps the note; a removed scene drops out
    (tmp_path / "0001" / "images" / "00000.png").unlink()
    (tmp_path / "0001" / "images" / "00001.png").unlink()
    (tmp_path / "0001" / "images").rmdir()
    (tmp_path / "0001" / "data_ts.txt").unlink()
    (tmp_path / "0001").rmdir()
    doc = disk_usage.update(tmp_path, {"0000"})
    assert doc["scenes"]["0000"]["frames"] == 4 and doc["scenes"]["0000"]["note"] == "keep me"
    assert list(doc["scenes"]) == ["0000"]


def test_a_file_from_an_older_schema_is_rescanned_whole(tmp_path):
    make_scene(tmp_path, "0000", 2)
    stale = {"schema": 1, "scenes": {"0000": {"bytes": 1, "modalities": {}, "note": "keep me"}}}
    (tmp_path / "disk_usage.json").write_text(json.dumps(stale))
    s = disk_usage.update(tmp_path)["scenes"]["0000"]
    assert s["bytes"] == 32 and s["files"] == {"data_ts.txt": 12} and s["note"] == "keep me"
