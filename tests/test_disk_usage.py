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


def test_what_a_link_leads_to_is_counted_apart(tmp_path):
    """A root built from links into another one (a release linking scenes of
    the one before): the linked frames are this root's frames, the linked bytes
    are not its disk usage."""
    make_scene(tmp_path / "other_root", "0000", 3)
    (tmp_path / "other_root" / "0000" / "clouds").mkdir()
    (tmp_path / "other_root" / "0000" / "clouds" / "00000.npy").write_bytes(b"x" * 7)
    root = tmp_path / "root"
    scene = root / "0000"
    scene.mkdir(parents=True)
    (scene / "images").symlink_to("../../other_root/0000/images")  # a linked directory
    (scene / "data_ts.txt").symlink_to("../../other_root/0000/data_ts.txt")  # a linked file
    (scene / "clouds").mkdir()  # a directory of its own, with a link inside
    (scene / "clouds" / "00000.npy").symlink_to("../../../other_root/0000/clouds/00000.npy")
    (scene / "clouds" / "00001.npy").write_bytes(b"x" * 5)
    (scene / "gps.npy").symlink_to("../../other_root/0000/nothing_here")  # broken
    (scene / "source_bag.txt").write_text("bag\n")

    doc = disk_usage.update(root)
    s = doc["scenes"]["0000"]
    assert s["modalities"]["images"] == {
        "bytes": 0,
        "files": 0,
        "linked_bytes": 30,
        "linked_files": 3,
        "link": "../../other_root/0000/images",
    }
    assert s["modalities"]["clouds"] == {"bytes": 5, "files": 1, "linked_bytes": 7, "linked_files": 1}
    assert s["modalities"]["other"] == {"bytes": 4, "files": 1, "linked_bytes": 12, "linked_files": 1}
    assert s["files"] == {"source_bag.txt": 4}
    assert s["linked_files"] == {"data_ts.txt": {"bytes": 12, "link": "../../other_root/0000/data_ts.txt"}}
    assert s["frames"] == 3 and s["span_s"] == 2.5
    assert s["bytes"] == 9 and s["linked_bytes"] == 49 and s["broken_links"] == 1
    t = doc["totals"]
    assert (t["bytes"], t["files"], t["linked_bytes"], t["linked_files"]) == (9, 2, 49, 5)
    assert t["frames"] == 3 and t["broken_links"] == 1
    assert doc["modalities"][0] == "images"  # largest by what it holds, own or linked


def test_a_scene_without_links_says_so_in_every_measured_key(tmp_path):
    """The measured keys are always written, so a rescan of a scene that no
    longer has links cannot leave the old figures behind."""
    make_scene(tmp_path, "0000", 2)
    s = disk_usage.update(tmp_path)["scenes"]["0000"]
    assert s["linked_bytes"] == 0 and s["linked_files"] == {} and s["broken_links"] == 0
    assert "linked_bytes" not in s["modalities"]["images"]
