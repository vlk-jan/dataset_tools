"""What a root's images are stored as, and who says so (dataset_tools.image_format)."""

import json

import pytest

from dataset_tools import image_format as imf


def make_scene(root, sid, ext, n=3):
    d = root / sid / "images"
    d.mkdir(parents=True)
    for i in range(n):
        (d / f"{i:05d}{ext}").write_bytes(b"x")
    return root / sid


def test_a_payload_says_what_it_is():
    assert imf.sniff(b"\xff\xd8\xff\xe0\x00\x10JFIF") == ".jpg"
    assert imf.sniff(b"\x89PNG\r\n\x1a\n....") == ".png"
    assert imf.sniff(b"RIFF....WEBP") is None
    assert imf.sniff(b"") is None


def test_a_root_that_declares_nothing_is_png(tmp_path):
    scene = make_scene(tmp_path, "0000", ".png")
    assert imf.declared(tmp_path) is None
    assert imf.of_scene(scene) == ".png"
    # ... and a scene with no images yet is whatever the default is.
    (tmp_path / "0001").mkdir()
    assert imf.of_scene(tmp_path / "0001") == imf.DEFAULT


def test_a_scene_fresh_from_the_extractor_is_what_it_holds(tmp_path):
    """No manifest above it yet, so the directory speaks for itself."""
    assert imf.of_scene(make_scene(tmp_path, "0000", ".jpg")) == ".jpg"


def test_declare_writes_the_block_and_keeps_the_rest(tmp_path):
    make_scene(tmp_path, "0000", ".jpg")
    make_scene(tmp_path, "0001", ".jpg")
    (tmp_path / "manifest.json").write_text(json.dumps({"rig": "x", "images": {"note": "kept"}}))
    assert imf.declare(tmp_path) == ".jpg"
    doc = json.loads((tmp_path / "manifest.json").read_text())
    assert doc["rig"] == "x"
    assert doc["images"] == {"note": "kept", "extension": "jpg", "encoding": imf.ENCODINGS[".jpg"]}
    assert imf.declared(tmp_path) == ".jpg"
    assert not (tmp_path / "manifest.json.tmp").exists()


def test_a_png_root_is_left_declaring_nothing(tmp_path):
    make_scene(tmp_path, "0000", ".png")
    assert imf.declare(tmp_path) == ".png"
    assert not (tmp_path / "manifest.json").exists()


def test_a_scene_that_holds_another_format_than_declared_is_an_error(tmp_path):
    """Not a scene with no frames: a consumer that tests for the declared file
    would list nothing and say nothing."""
    scene = make_scene(tmp_path, "0000", ".png")
    (tmp_path / "manifest.json").write_text(json.dumps({"images": {"extension": "jpg"}}))
    with pytest.raises(imf.ImageFormatError, match="declares .jpg"):
        imf.of_scene(scene)
    with pytest.raises(imf.ImageFormatError, match="declares .jpg"):
        imf.declare(tmp_path)


def test_two_formats_in_one_scene_or_one_root_are_an_error(tmp_path):
    scene = make_scene(tmp_path, "0000", ".png")
    (scene / "images" / "00009.jpg").write_bytes(b"x")
    with pytest.raises(imf.ImageFormatError, match="more than one format"):
        imf.of_scene(scene)
    (scene / "images" / "00009.jpg").unlink()
    make_scene(tmp_path, "0001", ".jpg")
    with pytest.raises(imf.ImageFormatError, match="do not agree"):
        imf.of_root(tmp_path)


def test_an_extension_nobody_knows_is_refused(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"images": {"extension": "webp"}}))
    with pytest.raises(imf.ImageFormatError, match="expected one of"):
        imf.declared(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"images": {"extension": ".JPG"}}))
    assert imf.declared(tmp_path) == ".jpg"
