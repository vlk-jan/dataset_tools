"""scripts/images_from_bag.py: a scene of PNGs gets the camera's JPEGs back."""

import importlib.util
import sys
from pathlib import Path

import pytest
from test_extract_bag import COMMON, FRAMES, _image_payload, write_bag

from dataset_tools import extract_data as ed
from dataset_tools import image_format

spec = importlib.util.spec_from_file_location(
    "images_from_bag", Path(__file__).resolve().parents[1] / "scripts" / "images_from_bag.py"
)
ifb = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ifb  # so that its functions pickle for the worker processes
spec.loader.exec_module(ifb)

CAM = "/cam/image/compressed"


@pytest.fixture
def png_root(tmp_path: Path):
    """A root as they were made before: the camera published JPEG, the scene holds PNG."""
    bag = write_bag(tmp_path / "bags" / "bag_a", image_ext=".jpg")
    out = tmp_path / "out"
    assert ed.main([str(tmp_path / "bags"), str(out), *COMMON, "--image_format", "png"]) == 0
    return bag, out


def test_every_frame_gets_the_bytes_the_camera_published(png_root):
    bag, out = png_root
    scene = out / "0000"
    before = {p.name: p.read_bytes() for p in scene.iterdir() if p.is_file()}
    assert ifb.recover(bag, scene, CAM, jobs=2) == FRAMES
    ifb.swap(scene)
    assert image_format.held(scene) == {".jpg": FRAMES}
    for i in range(FRAMES):
        assert (scene / "images" / f"{i:05d}.jpg").read_bytes() == _image_payload(i, ".jpg").tobytes()
    # the PNGs are kept until they are asked to go, and nothing else changed
    assert len(list((scene / "images.png_old").glob("*.png"))) == FRAMES
    assert {p.name: p.read_bytes() for p in scene.iterdir() if p.is_file()} == before
    ifb.drop_old(scene)
    assert not (scene / "images.png_old").exists() and ed.scene_is_complete(scene)
    assert image_format.declare(out) == ".jpg"


def test_a_scene_of_links_is_converted_without_touching_what_it_links_to(png_root, tmp_path: Path):
    """v2.1 over v2.0: images/ is a link into the other root. The JPEGs go
    into the scene, the link goes, and the other root keeps its PNGs."""
    bag, out = png_root
    scene = tmp_path / "linked" / "0000"
    scene.mkdir(parents=True)
    for entry in (out / "0000").iterdir():
        (scene / entry.name).symlink_to(Path("../../out/0000") / entry.name)
    assert ifb.recover(bag, scene, CAM, jobs=2) == FRAMES
    ifb.swap(scene)
    assert not (scene / "images").is_symlink() and image_format.held(scene) == {".jpg": FRAMES}
    assert (scene / "clouds").is_symlink()
    assert image_format.held(out / "0000") == {".png": FRAMES}
    assert not (out / "0000" / "images.new").exists() and not (scene / "images.png_old").exists()


def test_a_frame_the_bag_does_not_hold_changes_nothing(png_root):
    bag, out = png_root
    scene = out / "0000"
    import cv2
    import numpy as np

    cv2.imwrite(str(scene / "images" / "00004.png"), np.full((8, 12, 3), 200, np.uint8))
    with pytest.raises(ValueError, match="1 of .* frames have no bag image"):
        ifb.recover(bag, scene, CAM, jobs=2)
    assert not (scene / "images.new").exists() and image_format.held(scene) == {".png": FRAMES}


def test_nothing_is_written_through_a_link(png_root, tmp_path: Path):
    bag, out = png_root
    link = tmp_path / "scene_link"
    link.symlink_to(out / "0000")
    with pytest.raises(ValueError, match="symbolic link"):
        ifb.recover(bag, link, CAM, jobs=2)
    (out / "0000" / "images.new").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="refusing to write through a link"):
        ifb.recover(bag, out / "0000", CAM, jobs=2)
