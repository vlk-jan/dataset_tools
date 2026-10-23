"""make_atlas never writes through a link into another release's atlas.

A root may have a real ``atlas/`` whose ``thumbs`` is a link into another
release's, so a guard on ``atlas/`` alone would let a thumbnail rewrite land there.
"""

import pytest
from make_atlas import resolve_atlas_dir


def _linked(tmp_path):
    older = tmp_path / "v14" / "atlas" / "thumbs"
    older.mkdir(parents=True)
    (older / "0000a.jpg").write_bytes(b"jpeg")
    atlas = tmp_path / "v15" / "atlas"
    atlas.mkdir(parents=True)
    (atlas / "thumbs").symlink_to(older)
    return atlas, older


def test_a_linked_thumbs_directory_is_refused(tmp_path):
    atlas, _ = _linked(tmp_path)
    with pytest.raises(SystemExit, match="symlink"):
        resolve_atlas_dir(atlas, materialize=False)
    assert (atlas / "thumbs").is_symlink()


def test_materialize_copies_the_thumbnails_and_leaves_the_older_root_alone(tmp_path):
    atlas, older = _linked(tmp_path)
    resolve_atlas_dir(atlas, materialize=True)

    thumbs = atlas / "thumbs"
    assert thumbs.is_dir() and not thumbs.is_symlink()
    assert (thumbs / "0000a.jpg").read_bytes() == b"jpeg"
    (thumbs / "0000a.jpg").write_bytes(b"new")
    assert (older / "0000a.jpg").read_bytes() == b"jpeg"


def test_a_real_thumbs_directory_is_left_as_it_is(tmp_path):
    atlas = tmp_path / "atlas"
    (atlas / "thumbs").mkdir(parents=True)
    resolve_atlas_dir(atlas, materialize=False)
    assert (atlas / "thumbs").is_dir()
