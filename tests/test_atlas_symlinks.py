"""make_atlas never writes through a link into another release's atlas.

HELHEST v1.5 has a real ``atlas/`` whose ``thumbs`` is a link into v1.4's, so
the guard on ``atlas/`` alone let a thumbnail rewrite land in v1.4.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from make_atlas import resolve_thumbs_dir  # noqa: E402


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
        resolve_thumbs_dir(atlas, materialize=False)
    assert (atlas / "thumbs").is_symlink()


def test_materialize_copies_the_thumbnails_and_leaves_the_older_root_alone(tmp_path):
    atlas, older = _linked(tmp_path)
    resolve_thumbs_dir(atlas, materialize=True)

    thumbs = atlas / "thumbs"
    assert thumbs.is_dir() and not thumbs.is_symlink()
    assert (thumbs / "0000a.jpg").read_bytes() == b"jpeg"
    (thumbs / "0000a.jpg").write_bytes(b"new")
    assert (older / "0000a.jpg").read_bytes() == b"jpeg"


def test_a_real_thumbs_directory_is_left_as_it_is(tmp_path):
    atlas = tmp_path / "atlas"
    (atlas / "thumbs").mkdir(parents=True)
    resolve_thumbs_dir(atlas, materialize=False)
    assert (atlas / "thumbs").is_dir()
