#!/usr/bin/env python3
"""Which file format a root's camera images are stored in, and who says so.

    helhest-image-format <root>            # declare what the scenes hold
    helhest-image-format <root> --check    # say it, write nothing

The cameras of both rigs publish ``sensor_msgs/CompressedImage``: a JPEG the
camera has already made. Until 2026-09-29 the extractor decoded it and wrote
the pixels out again as a PNG, which is a lossless copy of a lossy image --
pixel for pixel what the JPEG decodes to, at four times its size (2.9 MB
against 0.7 MB for the Odin1, measured; images were 91 % of a root). A root
now holds the camera's own bytes, ``msg.data`` unmodified, under the extension
their first bytes name.

So a root's images are one of two things, and nothing may have to guess which:

``<root>/manifest.json`` declares it, in the key this module owns::

    "images": {"extension": "jpg",
               "encoding": "camera JPEG bytes verbatim, msg.data unmodified"}

A root without the key is a PNG root, which is every root written before the
change. A root with the key holds that extension in every scene, and a scene
that does not is an error rather than a scene with fewer frames: a consumer
that lists frames by testing for the file would otherwise hand back a short
list, and the first anyone would hear of it is a training run on less data.

Inside this package the extension of one scene is `of_scene`: what the root
declares, checked against what the directory holds, and what the directory
holds where nothing is declared -- a scene fresh out of the extractor has no
manifest above it yet.
"""

import argparse
import json
import os
import re
from pathlib import Path

from dataset_tools.gt_io import _replace_atomically

NAME = "manifest.json"
KEY = "images"

#: What a root that declares nothing holds.
DEFAULT = ".png"

#: Extension -> what the manifest says the files are.
ENCODINGS = {
    ".jpg": "camera JPEG bytes verbatim, msg.data unmodified",
    ".png": "PNG",
}
EXTENSIONS = tuple(ENCODINGS)

#: How a payload says what it is: (first bytes, extension).
MAGIC = ((b"\xff\xd8\xff", ".jpg"), (b"\x89PNG\r\n\x1a\n", ".png"))


class ImageFormatError(ValueError):
    """The images of a scene or a root are not what was declared, or not one thing."""


def sniff(payload) -> str | None:
    """The extension a compressed image payload names by its first bytes, or None."""
    head = bytes(payload[:8])
    for magic, ext in MAGIC:
        if head.startswith(magic):
            return ext
    return None


def _normalise(value, where) -> str:
    ext = "." + str(value).lower().lstrip(".")
    if ext not in ENCODINGS:
        known = ", ".join(e.lstrip(".") for e in EXTENSIONS)
        raise ImageFormatError(f"{where}: images.extension is {value!r}; expected one of {known}")
    return ext


def declared(root) -> str | None:
    """The extension ``<root>/manifest.json`` declares, or None where it declares none."""
    path = Path(root) / NAME
    try:
        doc = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise ImageFormatError(f"{path}: cannot be read ({exc})") from exc
    block = doc.get(KEY) if isinstance(doc, dict) else None
    if not isinstance(block, dict) or "extension" not in block:
        return None
    return _normalise(block["extension"], path)


def held(scene_dir) -> dict[str, int]:
    """Image files in ``<scene>/images`` per known extension; absent ones are not listed."""
    counts: dict[str, int] = {}
    try:
        names = os.listdir(Path(scene_dir) / "images")
    except OSError:
        return counts
    for name in names:
        ext = os.path.splitext(name)[1]
        if ext in ENCODINGS:
            counts[ext] = counts.get(ext, 0) + 1
    return counts


def of_scene(scene_dir, root=None) -> str:
    """The extension of this scene's images.

    ``root`` is where the manifest is looked for; by default the directory the
    scene is in. Raises `ImageFormatError` for a scene that holds two formats,
    or one other than its root declares.
    """
    scene_dir = Path(scene_dir)
    have = held(scene_dir)
    want = declared(scene_dir.parent if root is None else root)
    if len(have) > 1:
        found = ", ".join(f"{n} {e}" for e, n in sorted(have.items()))
        raise ImageFormatError(f"{scene_dir}/images holds more than one format: {found}")
    if want is None:
        return next(iter(have), DEFAULT)
    if have and want not in have:
        ext, n = next(iter(have.items()))
        raise ImageFormatError(f"{scene_dir}/images holds {n} {ext} files; the root declares {want}")
    return want


#: What a scene directory is called: its index, zero-padded to four digits.
SCENE_DIR_RE = re.compile(r"\d{4,}")


def scenes(root) -> list[Path]:
    """The scene directories of a root, in index order."""
    return sorted(d for d in Path(root).iterdir() if SCENE_DIR_RE.fullmatch(d.name) and d.is_dir())


def of_root(root) -> str | None:
    """The one extension every scene with images holds, or None for a root with no images.

    Looks at the scenes only, not at the manifest: this is what `declare` writes.
    """
    found: dict[str, list[str]] = {}
    for scene in scenes(root):
        have = held(scene)
        if len(have) > 1:
            raise ImageFormatError(f"{scene}/images holds more than one format: {sorted(have)}")
        for ext in have:
            found.setdefault(ext, []).append(scene.name)
    if len(found) > 1:
        parts = "; ".join(f"{e}: {' '.join(s)}" for e, s in sorted(found.items()))
        raise ImageFormatError(f"{root}: scenes do not agree on the image format ({parts})")
    return next(iter(found), None)


def block(ext) -> dict[str, str]:
    """The manifest's ``images`` block for an extension."""
    ext = _normalise(ext, "images")
    return {"extension": ext.lstrip("."), "encoding": ENCODINGS[ext]}


def declare(root, ext=None) -> str | None:
    """Write what the scenes hold into ``<root>/manifest.json`` and return it.

    Every other key of the manifest is kept as it is, and so is anything a
    person added to the ``images`` block. A PNG root that declares nothing is
    left declaring nothing, which is what it already means. Refuses a root
    that declares one format and holds another.
    """
    root = Path(root)
    have = of_root(root)
    ext = have if ext is None else _normalise(ext, "images")
    if have is not None and ext != have:
        raise ImageFormatError(f"{root}: the scenes hold {have}, not {ext}")
    was = declared(root)
    if ext is None or (was is None and ext == DEFAULT):
        return was or ext
    if was is not None and was != ext:
        raise ImageFormatError(f"{root}: {NAME} declares {was} and the scenes hold {ext}")
    path = root / NAME
    try:
        doc = json.loads(path.read_text())
    except FileNotFoundError:
        doc = {}
    if not isinstance(doc, dict):
        raise ImageFormatError(f"{path}: not a JSON object")
    old = doc.get(KEY) if isinstance(doc.get(KEY), dict) else {}
    doc[KEY] = {**old, **block(ext)}
    _replace_atomically(path, lambda f: f.write(json.dumps(doc, indent=1, ensure_ascii=False) + "\n"))
    return ext


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--check", action="store_true", help="report only; write nothing")
    a = ap.parse_args()
    try:
        was, have = declared(a.root), of_root(a.root)
        for scene in scenes(a.root):
            of_scene(scene, a.root)
        ext = (was or have) if a.check else declare(a.root)
    except ImageFormatError as exc:
        ap.exit(1, f"{exc}\n")
    said = f"declares {was}" if was else f"declares nothing, which means {DEFAULT}"
    print(f"{a.root}: scenes hold {have or 'no images'}; {NAME} {said}" + ("" if a.check else f"; now {ext}"))


if __name__ == "__main__":
    main()
