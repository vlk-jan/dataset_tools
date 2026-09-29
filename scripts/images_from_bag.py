#!/usr/bin/env python3
"""Give a scene of PNGs the camera's own JPEGs back, from the bag it came from.

    images_from_bag.py <bag_dir> <scene_dir> [--cam TOPIC] [--jobs N] [--swap] [--drop-old]

A scene extracted before 2026-09-29 holds a PNG of every camera frame: the
pixels the camera's JPEG decodes to, at four times the size. The JPEG itself is
still in the bag. This finds, for every frame of the scene, the bag message
that decodes to exactly its pixels, and writes that message's bytes unmodified
to ``images.new/NNNNN.jpg``. Nothing else of the scene is read or written, and
no frame changes its index, so clouds, maps, poses and labels stay what they
were.

Matching is by pixel identity and nothing weaker: a frame whose pixels no
message of the bag decodes to is an error, the scene is left as it was, and
nothing is swapped. So a scene that comes out of this is, decoded, bit for bit
the scene that went in.

``--swap`` then puts ``images.new`` in the place of ``images``. An ``images``
that is a symbolic link (a root built from links into another) is unlinked,
which leaves the other root's files alone; one that is a directory is renamed
to ``images.png_old``. ``--drop-old`` removes that directory afterwards. Every
path written or removed is checked to lie inside the scene directory itself:
writing through a link would put the files into another root.

After the last scene of a root: ``helhest-image-format <root>`` declares the
new format in ``manifest.json``.
"""

import argparse
import hashlib
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rosbags.highlevel import AnyReader  # noqa: E402

from dataset_tools import image_format, rosbags_compat  # noqa: E402, F401  (rosbags_compat patches rosbags)

LOG = logging.getLogger("images_from_bag")
NEW, OLD = "images.new", "images.png_old"


def pixels_hash(img: np.ndarray) -> bytes:
    return hashlib.md5(np.ascontiguousarray(img).tobytes() + repr(img.shape).encode()).digest()


def hash_file(path: str) -> bytes | None:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return None if img is None else pixels_hash(img)


def inside(path: Path, scene: Path) -> Path:
    """``path``, which must be an entry of ``scene`` itself and not reached through a link."""
    if path.parent.resolve() != scene.resolve() or path.is_symlink():
        raise ValueError(f"{path} is not a directory of {scene} itself; refusing to write through a link")
    return path


def recover(bag: Path, scene: Path, topic: str, jobs: int) -> int:
    if scene.is_symlink():
        raise ValueError(f"{scene} is a symbolic link; convert the scene it points to, or a directory of links")
    have = image_format.held(scene)
    if set(have) != {".png"}:
        raise ValueError(f"{scene}/images holds {have or 'no images'}; only a scene of PNGs is converted")
    frames = sorted(p for p in (scene / "images").iterdir() if p.suffix == ".png")
    if [p.stem for p in frames] != [f"{i:05d}" for i in range(len(frames))]:
        raise ValueError(f"{scene}/images is not a contiguous 00000..{len(frames) - 1:05d} run")

    t0 = time.time()
    with ProcessPoolExecutor(jobs) as pool:
        hashes = list(pool.map(hash_file, [str(p) for p in frames], chunksize=32))
    unreadable = [frames[i].name for i, h in enumerate(hashes) if h is None]
    if unreadable:
        raise ValueError(f"{len(unreadable)} PNGs of {scene} cannot be read, first {unreadable[0]}")
    wanted: dict[bytes, list[int]] = {}
    for i, h in enumerate(hashes):
        wanted.setdefault(h, []).append(i)
    repeated = sum(len(v) - 1 for v in wanted.values())
    LOG.info(
        "%s: %d frames hashed in %.0f s (%d with pixels another frame also has)",
        scene.name,
        len(frames),
        time.time() - t0,
        repeated,
    )

    new = scene / NEW
    if new.exists() or new.is_symlink():
        inside(new, scene)
        shutil.rmtree(new)
    new.mkdir()
    inside(new, scene)

    todo, written, seen, sizes, formats = len(frames), 0, 0, 0, {}
    t0 = time.time()
    with AnyReader([bag]) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            raise ValueError(f"{bag} has no topic {topic}")
        for conn, _, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            payload = np.frombuffer(msg.data, np.uint8)
            seen += 1
            ext = image_format.sniff(payload)
            formats[ext] = formats.get(ext, 0) + 1
            if ext != ".jpg":
                continue
            img = cv2.imdecode(payload, cv2.IMREAD_COLOR)
            if img is None:
                continue
            slots = wanted.get(pixels_hash(img))
            if not slots:
                continue
            index = slots.pop(0)  # frames with the same pixels take the messages in order
            (new / f"{index:05d}.jpg").write_bytes(payload.tobytes())
            written += 1
            sizes += len(payload)
            if written == todo:
                break
    LOG.info("%s: %d bag images read in %.0f s, formats %s", scene.name, seen, time.time() - t0, formats)

    missing = sorted(i for slots in wanted.values() for i in slots)
    if missing:
        shutil.rmtree(new)
        raise ValueError(
            f"{scene.name}: {len(missing)} of {todo} frames have no bag image that decodes to their pixels "
            f"(first {missing[0]:05d}); nothing was changed"
        )
    png = sum(p.stat().st_size for p in frames)
    LOG.info(
        "%s: %d frames, %.2f GB of PNG -> %.2f GB of camera JPEG (%.1fx)",
        scene.name,
        todo,
        png / 1e9,
        sizes / 1e9,
        png / sizes,
    )
    return todo


def swap(scene: Path) -> None:
    images, new, old = scene / "images", inside(scene / NEW, scene), scene / OLD
    n_new = sum(1 for p in new.iterdir() if p.suffix == ".jpg")
    n_old = sum(image_format.held(scene).values())
    if n_new != n_old or not n_new:
        raise ValueError(f"{scene.name}: {NEW} holds {n_new} images and images {n_old}; not swapped")
    if images.parent.resolve() != scene.resolve():
        raise ValueError(f"{images} is not an entry of {scene}")
    if images.is_symlink():
        target = os.readlink(images)
        images.unlink()  # the link, not what it points to
        LOG.info("%s: images was a link to %s; unlinked", scene.name, target)
    else:
        if old.exists():
            raise ValueError(f"{old} exists already")
        images.rename(old)
    new.rename(images)


def drop_old(scene: Path) -> None:
    old = scene / OLD
    if not old.exists():
        return
    inside(old, scene)
    if set(image_format.held(scene)) != {".jpg"}:
        raise ValueError(f"{scene.name}: images is not a directory of JPEGs; {OLD} kept")
    n_new, n_old = image_format.held(scene)[".jpg"], sum(1 for p in old.iterdir() if p.suffix == ".png")
    if n_new != n_old:
        raise ValueError(f"{scene.name}: {n_new} JPEGs for {n_old} PNGs; {OLD} kept")
    shutil.rmtree(old)
    LOG.info("%s: %s removed (%d PNGs)", scene.name, OLD, n_old)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", type=Path)
    ap.add_argument("scene", type=Path)
    ap.add_argument("--cam", default="/odin1/image/compressed", help="compressed image topic")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4, help="processes hashing the PNGs")
    ap.add_argument("--swap", action="store_true", help=f"put {NEW} in the place of images")
    ap.add_argument("--drop-old", action="store_true", help=f"remove {OLD} once the JPEGs are in place")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    try:
        if set(image_format.held(a.scene)) == {".jpg"} and not (a.scene / NEW).exists():
            LOG.info("%s: holds JPEGs already", a.scene.name)
        else:
            recover(a.bag, a.scene, a.cam, a.jobs)
            if a.swap:
                swap(a.scene)
        if a.drop_old:
            drop_old(a.scene)
    except (ValueError, OSError) as exc:
        LOG.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
