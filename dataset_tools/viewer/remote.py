"""A dataset root served over HTTP (an Apache-style directory index), mirrored on demand.

The rest of the viewer only reads local files. :func:`mount` maps a URL root onto
a local mirror directory, which is then used as the root:

- a directory's listing is synced by :func:`refresh`, at most every
  :data:`REFRESH_SECONDS`; the root syncs everything but its scenes, a scene
  syncs when it is first looked at;
- files directly in the root, a scene or another top-level directory (manifest,
  odometry, calibration, ``atlas/atlas.json``, ...) are downloaded by the sync;
- files deeper down (``clouds/00000.npy``, ``images/``, atlas thumbnails) and
  every ``.npy`` (a long scene's ``joint_states.npy`` alone is 15 MB) are only
  listed until :func:`local` is asked for one; :func:`head` reads the start of
  one without fetching it. Not writing them as empty placeholders spares ~1.2M
  file creations: a cold index of the 36-scene public root went 78 s -> 16 s.

A root can publish its scenes' per-frame directory listings as
``listings/<scene>.json.gz`` (:func:`write_listings`, ``python -m
dataset_tools.viewer.remote ROOT``): one request per scene instead of an index
page per directory, which the server is slow to build at 16k entries (a cold
index of the 36-scene public root: 15 s). A directory newer than its scene's
file is listed by the server as before.

Each mirrored directory keeps the listing it was synced from in ``.listing.json``;
:func:`listdir` and :func:`is_file` answer from it, so a file not fetched yet
looks present to the rest of the viewer.
A scene's fingerprint is that file's digest, so fetching a frame never moves it.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
import logging
import os
import re
import shutil
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, unquote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

logger = logging.getLogger("dataset_tools.viewer.remote")

LISTING = ".listing.json"
#: How long a synced listing is trusted before it is fetched again (s).
REFRESH_SECONDS = 300.0
TIMEOUT = 30
#: Largest response accepted, compressed or not (a 28k-frame scene's
#: ``joint_states.npy`` is 15 MB), so a hostile server cannot exhaust memory.
MAX_BYTES = 512 * 2**20

#: (URL ending in "/", resolved mirror directory) of the mounted root.
_mount: tuple[str, Path] | None = None
_synced: dict[Path, float] = {}
_locks: dict[Path, threading.Lock] = {}
_locks_guard = threading.Lock()
#: Parsed ``.listing.json`` of the most recently used mirrored directories, with the
#: file's (mtime, inode). Bounded: a cold index reads every one (36 scenes: ~90 MB).
_listings: dict[Path, tuple[tuple[int, int], dict[str, list]]] = {}
_listings_guard = threading.Lock()
LISTINGS_KEPT = 64

#: Root directory of the published per-scene listings, and their format.
PUBLISHED_DIR = "listings"
PUBLISHED_FORMAT = 1
#: The minute an Apache index prints as an entry's date.
_MINUTE = re.compile(r"\d{4}-\d\d-\d\d \d\d:\d\d")
#: A name that stays in its directory and is not hidden, as :func:`parse_listing` keeps.
_NAME = re.compile(r"[^./\\\0][^/\\\0]*")
_pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="remote-fetch")


class _SameHost(HTTPRedirectHandler):
    """Follow a redirect only within the root's scheme and host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlsplit(newurl)[:2] != urlsplit(req.full_url)[:2]:
            raise HTTPError(newurl, code, f"redirect off {urlsplit(req.full_url).netloc} refused", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_opener = build_opener(_SameHost)

#: One link of a directory index and the text after it (Apache: date, size).
_LINK = re.compile(r'<a href="([^"]+)"[^>]*>.*?</a>(.*?)(?=<a |</tr>|</li>|\Z)', re.S | re.I)


def is_url(root: str) -> bool:
    return root.startswith(("http://", "https://"))


def mount(url: str, cache_dir: Path, clear: bool = False) -> Path:
    """Mirror ``url`` under ``cache_dir/mirror`` and return that directory.

    Syncs the root's listing first, so an unreachable URL raises ``OSError`` here.
    """
    # ponytail: the mirror is unbounded; evict like bundles if public users fill disks.
    global _mount
    mirror = Path(cache_dir) / "mirror"
    # Only a mirror's own listing proves the directory is ours to clear and overwrite.
    if mirror.is_dir() and not (mirror / LISTING).is_file() and any(mirror.iterdir()):
        raise OSError(f"{mirror} holds files this viewer did not mirror; pass another --cache-dir")
    if clear:
        shutil.rmtree(mirror, ignore_errors=True)
    mirror.mkdir(parents=True, exist_ok=True)
    mirror = mirror.resolve()
    _mount = (url.rstrip("/") + "/", mirror)
    _synced.clear()
    _sync(mirror, ())
    _synced[mirror] = time.monotonic()
    return mirror


def parse_listing(text: str) -> dict[str, list]:
    """``{name: [is_dir, stamp]}`` of a directory index page.

    ``stamp`` is the text the index prints after the link (Apache: date and
    size), so it moves when the entry does. Parent links, sort links, hidden
    names and anything that could leave the directory are dropped.
    """
    entries: dict[str, list] = {}
    for m in _LINK.finditer(text):
        href = html.unescape(m[1])
        if href.startswith("?"):  # sort links; the name check drops the rest
            continue
        is_dir = href.endswith("/")
        name = unquote(href.rstrip("/"))
        if not name or name.startswith(".") or any(c in name for c in "/\\\0"):
            continue
        entries[name] = [is_dir, " ".join(re.sub(r"<[^>]+>|&nbsp;", " ", m[2]).split())]
    return entries


def refresh(path: Path) -> None:
    """Sync a mirrored directory's listing if it is older than :data:`REFRESH_SECONDS`.

    A no-op outside the mirror. A failed sync keeps what the mirror has.
    """
    parts = _parts(path)
    if parts is None:
        return
    path = _mount[1].joinpath(*parts)
    with _locks_guard:
        lock = _locks.setdefault(path, threading.Lock())
    with lock:
        if time.monotonic() - _synced.get(path, -REFRESH_SECONDS) < REFRESH_SECONDS:
            return
        try:
            _sync(path, parts)
        except OSError as exc:
            logger.warning("cannot refresh %s: %s", _url(parts, True), exc)
        _synced[path] = time.monotonic()


def local(path: Path) -> Path:
    """``path``, downloaded first when it is a mirrored file not fetched yet.

    Raises ``OSError`` when the download fails; a missing file is left to the reader.
    """
    parts = _parts(path)
    if _unfetched(path, parts):
        _download(_url(parts), path)
    return path


def head(path: Path) -> bytes | None:
    """The first 4 KiB of a mirrored file not fetched yet, by a range request.

    ``None`` for any other path: read that one directly.
    """
    parts = _parts(path)
    if not _unfetched(path, parts):
        return None
    return _get(_url(parts), {"Range": "bytes=0-4095"})[:4096]


def listdir(path: Path) -> list[str]:
    """``os.listdir(path)``; for a synced mirrored directory, the names in its listing."""
    entries = _listing(Path(path))
    return os.listdir(path) if entries is None else list(entries)


def is_file(path: Path) -> bool:
    """``path.is_file()``, also true for a listed mirrored file not fetched yet."""
    path = Path(path)
    parts = _parts(path)
    if parts and _lazy(parts):
        entries = _listing(path.parent)
        if entries is not None:
            entry = entries.get(path.name)
            return entry is not None and not entry[0]
    return path.is_file()


def fingerprint(scene_dir: Path) -> str | None:
    """Digest of a mirrored directory's synced listing; ``None`` outside the mirror.

    Raises ``OSError`` when the directory was never synced (or is gone).
    """
    if _parts(scene_dir) is None:
        return None
    return hashlib.sha1((scene_dir / LISTING).read_bytes()).hexdigest()[:16]


# --------------------------------------------------------------------------


def _parts(path: Path) -> tuple[str, ...] | None:
    if _mount is None:
        return None
    try:
        return Path(path).relative_to(_mount[1]).parts
    except ValueError:
        return None


def _lazy(parts: tuple[str, ...]) -> bool:
    # Deeper than a scene's own files, an array, or a scene's published listing (fetched with the scene).
    return len(parts) > 2 or parts[-1].endswith(".npy") or parts[0] == PUBLISHED_DIR


def _unfetched(path: Path, parts: tuple[str, ...] | None) -> bool:
    if not parts or not _lazy(parts):
        return False
    try:
        return path.stat().st_size == 0  # an empty placeholder, as mirrors before 2026-10-29 wrote
    except FileNotFoundError:
        return is_file(path)


def _listing(path: Path) -> dict[str, list] | None:
    """The synced listing of a mirrored directory; ``None`` outside the mirror or before a sync."""
    if _parts(path) is None:
        return None
    try:
        st = (path / LISTING).stat()
    except OSError:
        return None
    stamp = (st.st_mtime_ns, st.st_ino)
    with _listings_guard:
        cached = _listings.pop(path, None)
        if cached is None or cached[0] != stamp:
            try:
                cached = (stamp, json.loads((path / LISTING).read_text()))
            except (OSError, ValueError):
                return None
        _listings[path] = cached  # most recent last
        while len(_listings) > LISTINGS_KEPT:
            del _listings[next(iter(_listings))]
    return cached[1]


def _url(parts: tuple[str, ...], is_dir: bool = False) -> str:
    rel = "/".join(quote(p) for p in parts)
    return _mount[0] + rel + ("/" if is_dir and rel else "")


def _get(url: str, headers: dict | None = None) -> bytes:
    headers = headers or {"Accept-Encoding": "gzip"}
    req = Request(url, headers={**headers, "User-Agent": "dataset_tools-viewer"})
    with _opener.open(req, timeout=TIMEOUT) as resp:
        data = resp.read(MAX_BYTES + 1)
        if len(data) <= MAX_BYTES and resp.headers.get("Content-Encoding") == "gzip":
            data = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(data, MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise OSError(f"{url}: response over {MAX_BYTES >> 20} MB refused")
    return data


def _download(url: str, dest: Path) -> None:
    """Fetch ``url`` into ``dest`` through a temp file, so no reader sees half of it."""
    data = _get(url)
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, dest)


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def _sync(path: Path, parts: tuple[str, ...], entries: dict[str, list] | None = None) -> None:
    """Bring one mirrored directory (and its non-scene subdirectories) in line with its
    listing: ``entries``, else the server's index page.

    ponytail: Apache stamps are to the minute with rounded sizes, so a file
    rewritten within a minute at the same size is missed; fine for a published
    release, add ETags if a root is ever rewritten in place.
    """
    if entries is None:
        entries = parse_listing(_get(_url(parts, True)).decode("utf-8", "replace"))
    published = _published(parts[0], entries) if len(parts) == 1 and parts[0].isdigit() else {}
    try:
        old = json.loads((path / LISTING).read_text())
    except (OSError, ValueError):
        old = {}
    for name in old.keys() - entries.keys():
        _remove(path / name)

    synced: dict[str, list] = {}
    fetch: list[str] = []
    for name, entry in entries.items():
        changed = old.get(name) != entry
        if entry[0]:
            dest = path / name
            dest.mkdir(exist_ok=True)
            synced[name] = entry
            if not parts and name.isdigit():  # a scene: synced when it is looked at
                if changed:
                    _synced.pop(dest, None)
            elif changed or not (dest / LISTING).is_file():
                _sync(dest, (*parts, name), published.get(name))
        elif _lazy((*parts, name)):  # no Path per entry: 166k of them in a long scene
            if changed and name in old:
                (path / name).unlink(missing_ok=True)  # fetched again by local()
            synced[name] = entry
        elif not changed and (path / name).is_file():
            synced[name] = entry
        else:
            fetch.append(name)

    def get(name: str) -> str | None:
        try:
            _download(_url((*parts, name)), path / name)
            return name
        except OSError as exc:  # left out of the listing, so the next sync retries it
            logger.warning("cannot fetch %s: %s", _url((*parts, name)), exc)
            return None

    for name in _pool.map(get, fetch):
        if name:
            synced[name] = entries[name]
    if synced != old:
        tmp = path / f"{LISTING}.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(synced, sort_keys=True))
        os.replace(tmp, path / LISTING)
        st = (path / LISTING).stat()
        with _listings_guard:  # read next by listdir/is_file: spare parsing it again
            _listings[path] = ((st.st_mtime_ns, st.st_ino), synced)
            while len(_listings) > LISTINGS_KEPT:
                del _listings[next(iter(_listings))]


def _published(scene: str, entries: dict[str, list]) -> dict[str, dict[str, list]]:
    """Listings of the scene's subdirectories from ``listings/<scene>.json.gz``, for
    those the file is newer than (both by the dates the server prints); {} without one."""
    root = _mount[1]
    name = f"{scene}.json.gz"
    try:
        stamp = json.loads((root / PUBLISHED_DIR / LISTING).read_text())[name][1]
        z = zlib.decompressobj(16 + zlib.MAX_WBITS)
        raw = z.decompress(local(root / PUBLISHED_DIR / name).read_bytes(), MAX_BYTES)
        if z.unconsumed_tail:
            raise ValueError(f"{name} inflates past {MAX_BYTES >> 20} MB")
        doc = json.loads(raw)
        dirs = doc["dirs"] if doc.get("format") == PUBLISHED_FORMAT else {}
    except (OSError, ValueError, KeyError, TypeError, AttributeError, zlib.error) as exc:
        if not isinstance(exc, FileNotFoundError | KeyError):
            logger.warning("ignoring %s/%s: %s", PUBLISHED_DIR, name, exc)
        return {}
    if not _MINUTE.match(stamp):
        return {}
    out: dict[str, dict[str, list]] = {}
    for sub, files in dirs.items() if isinstance(dirs, dict) else ():
        entry = entries.get(sub)
        # To the minute: the same minute could be either order, so it is not trusted.
        if not (entry and entry[0] and _MINUTE.match(entry[1]) and stamp[:16] > entry[1][:16]):
            continue
        if isinstance(files, dict):
            # Already listing entries, [False, mtime]; held to parse_listing's names.
            out[sub] = {
                n: e for n, e in files.items() if _NAME.fullmatch(n) and type(e) is list and len(e) == 2 and not e[0]
            }
    return out


def write_listings(root: Path, workers: int = 16) -> int:
    """Write ``listings/<scene>.json.gz`` for every scene of a local root: each
    subdirectory holding only files, as listing entries ``{name: [false, mtime]}``
    (the mtime says when to fetch a file again; sizes would only cost bytes). Returns the
    number of files listed. Run it last, after anything else writes into the scenes."""
    root = Path(root)
    out = root / PUBLISHED_DIR
    out.mkdir(exist_ok=True)
    scenes = sorted(d.name for d in root.iterdir() if d.name.isdigit() and d.is_dir())

    def one(scene: str) -> int:
        dirs: dict[str, dict[str, list]] = {}
        with os.scandir(root / scene) as entries:
            for d in entries:
                if not d.is_dir() or d.name.startswith("."):
                    continue
                files: dict[str, list] | None = {}
                with os.scandir(d.path) as inner:
                    for f in inner:
                        if f.name.startswith("."):
                            continue
                        if not f.is_file():
                            files = None  # nested: left to the server's index
                            break
                        st = f.stat()
                        files[f.name] = [False, int(st.st_mtime)]
                if files is not None:
                    dirs[d.name] = files
        doc = json.dumps({"format": PUBLISHED_FORMAT, "dirs": dirs}, separators=(",", ":"))
        tmp = out / f".{scene}.{os.getpid()}.tmp"
        tmp.write_bytes(gzip.compress(doc.encode(), 6))
        os.replace(tmp, out / f"{scene}.json.gz")
        return sum(len(f) for f in dirs.values())

    with ThreadPoolExecutor(max_workers=workers) as pool:  # network storage: overlap the stats
        listed = sum(pool.map(one, scenes))
    for stale in out.glob("*.json.gz"):
        if stale.name.split(".")[0] not in scenes:
            stale.unlink()
    return listed


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Write a root's listings/ for viewers that open it by URL.")
    ap.add_argument("root", type=Path)
    a = ap.parse_args()
    print(f"{a.root / PUBLISHED_DIR}: {write_listings(a.root)} files listed")
