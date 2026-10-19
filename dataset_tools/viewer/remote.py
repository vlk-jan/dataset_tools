"""A dataset root served over HTTP (an Apache-style directory index), mirrored on demand.

The rest of the viewer only reads local files. :func:`mount` maps a URL root onto
a local mirror directory, which is then used as the root:

- a directory's listing is synced by :func:`refresh`, at most every
  :data:`REFRESH_SECONDS`; the root syncs everything but its scenes, a scene
  syncs when it is first looked at;
- files directly in the root, a scene or another top-level directory (manifest,
  odometry, calibration, ``atlas/atlas.json``, ...) are downloaded by the sync;
- files deeper down (``clouds/00000.npy``, ``images/``, atlas thumbnails) and
  every ``.npy`` (a long scene's ``joint_states.npy`` alone is 15 MB) are empty
  placeholders until :func:`local` is asked for one; :func:`head` reads the
  start of one without fetching it.

Each mirrored directory keeps the listing it was synced from in ``.listing.json``.
A scene's fingerprint is that file's digest, so fetching a frame never moves it.
"""

from __future__ import annotations

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
    """``path``, downloaded first when it is a mirrored placeholder.

    Raises ``OSError`` when the download fails; a missing file is left to the reader.
    """
    parts = _parts(path)
    if _placeholder(path, parts):
        _download(_url(parts), path)
    return path


def head(path: Path) -> bytes | None:
    """The first 4 KiB of a mirrored placeholder, by a range request.

    ``None`` for any other path: read that one directly.
    """
    parts = _parts(path)
    if not _placeholder(path, parts):
        return None
    return _get(_url(parts), {"Range": "bytes=0-4095"})[:4096]


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
    return len(parts) > 2 or parts[-1].endswith(".npy")  # deeper than a scene's own files, or an array


def _placeholder(path: Path, parts: tuple[str, ...] | None) -> bool:
    if not parts or not _lazy(parts):
        return False
    try:
        return path.stat().st_size == 0
    except FileNotFoundError:
        return False


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
    shutil.rmtree(path, ignore_errors=True) if path.is_dir() else path.unlink(missing_ok=True)


def _sync(path: Path, parts: tuple[str, ...]) -> None:
    """Bring one mirrored directory (and its non-scene subdirectories) in line with its listing.

    ponytail: Apache stamps are to the minute with rounded sizes, so a file
    rewritten within a minute at the same size is missed; fine for a published
    release, add ETags if a root is ever rewritten in place.
    """
    entries = parse_listing(_get(_url(parts, True)).decode("utf-8", "replace"))
    try:
        old = json.loads((path / LISTING).read_text())
    except (OSError, ValueError):
        old = {}
    for name in old.keys() - entries.keys():
        _remove(path / name)

    synced: dict[str, list] = {}
    fetch: list[str] = []
    for name, entry in entries.items():
        dest = path / name
        changed = old.get(name) != entry
        if entry[0]:
            dest.mkdir(exist_ok=True)
            synced[name] = entry
            if not parts and name.isdigit():  # a scene: synced when it is looked at
                if changed:
                    _synced.pop(dest, None)
            elif changed or not (dest / LISTING).is_file():
                _sync(dest, (*parts, name))
        elif not changed and dest.is_file():
            synced[name] = entry
        elif not _lazy((*parts, name)):
            fetch.append(name)
        else:
            open(dest, "wb").close()  # placeholder: fetched by local()
            synced[name] = entry

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
