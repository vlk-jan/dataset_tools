"""The viewer's web application.

Three groups of routes:

``/``, ``/scene/{id}``, ``/view/{id}/{window}``
    Server-rendered pages: the scene index, one scene's detail, and the 3D
    viewer wrapper. The wrapper is SimView's own page (read from the installed
    package at request time) plus an overlay panel of our own.

``/api/data/{scene}/{window}/...``
    The static-mode bundle the SimView front-end fetches: ``model.json``,
    ``states.json`` and ``blob/{n}``. Built on demand, cached on disk, served
    with byte-range support so the front-end can window long trajectories.

``/api/scene/{id}/frame/{n}/...``
    Per-frame artefacts for the side panel -- camera image, BEV height map,
    validity mask, cloud top-down -- rendered one frame at a time.

Every dataset read happens in a worker thread: the root is usually on contended
network storage, and a slow read must not block the event loop serving the page.
"""

from __future__ import annotations

import asyncio
import bisect
import csv
import dataclasses
import functools
import hashlib
import io
import json
import logging
import re
import shutil
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import cv2
import numpy as np
import simview
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi import Path as PathParam
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import geo, render
from .build import META_VERSION, BuildOptions, SceneAccess, build_scene, read_driven_similarity
from .index_page import THUMB_SLOTS, build_index, split_mark, thumb_index
from .loaders import (
    HM_EXTENT,
    HM_RES,
    VALIDITY_LABELS,
    load_cloud,
    load_npz_key,
    read_timestamps,
    roll_pitch_from_gravity_align,
)
from .notes import NoteStore, NotesUnavailable
from .probe import ARTEFACTS, BUNDLE_DIR_NAME, DatasetRoot, _npy_shape, scene_fingerprint
from .scene_page import build_scene_page, map_card
from .segment_page import COLUMN_TITLES, attitude_per_window, build_segment_page, segment_windows
from .segment_page import find as find_segment
from .signals import MAX_GT_SPEED, signal_charts
from .splits import SPLIT_NAMES, SPLIT_ORDER
from .windows import (
    DEFAULT_MAX_FRAMES,
    DEFAULT_MAX_SECONDS,
    Window,
    describe_reason,
    frame_window_key,
    neighbour_windows,
)
from .wire import render_viewer_page, scene_document, write_static_bundle


class _CachedStatic(StaticFiles):
    """StaticFiles that sends one ``Cache-Control`` with every file it serves."""

    def __init__(self, *, cache_control: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.cache_control = cache_control

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = self.cache_control
        return response


logger = logging.getLogger("dataset_tools.viewer.app")

HERE = Path(__file__).parent
#: Pixel size of a frame-page BEV tile; the page's zoom asks for up to 1536.
TILE_SIZE = 384
#: Per-frame renders and thumbnails: kept by the browser, but asked about every
#: time (``no-cache``), and answered with a 304 while the scene's fingerprint,
#: the atlas and the viewer's code are what the ETag says. Their URLs name only
#: a scene number and a frame index, which a re-extraction or a renumbering
#: points at other pictures -- a lifetime of any length shows the old ones.
SCENE_ASSET_RE = re.compile(r"^/api/scene/(\d+)/(frame/\d+/[a-z]+\.(jpg|png|json)|thumb/[a-z]+\.jpg)$")
#: The shapes a window key (``windows.py``) and a build-options key
#: (``BuildOptions.cache_key``) can take. Both are URL segments that end up in
#: a cache *path*, so anything else is refused before a path is formed.
WINDOW_KEY_RE = re.compile(r"^(seg\d+(\.\d+)?|raw(\.\d+)?|frame\d+)$")
OPTS_KEY_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
#: Scene handles kept at once, least recently used dropped first.
CACHE_SCENES = 4
#: Default ceiling on the built 3D windows kept on disk.
DEFAULT_CACHE_MAX_GB = 5.0


def _warm_build_imports() -> None:
    """Import SimView's authoring API and torch ahead of the first build.

    ``build_scene`` imports them lazily so that ``import dataset_tools.viewer``
    stays cheap, but the first window anyone opens after the viewer starts then
    paid about a second for the torch import inside the request. Done here on
    a background thread, that second is spent while the index is being looked
    at instead.
    """
    try:
        import simview.scene  # noqa: F401
        import simview.state  # noqa: F401
        import torch  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - the build reports it properly on demand
        logger.debug("build imports not warmed: %s", exc)


class ViewerState:
    """Shared, per-process state: the dataset root, caches and build locks."""

    def __init__(
        self,
        root: DatasetRoot,
        max_frames: int,
        max_seconds: float = DEFAULT_MAX_SECONDS,
        cache_max_bytes: float = DEFAULT_CACHE_MAX_GB * 1e9,
    ) -> None:
        self.root = root
        self.max_frames = max_frames
        self.max_seconds = max_seconds
        self.cache_max_bytes = cache_max_bytes
        self._access: OrderedDict[str, SceneAccess] = OrderedDict()
        self._locks: dict[str, asyncio.Lock] = {}
        self.bundle_dir = root.cache_dir / BUNDLE_DIR_NAME
        self.bundle_dir.mkdir(parents=True, exist_ok=True)
        #: What each bundle's meta.json said, keyed by its path and mtime: every
        #: blob of a window asks whether the bundle is current, and parsing a
        #: 0.7 MB meta.json each time blocked the event loop for ~0.3 s a window.
        self.meta_memo: dict[Path, tuple[int, tuple]] = {}

    def access(self, scene_id: str) -> SceneAccess:
        """A cached per-scene handle. Bounded so a long session browsing many
        scenes does not accumulate every scene's arrays in memory.

        Dropped when the scene directory has changed since the handle was
        made: a scene still being written gains height maps and a timestamp
        file while it is looked at, and a handle from before that would keep
        serving the old frame list. ``root.probe`` is one ``stat`` while the
        mtime holds, so the check costs nothing on a finished root.
        """
        cached = self._access.pop(scene_id, None)
        if cached is None or self.root.probe(scene_id) is not cached.probe:
            cached = SceneAccess(self.root, scene_id)
        self._access[scene_id] = cached
        while len(self._access) > CACHE_SCENES:
            self._access.popitem(last=False)
        return cached

    def lock(self, key: str) -> asyncio.Lock:
        return self._locks.setdefault(key, asyncio.Lock())


def create_app(
    root_path: str | Path,
    max_frames: int = DEFAULT_MAX_FRAMES,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    cache_dir: str | Path | None = None,
    notes_dir: str | Path | None = None,
    notes: bool = True,
    author: str = "",
    clear_cache: bool = False,
    allowed_hosts: Sequence[str] | None = None,
    prefetch: bool = True,
    cache_max_gb: float = DEFAULT_CACHE_MAX_GB,
) -> FastAPI:
    """The viewer as an ASGI app.

    ``allowed_hosts`` restricts the ``Host`` header the server answers to. The
    server has no authentication -- notes are writes -- so a viewer bound to
    the loopback interface passes the loopback names here, which stops a page
    on another site from reaching it through a hostname that resolves to
    127.0.0.1 (DNS rebinding). ``None`` answers every host.

    ``prefetch`` starts building the window after the one just opened, so
    paging through a scene with ``[`` / ``]`` does not wait on a build; off
    for a root on storage slow enough that an unasked build competes with the
    one being looked at.
    """
    root = DatasetRoot(root_path, cache_dir=Path(cache_dir) if cache_dir else None, clear_cache=clear_cache)
    state = ViewerState(root, max_frames=max_frames, max_seconds=max_seconds, cache_max_bytes=cache_max_gb * 1e9)
    # Resolved once: the pages report where notes go and why they cannot be
    # written when they cannot, and probing that per request would put a write
    # on the hot path of an otherwise read-only server.
    notes_store = NoteStore(
        root.path,
        notes_dir=Path(notes_dir) if notes_dir else None,
        enabled=notes,
        author=author,
        dataset_version=str((root.manifest() or {}).get("dataset_version") or ""),
    )
    if notes_store.enabled and not notes_store.writable:
        logger.warning("notes are read-only: %s", notes_store.reason)

    app = FastAPI(title="dataset_tools viewer", docs_url=None, redoc_url=None)
    if allowed_hosts is not None:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(allowed_hosts))

    templates = Jinja2Templates(directory=str(HERE / "templates"))
    templates.env.globals["reason_text"] = describe_reason
    templates.env.globals["column_titles"] = COLUMN_TITLES

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        """A page for a person, JSON for everything else: a mistyped scene or
        frame number used to answer a browser with ``{"detail": ...}`` and no
        way back to the index."""
        if request.url.path.startswith("/api/") or "text/html" not in request.headers.get("accept", ""):
            return await http_exception_handler(request, exc)
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={"root": root, "status": exc.status_code, "detail": exc.detail},
            status_code=exc.status_code,
        )

    app.mount(
        "/viewer-static",
        # Only SimView's entry point carries a cache-bust; with no
        # Cache-Control a browser keeps OLD modules after an upgrade.
        # ``no-cache`` keeps the file cached but forces a conditional request:
        # a cheap 304, or the upgrade picked up at once.
        _CachedStatic(directory=str(HERE / "static"), cache_control="no-cache"),
        name="viewer_static",
    )

    simview_static = str(files("simview") / "static")
    simview_ver = simview.__version__
    # The entry point and every ES module it imports resolve under a stamped
    # path, so a change to SimView changes every module's URL at once and the
    # browser cannot run a stale mix of old and new modules: the file at a URL
    # never changes, so it is cached forever. This is what render_viewer_page
    # points at. The stamp is the version *and* a fingerprint of the files: an
    # editable install changes them without changing the version, and a
    # version-only stamp left browsers on a year-cached module that imported
    # what the new page no longer mapped ("chroma", simview dfc3294).
    simview_stamp = f"{simview_ver}-{_static_fingerprint(Path(simview_static))}"
    app.mount(
        f"/simview-static/{simview_stamp}",
        _CachedStatic(directory=simview_static, cache_control="public, max-age=31536000, immutable"),
        name="simview_static_versioned",
    )
    # SimView's own document-relative asset paths (config.js: "static/textures/...").
    app.mount(
        "/static",
        _CachedStatic(directory=simview_static, cache_control="no-cache"),
        name="simview_static_root",
    )
    threading.Thread(target=_warm_build_imports, name="warm-build-imports", daemon=True).start()

    # ------------------------------------------------------------------ pages

    # Cache-immunity stamp, folded into every /api/data URL: bundle content
    # can change under an unchanged URL when the viewer code changes, and a
    # browser that cached the old response (with any freshness lifetime) would
    # mix stale trajectory blobs with fresh model.json. A new stamp on every
    # code change makes previously cached entries unaddressable.
    _h = hashlib.sha1()
    # Every module whose output ends up inside a bundle, not just the three
    # that write it: a change to how windows are cut (windows.py), how poses
    # are interpolated (loaders.py) or how cloud colours are chosen (render.py)
    # changes the bytes at an unchanged URL just as surely as build.py does.
    for _f in (
        "build.py",
        "wire.py",
        "app.py",
        "windows.py",
        "loaders.py",
        "render.py",
    ):
        try:
            _h.update(str((Path(__file__).parent / _f).stat().st_mtime_ns).encode())
        except OSError:  # pragma: no cover - a zipped/partial install
            _h.update(_f.encode())
    stamp = _h.hexdigest()[:8]

    @app.middleware("http")
    async def revalidate_scene_assets(request: Request, call_next):
        """ETag + ``no-cache`` on every per-frame render and thumbnail (``SCENE_ASSET_RE``).

        The tag is computed before anything is rendered, from what the picture
        depends on: the scene's fingerprint, the atlas (curated thumbnails), the
        viewer's code and the URL with its query. A browser that has the bytes
        gets a 304 and no render at all.
        """
        match = SCENE_ASSET_RE.match(request.url.path)
        if not match:
            return await call_next(request)
        try:
            fingerprint = scene_fingerprint(root.path / match.group(1))
        except OSError:
            return await call_next(request)  # unknown scene: the route says so
        try:
            atlas = (root.atlas_dir / "atlas.json").stat().st_mtime_ns
        except OSError:
            atlas = 0
        key = f"{fingerprint}|{atlas}|{stamp}|{request.url.path}?{request.url.query}"
        etag = f'"{hashlib.sha1(key.encode()).hexdigest()[:20]}"'
        headers = {"ETag": etag, "Cache-Control": "no-cache"}
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers=headers)
        response = await call_next(request)
        if response.status_code == 200:
            response.headers.update(headers)
        return response

    @app.get("/", response_class=HTMLResponse)
    async def index(
        request: Request,
        status: str | None = Query(None, pattern="^(ok|warn|fail)$"),
        verdict: str | None = Query(None, pattern="^(keep|drop|fix)$"),
        split: str = Query("", pattern=f"^({'|'.join(SPLIT_ORDER)})?$"),
    ):
        def work():
            return build_index(root, root.probe_all(), status, notes_store.list_all(), verdict, split)

        page = await run_in_threadpool(work)
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "root": root,
                "artefacts": ARTEFACTS,
                "manifest": root.manifest(),
                "simview_version": simview_ver,
                "notes_enabled": notes_store.enabled,
                **page,
            },
        )

    @app.get("/notes", response_class=HTMLResponse)
    async def notes_index(request: Request, topic: str | None = Query(None, max_length=32)):
        """Every note in the root on one page, grouped by topic.

        Notes are written one scene at a time, from wherever a bad stretch was
        noticed; this is where they are read back as a list of what is wrong
        with the root. With notes off the page still answers, saying so.
        """
        page = await run_in_threadpool(_notes_index, state, notes_store, topic)
        return templates.TemplateResponse(request=request, name="notes.html", context={"root": root, **page})

    @app.get("/api/notes")
    async def api_notes_index(topic: str | None = Query(None, max_length=32)):
        return await run_in_threadpool(_notes_index, state, notes_store, topic)

    #: How the segments table can be ordered: key -> (label, sort key, descending).
    segment_sorts = {
        "scene": ("scene", lambda r: (r["scene"], r["window"].frame_start), False),
        "att": ("GT attitude", lambda r: r["att"]["median"] if r["att"] else -1.0, True),
        "rate": ("pose rate", lambda r: r["window"].pose_rate or 0.0, False),
        "duration": ("duration", lambda r: r["window"].duration, True),
    }
    segment_rows: dict[tuple, list[dict]] = {}

    @app.get("/segments", response_class=HTMLResponse)
    async def segments_page(
        request: Request,
        sort: str = Query("scene", pattern="^(scene|att|rate|duration)$"),
        split: str = Query("", pattern=f"^({'|'.join(SPLIT_ORDER)})?$"),
    ):
        """Every ground-truth window of the root in one table.

        The GT-versus-IMU verdict means reading each scene's odometry and
        gravity, so the rows are kept for as long as no scene's fingerprint
        moves; notes are read fresh every time.
        """

        def work() -> list[dict]:
            ids = root.scene_ids(refresh=True)
            key = tuple((sid, root.probe(sid).fingerprint) for sid in ids) + (state.max_frames, state.max_seconds)
            if key not in segment_rows:
                rows = []
                for sid in ids:
                    access = state.access(sid)
                    wins = [w for w in access.windows(state.max_frames, state.max_seconds) if w.kind == "segment"]
                    att = attitude_per_window(access, wins)
                    rows += [{"scene": sid, "window": w, "att": att.get(w.key)} for w in wins]
                segment_rows.clear()
                segment_rows[key] = rows
            notes = notes_store.list_all()
            # Read fresh like the notes: splits.json is not in any scene's fingerprint.
            marks = {sid: split_mark(root, sid) for sid in ids}
            out = []
            for r in segment_rows[key]:
                w = r["window"]
                touching = [n for n in notes.get(r["scene"], []) if n.overlaps(w.frame_start, w.frame_end)]
                verdicts = sorted({n.verdict for n in touching if n.verdict})
                out.append({**r, "notes": touching, "verdicts": verdicts, "split": marks[r["scene"]]})
            return out

        rows = await run_in_threadpool(work)
        # Which sides the menu offers: the ones some row is on, in drawing order.
        sides = [s for s in SPLIT_ORDER if any(r["split"] and r["split"]["which"] == s for r in rows)]
        if split:
            rows = [r for r in rows if r["split"] and r["split"]["which"] == split]
        _, by, descending = segment_sorts[sort]
        rows.sort(key=by, reverse=descending)
        return templates.TemplateResponse(
            request=request,
            name="segments.html",
            context={
                "root": root,
                "rows": rows,
                "n_scenes": len({r["scene"] for r in rows}),
                "sort": sort,
                "sorts": [(k, v[0]) for k, v in segment_sorts.items()],
                "split": split,
                "splits": [(s, SPLIT_NAMES[s]) for s in sides],
            },
        )

    @app.get("/scene/{scene_id}", response_class=HTMLResponse)
    async def scene_page(
        request: Request,
        scene_id: str,
        tab: str = Query("windows", pattern="^(windows|frames)$"),
        page: int = Query(0, ge=0),
    ):
        access = _access_or_404(state, scene_id)
        windows = await run_in_threadpool(access.windows, state.max_frames, state.max_seconds)
        # The thumbnails and the four cards. Cached on the scene handle, so
        # switching tabs does not re-read this scene's small arrays.
        card_page = await run_in_threadpool(build_scene_page, root, access)
        # The frame strip is only assembled when it is the tab being shown: it
        # is the one part of this page that may list a directory.
        frames = await run_in_threadpool(_frame_page, access, page) if tab == "frames" else None
        scene_notes = await run_in_threadpool(notes_store.list, scene_id)
        return templates.TemplateResponse(
            request=request,
            name="scene.html",
            context={
                "root": root,
                "data_problem": await run_in_threadpool(lambda: access.clock_problem),
                "scene_id": scene_id,
                "scene_ids": root.scene_ids(),
                "probe": access.probe,
                "windows": windows,
                "artefacts": ARTEFACTS,
                "tab": tab,
                "frames": frames,
                "page": card_page,
                "atlas": root.atlas_scene(scene_id),
                "notes": scene_notes,
                "notes_status": notes_store.status(),
                "notes_stale": await run_in_threadpool(notes_store.stale_version, scene_id),
                # Which windows a note lands in, so the table says where the
                # flagged stretches are without anyone reading every note.
                "notes_per_window": _notes_per_window(scene_notes, windows),
                # How far each window's ground truth has wandered from the
                # IMU, so the segments worth opening are told apart here.
                "attitude_per_window": await run_in_threadpool(attitude_per_window, access, windows),
            },
        )

    @app.get("/scene/{scene_id}/segment/{segment_index}", response_class=HTMLResponse)
    async def segment_detail(
        request: Request,
        scene_id: str,
        segment_index: int,
        page: int = Query(0, ge=0),
    ):
        """One ground-truth segment: what it is, what happened inside it, where.

        Addressed by the number in ``odometry_segments.txt`` -- the same number
        the window keys carry -- so a link from a window, from a frame or from
        another segment all resolve the same way.
        """
        access = _access_or_404(state, scene_id)
        found = await run_in_threadpool(find_segment, access, segment_index)
        if found is None:
            raise HTTPException(404, f"scene {scene_id} has no segment {segment_index}")
        position, seg = found
        detail = await run_in_threadpool(build_segment_page, access, position, seg, state.max_frames, state.max_seconds)
        lo, hi = detail["frames"]
        frames = await run_in_threadpool(_frame_page, access, page, (lo, hi))
        scene_notes = await run_in_threadpool(notes_store.list, scene_id)
        return templates.TemplateResponse(
            request=request,
            name="segment.html",
            context={
                "root": root,
                "data_problem": await run_in_threadpool(lambda: access.clock_problem),
                "scene_id": scene_id,
                "seg": seg,
                "probe": access.probe,
                "page": detail,
                "frames": frames,
                # Only the notes that touch this segment; the card says so and
                # links to the scene page, which has all of them.
                "notes": [n for n in scene_notes if n.overlaps(lo, hi)],
                "notes_status": notes_store.status(),
                "notes_stale": await run_in_threadpool(notes_store.stale_version, scene_id),
                "attitude_per_window": await run_in_threadpool(attitude_per_window, access, detail["windows"]),
            },
        )

    @app.get("/scene/{scene_id}/jump")
    async def frame_jump(scene_id: str, index: int = Query(0, ge=0)):
        """The frames tab's jump box: clamp to the scene and go to that frame."""
        access = _access_or_404(state, scene_id)
        last = max(0, access.probe.n_frames - 1)
        return RedirectResponse(f"/scene/{scene_id}/frame/{min(index, last)}", status_code=303)

    @app.get("/scene/{scene_id}/frame/{index}", response_class=HTMLResponse)
    async def frame_page(request: Request, scene_id: str, index: int):
        access = _access_or_404(state, scene_id)
        n_frames = access.probe.n_frames
        if index < 0 or (n_frames and index >= n_frames):
            raise HTTPException(404, f"scene {scene_id} has no frame {index}")
        details = await run_in_threadpool(_frame_details, access, index, state.max_frames, state.max_seconds)
        # A frame past the end of data_ts.txt has no clock to sit on, so there
        # is no single-frame window to build for it; offering the button anyway
        # led straight to a 500.
        single = await run_in_threadpool(access.window, frame_window_key(index), state.max_frames, state.max_seconds)
        # Where this frame was, when the recording says: the scene's track with
        # a marker on it. None without gps.npy, and the page then has no map.
        frame_map = await run_in_threadpool(map_card, access, None, index)
        scene_notes = await run_in_threadpool(notes_store.list, scene_id)
        return templates.TemplateResponse(
            request=request,
            name="frame.html",
            context={
                "root": root,
                "data_problem": await run_in_threadpool(lambda: access.clock_problem),
                "scene_id": scene_id,
                "index": index,
                "probe": access.probe,
                "details": details,
                "frame_window": single.key if single is not None else None,
                "map": frame_map,
                "signals": await run_in_threadpool(_frame_signals, access, details.get("timestamp")),
                # Only the notes whose range holds this frame.
                "notes": [n for n in scene_notes if n.overlaps(index, index)],
                "notes_status": notes_store.status(),
                "notes_stale": await run_in_threadpool(notes_store.stale_version, scene_id),
            },
        )

    @app.get("/view/{scene_id}/{window_key}", response_class=HTMLResponse)
    async def viewer_page(request: Request, scene_id: str, window_key: str):
        access = _access_or_404(state, scene_id)
        window = await run_in_threadpool(access.window, window_key, state.max_frames, state.max_seconds)
        if window is None:
            raise HTTPException(404, f"scene {scene_id} has no window {window_key!r}")

        opts = _options_from_query(request.query_params)
        bundle_key = _bundle_key(scene_id, window_key, opts)
        # The windows either side of this one, so the panel can page through a
        # scene's segments without going back to the scene page each time.
        windows = await run_in_threadpool(access.windows, state.max_frames, state.max_seconds)
        prev_window, next_window = neighbour_windows(windows, window)
        # ?frame=N -- arrived here from that frame's page, so open the playhead
        # on it instead of at the start of the window.
        try:
            initial_frame = int(request.query_params["frame"])
        except (KeyError, TypeError, ValueError):
            initial_frame = None
        overlay = templates.get_template("overlay.html").render(
            scene_id=scene_id,
            window=window,
            prev_window=prev_window,
            next_window=next_window,
            # Build options are carried across, so paging to the next segment
            # keeps the cloud/terrain settings the current view was opened
            # with; ?frame= is not -- it names a frame of *this* window.
            view_query=_view_query(request.query_params),
            probe=access.probe,
            # Whether the camera tile's zoom can project the cloud onto it.
            cloud_on_image=_overlay_possible(access),
            # How many segments the scene has, so the panel can title the view
            # "Segment 3/17" rather than leaving "3 of how many?" to a trip
            # back to the scene page.
            n_segments=len(access.segments),
            scene_url=f"/scene/{scene_id}",
            notes=notes_store.status(),
        )
        config = json.dumps(
            {
                "frameBase": f"/api/scene/{scene_id}/frame",
                "framePage": f"/scene/{scene_id}/frame",
                "metaUrl": f"/api/data/{bundle_key}/{stamp}/meta.json",
                "initialFrame": initial_frame,
                "notesUrl": f"/api/scene/{scene_id}/notes",
                "notes": notes_store.status(),
                "frameStart": window.frame_start,
                "frameEnd": window.frame_end - 1,
            }
        )
        # json.dumps leaves "<" alone, and a "</script>" inside any value (the
        # notes status carries an OS error string) would end the block early.
        config = config.replace("<", "\\u003c")
        head = (
            '<link rel="icon" href="/viewer-static/favicon.svg" type="image/svg+xml">'
            '<link rel="stylesheet" href="/viewer-static/overlay.css">'
            f"<script>window.__dtViewer = {config};</script>"
        )
        body = overlay + '<script type="module" src="/viewer-static/overlay.js"></script>'
        page = await run_in_threadpool(
            render_viewer_page,
            f"/simview-static/{simview_stamp}",
            f"/api/data/{bundle_key}/{stamp}",
            body,
            head,
            f"Scene {scene_id} {window_key} - 3D view",
        )
        # Defer SimView's own entry point: it fetches the scene the instant it
        # runs, and the bundle may still be building. The overlay controller
        # re-inserts this exact script once the build reports ready.
        page = _defer_main_script(page)
        if prefetch and next_window is not None:
            # The window ] pages to, built now rather than when asked for. It
            # takes the same per-bundle lock a request would, so a reader who
            # pages before it is done joins the build instead of starting a
            # second one; and it never surfaces an error -- nobody asked.
            asyncio.ensure_future(_prefetch(state, scene_id, next_window.key, opts.cache_key()))
        return HTMLResponse(page)

    # ------------------------------------------------------------------- api

    @app.get("/api/scene/{scene_id}/gps.json")
    async def api_scene_gps(
        scene_id: str,
        segment: int | None = Query(None, ge=0),
        frame: int | None = Query(None, ge=0),
        max_points: int = Query(2000, ge=2, le=20000),
    ):
        """This scene's GNSS track, thinned for drawing.

        ``segment`` highlights one segment's stretch (the whole scene comes
        back as ``context`` behind it); ``frame`` adds a marker where that
        frame was. A scene without ``gps.npy`` is a 404, not an empty track:
        the page then draws no map at all rather than an empty one.
        """
        access = _access_or_404(state, scene_id)
        payload = await run_in_threadpool(_gps_payload, access, segment, frame, max_points)
        if payload is None:
            raise HTTPException(404, f"scene {scene_id} has no gps.npy")
        return payload

    @app.get("/api/scene/{scene_id}/frame/{index}/image.jpg")
    async def api_frame_image(scene_id: str, index: int, w: int = Query(640, ge=64, le=1920)):
        access = _access_or_404(state, scene_id)
        path = access.frame_path("images", index)
        data = await run_in_threadpool(_safe_render, render.render_camera_image, path, w)
        if data is None:
            return _absent_or_unreadable(path, f"image for frame {index}")
        return Response(data, media_type="image/jpeg")

    @app.get("/api/scene/{scene_id}/frame/{index}/overlay.jpg")
    async def api_frame_overlay(scene_id: str, index: int, w: int = Query(1280, ge=64, le=1920)):
        """The frame's cloud projected into its image: a check on the extrinsic.

        404 rather than a bare picture when the scene cannot support it -- no
        ``T_cam_to_base.txt`` (a v1 root's ``T_cam_to_lidar.txt`` is in the
        lidar frame and would draw the cloud in the wrong place), no ``K``, or
        no image or cloud for this frame.
        """
        access = _access_or_404(state, scene_id)
        if not _overlay_possible(access):
            raise HTTPException(404, f"scene {scene_id} has no camera calibration to project with")

        def work():
            image = access.frame_path("images", index)
            cloud_path = access.frame_path("clouds", index)
            cloud = load_cloud(cloud_path)
            if cloud is None or not image.is_file():
                return cloud_path if cloud is None else image
            info = access.camera_info or {}
            return _safe_render(
                render.render_cloud_on_image,
                image,
                cloud,
                access.K,
                access.T_cam,
                w,
                distortion_model=info.get("distortion_model"),
                distortion_coeffs=info.get("distortion_coefficients"),
            )

        data = await run_in_threadpool(work)
        if data is None or isinstance(data, Path):
            return _absent_or_unreadable(data, f"image and cloud to overlay for frame {index}")
        jpeg, span = data
        # The depth the colours run over, for the legend the page draws on top.
        headers = {"X-Depth-Range": f"{span[0]:.2f} {span[1]:.2f}"} if span else {}
        return Response(jpeg, media_type="image/jpeg", headers=headers)

    @app.get("/api/scene/{scene_id}/thumb/{slot}.jpg")
    async def api_scene_thumb(scene_id: str, slot: str):
        """One index thumbnail: the curated file if the root has one, else a live frame.

        The two cases are deliberately the same URL. A root with an ``atlas/``
        serves a stored, immutable JPEG (43 scenes x 3 of them land in one page
        load, so they must be cacheable forever and never re-encoded); a root
        without one renders the frame at the same fraction of the scene. The
        index does not have to know which kind of root it is showing.
        """
        access = _access_or_404(state, scene_id)
        if slot not in THUMB_SLOTS:
            raise HTTPException(404, f"no thumbnail slot {slot!r}")
        stored = root.atlas_thumb(scene_id, slot)
        if stored is not None:
            return FileResponse(stored, media_type="image/jpeg")

        n_frames = access.probe.n_frames
        if not n_frames or not access.probe.has("images"):
            raise HTTPException(404, f"scene {scene_id} has no images to thumbnail")
        # The same resolver the index and the scene page use, so a curated index
        # with no file behind it still renders the frame it names.
        index = thumb_index(root.atlas_scene(scene_id), slot, n_frames)

        def work():
            nearest = access.nearest_frame("images", index)
            if nearest is None:
                return None
            return _safe_render(render.render_camera_image, access.frame_path("images", nearest), 384)

        data = await run_in_threadpool(work)
        if data is None:
            raise HTTPException(404, f"no image to thumbnail for scene {scene_id}")
        return Response(data, media_type="image/jpeg")

    # ``size`` is the tile's pixel size: the frame page's tiles ask for the
    # default, its click-to-zoom asks for a bigger copy of the same picture.
    @app.get("/api/scene/{scene_id}/frame/{index}/heightmap.png")
    async def api_frame_heightmap(scene_id: str, index: int, size: int = Query(TILE_SIZE, ge=128, le=1536)):
        access = _access_or_404(state, scene_id)
        return await _npz_png(access, index, "height_maps", _npz_key("height_map"), render.render_height_map, size)

    @app.get("/api/scene/{scene_id}/frame/{index}/validity.png")
    async def api_frame_validity(scene_id: str, index: int, size: int = Query(TILE_SIZE, ge=128, le=1536)):
        access = _access_or_404(state, scene_id)
        return await _npz_png(access, index, "height_masks", _npz_key("validity"), render.render_validity, size)

    @app.get("/api/scene/{scene_id}/frame/{index}/driven.png")
    async def api_frame_driven(scene_id: str, index: int, size: int = Query(TILE_SIZE, ge=128, le=1536)):
        access = _access_or_404(state, scene_id)
        return await _npz_png(access, index, "driven_similarity", read_driven_similarity, _render_driven, size)

    @app.get("/api/scene/{scene_id}/frame/{index}/grid.json")
    async def api_frame_grid(scene_id: str, index: int):
        """The frame's BEV layers as numbers, for the zoomed tile's probe."""
        access = _access_or_404(state, scene_id)
        grid = await run_in_threadpool(_frame_grid, access, index)
        if grid is None:
            raise HTTPException(404, f"no BEV grid for frame {index}")
        return Response(json.dumps(grid, separators=(",", ":")), media_type="application/json")

    @app.get("/api/scene/{scene_id}/frame/{index}/cloud.png")
    async def api_frame_cloud(scene_id: str, index: int, size: int = Query(TILE_SIZE, ge=128, le=1536)):
        access = _access_or_404(state, scene_id)

        path = access.frame_path("clouds", index)

        def work():
            cloud = load_cloud(path)
            if cloud is None:
                return None
            return render.render_cloud_bev(cloud, size=size)

        data = await run_in_threadpool(work)
        if data is None:
            return _absent_or_unreadable(path, f"cloud for frame {index}")
        return Response(data, media_type="image/png")

    # ------------------------------------------------------------------ notes

    # Deliberately NOT under /api/data/: a bundle URL carries a content stamp
    # and its meta.json is cached on disk per window, so a note folded in there
    # would either be invisible until the next code change or force every
    # window it touches to rebuild. Notes are their own resource, fetched
    # uncached, and nothing about them can invalidate a built window.

    @app.get("/api/notes.csv")
    async def api_notes_csv():
        """Every note in the root as one table, for whatever applies the
        verdicts: frame range and the same range as frame times
        (``data_ts.txt``), so a consumer needs neither the viewer nor the
        scene's clock to cut a ``drop`` stretch out."""

        def work() -> str:
            out = io.StringIO()
            writer = csv.writer(out)
            writer.writerow(
                [
                    "scene",
                    "frame_start",
                    "frame_end",
                    "t_start",
                    "t_end",
                    "topic",
                    "verdict",
                    "author",
                    "created",
                    "text",
                ]
            )
            for scene_id, scene_notes in notes_store.list_all().items():
                ts = read_timestamps(root.path / scene_id / "data_ts.txt")
                for n in scene_notes:
                    span = ["", ""]
                    if ts is not None and len(ts):
                        last = len(ts) - 1
                        span = [f"{ts[min(n.frame_start, last)]:.6f}", f"{ts[min(n.frame_end, last)]:.6f}"]
                    writer.writerow(
                        [scene_id, n.frame_start, n.frame_end, *span, n.topic, n.verdict, n.author, n.created, n.text]
                    )
            return out.getvalue()

        return Response(
            await run_in_threadpool(work),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="notes.csv"'},
        )

    @app.get("/api/scene/{scene_id}/notes")
    async def api_notes(scene_id: str):
        access = _access_or_404(state, scene_id)
        scene_notes = await run_in_threadpool(notes_store.list, scene_id)
        return {
            "scene": scene_id,
            "n_frames": access.probe.n_frames,
            "notes": [n.to_json() for n in scene_notes],
            "status": notes_store.status(),
            "stale_version": await run_in_threadpool(notes_store.stale_version, scene_id),
        }

    @app.post("/api/scene/{scene_id}/notes", status_code=201)
    async def api_add_note(scene_id: str, payload: dict = Body(...)):
        access = _access_or_404(state, scene_id)
        start, end = _frames_from_payload(payload)
        note = await _notes_write(
            state,
            scene_id,
            notes_store.add,
            scene_id,
            start,
            end,
            payload.get("text", ""),
            topic=payload.get("topic", "other"),
            author=payload.get("author", ""),
            n_frames=access.probe.n_frames,
            verdict=payload.get("verdict", ""),
        )
        return note.to_json()

    @app.patch("/api/scene/{scene_id}/notes/{note_id}")
    async def api_edit_note(scene_id: str, note_id: str, payload: dict = Body(...)):
        access = _access_or_404(state, scene_id)
        start, end = _frames_from_payload(payload, required=False)
        note = await _notes_write(
            state,
            scene_id,
            notes_store.update,
            scene_id,
            note_id,
            text=payload.get("text"),
            topic=payload.get("topic"),
            frame_start=start,
            frame_end=end,
            n_frames=access.probe.n_frames,
            verdict=payload.get("verdict"),
        )
        if note is None:
            raise HTTPException(404, f"scene {scene_id} has no note {note_id!r}")
        return note.to_json()

    @app.delete("/api/scene/{scene_id}/notes/{note_id}")
    async def api_delete_note(scene_id: str, note_id: str):
        _access_or_404(state, scene_id)
        removed = await _notes_write(state, scene_id, notes_store.delete, scene_id, note_id)
        if not removed:
            raise HTTPException(404, f"scene {scene_id} has no note {note_id!r}")
        return {"deleted": note_id}

    # ------------------------------------------------------------ scene data

    @app.get("/api/data/{scene_id}/{window_key}/{opts_key}/{stamp}/{name}")
    async def api_bundle_json(
        scene_id: str,
        window_key: str,
        opts_key: str,
        stamp: str,
        name: str = PathParam(pattern=r"^(meta|model|states)\.json$"),
    ):
        bundle = await _ensure_bundle(state, scene_id, window_key, opts_key)
        return FileResponse(bundle / name, media_type="application/json")

    @app.get("/api/data/{scene_id}/{window_key}/{opts_key}/{stamp}/blob/{blob_id}")
    async def api_blob(scene_id: str, window_key: str, opts_key: str, stamp: str, blob_id: str):
        if not blob_id.isdigit():
            raise HTTPException(404, "bad blob id")
        bundle = await _ensure_bundle(state, scene_id, window_key, opts_key)
        path = bundle / "blob" / blob_id
        if not path.is_file():
            raise HTTPException(404, f"no blob {blob_id}")
        # FileResponse handles Range itself, which is what lets the SimView
        # front-end window a long trajectory instead of loading it whole.
        # no-cache (NOT max-age): the bundle URL stays the same when a code
        # change rebuilds its content, so the browser must revalidate against
        # the ETag/Last-Modified rather than serve a stale blob for a day --
        # a stale blob mixed with a fresh model.json mangles the robot.
        return FileResponse(
            path,
            media_type="application/octet-stream",
            headers={"Cache-Control": "no-cache", "Accept-Ranges": "bytes"},
        )

    @app.get("/healthz")
    async def healthz():
        return {
            "ok": True,
            "root": str(root.path),
            "scenes": len(root.scene_ids()),
            "notes": notes_store.status(),
        }

    return app


# ---------------------------------------------------------------- helpers


def _frames_from_payload(payload: dict, required: bool = True) -> tuple[int | None, int | None]:
    """The inclusive ``{"frames": [a, b]}`` range a note request names -- the
    stored form, and the only one the pages send.

    An edit may leave the range out (``required=False``) and gets
    ``(None, None)``: the note keeps the range it has.
    """
    frames = payload.get("frames")
    if frames is None and not required:
        return None, None
    try:
        if not isinstance(frames, list | tuple) or len(frames) != 2:
            raise ValueError(frames)
        return int(frames[0]), int(frames[1])
    except (TypeError, ValueError):
        raise HTTPException(400, "a note needs a frame range: frames [first, last]") from None


async def _notes_write(state: ViewerState, scene_id: str, fn, *args, **kwargs):
    """Run one note mutation under the scene's lock, in a worker thread.

    Serialised per scene because every write replaces the whole file: two
    concurrent adds without the lock would each save their own read of it, and
    the second would drop the first.
    """
    async with state.lock(f"notes:{scene_id}"):
        try:
            return await run_in_threadpool(fn, *args, **kwargs)
        except NotesUnavailable as exc:
            raise HTTPException(403, str(exc)) from None
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        except OSError as exc:
            raise HTTPException(500, f"could not save the note: {exc}") from None


def _notes_per_window(notes: list, windows: list) -> dict[str, list]:
    """Which notes touch each window, keyed by window key.

    A note is a range of frames and a window is a range of frames; a note that
    spans a segment boundary legitimately shows up under both windows, since
    both are stretches you would want warned about when you open them.
    """
    return {w.key: [n for n in notes if n.overlaps(w.frame_start, w.frame_end)] for w in windows}


def _bundle_path(state: ViewerState, scene_id: str, window_key: str, opts_key: str) -> Path:
    """The on-disk bundle directory for a request, after checking its parts.

    The options key has to be the one ``BuildOptions.cache_key`` would write
    for the options it names -- so ``..`` is refused before it becomes a path,
    and so is every spelling of the same options but the canonical one, which
    is what keeps a request from minting a fresh build (and a fresh directory)
    per variant of an otherwise identical key.
    """
    if not scene_id.isdigit() or not WINDOW_KEY_RE.match(window_key) or not OPTS_KEY_RE.match(opts_key):
        raise HTTPException(404, f"no bundle {scene_id}/{window_key}/{opts_key}")
    if _options_from_key(opts_key).cache_key() != opts_key:
        raise HTTPException(404, f"no bundle {scene_id}/{window_key}/{opts_key}")
    return state.bundle_dir / scene_id / window_key / opts_key


def _access_or_404(state: ViewerState, scene_id: str) -> SceneAccess:
    try:
        return state.access(scene_id)
    except KeyError:
        raise HTTPException(404, f"unknown scene {scene_id!r}") from None


def _defer_main_script(page: str) -> str:
    """Replace SimView's module entry point with a marker our controller uses.

    SimView's ``main.js`` calls ``SimView.run()`` as soon as it is parsed, which
    would fire the data fetches before the bundle exists. Removing the tag and
    recording its URL lets the overlay insert it after the build is ready --
    the front-end itself is untouched and runs exactly as it normally does.
    """
    match = re.search(r'<script\s+type="module"\s+src="([^"]*main\.js[^"]*)"\s*></script>', page)
    if not match:
        logger.warning("could not find simview's main.js script tag; loading it inline")
        return page
    return page.replace(
        match.group(0),
        f'<script>window.__dtSimviewEntry = "{match.group(1)}";</script>',
        1,
    )


def _options_from_query(params) -> BuildOptions:
    def get_int(name: str, default: int) -> int:
        try:
            return int(params.get(name, default))
        except (TypeError, ValueError):
            return default

    return BuildOptions(
        max_cloud_frames=get_int("cloud_frames", BuildOptions.max_cloud_frames),
        max_cloud_points=get_int("cloud_points", BuildOptions.max_cloud_points),
        max_hm_frames=get_int("hm_frames", BuildOptions.max_hm_frames),
        max_terrain_cells=get_int("terrain_cells", BuildOptions.max_terrain_cells),
        cloud_color=params.get("cloud_color", BuildOptions.cloud_color),
    ).sanitized()


def _view_query(params) -> str:
    """This view's query string, minus ``frame``, ready to append to a URL.

    Empty when nothing is left to carry, so the common link has no trailing
    "?" on it.
    """
    kept = [(k, v) for k, v in params.multi_items() if k != "frame"]
    return f"?{urlencode(kept)}" if kept else ""


def _bundle_key(scene_id: str, window_key: str, opts: BuildOptions) -> str:
    return f"{scene_id}/{window_key}/{opts.cache_key()}"


def _options_from_key(opts_key: str) -> BuildOptions:
    """Recover the build options a bundle path encodes (see ``BuildOptions.cache_key``)."""
    parts = opts_key.split("-")
    fields = dataclasses.fields(BuildOptions)
    if len(parts) != len(fields):
        return BuildOptions()
    try:
        # Each field parsed as the type of its default: int, float or str.
        return BuildOptions(**{f.name: type(f.default)(v) for f, v in zip(fields, parts, strict=True)}).sanitized()
    except ValueError:
        return BuildOptions()


def _bundle_is_current(
    dest: Path, window: Window | None, data: str = "", memo: dict[Path, tuple[int, tuple]] | None = None
) -> bool:
    """Is this cached bundle complete, built by today's code, and still this
    window?

    The cache is keyed by what was asked for (scene, window, build options),
    none of which changes when the builder does -- so a bundle written before a
    viewer upgrade would keep being served, with whatever fields its
    ``meta.json`` happened to have at the time. The payload carries its own
    version (``build.META_VERSION``); anything older is rebuilt rather than
    served. Rebuilding a window costs a couple of seconds, once.

    A window *key* outlives its frame range, which is the other way a cached
    bundle goes stale: ``seg000.0`` is the first part of segment 0 however many
    parts there are, so a cache filled under one ``--max-seconds`` holds an
    entry that a run with another would otherwise serve for a different stretch
    of the scene. So the range is checked against the window as it is cut now,
    and ``window=None`` -- a key this scene no longer offers, such as a part of
    a segment that is no longer split that far -- is never current: there is
    nothing left for that bundle to be a build of.

    The third way is the data itself: a re-extraction, a regenerated
    odometry.txt or new height maps leave the key and the frame range alone.
    ``data`` is the scene's fingerprint (``probe.scene_fingerprint``) now; the
    bundle records the one it was built from.
    """
    if window is None:
        return False
    if not (dest / "model.json").is_file():
        return False
    try:
        mtime = (dest / "meta.json").stat().st_mtime_ns
    except OSError:
        return False
    hit = memo.get(dest) if memo is not None else None
    if hit is not None and hit[0] == mtime:
        summary = hit[1]
    else:
        try:
            meta = json.loads((dest / "meta.json").read_text())
        except (OSError, ValueError):
            return False
        cached = meta.get("window") or {}
        summary = (
            meta.get("meta_version"),
            cached.get("frame_start"),
            cached.get("frame_end"),
            (meta.get("build") or {}).get("data", ""),
        )
        if memo is not None:
            memo[dest] = (mtime, summary)
    return summary == (META_VERSION, window.frame_start, window.frame_end, data)


async def _ensure_bundle(state: ViewerState, scene_id: str, window_key: str, opts_key: str) -> Path:
    """Return the on-disk bundle for a window, building it if necessary.

    One lock per bundle key: several browser fetches for the same window arrive
    together (model, states, then every blob), and without it they would each
    kick off their own build of the same thing.
    """
    dest = _bundle_path(state, scene_id, window_key, opts_key)
    access = _access_or_404(state, scene_id)
    window = await run_in_threadpool(access.window, window_key, state.max_frames, state.max_seconds)
    data = access.probe.fingerprint
    if _bundle_is_current(dest, window, data, state.meta_memo):
        return dest

    async with state.lock(f"{scene_id}/{window_key}/{opts_key}"):
        if _bundle_is_current(dest, window, data, state.meta_memo):
            return dest
        opts = _options_from_key(opts_key)

        def work() -> Path:
            started = time.time()
            scene, meta = build_scene(
                state.root, scene_id, window_key, opts, max_frames=state.max_frames, max_seconds=state.max_seconds
            )
            doc = scene_document(scene)
            info = write_static_bundle(doc, dest)
            meta["build"]["seconds"] = round(time.time() - started, 2)
            meta["build"]["bundle"] = info
            meta["build"]["data"] = data
            (dest / "meta.json").write_text(json.dumps(meta))
            _evict_bundles(state.bundle_dir, dest, state.cache_max_bytes)
            logger.info(
                "built %s/%s in %.1fs (%d blobs, %.1f MB)",
                scene_id,
                window_key,
                time.time() - started,
                info["blobs"],
                info["bytes"] / 1e6,
            )
            return dest

        try:
            return await run_in_threadpool(work)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - logged in full, named to the browser
            # The traceback goes to the log; the browser gets the exception's
            # type, not its text, which tends to carry file paths.
            logger.exception("build failed for %s/%s", scene_id, window_key)
            raise HTTPException(500, f"build failed ({type(exc).__name__}); see the viewer's log") from exc


def _evict_bundles(bundle_dir: Path, keep: Path, max_bytes: float) -> None:
    """Delete the oldest built windows until the cache is under ``max_bytes``.

    Every distinct set of build options is its own bundle (up to ~75 MB), so
    without a ceiling the cache only grows. Oldest by build time; ``keep``, the
    one just built, never goes.
    """
    bundles = [d for d in bundle_dir.glob("*/*/*") if d.is_dir()]
    sizes = {d: sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) for d in bundles}
    total = sum(sizes.values())
    for d in sorted(bundles, key=lambda d: d.stat().st_mtime):
        if total <= max_bytes:
            break
        if d == keep:
            continue
        shutil.rmtree(d, ignore_errors=True)
        total -= sizes[d]
        logger.info("cache over %.1f GB: removed %s", max_bytes / 1e9, d.relative_to(bundle_dir))


async def _prefetch(state: ViewerState, scene_id: str, window_key: str, opts_key: str) -> None:
    """Build a window nobody has asked for yet, quietly."""
    try:
        await _ensure_bundle(state, scene_id, window_key, opts_key)
    except Exception as exc:  # noqa: BLE001 - the request that wants it will report properly
        logger.debug("prefetch of %s/%s skipped: %s", scene_id, window_key, exc)


async def _npz_png(access: SceneAccess, index: int, kind: str, load, renderer, size: int) -> Response:
    """One frame's ``kind`` file as a PNG tile: ``renderer(load(path), size=size)``,
    absent or unreadable when ``load`` gives None."""
    path = access.frame_path(kind, index)

    def work():
        data = load(path)
        return None if data is None else renderer(data, size=size)

    data = await run_in_threadpool(work)
    if data is None:
        return _absent_or_unreadable(path, f"{kind} for frame {index}")
    return Response(data, media_type="image/png")


def _npz_key(key: str):
    """A loader for ``_npz_png`` reading one array out of an ``.npz``."""
    return lambda path: load_npz_key(path, key)


def _render_driven(sidecar: dict[str, Any], size: int) -> bytes:
    """The driven-similarity tile: three keys of one small npz, one image."""
    return render.render_driven_similarity(sidecar["driven"], sidecar["max_sim"], sidecar["threshold"], size=size)


def _frame_grid(access: SceneAccess, index: int) -> dict[str, Any] | None:
    """Every BEV layer of one frame, flattened, for the probe on a zoomed tile.

    The layers keep the files' own ``[row=x forward, col=y left]`` layout,
    row-major; the page flips them the way the tiles are drawn. A layer whose
    shape disagrees with the height map's is left out rather than misread, and
    NaN goes out as null (JSON has no NaN). None when the frame has no layer.
    """
    layers: dict[str, np.ndarray] = {}
    out: dict[str, Any] = {"res": HM_RES, "x0": HM_EXTENT[0], "validity_labels": VALIDITY_LABELS}
    hm = (
        load_npz_key(access.frame_path("height_maps", index), "height_map") if access.probe.has("height_maps") else None
    )
    if hm is not None:
        layers["height"] = hm
        finite = hm[np.isfinite(hm)]
        if finite.size:
            # The same 2nd-98th percentile span render.colorize stretches the tile over.
            out["height_range"] = [round(float(np.percentile(finite, q)), 3) for q in (2, 98)]
    if access.probe.has("height_masks"):
        validity = load_npz_key(access.frame_path("height_masks", index), "validity")
        if validity is not None:
            layers["validity"] = validity
    if access.probe.has("driven_similarity"):
        sidecar = read_driven_similarity(access.frame_path("driven_similarity", index))
        if sidecar is not None:
            out["threshold"] = sidecar.pop("threshold")
            layers.update(sidecar)
    if not layers:
        return None
    shape = next(iter(layers.values())).shape
    if len(shape) != 2:
        return None
    out["rows"], out["cols"] = shape
    out["layers"] = {}
    for name, arr in layers.items():
        if arr.shape != shape:
            continue
        vals = np.round(np.asarray(arr, dtype=np.float64), 3)
        flat = vals.ravel().astype(object)
        flat[~np.isfinite(vals.ravel())] = None
        out["layers"][name] = flat.tolist()
    return out


def _overlay_possible(access: SceneAccess) -> bool:
    """Can this scene's clouds be drawn on its images at all?

    Needs ``K`` and ``T_cam_to_base.txt`` specifically: ``access.T_cam`` falls
    back to a v1 root's ``T_cam_to_lidar.txt``, which is the camera in the raw
    lidar frame, not the base frame the clouds are in.
    """
    p = access.probe
    if not (p.has("images") and p.has("clouds") and p.has("camera_intrinsics") and p.has("T_cam_to_base")):
        return False
    return access.K is not None and access.T_cam is not None


def _static_fingerprint(root: Path) -> str:
    """A short hash of every file's path, size and mtime under ``root``.

    Read once at startup: it changes whenever a file does, which is all a
    cache-busting URL needs, without hashing the contents.
    """
    h = hashlib.sha1()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            st = path.stat()
            h.update(f"{path.relative_to(root)}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
    return h.hexdigest()[:10]


def _safe_render(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except (OSError, ValueError) as exc:
        logger.debug("render failed: %s", exc)
        return None


@functools.cache
def _unreadable_tile() -> bytes:
    """A red square that says so, drawn where a file exists but cannot be read."""
    tile = np.full((96, 96, 3), (40, 40, 190), dtype=np.uint8)  # BGR
    cv2.putText(tile, "unreadable", (6, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.imencode(".png", tile)[1].tobytes()


def _absent_or_unreadable(path: Path | None, what: str) -> Response:
    """The answer when nothing could be drawn: 404 for a file the scene does
    not have, 422 -- with a red tile, which the page shows as it is -- for one
    it has but that will not load. The two looked the same, and the second is
    exactly what reviewing a dataset is for."""
    if path is not None and path.is_file():
        logger.warning("%s: %s exists but cannot be read", what, path)
        return Response(_unreadable_tile(), status_code=422, media_type="image/png")
    raise HTTPException(404, f"no {what}")


#: Fixes drawn for the *context* track -- the whole scene behind a highlighted
#: stretch. Thinner than the highlight: it is there for shape, not for detail.
GPS_CONTEXT_POINTS = 800


def _gps_payload(
    access: SceneAccess,
    segment_index: int | None,
    frame_index: int | None,
    max_points: int,
) -> dict[str, Any] | None:
    """What the map draws for one scene: the track, its bounds, a marker, the context.

    Read straight from ``gps.npy`` and thinned here rather than in the browser,
    so a 16-minute scene ships tens of kilobytes instead of ten thousand pairs
    of doubles. Deliberately its own endpoint and never folded into a window
    bundle: bundles are cached per window under a content stamp, and a track
    belongs to the scene, not to a view of it.
    """
    gps = access.gps
    if not geo.is_track(gps):
        return None

    track = gps
    if segment_index is not None:
        seg = next((s for s in access.segments if s.index == segment_index), None)
        if seg is None:
            raise HTTPException(404, f"scene {access.scene_id} has no segment {segment_index}")
        track = geo.slice_time(gps, seg.start_ts, seg.end_ts)

    marker = None
    if frame_index is not None:
        ts = access.frame_ts
        if ts is not None and 0 <= frame_index < len(ts):
            marker = geo.locate(gps, float(ts[frame_index]))

    return {
        "points": geo.points(geo.decimate(track, max_points)),
        "bounds": geo.bounds(track),
        "marker": marker,
        # The rest of the scene, faint, behind a highlighted stretch -- so a
        # segment is seen where in the outing it happened, not just on its own.
        "context": (
            geo.points(geo.decimate(gps, GPS_CONTEXT_POINTS)) if track is not gps and len(gps) > len(track) else None
        ),
    }


#: Thumbnails per page of the frame browser. One HTTP request each, lazily,
#: so this is a bound on concurrent image renders rather than on bytes read.
FRAMES_PER_PAGE = 100


def _frame_indices(access: SceneAccess, frame_range: tuple[int, int] | None = None) -> list[int]:
    """Every frame index this scene can show, in order.

    Normally just ``range(n_frames)``. Only when the per-frame directory is
    known to be incomplete is it listed, so the common case costs nothing.

    A present-but-empty directory says nothing about which frames exist, so it
    is skipped rather than accepted as the answer -- otherwise a scene whose
    ``images/`` had been created but not yet filled fell back to
    ``range(n_frames)`` and offered a page of tiles that all 404.

    ``frame_range`` narrows the result to a half-open range of frames, which is
    how the segment page shows a strip of its own frames rather than the
    scene's.
    """
    probe = access.probe
    indices = list(range(probe.n_frames))
    for kind in ("images", "clouds"):
        if not probe.has(kind) or not probe.counts.get(kind):
            continue
        if probe.counts[kind] != probe.n_frames:
            indices = sorted(access.available_frames(kind))
        break
    if frame_range is not None:
        lo, hi = frame_range
        indices = [i for i in indices if lo <= i < hi]
    return indices


def _page_of(indices: list[int], index: int) -> int:
    """Which page of a frame strip over ``indices`` holds ``index`` (0 when empty).

    Not ``index // FRAMES_PER_PAGE``: on a scene with a sparse per-frame
    directory the strip pages through the frames that *exist*, so a frame's
    page is its position in that list, not its own number.
    """
    if not indices:
        return 0
    # The list is sorted, so the position is a search -- and for the usual
    # dense scene (0..n-1) that search lands on `index` itself.
    position = min(bisect.bisect_left(indices, index), len(indices) - 1)
    return position // FRAMES_PER_PAGE


def _frame_page(access: SceneAccess, page: int, frame_range: tuple[int, int] | None = None) -> dict[str, Any]:
    """One page of the frame browser, over the scene or over one segment."""
    indices = _frame_indices(access, frame_range)
    n_pages = max(1, -(-len(indices) // FRAMES_PER_PAGE))
    page = min(max(page, 0), n_pages - 1)
    start = page * FRAMES_PER_PAGE
    shown = indices[start : start + FRAMES_PER_PAGE]
    ts = access.frame_ts
    t0 = float(ts[0]) if ts is not None and len(ts) else None
    return {
        "page": page,
        "n_pages": n_pages,
        "per_page": FRAMES_PER_PAGE,
        "total": len(indices),
        "first": shown[0] if shown else None,
        "last": shown[-1] if shown else None,
        "entries": [
            {
                "index": i,
                "t": round(float(ts[i]) - t0, 2) if ts is not None and t0 is not None and i < len(ts) else None,
            }
            for i in shown
        ],
        # A window of pages around the current one, so a 200-page scene does
        # not render 200 links.
        "page_links": sorted({0, n_pages - 1, *range(max(0, page - 3), min(n_pages, page + 4))}),
    }


def _frame_details(
    access: SceneAccess, index: int, max_frames: int, max_seconds: float = DEFAULT_MAX_SECONDS
) -> dict[str, Any]:
    """Everything the per-frame page states about one frame.

    Cheap by construction: two timestamp lookups in arrays already in memory,
    one ``.npy`` header read for the cloud's point count, and ``is_file`` for
    the per-frame artefacts. No cloud, image or height map is decoded here --
    those are separate requests the browser makes for itself.
    """
    probe = access.probe
    details: dict[str, Any] = {
        "index": index,
        "n_frames": probe.n_frames,
        # Which page of the frame browser this frame sits on, so "back to the
        # frame strip" lands where the frame actually is even on a scene whose
        # per-frame directory is sparse.
        "strip_page": _page_of(_frame_indices(access), index),
    }

    ts = access.frame_ts
    if ts is not None and 0 <= index < len(ts):
        details["timestamp"] = float(ts[index])
        details["t_since_start"] = float(ts[index] - ts[0])
    pcd = access.pcd_ts
    if pcd is not None and 0 <= index < len(pcd):
        details["lidar_timestamp"] = float(pcd[index])
        if "timestamp" in details:
            details["camera_lidar_offset_ms"] = round((details["timestamp"] - float(pcd[index])) * 1000, 2)

    present: dict[str, bool] = {}
    for kind in ("images", "clouds", "height_maps", "height_masks", "driven_similarity"):
        present[kind] = probe.has(kind) and access.frame_path(kind, index).is_file()
    details["present"] = present
    # The cloud-on-image check needs both per-frame files and the calibration.
    details["overlay"] = present["images"] and present["clouds"] and _overlay_possible(access)

    if present["driven_similarity"]:
        # The one per-frame array this page does decode: a 128x128 bool and a
        # float16 of the same size, a few kilobytes, and the caption under the
        # tile has to say how many cells the wheels touched and how many more
        # the trust gate admits.
        sidecar = read_driven_similarity(access.frame_path("driven_similarity", index))
        if sidecar is not None:
            driven, threshold = sidecar["driven"], sidecar["threshold"]
            details["driven"] = {
                "n_driven": int(driven.sum()),
                "n_trusted": int((driven | (sidecar["max_sim"] > threshold)).sum()),
                "threshold": threshold,
            }

    if present["clouds"]:
        shape = _npy_shape(access.frame_path("clouds", index))
        if shape:
            details["cloud_points"] = int(shape[0])
            details["cloud_columns"] = int(shape[1]) if len(shape) > 1 else 1

    ga = access.gravity_align
    if ga is not None and 0 <= index < len(ga):
        roll, pitch = roll_pitch_from_gravity_align(np.asarray(ga[index])[None])
        details["roll_deg"] = round(float(np.degrees(roll[0])), 2)
        details["pitch_deg"] = round(float(np.degrees(pitch[0])), 2)

    seg = access.segment_of_frame(index)
    if seg is not None:
        # The segment page's strip covers the frames of the segment's windows.
        wins = segment_windows(access, seg, max_frames, max_seconds)
        strip = _frame_indices(access, (wins[0].frame_start, wins[-1].frame_end)) if wins else []
        details["segment"] = {
            "index": seg.index,
            "reason": seg.reason,
            "duration": round(seg.duration, 2),
            "n_poses": seg.n_poses,
            # Clamped: a frame within the boundary slack sits a few ms before the
            # segment's first pose, and "-0.0 s into the segment" reads as a bug.
            "t_into_segment": max(
                0.0,
                round(float(details.get("timestamp", seg.start_ts)) - seg.start_ts, 2),
            ),
            "strip_page": _page_of(strip, index),
        }
    else:
        details["segment"] = None

    if "timestamp" in details:
        details["motion"] = _frame_motion(access, index, details["timestamp"], seg)

    best = _window_of_frame(access, index, max_frames, max_seconds)
    if best is not None:
        details["window"] = {
            "key": best.key,
            "label": best.label,
            "kind": best.kind,
            "pose_source": best.pose_source,
            "offset": index - best.frame_start,
            "n_frames": best.n_frames,
        }
    else:
        details["window"] = None
    return details


#: How much of the recording either side of a frame its signal charts cover (s).
FRAME_SIGNAL_SECONDS = 15.0

#: The span a frame's speed and turn rate are differenced over, either side (s):
#: a few 10 Hz poses, so one noisy pose does not make the number.
FRAME_MOTION_HALF = 0.25


def _frame_signals(access: SceneAccess, t: float | None) -> dict[str, Any] | None:
    """The scene's signal charts over the seconds around one frame, and where it sits.

    ``now`` is the frame's position across the charts in percent; the span is
    clipped to the recording, so near either end the frame is off centre.
    """
    ts = access.frame_ts
    if t is None or ts is None or len(ts) < 2:
        return None
    t0, t1 = max(float(ts[0]), t - FRAME_SIGNAL_SECONDS), min(float(ts[-1]), t + FRAME_SIGNAL_SECONDS)
    # Times read from the scene's start, as "Into the scene" does, not from the charts' edge.
    signals = signal_charts(access, t0, t1, offset=t0 - float(ts[0]))
    if signals is not None:
        signals["now"] = (t - t0) / (t1 - t0) * 100
    return signals


def _frame_motion(access: SceneAccess, index: int, t: float, seg) -> dict[str, Any]:
    """How the robot was moving at one frame, and what the GNSS said there.

    Ground-truth speed and turn rate are differenced across the poses within
    :data:`FRAME_MOTION_HALF` of the frame and never outside its own segment
    (two segments share no anchor); the robot's own odometry across the frames
    either side. Each number is left out where its source has nothing there.
    """
    out: dict[str, Any] = {}
    odom = access.odometry
    if odom is not None and seg is not None:
        ots, poses = odom
        take = (ots >= max(seg.start_ts, t - FRAME_MOTION_HALF)) & (ots <= min(seg.end_ts, t + FRAME_MOTION_HALF))
        if take.sum() >= 2:
            order = np.argsort(ots[take], kind="stable")
            st, sp = ots[take][order], poses[take][order]
            dt = float(st[-1] - st[0])
            speed = float(np.linalg.norm(sp[-1, :2, 3] - sp[0, :2, 3])) / dt if dt > 0.05 else None
            if speed is not None and speed <= MAX_GT_SPEED:
                yaw = np.arctan2(sp[[0, -1], 1, 0], sp[[0, -1], 0, 0])
                dyaw = (yaw[1] - yaw[0] + np.pi) % (2 * np.pi) - np.pi
                out["gt_speed"] = round(speed, 2)
                out["gt_yaw_rate"] = round(float(np.degrees(dyaw)) / dt, 1)
    tf, ts = access.transforms, access.frame_ts
    if tf is not None and ts is not None and np.ndim(tf) == 3:
        lo, hi = max(0, index - 1), min(len(tf), len(ts)) - 1
        hi = min(hi, index + 1)
        if hi > lo and ts[hi] - ts[lo] > 1e-3:
            out["odom_speed"] = round(float(np.linalg.norm(tf[hi][:2, 3] - tf[lo][:2, 3]) / (ts[hi] - ts[lo])), 2)
    gps = access.gps
    if geo.is_track(gps) and gps.shape[1] > geo.STATUS:
        k = int(np.argmin(np.abs(gps[:, geo.T] - t)))
        if abs(gps[k, geo.T] - t) <= geo.MAX_INTERP_GAP:
            status = int(gps[k, geo.STATUS])
            fix = {"status": geo.FIX_LABELS.get(status, str(status)), "alt": round(float(gps[k, geo.ALT]), 1)}
            if gps.shape[1] >= geo.COV.stop:
                horiz = float(gps[k, geo.COV.start] + gps[k, geo.COV.start + 1])
                if horiz > 0:
                    fix["sigma"] = round(float(np.sqrt(horiz)), 2)
            out["gnss"] = fix
    return out


def _window_of_frame(access: SceneAccess, index: int, max_frames: int, max_seconds: float) -> Window | None:
    """The enumerated window a frame sits in: a ground-truth one if there is
    one, otherwise the whole-scene fallback. This is the "show me the motion
    around this frame" link, as opposed to the static single-frame view."""
    containing = [w for w in access.windows(max_frames, max_seconds) if w.frame_start <= index < w.frame_end]
    return next((w for w in containing if w.kind == "segment"), None) or (containing[0] if containing else None)


def _notes_index(state: ViewerState, notes_store: NoteStore, topic: str | None) -> dict[str, Any]:
    """Every note in the root, grouped by topic, for the ``/notes`` page.

    Each note carries the links the page draws: its scene, its first frame and
    the window holding that frame opened on it. A notes file for a scene the
    root does not have is listed and said to be orphaned rather than dropped --
    the note is a person's work, and a scene renumbered out from under it is
    exactly the case worth seeing.
    """
    by_scene = notes_store.list_all()
    known = set(state.root.scene_ids(refresh=True))
    rows: list[dict[str, Any]] = []
    for scene_id, notes in by_scene.items():
        access = state.access(scene_id) if scene_id in known else None
        for note in notes:
            window = _window_of_frame(access, note.frame_start, state.max_frames, state.max_seconds) if access else None
            rows.append(
                {
                    **note.to_json(),
                    "scene": scene_id,
                    "orphan": access is None,
                    "view_url": f"/view/{scene_id}/{window.key}?frame={note.frame_start}" if window else None,
                }
            )
    # The UI's own topics in their own order, then anything hand-written after.
    offered = notes_store.status()["topics"]
    topics = sorted({r["topic"] for r in rows}, key=lambda t: (offered.index(t) if t in offered else len(offered), t))
    if topic:
        rows = [r for r in rows if r["topic"] == topic]
    groups = [{"topic": t, "notes": [r for r in rows if r["topic"] == t]} for t in topics]
    return {
        "topic": topic,
        "topics": topics,
        "groups": [g for g in groups if g["notes"]],
        "n_notes": len(rows),
        "n_scenes": len({r["scene"] for r in rows}),
        "status": notes_store.status(),
    }
