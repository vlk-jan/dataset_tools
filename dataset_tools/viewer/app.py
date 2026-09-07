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
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from . import render
from .build import META_VERSION, BuildOptions, SceneAccess, build_scene
from .index_page import THUMB_SLOTS, build_index, thumb_index
from .notes import NoteStore, NotesUnavailable
from .probe import ARTEFACTS, BUNDLE_DIR_NAME, DatasetRoot
from .scene_page import build_scene_page
from .windows import describe_reason, frame_window_key, neighbour_windows
from .wire import render_viewer_page, scene_document, write_static_bundle


class _RevalidatingStatic(StaticFiles):
    """StaticFiles that forces the browser to revalidate every asset.

    Only SimView's entry point carries a cache-bust (``main.js?v=<ver>``); the
    ES modules it imports do not. With no ``Cache-Control``, a browser keeps the
    OLD modules after an upgrade -- the server reports the new version and our
    version-gated workarounds stand down while the tab still runs the buggy JS
    (blank Analysis plots after 4.2.1). ``no-cache`` keeps the file cached but
    forces a conditional request: a cheap 304, or the upgrade picked up at once.
    """

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


class _ImmutableStatic(StaticFiles):
    """StaticFiles for URLs that already carry a version in their path.

    When the URL is unique per version (``/simview-static/<ver>/...``) the file
    at it never changes, so it is safe -- and best -- to let the browser cache
    it forever. An upgrade changes the version segment, so the new build is a
    new URL the browser has never seen; there is no way to serve a stale module
    the way an unversioned path invites. This is the belt to the no-cache
    braces: it is what actually guarantees the JS the browser runs matches the
    version the server reports.
    """

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response


logger = logging.getLogger("dataset_tools.viewer.app")

HERE = Path(__file__).parent
#: Per-frame renders of files that never change once extracted.
DAY_CACHE = {"Cache-Control": "public, max-age=86400"}
#: The shapes a window key (``windows.py``) and a build-options key
#: (``BuildOptions.cache_key``) can take. Both are URL segments that end up in
#: a cache *path*, so anything else is refused before a path is formed.
WINDOW_KEY_RE = re.compile(r"^(seg\d+(\.\d+)?|raw(\.\d+)?|frame\d+)$")
OPTS_KEY_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
#: Stored atlas thumbnails are curated files that only change when someone
#: regenerates them by hand, and the index asks for 129 of them at once.
CURATED = {"Cache-Control": "public, max-age=31536000, immutable"}


def _simview_static_dir() -> Path:
    from importlib.resources import files

    return Path(str(files("simview") / "static"))


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


def _simview_version() -> str:
    try:
        import simview

        return getattr(simview, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        return "unavailable"


class ViewerState:
    """Shared, per-process state: the dataset root, caches and build locks."""

    def __init__(self, root: DatasetRoot, max_frames: int, cache_scenes: int = 4) -> None:
        self.root = root
        self.max_frames = max_frames
        self._access: dict[str, SceneAccess] = {}
        self._access_order: list[str] = []
        self._cache_scenes = cache_scenes
        self._locks: dict[str, asyncio.Lock] = {}
        self.bundle_dir = root.cache_dir / BUNDLE_DIR_NAME
        self.bundle_dir.mkdir(parents=True, exist_ok=True)

    def access(self, scene_id: str) -> SceneAccess:
        """A cached per-scene handle. Bounded so a long session browsing many
        scenes does not accumulate every scene's arrays in memory."""
        if scene_id in self._access:
            self._access_order.remove(scene_id)
            self._access_order.append(scene_id)
            return self._access[scene_id]
        access = SceneAccess(self.root, scene_id)
        self._access[scene_id] = access
        self._access_order.append(scene_id)
        while len(self._access_order) > self._cache_scenes:
            self._access.pop(self._access_order.pop(0), None)
        return access

    def lock(self, key: str) -> asyncio.Lock:
        return self._locks.setdefault(key, asyncio.Lock())


def create_app(
    root_path: str | Path,
    max_frames: int = 1500,
    cache_dir: str | Path | None = None,
    notes_dir: str | Path | None = None,
    notes: bool = True,
    author: str = "",
    clear_cache: bool = False,
) -> FastAPI:
    root = DatasetRoot(root_path, cache_dir=Path(cache_dir) if cache_dir else None, clear_cache=clear_cache)
    state = ViewerState(root, max_frames=max_frames)
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
    app.state.viewer = state

    templates = Jinja2Templates(directory=str(HERE / "templates"))
    templates.env.globals["reason_text"] = describe_reason
    app.mount(
        "/viewer-static",
        _RevalidatingStatic(directory=str(HERE / "static")),
        name="viewer_static",
    )

    simview_static = _simview_static_dir()
    simview_ok = simview_static.is_dir()
    simview_ver = _simview_version()
    if simview_ok:
        # Version-stamped path first (most specific): the entry point and every
        # ES module it imports resolve under it, so a package upgrade changes
        # every module's URL at once and the browser cannot serve a stale mix
        # of old and new modules. This is what render_viewer_page points at.
        app.mount(
            f"/simview-static/{simview_ver}",
            _ImmutableStatic(directory=str(simview_static)),
            name="simview_static_versioned",
        )
        # Unversioned fallback (revalidating), still needed for SimView's own
        # document-relative asset paths (config.js: "static/textures/...").
        app.mount(
            "/simview-static",
            _RevalidatingStatic(directory=str(simview_static)),
            name="simview_static",
        )
        app.mount(
            "/static",
            _RevalidatingStatic(directory=str(simview_static)),
            name="simview_static_root",
        )
    else:  # pragma: no cover - only when simview is missing/broken
        logger.error(
            "simview static assets not found at %s; the 3D view will not work",
            simview_static,
        )
    threading.Thread(target=_warm_build_imports, name="warm-build-imports", daemon=True).start()

    # ------------------------------------------------------------------ pages

    # Cache-immunity stamp, folded into every /api/data URL: bundle content
    # can change under an unchanged URL when the viewer code changes, and a
    # browser that cached the old response (with any freshness lifetime) would
    # mix stale trajectory blobs with fresh model.json. A new stamp on every
    # code change makes previously cached entries unaddressable.
    import hashlib

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

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        def work():
            return build_index(root, root.probe_all())

        page = await run_in_threadpool(work)
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "root": root,
                "artefacts": ARTEFACTS,
                "manifest": root.manifest(),
                "simview_version": _simview_version(),
                "simview_ok": simview_ok,
                **page,
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
        windows = await run_in_threadpool(access.windows, state.max_frames)
        # `_scene_details` is deliberately NOT computed here: the page renders
        # none of it (the four cards below carry the same facts, read through a
        # memory map instead), and asking for it loaded joint_states.npy,
        # cmd_vel.npy and joint_setpoints.npy whole on every page view. It
        # remains what /api/scene/{id} returns, which is where it is read.
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
                "scene_id": scene_id,
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
            },
        )

    @app.get("/scene/{scene_id}/jump")
    async def frame_jump(scene_id: str, index: int = Query(0, ge=0)):
        """The frames tab's jump box: clamp to the scene and go to that frame."""
        from fastapi.responses import RedirectResponse

        access = _access_or_404(state, scene_id)
        last = max(0, access.probe.n_frames - 1)
        return RedirectResponse(f"/scene/{scene_id}/frame/{min(index, last)}", status_code=303)

    @app.get("/scene/{scene_id}/frame/{index}", response_class=HTMLResponse)
    async def frame_page(request: Request, scene_id: str, index: int):
        access = _access_or_404(state, scene_id)
        n_frames = access.probe.n_frames
        if index < 0 or (n_frames and index >= n_frames):
            raise HTTPException(404, f"scene {scene_id} has no frame {index}")
        details = await run_in_threadpool(_frame_details, access, index, state.max_frames)
        # A frame past the end of data_ts.txt has no clock to sit on, so there
        # is no single-frame window to build for it; offering the button anyway
        # led straight to a 500.
        single = await run_in_threadpool(access.window, frame_window_key(index), state.max_frames)
        return templates.TemplateResponse(
            request=request,
            name="frame.html",
            context={
                "root": root,
                "scene_id": scene_id,
                "index": index,
                "probe": access.probe,
                "details": details,
                "simview_ok": simview_ok,
                "frame_window": single.key if single is not None else None,
            },
        )

    @app.get("/view/{scene_id}/{window_key}", response_class=HTMLResponse)
    async def viewer_page(request: Request, scene_id: str, window_key: str):
        access = _access_or_404(state, scene_id)
        window = await run_in_threadpool(access.window, window_key, state.max_frames)
        if window is None:
            raise HTTPException(404, f"scene {scene_id} has no window {window_key!r}")
        if not simview_ok:
            raise HTTPException(500, "simview's static assets are not installed")

        opts = _options_from_query(request.query_params)
        bundle_key = _bundle_key(scene_id, window_key, opts)
        # The windows either side of this one, so the panel can page through a
        # scene's segments without going back to the scene page each time.
        windows = await run_in_threadpool(access.windows, state.max_frames)
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
            data_base=f"/api/data/{bundle_key}/{stamp}",
            meta_url=f"/api/data/{bundle_key}/{stamp}/meta.json",
            status_url=f"/api/data/{bundle_key}/{stamp}/status",
            frame_base=f"/api/scene/{scene_id}/frame",
            scene_url=f"/scene/{scene_id}",
        )
        config = json.dumps(
            {
                "sceneId": scene_id,
                "windowKey": window_key,
                "dataBase": f"/api/data/{bundle_key}/{stamp}",
                "frameBase": f"/api/scene/{scene_id}/frame",
                "statusUrl": f"/api/data/{bundle_key}/{stamp}/status",
                "metaUrl": f"/api/data/{bundle_key}/{stamp}/meta.json",
                "initialFrame": initial_frame,
                "sceneUrl": f"/scene/{scene_id}",
                "notesUrl": f"/api/scene/{scene_id}/notes",
                "notes": notes_store.status(),
                "frameStart": window.frame_start,
                "frameEnd": window.frame_end - 1,
                # Which SimView the page is running against. The overlay gates
                # its workarounds on this so that upgrading the package turns
                # them off by itself -- a stopgap left running over a fixed
                # SimView would override the real fix (see simview_patch.js).
                "simviewVersion": _simview_version(),
            }
        )
        head = (
            '<link rel="stylesheet" href="/viewer-static/overlay.css">'
            f"<script>window.__dtViewer = {config};</script>"
        )
        body = overlay + '<script type="module" src="/viewer-static/overlay.js"></script>'
        page = await run_in_threadpool(
            render_viewer_page,
            f"/simview-static/{_simview_version()}",
            f"/api/data/{bundle_key}/{stamp}",
            _simview_version(),
            body,
            head,
        )
        # Defer SimView's own entry point: it fetches the scene the instant it
        # runs, and the bundle may still be building. The overlay controller
        # re-inserts this exact script once the build reports ready.
        page = _defer_main_script(page)
        return HTMLResponse(page)

    # ------------------------------------------------------------------- api

    @app.get("/api/scenes")
    async def api_scenes():
        probes = await run_in_threadpool(root.probe_all)
        return {
            "root": str(root.path),
            "scenes": [p.to_json() for p in probes],
            "artefacts": [a.__dict__ for a in ARTEFACTS],
        }

    @app.get("/api/scene/{scene_id}")
    async def api_scene(scene_id: str):
        access = _access_or_404(state, scene_id)
        windows = await run_in_threadpool(access.windows, state.max_frames)
        details = await run_in_threadpool(_scene_details, access)
        return {
            "scene": scene_id,
            "probe": access.probe.to_json(),
            "windows": [w.to_json() for w in windows],
            "details": details,
        }

    @app.get("/api/scene/{scene_id}/frame/{index}/info")
    async def api_frame_info(scene_id: str, index: int):
        access = _access_or_404(state, scene_id)
        return await run_in_threadpool(_frame_info, access, index)

    @app.get("/api/scene/{scene_id}/frame/{index}/image.jpg")
    async def api_frame_image(scene_id: str, index: int, w: int = Query(640, ge=64, le=1920)):
        access = _access_or_404(state, scene_id)
        path = access.frame_path("images", index)
        data = await run_in_threadpool(_safe_render, render.render_camera_image, path, w)
        if data is None:
            raise HTTPException(404, f"no image for frame {index}")
        return Response(data, media_type="image/jpeg", headers=DAY_CACHE)

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
        stored = root.atlas_thumb(scene_id, slot)
        if stored is not None:
            return FileResponse(stored, media_type="image/jpeg", headers=CURATED)

        if slot not in THUMB_SLOTS:
            raise HTTPException(404, f"no thumbnail slot {slot!r}")
        n_frames = access.probe.n_frames
        if not n_frames or not access.probe.has("images"):
            raise HTTPException(404, f"scene {scene_id} has no images to thumbnail")
        # The same resolver the index and the scene page use, so a curated index
        # with no file behind it still renders the frame it names.
        index = thumb_index(root.atlas_scene(scene_id), slot, n_frames)

        def work():
            path = access.frame_path("images", index)
            if not path.is_file():
                # Sparse images/: fall back to the nearest index that exists.
                available = sorted(access.available_frames("images"))
                if not available:
                    return None
                path = access.frame_path("images", min(available, key=lambda i: abs(i - index)))
            return _safe_render(render.render_camera_image, path, 384)

        data = await run_in_threadpool(work)
        if data is None:
            raise HTTPException(404, f"no image to thumbnail for scene {scene_id}")
        return Response(data, media_type="image/jpeg", headers=DAY_CACHE)

    @app.get("/api/scene/{scene_id}/frame/{index}/heightmap.png")
    async def api_frame_heightmap(scene_id: str, index: int):
        access = _access_or_404(state, scene_id)
        return await _npz_png(access, index, "height_maps", "height_map", render.render_height_map)

    @app.get("/api/scene/{scene_id}/frame/{index}/validity.png")
    async def api_frame_validity(scene_id: str, index: int):
        access = _access_or_404(state, scene_id)
        return await _npz_png(access, index, "height_masks", "validity", render.render_validity)

    @app.get("/api/scene/{scene_id}/frame/{index}/cloud.png")
    async def api_frame_cloud(scene_id: str, index: int, color: str = "z"):
        access = _access_or_404(state, scene_id)
        from .loaders import load_cloud

        def work():
            cloud = load_cloud(access.frame_path("clouds", index))
            if cloud is None:
                return None
            return render.render_cloud_bev(cloud, color_by=color)

        data = await run_in_threadpool(work)
        if data is None:
            raise HTTPException(404, f"no cloud for frame {index}")
        return Response(data, media_type="image/png", headers=DAY_CACHE)

    # ------------------------------------------------------------------ notes

    # Deliberately NOT under /api/data/: a bundle URL carries a content stamp
    # and its meta.json is cached on disk per window, so a note folded in there
    # would either be invisible until the next code change or force every
    # window it touches to rebuild. Notes are their own resource, fetched
    # uncached, and nothing about them can invalidate a built window.

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
        )
        return note.to_json()

    @app.patch("/api/scene/{scene_id}/notes/{note_id}")
    async def api_edit_note(scene_id: str, note_id: str, payload: dict = Body(...)):
        access = _access_or_404(state, scene_id)
        # Either end may move on its own -- ``update`` leaves out whatever is
        # None -- so this does not go through _frames_from_payload, which is
        # for a whole range being named at once.
        start, end = _optional_frames(payload)
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

    @app.get("/api/data/{scene_id}/{window_key}/{opts_key}/{stamp}/status")
    async def api_status(request: Request, scene_id: str, window_key: str, opts_key: str, stamp: str):
        path = _bundle_path(state, scene_id, window_key, opts_key)
        # The same test _ensure_bundle uses. Reporting "ready" for a bundle
        # written by an older viewer would promise a fetch that in fact blocks
        # on a full rebuild.
        return {"ready": _bundle_is_current(path), "path": str(path)}

    @app.get("/api/data/{scene_id}/{window_key}/{opts_key}/{stamp}/meta.json")
    async def api_meta(request: Request, scene_id: str, window_key: str, opts_key: str, stamp: str):
        bundle = await _ensure_bundle(state, scene_id, window_key, opts_key)
        return FileResponse(bundle / "meta.json", media_type="application/json")

    @app.get("/api/data/{scene_id}/{window_key}/{opts_key}/{stamp}/model.json")
    async def api_model(scene_id: str, window_key: str, opts_key: str, stamp: str):
        bundle = await _ensure_bundle(state, scene_id, window_key, opts_key)
        return FileResponse(bundle / "model.json", media_type="application/json")

    @app.get("/api/data/{scene_id}/{window_key}/{opts_key}/{stamp}/states.json")
    async def api_states(scene_id: str, window_key: str, opts_key: str, stamp: str):
        bundle = await _ensure_bundle(state, scene_id, window_key, opts_key)
        return FileResponse(bundle / "states.json", media_type="application/json")

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


def _frames_from_payload(payload: dict) -> tuple[int, int]:
    """The inclusive frame range a request names.

    Accepts ``{"frames": [a, b]}`` (the stored form) or the two fields spelled
    out, and a bare ``{"frame": n}`` for a note on one frame -- which is what
    the overlay sends when you mark the frame you are paused on.
    """
    frames = payload.get("frames")
    if isinstance(frames, list | tuple) and len(frames) == 2:
        start, end = frames
    elif "frame" in payload:
        start = end = payload["frame"]
    else:
        start = payload.get("frame_start")
        end = payload.get("frame_end", start)
    try:
        return int(start), int(end)
    except (TypeError, ValueError):
        raise HTTPException(400, "a note needs a frame range") from None


def _optional_frames(payload: dict) -> tuple[int | None, int | None]:
    """The ends of a range an edit names, ``None`` for each one it leaves alone."""
    frames = payload.get("frames")
    if isinstance(frames, list | tuple) and len(frames) == 2:
        pair = list(frames)
    else:
        pair = [payload.get("frame_start"), payload.get("frame_end")]
    try:
        return tuple(None if v is None else int(v) for v in pair)  # type: ignore[return-value]
    except (TypeError, ValueError):
        raise HTTPException(400, "frame numbers must be integers") from None


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
    """The on-disk bundle directory for a request, after checking its parts."""
    if not scene_id.isdigit() or not WINDOW_KEY_RE.match(window_key) or not OPTS_KEY_RE.match(opts_key):
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
    import re

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
    if len(parts) != 6:
        return BuildOptions()
    try:
        return BuildOptions(
            max_cloud_frames=int(parts[0]),
            max_cloud_points=int(parts[1]),
            max_hm_frames=int(parts[2]),
            max_terrain_cells=int(parts[3]),
            terrain_res=float(parts[4]),
            cloud_color=parts[5],
        ).sanitized()
    except ValueError:
        return BuildOptions()


def _bundle_is_current(dest: Path) -> bool:
    """Is this cached bundle complete *and* built by today's code?

    The cache is keyed by what was asked for (scene, window, build options),
    none of which changes when the builder does -- so a bundle written before a
    viewer upgrade would keep being served, with whatever fields its
    ``meta.json`` happened to have at the time. The payload carries its own
    version (``build.META_VERSION``); anything older is rebuilt rather than
    served. Rebuilding a window costs a couple of seconds, once.
    """
    if not ((dest / "model.json").is_file() and (dest / "meta.json").is_file()):
        return False
    try:
        meta = json.loads((dest / "meta.json").read_text())
    except (OSError, ValueError):
        return False
    return meta.get("meta_version") == META_VERSION


async def _ensure_bundle(state: ViewerState, scene_id: str, window_key: str, opts_key: str) -> Path:
    """Return the on-disk bundle for a window, building it if necessary.

    One lock per bundle key: several browser fetches for the same window arrive
    together (model, states, then every blob), and without it they would each
    kick off their own build of the same thing.
    """
    dest = _bundle_path(state, scene_id, window_key, opts_key)
    if _bundle_is_current(dest):
        return dest

    async with state.lock(f"{scene_id}/{window_key}/{opts_key}"):
        if _bundle_is_current(dest):
            return dest
        opts = _options_from_key(opts_key)

        def work() -> Path:
            started = time.time()
            scene, meta = build_scene(state.root, scene_id, window_key, opts, max_frames=state.max_frames)
            doc = scene_document(scene)
            info = write_static_bundle(doc, dest)
            meta["build"]["seconds"] = round(time.time() - started, 2)
            meta["build"]["bundle"] = info
            (dest / "meta.json").write_text(json.dumps(meta))
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
        except Exception as exc:  # noqa: BLE001 - surfaced to the browser
            logger.exception("build failed for %s/%s", scene_id, window_key)
            raise HTTPException(500, f"build failed: {type(exc).__name__}: {exc}") from exc


async def _npz_png(access: SceneAccess, index: int, kind: str, key: str, renderer) -> Response:
    from .loaders import load_npz_key

    def work():
        arr = load_npz_key(access.frame_path(kind, index), key)
        return None if arr is None else renderer(arr)

    data = await run_in_threadpool(work)
    if data is None:
        raise HTTPException(404, f"no {kind} for frame {index}")
    return Response(data, media_type="image/png", headers=DAY_CACHE)


def _safe_render(fn, *args):
    try:
        return fn(*args)
    except (OSError, ValueError) as exc:
        logger.debug("render failed: %s", exc)
        return None


def _scene_details(access: SceneAccess) -> dict[str, Any]:
    """Small, cheap, human-facing extras for the scene page."""
    import numpy as np

    from .loaders import read_text

    details: dict[str, Any] = {}
    if access.K is not None:
        details["camera_intrinsics"] = np.asarray(access.K).round(3).tolist()
    if access.T_cam is not None:
        details["T_cam_to_base"] = np.asarray(access.T_cam).round(4).tolist()
    bag = read_text(access.dir / "source_bag.txt", 2048)
    if bag:
        details["source_bag"] = bag.strip()
    segments = access.segments
    details["segments"] = [
        {
            "index": s.index,
            "start_ts": s.start_ts,
            "end_ts": s.end_ts,
            "duration": s.duration,
            "n_poses": s.n_poses,
            "reason": s.reason,
        }
        for s in segments
    ]
    ts = access.frame_ts
    if ts is not None and len(ts) > 1:
        diffs = np.diff(ts)
        details["frame_rate_hz"] = round(float(1.0 / np.median(diffs)), 2) if np.median(diffs) > 0 else None
        details["duration_s"] = round(float(ts[-1] - ts[0]), 1)
    for key in ("joint_states", "cmd_vel", "joint_setpoints"):
        arr = getattr(access, key)
        if arr is not None:
            details[f"{key}_shape"] = list(arr.shape)
            if arr.ndim == 2 and arr.shape[0] > 1:
                span = float(arr[-1, 0] - arr[0, 0])
                details[f"{key}_rate_hz"] = round(arr.shape[0] / span, 1) if span > 0 else None
    return details


#: Thumbnails per page of the frame browser. One HTTP request each, lazily,
#: so this is a bound on concurrent image renders rather than on bytes read.
FRAMES_PER_PAGE = 100


def _frame_indices(access: SceneAccess) -> list[int]:
    """Every frame index this scene can show, in order.

    Normally just ``range(n_frames)``. Only when the per-frame directory is
    known to be incomplete is it listed, so the common case costs nothing.

    A present-but-empty directory says nothing about which frames exist, so it
    is skipped rather than accepted as the answer -- otherwise a scene whose
    ``images/`` had been created but not yet filled fell back to
    ``range(n_frames)`` and offered a page of tiles that all 404.
    """
    probe = access.probe
    for kind in ("images", "clouds"):
        if not probe.has(kind) or not probe.counts.get(kind):
            continue
        if probe.counts[kind] != probe.n_frames:
            return sorted(access.available_frames(kind))
        break
    return list(range(probe.n_frames))


def _page_of_frame(access: SceneAccess, index: int) -> int:
    """Which page of the frame browser holds ``index``.

    Not ``index // FRAMES_PER_PAGE``: on a scene with a sparse per-frame
    directory the browser pages through the frames that *exist*, so a frame's
    page is its position in that list, not its own number.
    """
    import bisect

    indices = _frame_indices(access)
    if not indices:
        return 0
    # The list is sorted, so the position is a search -- and for the usual
    # dense scene (0..n-1) that search lands on `index` itself.
    position = min(bisect.bisect_left(indices, index), len(indices) - 1)
    return position // FRAMES_PER_PAGE


def _frame_page(access: SceneAccess, page: int) -> dict[str, Any]:
    """One page of the frame browser."""
    indices = _frame_indices(access)
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


def _frame_details(access: SceneAccess, index: int, max_frames: int) -> dict[str, Any]:
    """Everything the per-frame page states about one frame.

    Cheap by construction: two timestamp lookups in arrays already in memory,
    one ``.npy`` header read for the cloud's point count, and ``is_file`` for
    the per-frame artefacts. No cloud, image or height map is decoded here --
    those are separate requests the browser makes for itself.
    """
    import numpy as np

    from .probe import _npy_shape

    probe = access.probe
    details: dict[str, Any] = {
        "index": index,
        "n_frames": probe.n_frames,
        # Which page of the frame browser this frame sits on, so "back to the
        # frame strip" lands where the frame actually is even on a scene whose
        # per-frame directory is sparse.
        "strip_page": _page_of_frame(access, index),
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
    for kind in ("images", "clouds", "height_maps", "height_masks"):
        present[kind] = probe.has(kind) and access.frame_path(kind, index).is_file()
    details["present"] = present

    if present["clouds"]:
        shape = _npy_shape(access.frame_path("clouds", index))
        if shape:
            details["cloud_points"] = int(shape[0])
            details["cloud_columns"] = int(shape[1]) if len(shape) > 1 else 1

    ga = access.gravity_align
    if ga is not None and 0 <= index < len(ga):
        from .loaders import roll_pitch_from_gravity_align

        roll, pitch = roll_pitch_from_gravity_align(np.asarray(ga[index])[None])
        details["roll_deg"] = round(float(np.degrees(roll[0])), 2)
        details["pitch_deg"] = round(float(np.degrees(pitch[0])), 2)

    seg = access.segment_of_frame(index)
    if seg is not None:
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
        }
    else:
        details["segment"] = None

    # The enumerated window this frame sits in: a ground-truth one if there is
    # one, otherwise the whole-scene fallback. This is the "show me the motion
    # around this frame" link, as opposed to the static single-frame view.
    containing = [w for w in access.windows(max_frames) if w.frame_start <= index < w.frame_end]
    best = next((w for w in containing if w.kind == "segment"), None) or (containing[0] if containing else None)
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


def _frame_info(access: SceneAccess, index: int) -> dict[str, Any]:
    """Per-frame numbers for the overlay's readout."""
    import numpy as np

    info: dict[str, Any] = {"frame": index}
    ts = access.frame_ts
    if ts is not None and 0 <= index < len(ts):
        info["timestamp"] = float(ts[index])
    pcd = access.pcd_ts
    if pcd is not None and 0 <= index < len(pcd):
        info["lidar_timestamp"] = float(pcd[index])
        if "timestamp" in info:
            info["camera_lidar_offset_ms"] = round((info["timestamp"] - float(pcd[index])) * 1000, 2)
    ga = access.gravity_align
    if ga is not None and 0 <= index < len(ga):
        from .loaders import roll_pitch_from_gravity_align

        roll, pitch = roll_pitch_from_gravity_align(np.asarray(ga[index])[None])
        info["roll_deg"] = round(float(np.degrees(roll[0])), 2)
        info["pitch_deg"] = round(float(np.degrees(pitch[0])), 2)
    info["has"] = {
        kind: index in access.available_frames(kind)
        for kind in ("images", "clouds", "height_maps", "height_masks")
        if access.probe.has(kind)
    }
    return info
