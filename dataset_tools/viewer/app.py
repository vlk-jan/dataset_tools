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

Reading the dataset happens in a worker thread throughout: the root normally
lives on contended network storage, and a slow read must never block the event
loop that is still serving the page around it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from . import render
from .build import BuildOptions, SceneAccess, build_scene
from .probe import ARTEFACTS, DatasetRoot
from .wire import render_viewer_page, scene_document, write_static_bundle

logger = logging.getLogger("dataset_tools.viewer.app")

HERE = Path(__file__).parent
IMMUTABLE = {"Cache-Control": "public, max-age=86400"}


def _simview_static_dir() -> Path:
    from importlib.resources import files

    return Path(str(files("simview") / "static"))


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
        self.bundle_dir = root.cache_dir / "bundles"
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
) -> FastAPI:
    root = DatasetRoot(root_path, cache_dir=Path(cache_dir) if cache_dir else None)
    state = ViewerState(root, max_frames=max_frames)

    app = FastAPI(title="dataset_tools viewer", docs_url=None, redoc_url=None)
    app.state.viewer = state

    templates = Jinja2Templates(directory=str(HERE / "templates"))
    app.mount("/viewer-static", StaticFiles(directory=str(HERE / "static")), name="viewer_static")

    simview_static = _simview_static_dir()
    simview_ok = simview_static.is_dir()
    if simview_ok:
        app.mount("/simview-static", StaticFiles(directory=str(simview_static)), name="simview_static")
        # Also at /static, where SimView's own document-relative asset paths
        # (config.js: "static/textures/points/ball1.png") expect to find them.
        app.mount("/static", StaticFiles(directory=str(simview_static)), name="simview_static_root")
    else:  # pragma: no cover - only when simview is missing/broken
        logger.error("simview static assets not found at %s; the 3D view will not work", simview_static)

    # ------------------------------------------------------------------ pages

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        probes = await run_in_threadpool(root.probe_all)
        rows = []
        csv_rows = root.csv_rows()
        for probe in probes:
            row = csv_rows.get(probe.scene_id, {})
            rows.append({"probe": probe, "csv": row})
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "root": root,
                "rows": rows,
                "artefacts": ARTEFACTS,
                "manifest": root.manifest(),
                "simview_version": _simview_version(),
                "simview_ok": simview_ok,
            },
        )

    @app.get("/scene/{scene_id}", response_class=HTMLResponse)
    async def scene_page(request: Request, scene_id: str):
        access = _access_or_404(state, scene_id)
        windows = await run_in_threadpool(access.windows, state.max_frames)
        details = await run_in_threadpool(_scene_details, access)
        return templates.TemplateResponse(
            request=request,
            name="scene.html",
            context={
                "root": root,
                "scene_id": scene_id,
                "probe": access.probe,
                "windows": windows,
                "artefacts": ARTEFACTS,
                "details": details,
                "csv": root.csv_rows().get(scene_id, {}),
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
        overlay = templates.get_template("overlay.html").render(
            scene_id=scene_id,
            window=window,
            probe=access.probe,
            data_base=f"/api/data/{bundle_key}",
            meta_url=f"/api/data/{bundle_key}/meta.json",
            status_url=f"/api/data/{bundle_key}/status",
            frame_base=f"/api/scene/{scene_id}/frame",
            scene_url=f"/scene/{scene_id}",
        )
        config = json.dumps(
            {
                "sceneId": scene_id,
                "windowKey": window_key,
                "dataBase": f"/api/data/{bundle_key}",
                "frameBase": f"/api/scene/{scene_id}/frame",
                "statusUrl": f"/api/data/{bundle_key}/status",
                "metaUrl": f"/api/data/{bundle_key}/meta.json",
            }
        )
        head = (
            '<link rel="stylesheet" href="/viewer-static/overlay.css">'
            f"<script>window.__dtViewer = {config};</script>"
        )
        body = overlay + '<script type="module" src="/viewer-static/overlay.js"></script>'
        page = await run_in_threadpool(
            render_viewer_page,
            "/simview-static",
            f"/api/data/{bundle_key}",
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
        return Response(data, media_type="image/jpeg", headers=IMMUTABLE)

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
        return Response(data, media_type="image/png", headers=IMMUTABLE)

    # ------------------------------------------------------------ scene data

    @app.get("/api/data/{scene_id}/{window_key}/{opts_key}/status")
    async def api_status(request: Request, scene_id: str, window_key: str, opts_key: str):
        path = state.bundle_dir / scene_id / window_key / opts_key
        return {"ready": (path / "model.json").is_file(), "path": str(path)}

    @app.get("/api/data/{scene_id}/{window_key}/{opts_key}/meta.json")
    async def api_meta(request: Request, scene_id: str, window_key: str, opts_key: str):
        bundle = await _ensure_bundle(state, scene_id, window_key, opts_key)
        return FileResponse(bundle / "meta.json", media_type="application/json")

    @app.get("/api/data/{scene_id}/{window_key}/{opts_key}/model.json")
    async def api_model(scene_id: str, window_key: str, opts_key: str):
        bundle = await _ensure_bundle(state, scene_id, window_key, opts_key)
        return FileResponse(bundle / "model.json", media_type="application/json")

    @app.get("/api/data/{scene_id}/{window_key}/{opts_key}/states.json")
    async def api_states(scene_id: str, window_key: str, opts_key: str):
        bundle = await _ensure_bundle(state, scene_id, window_key, opts_key)
        return FileResponse(bundle / "states.json", media_type="application/json")

    @app.get("/api/data/{scene_id}/{window_key}/{opts_key}/blob/{blob_id}")
    async def api_blob(scene_id: str, window_key: str, opts_key: str, blob_id: str):
        if not blob_id.isdigit():
            raise HTTPException(404, "bad blob id")
        bundle = await _ensure_bundle(state, scene_id, window_key, opts_key)
        path = bundle / "blob" / blob_id
        if not path.is_file():
            raise HTTPException(404, f"no blob {blob_id}")
        # FileResponse handles Range itself, which is what lets the SimView
        # front-end window a long trajectory instead of loading it whole.
        return FileResponse(
            path,
            media_type="application/octet-stream",
            headers={"Cache-Control": "public, max-age=86400", "Accept-Ranges": "bytes"},
        )

    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "root": str(root.path), "scenes": len(root.scene_ids())}

    return app


# ---------------------------------------------------------------- helpers


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


async def _ensure_bundle(state: ViewerState, scene_id: str, window_key: str, opts_key: str) -> Path:
    """Return the on-disk bundle for a window, building it if necessary.

    One lock per bundle key: several browser fetches for the same window arrive
    together (model, states, then every blob), and without it they would each
    kick off their own build of the same thing.
    """
    dest = state.bundle_dir / scene_id / window_key / opts_key
    if (dest / "model.json").is_file() and (dest / "meta.json").is_file():
        return dest

    async with state.lock(f"{scene_id}/{window_key}/{opts_key}"):
        if (dest / "model.json").is_file() and (dest / "meta.json").is_file():
            return dest
        opts = _options_from_key(opts_key)

        def work() -> Path:
            started = time.time()
            scene, meta = build_scene(
                state.root, scene_id, window_key, opts, max_frames=state.max_frames
            )
            doc = scene_document(scene)
            info = write_static_bundle(doc, dest)
            meta["build"]["seconds"] = round(time.time() - started, 2)
            meta["build"]["bundle"] = info
            (dest / "meta.json").write_text(json.dumps(meta))
            logger.info(
                "built %s/%s in %.1fs (%d blobs, %.1f MB)",
                scene_id, window_key, time.time() - started, info["blobs"], info["bytes"] / 1e6,
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
    return Response(data, media_type="image/png", headers=IMMUTABLE)


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
