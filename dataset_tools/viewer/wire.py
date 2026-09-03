"""Serving a built scene to the SimView front-end without running SimView's server.

SimView's front-end normally fetches ``/model``, ``/states`` and ``/blob/...``
from its own app, rooted at ``/`` and so unmountable under a prefix. But it has
a documented static mode: with ``window.__simviewStaticBase`` set, every fetch
goes to ``{base}/model.json``, ``{base}/states.json`` and ``{base}/blob/{id}``
(SimView.js ``loadData`` / ``fetchBlobs``) -- the mode its own GitHub Pages demo
uses. This module splits ``SimulationScene.save``'s document into those flat
files. Nothing in SimView is modified or subclassed; the only coupling is its
documented, versioned on-disk format.

Splitting the base64 payloads back into standalone binaries is not just size:
the viewer range-requests them to window long trajectories, and files on disk
get byte-range support from Starlette for free.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger("dataset_tools.viewer.wire")

BLOB_PREFIX = "__b64__"


def scene_document(scene) -> dict:
    """The saved-scene document for a built :class:`SimulationScene`, in memory.

    What ``scene.save(path)`` would write, without the detour through a file.
    The important part is ``columnarize_states``: ``add_trajectory`` leaves
    ``scene.states`` as one dict per frame with a small blob per body, so
    splitting that yields thousands of tiny blobs and as many requests. The
    columnar layout packs each body's whole trajectory into one blob -- tens
    per scene, and the layout windowed range-fetching relies on.
    """
    model_json = scene.model.to_json()
    states = scene.states
    try:
        from simview.columnar import columnarize_states, inline_blob

        columnar = columnarize_states(states, model_json, inline_blob)
        if columnar is not None:
            states = columnar
        else:
            logger.warning(
                "states could not be packed columnar; falling back to the "
                "per-frame layout (slower to load, many more blobs)"
            )
    except ImportError:  # pragma: no cover - very old/newer simview layout
        logger.warning("simview.columnar is unavailable; using the per-frame layout")
    return {"model": model_json, "states": states}


def split_scene_document(doc: dict) -> tuple[dict, object, list[bytes]]:
    """Split a saved scene into ``(model, states, blobs)``.

    Every ``"__b64__<payload>"`` string anywhere in the document is replaced in
    place by a ``"/blob/static/{id}"`` reference and its decoded bytes appended
    to the returned list. The viewer only reads the final path segment of a
    blob reference in static mode, so the middle component is arbitrary.
    """
    import base64

    blobs: list[bytes] = []

    def rewrite(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, str) and value.startswith(BLOB_PREFIX):
                    node[key] = _register(value)
                else:
                    rewrite(value)
        elif isinstance(node, list):
            for i, value in enumerate(node):
                if isinstance(value, str) and value.startswith(BLOB_PREFIX):
                    node[i] = _register(value)
                else:
                    rewrite(value)

    def _register(value: str) -> str:
        blob_id = len(blobs)
        blobs.append(base64.b64decode(value[len(BLOB_PREFIX) :]))
        return f"/blob/static/{blob_id}"

    model = doc.get("model")
    states = doc.get("states")
    if model is None or states is None:
        raise ValueError("scene document is missing 'model' or 'states'")
    rewrite(model)
    rewrite(states)
    return model, states, blobs


def write_static_bundle(doc: dict, dest: Path) -> dict:
    """Write ``model.json``, ``states.json`` and ``blob/N`` under ``dest``.

    Written to a sibling temporary directory and moved into place, so a reader
    never sees a half-written bundle and two concurrent builds of the same
    window cannot interleave.
    """
    import os
    import shutil
    import tempfile

    model, states, blobs = split_scene_document(doc)
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{dest.name}.", dir=dest.parent))
    try:
        (staging / "blob").mkdir()
        (staging / "model.json").write_bytes(json.dumps(model).encode())
        (staging / "states.json").write_bytes(json.dumps(states).encode())
        for i, blob in enumerate(blobs):
            (staging / "blob" / str(i)).write_bytes(blob)
        info = {
            "blobs": len(blobs),
            "bytes": sum(len(b) for b in blobs),
        }
        (staging / "bundle.json").write_bytes(json.dumps(info).encode())
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        os.replace(staging, dest)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return info


def render_viewer_page(
    static_prefix: str,
    data_base: str,
    version: str,
    body_html: str = "",
    head_html: str = "",
) -> str:
    """Render SimView's own ``index.html`` against our asset and data URLs.

    The template ships inside the installed ``simview`` package and is read at
    request time rather than copied, so the page always matches whatever
    SimView version is installed -- including its import map and whichever
    third-party libraries that version vendors. Only two things are added: the
    ``__simviewStaticBase`` declaration (before the module script runs, which
    is what the front-end requires) and our own overlay markup.
    """
    from importlib.resources import files

    template = (files("simview") / "templates" / "index.html").read_text()

    # The template is Jinja2, but the only expressions in it are asset URLs and
    # a cache-busting stamp. Substituting them directly avoids depending on how
    # a given SimView version wires up its Jinja environment.
    import re

    def sub_url(match: re.Match) -> str:
        path = match.group(1)
        return f"{static_prefix.rstrip('/')}/{path}"

    page = re.sub(r"\{\{\s*url_for\('static',\s*path='([^']+)'\)\s*\}\}", sub_url, template)
    page = re.sub(r"\{\{\s*t\s*\}\}", version, page)

    if "{{" in page:
        logger.warning("simview's index.html has unsubstituted template expressions")

    # SimView's config.js refers to a few assets by the document-relative path
    # "static/textures/...", which resolves correctly only when the page is
    # served from the site root (as SimView's own server does). This page lives
    # at /view/<scene>/<window>, so without a base the browser would look for
    # /view/<scene>/static/... and the point sprite would 404. Declaring the
    # base restores root-relative resolution; every URL we emit ourselves is
    # already absolute, so nothing else is affected.
    page = page.replace("<head>", '<head>\n    <base href="/">', 1)

    injection = (
        f'<script>window.__simviewStaticBase = "{data_base.rstrip("/")}";</script>\n'
        f"{head_html}\n</head>"
    )
    page = page.replace("</head>", injection, 1)
    page = page.replace("</body>", f"{body_html}\n</body>", 1)
    return page
