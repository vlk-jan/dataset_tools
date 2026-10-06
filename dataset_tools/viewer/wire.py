"""Serving a built scene to the SimView front-end without running SimView's server.

SimView's front-end normally fetches ``/model``, ``/states`` and ``/blob/...``
from its own app, rooted at ``/`` and so unmountable under a prefix. But it has
a documented static mode: with ``window.__simviewStaticBase`` set, every fetch
goes to ``{base}/model.json``, ``{base}/states.json`` and ``{base}/blob/{id}``
(SimView.js ``loadData`` / ``fetchBlobs``) -- the mode its own GitHub Pages demo
uses. SimView writes those flat files itself
(``simview.columnar.write_static_bundle``); this module builds the document in
memory and moves the bundle into place atomically.

Standalone blob files are not just size: the viewer range-requests them to
window long trajectories, and files on disk get byte-range support from
Starlette for free.
"""

from __future__ import annotations

import html
import logging
import re
from pathlib import Path

logger = logging.getLogger("dataset_tools.viewer.wire")


def scene_document(scene) -> dict:
    """The saved-scene document for a built :class:`SimulationScene`, in memory.

    What ``scene.save(path)`` would write, without the detour through a file.
    The important part is ``columnarize_states``: ``add_trajectory`` leaves
    ``scene.states`` as one dict per frame with a small blob per body, so
    splitting that yields thousands of tiny blobs and as many requests. The
    columnar layout packs each body's whole trajectory into one blob -- tens
    per scene, and the layout windowed range-fetching relies on.
    """
    from simview.columnar import columnarize_states, inline_blob

    model_json = scene.model.to_json()
    states = columnarize_states(scene.states, model_json, inline_blob)
    if states is None:
        logger.warning(
            "states could not be packed columnar; falling back to the "
            "per-frame layout (slower to load, many more blobs)"
        )
        states = scene.states
    return {"model": model_json, "states": states}


def write_static_bundle(doc: dict, dest: Path) -> dict:
    """Write ``model.json``, ``states.json`` and ``blob/N`` under ``dest``.

    The layout is SimView's own (``simview.columnar.write_static_bundle``,
    which decodes every inline ``__b64__`` payload into ``blob/N``). What this
    adds is the atomic move: the bundle is written to a sibling temporary
    directory and moved into place, so a reader never sees a half-written
    bundle and two concurrent builds of the same window cannot interleave.
    """
    import os
    import shutil
    import tempfile

    from simview.columnar import write_static_bundle as simview_write

    if doc.get("model") is None or doc.get("states") is None:
        raise ValueError("scene document is missing 'model' or 'states'")
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{dest.name}.", dir=dest.parent))
    try:
        simview_write(doc, staging)
        blobs = list((staging / "blob").iterdir())
        info = {"blobs": len(blobs), "bytes": sum(p.stat().st_size for p in blobs)}
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
    body_html: str = "",
    head_html: str = "",
    title: str | None = None,
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

    # SimView refers to its assets page-relatively ("./static/..."), which
    # from /view/<scene>/<window> would resolve under /view/<scene>/; point
    # them at our versioned mount instead. Anything SimView loads from inside
    # its modules resolves against the module's own URL, so needs nothing.
    page = template.replace('"./static/', f'"{static_prefix.rstrip("/")}/')

    if title is not None:
        page = re.sub(r"<title>.*?</title>", lambda _: f"<title>{html.escape(title)}</title>", page, count=1)

    injection = f'<script>window.__simviewStaticBase = "{data_base.rstrip("/")}";</script>\n' f"{head_html}\n</head>"
    page = page.replace("</head>", injection, 1)
    page = page.replace("</body>", f"{body_html}\n</body>", 1)
    return page
