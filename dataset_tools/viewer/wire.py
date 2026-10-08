"""Serving a built scene to SimView's front-end in its static mode.

With ``window.__simviewStaticBase`` set, the front-end fetches
``{base}/model.json``, ``{base}/states.json`` and ``{base}/blob/{id}``, which
Starlette serves with byte-range support.
"""

from __future__ import annotations

import html
import logging
import re
from pathlib import Path

logger = logging.getLogger("dataset_tools.viewer.wire")


def scene_document(scene) -> dict:
    """The saved-scene document for a built :class:`SimulationScene`, in memory.

    What ``scene.save(path)`` would write. States are columnarised: one blob
    per body instead of thousands of per-frame ones, which range-fetching relies on.
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

    SimView's own layout, written to a sibling temp dir and moved into place so
    readers never see a half-written bundle and concurrent builds cannot interleave.
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

    Read from the installed package at request time so it always matches that
    SimView version; adds ``__simviewStaticBase`` (before the module script) and our markup.
    """
    from importlib.resources import files

    template = (files("simview") / "templates" / "index.html").read_text()

    # Page-relative "./static/" would resolve under /view/<scene>/; point it at our mount.
    page = template.replace('"./static/', f'"{static_prefix.rstrip("/")}/')

    if title is not None:
        page = re.sub(r"<title>.*?</title>", lambda _: f"<title>{html.escape(title)}</title>", page, count=1)

    injection = f'<script>window.__simviewStaticBase = "{data_base.rstrip("/")}";</script>\n' f"{head_html}\n</head>"
    page = page.replace("</head>", injection, 1)
    page = page.replace("</body>", f"{body_html}\n</body>", 1)
    return page
