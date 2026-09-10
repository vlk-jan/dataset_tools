"""Command-line entry point: ``python -m dataset_tools.viewer /path/to/root``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .windows import DEFAULT_MAX_FRAMES, DEFAULT_MAX_SECONDS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dataset_tools.viewer",
        description=(
            "Browse an extracted dataset root in a web viewer: scene index, "
            "per-scene file inventory, and a 3D view (SimView) of each odometry "
            "segment with the camera image, height map and cloud alongside."
        ),
    )
    parser.add_argument("root", type=Path, help="dataset root directory (the one holding 0000/, 0001/, ...)")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: %(default)s)")
    parser.add_argument("--port", type=int, default=8080, help="port (default: %(default)s)")
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=DEFAULT_MAX_SECONDS,
        help="seconds per viewing window; longer segments are split into equal parts (default: %(default)s)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help=(
            "frames per viewing window: a backstop under --max-seconds, which is "
            "the cap that normally decides (default: %(default)s)"
        ),
    )
    parser.add_argument("--cache-dir", type=Path, default=None, help="where built scenes are cached")
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="delete this root's cached probes and built scenes before starting, so everything is rebuilt",
    )
    parser.add_argument(
        "--notes-dir",
        type=Path,
        default=None,
        help="where reviewer notes are kept (default: <root>/notes, so they travel with the dataset)",
    )
    parser.add_argument(
        "--no-notes",
        dest="notes",
        action="store_false",
        help="turn reviewer notes off entirely: nothing is read, written or shown",
    )
    parser.add_argument(
        "--author",
        default="",
        help="who new notes are from (default: $USER)",
    )
    parser.add_argument("--open", action="store_true", help="open a browser at the index page")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not args.root.is_dir():
        print(f"error: {args.root} is not a directory", file=sys.stderr)
        return 2

    try:
        from .app import create_app
    except ImportError as exc:  # pragma: no cover - dependency guidance
        print(
            f"error: the viewer's dependencies are not installed ({exc}).\n"
            'Install them with:  pip install "dataset_tools[viewer]"',
            file=sys.stderr,
        )
        return 2

    import uvicorn

    app = create_app(
        args.root,
        max_frames=args.max_frames,
        max_seconds=args.max_seconds,
        cache_dir=args.cache_dir,
        notes_dir=args.notes_dir,
        notes=args.notes,
        author=args.author,
        clear_cache=args.clear_cache,
    )
    url = f"http://{args.host}:{args.port}/"
    print(f"dataset viewer for {args.root}\n  {url}")
    if args.open:
        import threading
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="debug" if args.verbose else "info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
