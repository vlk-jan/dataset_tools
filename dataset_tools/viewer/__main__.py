"""Command-line entry point: ``python -m dataset_tools.viewer /path/to/root``."""

from __future__ import annotations

import argparse
import ipaddress
import logging
import sys
from pathlib import Path

from .windows import DEFAULT_MAX_FRAMES, DEFAULT_MAX_SECONDS

#: Bind addresses that mean "this machine only", and the Host names a browser
#: sends when it reaches one.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost"})


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
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind address (default: %(default)s). The server has no authentication, and notes "
        "are writes: keep it on the loopback interface and reach it over an SSH tunnel, or "
        "pass --no-notes before exposing it with 0.0.0.0",
    )
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
    parser.add_argument(
        "--no-prefetch",
        dest="prefetch",
        action="store_false",
        help="do not build the next window in the background when one is opened: for a root on storage "
        "slow enough that an unasked build competes with the one being looked at",
    )
    parser.add_argument(
        "--cache-max-gb",
        type=float,
        default=5.0,
        help="keep at most this much of built 3D windows on disk, oldest removed first (default: %(default)s)",
    )
    parser.add_argument("--open", action="store_true", help="open a browser at the index page")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser


#: Keep in step with pyproject.toml's simview requirement.
SIMVIEW_MIN = (6, 1, 1)


def _is_loopback(host: str) -> bool:
    """``localhost``, ``::1`` and any ``127.x.x.x``."""
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host in LOOPBACK_HOSTS


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cache_max_gb <= 0:
        parser.error("--cache-max-gb must be above 0")
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

    # An older SimView imports fine and would only fail per window build.
    import simview

    found = tuple(int(p) for p in simview.__version__.split(".")[:3] if p.isdigit())
    if found < SIMVIEW_MIN:
        print(
            f"error: the viewer needs simview >= {'.'.join(map(str, SIMVIEW_MIN))}, "
            f"this interpreter ({sys.executable}) has {simview.__version__}.\n"
            'Upgrade it with:  pip install -U "simview[authoring]"',
            file=sys.stderr,
        )
        return 2

    import uvicorn

    # Loopback binds answer only to loopback Host names (DNS-rebinding guard).
    loopback = _is_loopback(args.host)
    if not loopback and args.notes:
        logging.getLogger("dataset_tools.viewer").warning(
            "binding %s with notes on: anyone who can reach the port can write notes into the root "
            "(--no-notes turns that off)",
            args.host,
        )
    app = create_app(
        args.root,
        max_frames=args.max_frames,
        max_seconds=args.max_seconds,
        cache_dir=args.cache_dir,
        notes_dir=args.notes_dir,
        notes=args.notes,
        author=args.author,
        clear_cache=args.clear_cache,
        allowed_hosts=sorted({*LOOPBACK_HOSTS, args.host}) if loopback else None,
        prefetch=args.prefetch,
        cache_max_gb=args.cache_max_gb,
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
