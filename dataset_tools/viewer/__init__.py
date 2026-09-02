"""Web visualizer for extracted HELHEST-style dataset roots.

The viewer is driven entirely by the *file structure* of a dataset root, not by
any particular sensor rig: every per-scene file is probed and the UI degrades to
whatever is actually on disk (see :mod:`dataset_tools.viewer.probe`).

Run it with::

    python -m dataset_tools.viewer /path/to/dataset_root

or via the ``helhest-viewer`` console script.
"""

__all__ = ["create_app", "main"]


def create_app(*args, **kwargs):
    """Lazily import the FastAPI app factory (keeps ``import dataset_tools``
    free of the viewer's heavier optional dependencies)."""
    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)


def main(argv=None):
    from .__main__ import main as _main

    return _main(argv)
