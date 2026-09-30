"""Web visualizer for extracted HELHEST-style dataset roots.

The viewer is driven entirely by the *file structure* of a dataset root, not by
any particular sensor rig: every per-scene file is probed and the UI degrades to
whatever is actually on disk (see :mod:`dataset_tools.viewer.probe`).

Run it with::

    python -m dataset_tools.viewer /path/to/dataset_root

or via the ``helhest-viewer`` console script.
"""
