"""Tools for extracting the HELHEST robot dataset from ROS 2 bags.

See README.md for the output format; the extraction entry point is
``dataset_tools.extract_data`` and the checker is
``dataset_tools.validate_scene``. The ground-truth files are read and written
through ``dataset_tools.gt_io``, and ``dataset_tools.viewer`` browses a root.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dataset_tools")
except PackageNotFoundError:  # a checkout on sys.path that was never installed
    __version__ = "0+unknown"
