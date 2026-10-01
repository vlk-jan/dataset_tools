"""Tools for extracting the HELHEST robot dataset from ROS 2 bags.

See README.md for the output format; the extraction entry point is
``dataset_tools.extract`` (``helhest-extract``) and the checker is
``dataset_tools.validate_scene``. The ground-truth files are read and written
through ``dataset_tools.gt_io``, and ``dataset_tools.viewer`` browses a root.
"""
