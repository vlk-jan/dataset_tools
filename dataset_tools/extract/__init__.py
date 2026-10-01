"""Bag -> scene extraction (``helhest-extract``), split by stage.

- :mod:`.cli` -- arguments, bag discovery and the worker pool
- :mod:`.scene` -- one bag into one scene directory
- :mod:`.scene_io` -- what a scene directory holds, and whether it is finished
- :mod:`.sync`, :mod:`.pointcloud`, :mod:`.gps`, :mod:`.tf_graph`, :mod:`.camera`,
  :mod:`.ground_truth` -- the pieces :mod:`.scene` is built from
"""
