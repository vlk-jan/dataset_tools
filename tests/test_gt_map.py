"""The wheel-bottom-against-height-map test behind the map-mismatch cut."""

from __future__ import annotations

import numpy as np

from dataset_tools import gt_map


def test_a_robot_standing_on_the_map_has_zero_offset():
    """The wheel radius is the URDF's 0.36 m: measured flat ground 0.36 m under the axle is where the wheels sit."""
    T = np.tile(np.eye(4), (4, 1, 1))
    height = np.full((gt_map.MAP_N, gt_map.MAP_N), -0.36)
    validity = np.full((gt_map.MAP_N, gt_map.MAP_N), gt_map.VALIDITY_MEASURED)
    off, n = gt_map.window_offset(T, np.eye(3), height, validity, min_cells=1)
    assert off == 0.0 and n == 6
