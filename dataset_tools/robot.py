"""HELHEST geometry shared by the height-map builder and the viewer."""

import numpy as np

#: Wheel contacts in base_link (URDF: base_link is the front axle), rows left, right, rear; and the wheel radius.
WHEELS = np.array([[0.0, 0.36], [0.0, -0.36], [-0.697, 0.0]])
WHEEL_RADIUS = 0.36
