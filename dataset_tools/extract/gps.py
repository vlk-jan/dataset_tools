"""GNSS: which NavSatFix topic ``gps.npy`` comes from, and one row of it."""

from __future__ import annotations

import math

import numpy as np

#: The only message type ``gps.npy`` is ever built from. Both rig generations
#: publish one -- Fixposition on the new stack, Septentrio on the old -- under
#: names that have nothing in common, which is why the topic is found by type.
GPS_MSGTYPE = "sensor_msgs/msg/NavSatFix"

#: Ordered substrings preferred when ``--gps auto`` has several NavSatFix
#: topics to choose between. A fused solution (Fixposition's ``odometry_llh``,
#: which is its ENU odometry expressed geodetically) says more about where the
#: robot was than one antenna's raw fix, and a receiver's own ``navsatfix`` /
#: ``fix`` outranks whatever else a rig happens to publish.
GPS_TOPIC_PREFERENCE = ("odometry_llh", "navsatfix", "fix", "gnss", "gps")


def pick_gps_topic(connections, requested: str) -> str | None:
    """Which topic ``gps.npy`` comes from: the one asked for, or the best fix.

    ``off`` writes no GPS; an explicit topic is taken as given (and simply
    finds nothing if the bag does not carry it); ``auto`` searches by message
    type, so it needs no knowledge of a rig's naming. Topics recorded with
    zero messages are never chosen -- the new bags declare an empty
    ``/fixposition/gnss1`` next to the fix that is actually populated.
    """
    if requested == "off":
        return None
    if requested != "auto":
        return requested
    counts: dict[str, int] = {}
    for conn in connections:
        if conn.msgtype != GPS_MSGTYPE:
            continue
        counts[conn.topic] = counts.get(conn.topic, 0) + int(getattr(conn, "msgcount", 1) or 0)
    populated = {topic: n for topic, n in counts.items() if n}
    if not populated:
        return None

    def rank(topic: str) -> tuple[int, int, str]:
        for i, token in enumerate(GPS_TOPIC_PREFERENCE):
            if token in topic:
                return (i, -populated[topic], topic)
        return (len(GPS_TOPIC_PREFERENCE), -populated[topic], topic)

    return min(populated, key=rank)


def gps_row(t: float, msg) -> np.ndarray | None:
    """One ``gps.npy`` row from a ``NavSatFix``, or None for a fix with no position.

    ``[t, lat, lon, alt, status, cov_ee, cov_nn, cov_uu]``. The covariance is
    the diagonal of the message's row-major 3x3, in m^2, and is NaN when the
    message declares ``COVARIANCE_TYPE_UNKNOWN`` -- the numbers mean nothing
    then, and NaN says so without spending a column on the type constant.
    """
    lat, lon = float(msg.latitude), float(msg.longitude)
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return None
    # A Fixposition that has lost its solution publishes lat = lon = alt = 0
    # with position_covariance_type UNKNOWN -- while status still claims
    # GBAS_FIX, so the status field cannot be the filter. Exact null island is
    # never a fix a ground robot holds. Seen in the 2026-09-08 recordings:
    # 466 such rows in scene 0010 and 29 in 0014, each run spanning exactly one
    # ground-truth segment (the solution resets, the ENU pose teleports and the
    # attitude flips at the same moment).
    if lat == 0.0 and lon == 0.0:
        return None
    cov = np.asarray(msg.position_covariance, dtype=np.float64).ravel()
    unknown = int(getattr(msg, "position_covariance_type", 0)) == 0
    diag = np.full(3, np.nan) if unknown or cov.size < 9 else cov[[0, 4, 8]]
    return np.r_[t, lat, lon, float(msg.altitude), float(msg.status.status), diag]
