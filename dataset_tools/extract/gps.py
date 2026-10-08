"""GNSS: which NavSatFix topic ``gps.npy`` comes from, and one row of it."""

from __future__ import annotations

import math

import numpy as np

#: ``gps.npy``'s source type; found by type since the rigs name topics differently.
GPS_MSGTYPE = "sensor_msgs/msg/NavSatFix"

#: ``--gps auto`` preference order; a fused solution beats a raw antenna fix.
GPS_TOPIC_PREFERENCE = ("odometry_llh", "navsatfix", "fix", "gnss", "gps")


def pick_gps_topic(connections, requested: str) -> str | None:
    """The topic ``gps.npy`` comes from: ``off`` -> None, explicit as given,
    ``auto`` -> the best populated NavSatFix topic."""
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

    ``[t, lat, lon, alt, status, cov_ee, cov_nn, cov_uu]``; covariance (m^2) is NaN
    when its type is unknown.
    """
    lat, lon = float(msg.latitude), float(msg.longitude)
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return None
    # A Fixposition without a solution publishes null island while status still claims a fix.
    if lat == 0.0 and lon == 0.0:
        return None
    cov = np.asarray(msg.position_covariance, dtype=np.float64).ravel()
    unknown = int(getattr(msg, "position_covariance_type", 0)) == 0
    diag = np.full(3, np.nan) if unknown or cov.size < 9 else cov[[0, 4, 8]]
    return np.r_[t, lat, lon, float(msg.altitude), float(msg.status.status), diag]
