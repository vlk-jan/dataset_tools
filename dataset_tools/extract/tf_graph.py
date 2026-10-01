"""The static TF tree as a graph of point maps, and paths through it."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

import numpy as np

from dataset_tools.transforms import quaternion_matrix


def build_tf_graph(tf_msgs: Iterable) -> dict[str, dict[str, np.ndarray]]:
    """Build a bidirectional frame graph from TF messages.

    A TF message with ``header.frame_id = parent`` and ``child_frame_id = child``
    carries the pose of the child in the parent frame, i.e. the matrix ``M``
    with ``p_parent = M @ p_child``. Every edge ``G[a][b]`` of the graph is the
    *point map* from ``a`` to ``b`` (``p_b = E @ p_a``), so ``G[parent][child]``
    holds ``inv(M)`` and ``G[child][parent]`` holds ``M``.
    """
    G: dict[str, dict[str, np.ndarray]] = {}
    for tf in tf_msgs:
        q = tf.transform.rotation
        M = quaternion_matrix([q.x, q.y, q.z, q.w])
        M[:3, 3] = [
            tf.transform.translation.x,
            tf.transform.translation.y,
            tf.transform.translation.z,
        ]
        G.setdefault(tf.header.frame_id, {})[tf.child_frame_id] = np.linalg.inv(M)
        G.setdefault(tf.child_frame_id, {})[tf.header.frame_id] = M
    return G


def get_transform(G: dict[str, dict[str, np.ndarray]], src_frame, dst_frame) -> np.ndarray | None:
    """Transform T such that ``p_dst = T @ p_src``, or None if no TF path exists.

    Edges store point maps along their direction (see :func:`build_tf_graph`),
    so the maps are chained right-to-left along the shortest path (BFS).
    """
    if src_frame == dst_frame:
        return np.eye(4)
    if src_frame not in G or dst_frame not in G:
        return None
    parent = {src_frame: None}
    queue = deque([src_frame])
    while queue and dst_frame not in parent:
        a = queue.popleft()
        for b in G[a]:
            if b not in parent:
                parent[b] = a
                queue.append(b)
    if dst_frame not in parent:
        return None
    T = np.eye(4)
    b = dst_frame
    while parent[b] is not None:
        T = T @ G[parent[b]][b]
        b = parent[b]
    return T
