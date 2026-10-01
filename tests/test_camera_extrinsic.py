import numpy as np

from dataset_tools.extract.camera import save_camera_extrinsic
from dataset_tools.extract.tf_graph import get_transform


class _TF:
    def __init__(self, parent, child, t, q):
        class NS:
            pass

        self.header = NS()
        self.header.frame_id = parent
        self.child_frame_id = child
        self.transform = NS()
        self.transform.translation = NS()
        self.transform.rotation = NS()
        self.transform.translation.x, self.transform.translation.y, self.transform.translation.z = t
        self.transform.rotation.x, self.transform.rotation.y, self.transform.rotation.z, self.transform.rotation.w = q


def _graph():
    from dataset_tools.extract.tf_graph import build_tf_graph

    # base -> cam: pure translation
    return build_tf_graph([_TF("base_link", "cam_optical", (0.155, 0.0, 0.23), (0, 0, 0, 1))])


def test_writes_cam_to_base(tmp_path):
    G = _graph()
    assert save_camera_extrinsic(G, "cam_optical", "base_link", tmp_path)
    T = np.loadtxt(tmp_path / "T_cam_to_base.txt")
    # p_base = T @ p_cam: the camera origin lands at its mount position
    assert np.allclose(T @ [0, 0, 0, 1], [0.155, 0.0, 0.23, 1.0])
    assert np.allclose(T, get_transform(G, "cam_optical", "base_link"))


def test_missing_path_is_not_fatal(tmp_path):
    G = _graph()
    assert not save_camera_extrinsic(G, "unknown_frame", "base_link", tmp_path)
    assert not (tmp_path / "T_cam_to_base.txt").exists()
    assert not save_camera_extrinsic(G, "", "base_link", tmp_path)
