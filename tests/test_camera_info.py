from types import SimpleNamespace

import numpy as np

from dataset_tools.extract.camera import write_camera_info_yaml

try:
    import yaml
except ImportError:
    yaml = None


def _msg():
    return SimpleNamespace(
        header=SimpleNamespace(frame_id="oak_rgb_camera_optical_frame"),
        width=1280,
        height=800,
        distortion_model="rational_polynomial",
        d=[0.1, -0.2, 0.001, 0.002, 0.05, 0.0, 0.0, 0.0],
        k=[761.6, 0, 633.9, 0, 761.6, 407.0, 0, 0, 1],
        r=list(np.eye(3).ravel()),
        p=[761.6, 0, 633.9, 0, 0, 761.6, 407.0, 0, 0, 0, 1, 0],
        binning_x=0,
        binning_y=0,
    )


def test_camera_info_yaml_round_trips(tmp_path):
    out = tmp_path / "camera_info.yaml"
    write_camera_info_yaml(_msg(), out)
    text = out.read_text()
    assert "distortion_model: rational_polynomial" in text
    assert "image_width: 1280" in text
    if yaml is not None:
        doc = yaml.safe_load(text)
        assert doc["distortion_coefficients"]["cols"] == 8
        assert np.allclose(doc["camera_matrix"]["data"], _msg().k)
        assert np.allclose(doc["projection_matrix"]["data"], _msg().p)
