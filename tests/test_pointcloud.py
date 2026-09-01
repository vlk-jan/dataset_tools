"""PointCloud2 field parsing, on synthetic message stand-ins."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from dataset_tools.extract_data import (
    PC2_DTYPES,
    cloud_to_xyzi,
    drop_invalid_returns,
    fields_to_dtype,
)

FLOAT32 = 7
UINT16 = 4


def field(name: str, offset: int, datatype: int = FLOAT32) -> SimpleNamespace:
    return SimpleNamespace(name=name, offset=offset, datatype=datatype, count=1)


def make_msg(points: np.ndarray, with_intensity: bool = True, padding: int = 0):
    """Build a PointCloud2-like object from an (N, 3 or 4) float array."""
    fields = [field("x", 0), field("y", 4), field("z", 8)]
    itemsize = 12
    if with_intensity:
        fields.append(field("intensity", 12))
        itemsize = 16
    itemsize += padding

    dtype = np.dtype(
        {
            "names": [f.name for f in fields],
            "formats": ["<f4"] * len(fields),
            "offsets": [f.offset for f in fields],
            "itemsize": itemsize,
        }
    )
    buffer = np.zeros(len(points), dtype=dtype)
    for i, name in enumerate(("x", "y", "z")):
        buffer[name] = points[:, i]
    if with_intensity:
        buffer["intensity"] = points[:, 3]
    return SimpleNamespace(
        fields=fields,
        point_step=itemsize,
        is_bigendian=False,
        width=len(points),
        height=1,
        data=buffer.tobytes(),
    )


def test_fields_to_dtype_respects_offsets_and_point_step():
    fields = [field("x", 0), field("y", 4), field("z", 8), field("ring", 16, UINT16)]
    dtype = fields_to_dtype(fields, point_step=32)
    assert dtype.itemsize == 32
    assert dtype.names == ("x", "y", "z", "ring")
    assert dtype["ring"] == np.dtype("<u2")
    assert dtype.fields["z"][1] == 8


def test_fields_to_dtype_big_endian():
    dtype = fields_to_dtype([field("x", 0)], point_step=4, big_endian=True)
    assert dtype["x"].byteorder in (
        ">",
        "=" if np.dtype(">f4").byteorder == "=" else ">",
    )


def test_pc2_dtype_table_matches_pointfield_enum():
    # sensor_msgs/PointField: INT8=1 ... FLOAT64=8
    assert [PC2_DTYPES[i] for i in range(1, 9)] == [
        "i1",
        "u1",
        "i2",
        "u2",
        "i4",
        "u4",
        "f4",
        "f8",
    ]


def test_cloud_to_xyzi_round_trip():
    rng = np.random.default_rng(1)
    points = rng.normal(size=(17, 4)).astype(np.float32)
    out = cloud_to_xyzi(make_msg(points))
    assert out.shape == (17, 4)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, points, rtol=0, atol=0)


def test_cloud_to_xyzi_with_padding_between_points():
    rng = np.random.default_rng(2)
    points = rng.normal(size=(5, 4)).astype(np.float32)
    out = cloud_to_xyzi(make_msg(points, padding=16))
    np.testing.assert_allclose(out, points)


def test_cloud_to_xyzi_without_intensity_field():
    rng = np.random.default_rng(3)
    points = rng.normal(size=(6, 4)).astype(np.float32)
    out = cloud_to_xyzi(make_msg(points, with_intensity=False))
    np.testing.assert_allclose(out[:, :3], points[:, :3])
    assert np.all(out[:, 3] == 0.0)


def test_cloud_to_xyzi_uses_compressed_data_when_present(monkeypatch):
    from dataset_tools import cloudini

    rng = np.random.default_rng(4)
    points = rng.normal(size=(4, 4)).astype(np.float32)
    msg = make_msg(points)
    raw = msg.data
    msg.compressed_data = b"pretend-this-is-cloudini"
    msg.data = b""

    monkeypatch.setattr(cloudini, "decode", lambda blob: raw)
    out = cloud_to_xyzi(msg)
    np.testing.assert_allclose(out, points)


def test_cloud_to_xyzi_organized_cloud_uses_width_times_height():
    rng = np.random.default_rng(5)
    points = rng.normal(size=(12, 4)).astype(np.float32)
    msg = make_msg(points)
    msg.width, msg.height = 4, 3
    assert cloud_to_xyzi(msg).shape == (12, 4)


# ---------------------------------------------------------------------------
# drop_invalid_returns: the guard that keeps the HELHEST v1 origin-delta bug
# from being reproduced on the Odin1 rig. See the function's docstring.
# ---------------------------------------------------------------------------
def test_drop_invalid_returns_drops_nan_rows():
    pts = np.array(
        [[1.0, 2.0, 3.0, 9.0], [np.nan, 2.0, 3.0, 9.0], [4.0, 5.0, 6.0, 9.0]],
        dtype=np.float32,
    )
    kept = drop_invalid_returns(pts)
    assert kept.shape == (2, 4)
    assert np.array_equal(kept[:, 0], [1.0, 4.0])


def test_drop_invalid_returns_drops_exact_origin():
    pts = np.array(
        [[1.0, 2.0, 3.0, 9.0], [0.0, 0.0, 0.0, 0.0], [4.0, 5.0, 6.0, 9.0]],
        dtype=np.float32,
    )
    kept = drop_invalid_returns(pts)
    assert kept.shape == (2, 4)
    assert not np.any(np.all(kept[:, :3] == 0, axis=1))


def test_drop_invalid_returns_ignores_intensity_when_deciding():
    """A dropped shot carrying a non-zero intensity is still a dropped shot."""
    pts = np.array([[0.0, 0.0, 0.0, 42.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    kept = drop_invalid_returns(pts)
    assert kept.shape == (1, 4)
    assert kept[0, 0] == 1.0


def test_drop_invalid_returns_keeps_points_near_but_not_at_origin():
    """Exact zeros only -- range gating belongs to the consumer, not here."""
    pts = np.array([[0.0, 0.0, 1e-6, 1.0], [0.0, 1e-7, 0.0, 1.0]], dtype=np.float32)
    assert drop_invalid_returns(pts).shape == (2, 4)


def test_drop_invalid_returns_preserves_order_and_columns():
    pts = np.array(
        [[3.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 2.0]],
        dtype=np.float32,
    )
    kept = drop_invalid_returns(pts)
    assert np.array_equal(kept[:, 0], [3.0, 1.0])
    assert np.array_equal(kept[:, 3], [1.0, 2.0])


def test_drop_invalid_returns_on_an_all_invalid_sweep():
    pts = np.array([[0.0, 0.0, 0.0, 0.0], [np.nan, np.nan, np.nan, 0.0]], dtype=np.float32)
    kept = drop_invalid_returns(pts)
    assert kept.shape == (0, 4)
