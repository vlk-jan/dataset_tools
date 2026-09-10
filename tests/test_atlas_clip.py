"""Ground-truth statistics are clipped to the frame window.

The Odin comes up later than the Fixposition in several 2026-09 bags, so
``odometry.txt`` legitimately covers time with no image and no cloud. Counting
it made the atlas report more time "with GT" than the scene's own duration,
which cannot be true of a subset.
"""

import sys
from pathlib import Path

import pytest

from dataset_tools.viewer.probe import SceneProbe

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from make_atlas import clip, frame_window  # noqa: E402


def test_clip_keeps_only_the_overlap():
    assert clip((10.0, 20.0), None) == pytest.approx(10.0)  # unwindowed: the whole span
    assert clip((10.0, 20.0), (0.0, 100.0)) == pytest.approx(10.0)
    assert clip((10.0, 20.0), (15.0, 100.0)) == pytest.approx(5.0)  # starts before the frames
    assert clip((10.0, 20.0), (0.0, 12.0)) == pytest.approx(2.0)  # runs past the last frame
    assert clip((10.0, 20.0), (12.0, 15.0)) == pytest.approx(3.0)  # window inside the segment
    assert clip((10.0, 20.0), (30.0, 40.0)) == 0.0  # wholly outside


def test_frame_window_needs_a_real_span():
    assert frame_window(SceneProbe("0000")) is None
    assert frame_window(SceneProbe("0000", start_ts=5.0, end_ts=5.0)) is None
    assert frame_window(SceneProbe("0000", start_ts=5.0, end_ts=9.0)) == (5.0, 9.0)


def test_a_segment_that_predates_the_frames_contributes_nothing():
    """Scene 0000's shape: 141 s of ground truth before the first frame."""
    window = (1000.0, 1290.0)
    assert clip((858.5, 1290.0), window) == pytest.approx(290.0)
    assert clip((858.5, 1290.0), None) == pytest.approx(431.5)
