"""The mutual nearest-neighbour sync must match the naive O(n*m) version."""

from __future__ import annotations

import numpy as np
import pytest

from dataset_tools.extract_data import nearest_idx, sync_timestamps


def brute_force_sync(cam, pcd):
    cam = np.sort(np.asarray(cam, dtype=float))
    pcd = np.sort(np.asarray(pcd, dtype=float))
    if len(cam) == 0 or len(pcd) == 0:
        return cam[:0], pcd[:0]
    distance = np.abs(cam[:, None] - pcd[None, :])
    cam_to_pcd = distance.argmin(axis=1)
    pcd_to_cam = distance.argmin(axis=0)
    mask = pcd_to_cam[cam_to_pcd] == np.arange(len(cam))
    return cam[mask], pcd[cam_to_pcd[mask]]


@pytest.mark.parametrize("seed", range(8))
def test_sync_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    n_cam = int(rng.integers(1, 60))
    n_pcd = int(rng.integers(1, 60))
    cam = 1000.0 + np.cumsum(rng.uniform(0.01, 0.4, n_cam))
    pcd = 1000.0 + np.cumsum(rng.uniform(0.01, 0.4, n_pcd))
    rng.shuffle(cam)  # bags may hold out-of-order messages
    rng.shuffle(pcd)

    got_cam, got_pcd = sync_timestamps(cam, pcd)
    exp_cam, exp_pcd = brute_force_sync(cam, pcd)
    np.testing.assert_allclose(got_cam, exp_cam)
    np.testing.assert_allclose(got_pcd, exp_pcd)


def test_sync_regular_streams_pair_one_to_one():
    cam = np.arange(10) * 0.1
    pcd = cam + 0.01
    got_cam, got_pcd = sync_timestamps(cam, pcd)
    assert len(got_cam) == 10
    np.testing.assert_allclose(got_pcd - got_cam, 0.01)


def test_sync_drops_unmatched_frames():
    cam = np.array([0.0, 1.0, 1.05, 5.0])
    pcd = np.array([1.02])
    got_cam, got_pcd = sync_timestamps(cam, pcd)
    assert len(got_cam) == 1
    assert got_cam[0] == pytest.approx(1.0)  # 1.0 is closer to 1.02 than 1.05 is
    assert got_pcd[0] == pytest.approx(1.02)


def test_sync_outputs_are_sorted_and_equal_length():
    rng = np.random.default_rng(3)
    cam = rng.uniform(0, 10, 40)
    pcd = rng.uniform(0, 10, 35)
    got_cam, got_pcd = sync_timestamps(cam, pcd)
    assert len(got_cam) == len(got_pcd)
    assert np.all(np.diff(got_cam) >= 0)
    assert np.all(np.diff(got_pcd) >= 0)


def test_sync_empty_inputs():
    for cam, pcd in (([], [1.0]), ([1.0], []), ([], [])):
        got_cam, got_pcd = sync_timestamps(cam, pcd)
        assert len(got_cam) == 0 and len(got_pcd) == 0


def test_nearest_idx_single_element_and_ties():
    assert nearest_idx(np.array([-5.0, 5.0]), np.array([1.0])).tolist() == [0, 0]
    # Exactly between two candidates -> the lower index wins (as does argmin).
    assert nearest_idx(np.array([1.5]), np.array([1.0, 2.0])).tolist() == [0]
