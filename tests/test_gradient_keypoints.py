"""
The 'gradients' spot detection of `feature_point_ransac_affine_align`:
local maxima of the gaussian gradient magnitude.
"""

import numpy as np
import pytest

from scipy.ndimage import gaussian_filter, shift

from bigstream.align import feature_point_ransac_affine_align
from bigstream.features import gradient_keypoints


@pytest.fixture
def texture():
    rng = np.random.default_rng(0)
    return gaussian_filter(rng.random((48, 96, 96)), 2).astype(np.float32)


def test_keypoints_are_separated_and_inside_the_border(texture):
    cc_radius = (3, 5, 5)
    spots = gradient_keypoints(texture, 2, cc_radius)
    assert len(spots) > 50
    coords = spots[:, :3].astype(int)
    for axis, r in enumerate(cc_radius):
        assert np.all(coords[:, axis] >= r)
        assert np.all(coords[:, axis] < texture.shape[axis] - r)
    # one maximum per (2r+1) window: no two keypoints within r on all axes
    diff = np.abs(coords[:, None, :] - coords[None, :, :])
    close = np.all(diff <= np.array(cc_radius), axis=-1)
    np.fill_diagonal(close, False)
    assert not np.any(close)
    assert np.all(spots[:, -1] > 0)


def test_mask_keeps_only_foreground_keypoints(texture):
    mask = np.zeros(texture.shape, dtype=np.uint8)
    mask[:, :48, :] = 1
    spots = gradient_keypoints(texture, 2, 4, mask=mask)
    assert len(spots) > 0
    assert np.all(spots[:, 1] < 48)


def test_flat_image_has_no_keypoints():
    spots = gradient_keypoints(np.ones((20, 20, 20), dtype=np.float32), 2, 3)
    assert spots.shape == (0, 4)


def test_ransac_with_gradient_spots_recovers_a_translation(texture):
    offset = np.array([2.0, -3.0, 4.0])
    # mov(x) = fix(x - offset), so the fix -> mov affine adds offset
    mov = shift(texture, offset, order=1)
    affine = feature_point_ransac_affine_align(
        texture, mov, np.ones(3), np.ones(3),
        blob_sizes=[2, 4],
        cc_radius=(4, 6, 6),
        spot_detection_method='gradients',
        fix_spot_detection_kwargs={'blob_method': 'log'},
        match_threshold=0.9,
        max_spot_match_distance=10,
        point_matches_threshold=20,
        fix_spots_count_threshold=20,
        mov_spots_count_threshold=20,
    )
    np.testing.assert_allclose(affine[:3, :3], np.eye(3), atol=0.02)
    np.testing.assert_allclose(affine[:3, 3], offset, atol=0.5)


def test_unknown_spot_detection_method_is_refused(texture):
    with pytest.raises(ValueError, match='spot_detection_method'):
        feature_point_ransac_affine_align(
            texture, texture, np.ones(3), np.ones(3),
            blob_sizes=[2, 4], spot_detection_method='harris',
        )
