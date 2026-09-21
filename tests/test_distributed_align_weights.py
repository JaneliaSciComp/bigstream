import numpy as np
import pytest

from bigstream.distributed_align import _get_transform_weights


BLOCK_SIZE = (128, 128, 128)
OVERLAPS = (64, 64, 64)
NBLOCKS = (4, 4, 4)


def _neighbors(block_index, nblocks, missing=()):
    """Neighbor presence flags as the block partitioning builds them.

    Every offset whose target is off the volume is absent, plus any offset
    listed in `missing` (a block that exists but was dropped by the mask).
    """
    missing = {tuple(m) for m in missing}
    flags = {}
    for offset in np.array(list(np.ndindex(*(3,) * len(nblocks)))) - 1:
        offset = tuple(int(o) for o in offset)
        target = tuple(a + b for a, b in zip(block_index, offset))
        in_volume = all(0 <= i < n for i, n in zip(target, nblocks))
        flags[offset] = in_volume and offset not in missing
    return flags


def _profile(weights, axis=2):
    """1d slice through the middle of the weights along `axis`."""
    index = [s // 2 for s in weights.shape]
    index[axis] = slice(None)
    return weights[tuple(index)]


def test_interior_block_with_all_neighbors_ramps_to_zero():
    """The baseline: a fully surrounded block fades out over its overlap."""
    block_index = (2, 2, 2)
    weights = _get_transform_weights(
        block_index, BLOCK_SIZE, OVERLAPS,
        _neighbors(block_index, NBLOCKS), NBLOCKS, True,
    )
    profile = _profile(weights)
    assert profile[0] == pytest.approx(0.0)
    assert profile[-1] == pytest.approx(0.0)
    assert profile.max() == pytest.approx(1.0)


def test_unaligned_neighbor_does_not_get_rebalanced():
    """
    A block whose neighbor was dropped by the mask keeps its blending ramp.

    Rebalancing toward that neighbor would hold this block's displacement at
    full weight out to its last voxel and then step to zero, because nothing
    writes the territory the dropped block would have covered. That one voxel
    cliff carries the block's full displacement and folds the stitched field
    in a sheet along the mask boundary - outside the mask, where the
    deformation was never constrained by image data.
    """
    block_index = (2, 2, 2)
    weights = _get_transform_weights(
        block_index, BLOCK_SIZE, OVERLAPS,
        _neighbors(block_index, NBLOCKS, missing=[(0, 0, 1)]), NBLOCKS, True,
    )
    profile = _profile(weights)

    # the face toward the dropped neighbor still ramps all the way down
    assert profile[-1] == pytest.approx(0.0)
    # and it is a ramp, not a cliff: no full weight voxel next to the face
    assert profile[-2] < 0.05
    # a single voxel step must not carry a meaningful fraction of the field
    assert np.abs(np.diff(profile)).max() < 0.05
    # the untouched far face is unaffected
    assert profile[0] == pytest.approx(0.0)


def test_volume_edge_neighbor_is_still_rebalanced():
    """Off the volume there is no territory to fade into, so rebalance stands."""
    block_index = (2, 2, 3)  # last block along axis 2
    weights = _get_transform_weights(
        block_index, BLOCK_SIZE, OVERLAPS,
        _neighbors(block_index, NBLOCKS), NBLOCKS, True,
    )
    profile = _profile(weights)
    # the volume edge face is cropped and carries full weight
    assert profile[-1] == pytest.approx(1.0)
    # the interior face still ramps down toward its present neighbor
    assert profile[0] == pytest.approx(0.0)


def test_edge_and_unaligned_neighbors_are_handled_independently():
    """A block can sit on the volume edge and next to a dropped block at once."""
    block_index = (2, 2, 3)  # volume edge along axis 2
    weights = _get_transform_weights(
        block_index, BLOCK_SIZE, OVERLAPS,
        # axis 1 neighbor exists but was dropped by the mask
        _neighbors(block_index, NBLOCKS, missing=[(0, 1, 0)]), NBLOCKS, True,
    )
    # volume edge along axis 2 -> rebalanced to full weight
    assert _profile(weights, axis=2)[-1] == pytest.approx(1.0)
    # dropped neighbor along axis 1 -> ramp preserved
    axis1 = _profile(weights, axis=1)
    assert axis1[-1] == pytest.approx(0.0)
    assert np.abs(np.diff(axis1)).max() < 0.05


def test_rebalance_disabled_leaves_every_face_ramping():
    block_index = (2, 2, 3)
    weights = _get_transform_weights(
        block_index, BLOCK_SIZE, OVERLAPS,
        _neighbors(block_index, NBLOCKS), NBLOCKS, False,
    )
    profile = _profile(weights)
    assert profile[0] == pytest.approx(0.0)
    assert np.abs(np.diff(profile)).max() < 0.05


# --------------------------------------------------------------------------
# the same handling, under a non-linear blending ramp
#
# `rebalance_for_missing_neighbors` divides by `1 - missing_weights`, where
# `missing_weights` is gathered from the last `2*overlap` voxels of the face
# an off-volume neighbour would have covered. That arithmetic assumes nothing
# about the ramp's shape - only that the two faces are complementary - but it
# was never exercised against anything but the linear ramp.
# --------------------------------------------------------------------------


@pytest.mark.parametrize('ramp', ['linear', 'cosine'])
def test_rebalance_reaches_full_weight_at_the_volume_edge(ramp):
    """A rebalanced edge face carries the whole field, whatever the ramp."""
    block_index = (2, 2, 3)  # last block along axis 2
    weights = _get_transform_weights(
        block_index, BLOCK_SIZE, OVERLAPS,
        _neighbors(block_index, NBLOCKS), NBLOCKS, True, blend_ramp=ramp,
    )
    profile = _profile(weights)
    assert profile[-1] == pytest.approx(1.0)
    assert profile[0] == pytest.approx(0.0)
    # and it gets there monotonically - a rebalance that overshot would put a
    # weight above 1 next to the face and fold the field there
    assert weights.max() == pytest.approx(1.0)
    assert np.all(np.diff(profile[len(profile) // 2:]) >= -1e-12)


@pytest.mark.parametrize('ramp', ['linear', 'cosine'])
def test_unaligned_neighbor_keeps_its_ramp_under_any_shape(ramp):
    """
    The mask-boundary case, checked per shape: the face toward a dropped
    block must still fade to zero rather than step off a cliff.
    """
    block_index = (2, 2, 2)
    weights = _get_transform_weights(
        block_index, BLOCK_SIZE, OVERLAPS,
        _neighbors(block_index, NBLOCKS, missing=[(0, 0, 1)]), NBLOCKS, True,
        blend_ramp=ramp,
    )
    profile = _profile(weights)
    assert profile[-1] == pytest.approx(0.0)
    # cosine is pi/2 steeper mid-ramp, so the per-voxel step budget has to
    # allow for that - it is still nowhere near a cliff
    assert np.abs(np.diff(profile)).max() < 0.05


def test_cosine_and_linear_differ_where_it_matters():
    """
    A guard against the ramp argument being quietly ignored: the two shapes
    must actually produce different weights, agreeing only at the ramp ends
    and the midpoint (where both pass through 1/2).
    """
    block_index = (2, 2, 2)
    linear = _get_transform_weights(
        block_index, BLOCK_SIZE, OVERLAPS,
        _neighbors(block_index, NBLOCKS), NBLOCKS, True, blend_ramp='linear')
    cosine = _get_transform_weights(
        block_index, BLOCK_SIZE, OVERLAPS,
        _neighbors(block_index, NBLOCKS), NBLOCKS, True, blend_ramp='cosine')
    assert linear.shape == cosine.shape
    assert np.abs(linear - cosine).max() > 0.1
    # unset means linear
    default = _get_transform_weights(
        block_index, BLOCK_SIZE, OVERLAPS,
        _neighbors(block_index, NBLOCKS), NBLOCKS, True)
    assert np.array_equal(default, linear)


def test_unknown_ramp_is_refused_by_the_weight_builder():
    block_index = (2, 2, 2)
    with pytest.raises(ValueError, match='unsupported blend_ramp'):
        _get_transform_weights(
            block_index, BLOCK_SIZE, OVERLAPS,
            _neighbors(block_index, NBLOCKS), NBLOCKS, True,
            blend_ramp='hann')
