import numpy as np
import pytest

from bigstream.distributed_align import _get_transform_weights


BLOCK_SIZE = (128, 128, 128)
OVERLAPS = (64, 64, 64)
NBLOCKS = (4, 4, 4)


def _neighbors(block_index, nblocks, missing=()):
    """Neighbor presence flags as distributed_alignment_pipeline builds them.

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
