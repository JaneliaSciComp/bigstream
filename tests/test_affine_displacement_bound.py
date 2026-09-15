"""
Tests for the per-block affine displacement bound (`max_displacement` on
`affine_align`).

An affine has no coefficient grid, so C4 does not apply to it - the only way
it folds is by amplitude, when `distributed_align` blends it against a
neighbour that disagrees. These matrices are the real regression this bound
exists for: block (0,1,3) of a local registration run (blocksize 512) came
back with scale 0.78 and a 113-unit offset on x, while its healthy neighbour
block (0,1,2) is near identity. The affine is individually valid - det > 0,
orientation preserving - but blends into a fold anyway.
"""
import numpy as np
import pytest
import SimpleITK as sitk

import bigstream.transform as bst
from bigstream.align_constraints import (
    bound_affine_displacement,
    validate_affine_displacement_bound,
    is_orientation_preserving,
    blend_safe_displacement_bound,
)
from bigstream.distributed_align import _get_transform_weights


NDIM = 3

# real local-align affines from the same run, both blocks 512^3
DEGENERATE_AFFINE = np.array([
    [7.81793667e-01,  2.88132772e-03, -3.87507423e-02, 1.13021386e+02],
    [9.94930724e-06,  9.99974938e-01,  1.13753181e-04, 1.86404317e+00],
    [2.67358021e-04, -4.01985270e-04,  1.00013243e+00, 4.10692030e+01],
    [0.0,             0.0,             0.0,             1.0],
])
HEALTHY_NEIGHBOR_AFFINE = np.array([
    [1.00000999e+00,  5.85984105e-06, -3.18982597e-05, 4.13883489e+00],
    [3.56229926e-06,  9.99981790e-01, -1.06662946e-06, -4.91955853e+00],
    [9.38792943e-06, -1.55802307e-05,  9.99998930e-01, 3.87825682e+00],
    [0.0,             0.0,             0.0,             1.0],
])
BLOCK_EXTENT = np.array([512.0, 512.0, 512.0])


# ---------------------------------------------------------------------------
# 1. config validation
# ---------------------------------------------------------------------------

def test_none_disables_the_bound():
    assert validate_affine_displacement_bound(None, NDIM) is None


def test_scalar_broadcasts_to_zyx():
    np.testing.assert_allclose(
        validate_affine_displacement_bound(16.0, NDIM), [16.0] * NDIM)


def test_non_positive_bound_is_rejected():
    with pytest.raises(ValueError, match='max_displacement'):
        validate_affine_displacement_bound(0.0, NDIM)
    with pytest.raises(ValueError, match='max_displacement'):
        validate_affine_displacement_bound(-1.0, NDIM)


# ---------------------------------------------------------------------------
# 2. dead zone / no-op behaviour
# ---------------------------------------------------------------------------

def test_healthy_affine_is_left_unchanged():
    """A compliant affine is returned byte-identical - a dead zone operator."""
    matrix, info = bound_affine_displacement(
        HEALTHY_NEIGHBOR_AFFINE, BLOCK_EXTENT, 16.0)
    np.testing.assert_array_equal(matrix, HEALTHY_NEIGHBOR_AFFINE)
    assert info['clamped'] is False
    assert info['alpha'] == 1.0


# ---------------------------------------------------------------------------
# 3. clamping the real degenerate affine
# ---------------------------------------------------------------------------

def _displacement_at_corners(matrix, extent):
    """
    Independent check of the corner displacement: matrix_to_displacement_field
    on a 2x2x2 grid samples voxel index {0, 1}, which at spacing=extent lands
    exactly on the box corners {0, extent} - a different code path than
    bound_affine_displacement's own corner enumeration.
    """
    field = bst.matrix_to_displacement_field(
        matrix, (2, 2, 2), spacing=extent)
    return field.reshape(-1, NDIM)


def test_degenerate_affine_is_clamped_within_bound():
    bound = 16.0
    matrix, info = bound_affine_displacement(
        DEGENERATE_AFFINE, BLOCK_EXTENT, bound)

    assert info['clamped'] is True
    assert 0.0 < info['alpha'] < 1.0

    corner_displacements = _displacement_at_corners(matrix, BLOCK_EXTENT)
    assert np.abs(corner_displacements).max() <= bound + 1e-9


def test_clamping_preserves_orientation_and_shrinks_toward_identity():
    matrix, info = bound_affine_displacement(
        DEGENERATE_AFFINE, BLOCK_EXTENT, 16.0)

    assert is_orientation_preserving(matrix)
    assert info['orientation_preserved'] is True
    # det(A) = 0.782 is not degenerate on its own (that is exactly why a
    # determinant/orientation check alone cannot catch this affine) - scaling
    # toward identity moves it closer to 1, not further away
    det_before = np.linalg.det(DEGENERATE_AFFINE[:NDIM, :NDIM])
    det_after = np.linalg.det(matrix[:NDIM, :NDIM])
    assert det_before == pytest.approx(0.7818879714105799, rel=1e-6)
    assert abs(det_after - 1.0) < abs(det_before - 1.0)


def test_bound_is_per_component():
    U = [4.0, 50.0, 4.0]  # zyx: only y is generous
    matrix, info = bound_affine_displacement(DEGENERATE_AFFINE, BLOCK_EXTENT, U)
    corner_displacements = _displacement_at_corners(matrix, BLOCK_EXTENT)
    for q, limit in enumerate(U):
        assert np.abs(corner_displacements[:, q]).max() <= limit + 1e-9


# ---------------------------------------------------------------------------
# 4. the blend-level regression - what the bound is actually there to prevent
# ---------------------------------------------------------------------------

def _assemble(block_fields, blocksize, overlaps, nblocks, dims):
    """Stitch per-block fields exactly as _compute_block_transform does."""
    out = np.zeros(tuple(dims) + (NDIM,), dtype=np.float64)
    offsets = np.array(list(np.ndindex(*(3,) * NDIM))) - 1
    all_blocks = set(map(tuple, np.argwhere(np.ones(nblocks))))
    for bi in sorted(all_blocks):
        start = np.maximum(0, blocksize * np.array(bi) - overlaps)
        stop = np.minimum(dims, blocksize * np.array(bi) + blocksize + overlaps)
        coords = tuple(slice(a, b) for a, b in zip(start, stop))
        shape = tuple(b - a for a, b in zip(start, stop))
        nbrs = {tuple(int(x) for x in o): tuple(np.array(bi) + o) in all_blocks
                for o in offsets}
        w = _get_transform_weights(bi, (blocksize,) * NDIM, tuple(overlaps),
                                   nbrs, tuple(nblocks), True)
        w = w[tuple(slice(0, s) for s in shape)]
        out[coords] += block_fields(bi, shape) * w[..., None]
    return out


def _folded_fraction(field):
    disp = sitk.GetImageFromArray(field[..., ::-1].astype(np.float64),
                                  isVector=True)
    jac = sitk.GetArrayFromImage(sitk.DisplacementFieldJacobianDeterminant(disp))
    return float((jac <= 0).mean())


def test_unbounded_affine_folds_on_blend_but_bounded_one_does_not():
    """
    The real (0,1,3) affine, blended against a zero-displacement neighbour
    the way (0,1,2)'s near-identity affine effectively is, folds the stitched
    field - reproducing the bug. Clamping it with `bound_affine_displacement`
    at a `blend_safe_displacement_bound`-derived ceiling removes the fold,
    the same way `max_displacement` already does for the deform step.
    """
    blocksize, overlaps = 32, np.array([8, 8, 8])
    nblocks = np.array([3, 3, 3])
    dims = nblocks * blocksize
    hot = (1, 1, 1)

    def raw_fields(bi, shape):
        f = np.zeros(shape + (NDIM,), dtype=np.float64)
        if bi == hot:
            f[...] = bst.matrix_to_displacement_field(
                DEGENERATE_AFFINE, shape, spacing=np.ones(NDIM))
        return f

    folded_before = _folded_fraction(
        _assemble(raw_fields, blocksize, overlaps, nblocks, dims))
    assert folded_before > 0, 'expected the unbounded affine to fold on blend'

    # a bound derived the same way the deform step's own is - see
    # test_bound_excludes_the_measured_folding_threshold
    safe_bound = blend_safe_displacement_bound(
        overlaps, np.ones(NDIM), 1e-12, min_jacobian=0.0)
    clamped_affine, info = bound_affine_displacement(
        DEGENERATE_AFFINE, np.full(NDIM, float(blocksize)), safe_bound)
    assert info['clamped'] is True

    def bounded_fields(bi, shape):
        f = np.zeros(shape + (NDIM,), dtype=np.float64)
        if bi == hot:
            f[...] = bst.matrix_to_displacement_field(
                clamped_affine, shape, spacing=np.ones(NDIM))
        return f

    folded_after = _folded_fraction(
        _assemble(bounded_fields, blocksize, overlaps, nblocks, dims))
    assert folded_after == 0.0, (
        f'bounded affine should not fold on blend, got '
        f'{100 * folded_after:.3f}% folded'
    )
