import numpy as np
import pytest
from scipy import ndimage

import bigstream.align
import bigstream.contrib.demons_align
import bigstream.transform as bst
from bigstream.align import alignment_pipeline


SHAPE = (24, 28, 32)
SPACING = np.array([2.0, 1.0, 0.5])
ORIGIN = np.array([10.0, -5.0, 3.0])
AFFINE = np.array([
    [1.04, 0.02, 0.00, 1.5],
    [0.00, 0.97, 0.03, -2.0],
    [0.01, 0.00, 1.02, 0.5],
    [0.00, 0.00, 0.00, 1.0],
])


def _smooth_field(shape=SHAPE, amplitude=1.0, seed=3):
    rng = np.random.default_rng(seed)
    field = ndimage.gaussian_filter(rng.standard_normal(shape + (3,)), (4, 4, 4, 0))
    return (amplitude * field / np.abs(field).max()).astype(np.float32)


def _smooth_volume(shape=SHAPE, seed=42):
    rng = np.random.default_rng(seed)
    return ndimage.gaussian_filter(rng.random(shape), 2).astype(np.float32)


def test_compose_affine_with_field_spacings_as_list_with_none():
    # what main_global_align_pipeline passes after get_transforms: a static
    # affine has spacing None, the global field its own spacing
    field = _smooth_field()
    composed = bst.compose_transform_list([AFFINE, field], [None, SPACING])
    assert composed.shape == SHAPE + (3,)


def _grid(shape=SHAPE, spacing=SPACING):
    return np.stack(np.meshgrid(
        *[np.arange(n) * s for n, s in zip(shape, spacing)], indexing='ij',
    ), axis=-1)


@pytest.mark.parametrize('affine', [
    AFFINE,
    # 180 degree rotation in yx about the grid center: the prealign case
    # (rounds imaged rotated) that used to mirror the image near the edges
    np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, (SHAPE[1] - 1) * SPACING[1]],
        [0.0, 0.0, -1.0, (SHAPE[2] - 1) * SPACING[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]),
])
def test_compose_affine_with_field_is_exact(affine):
    # d(x) = A (x + s(x)) - x must hold everywhere, including where the
    # field pushes x + s(x) outside the grid: a large field guarantees that
    field = _smooth_field(amplitude=4.0)
    grid = _grid()
    expected = (grid + field) @ affine[:3, :3].T + affine[:3, -1] - grid
    composed = bst.compose_transforms(affine, field, None, SPACING)
    assert composed.shape == field.shape
    assert composed.dtype == field.dtype
    np.testing.assert_allclose(composed, expected, atol=1e-4)


def test_compose_affine_with_field_off_grid_keeps_linear_part():
    # a field pushing the whole grid outside along x: the composite must
    # still flip x, not collapse to the edge value of the affine
    flip_x = np.eye(4)
    flip_x[2, 2] = -1.0
    flip_x[2, 3] = (SHAPE[2] - 1) * SPACING[2]
    field = np.zeros(SHAPE + (3,), dtype=np.float32)
    field[..., 2] = -SHAPE[2] * SPACING[2]
    composed = bst.compose_transforms(flip_x, field, None, SPACING)
    mov_x = _grid()[..., 2] + composed[..., 2]
    # x + s(x) = x - W, so the flip maps it to (W - 1) s - (x - W): decreasing in x
    assert np.all(np.diff(mov_x, axis=2) < 0)
    np.testing.assert_allclose(
        np.diff(mov_x, axis=2), -SPACING[2], atol=1e-5,
    )


def test_alignment_pipeline_flatten_with_fix_origin():
    # 'flatten' composes the affine step into the field step's grid, which
    # starts at fix_origin
    fix = _smooth_volume()
    mov = bst.apply_transform(
        fix, fix, SPACING, SPACING, transform_list=[bst.invert_affine(AFFINE)],
        fix_origin=ORIGIN, mov_origin=ORIGIN,
    ).astype(np.float32)
    steps = [
        ('affine', dict(metric='MS', shrink_factors=(2, 1), smooth_sigmas=(1.0, 0.0),
                        alignment_spacing=None)),
        ('demons', dict(iterations=[5], smooth_sigmas=[1.0], shrink_factors=[1])),
    ]
    kwargs = dict(fix_origin=ORIGIN, mov_origin=ORIGIN)
    independent = alignment_pipeline(fix, mov, SPACING, SPACING, steps,
                                     return_format='independent', **kwargs)
    flattened = alignment_pipeline(fix, mov, SPACING, SPACING, steps,
                                   return_format='flatten', **kwargs)
    by_list = bst.apply_transform(
        fix, mov, SPACING, SPACING, transform_list=independent,
        transform_origin=(ORIGIN, ORIGIN), **kwargs,
    )
    by_flattened = bst.apply_transform(
        fix, mov, SPACING, SPACING, transform_list=[flattened],
        transform_origin=ORIGIN, **kwargs,
    )
    interior = (slice(4, -4),) * 3
    np.testing.assert_allclose(by_flattened[interior], by_list[interior], atol=1e-3)


# ---------------------------------------------------------------------------
# 'flatten' with a nonzero fix_origin, with the registration steps replaced
# by fixed transforms so only the composition is measured.
#
# o is fix_origin and T(o) the translation by o. Composing turns an affine A
# into a field at grid positions measured from 0, while the fix grid starts
# at o; alignment_pipeline therefore composes T(-o) A T(o) instead. Without
# that, every voxel of the composed field is off by -(L - I) o, L being A's
# linear part (in the opposite direction).
# ---------------------------------------------------------------------------

FAR_ORIGIN = np.array([200.0, -150.0, 300.0])
FIELD_A = _smooth_field(amplitude=1.5, seed=3)
FIELD_B = _smooth_field(amplitude=1.0, seed=5)


def _translation(offset):
    matrix = np.eye(4)
    matrix[:3, -1] = offset
    return matrix


def _in_grid_frame(affine, origin):
    """T(-o) A T(o): the same affine for positions measured from the grid's start"""
    return _translation(-origin) @ affine @ _translation(origin)


# AFFINE about the block's own centre, as a fitted affine is: it moves the
# block a little, so the moved points stay on the fields' grid. The naive
# error -(L - I) o depends only on L and o, so it is as large as for AFFINE.
BLOCK_CENTRE = FAR_ORIGIN + (np.array(SHAPE) - 1) / 2 * SPACING
BLOCK_AFFINE = _translation(BLOCK_CENTRE) @ AFFINE @ _translation(-BLOCK_CENTRE)


@pytest.fixture
def fixed_steps(monkeypatch):
    """'affine' returns BLOCK_AFFINE; 'demons' returns FIELD_A, then FIELD_B"""
    fields = [FIELD_A, FIELD_B]
    monkeypatch.setattr(bigstream.align, 'affine_align', lambda *a, **k: BLOCK_AFFINE.copy())
    monkeypatch.setattr(bigstream.contrib.demons_align, 'demons_align',
                        lambda *a, **k: (None, fields.pop(0)))


def _flatten(steps, origin):
    fix = _smooth_volume()
    return alignment_pipeline(fix, fix, SPACING, SPACING, [(s, {}) for s in steps],
                              fix_origin=origin, mov_origin=origin)


def _warp(transforms, origin):
    fix = _smooth_volume()
    mov = _smooth_volume(seed=7)
    return bst.apply_transform(
        fix, mov, SPACING, SPACING, transform_list=transforms,
        transform_origin=tuple(origin for _ in transforms),
        fix_origin=origin, mov_origin=origin,
    )


@pytest.mark.parametrize('steps', [('affine', 'demons'), ('demons', 'affine')])
def test_flatten_affine_and_field_uses_the_grid_frame(fixed_steps, steps):
    transforms = {'affine': BLOCK_AFFINE, 'demons': FIELD_A}
    flattened = _flatten(steps, FAR_ORIGIN)

    # the fix: exactly the composition with each affine in the grid's frame
    expected = bst.compose_transform_list(
        [_in_grid_frame(transforms[s], FAR_ORIGIN) if s == 'affine' else transforms[s]
         for s in steps],
        SPACING,
    )
    np.testing.assert_array_equal(flattened, expected)

    # and it warps the moving image like the separate transforms do
    interior = (slice(4, -4),) * 3
    np.testing.assert_allclose(
        _warp([flattened], FAR_ORIGIN)[interior],
        _warp([transforms[s] for s in steps], FAR_ORIGIN)[interior],
        atol=1e-3,
    )

    # the bug it replaces: composing A as is evaluates it at g instead of
    # o + g, so (A g - g) - (A (o + g) - (o + g)) = -(L - I) o everywhere
    naive = bst.compose_transform_list([transforms[s] for s in steps], SPACING)
    if steps[0] == 'affine':
        # the affine applies last, so the offset is added unchanged everywhere
        error = -(BLOCK_AFFINE[:3, :3] - np.eye(3)) @ FAR_ORIGIN
        np.testing.assert_allclose(naive - flattened, np.broadcast_to(error, naive.shape), atol=1e-4)
    assert np.abs(naive - flattened).max() > 5.0


def test_flatten_fields_only_needs_no_origin(fixed_steps):
    flattened = _flatten(('demons', 'demons'), FAR_ORIGIN)
    # the same as composing the fields directly, and as with origin 0 ...
    np.testing.assert_array_equal(
        flattened, bst.compose_transform_list([FIELD_A, FIELD_B], SPACING),
    )
    # ... and correct at the far origin
    interior = (slice(4, -4),) * 3
    np.testing.assert_allclose(
        _warp([flattened], FAR_ORIGIN)[interior],
        _warp([FIELD_A, FIELD_B], FAR_ORIGIN)[interior],
        atol=1e-3,
    )


def test_flatten_affines_only_needs_no_origin(fixed_steps):
    np.testing.assert_allclose(
        _flatten(('affine', 'affine'), FAR_ORIGIN), BLOCK_AFFINE @ BLOCK_AFFINE, atol=1e-9,
    )
