"""
Moving-block bounds in `bigstream.distributed_transform._transform_single_block`.

The log line `mov block start (voxels - unclipped)` can show negative
starts: it is printed before the bounds are clipped to the moving image.
These tests pin that the slices actually read from the moving image are
always inside it, and that clipping does not shift the result - the block
still matches a whole-image resample.
"""

import numpy as np
import pytest
import SimpleITK as sitk

import bigstream.transform as bs_transform
from bigstream.distributed_transform import _transform_single_block


SHAPE = (24, 28, 32)
SPACING = np.array([2.0, 1.0, 1.0])
# a shift that maps the block at the origin to negative moving coordinates
SHIFT = np.array([-6.0, -8.0, -5.0])      # physical units


def _smooth(seed):
    rng = np.random.default_rng(seed)
    img = sitk.GetImageFromArray(rng.random(SHAPE).astype(np.float32))
    return sitk.GetArrayFromImage(sitk.SmoothingRecursiveGaussian(img, 1.5))


class _RecordingWriter:
    def __init__(self, shape):
        self.output = np.zeros(shape, dtype=np.float32)

    def write(self, block_index, coords, block, context=''):
        self.output[coords] = block
        return coords


def _transform(kind):
    if kind == 'affine':
        affine = np.eye(4)
        affine[:3, 3] = SHIFT
        return affine, None
    field = np.broadcast_to(SHIFT.astype(np.float32), SHAPE + (3,)).copy()
    return field, SPACING


@pytest.mark.parametrize('kind', ['affine', 'deform'])
def test_negative_moving_block_is_clipped_and_matches_a_whole_image_run(kind):
    fix, mov = _smooth(0), _smooth(1)
    transform, transform_spacing = _transform(kind)

    reads = []

    def read_mov(slices):
        reads.append(slices)
        return mov[slices]

    block = (slice(0, 12), slice(0, 14), slice(0, 16))
    writer = _RecordingWriter(SHAPE)
    _transform_single_block(
        lambda coords: fix[coords], read_mov, ((0, 0, 0), block),
        full_mov_shape=SHAPE,
        fix_spacing=SPACING, mov_spacing=SPACING,
        blocksize=(12, 14, 16), blockoverlaps=(0, 0, 0),
        transform_list=[transform],
        transform_spacing_list=[transform_spacing],
        block_writer=writer,
    )

    # the corners map to negative voxels, but what is read is in bounds
    assert len(reads) == 1
    for s, n in zip(reads[0], SHAPE):
        assert 0 <= s.start < s.stop <= n
    assert all(s.start == 0 for s in reads[0])

    expected = bs_transform.apply_transform(
        fix, mov, SPACING, SPACING,
        transform_list=[transform],
        transform_spacing=(transform_spacing,),
    )
    np.testing.assert_allclose(writer.output[block], expected[block],
                               rtol=1e-5, atol=1e-5)


def test_block_mapped_entirely_outside_the_moving_image_is_skipped():
    fix, mov = _smooth(0), _smooth(1)
    affine = np.eye(4)
    affine[:3, 3] = -1000.0

    reads = []
    result = _transform_single_block(
        lambda coords: fix[coords],
        lambda slices: reads.append(slices) or mov[slices],
        ((0, 0, 0), (slice(0, 12), slice(0, 14), slice(0, 16))),
        full_mov_shape=SHAPE,
        fix_spacing=SPACING, mov_spacing=SPACING,
        blocksize=(12, 14, 16), blockoverlaps=(0, 0, 0),
        transform_list=[affine], transform_spacing_list=[None],
        block_writer=_RecordingWriter(SHAPE),
    )
    assert result == ((0, 0, 0), None)
    assert reads == []


def _flip_x_field(shape=SHAPE, spacing=SPACING):
    # displacement of x -> (W - 1) s - x along the last axis: a mirror, the
    # simplest field with a large linear part (like a global field carrying
    # a 180 degree rotation)
    field = np.zeros(shape + (3,), dtype=np.float32)
    x = np.arange(shape[2]) * spacing[2]
    field[..., 2] = (shape[2] - 1) * spacing[2] - 2 * x
    return field


def _varying_x_shift(shape=SHAPE, amplitude=6.0):
    # an x displacement that ramps along y from 0 to -amplitude, so inside a
    # block the points move well past the block's own x extent
    field = np.zeros(shape + (3,), dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, shape[1], dtype=np.float32)
    field[..., 2] = -amplitude * ramp[None, :, None]
    return field


def test_field_after_a_field_is_read_where_the_points_land():
    # [global, local]: local is applied first, so the global field is sampled
    # at the locally displaced positions - outside the fixed block's extent.
    # Reading the global field only over the fixed block clamped it there and
    # left zero (black) boxes at block boundaries
    fix, mov = _smooth(0), _smooth(1) + 1.0   # mov > 0 everywhere
    transform_list = [_flip_x_field(), _varying_x_shift()]
    spacings = [SPACING, SPACING]
    blocksize = (12, 14, 8)

    writer = _RecordingWriter(SHAPE)
    for bi in np.ndindex(*(np.array(SHAPE) // blocksize)):
        start = np.array(bi) * blocksize
        block = tuple(slice(a, a + b) for a, b in zip(start, blocksize))
        _transform_single_block(
            lambda coords: fix[coords], lambda slices: mov[slices], (bi, block),
            full_mov_shape=SHAPE,
            fix_spacing=SPACING, mov_spacing=SPACING,
            blocksize=blocksize, blockoverlaps=(0, 0, 0),
            transform_list=transform_list,
            transform_spacing_list=spacings,
            block_writer=writer,
        )

    expected = bs_transform.apply_transform(
        fix, mov, SPACING, SPACING,
        transform_list=transform_list,
        transform_spacing=tuple(spacings),
    )
    # wherever the whole-image apply lands inside mov, the blockwise one must
    # too, with the same value
    inside = expected > 0
    assert inside.mean() > 0.5
    assert np.all(writer.output[inside] > 0)
    np.testing.assert_allclose(writer.output, expected, rtol=1e-5, atol=1e-5)
