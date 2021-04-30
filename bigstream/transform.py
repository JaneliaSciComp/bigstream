import numpy as np
from scipy.ndimage import map_coordinates
import zarr
import dask.array as da
import dask.delayed as delayed
from dask_stitch.local_affine import local_affines_to_field


def position_grid(shape):
    """
    """

    coords = np.meshgrid(*[range(x) for x in shape], indexing='ij')
    coords = np.array(coords).astype(np.int16)
    return np.ascontiguousarray(np.moveaxis(coords, 0, -1))


def affine_to_grid(matrix, grid, displacement=True):
    """
    """

    mm = matrix[:3, :3]
    tt = matrix[:3, -1]
    result = np.einsum('...ij,...j->...i', mm, grid) + tt
    if displacement:
        result = result - grid
    return result


def interpolate_image(image, X, order=1):
    """
    """

    X = np.moveaxis(X, -1, 0)
    return map_coordinates(image, X, order=order, mode='constant')


def apply_global_affine(
    fix, mov,
    fix_spacing, mov_spacing,
    affine,
    order=1,
):
    """
    """

    grid = position_grid(fix.shape) * fix_spacing
    coords = affine_to_grid(affine, grid, displacement=False) / mov_spacing
    return interpolate_image(mov, coords, order=order)


def compose_affines(global_affine, local_affines):
    """
    """

    # get block info
    block_grid = local_affines.shape[:3]
    nblocks = np.prod(block_grid)

    # compose with global affine
    total_affines = np.copy(local_affines)
    for i in range(nblocks):
        x, y, z = np.unravel_index(i, block_grid)
        total_affines[x, y, z] = np.matmul(
            global_affine, local_affines[x, y, z]
        )
    return total_affines


def prepare_apply_position_field(
    fix, mov,
    fix_spacing, mov_spacing,
    transform,
    blocksize,
    transpose=[False,]*3,
):
    """
    """

    # wrap transform as dask array, define chunks
    transform_da = transform
    if not isinstance(transform, da.Array):
        transform_da = da.from_array(transform)
    if transpose[2]:
        transform_da = transform_da.transpose(2,1,0,3)
        transform_da = transform_da[..., ::-1]
    transform_da = transform_da.rechunk(tuple(blocksize) + (3,))

    # wrap moving data appropriately
    if isinstance(mov, np.ndarray):
        mov_s = delayed(mov)
    elif isinstance(mov, zarr.Array):
        mov_s = mov

    # function to get per block origin and span in voxel units
    def transform_block(transform, mov):
        # convert to voxel units
        t = transform / mov_spacing
        # get moving data block coordinates
        s = np.floor(t.min(axis=(0,1,2))).astype(int)
        s = np.maximum(0, s)
        e = np.ceil(t.max(axis=(0,1,2))).astype(int) + 1
        slc = tuple(slice(x, y) for x, y in zip(s, e))
        # check transpose
        if transpose[1]: slc = slc[::-1]
        # slice data
        mov_block = mov[slc]
        # check transpose
        if transpose[1]: mov_block = mov_block.transpose(2,1,0)
        # interpolate block (adjust transform to local origin)
        return interpolate_image(mov_block, t - s)

    # map the interpolate function
    return da.map_blocks(
        transform_block,
        transform_da, mov=mov_s,
        dtype=mov.dtype,
        chunks=transform_da.chunks[:-1],
        drop_axis=[3,],
    )


def prepare_apply_local_affines(
    fix, mov,
    fix_spacing, mov_spacing,
    local_affines,
    blocksize,
    global_affine=None,
    transpose=[False,]*3,
):
    """
    """

    # get block grid info
    block_grid = local_affines.shape[:3]
    nblocks = np.prod(block_grid)

    # compose global/local affines
    total_affines = np.copy(local_affines)
    if global_affine is not None:
        total_affines = compose_affines(global_affine, local_affines)

    # get shape and overlap for position field
    pf_shape = fix.shape if not transpose[0] else fix.shape[::-1]
    overlap = [int(round(x/8)) for x in blocksize]

    # get task graph for local affines to position field
    position_field = local_affines_to_field(
        pf_shape, fix_spacing, total_affines,
        blocksize, overlap,
        displacement=False,
    )

    # align
    return prepare_apply_position_field(
        fix, mov, fix_spacing, mov_spacing,
        position_field, blocksize,
        transpose=transpose,
    )

