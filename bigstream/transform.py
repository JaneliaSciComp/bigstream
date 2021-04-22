import numpy as np
from scipy.ndimage import map_coordinates
import zarr
from numcodecs import Blosc
import dask.array as da
import dask.delayed as delayed
from scipy.ndimage import zoom
import ClusterWrap
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


# DASK functions
def apply_position_field(
    fix, mov,
    fix_spacing, mov_spacing,
    transform,
    blocksize,
    transpose=[False,]*3,
    write_path=None,
    lazy=True,
    cluster_kwargs={},
):
    """
    """

    # get number of jobs needed
    nblocks = np.prod(np.ceil(np.array(fix.shape) / blocksize))

    # start cluster
    with ClusterWrap.cluster(**cluster_kwargs) as cluster:
        if write_path is not None or not lazy:
            cluster.scale_cluster(nblocks + 1)

        # wrap transform as dask array, define chunks
        transform_da = transform
        if not isinstance(transform, da.Array):
            transform_da = da.from_array(transform)
        if transpose[2]:
            transform_da = transform_da.transpose(2,1,0,3)
            transform_da = transform_da[..., ::-1]
        transform_da = transform_da.rechunk(tuple(blocksize) + (3,))

        # function to get per block origin and span in voxel units
        def get_origin_and_span(t_block, mov_spacing):
            mins = t_block.min(axis=(0,1,2))
            maxs = t_block.max(axis=(0,1,2))
            mins = np.floor(mins / mov_spacing).astype(int)
            maxs = np.ceil(maxs / mov_spacing).astype(int)
            os = np.empty((2,3))
            os[0] = np.maximum(0, mins - 3)
            os[1] = maxs - mins + 6
            return os.reshape((1,1,1,2,3))

        # get per block origins and spans
        os = da.map_blocks(
            get_origin_and_span,
            transform_da,
            mov_spacing=mov_spacing,
            dtype=int,
            new_axis=[4,],
            chunks=(1,1,1,2,3),
        )

        # extract crop info
        origins_da = os[..., 0, :]
        span = np.max(os[..., 1, :], axis=(0,1,2))

        # wrap moving data access with a function
        def moving_data_chunk(origin, span):
            origin = origin.astype(int)
            span = span.astype(int)
            return mov[tuple(slice(o, o+s) for o, s in zip(origin, span))]

        # delay the data access function
        mdc = delayed(moving_data_chunk, pure=True)

        # cut moving data into blocks
        sh = origins_da.shape[:3]
        mov_da = [[[da.from_delayed(
                    mdc(origins_da[i, j, k], span),
                    dtype=mov.dtype,
                    shape=span) 
                    for k in range(sh[2])]
                    for j in range(sh[1])]
                    for i in range(sh[0])]

        # reformat to dask array, extra dimension for map_blocks
        mov_da = da.block(mov_da)[..., None]

        # wrap interpolate function
        def wrapped_interpolate_image(t, mov, origin, mov_spacing):
            mov, origin = mov.squeeze(), origin.squeeze()
            t = t / mov_spacing - origin
            return interpolate_image(mov, t)

        print(transform_da)
        print(mov_da)
        print(origins_da)
        return 0

        # map the interpolate function
        aligned = da.map_blocks(
            wrapped_interpolate_image,
            transform_da, mov_da, origins_da,
            mov_spacing=mov_spacing,
            dtype=mov.dtype,
            chunks=tuple(blocksize),
            drop_axis=[3,],
        )

        # if user wants to write to disk
        if write_path is not None:
            compressor = Blosc(
                cname='zstd',
                clevel=4,
                shuffle=Blosc.BITSHUFFLE,
            )
            aligned_disk = zarr.open(
                write_path, 'w',
                shape=transform_da.shape[:-1],
                chunks=aligned.chunksize,
                dtype=aligned.dtype,
                compressor=compressor,
            )
            da.to_zarr(aligned, aligned_disk)
            return aligned_disk

        # if user wants to compute and return full resampled image
        if not lazy:
            return aligned.compute()

        # if user wants to return compute graph w/o executing
        if lazy:
            return aligned


def apply_local_affines(
    fix, mov,
    fix_spacing, mov_spacing,
    local_affines,
    blocksize,
    global_affine=None,
    write_path=None,
    lazy=True,
    transpose=[False,]*3,
    cluster_kwargs={},
):
    """
    """

    # get block grid info
    block_grid = local_affines.shape[:3]
    nblocks = np.prod(block_grid)

    # compose with global affine
    total_affines = np.copy(local_affines)
    if global_affine is not None:
        for i in range(nblocks):
            x, y, z = np.unravel_index(i, block_grid)
            total_affines[x, y, z] = np.matmul(
                global_affine, local_affines[x, y, z]
            )

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
    aligned = apply_position_field(
        fix, mov, fix_spacing, mov_spacing,
        position_field, blocksize,
        write_path=write_path,
        lazy=lazy,
        transpose=transpose,
        cluster_kwargs=cluster_kwargs,
    )

    return aligned

