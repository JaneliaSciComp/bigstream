import numpy as np
from scipy.ndimage import map_coordinates
import zarr
from numcodecs import Blosc
from bigstream import stitch
import dask.array as da
from scipy.ndimage import zoom
from ClusterWrap.clusters import janelia_lsf_cluster


def position_grid(shape, dtype=np.uint16):
    """
    """

    coords = np.array(
        np.meshgrid(*[range(x) for x in shape], indexing='ij'),
        dtype=dtype,
    )
    return np.ascontiguousarray(np.moveaxis(coords, 0, -1))


def affine_to_grid(matrix, grid, displacement=False):
    """
    """

    mm = matrix[:3, :-1]
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


def apply_global_affine(fix, mov, fix_spacing, mov_spacing, affine, order=1):
    """
    """

    grid = position_grid(fix.shape) * fix_spacing
    coords = affine_to_grid(affine, grid) / mov_spacing
    return interpolate_image(mov, coords, order=order)




# DASK versions for big data
def position_grid_dask(shape, blocksize):
    """
    """

    coords = da.meshgrid(*[range(x) for x in shape], indexing='ij')
    coords = da.stack(coords, axis=-1).astype(np.uint16)
    return da.rechunk(coords, chunks=tuple(blocksize + [3,]))


def affine_to_grid_dask(matrix, grid, displacement=False):
    """
    """

    ndims = len(matrix.shape)
    matrix = matrix.astype(np.float32).squeeze()
    lost_dims = ndims - len(matrix.shape)

    mm = matrix[:3, :-1]
    tt = matrix[:3, -1]
    result = da.einsum('...ij,...j->...i', mm, grid) + tt

    if displacement:
        result = result - grid

    if lost_dims > 0:
        result = result.reshape((1,)*lost_dims + result.shape)
    return result


def interpolate_image_dask(fix, mov, X, blocksize, margin, order=1, block_info=None):
    """
    """

    # resample transform to match reference dimensions
    if fix.shape[:3] != X.shape[:3]:
        X_ = np.empty(fix.shape[:3] + (3,), dtype=X.dtype)
        for i in range(3):
            X_[..., i] = zoom(X[..., i], np.array(fix.shape[:3])/X.shape[:3], order=1)
        X = X_

    # subtract offset from coordinates
    block_idx = block_info[1]['chunk-location'][:3]
    offset = np.array(block_idx) * blocksize - margin
    X = X - np.array(offset)

    # reformat: map_overlap requires matching ndims, remove before interpolation
    mov = mov.squeeze()
    X = np.moveaxis(X, -1, 0)

    # interpolate, add degenerate dimension back, and return
    result = map_coordinates(mov, X, order=order, mode='constant')
    return np.expand_dims(result, -1)


def global_affine_to_position_field(
    shape, spacing, affine, output, blocksize=[256,]*3, cluster_kwargs={},
):
    """
    """

    # get number of jobs needed
    block_grid = np.ceil(np.array(shape) / blocksize).astype(int)
    nblocks = np.prod(block_grid)

    with janelia_lsf_cluster(**cluster_kwargs) as cluster:
        cluster.scale_cluster(nblocks)

        # compute affine transform as position coordinates, lazy dask arrays
        grid = position_grid_dask(shape, blocksize) * spacing.astype(np.float32)
        coords = affine_to_grid_dask(affine, grid)
        coords = da.around(coords, decimals=2)

        # write in parallel as 4D array to zarr file
        compressor = Blosc(cname='zstd', clevel=9, shuffle=Blosc.BITSHUFFLE)
        coords_disk = zarr.open(output, 'w',
            shape=coords.shape, chunks=tuple(blocksize + [3,]),
            dtype=coords.dtype, compressor=compressor,
        )
        da.to_zarr(coords, coords_disk)

        # return pointer to zarr file
        return coords_disk


def local_affine_to_position_field(shape, spacing, local_affines, output,
    blocksize=[256,]*3, block_multiplier=[1,]*3, cluster_kwargs={},
    ):
    """
    """

    # get number of jobs needed
    block_grid = local_affines.shape[:3]
    nblocks = np.prod(block_grid)

    # need lots of RAM per worker
    cluster_kwargs["cores"] = 4

    with janelia_lsf_cluster(**cluster_kwargs) as cluster:
        cluster.scale_cluster(nblocks)

        # augment the blocksize by the fixed overlap size
        pads = [2*int(round(x/8)) for x in blocksize]
        blocksize_with_overlap = np.array(blocksize) + pads
        blocksize_with_overlap = blocksize_with_overlap * block_multiplier

        # get a grid used for each affine
        grid = position_grid_dask(blocksize_with_overlap, list(blocksize_with_overlap))
        grid = grid * spacing.astype(np.float32)

        # wrap local_affines as dask array
        local_affines_da = da.from_array(local_affines, chunks=(1, 1, 1, 3, 4))

        # compute affine transforms as position coordinates, lazy dask arrays
        coords = da.map_blocks(
            affine_to_grid_dask, local_affines_da, grid=grid, displacement=True,
            new_axis=[5,6], chunks=(1,1,1,)+tuple(grid.shape), dtype=np.float32,
        )

        # stitch affine position fields
        coords = stitch.stitch_fields(coords, blocksize)

        # crop to original shape and rechunk
        coords = coords[:shape[0], :shape[1], :shape[2]]
        coords = coords.rechunk(tuple(blocksize + [1,]))

        # convert to position field
        coords = coords + position_grid_dask(shape, blocksize) * spacing.astype(np.float32)
        coords = da.around(coords, decimals=2)

        # write in parallel as 3D array to zarr file
        compressor = Blosc(cname='zstd', clevel=9, shuffle=Blosc.BITSHUFFLE)
        coords_disk = zarr.open(output, 'w',
            shape=coords.shape, chunks=tuple(blocksize + [3,]),
            dtype=coords.dtype, compressor=compressor,
        )
        da.to_zarr(coords, coords_disk)

        # return pointer to zarr file
        return coords_disk


def apply_position_field(
    mov, mov_spacing,
    fix, fix_spacing,
    transform, output,
    blocksize=[256,]*3, order=1,
    transform_spacing=None,
    transpose=[False,]*3,
    depth=(32, 32, 32),
    cluster_kwargs={},
):
    """
    """

    # get number of jobs needed
    block_grid = np.ceil(np.array(mov.shape) / blocksize).astype(int)
    nblocks = np.prod(block_grid)

    with janelia_lsf_cluster(**cluster_kwargs) as cluster:
        cluster.scale_cluster(nblocks)

        # determine mov/fix relative chunking
        m_blocksize = blocksize * fix_spacing / mov_spacing
        m_blocksize = list(np.round(m_blocksize).astype(np.int16))
        m_depth = depth * fix_spacing / mov_spacing
        m_depth = tuple(np.round(m_depth).astype(np.int16))

        # determine trans/fix relative chunking
        if transform_spacing is not None:
            t_blocksize = blocksize * fix_spacing / transform_spacing
            t_blocksize = list(np.round(t_blocksize).astype(np.int16))
            t_depth = depth * fix_spacing / transform_spacing
            t_depth = tuple(np.round(t_depth).astype(np.int16))
        else:
            t_blocksize = blocksize
            t_depth = depth

        # wrap objects as dask arrays
        fix_da = da.from_array(fix)
        if transpose[0]:
            fix_da = fix_da.transpose(2,1,0)

        mov_da = da.from_array(mov)
        if transpose[1]:
            mov_da = mov_da.transpose(2,1,0)
            block_grid = block_grid[::-1]

        transform_da = da.from_array(transform)
        if transpose[2]:
            transform_da = transform_da.transpose(2,1,0,3)
            transform_da = transform_da[..., ::-1]

        # chunk dask arrays
        fix_da = da.reshape(fix_da, fix_da.shape + (1,)).rechunk(tuple(blocksize + [1,]))
        mov_da = da.reshape(mov_da, mov_da.shape + (1,)).rechunk(tuple(m_blocksize + [1,]))
        transform_da = transform_da.rechunk(tuple(t_blocksize + [3,]))

        # put transform in voxel units
        transform_da = transform_da / mov_spacing

        # map the interpolate function with overlaps
        # TODO: depth should be computed automatically from transform maximum?
        d = [depth+(0,), m_depth+(0,), t_depth+(0,)]
        aligned = da.map_overlap(
            interpolate_image_dask, fix_da, mov_da, transform_da,
            blocksize=m_blocksize, margin=m_depth,
            depth=d, boundary=0, dtype=np.uint16, align_arrays=False,
        )

        # remove degenerate dimension
        aligned = da.reshape(aligned, aligned.shape[:-1])

        # write in parallel as 3D array to zarr file
        compressor = Blosc(cname='zstd', clevel=9, shuffle=Blosc.BITSHUFFLE)
        aligned_disk = zarr.open(output, 'w',
            shape=aligned.shape, chunks=aligned.chunksize,
            dtype=aligned.dtype, compressor=compressor,
        )
        da.to_zarr(aligned, aligned_disk)

        # return pointer to zarr file
        return aligned_disk


def compose_position_fields(
    fields, spacing, output,
    blocksize=[256,]*3, displacement=None,
    cluster_kwargs={},
):
    """
    """

    # get number of jobs needed
    block_grid = np.ceil(np.array(fields[0].shape[:-1]) / blocksize).astype(int)
    nblocks = np.prod(block_grid)

    with janelia_lsf_cluster(**cluster_kwargs) as cluster:
        cluster.scale_cluster(nblocks)
    
        # wrap fields as dask arrays
        fields_da = da.stack([da.from_array(f, chunks=blocksize+[3,]) for f in fields])

        # accumulate
        composed = da.sum(fields_da, axis=0)

        # modify for multiple position fields
        if displacement is not None:
            raise NotImplementedError("composing displacement fields not implemented yet")
        else:
            grid = position_grid_dask(composed.shape[:3], blocksize) * spacing.astype(np.float32)
            composed = composed - (len(fields) - 1)*grid

        # write in parallel as 3D array to zarr file
        compressor = Blosc(cname='zstd', clevel=9, shuffle=Blosc.BITSHUFFLE)
        composed_disk = zarr.open(output, 'w',
            shape=composed.shape, chunks=composed.chunksize,
            dtype=composed.dtype, compressor=compressor,
        )
        da.to_zarr(composed, composed_disk)

        # return pointer to zarr file
        return composed_disk

