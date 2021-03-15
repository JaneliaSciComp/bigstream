import numpy as np
from scipy.ndimage import map_coordinates
import zarr
from numcodecs import Blosc
from bigstream import stitch
import dask.array as da
from scipy.ndimage import zoom
from ClusterWrap.clusters import janelia_lsf_cluster

WORKER_BUFFER = 8

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


def local_affines_to_position_field(
    shape, spacing, blocksize,
    local_affines,
    global_affine=None,
    write_path=None,
    lazy=True,
    cluster_kwargs={},
):
    """
    """

    # get number of jobs needed
    block_grid = local_affines.shape[:3]
    nblocks = np.prod(block_grid)
    overlap = [int(round(x/8)) for x in blocksize]

    # compose with global affine
    total_affines = np.copy(local_affines)
    if global_affine is not None:
        g = np.eye(4)
        g[:3, :] = global_affine
        for i in range(nblocks):
            x, y, z = np.unravel_index(i, block_grid)
            l = np.eye(4)
            l[:3, :] = total_affines[x, y, z]
            total_affines[x, y, z] = np.matmul(g, l)[:3, :]

    # create cluster
    with janelia_lsf_cluster(**cluster_kwargs) as cluster:
        if write_path is not None or not lazy:
            cluster.scale_cluster(nblocks + WORKER_BUFFER)

        # create position grid
        grid_da = position_grid_dask(
            np.array(blocksize)*block_grid,
            blocksize,
        )
        grid_da = grid_da * spacing.astype(np.float32)
        grid_da = grid_da[..., None]  # needed for map_overlap

        # wrap total_affines as dask array
        total_affines_da = da.from_array(
            total_affines.astype(np.float32),
            chunks=(1, 1, 1, 3, 4),
        )

        # strip off dummy axis from position grid block
        def wrapped_affine_to_grid_dask(x, y):
            y = y.squeeze()
            return affine_to_grid_dask(x, y)

        # compute affine transforms as position coordinates
        coords = da.map_overlap(
            wrapped_affine_to_grid_dask,
            total_affines_da, grid_da,
            depth=[0, tuple(overlap)+(0,)],
            boundary=0,
            trim=False,
            align_arrays=False,
            dtype=np.float32,
            new_axis=[5,6],
            chunks=(1,1,1,)+tuple(x+2*y for x, y in zip(blocksize, overlap))+(3,),
        )

        # stitch affine position fields
        coords = stitch.stitch_fields(coords, blocksize)

        # crop to original shape and rechunk
        coords = coords[:shape[0], :shape[1], :shape[2]]
        coords = coords.rechunk(tuple(blocksize + [3,]))

        # if user wants to write to disk
        if write_path is not None:
            compressor = Blosc(cname='zstd', clevel=9, shuffle=Blosc.BITSHUFFLE)
            coords_disk = zarr.open(write_path, 'w',
                shape=coords.shape, chunks=tuple(blocksize + [3,]),
                dtype=coords.dtype, compressor=compressor,
            )
            da.to_zarr(coords, coords_disk)
            return coords_disk

        # if user wants to compute and return full field
        if not lazy:
            return coords.compute()

        # if user wants to return compute graph w/o executing
        if lazy:
            return coords


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
    block_grid = np.ceil(np.array(fix.shape) / blocksize).astype(int)
    if transpose[0]:
        block_grid = block_grid[::-1]
    nblocks = np.prod(block_grid)

    # start cluster
    with janelia_lsf_cluster(**cluster_kwargs) as cluster:
        if write_path is not None or not lazy:
            cluster.scale_cluster(nblocks + WORKER_BUFFER)

        # wrap transform as dask array, define chunks
        transform_da = transform
        if not isinstance(transform, da.Array):
            transform_da = da.from_array(transform)
        if transpose[2]:
            transform_da = transform_da.transpose(2,1,0,3)
            transform_da = transform_da[..., ::-1]
        transform_da = transform_da.rechunk(tuple(blocksize + [3,]))

        # function for getting per block origins and spans
        def get_origin_and_span(t_block, mov_spacing):
            mins = t_block.min(axis=(0,1,2))
            maxs = t_block.max(axis=(0,1,2))
            os = np.empty((2,3))
            os[0] = np.maximum(0, mins - 3*mov_spacing)
            os[1] = maxs - mins + 6*mov_spacing
            return os.reshape((1,1,1,2,3))

        # get per block origins and spans
        os = da.map_blocks(
            get_origin_and_span,
            transform_da, mov_spacing=mov_spacing,
            dtype=np.float32,
            new_axis=[4,],
            chunks=(1,1,1,2,3),
        ).compute()

        # extract crop info and convert to voxel units
        origins = os[..., 0, :]
        span = np.max(os[..., 1, :], axis=(0,1,2))
        origins_vox = np.round(origins / mov_spacing).astype(int)
        span = np.ceil(span / mov_spacing).astype(int)

        # wrap moving data as dask array
        mov_da = da.from_array(mov)
        if transpose[1]:
            mov_da = mov_da.transpose(2,1,0)

        # pad moving data dask array for blocking
        pads = []
        for sh, sp in zip(mov_da.shape, span):
            diff = sp - sh % sp
            pad = (0, diff) if sh % sp > 0 else (0, 0)
            pads.append(pad)
        mov_da = da.pad(mov_da, pads)

        # construct moving blocks
        o, s, bg = origins_vox, span, block_grid
        mov_blocks = [[[mov_da[o[i,j,k,0]:o[i,j,k,0]+s[0],
                               o[i,j,k,1]:o[i,j,k,1]+s[1],
                               o[i,j,k,2]:o[i,j,k,2]+s[2]] for k in range(bg[2])]
                                                           for j in range(bg[1])]
                                                           for i in range(bg[0])]
        mov_da = da.block(mov_blocks)[..., None]
        mov_da = mov_da.rechunk(tuple(span) + (1,))

        # wrap origins as dask array, define chunks
        origins_da = da.from_array(origins)
        origins_da = origins_da.rechunk((1,1,1,3))

        # put transform in voxel units
        # position vectors are in moving coordinate system
        origins_da = origins_da / mov_spacing
        transform_da = transform_da / mov_spacing

        # wrap interpolate function
        def wrapped_interpolate_image(x, y, origin):
            x = x.squeeze()
            y = y - origin
            return interpolate_image(x, y)

        # map the interpolate function
        aligned = da.map_blocks(
            wrapped_interpolate_image,
            mov_da, transform_da, origins_da,
            dtype=np.uint16,
            chunks=tuple(blocksize),
            drop_axis=[3,],
        )

        # if user wants to write to disk
        if write_path is not None:
            compressor = Blosc(cname='zstd', clevel=9, shuffle=Blosc.BITSHUFFLE)
            aligned_disk = zarr.open(write_path, 'w',
                shape=transform_da.shape[:-1], chunks=aligned.chunksize,
                dtype=aligned.dtype, compressor=compressor,
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

    # get position field shape
    pf_shape = fix.shape
    if transpose[0]:
        pf_shape = pf_shape[::-1]

    # get task graph for local affines position field
    local_affines_pf = local_affines_to_position_field(
        pf_shape, fix_spacing, blocksize, local_affines,
        global_affine=global_affine, lazy=True,
        cluster_kwargs=cluster_kwargs,
    )

    # align
    aligned = apply_position_field(
        fix, mov, fix_spacing, mov_spacing,
        local_affines_pf, blocksize,
        write_path=write_path,
        lazy=lazy,
        transpose=transpose,
        cluster_kwargs=cluster_kwargs,
    )

    return aligned


# TODO: refactor this function
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
        cluster.scale_cluster(nblocks + WORKER_BUFFER)
    
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


