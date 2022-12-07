import numpy as np
from itertools import product
import bigstream.utility as ut
import os, tempfile
from ClusterWrap.decorator import cluster
import dask.array as da
import zarr
import bigstream.transform as cs_transform


@cluster
def distributed_apply_transform(
    fix_zarr, mov_zarr,
    fix_spacing, mov_spacing,
    transform_list,
    blocksize,
    write_path=None,
    overlap=0.5,
    inverse_transforms=False,
    dataset_path=None,
    temporary_directory=None,
    cluster=None,
    cluster_kwargs={},
    **kwargs,
):
    """
    """

    # temporary file paths and ensure inputs are zarr
    temporary_directory = tempfile.TemporaryDirectory(
        prefix='.', dir=temporary_directory or os.getcwd(),
    )
    fix_zarr_path = temporary_directory.name + '/fix.zarr'
    mov_zarr_path = temporary_directory.name + '/mov.zarr'
    zarr_blocks = (128,)*fix_zarr.ndim
    fix_zarr = ut.numpy_to_zarr(fix_zarr, zarr_blocks, fix_zarr_path)
    mov_zarr = ut.numpy_to_zarr(mov_zarr, zarr_blocks, mov_zarr_path)

    # ensure all deforms are zarr
    new_list = []
    zarr_blocks = (128,)*3 + (3,)
    for iii, transform in enumerate(transform_list):
        if transform.shape != (4, 4):
            zarr_path = temporary_directory.name + f'/deform{iii}.zarr'
            transform = ut.numpy_to_zarr(transform, zarr_blocks, zarr_path)
        new_list.append(transform)
    transform_list = new_list

    # ensure transform spacing is set explicitly
    if 'transform_spacing' not in kwargs.keys():
        kwargs['transform_spacing'] = np.array(fix_spacing)
    if not isinstance(kwargs['transform_spacing'], tuple):
        kwargs['transform_spacing'] = (kwargs['transform_spacing'],) * len(transform_list)

    # get overlap and number of blocks
    blocksize = np.array(blocksize)
    overlap = np.round(blocksize * overlap).astype(int)  # NOTE: default overlap too big?
    nblocks = np.ceil(np.array(fix_zarr.shape) / blocksize).astype(int)

    # store block coordinates in a dask array
    block_coords = np.empty(nblocks, dtype=tuple)
    for (i, j, k) in np.ndindex(*nblocks):
        start = blocksize * (i, j, k) - overlap
        stop = start + blocksize + 2 * overlap
        start = np.maximum(0, start)
        stop = np.minimum(fix_zarr.shape, stop)
        block_coords[i, j, k] = tuple(slice(x, y) for x, y in zip(start, stop))
    block_coords = da.from_array(block_coords, chunks=(1,)*block_coords.ndim)

    # pipeline to run on each block
    def transform_single_block(coords, transform_list):

        # fetch fixed image slices and read fix
        fix_slices = coords.item()
        fix = fix_zarr[fix_slices]
        fix_origin = fix_spacing * [s.start for s in fix_slices]

        # read relevant region of transforms
        new_list = []
        transform_origin = [fix_origin,] * len(transform_list)
        shape = fix_zarr.shape if not inverse_transforms else mov_zarr.shape
        for iii, transform in enumerate(transform_list):
            if transform.shape != (4, 4):
                start = np.floor(fix_origin / kwargs['transform_spacing'][iii]).astype(int)
                stop = [s.stop for s in fix_slices] * fix_spacing / kwargs['transform_spacing'][iii]
                stop = np.ceil(stop).astype(int)
                transform = transform[tuple(slice(a, b) for a, b in zip(start, stop))]
                transform_origin[iii] = start * kwargs['transform_spacing'][iii]
            new_list.append(transform)
        transform_list = new_list
        transform_origin = tuple(transform_origin)

        # transform fixed block corners, read moving data
        fix_block_coords = []
        for corner in list(product([0, 1], repeat=3)):
            a = [x.stop-1 if y else x.start for x, y in zip(fix_slices, corner)]
            fix_block_coords.append(a)
        fix_block_coords = np.array(fix_block_coords) * fix_spacing
        mov_block_coords = cs_transform.apply_transform_to_coordinates(
            fix_block_coords, transform_list, kwargs['transform_spacing'], transform_origin,
        )
        mov_block_coords = np.round(mov_block_coords / mov_spacing).astype(int)
        mov_block_coords = np.maximum(0, mov_block_coords)
        mov_block_coords = np.minimum(mov_zarr.shape, mov_block_coords)
        mov_start = np.min(mov_block_coords, axis=0)
        mov_stop = np.max(mov_block_coords, axis=0)
        mov_slices = tuple(slice(a, b) for a, b in zip(mov_start, mov_stop))
        mov = mov_zarr[mov_slices]
        mov_origin = mov_spacing * [s.start for s in mov_slices]

        # resample
        aligned = cs_transform.apply_transform(
            fix, mov, fix_spacing, mov_spacing,
            transform_list=transform_list,
            transform_origin=transform_origin,
            fix_origin=fix_origin,
            mov_origin=mov_origin,
            **kwargs,
        )

        # crop out overlap
        for axis in range(aligned.ndim):

            # left side
            slc = [slice(None),]*aligned.ndim
            if fix_slices[axis].start != 0:
                slc[axis] = slice(overlap[axis], None)
                aligned = aligned[tuple(slc)]

            # right side
            slc = [slice(None),]*aligned.ndim
            if aligned.shape[axis] > blocksize[axis]:
                slc[axis] = slice(None, blocksize[axis])
                aligned = aligned[tuple(slc)]

        # return result
        return aligned
    # END: closure

    # align all blocks
    aligned = da.map_blocks(
        transform_single_block,
        block_coords,
        transform_list=transform_list,
        dtype=fix_zarr.dtype,
        chunks=blocksize,
    )

    # crop to original size
    aligned = aligned[tuple(slice(s) for s in fix_zarr.shape)]

    # return
    if write_path:
        da.to_zarr(aligned, write_path, component=dataset_path)
        return zarr.open(write_path, 'r+')
    else:
        return aligned.compute()


@cluster
def distributed_apply_transform_to_coordinates(
    coordinates,
    transform_list,
    partition_size=30.,
    transform_spacing=None,
    transform_origin=None,
    temporary_directory=None,
    cluster=None,
    cluster_kwargs={},
):
    """
    """

    # TODO: check this for multiple deforms and transform_origin as a list

    # ensure temporary directory exists
    temporary_directory = temporary_directory or os.getcwd()
    temporary_directory = tempfile.TemporaryDirectory(
        prefix='.', dir=temporary_directory,
    )

    # ensure all deforms are zarr
    new_list = []
    zarr_blocks = (128,)*3 + (3,)  # TODO: generalize
    for iii, transform in enumerate(transform_list):
        if transform.shape != (4, 4):
            zarr_path = temporary_directory.name + f'/deform{iii}.zarr'
            transform = ut.numpy_to_zarr(transform, zarr_blocks, zarr_path)
        new_list.append(transform)
    transform_list = new_list

    # determine partitions of coordinates
    origin = np.min(coordinates, axis=0)
    nblocks = np.max(coordinates, axis=0) - origin
    nblocks = np.ceil(nblocks / partition_size).astype(int)
    partitions = []
    for (i, j, k) in np.ndindex(*nblocks):
        lower_bound = origin + partition_size * np.array((i, j, k))
        upper_bound = lower_bound + partition_size
        not_too_low = np.all(coordinates >= lower_bound, axis=1)
        not_too_high = np.all(coordinates < upper_bound, axis=1)
        coords = coordinates[ not_too_low * not_too_high ]
        if coords.size != 0: partitions.append(coords)

    def transform_partition(coordinates, transform_list):

        # read relevant region of transform
        a = np.min(coordinates, axis=0)
        b = np.max(coordinates, axis=0)
        new_list = []
        for iii, transform in enumerate(transform_list):
            if transform.shape != (4, 4):
                spacing = transform_spacing
                if isinstance(spacing, tuple): spacing = spacing[iii]
                start = np.floor(a / spacing).astype(int)
                stop = np.ceil(b / spacing).astype(int) + 1
                crop = tuple(slice(x, y) for x, y in zip(start, stop))
                transform = transform[crop]
            new_list.append(transform)
        transform_list = new_list

        # apply transforms
        return cs_transform.apply_transform_to_coordinates(
            coordinates, transform_list,
            transform_spacing,
            transform_origin=a,
        )

    # transform all partitions and return
    futures = cluster.client.map(
        transform_partition, partitions,
        transform_list=transform_list,
    )
    results = cluster.client.gather(futures)

    return np.concatenate(results, axis=0)


@cluster
def distributed_invert_displacement_vector_field(
    field_zarr,
    spacing,
    blocksize,
    write_path,
    iterations=10,
    order=2,
    sqrt_iterations=5,
    cluster=None,
    cluster_kwargs={},
):
    """
    """

    # get overlap and number of blocks
    blocksize = np.array(blocksize)
    overlap = np.round(blocksize * 0.25).astype(int)
    nblocks = np.ceil(np.array(field_zarr.shape[:-1]) / blocksize).astype(int)

    # store block coordinates in a dask array
    block_coords = np.empty(nblocks, dtype=tuple)
    for (i, j, k) in np.ndindex(*nblocks):
        start = blocksize * (i, j, k) - overlap
        stop = start + blocksize + 2 * overlap
        start = np.maximum(0, start)
        stop = np.minimum(field_zarr.shape[:-1], stop)
        coords = tuple(slice(x, y) for x, y in zip(start, stop))
        block_coords[i, j, k] = coords
    block_coords = da.from_array(block_coords, chunks=(1,)*block_coords.ndim)


    def invert_block(slices):

        slices = slices.item()
        field = field_zarr[slices]
        inverse = cs_transform.invert_displacement_vector_field(
            field, spacing, iterations, order, sqrt_iterations,
        )
        
        # crop out overlap
        for axis in range(inverse.ndim - 1):

            # left side
            slc = [slice(None),]*(inverse.ndim - 1)
            if slices[axis].start != 0:
                slc[axis] = slice(overlap[axis], None)
                inverse = inverse[tuple(slc)]

            # right side
            slc = [slice(None),]*(inverse.ndim - 1)
            if inverse.shape[axis] > blocksize[axis]:
                slc[axis] = slice(None, blocksize[axis])
                inverse = inverse[tuple(slc)]

        # return result
        return inverse

    # invert all blocks
    inverse = da.map_blocks(
        invert_block,
        block_coords,
        dtype=field_zarr.dtype,
        new_axis=[3,],
        chunks=tuple(blocksize) + (3,),
    )

    # crop to original size
    inverse = inverse[tuple(slice(0, s) for s in field_zarr.shape[:-1])]

    # compute result, write to zarr array
    da.to_zarr(inverse, write_path)

    # return reference to result
    return zarr.open(write_path, 'r+')

