import numpy as np
import bigstream.utility as ut
import time
import traceback

from bigstream.align import alignment_pipeline
from bigstream.transform import apply_transform_to_coordinates
from ClusterWrap.decorator import cluster
from dask.distributed import as_completed, MultiLock
from itertools import product


def _prepare_compute_block_transform_params(block_info,
                                            full_fix=None,
                                            full_mov=None,
                                            fix_spacing=None,
                                            mov_spacing=None,
                                            full_fix_mask=None,
                                            full_mov_mask=None,
                                            static_transform_list=[]):

    block_index, block_slice_coords = block_info

    print(f'{time.ctime(time.time())} Get blocks data',
          block_index, block_slice_coords,
          flush=True)

    # read fix block
    fix_block = full_fix[block_slice_coords]

    (fix_block_voxel_coords,
     fix_block_phys_coords) = _get_block_corner_coords(block_slice_coords,
                                                       fix_spacing)

    # parse initial transforms
    # recenter affines, read deforms, apply transforms to crop coordinates
    updated_block_transform_list = []
    mov_block_phys_coords = np.copy(fix_block_phys_coords)
    # traverse current transformations in reverse order
    for transform in static_transform_list[::-1]:
        mov_block_phys_coords, block_transform = _get_moving_block(
            fix_block,
            fix_spacing,
            full_fix.shape,
            fix_block_voxel_coords[0],
            fix_block_voxel_coords[-1],
            fix_block_phys_coords,
            mov_block_phys_coords,
            transform)
        updated_block_transform_list.append(block_transform)

    block_transform_list = updated_block_transform_list[::-1]  # reverse it

    # get moving image crop, read moving data
    mov_block_coords = np.round(
        mov_block_phys_coords / mov_spacing).astype(int)
    mov_start = np.min(mov_block_coords, axis=0)
    mov_stop = np.max(mov_block_coords, axis=0)
    mov_start = np.maximum(0, mov_start)
    mov_stop = np.minimum(np.array(full_mov.shape)-1, mov_stop)
    mov_slices = tuple(slice(a, b) for a, b in zip(mov_start, mov_stop))
    mov_block = full_mov[mov_slices]

    # get moving image origin
    new_origin = mov_start * mov_spacing - fix_block_phys_coords[0]

    # read masks
    fix_block_mask, mov_block_mask = None, None
    if full_fix_mask is not None:
        ratio = np.array(full_fix_mask.shape) / full_fix.shape
        fix_mask_start = np.round(ratio * fix_block_voxel_coords[0]).astype(int)
        fix_mask_stop = np.round(
            ratio * (fix_block_voxel_coords[-1] + 1)).astype(int)
        fix_mask_slices = tuple(slice(a, b)
                                for a, b in zip(fix_mask_start, fix_mask_stop))
        fix_block_mask = full_fix_mask[fix_mask_slices]

    if full_mov_mask is not None:
        ratio = np.array(full_mov_mask.shape) / full_mov.shape
        mov_mask_start = np.round(ratio * mov_start).astype(int)
        mov_mask_stop = np.round(ratio * mov_stop).astype(int)
        mov_mask_slices = tuple(slice(a, b)
                                for a, b in zip(mov_mask_start, mov_mask_stop))
        mov_block_mask = full_mov_mask[mov_mask_slices]

    print(f'{time.ctime(time.time())} Return blocks data',
          block_index, block_slice_coords,
          flush=True)

    return (block_index,
            block_slice_coords,
            fix_block,
            mov_block,
            fix_block_mask,
            mov_block_mask,
            new_origin,
            block_transform_list)


# get image block corners both in voxel and physical units
def _get_block_corner_coords(block_slice_coords, voxel_spacing):
    block_coords_list = []
    for corner in list(product([0, 1], repeat=3)):
        a = [x.stop-1 if y else x.start
             for x, y in zip(block_slice_coords, corner)]
        block_coords_list.append(a)
    block_corners_voxel_units = np.array(block_coords_list)
    block_corners_phys_units = block_corners_voxel_units * voxel_spacing
    return block_corners_voxel_units, block_corners_phys_units


def _get_moving_block(fix_block,
                      fix_spacing,
                      full_fix_shape,
                      min_fix_block_voxel_coords,
                      max_fix_block_voxel_coords,
                      fix_block_phys_coords,
                      original_mov_block_phys_coords,
                      original_transform):
    if original_transform.shape == (4, 4):
        mov_block_phys_coords = apply_transform_to_coordinates(
            original_mov_block_phys_coords,
            [original_transform,],
        )
        block_transform = ut.change_affine_matrix_origin(
            original_transform, fix_block_phys_coords[0])
    else:
        ratio = np.array(original_transform.shape[:-1]) / full_fix_shape
        start = np.round(ratio * min_fix_block_voxel_coords).astype(int)
        stop = np.round(ratio * (max_fix_block_voxel_coords + 1)).astype(int)
        transform_slices = tuple(slice(a, b)
                                 for a, b in zip(start, stop))
        block_transform = original_transform[transform_slices]
        spacing = ut.relative_spacing(block_transform, fix_block, fix_spacing)
        origin = spacing * start
        mov_block_phys_coords = apply_transform_to_coordinates(
            original_mov_block_phys_coords, [block_transform,], spacing, origin
        )
    return mov_block_phys_coords, block_transform


def _compute_block_trasform(compute_transform_params,
                            fix_spacing=None,
                            mov_spacing=None,
                            block_size=None,
                            block_overlaps=None,
                            nblocks=None,
                            align_steps=[]):
    start_time = time.time()
    (block_index,
     block_slice_coords,
     fix_block, mov_block,
     fix_block_mask, mov_block_mask,
     new_origin_phys,
     static_block_transform_list) = compute_transform_params
    print(f'{time.ctime(start_time)} Compute block transform',
          block_index,
          flush=True)

    # run alignment pipeline
    transform = alignment_pipeline(
        fix_block, mov_block,
        fix_spacing, mov_spacing,
        align_steps,
        fix_mask=fix_block_mask, mov_mask=mov_block_mask,
        mov_origin=new_origin_phys,
        static_transform_list=static_block_transform_list,
    )
    # ensure transform is a vector field
    if transform.shape == (4, 4):
        transform = ut.matrix_to_displacement_field(
            transform, fix_block.shape, spacing=fix_spacing,
        )

    weights = _get_transform_weights(block_index, 
                                     block_size,
                                     block_overlaps,
                                     nblocks)

    # handle end blocks
    if np.any(weights.shape != transform.shape[:-1]):
        crop = tuple(slice(0, s) for s in transform.shape[:-1])
        print('Crop weights for', block_index,
              'from', transform.shape, 'to', weights.shape,
              flush=True)
        weights = weights[crop]

    # apply weights
    transform = transform * weights[..., None]

    end_time = time.time()

    print(f'{time.ctime(end_time)} Finished computing block transform',
          f'in {end_time-start_time}s',
          block_index,
          flush=True)

    return block_index, block_slice_coords, transform


def _get_transform_weights(block_index,
                           block_size,
                           block_overlaps,
                           nblocks):
    print(f'{time.ctime(time.time())} Adjust transform',
          block_index,
          flush=True)

    neighbor_offsets = np.array(list(product([-1, 0, 1], repeat=3)))
    block_neighbors = {tuple(o): (all(x <= y 
                                      for x, y in zip(tuple(block_index + o),
                                                      nblocks)),)
                       for o in neighbor_offsets}

    # create the standard weights array
    core = tuple(x - 2*y + 2 for x, y in zip(block_size, block_overlaps))
    pad = tuple((2*y - 1, 2*y - 1) for y in block_overlaps)
    weights = np.pad(np.ones(core, dtype=np.float64),
                        pad, mode='linear_ramp')

    # rebalance if any neighbors are missing
    if not np.all(list(block_neighbors.values())):
        print(f'{time.ctime(time.time())} Rebalance transform weights',
            block_index,
            flush=True)
        # define overlap slices
        slices = {}
        slices[-1] = tuple(slice(0, 2*y) for y in block_overlaps)
        slices[0] = (slice(None),) * len(block_overlaps)
        slices[1] = tuple(slice(-2*y, None) for y in block_overlaps)

        missing_weights = np.zeros_like(weights)
        for neighbor, flag in block_neighbors.items():
            if not flag:
                neighbor_region = tuple(slices[-1*b][a]
                                        for a, b in enumerate(neighbor))
                region = tuple(slices[b][a]
                                for a, b in enumerate(neighbor))
                missing_weights[region] += weights[neighbor_region]

        # rebalance the weights
        weights_adjustment = 1 - missing_weights
        weights = np.divide(weights, weights_adjustment,
                            out=np.zeros_like(weights),
                            where=weights_adjustment != 0).astype(np.float32)

    # crop weights if block is on edge of domain
    block_dim = len(block_index)
    region = [slice(None),]*block_dim
    for i in range(block_dim):
        if block_index[i] == 0:
            region[i] = slice(block_overlaps[i], None)
        elif block_index[i] == nblocks[i] - 1:
            region[i] = slice(None, -block_overlaps[i])

    return weights[tuple(region)]


def _write_block_trasform(block_transform_results,
                          with_lock=False,
                          output_transform=None):
    start_time = time.time()
    (block_index,
     block_slice_coords,
     block_transform) = block_transform_results
    print(f'{time.ctime(start_time)} Write block transform results',
          block_index,
          flush=True)

    if output_transform is not None:
        lock = None
        if with_lock:
            lock_strs = []
            for delta in product((-1, 0, 1,), repeat=3):
                lock_strs.append(str(tuple(a + b 
                                           for a, b in zip(block_index, delta))))
            lock = MultiLock(lock_strs)
            lock.acquire()
            print(f'{time.ctime(start_time)} Lock acquired for writing block',
                  block_index,
                  flush=True)

        output_transform[block_slice_coords] += block_transform
        print(f'{time.ctime(time.time())} Updated vector field for block: ',
                block_index,
                'at', block_slice_coords,
                flush=True)

        if lock is not None:
            lock.release()

    end_time = time.time()
    print(f'{time.ctime(end_time)} Finished writing vector field for block: ',
            block_index,
            flush=True)

    return block_index, block_slice_coords, block_transform


def _complete_block(block_transform_results,
                    with_lock=False,
                    output_transform=None):
    (block_index, block_slice_coords, block_transform) = block_transform_results
    print(f'{time.ctime(time.time())} Completed block',
          block_index, 'block shape:' block_transform.shape,
          flush=True)
    return block_index, block_slice_coords, block_transform.shape


@cluster
def distributed_alignment_pipeline(
    fix,
    mov,
    fix_spacing,
    mov_spacing,
    steps,
    blocksize,
    overlap_factor=0.5,
    fix_mask=None,
    mov_mask=None,
    foreground_percentage=0.5,
    static_transform_list=[],
    output_transform=None,
    retries=2,
    cluster=None,
    cluster_kwargs={},
    **kwargs,
):
    """
    Piecewise alignment of moving to fixed image.
    Overlapping blocks are given to `alignment_pipeline` in parallel
    on distributed hardware. Can include random, rigid, affine, and
    deformable alignment. Inputs can be numpy or zarr arrays. Output
    is a single displacement vector field for the entire domain.
    Output can be returned to main process memory as a numpy array
    or written to disk as a zarr array.

    Parameters
    ----------
    fix : ndarray (zarr.Array)
        the fixed image

    mov : ndarray (zarr.Array)
        the moving image; `fix.shape` must equal `mov.shape`
        I.e. typically piecewise affine alignment is done after
        a global affine alignment wherein the moving image has
        been resampled onto the fixed image voxel grid.

    fix_spacing : 1d array
        The spacing in physical units (e.g. mm or um) between voxels
        of the fixed image.
        Length must equal `fix.ndim`

    mov_spacing : 1d array
        The spacing in physical units (e.g. mm or um) between voxels
        of the moving image.
        Length must equal `mov.ndim`

    steps : list of tuples in this form [(str, dict), (str, dict), ...]
        For each tuple, the str specifies which alignment to run. The options are:
        'random' : run `random_affine_search`
        'rigid' : run `affine_align` with `rigid=True`
        'affine' : run `affine_align`
        'deform' : run `deformable_align`
        For each tuple, the dict specifies the arguments to that alignment function
        Arguments specified here override any global arguments given through kwargs
        for their specific step only.

    blocksize : tuple
        Partition or block size for distributing the work

    overlap_factor : float in range [0, 1] (default: 0.5)
        Block overlap size as a percentage of block size

    fix_mask : binary ndarray (zarr.Array) (default: None)
        A mask limiting metric evaluation region of the fixed image
        Assumed to have the same domain as the fixed image, though sampling
        can be different. I.e. the origin and span are the same (in physical
        units) but the number of voxels can be different.

    mov_mask : binary ndarray (zarr.Array) (default: None)
        A mask limiting metric evaluation region of the moving image
        Assumed to have the same domain as the moving image, though sampling
        can be different. I.e. the origin and span are the same (in physical
        units) but the number of voxels can be different.

    static_transform_list : list of numpy arrays (default: [])
        Transforms applied to moving image before applying query transform
        Assumed to have the same domain as the fixed image, though sampling
        can be different. I.e. the origin and span are the same (in physical
        units) but the number of voxels can be different.

    output_transform : ndarray (default: None)
        Output transform

    write_group_interval : float (default: 30.)
        The time each of the 27 mutually exclusive write block groups have
        each round to write finished data.

    cluster : ClusterWrap.cluster object (default: None)
        Only set if you have constructed your own static cluster. The default behavior
        is to construct a cluster for the duration of this function, then close it
        when the function is finished.

    cluster_kwargs : dict (default: {})
        Arguments passed to ClusterWrap.cluster
        If working with an LSF cluster, this will be
        ClusterWrap.janelia_lsf_cluster. If on a workstation
        this will be ClusterWrap.local_cluster.
        This is how distribution parameters are specified.

    kwargs : any additional arguments
        Arguments that will apply to all alignment steps. These are overruled by
        arguments for specific steps e.g. `random_kwargs` etc.

    Returns
    -------
    field : nd array or zarr.core.Array
        Local affines stitched together into a displacement field
        Shape is `fix.shape` + (3,) as the last dimension contains
        the displacement vector.
    """

    # there's no need to convert anything to a zarr array 
    # since they are already zarr arrays

    # determine fixed image slices for blocking
    block_partition_size = np.array(blocksize)
    nblocks = np.ceil(np.array(fix.shape) / block_partition_size).astype(int)
    overlaps = np.round(block_partition_size * overlap_factor).astype(int)
    
    fix_blocks_infos = []
    for (i, j, k) in np.ndindex(*nblocks):
        start = block_partition_size * (i, j, k) - overlaps
        stop = start + block_partition_size + 2 * overlaps
        start = np.maximum(0, start)
        stop = np.minimum(fix.shape, stop)
        block_slice = tuple(slice(x, y) for x, y in zip(start, stop))

        foreground = True
        if fix_mask is not None:
            start = block_partition_size * (i, j, k)
            stop = start + block_partition_size
            ratio = np.array(fix_mask.shape) / fix.shape
            start = np.round(ratio * start).astype(int)
            stop = np.round(ratio * stop).astype(int)
            mask_crop = fix_mask[tuple(slice(a, b)
                                       for a, b in zip(start, stop))]
            if (np.sum(mask_crop) / np.prod(mask_crop.shape) <
                foreground_percentage):
                foreground = False

        if foreground:
            fix_blocks_infos.append(((i, j, k,), block_slice))

    # establish all keyword arguments
    block_align_steps = [(a, {**kwargs, **b}) for a, b in steps]

    print('Prepare params for', len(fix_blocks_infos), 'bocks', flush=True)
    blocks = cluster.client.map(_prepare_compute_block_transform_params,
                                fix_blocks_infos,
                                full_fix=fix,
                                full_mov=mov,
                                fix_spacing=fix_spacing,
                                mov_spacing=mov_spacing,
                                full_fix_mask=fix_mask,
                                full_mov_mask=mov_mask,
                                static_transform_list=static_transform_list)

    print('Submit compute transform for',
          len(blocks), 'bocks', flush=True)
    block_transform_res = cluster.client.map(_compute_block_trasform, 
                                             blocks,
                                             fix_spacing=fix_spacing,
                                             mov_spacing=mov_spacing,
                                             block_size=block_partition_size,
                                             block_overlaps=overlaps,
                                             nblocks=nblocks,
                                             align_steps=block_align_steps)

    print('Submit write tasks for',
          len(block_transform_res), 'bocks', flush=True)
    wrire_res = cluster.client.map(_write_block_trasform,
                                   block_transform_res,
                                   with_lock=True,
                                   output_transform=output_transform)

    last_exc = None
    for retry in range(retries):
        try:
            _collect_results(res)
            print(f'{time.ctime(time.time())} Distributed alignment completed successfully',
                    flush=True)
            return output_transform
        except Exception as e:
            if retry < retries -1:
                print(f'{time.ctime(time.time())} restart cluster due to exception',
                    traceback.format_exception(e),
                    flush=True)
                cluster.client.restart()
            else:
                last_exc = e

    print(f'{time.ctime(time.time())} Distributed alignment failed',
            traceback.format_exception(last_exc),
            flush=True)
    raise last_exc


def _collect_results(futures_res):
    for batch in as_completed(futures_res, with_results=True).batches():
        for _, result in batch:
            _complete_block(result)
