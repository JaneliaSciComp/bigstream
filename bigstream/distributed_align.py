import numpy as np
import bigstream.utility as ut
import time
import traceback

from functools import partial
from itertools import product
from dask.distributed import as_completed
from ClusterWrap.decorator import cluster
from bigstream.align import alignment_pipeline
from bigstream.transform import apply_transform_to_coordinates


def _get_moving_block(fix_block,
                      fix_spacing,
                      full_fix_shape,
                      fix_block_coords,
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
        start = np.round(ratio * fix_block_coords[0]).astype(int)
        stop = np.round(ratio * (fix_block_coords[-1] + 1)).astype(int)
        transform_slices = tuple(slice(a, b)
                                 for a, b in zip(start, stop))
        block_transform = original_transform[transform_slices]
        spacing = ut.relative_spacing(block_transform, fix_block, fix_spacing)
        origin = spacing * start
        mov_block_phys_coords = apply_transform_to_coordinates(
            original_mov_block_phys_coords, [block_transform,], spacing, origin
        )
    return mov_block_phys_coords, block_transform


def _get_nblocks(full_size, block_size):
    return np.ceil(np.array(full_size) / np.array(block_size)).astype(int)


def _align_single_block(block_index,
                        block_coords,
                        block_neighbors,
                        blocksize,
                        overlaps,
                        full_fix, full_mov,
                        fix_spacing, mov_spacing,
                        fix_mask, mov_mask,
                        align_steps,
                        static_transform_list,
                        result_transform,
                        write_group_interval):

    # print some feedback
    print('Align block: ', block_index,
          '\nBlock coords: ', block_coords,
          '\nBlock neighbors: ', block_neighbors,
          flush=True)

    # get the coordinates, read fixed data
    fix_block = full_fix[block_coords]

    # get fixed image block corners in physical units
    fix_block_coords = []
    for corner in list(product([0, 1], repeat=3)):
        a = [x.stop-1 if y else x.start
             for x, y in zip(block_coords, corner)]
        fix_block_coords.append(a)

    fix_block_coords = np.array(fix_block_coords)
    fix_block_phys_coords = fix_block_coords * fix_spacing

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
            fix_block_coords,
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

    # read masks
    fix_block_mask, mov_block_mask = None, None
    if fix_mask:
        ratio = np.array(fix_mask.shape) / full_fix.shape
        fix_mask_start = np.round(ratio * fix_block_coords[0]).astype(int)
        fix_mask_stop = np.round(
            ratio * (fix_block_coords[-1] + 1)).astype(int)
        fix_mask_slices = tuple(slice(a, b)
                                for a, b in zip(fix_mask_start, fix_mask_stop))
        fix_block_mask = fix_mask[fix_mask_slices]

    if mov_mask:
        ratio = np.array(mov_mask.shape) / full_mov.shape
        mov_mask_start = np.round(ratio * mov_start).astype(int)
        mov_mask_stop = np.round(ratio * mov_stop).astype(int)
        mov_mask_slices = tuple(slice(a, b)
                                for a, b in zip(mov_mask_start, mov_mask_stop))
        mov_block_mask = mov_mask[mov_mask_slices]

    # get moving image origin
    mov_origin = mov_start * mov_spacing - fix_block_phys_coords[0]

    try:
        # run alignment pipeline
        transform = alignment_pipeline(
            fix_block, mov_block,
            fix_spacing, mov_spacing,
            align_steps,
            fix_mask=fix_block_mask, mov_mask=mov_block_mask,
            mov_origin=mov_origin,
            static_transform_list=block_transform_list,
        )

        # ensure transform is a vector field
        if transform.shape == (4, 4):
            transform = ut.matrix_to_displacement_field(
                transform, fix_block.shape, spacing=fix_spacing,
            )

        print('Completed alignment for block', block_index)
    except Exception as e:
        print('Alignment pipeline failed for block', block_index,
              traceback.format_exception(e))
        return

    try:
        # create the standard weights array
        core = tuple(x - 2*y + 2 for x, y in zip(blocksize, overlaps))
        pad = tuple((2*y - 1, 2*y - 1) for y in overlaps)
        weights = np.pad(np.ones(core, dtype=np.float64),
                         pad, mode='linear_ramp')
        # rebalance if any neighbors are missing
        if not np.all(list(block_neighbors.values())):
            print('Rebalance', block_index)

            # define overlap slices
            slices = {}
            slices[-1] = tuple(slice(0, 2*y) for y in overlaps)
            slices[0] = (slice(None),) * len(overlaps)
            slices[1] = tuple(slice(-2*y, None) for y in overlaps)

            missing_weights = np.zeros_like(weights)
            for neighbor, flag in block_neighbors.items():
                if not flag:
                    neighbor_region = tuple(
                        slices[-1*b][a] for a, b in enumerate(neighbor))
                    region = tuple(slices[b][a]
                                   for a, b in enumerate(neighbor))
                    missing_weights[region] += weights[neighbor_region]

            # rebalance the weights
            weights_adjustment = 1 - missing_weights
            weights = np.divide(weights, weights_adjustment,
                                out=np.zeros_like(weights),
                                where=weights_adjustment != 0).astype(np.float32)

        # crop weights if block is on edge of domain
        nblocks = _get_nblocks(full_fix.shape, blocksize)
        region = [slice(None),]*fix_block.ndim
        for i in range(fix_block.ndim):
            if block_index[i] == 0:
                region[i] = slice(overlaps[i], None)
            elif block_index[i] == nblocks[i] - 1:
                region[i] = slice(None, -overlaps[i])
        weights = weights[tuple(region)]

        # crop any incomplete blocks (on the ends)
        if np.any(weights.shape != transform.shape[:-1]):
            crop = tuple(slice(0, s) for s in transform.shape[:-1])
            print('Crop weights for', block_index, block_coords,
                  'from', weights.shape, 'to', transform.shape)
            weights = weights[crop]

        # apply weights
        print('Apply weights for', block_index, block_coords,
              'from', weights.shape, 'to', transform.shape)
        transform = transform * weights[..., None]

        # write the data
        if result_transform:
            # wait until the correct write window for this write group
            # if a worker can query the set of running tasks, I may be able to skip
            # groups that are completely written
            write_group = np.sum(np.array(block_index) % 3 * (9, 3, 1))
            while not (write_group < time.time() / write_group_interval % 27 < write_group + .5):
                time.sleep(1)
            print('Write results for block', block_index,
                  'at', block_coords)
            result_transform[block_coords] += transform
    except Exception as e:
        print('Balancing weights failed for', block_index, block_coords,
              traceback.format_exception(e))

    return transform


def _create_single_block_align_args_from_index(block_info,
                                               blocksize,
                                               blockoverlaps,
                                               fix_vol, mov_vol,
                                               fix_spacing, mov_spacing,
                                               fix_mask, mov_mask,
                                               align_steps,
                                               transforms_list,
                                               result_transform,
                                               write_group_interval):
    return [
        block_info[0],  # block_index
        block_info[1],  # ndim tuple of block slices
        block_info[2],  # block neighbors
        blocksize,
        blockoverlaps,
        fix_vol, mov_vol,
        fix_spacing, mov_spacing,
        fix_mask, mov_mask,
        align_steps,
        transforms_list,
        result_transform,
        write_group_interval,
    ]


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
    write_group_interval=30,
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
    blocksize_array = np.array(blocksize)
    nblocks = np.ceil(np.array(fix.shape) / blocksize_array).astype(int)
    overlaps = np.round(blocksize_array * overlap_factor).astype(int)
    indices, slices = [], []
    for (i, j, k) in np.ndindex(*nblocks):
        start = blocksize_array * (i, j, k) - overlaps
        stop = start + blocksize_array + 2 * overlaps
        start = np.maximum(0, start)
        stop = np.minimum(fix.shape, stop)
        coords = tuple(slice(x, y) for x, y in zip(start, stop))

        foreground = True
        if fix_mask is not None:
            start = blocksize_array * (i, j, k)
            stop = start + blocksize_array
            ratio = np.array(fix_mask.shape) / fix.shape
            start = np.round(ratio * start).astype(int)
            stop = np.round(ratio * stop).astype(int)
            mask_crop = fix_mask[tuple(slice(a, b)
                                       for a, b in zip(start, stop))]
            if not np.sum(mask_crop) / np.prod(mask_crop.shape) >= foreground_percentage:
                foreground = False

        if foreground:
            indices.append((i, j, k,))
            slices.append(coords)

    # determine foreground neighbor structure
    new_indices = []
    neighbor_offsets = np.array(list(product([-1, 0, 1], repeat=3)))
    for index, coords in zip(indices, slices):
        neighbor_flags = {tuple(o): tuple(index + o)
                          in indices for o in neighbor_offsets}
        new_indices.append((index, coords, neighbor_flags))
    indices = new_indices

    # establish all keyword arguments
    steps = [(a, {**kwargs, **b}) for a, b in steps]

    print('Submit alignment for', len(indices), 'bocks')
    align_blocks_args = [_create_single_block_align_args_from_index(
        block_info,
        blocksize_array,
        overlaps,
        fix, mov,
        fix_spacing, mov_spacing,
        fix_mask, mov_mask,
        steps,
        static_transform_list,
        output_transform,
        write_group_interval
    ) for block_info in indices]

    print('Align', len(align_blocks_args), 'blocks')
    futures = cluster.client.map(
        _align_single_block,
        *list(zip(*align_blocks_args)),  # transpose arguments
        pure=False
    )
    future_keys = [f.key for f in futures]

    for batch in as_completed(futures, with_results=True).batches():
        for future, _ in batch:
            iii = future_keys.index(future.key)
            result_block_info = indices[iii]
            print('Completed block: ', result_block_info[0],
                    flush=True)

    return output_transform
