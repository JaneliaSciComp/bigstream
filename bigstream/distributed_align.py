import functools
import numpy as np
import bigstream.utility as ut
import sys
import time
import traceback

from dask.distributed import as_completed
from itertools import product

from .align import alignment_pipeline
from .transform import apply_transform_to_coordinates
from .distributed_utils import throttle_method_invocations
from .image_data import ImageData, as_image_data
from .io_utility import read_block as io_utility_read_block


def _prepare_compute_block_transform_params(block_info,
                                            fix_shape=None,
                                            mov_shape=None,
                                            fix_spacing=None,
                                            mov_spacing=None,
                                            fix_fullmask_shape=None,
                                            mov_fullmask_shape=None,
                                            static_transform_list=[]):

    print(f'{time.ctime(time.time())} Prepare block coords',
          block_info, flush=True)

    block_index, fix_block_coords = block_info
    (fix_block_voxel_coords,
     fix_block_phys_coords) = _get_block_corner_coords(fix_block_coords,
                                                       fix_spacing)

    # parse initial transforms
    # recenter affines, read deforms, apply transforms to crop coordinates
    updated_block_transform_list = []
    mov_block_phys_coords = np.copy(fix_block_phys_coords)
    # traverse current transformations in reverse order
    for transform in static_transform_list[::-1]:
        mov_block_phys_coords, block_transform = _get_moving_block_coords(
            fix_shape,
            fix_spacing,
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
    mov_stop = np.minimum(np.array(mov_shape)-1, mov_stop)
    mov_slices = tuple(slice(a, b) for a, b in zip(mov_start, mov_stop))

    # get moving image origin
    new_origin = mov_start * mov_spacing - fix_block_phys_coords[0]

    print(f'{time.ctime(time.time())} Block {block_index} :'
          f'fix voxel coords {fix_block_voxel_coords},',
          f'fix phys coords {fix_block_phys_coords} -> ',
          f'mov coords {mov_slices},',
          f'mov phys coords {mov_block_phys_coords} -> ',
          flush=True)

    # read masks
    fix_blockmask_coords, mov_blockmask_coords = None, None
    if fix_fullmask_shape is not None:
        ratio = np.array(fix_fullmask_shape) / fix_shape
        fix_mask_start = np.round(ratio * fix_block_voxel_coords[0]).astype(int)
        fix_mask_stop = np.round(
            ratio * (fix_block_voxel_coords[-1] + 1)).astype(int)
        fix_blockmask_coords = tuple(slice(a, b)
                                     for a, b in zip(fix_mask_start,
                                                     fix_mask_stop))

    if mov_fullmask_shape is not None:
        ratio = np.array(mov_fullmask_shape) / mov_shape
        mov_mask_start = np.round(ratio * mov_start).astype(int)
        mov_mask_stop = np.round(ratio * mov_stop).astype(int)
        mov_blockmask_coords = tuple(slice(a, b)
                                    for a, b in zip(mov_mask_start,
                                                    mov_mask_stop))

    print(f'{time.ctime(time.time())} Return blocks data',
          block_index, fix_block_coords,
          flush=True)

    return (block_index,
            fix_block_coords,
            mov_slices,
            fix_blockmask_coords,
            mov_blockmask_coords,
            new_origin,
            block_transform_list)


def _read_blocks_for_processing(blocks_info,
                                fix=None,
                                mov=None,
                                fix_mask=None,
                                mov_mask=None):
    # blocks_info is a tuple containing the fields below 
    # and the extract method knows to get the coords of the block to be read
    #    block_index,
    #    fix_block_coords,
    #    mov_block_coords,
    #    fix_mask_block_coords,
    #    mov_mask_block_coords,
    #    origin,
    #    block_transforms
    print(f'{time.ctime(time.time())} '
          f'Read blocks: {blocks_info}', flush=True)
    fix_block = _read_block(blocks_info[1], fix)
    mov_block = _read_block(blocks_info[2], mov)
    fix_mask_block = _read_block(blocks_info[3], fix_mask)
    mov_mask_block = _read_block(blocks_info[4], mov_mask)

    return (blocks_info,
            fix_block,
            mov_block,
            fix_mask_block,
            mov_mask_block)


def _read_block(block_coords, image_data):
    image_repr = as_image_data(image_data)
    if image_repr is not None:
        return io_utility_read_block(block_coords,
                                     image=image_repr.image_array,
                                     image_path=image_repr.image_path,
                                     image_subpath=image_repr.image_subpath)


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


def _get_moving_block_coords(fix_shape,
                             fix_spacing,
                             fix_block_min_voxel_coords,
                             fix_block_max_voxel_coords,
                             fix_block_phys_coords,
                             original_mov_block_phys_coords,
                             original_transform):
    if len(original_transform.shape) == 2:
        mov_block_phys_coords = apply_transform_to_coordinates(
            original_mov_block_phys_coords,
            [original_transform,],
        )
        block_transform = ut.change_affine_matrix_origin(
            original_transform, fix_block_phys_coords[0])
    else:
        ratio = np.array(original_transform.shape[:-1]) / fix_shape
        start = np.round(ratio * fix_block_min_voxel_coords).astype(int)
        stop = np.round(ratio * (fix_block_max_voxel_coords + 1)).astype(int)
        transform_slices = tuple(slice(a, b)
                                 for a, b in zip(start, stop))
        block_transform = original_transform[transform_slices]
        spacing = ut.relative_spacing(block_transform.shape,
                                      fix_shape,
                                      fix_spacing)
        origin = spacing * start
        mov_block_phys_coords = apply_transform_to_coordinates(
            original_mov_block_phys_coords, [block_transform,], spacing, origin
        )
    return mov_block_phys_coords, block_transform


def _compute_block_transform(compute_transform_params,
                             fix_spacing=None,
                             mov_spacing=None,
                             block_size=None,
                             block_overlaps=None,
                             nblocks=None,
                             align_steps=[]):
    start_time = time.time()
    ((block_index,
      block_coords,
      _, # mov_block_coords,
      _, # fix_mask_block_coords,
      _, # mov_mask_block_coords,
      new_origin_phys,
      static_block_transform_list,
     ),
     fix_block,
     mov_block,
     fix_mask_block, # this can be a mask descriptor
     mov_mask_block, # this can be a mask descriptor
     ) = compute_transform_params
    print(f'{time.ctime(start_time)} Compute block transform',
          f'{block_index}: {block_coords}, {new_origin_phys}',
          f'fix shape: {fix_block.shape}, mov_shape: {mov_block.shape}',
          static_block_transform_list,
          flush=True)
    # run alignment pipeline
    # some pipeline algorithms use "fancy indexing" (list of tuples)
    # which is not supported yet by dask arrays
    # so in order to avoid the problem we materialize the fix and moving blocks
    transform = alignment_pipeline(
        fix_block, mov_block, fix_spacing, mov_spacing, align_steps,
        fix_mask=fix_mask_block,
        mov_mask=mov_mask_block,
        mov_origin=new_origin_phys,
        static_transform_list=static_block_transform_list,
    )
    # ensure transform is a vector field
    if len(transform.shape) == 2:
        transform = ut.matrix_to_displacement_field(
            transform, fix_block.shape, spacing=fix_spacing,
        )

    print(f'{time.ctime(start_time)} Finished block alignment for ',
          f'{block_index}:{block_coords} -> {transform.shape}',
          flush=True)

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

    print(f'{time.ctime(end_time)} Finished computing {transform.shape}',
            f'block  transform in {end_time-start_time}s',
            block_index,
            flush=True)

    return block_index, block_coords, transform


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
    weights = np.pad(np.ones(core, dtype=np.float64), pad, mode='linear_ramp')

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
        weights = weights / (1 - missing_weights)
        weights[np.isnan(weights)] = 0.  # edges of blocks are 0/0
        weights = weights.astype(np.float32)

    # crop weights if block is on edge of domain
    block_dim = len(block_index)
    for i in range(block_dim):
        region = [slice(None),]*block_dim
        if block_index[i] == 0:
            region[i] = slice(block_overlaps[i], None)
            weights = weights[tuple(region)]
        elif block_index[i] == nblocks[i] - 1:
            region[i] = slice(None, -block_overlaps[i])
            weights = weights[tuple(region)]

    return weights


def _write_block_transform(block_transform_future,
                           output_transform=None):
    start_time = time.time()
    (block_index,
     block_slice_coords,
     block_transform) = block_transform_future.result()
    print(f'{time.ctime(start_time)} Write block transform results',
          block_index,
          flush=True)

    if output_transform is not None:
        # write result
        output_transform[block_slice_coords] = (output_transform[block_slice_coords] +
                                                block_transform)
        print(f'{time.ctime(time.time())} Updated vector field for block: ',
                block_index,
                'at', block_slice_coords,
                flush=True)

    end_time = time.time()
    print(f'{time.ctime(end_time)} Finished writing vector field for block: ',
            block_index,
            flush=True)

    return block_index, block_slice_coords


def distributed_alignment_pipeline(
    fix_image,
    mov_image,
    fix_spacing,
    mov_spacing,
    steps,
    blocksize,
    cluster_client,
    overlap_factor=0.5,
    fix_mask=None,
    mov_mask=None,
    foreground_percentage=0.5,
    static_transform_list=[],
    output_transform=None,
    max_tasks=0,
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
    fix_image : ImageData
        the fixed image

    mov_image : ImageData
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

    cluster_client : Dask cluster client proxy
        the cluster must exists before this method is invoked

    overlap_factor : float in range [0, 1] (default: 0.5)
        Block overlap size as a percentage of block size

    fix_mask : ImageData, tuple of floats, or function (default: None)
        A mask limiting metric evaluation region of the fixed image
        If an ImageData, any non-zero value is considered foreground and any
        zero value is considered background. If a tuple of floats, any voxel
        with value in the tuple is considered background. If a function, it
        must take a single nd-array argument as input and return an array
        of the same shape as the input but with dtype bool.

        If an ImageData, it is assumed to have the same domain as the fixed
        image, though sampling can be different. I.e. the origin and span
        are the same (in phyiscal units) but the number of voxels can
        be different.

    mov_mask : ImageData, tuple of floats, or function (default: None)
        A mask limiting metric evaluation region of the moving image
        If an ImageData, any non-zero value is considered foreground and any
        zero value is considered background. If a tuple of floats, any voxel
        with value in the tuple is considered background. If a function, it
        must take a single nd-array argument as input and return an array
        of the same shape as the input but with dtype bool.

        If an ImageData, it is assumed to have the same domain as the fixed
        image, though sampling can be different. I.e. the origin and span
        are the same (in phyiscal units) but the number of voxels can
        be different.

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
    fix_shape_arr = fix_image.shape_arr
    mov_shape_arr = mov_image.shape_arr
    # fix/mov mask shape gets set only the corresponding masks
    # refer to an image array (either ImageData or ndarray)
    fix_mask_image = as_image_data(fix_mask)
    mov_mask_image = as_image_data(mov_mask)
    fix_mask_shape_arr = (fix_mask_image.shape_arr
                          if fix_mask_image is not None
                          else None)
    mov_mask_shape_arr = (mov_mask_image.shape_arr
                          if mov_mask_image is not None
                          else None)

    block_partition_size = np.array(blocksize)
    nblocks = np.ceil(np.array(fix_shape_arr) / block_partition_size).astype(int)
    overlaps = np.round(block_partition_size * overlap_factor).astype(int)
    
    fix_blocks_infos = []
    for (i, j, k) in np.ndindex(*nblocks):
        start = block_partition_size * (i, j, k) - overlaps
        stop = start + block_partition_size + 2 * overlaps
        start = np.maximum(0, start)
        stop = np.minimum(fix_shape_arr, stop)
        block_slice = tuple(slice(x, y) for x, y in zip(start, stop))

        foreground = True
        if fix_mask_image is not None:
            start = block_partition_size * (i, j, k)
            stop = start + block_partition_size
            ratio = fix_mask_shape_arr / fix_shape_arr
            start = np.round(ratio * start).astype(int)
            stop = np.round(ratio * stop).astype(int)
            fix_mask_block_coords = tuple(slice(a, b)
                                                for a, b in zip(start, stop))
            fix_mask_crop = _read_block(fix_mask_block_coords, fix_mask_image)
            if (np.sum(fix_mask_crop) / np.prod(fix_mask_crop.shape) <
                foreground_percentage):
                foreground = False

        if foreground:
            fix_blocks_infos.append(((i, j, k,), block_slice))

    # establish all keyword arguments
    block_align_steps = [(a, {**kwargs, **b}) for a, b in steps]

    print(f'{time.ctime(time.time())} Prepare params for',
          len(fix_blocks_infos), 
          f'bocks for a {fix_shape_arr} volume',
          flush=True)

    prepare_blocks_method = functools.partial(
        _prepare_compute_block_transform_params,
        fix_shape=fix_shape_arr,
        mov_shape=mov_shape_arr,
        fix_spacing=fix_spacing,
        mov_spacing=mov_spacing,
        fix_fullmask_shape=fix_mask_shape_arr,
        mov_fullmask_shape=mov_mask_shape_arr,
        static_transform_list=static_transform_list,
    )

    blocks = cluster_client.map(prepare_blocks_method, fix_blocks_infos)
    # blocks = (block_index, fix_block_coords, mov_slices,
    #           fix_blockmask_coords, mov_blockmask_coords,
    #           new_origin, block_transform_list)

    blocks_to_process = cluster_client.map(
        _read_blocks_for_processing,
        blocks,
        fix=fix_image,
        mov=mov_image,
        fix_mask=fix_mask,
        mov_mask=mov_mask,
    )

    compute_block_transform = throttle_method_invocations(
        _compute_block_transform, max_tasks)

    print(f'{time.ctime(time.time())} Submit compute transform for',
          len(blocks), 'blocks', flush=True)
    block_transform_res = cluster_client.map(compute_block_transform,
                                             blocks_to_process,
                                             fix_spacing=fix_spacing,
                                             mov_spacing=mov_spacing,
                                             block_size=block_partition_size,
                                             block_overlaps=overlaps,
                                             nblocks=nblocks,
                                             align_steps=block_align_steps)
    print(f'{time.ctime(time.time())} Collect compute transform results for',
          len(block_transform_res), 'blocks', flush=True)

    res = _collect_results(block_transform_res, output=output_transform)
    print(f'{time.ctime(time.time())} Distributed alignment completed successfully',
            flush=True)
    return res


def _collect_results(futures, output=None):
    res = True
    for f in as_completed(futures):
        if f.cancelled():
            exc = f.exception()
            print(f'{time.ctime(time.time())} Block exception:', exc,
                file=sys.stderr, flush=True)
            tb = f.traceback()
            traceback.print_tb(tb)
            res = False
        else:
            _write_block_transform(f, output_transform=output)

    return res
