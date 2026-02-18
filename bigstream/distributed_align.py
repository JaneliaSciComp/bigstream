import gc
import logging
import bigstream.transform as bst
import numpy as np
import bigstream.utility as ut
import time
import traceback

from dask.distributed import as_completed, MultiLock
from itertools import product
from toolz import partition_all

from .align import alignment_pipeline
from .distutils import validate_processing_block_size,ThrottledArraySliceReader
from .image_data import (ImageData, as_image_data)
from .ome_utils import get_spatial_values


logger = logging.getLogger(__name__)


def _prepare_compute_block_spatial_transform_params(block_info,
                                                    fix_shape=None,
                                                    mov_shape=None,
                                                    fix_spacing=None,
                                                    mov_spacing=None,
                                                    fix_fullmask_shape=None,
                                                    mov_fullmask_shape=None,
                                                    mov_origin_transform=None,
                                                    static_transform_list=[]):
    logger.debug(f'Prepare block coords {block_info}')
    block_index, fix_block_coords, fix_block_neighbors = block_info
    (fix_block_voxel_coords,
     fix_block_phys_coords) = _get_spatial_block_corner_coords(fix_block_coords,
                                                               fix_spacing)
    logger.debug((
        f'Block index: {block_index} - '
        f'fix block corner physical coords: {fix_block_phys_coords} '
        f'using fix spacing = {fix_spacing}'
    ))

    # parse initial transforms
    # recenter affines, read deforms, apply transforms to crop coordinates
    updated_block_transform_list = []
    mov_block_phys_coords = np.copy(fix_block_phys_coords)
    if mov_origin_transform is not None:
        # Convert to homogeneous coordinates (n, 3) -> (n, 4)
        homogeneous = np.c_[mov_block_phys_coords, np.ones(mov_block_phys_coords.shape[0])]
        # Apply transform and extract first 3 columns
        mov_block_phys_coords = (mov_origin_transform @ homogeneous.T).T[:, :3]
    # traverse current transformations in reverse order
    for transform in static_transform_list[::-1]:
        mov_block_phys_coords, block_transform = _get_spatial_moving_block_coords(
            fix_shape,
            fix_spacing,
            fix_block_voxel_coords[0],
            fix_block_voxel_coords[-1],
            fix_block_phys_coords,
            mov_block_phys_coords,
            transform)
        updated_block_transform_list.append(block_transform)

    logger.debug((
        f'Block {block_index} :'
        f'moving block physical coords {mov_block_phys_coords}, '
        f'using moving spacing = {mov_spacing}'
    ))

    block_transform_list = updated_block_transform_list[::-1]  # reverse it

    mov_start_phys_coords = np.min(mov_block_phys_coords, axis=0)
    mov_stop_phys_coords = np.max(mov_block_phys_coords, axis=0)

    logger.debug((
        f'Block {block_index}\n'
        f'fix block start physical coords: {fix_block_voxel_coords[0]}\n'
        f'fix block stop physical coords: {fix_block_voxel_coords[-1]}\n'
        f'moving block start physical coords: {mov_start_phys_coords}\n'
        f'moving block stop physical coords: {mov_stop_phys_coords}\n'
    ))

    # get moving image origin
    new_origin = mov_start_phys_coords - fix_block_phys_coords[0]

    # get moving image crop, read moving data
    mov_block_coords = np.round(
        mov_block_phys_coords / mov_spacing).astype(int)

    logger.debug((
        f'Block {block_index} :'
        f'moving block voxel coords {mov_block_coords},'
    ))

    mov_start = np.min(mov_block_coords, axis=0)
    mov_stop = np.max(mov_block_coords, axis=0)

    logger.debug((
        f'Block {block_index} : '
        f'non-truncated moving block start (voxel coords): {mov_start} '
        f'non-truncated moving block stop (voxel coords): {mov_stop} '
    ))

    mov_start = np.maximum(0, mov_start)
    mov_stop = np.minimum(np.array(mov_shape)-1, mov_stop)
    mov_slices = tuple(slice(a, b) for a, b in zip(mov_start, mov_stop))

    logger.debug((
        f'Block {block_index} : '
        f'fix voxel coords:\n {fix_block_voxel_coords}\n'
        f'fix phys coords:\n {fix_block_phys_coords}\n'
        f'mov origin phys coords: {new_origin}, '
        f'mov block slices {mov_slices}, '
        f'mov phys coords: \n{mov_block_phys_coords}\n'
    ))

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

    logger.debug('Return blocks data: '+ 
                 f'{block_index}, {fix_block_coords}, {new_origin}')

    return (block_index,
            fix_block_coords,
            fix_block_neighbors,
            mov_slices,
            fix_blockmask_coords,
            mov_blockmask_coords,
            new_origin,
            block_transform_list)


def _read_blocks_for_processing(blocks_info,
                                fix=None,
                                mov=None,
                                fix_mask=None,
                                mov_mask=None,
                                fix_block_reader=ThrottledArraySliceReader(),
                                mov_block_reader=ThrottledArraySliceReader(),
                                mask_reader=ThrottledArraySliceReader()):
    # blocks_info is a tuple containing the fields below 
    # and the extract method knows to get the coords of the block to be read
    #    block_index,
    #    fix_block_coords,
    #    fix_block_neighbors,
    #    mov_block_coords,
    #    fix_mask_block_coords,
    #    mov_mask_block_coords,
    #    origin,
    #    block_transforms
    logger.debug(f'Read blocks: {blocks_info}')
    fix_block = _read_imagedata_block(blocks_info[1], fix, fix_block_reader)

    mov_block_coords = blocks_info[3]
    if mov_block_coords is None or any(s.stop <= 0 for s in mov_block_coords):
        logger.info(f'Moving block corresponding to {blocks_info[0]} is out of range')
        mov_block = None
    else:
        mov_block = _read_imagedata_block(mov_block_coords, mov, mov_block_reader)

    fix_mask_block = _read_imagedata_block(blocks_info[4], fix_mask, mask_reader)

    mov_mask_block_coords = blocks_info[5]
    if mov_mask_block_coords is None or any(s.stop <= 0 for s in mov_mask_block_coords):
        mov_mask_block = None
    else:
        mov_mask_block = _read_imagedata_block(mov_mask_block_coords, mov_mask, mask_reader)

    return (blocks_info,
            fix_block,
            mov_block,
            fix_mask_block,
            mov_mask_block)


def _read_imagedata_block(block_coords, image_data, image_block_reader,
                          image_timeindex=None, image_channels=None):
    image_repr = as_image_data(image_data, image_timeindex=image_timeindex,
                               image_channels=image_channels)
    if image_repr is not None:
        b = image_block_reader.read_slice(
            block_coords,
            image=image_repr.image_array,
            image_path=image_repr.image_path,
            image_subpath=image_repr.image_subpath,
            image_timeindex=image_repr.image_timeindex,
            image_channel=image_repr.image_channel,
        )
    else:
        b = None
    if b is not None:
        if np.issubdtype(b.dtype, np.floating):
            return b.astype(np.float32)
    return b


# get image block corners both in voxel and physical units
def _get_spatial_block_corner_coords(block_slice_coords, voxel_spacing):
    """
    The method returns corner coo
    """
    block_coords_list = []
    for corner in list(product([0, 1], repeat=3)):
        a = [x.stop-1 if y else x.start
             for x, y in zip(block_slice_coords, corner)]
        block_coords_list.append(a)

    block_corners_voxel_units = np.array(block_coords_list)
    block_corners_phys_units = block_corners_voxel_units * voxel_spacing
    return block_corners_voxel_units, block_corners_phys_units


def _get_spatial_moving_block_coords(fix_shape,
                                     fix_spacing,
                                     fix_block_min_voxel_coords,
                                     fix_block_max_voxel_coords,
                                     fix_block_phys_coords,
                                     original_mov_block_phys_coords,
                                     static_transform):
    if len(static_transform.shape) == 2:
        logger.debug(f'Apply affine transform {static_transform} to moving block at: {original_mov_block_phys_coords}')
        mov_block_phys_coords = bst.apply_transform_to_coordinates(
            original_mov_block_phys_coords,
            [static_transform,],
        )
        block_transform = bst.change_affine_matrix_origin(
            static_transform, fix_block_phys_coords[0])
    else:
        logger.debug(f'Apply deform field of shape {static_transform.shape} to {original_mov_block_phys_coords}')
        spacing = ut.relative_spacing(static_transform.shape,
                                      fix_shape,
                                      fix_spacing)
        ratio = np.array(static_transform.shape[:-1]) / fix_shape
        start = np.round(ratio * fix_block_min_voxel_coords).astype(int)
        stop = np.round(ratio * (fix_block_max_voxel_coords + 1)).astype(int)
        transform_slices = tuple(slice(a, b)
                                 for a, b in zip(start, stop))
        block_transform = static_transform[transform_slices]
        origin = spacing * start
        mov_block_phys_coords = bst.apply_transform_to_coordinates(
            original_mov_block_phys_coords, [block_transform,], spacing, origin
        )
    return mov_block_phys_coords, block_transform


def _compute_block_transform(compute_transform_params,
                             fix_spacing=None,
                             mov_spacing=None,
                             block_size=None,
                             block_overlaps=None,
                             nblocks=None,
                             output_transform=None,
                             foreground_percentage=0.,
                             align_steps=[]):
    start_time = time.time()
    ((block_index,
      block_coords,
      block_neighbors,
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
    logger.info((
        'Compute block transform '
        f'{block_index}: {block_coords}, {new_origin_phys} '
        f'fix shape: {fix_block.shape if fix_block is not None else 0}, '
        f'mov_shape: {mov_block.shape if mov_block is not None else 0} '
        f'using {len(static_block_transform_list)} transforms '
    ))

    # check if blocks have sufficient foreground content
    skip_alignment = False
    if fix_block is None or mov_block is None:
        logger.warning(f'Block {block_index} has no data, skipping alignment')
        skip_alignment = True
    elif foreground_percentage > 0:
        fix_fg = np.count_nonzero(fix_block) / fix_block.size
        mov_fg = np.count_nonzero(mov_block) / mov_block.size
        if fix_fg < foreground_percentage or mov_fg < foreground_percentage:
            logger.warning((
                f'Block {block_index} has insufficient foreground '
                f'(fix: {fix_fg:.4f}, mov: {mov_fg:.4f}), skipping alignment'
            ))
            skip_alignment = True

    if skip_alignment:
        # identity deform: zero displacement field
        block_shape = tuple(s.stop - s.start for s in block_coords)
        transform = np.zeros(block_shape + (len(block_coords),), dtype=np.float32)
    else:
        # run alignment pipeline
        # some pipeline algorithms use "fancy indexing" (list of tuples)
        # which is not supported yet by dask arrays
        # so in order to avoid the problem we materialize the fix and moving blocks
        transform = alignment_pipeline(
            fix_block, mov_block,
            fix_spacing, mov_spacing,
            align_steps,
            fix_mask=fix_mask_block,
            mov_mask=mov_mask_block,
            mov_origin=new_origin_phys,
            static_transform_list=static_block_transform_list,
            context=f'{block_index}',
        )
        # ensure transform is a vector field
        if len(transform.shape) == 2:
            transform = bst.matrix_to_displacement_field(
                transform, fix_block.shape, spacing=fix_spacing,
            )
        else:
            #  if it's a displacement field validate it
            _validate_transform(transform)

    # Finished computing transformation for current block_index
    logger.info((
        'Finished block alignment for '
        f'{block_index}:{block_coords} -> {transform.shape} '
    ))

    weights = _get_transform_weights(block_index, 
                                     block_size,
                                     block_overlaps,
                                     block_neighbors,
                                     nblocks)

    # handle end blocks
    if np.any(weights.shape != transform.shape[:-1]):
        crop = tuple(slice(0, s) for s in transform.shape[:-1])
        logger.debug(f'Crop weights for {block_index} ' +
                     f'from {transform.shape} to {weights.shape}')
        weights = weights[crop]

    # apply weights
    logger.debug(f'Block {block_index} :' +
                 f'Apply weights {weights.shape},' +
                 f'to transform {transform.shape}')
    transform = transform * weights[..., None]

    end_time = time.time()

    logger.debug(f'Finished computing {transform.shape} ' +
                 f'block  {block_index} transform in {end_time-start_time}s')

    if output_transform is not None:
        # acquire a lock for writing this block and block all the neighbors from writing
        lock_strs = []
        for neighbor_offset in product((-1, 0, 1), repeat=len(block_index)):
            if block_neighbors.get(neighbor_offset, False):
                lock_index = tuple(x + y for x, y in zip(block_index, neighbor_offset))
                lock_strs.append(str(lock_index))
        lock = MultiLock(lock_strs)
        lock.acquire()
        logger.debug(f'Block {block_index} lock {lock_strs}')
        try:
            # write result to disk
            logger.info(f'Writing block {block_index} at {block_coords}')
            output_block = output_transform[block_coords] + transform
            # validate it again after applying the weights
            _validate_transform(output_block)
            output_transform[block_coords] = output_block
            logger.info(f'Finished writing block {block_index} at {block_coords}')
        finally:
            # release the lock
            lock.release()
            logger.debug(f'Block {block_index} released lock {lock_strs}')

    return block_index, block_coords


def _get_transform_weights(block_index,
                           block_size,
                           block_overlaps,
                           block_neighbors,
                           nblocks):
    logger.debug(f'Adjust transform for {block_index}')

    # create the standard weights array
    core = tuple(x - 2*y + 2 for x, y in zip(block_size, block_overlaps))
    pad = tuple((2*y - 1, 2*y - 1) for y in block_overlaps)
    weights = np.pad(np.ones(core, dtype=np.float64), pad, mode='linear_ramp')

    # rebalance if any neighbors are missing
    if not np.all(list(block_neighbors.values())):
        logger.debug(f'Rebalance transform weights for {block_index}')
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
        with np.errstate(divide='ignore', invalid='ignore'):
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
        if block_index[i] == nblocks[i] - 1:
            region[i] = slice(None, -block_overlaps[i])
            weights = weights[tuple(region)]

    return weights


def _validate_transform(transform, correct=False, throw_exc=True):
    """Check displacement field for NaN/Inf values and zero them out."""
    bad = np.isnan(transform)
    if bad.any():
        count = np.count_nonzero(bad)
        logger.warning(f'Transform has {count} NaN values - replacing with 0')
        if correct:
            transform[bad] = 0.
        elif throw_exc:
            raise ValueError(f'Invalid value error found at {bad}')
    bad = np.isinf(transform)
    if bad.any():
        count = np.count_nonzero(bad)
        logger.warning(f'Transform has {count} +Inf values - replacing with 0')
        if correct:
            transform[bad] = 0.
        elif throw_exc:
            raise ValueError(f'Invalid value error found at {bad}')


def distributed_alignment_pipeline(
    fix_image:ImageData,
    mov_image:ImageData,
    steps,
    blocksize,
    cluster_client,
    overlap_factor=0.5,
    fix_mask=None,
    mov_mask=None,
    foreground_percentage=0.,
    mov_origin_transform:np.ndarray|None=None,
    static_transform_list=[],
    output_transform=None,
    max_concurrent_reads=0,
    max_cluster_jobs=0,
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
        Spatial Partition or block size for distributing the work

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

    kwargs : any additional arguments
        Arguments that will apply to all alignment steps. These are overruled by
        arguments for specific steps e.g. `random_kwargs` etc.

    Returns
    -------
    field : nd array or zarr.Array
        Local affines stitched together into a displacement field
        Shape is `fix.shape` + (3,) as the last dimension contains
        the displacement vector.
    """

    # there's no need to convert anything to a zarr array 
    # since they are already zarr arrays
    logger.info(f'Run distributed alignment with: {steps}')

    # determine fixed image slices for blocking
    # For 4-D or 5-D images we only use the spatial dimensions
    # to partition the work
    fix_spatial_dims = fix_image.spatial_dims
    mov_spatial_dims = mov_image.spatial_dims
    fix_spacing = get_spatial_values(fix_image.voxel_spacing)
    mov_spacing = get_spatial_values(mov_image.voxel_spacing)
    if fix_mask is not None:
        if isinstance(fix_mask, (tuple, list)):
            fix_mask_image = None
            fix_mask_spatial_dims = np.array(fix_mask[3:]) - np.array(fix_mask[:3])
        else:
            fix_mask_image = as_image_data(fix_mask)
            fix_mask_spatial_dims = fix_mask_image.spatial_dims
    else:
        fix_mask_image = None
        fix_mask_spatial_dims = None

    if mov_mask is not None:
        if isinstance(mov_mask, (tuple, list)):
            mov_mask_image = None
            mov_mask_spatial_dims = np.array(mov_mask[3:]) - np.array(mov_mask[:3])
        else:
            mov_mask_image = as_image_data(mov_mask)
            mov_mask_spatial_dims = mov_mask_image.spatial_dims
    else:
        mov_mask_image = None
        mov_mask_spatial_dims = None

    block_partition_size = np.array(get_spatial_values(blocksize))

    # verify output zarr chunk size <= block size to prevent race conditions
    validate_processing_block_size(output_transform, block_partition_size, reverse_output_axes=True)

    # from here on we only use spatial coordinates
    # except for the block reading where we also use the 
    # image_timeindex and the image_channel if available
    nblocks = np.ceil(np.array(fix_spatial_dims) / block_partition_size).astype(int)
    overlaps = np.round(block_partition_size * overlap_factor).astype(int)
    logger.info((
        f'Partition {fix_spatial_dims} into {nblocks} using '
        f'{block_partition_size} blocksize and {overlaps} overlaps'
    ))
    fix_blocks_ids = []
    fix_blocks_coords = []
    fix_blocks_neighbors = []
    for bi in np.ndindex(*nblocks):
        start = block_partition_size * bi - overlaps
        stop = start + block_partition_size + 2 * overlaps
        start = np.maximum(0, start)
        stop = np.minimum(fix_spatial_dims, stop)
        block_slice = tuple(slice(x, y) for x, y in zip(start, stop))

        foreground = True
        if fix_mask_image is not None:
            # mask is provided as an image
            logger.info(f'Use image mask for block {bi}')
            mask_start = block_partition_size * bi
            mask_stop = mask_start + block_partition_size
            ratio = fix_mask_spatial_dims / fix_spatial_dims
            mask_start = np.round(ratio * mask_start).astype(int)
            mask_stop = np.round(ratio * mask_stop).astype(int)
            fix_mask_block_coords = tuple(slice(a, b)
                                                for a, b in zip(mask_start, mask_stop))
            fix_mask_crop = _read_imagedata_block(fix_mask_block_coords, fix_mask_image)
            foreground_ratio = np.sum(fix_mask_crop) / np.prod(fix_mask_crop.shape)
            logger.debug(f'Block {bi} fg ratio: {foreground_ratio}')
            if foreground_ratio < foreground_percentage:
                logger.debug(f'Ignore masked block {bi}')
                foreground = False
        elif fix_mask is not None:
            # mask is provided as 6 voxel coordinates: (z_min, y_min, x_min, z_max, y_max, x_max)
            mask_min = np.array(fix_mask[:3])
            mask_max = np.array(fix_mask[3:])
            block_min = np.array([s.start for s in block_slice])
            block_max = np.array([s.stop for s in block_slice])
            # check if block intersects the mask bounding box
            if np.any(block_max <= mask_min) or np.any(block_min >= mask_max):
                logger.debug(f'Block {bi} outside mask bbox {fix_mask}, skipping')
                foreground = False

        if foreground:
            fix_blocks_ids.append(bi)
            fix_blocks_coords.append(block_slice)

    neighbor_offsets = np.array(list(product([-1, 0, 1],
                                             repeat=fix_image.spatial_ndim)))
    for block_index in fix_blocks_ids:
        block_neighbors = { tuple(o): tuple(block_index + o) in fix_blocks_ids
                                      for o in neighbor_offsets }
        fix_blocks_neighbors.append(block_neighbors)

    fix_blocks_infos = list(zip(fix_blocks_ids,
                                fix_blocks_coords,
                                fix_blocks_neighbors))

    # establish all keyword arguments
    block_align_steps = [(a, {**kwargs, **b}) for a, b in steps]

    logger.info(f'Prepare params for {len(fix_blocks_infos)} ' +
                f'blocks for a {fix_spatial_dims} volume')

    # define partial functions that will be composed in the
    # final block processing method
    def prepare_blocks_method(block_info):
        return _prepare_compute_block_spatial_transform_params(
            block_info,
            fix_shape=fix_spatial_dims,
            mov_shape=mov_spatial_dims,
            fix_spacing=fix_spacing,
            mov_spacing=mov_spacing,
            fix_fullmask_shape=fix_mask_spatial_dims,
            mov_fullmask_shape=mov_mask_spatial_dims,
            mov_origin_transform=mov_origin_transform,
            static_transform_list=static_transform_list,
        )

    def read_block_method(blocks_info):
        return _read_blocks_for_processing(
            blocks_info,
            fix=fix_image,
            mov=mov_image,
            fix_mask=fix_mask_image,
            mov_mask=mov_mask_image,
            fix_block_reader=ThrottledArraySliceReader(max_concurrent_reads),
            mov_block_reader=ThrottledArraySliceReader(max_concurrent_reads),
        )

    def compute_block_transform_method(transform_params):
        return _compute_block_transform(
            transform_params,
            fix_spacing=fix_spacing,
            mov_spacing=mov_spacing,
            block_size=block_partition_size,
            block_overlaps=overlaps,
            nblocks=nblocks,
            align_steps=block_align_steps,
            foreground_percentage=foreground_percentage,
            output_transform=output_transform
        )

    def block_processing_method(block_info):
        # compose compute_block_transform_method . read_block_method . prepare_blocks_method
        r1 = prepare_blocks_method(block_info)
        r2 = read_block_method(r1)
        return compute_block_transform_method(r2)

    if max_cluster_jobs > 0:
        logger.info(f'Split {len(fix_blocks_infos)} total blocks into partitions of up to {max_cluster_jobs} blocks')
        partitioned_fix_blocks = partition_all(max_cluster_jobs, fix_blocks_infos)
    else:
        partitioned_fix_blocks = partition_all(len(fix_blocks_infos), fix_blocks_infos)

    res = True
    for pidx, part_fix_blocks_infos in enumerate(partitioned_fix_blocks):
        logger.info(f'Process partition {pidx} ({len(part_fix_blocks_infos)} blocks)')
        blocks_transform_res = cluster_client.map(block_processing_method, part_fix_blocks_infos, pure=False)
        logger.info('Collect compute transform results for ' +
                    f'{len(blocks_transform_res)} blocks')
        part_res = _collect_results(blocks_transform_res)
        if not part_res:
            logger.warning(f'Partition {pidx} had errors while collecting block results')
            res = False
        # Clear references and collect garbage between partitions
        del blocks_transform_res
        gc.collect()

    logger.info('Distributed alignment completed successfully')
    return res


def _collect_results(futures):
    res = True

    for f, r in as_completed(futures, with_results=True):
        if f.cancelled():
            exc = f.exception()
            logger.error(f'Block exception: {exc}')
            tb = f.traceback()
            traceback.print_tb(tb)
            res = False
        else:
            bi, bc = r
            logger.debug(f'Finished computing deformation field for {bi} at {bc}')
        # Release the future to free worker memory
        f.release()

    return res
