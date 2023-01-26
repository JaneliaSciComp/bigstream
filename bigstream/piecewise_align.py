import os, tempfile
import numpy as np
import time
from itertools import product
from scipy.interpolate import LinearNDInterpolator
from dask.distributed import as_completed, wait
from ClusterWrap.decorator import cluster
import bigstream.utility as ut
from bigstream.align import alignment_pipeline
from bigstream.transform import apply_transform
from bigstream.transform import apply_transform_to_coordinates
from bigstream.transform import compose_transforms


@cluster
def distributed_piecewise_alignment_pipeline(
    fix,
    mov,
    fix_spacing,
    mov_spacing,
    steps,
    blocksize,
    overlap=0.5,
    fix_mask=None,
    mov_mask=None,
    foreground_percentage=0.5,
    static_transform_list=[],
    cluster=None,
    cluster_kwargs={},
    temporary_directory=None,
    write_path=None,
    write_group_interval=30,
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
    fix : ndarray
        the fixed image

    mov : ndarray
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

    blocksize : iterable
        The shape of blocks in voxels

    overlap : float in range [0, 1] (default: 0.5)
        Block overlap size as a percentage of block size

    fix_mask : binary ndarray (default: None)
        A mask limiting metric evaluation region of the fixed image
        Assumed to have the same domain as the fixed image, though sampling
        can be different. I.e. the origin and span are the same (in physical
        units) but the number of voxels can be different.

    mov_mask : binary ndarray (default: None)
        A mask limiting metric evaluation region of the moving image
        Assumed to have the same domain as the moving image, though sampling
        can be different. I.e. the origin and span are the same (in physical
        units) but the number of voxels can be different.

    static_transform_list : list of numpy arrays (default: [])
        Transforms applied to moving image before applying query transform
        Assumed to have the same domain as the fixed image, though sampling
        can be different. I.e. the origin and span are the same (in physical
        units) but the number of voxels can be different.

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

    temporary_directory : string (default: None)
        Temporary files are created during alignment. The temporary files will be
        in their own folder within the `temporary_directory`. The default is the
        current directory. Temporary files are removed if the function completes
        successfully.

    write_path : string (default: None)
        If the transform found by this function is too large to fit into main
        process memory, set this parameter to a location where the transform
        can be written to disk as a zarr file.

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

    # temporary file paths and create zarr images
    temporary_directory = tempfile.TemporaryDirectory(
        prefix='.', dir=temporary_directory or os.getcwd(),
    )
    fix_zarr_path = temporary_directory.name + '/fix.zarr'
    mov_zarr_path = temporary_directory.name + '/mov.zarr'
    fix_mask_zarr_path = temporary_directory.name + '/fix_mask.zarr'
    mov_mask_zarr_path = temporary_directory.name + '/mov_mask.zarr'
    fix_zarr = ut.numpy_to_zarr(fix, blocksize, fix_zarr_path)
    mov_zarr = ut.numpy_to_zarr(mov, blocksize, mov_zarr_path)
    if fix_mask is not None: fix_mask_zarr = ut.numpy_to_zarr(fix_mask, blocksize, fix_mask_zarr_path)
    if mov_mask is not None: mov_mask_zarr = ut.numpy_to_zarr(mov_mask, blocksize, mov_mask_zarr_path)

    # zarr files for initial deformations
    new_list = []
    for iii, transform in enumerate(static_transform_list):
        if transform.shape != (4, 4) and len(transform.shape) != 1:
            path = temporary_directory.name + f'/deform{iii}.zarr'
            transform = ut.numpy_to_zarr(transform, tuple(blocksize) + (transform.shape[-1],), path)
        new_list.append(transform)
    static_transform_list = new_list

    # zarr file for output (if write_path is given)
    if write_path:
        output_transform = ut.create_zarr(
            write_path,
            fix.shape + (fix.ndim,),
            tuple(blocksize) + (fix.ndim,),
            np.float32,
        )

    # determine fixed image slices for blocking
    blocksize = np.array(blocksize)
    nblocks = np.ceil(np.array(fix.shape) / blocksize).astype(int)
    overlaps = np.round(blocksize * overlap).astype(int)
    indices, slices = [], []
    for (i, j, k) in np.ndindex(*nblocks):
        start = blocksize * (i, j, k) - overlaps
        stop = start + blocksize + 2 * overlaps
        start = np.maximum(0, start)
        stop = np.minimum(fix.shape, stop)
        coords = tuple(slice(x, y) for x, y in zip(start, stop))

        foreground = True
        if fix_mask is not None:
            start = blocksize * (i, j, k)
            stop = start + blocksize
            ratio = np.array(fix_mask.shape) / fix.shape
            start = np.round( ratio * start ).astype(int)
            stop = np.round( ratio * stop ).astype(int)
            mask_crop = fix_mask[tuple(slice(a, b) for a, b in zip(start, stop))]
            if not np.sum(mask_crop) / np.prod(mask_crop.shape) >= foreground_percentage:
                foreground = False

        if foreground:
            indices.append((i, j, k,))
            slices.append(coords)

    # determine foreground neighbor structure
    new_indices = []
    neighbor_offsets = np.array(list(product([-1, 0, 1], repeat=3)))
    for index, coords in zip(indices, slices):
        neighbor_flags = {tuple(o): tuple(index + o) in indices for o in neighbor_offsets}
        new_indices.append((index, coords, neighbor_flags))
    indices = new_indices

    # establish all keyword arguments
    steps = [(a, {**kwargs, **b}) for a, b in steps]

    # closure for alignment pipeline
    def align_single_block(
        indices,
        static_transform_list,
    ):

        # print some feedback
        print("Block index: ", indices[0], "\nSlices: ", indices[1], flush=True)

        # get the coordinates, read fixed data
        block_index, fix_slices, neighbor_flags = indices
        fix = fix_zarr[fix_slices]

        # get fixed image block corners in physical units
        fix_block_coords = []
        for corner in list(product([0, 1], repeat=3)):
            a = [x.stop-1 if y else x.start for x, y in zip(fix_slices, corner)]
            fix_block_coords.append(a)
        fix_block_coords = np.array(fix_block_coords)
        fix_block_coords_phys = fix_block_coords * fix_spacing

        # parse initial transforms
        # recenter affines, read deforms, apply transforms to crop coordinates
        new_list = []
        mov_block_coords_phys = np.copy(fix_block_coords_phys)
        for transform in static_transform_list[::-1]:
            if transform.shape == (4, 4):
                mov_block_coords_phys = apply_transform_to_coordinates(
                    mov_block_coords_phys, [transform,],
                )
                transform = ut.change_affine_matrix_origin(transform, fix_block_coords_phys[0])
            else:
                ratio = np.array(transform.shape[:-1]) / fix_zarr.shape
                start = np.round( ratio * fix_block_coords[0] ).astype(int)
                stop = np.round( ratio * (fix_block_coords[-1] + 1) ).astype(int)
                transform_slices = tuple(slice(a, b) for a, b in zip(start, stop))
                transform = transform[transform_slices]
                spacing = ut.relative_spacing(transform, fix, fix_spacing)
                origin = spacing * start
                mov_block_coords_phys = apply_transform_to_coordinates(
                    mov_block_coords_phys, [transform,], spacing, origin
                )
            new_list.append(transform)
        static_transform_list = new_list[::-1]

        # get moving image crop, read moving data 
        mov_block_coords = np.round(mov_block_coords_phys / mov_spacing).astype(int)
        mov_start = np.min(mov_block_coords, axis=0)
        mov_stop = np.max(mov_block_coords, axis=0)
        mov_start = np.maximum(0, mov_start)
        mov_stop = np.minimum(np.array(mov_zarr.shape)-1, mov_stop)
        mov_slices = tuple(slice(a, b) for a, b in zip(mov_start, mov_stop))
        mov = mov_zarr[mov_slices]

        # read masks
        fix_mask, mov_mask = None, None
        if os.path.isdir(fix_mask_zarr_path):
            ratio = np.array(fix_mask_zarr.shape) / fix_zarr.shape
            start = np.round( ratio * fix_block_coords[0] ).astype(int)
            stop = np.round( ratio * (fix_block_coords[-1] + 1) ).astype(int)
            fix_mask_slices = tuple(slice(a, b) for a, b in zip(start, stop))
            fix_mask = fix_mask_zarr[fix_mask_slices]
        if os.path.isdir(mov_mask_zarr_path):
            ratio = np.array(mov_mask_zarr.shape) / mov_zarr.shape
            start = np.round( ratio * mov_start ).astype(int)
            stop = np.round( ratio * mov_stop ).astype(int)
            mov_mask_slices = tuple(slice(a, b) for a, b in zip(start, stop))
            mov_mask = mov_mask_zarr[mov_mask_slices]

        # get moving image origin
        mov_origin = mov_start * mov_spacing - fix_block_coords_phys[0]

        # run alignment pipeline
        transform = alignment_pipeline(
            fix, mov, fix_spacing, mov_spacing, steps,
            fix_mask=fix_mask, mov_mask=mov_mask,
            mov_origin=mov_origin,
            static_transform_list=static_transform_list,
        )

        # ensure transform is a vector field
        if transform.shape == (4, 4):
            transform = ut.matrix_to_displacement_field(
                transform, fix.shape, spacing=fix_spacing,
            )

        # create the standard weight array
        core = tuple(x - 2*y + 2 for x, y in zip(blocksize, overlaps))
        pad = tuple((2*y - 1, 2*y - 1) for y in overlaps)
        weights = np.pad(np.ones(core, dtype=np.float64), pad, mode='linear_ramp')

        # rebalance if any neighbors are missing
        if not np.all(list(neighbor_flags.values())):

            # define overlap slices
            slices = {}
            slices[-1] = tuple(slice(0, 2*y) for y in overlaps)
            slices[0] = (slice(None),) * len(overlaps)
            slices[1] = tuple(slice(-2*y, None) for y in overlaps)

            missing_weights = np.zeros_like(weights)
            for neighbor, flag in neighbor_flags.items():
                if not flag:
                    neighbor_region = tuple(slices[-1*b][a] for a, b in enumerate(neighbor))
                    region = tuple(slices[b][a] for a, b in enumerate(neighbor))
                    missing_weights[region] += weights[neighbor_region]

            # rebalance the weights
            weights = weights / (1 - missing_weights)
            weights[np.isnan(weights)] = 0.  # edges of blocks are 0/0
            weights = weights.astype(np.float32)

        # crop weights if block is on edge of domain
        for i in range(3):
            region = [slice(None),]*3
            if block_index[i] == 0:
                region[i] = slice(overlaps[i], None)
                weights = weights[tuple(region)]
            if block_index[i] == nblocks[i] - 1:
                region[i] = slice(None, -overlaps[i])
                weights = weights[tuple(region)]

        # crop any incomplete blocks (on the ends)
        if np.any( weights.shape != transform.shape[:-1] ):
            crop = tuple(slice(0, s) for s in transform.shape[:-1])
            weights = weights[crop]

        # apply weights
        transform = transform * weights[..., None]

        # return or write the data
        if not write_path:
            return transform
        else:
            # wait until the correct write window for this write group
            # TODO: if a worker can query the set of running tasks, I may be able to skip
            #       groups that are completely written
            write_group = np.sum(np.array(block_index) % 3 * (9, 3, 1))
            while not (write_group < time.time() / write_group_interval % 27 < write_group + .5):
                time.sleep(1)
            output_transform[fix_slices] = output_transform[fix_slices] + transform
            return True
    # END CLOSURE

    # submit all alignments to cluster
    futures = cluster.client.map(
        align_single_block, indices,
        static_transform_list=static_transform_list,
    )

    # handle output for in memory and out of memory cases
    if not write_path:
        future_keys = [f.key for f in futures]
        transform = np.zeros(fix.shape + (fix.ndim,), dtype=np.float32)
        for batch in as_completed(futures, with_results=True).batches():
            for future, result in batch:
                iii = future_keys.index(future.key)
                transform[indices[iii][1]] += result
        return transform
    else:
        all_written = np.all( cluster.client.gather(futures) )
        return output_transform


# TODO: this function not yet refactored
@cluster
def nested_distributed_piecewise_alignment_pipeline(
    fix,
    mov,
    fix_spacing,
    mov_spacing,
    block_schedule,
    parameter_schedule=None,
    initial_transform_list=None,
    fix_mask=None,
    mov_mask=None,
    intermediates_path=None,
    cluster=None,
    cluster_kwargs={},
    **kwargs,
):
    """
    Nested piecewise affine alignments.
    Two levels of nesting: outer levels and inner levels.
    Transforms are averaged over inner levels and composed
    across outer levels. See the `block_schedule` parameter
    for more details.

    This method is good at capturing large bends and twists that
    cannot be captured with global rigid and affine alignment.

    Parameters
    ----------
    fix : ndarray
        the fixed image

    mov : ndarray
        the moving image; if `initial_transform_list` is None then
        `fix.shape` must equal `mov.shape`

    fix_spacing : 1d array
        The spacing in physical units (e.g. mm or um) between voxels
        of the fixed image.
        Length must equal `fix.ndim`

    mov_spacing : 1d array
        The spacing in physical units (e.g. mm or um) between voxels
        of the moving image.
        Length must equal `mov.ndim`

    block_schedule : list of lists of tuples of ints.
        Block structure for outer and inner levels.
        Tuples must all be of length `fix.ndim`

        Example:
            [ [(2, 1, 1), (1, 2, 1),],
              [(3, 1, 1), (1, 1, 2),],
              [(4, 1, 1), (2, 2, 1), (2, 2, 2),], ]

            This block schedule specifies three outer levels:
            1) This outer level contains two inner levels:
                1.1) Piecewise rigid+affine with 2 blocks along first axis
                1.2) Piecewise rigid+affine with 2 blocks along second axis
            2) This outer level contains two inner levels:
                2.1) Piecewise rigid+affine with 3 blocks along first axis
                2.2) Piecewise rigid+affine with 2 blocks along third axis
            3) This outer level contains three inner levels:
                3.1) Piecewise rigid+affine with 4 blocks along first axis
                3.2) Piecewise rigid+affine with 4 blocks total: the first
                     and second axis are each cut into 2 blocks
                3.3) Piecewise rigid+affine with 8 blocks total: all axes
                     are cut into 2 blocks

            1.1 and 1.2 are computed (serially) then averaged. This result
            is stored. 2.1 and 2.2 are computed (serially) then averaged.
            This is then composed with the result from the first level.
            This process proceeds for as many levels that are specified.

            Each instance of a piecewise rigid+affine alignment is handled
            by `distributed_piecewise_affine_alignment` and is therefore
            parallelized over blocks on distributed hardware.

    parameter_schedule : list of type dict (default: None)
        Overrides the general parameter `distributed_piecewise_affine_align`
        parameter settings for individual instances. Length of the list
        (total number of dictionaries) must equal the total number of
        tuples in `block_schedule`.

    initial_transform_list : list of ndarrays (default: None)
        A list of transforms to apply to the moving image before running
        twist alignment. If `fix.shape` does not equal `mov.shape`
        then an `initial_transform_list` must be given.

    fix_mask : binary ndarray (default: None)
        A mask limiting metric evaluation region of the fixed image

    mov_mask : binary ndarray (default: None)
        A mask limiting metric evaluation region of the moving image

    intermediates_path : string (default: None)
        Path to folder where intermediate results are written.
        The deform, transformed moving image, and transformed
        moving image mask (if given) are stored on disk as npy files.
    
    cluster_kwargs : dict (default: {})
        Arguments passed to ClusterWrap.cluster
        If working with an LSF cluster, this will be
        ClusterWrap.janelia_lsf_cluster. If on a workstation
        this will be ClusterWrap.local_cluster.
        This is how distribution parameters are specified.

    kwargs : any additional arguments
        Passed to `distributed_piecewise_affine_align`

    Returns
    -------
    field : ndarray
        Composition of all outer level transforms. A displacement vector
        field of the shape `fix.shape` + (3,) where the last dimension
        is the vector dimension.
    """

    # set working copies of moving data
    if initial_transform_list is not None:
        current_moving = apply_transform(
            fix, mov, fix_spacing, mov_spacing,
            transform_list=initial_transform_list,
        )
        current_moving_mask = None
        if mov_mask is not None:
            current_moving_mask = apply_transform(
                fix, mov_mask, fix_spacing, mov_spacing,
                transform_list=initial_transform_list,
            )
            current_moving_mask = (current_moving_mask > 0).astype(np.uint8)
    else:
        current_moving = np.copy(mov)
        current_moving_mask = None if mov_mask is None else np.copy(mov_mask)

    # initialize container and Loop over outer levels
    counter = 0  # count each call to distributed_piecewise_affine_align
    deform = np.zeros(fix.shape + (3,), dtype=np.float32)
    for outer_level, inner_list in enumerate(block_schedule):

        # initialize inner container and Loop over inner levels
        ddd = np.zeros_like(deform)
        for inner_level, nblocks in enumerate(inner_list):

            # determine parameter settings
            if parameter_schedule is not None:
                instance_kwargs = {**kwargs, **parameter_schedule[counter]}
            else:
                instance_kwargs = kwargs

            # align
            ddd += distributed_piecewise_alignment_pipeline(
                fix, current_moving,
                fix_spacing, fix_spacing,  # images should be on same grid
                nblocks=nblocks,
                fix_mask=fix_mask,
                mov_mask=current_moving_mask,
                cluster=cluster,
                cluster_kwargs=cluster_kwargs,
                **instance_kwargs,
            )

            # increment counter
            counter += 1

        # take mean
        ddd = ddd / len(inner_list)

        # if not first iteration, compose with existing deform
        if outer_level > 0:
            deform = compose_transforms(deform, ddd, fix_spacing,)
        else:
            deform = ddd

        # combine with initial transforms if given
        if initial_transform_list is not None:
            transform_list = initial_transform_list + [deform,]
        else:
            transform_list = [deform,]

        # update working copy of image
        current_moving = apply_transform(
            fix, mov, fix_spacing, mov_spacing,
            transform_list=transform_list,
        )
        # update working copy of mask
        if mov_mask is not None:
            current_moving_mask = apply_transform(
                fix, mov_mask, fix_spacing, mov_spacing,
                transform_list=transform_list,
            )
            current_moving_mask = (current_moving_mask > 0).astype(np.uint8)

        # write intermediates
        if intermediates_path is not None:
            ois = str(outer_level)
            deform_path = (intermediates_path + '/twist_deform_{}.npy').format(ois)
            image_path = (intermediates_path + '/twist_image_{}.npy').format(ois)
            mask_path = (intermediates_path + '/twist_mask_{}.npy').format(ois)
            np.save(deform_path, deform)
            np.save(image_path, current_moving)
            if mov_mask is not None:
                np.save(mask_path, current_moving_mask)

    # return deform
    return deform
    

