import os, tempfile
import numpy as np
import time
from itertools import product
from scipy.interpolate import LinearNDInterpolator
from dask.distributed import as_completed, wait
from ClusterWrap.decorator import cluster
import bigstream.utility as ut
from bigstream.align import alignment_pipeline
from bigstream.transform import apply_transform, compose_transform_list
from bigstream.transform import apply_transform_to_coordinates
from bigstream.transform import compose_transforms
from distributed import Lock, MultiLock


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
    rebalance_for_missing_neighbors=True,
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

    rebalance_for_missing_neighbors : bool (default: True)
        If True, when a block has a missing neighbor, it's linear blending weights
        are rebalanced to account for the missing neighbor. The ensures transforms
        aren't artificially dampened by missing neighbors, which have zero valued
        displacement. However, if it is desired that the edges of masked regions
        should more smoothly dampen into backgroun, then set this to false.

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

    ################  Input, output, and parameter formatting ############
    # temporary file paths and create zarr images
    temporary_directory = tempfile.TemporaryDirectory(
        prefix='.', dir=temporary_directory or os.getcwd(),
    )
    zarr_blocks = (128,) * fix.ndim
    fix_zarr_path = temporary_directory.name + '/fix.zarr'
    mov_zarr_path = temporary_directory.name + '/mov.zarr'
    fix_mask_zarr_path = temporary_directory.name + '/fix_mask.zarr'
    mov_mask_zarr_path = temporary_directory.name + '/mov_mask.zarr'
    fix_zarr = ut.numpy_to_zarr(fix, zarr_blocks, fix_zarr_path)
    mov_zarr = ut.numpy_to_zarr(mov, zarr_blocks, mov_zarr_path)
    fix_mask_zarr = None
    if fix_mask is not None:
        fix_mask_zarr = ut.numpy_to_zarr(fix_mask, zarr_blocks, fix_mask_zarr_path)
    mov_mask_zarr = None
    if mov_mask is not None:
        mov_mask_zarr = ut.numpy_to_zarr(mov_mask, zarr_blocks, mov_mask_zarr_path)

    # zarr files for initial deformations
    new_list = []
    zarr_blocks = (128,) * fix.ndim + (fix.ndim,)
    for iii, transform in enumerate(static_transform_list):
        if len(transform.shape) > fix.ndim:  # fields are always fix.ndim + 1
            path = temporary_directory.name + f'/deform{iii}.zarr'
            transform = ut.numpy_to_zarr(transform, zarr_blocks, path)
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
        # chunks will need locks for parallel writing
        locks = [Lock(f'{x}') for x in np.ndindex(*output_transform.cdata_shape)]

    # merge keyword arguments with step specific keyword arguments
    steps = [(a, {**kwargs, **b}) for a, b in steps]
    ######################################################################


    #################### Determine block structure #######################
    # determine fixed image slices for blocking
    blocksize = np.array(blocksize)
    nblocks = np.ceil(fix.shape / blocksize).astype(int)
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
    neighbor_flags = []
    neighbor_offsets = np.array(list(product([-1, 0, 1], repeat=fix.ndim)))
    for index in indices:
        flags = {tuple(o): tuple(index + o) in indices for o in neighbor_offsets}
        neighbor_flags.append(flags)

    # bundle the three parallel lists together
    block_data = list(zip(indices, slices, neighbor_flags))
    ######################################################################


    #################### The function to run on each block ###############
    def align_single_block(indices, static_transform_list):

        # parse input, log index and slices
        block_index, fix_slices, neighbor_flags = indices
        print("Block index: ", block_index, "\nSlices: ", fix_slices, flush=True)


        ########## Map fix block corners onto mov coordinates ############
        ############## Reads static transforms in the meantime ###########
        # get fixed image block corners in physical units
        fix_block_coords = []
        for corner in list(product([0, 1], repeat=len(fix_slices))):
            a = [x.stop-1 if y else x.start for x, y in zip(fix_slices, corner)]
            fix_block_coords.append(a)
        fix_block_coords = np.array(fix_block_coords)
        fix_block_coords_phys = fix_block_coords * fix_spacing

        # read static transforms: recenter affines, apply to crop coordinates
        new_list = []
        mov_block_coords_phys = np.copy(fix_block_coords_phys)
        for transform in static_transform_list[::-1]:
            if len(transform.shape) == 2:
                mov_block_coords_phys = apply_transform_to_coordinates(
                    mov_block_coords_phys, [transform,],
                )
                transform = ut.change_affine_matrix_origin(transform, fix_block_coords_phys[0])
            else:
                spacing = ut.relative_spacing(transform, fix_zarr, fix_spacing)
                ratio = np.array(transform.shape[:-1]) / fix_zarr.shape
                start = np.round( ratio * fix_block_coords[0] ).astype(int)
                stop = np.round( ratio * (fix_block_coords[-1] + 1) ).astype(int)
                transform_slices = tuple(slice(a, b) for a, b in zip(start, stop))
                transform = transform[transform_slices]
                origin = spacing * start
                mov_block_coords_phys = apply_transform_to_coordinates(
                    mov_block_coords_phys, [transform,], spacing, origin
                )
            new_list.append(transform)
        static_transform_list = new_list[::-1]

        # Now we can determine the moving image crop
        mov_block_coords = np.round(mov_block_coords_phys / mov_spacing).astype(int)
        mov_start = np.min(mov_block_coords, axis=0)
        mov_stop = np.max(mov_block_coords, axis=0)
        mov_start = np.maximum(0, mov_start)
        mov_stop = np.minimum(np.array(mov_zarr.shape)-1, mov_stop)
        mov_slices = tuple(slice(a, b) for a, b in zip(mov_start, mov_stop))

        # get moving crop origin relative to fixed crop
        mov_origin = mov_start * mov_spacing - fix_block_coords_phys[0]
        ##################################################################


        ################ Read fix and moving data ########################
        fix = fix_zarr[fix_slices]
        mov = mov_zarr[mov_slices]
        fix_mask, mov_mask = None, None
        if fix_mask_zarr is not None:
            ratio = np.array(fix_mask_zarr.shape) / fix_zarr.shape
            start = np.round( ratio * fix_block_coords[0] ).astype(int)
            stop = np.round( ratio * (fix_block_coords[-1] + 1) ).astype(int)
            fix_mask_slices = tuple(slice(a, b) for a, b in zip(start, stop))
            fix_mask = fix_mask_zarr[fix_mask_slices]
        if mov_mask_zarr is not None:
            ratio = np.array(mov_mask_zarr.shape) / mov_zarr.shape
            start = np.round( ratio * mov_start ).astype(int)
            stop = np.round( ratio * mov_stop ).astype(int)
            mov_mask_slices = tuple(slice(a, b) for a, b in zip(start, stop))
            mov_mask = mov_mask_zarr[mov_mask_slices]
        ##################################################################


        ############################ Align ###############################
        # run alignment pipeline
        transform = alignment_pipeline(
            fix, mov, fix_spacing, mov_spacing, steps,
            fix_mask=fix_mask, mov_mask=mov_mask,
            mov_origin=mov_origin,
            static_transform_list=static_transform_list,
        )

        # ensure transform is a vector field
        if len(transform.shape) == 2:
            transform = ut.matrix_to_displacement_field(
                transform, fix.shape, spacing=fix_spacing,
            )
        ##################################################################


        ################ Apply weights for linear blending ###############
        # create the standard weight array
        core = tuple(x - 2*y + 2 for x, y in zip(blocksize, overlaps))
        pad = tuple((2*y - 1, 2*y - 1) for y in overlaps)
        weights = np.pad(np.ones(core, dtype=np.float64), pad, mode='linear_ramp')

        # rebalance if any neighbors are missing
        if rebalance_for_missing_neighbors and not np.all(list(neighbor_flags.values())):

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
        for i in range(fix.ndim):
            region = [slice(None),]*fix.ndim
            if block_index[i] == 0:
                region[i] = slice(overlaps[i], None)
                weights = weights[tuple(region)]
            if block_index[i] == nblocks[i] - 1:
                region[i] = slice(None, -overlaps[i])
                weights = weights[tuple(region)]

        # crop any incomplete blocks (on the ends)
        if np.any( weights.shape != transform.shape[:-1] ):
            weights = weights[tuple(slice(0, s) for s in transform.shape[:-1])]

        # apply weights
        transform = transform * weights[..., None]
        ##################################################################


        ################ Write or return result ##########################
        # if there's no write path, just return the transform block
        if not write_path:
            return transform

        # otherwise, coordinate with neighboring blocks
        else:
            # block until locks for all write blocks are acquired
            lock_strs = []
            for delta in product((-1, 0, 1,), repeat=3):
                lock_strs.append(str(tuple(a + b for a, b in zip(block_index, delta))))
            lock = MultiLock(lock_strs)
            lock.acquire()

            # write result to disk
            print(f'WRITING BLOCK {block_index} at {time.ctime(time.time())}', flush=True)
            output_transform[fix_slices] = output_transform[fix_slices] + transform
            print(f'FINISHED WRITING BLOCK {block_index} at {time.ctime(time.time())}', flush=True)

            # release the lock
            lock.release()
            return True
        ##################################################################
    ######################################################################


    #################### Submit all blocks, parse output #################
    # submit all alignments to cluster
    futures = cluster.client.map(
        align_single_block, block_data,
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
    ######################################################################


# TODO: THIS FUNCTION CURRENTLY DOES NOT WORK FOR LARGER THAN MEMORY TRANSFORMS
@cluster
def nested_distributed_piecewise_alignment_pipeline(
    fix,
    mov,
    fix_spacing,
    mov_spacing,
    schedule,
    static_transform_list=None,
    fix_mask=None,
    mov_mask=None,
    cluster=None,
    cluster_kwargs={},
    temporary_directory=None,
    write_path=None,
    **kwargs,
):
    """
    Compose multiple calls to distributed_piecewise_alignment_pipeline with one function call.
    Be sure you understand how distributed_piecewise_alignment_pipeline and all its arguments
    work before you use this function.

    Parameters
    ----------
    fix : ndarray
        the fixed image

    mov : ndarray
        the moving image; if `static_transform_list` is None then
        `fix.shape` must equal `mov.shape`

    fix_spacing : 1d array
        The spacing in physical units (e.g. mm or um) between voxels
        of the fixed image.
        Length must equal `fix.ndim`

    mov_spacing : 1d array
        The spacing in physical units (e.g. mm or um) between voxels
        of the moving image.
        Length must equal `mov.ndim`

    schedule : list of tuples in this form [(tuple, [(str, dict), (str, dict), ...]), ...]
        The blocksize and steps arguments for each call to distributed_piecewise_alignment_pipeline
        An example:
            step1 = ( (256, 256, 256), [('affine', affine_kwargs)] )
            step2 = ( (128, 128, 128), [('affine', affine_kwargs), ('deform', deform_kwargs)] )
            schedule = [step1, step2]
        In this example, affine alignment will be done on all 256^3 blocks of the image.
        Subsequently, affine + deformable alignment will be done on all 128^3 blocks of the image.

    static_transform_list : list of numpy arrays (default: [])
        Transforms applied to moving image before applying query transform
        Assumed to have the same domain as the fixed image, though sampling
        can be different. I.e. the origin and span are the same (in physical
        units) but the number of voxels can be different.

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
        If the transforms found by this function are too large to fit into main
        process memory, set this parameter to a folder where the transforms
        can be written to disk as separate zarr files

    kwargs : any additional arguments
        Passed to `distributed_piecewise_alignment_pipeline`

    Returns
    -------
    field : nd array or zarr.core.Array
        Composition of all alignments into a single displacement vector field.
    """

    # temporary file paths and create zarr images
    temporary_directory = tempfile.TemporaryDirectory(
        prefix='.', dir=temporary_directory or os.getcwd(),
    )
    zarr_blocks = [128,] * fix.ndim
    fix_zarr_path = temporary_directory.name + '/fix.zarr'
    mov_zarr_path = temporary_directory.name + '/mov.zarr'
    fix_mask_zarr_path = temporary_directory.name + '/fix_mask.zarr'
    mov_mask_zarr_path = temporary_directory.name + '/mov_mask.zarr'
    fix_zarr = ut.numpy_to_zarr(fix, zarr_blocks, fix_zarr_path)
    mov_zarr = ut.numpy_to_zarr(mov, zarr_blocks, mov_zarr_path)
    fix_mask_zarr = None
    if fix_mask is not None: fix_mask_zarr = ut.numpy_to_zarr(fix_mask, zarr_blocks, fix_mask_zarr_path)
    mov_mask_zarr = None
    if mov_mask is not None: mov_mask_zarr = ut.numpy_to_zarr(mov_mask, zarr_blocks, mov_mask_zarr_path)

    # zarr files for initial deformations
    new_list = []
    for iii, transform in enumerate(static_transform_list):
        if transform.shape != (4, 4) and len(transform.shape) != 1:
            path = temporary_directory.name + f'/deform{iii}.zarr'
            transform = ut.numpy_to_zarr(transform, tuple(zarr_blocks) + (transform.shape[-1],), path)
        new_list.append(transform)
    static_transform_list = new_list

    # loop over the schedule
    for iii, (blocksize, steps) in enumerate(schedule):
        local_write_path = None
        if write_path: local_write_path = write_path + '/{iii}.zarr'
        deform = distributed_piecewise_alignment_pipeline(
            fix_zarr, mov_zarr, fix_spacing, mov_spacing,
            steps, blocksize,
            static_transform_list=static_transform_list,
            fix_mask=fix_mask_zarr,
            mov_mask=mov_mask_zarr,
            write_path=local_write_path,
            cluster=cluster,
            **kwargs,
        )
        # TODO: THIS DOES NOT WORK WITH LARGER THAN MEMORY TRANSFORMS
        if iii > 0:
            deform = compose_transforms(
                static_transform_list.pop(), deform,
                fix_spacing, fix_spacing,
            )
        static_transform_list.append(deform)

    # return in the requested format
    return static_transform_list.pop()

