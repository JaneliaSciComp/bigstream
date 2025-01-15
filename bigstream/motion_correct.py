import numpy as np
import os, json, tempfile, inspect
from ClusterWrap.decorator import cluster
import dask.array as da
from dask.distributed import as_completed
import bigstream.utility as ut
from bigstream.configure_irm import configure_irm
from bigstream.align import alignment_pipeline
from bigstream.transform import apply_transform
from bigstream.transform import compose_transforms
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom
from skimage.exposure import match_histograms
from networkx import DiGraph, shortest_path
import zarr


@cluster
def distributed_image_mean(
    fix_array,
    axis=0,
    cluster=None,
    cluster_kwargs={},
):
    """
    Voxelwise mean over images in a zarr array.
    Distributed with dask.

    Parameters
    ----------
    fix_array : zarr.Array
        The set of images over which you want to take a mean

    axis : int (default: 0)
        The axis over which you want to take the mean

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

    Returns
    -------
    image_mean : ndarray
        The voxelwise mean fix_array over axis
    """

    fix_array = da.from_zarr(fix_array)
    mean = fix_array.mean(axis=axis, dtype=np.float32).compute()
    return np.round(mean).astype(fix_array.dtype)


@cluster
def distributed_piecewise_image_mean(
    images_list,
    blocksize,
    write_path,
    cluster=None,
    cluster_kwargs={},
):
    """
    """

    # create and output zarr array
    output = ut.create_zarr(
        write_path,
        images_list[0].shape,
        images_list[0].chunks,
        images_list[0].dtype,
    )

    # define compute block coordinates
    blocksize = np.array(blocksize)
    nblocks = np.ceil( np.array(images_list[0].shape) / blocksize ).astype(int)
    crops = []
    for (i, j, k) in np.ndindex(*nblocks):
        start = blocksize * (i, j, k)
        stop = start + blocksize
        stop = np.minimum(images_list[0].shape, stop)
        crops.append( tuple(slice(x, y) for x, y in zip(start, stop)) )

    # define what to do on each block
    def average_block(crop):
        n_images = float(len(images_list))
        reference = images_list[0][crop]
        mean = reference / n_images
        for image in images_list[1:]:
            image = match_histograms(image[crop], reference)
            mean += image / n_images
        # TODO: rounding assumes an integer data type
        output[crop] = np.round(mean).astype(images_list[0].dtype)
        return True

    # run it
    futures = cluster.client.map(average_block, crops)
    all_written = np.all( cluster.client.gather(futures) )
    return output


@cluster
def motion_correct(
    target,
    timeseries,
    target_spacing,
    timeseries_spacing,
    steps,
    mask=None,
    cluster=None,
    cluster_kwargs={},
    temporary_directory=None,
    write_path=None,
    static_transforms=None,
    **kwargs,
):
    """
    Motion correct a timeseries by aligning every frame to a single fixed target.
    `bigstream.align.alignment_pipeline` is called to align each frame to the target.
    Control the alignment done for each pair of frames through `steps`.

    Parameters
    ----------
    target : ndarray
        The fixed image reference. All frames will be aligned to this image.
        This should be an in-memory numpy array.

    timeseries : zarr.Array
        The timeseries image frames. This should be a 4D zarr Array object that is
        only lazy loaded, e.g. not in memory. The time axis should be the first
        axis.

    target_spacing : 1d array
        The voxel spacing of the fixed reference image in micrometers

    timeseries_spacing : 1d array
        The voxel spacing of the moving images in micrometers

    steps : list of tuples in this form [(str, dict), (str, dict), ...]
        For each tuple, the str specifies which alignment to run. The options are:
        'ransac' : run `feature_point_ransac_affine_align`
        'random' : run `random_affine_search`
        'rigid' : run `affine_align` with `rigid=True`
        'affine' : run `affine_align`
        'deform' : run `deformable_align`
        For each tuple, the dict specifies the arguments to that alignment function.
        Arguments specified here override any global arguments give through kwargs
        for their specific step only.

    mask : binary ndarray
        Limits computation of the image matching metric to a region within the fixed image

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
        Temporary files may be created during alignment. The temporary files will be
        in their own folder within the `temporary_directory`. The default is the
        current directory. Temporary files are removed if the function completes
        successfully.

    write_path : string (default: None)
        If None then all transforms are returned in memory as a numpy array.
        If not None the array of transforms will be written to disk as a zarr array
        at this path.

    static_transforms : nd-array or zarr array (default: None)
        A transform to apply to each moving frame before aligning to target

    kwargs : any additional arguments
        Passed to alignment_pipeline. Control the nature of alignments through these arguments

    Returns
    -------
    transforms : nd-array or zarr array, see write_path
        All transforms aligning the timeseries frames to the target
    """

    # put timeseries (as zarr), target, and mask, in temporary directory
    temporary_directory = temporary_directory or os.getcwd()
    temporary_directory = tempfile.TemporaryDirectory(prefix='.', dir=temporary_directory)
    zarr_blocks = (1,) + timeseries.shape[1:]
    timeseries_zarr_path = temporary_directory.name + '/timeseries.zarr'
    timeseries_zarr = ut.numpy_to_zarr(timeseries, zarr_blocks, timeseries_zarr_path)
    np.save(temporary_directory.name + '/target.npy', target)
    if mask is not None: np.save(temporary_directory.name + '/mask.npy', mask)

    static_transforms_zarr = static_transforms
    if static_transforms_zarr is not None:
        zarr_blocks = (1,) + static_transforms.shape[1:]
        st_zarr_path = temporary_directory.name + '/static_transforms.zarr'
        static_transforms_zarr = ut.numpy_to_zarr(static_transforms, zarr_blocks, st_zarr_path)

    # format output
    output_shape = (timeseries.shape[0], 4, 4)
    if 'deform' in [x[0] for x in steps]:
        output_shape = (timeseries.shape[0],) + target.shape + (target.ndim,)
    if write_path:
        zarr_blocks = (1,) + output_shape[1:]
        output = ut.create_zarr(write_path, output_shape, zarr_blocks, np.float32)

    # define alignments
    steps = [(a, {**kwargs, **b}) for a, b in steps]
    def align_frame_pair(index):
        fix = np.load(temporary_directory.name + '/target.npy')
        mov = timeseries_zarr[index]
        if os.path.isfile(temporary_directory.name + '/mask.npy'):
            mask = np.load(temporary_directory.name + '/mask.npy')
        static_transform_list = []
        if static_transforms_zarr is not None:
            static_transform_list.append(static_transforms_zarr[index])
        transform = alignment_pipeline(
            fix, mov, target_spacing, timeseries_spacing,
            steps, fix_mask=mask,
            static_transform_list=static_transform_list,
        )
        if write_path:
            output[index] = transform
            return True
        return transform

    # submit all alignments
    futures = cluster.client.map(align_frame_pair, range(timeseries.shape[0]))
    if not write_path:
        output = np.empty(output_shape, dtype=np.float32)
        future_keys = [f.key for f in futures]
        for batch in as_completed(futures, with_results=True).batches():
            for future, result in batch:
                output[future_keys.index(future.key)] = result
    else:
        all_written = np.all(cluster.client.gather(futures))
    return output


@cluster
def delta_motion_correct(
    timeseries,
    spacing,
    steps,
    strides=(1,),
    stride_relaxation_radius=None,
    mask=None,
    cluster=None,
    cluster_kwargs={},
    temporary_directory=None,
    write_path=None,
    **kwargs,
):
    """
    Motion correct a time series by aligning each frame to the one before it.
    Frames at times T are fixed frames; frames at times T+1 are moving frames.
    `bigstream.align.alignment_pipeline` is called for each pair of frames.
    Control the alignment done for each pair of frames through `steps`.

    Parameters
    ----------
    timeseries : zarr.Array
        The timeseries image frames. This should be a 4D zarr Array object that is
        only lazy loaded, e.g. not in memory. The time axis should be the first
        axis.

    spacing : 1d array
        The voxel spacing of the frames in micrometers

    steps : list of tuples in this form [(str, dict), (str, dict), ...]
        For each tuple, the str specifies which alignment to run. The options are:
        'ransac' : run `feature_point_ransac_affine_align`
        'random' : run `random_affine_search`
        'rigid' : run `affine_align` with `rigid=True`
        'affine' : run `affine_align`
        'deform' : run `deformable_align`
        For each tuple, the dict specifies the arguments to that alignment function.
        Arguments specified here override any global arguments given through kwargs
        for their specific step only.

    strides : tuple of strictly increasing positive integers (default: (1,))
        The set of time strides used for registration pairs. Best described with an
        example. Suppose frames are labeled T0, T1, T2, and so on, TM-->TN indicates
        frame TM was registered as the moving image to fixed frame TN, and
        strides==(1,5,10). The 1 indicates the following registrations will be done:
        T1-->T0, T2-->T1, T3-->T2, and so on. The 5 indicates the following additional
        registrations will be done: T5-->T0, T10-->T5, T15-->T10, and so on. The 10
        indicates the following additional registrations will be done: T10-->T0,
        T20-->T10, T30-->T20, and so on. Higher level strides facilitate cumulative
        composition of transforms over long time intervals with less error.

        The exact stride intervals can be relaxed and replaced with a local search
        for optimal frame pairs by using stride_relaxation_radius.

    stride_relaxation_radius : float in interval (0, 1) (default: None)
        For fixed frame TF and stride N, an optimal moving frame is found in the range:
        [TF + N*(1 - stride_relaxation_radius), TF + N*(1 + stride_relaxation_radius)]
        rounding to the nearest integers. The chosen frame has the optimal image
        metric value with respect to TF of all frames in the range. The image metric
        used, and its parameters if any, are taken from the steps list. If multiple
        steps are defined with different metrics, values are taken from the end of
        the steps list. If None, no stride relaxation is used, i.e. the exact stride N
        determines the registration pairs.

        Stride interval relaxation helps identify frame pairs with smaller motion
        differences, minimizing expected registration error.
        
    mask : nd-array (default: None)
        This array should have the same dimensions and shape as a single frame of
        the timeseries. The mask will be used as the fixed and moving image mask
        for all frame pair alignments. A good way to make this is to use
        bigstream.level_set.foreground_segmentation on the max projection of all
        frames over time.

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
        Temporary files may be created during alignment. The temporary files will be
        in their own folder within the `temporary_directory`. The default is the
        current directory. Temporary files are removed if the function completes
        successfully.

    write_path : string (default: None)
        TODO: update for new write behavior with strides
        If not None, the final array of transforms will be written to disk as a zarr
        array at this path. If None then this array is returned in memory. Only use
        this if transforms are deformation and if the resultant set of transforms
        is too large to fit in memory. 

    kwargs : any additional arguments
        Passed to alignment_pipeline. Control the nature of alignments through these arguments

    Returns
    -------
    transforms : nd-array
        TODO: update for new return values with strides
        All transforms aligning the dataset to the reference. N == number of provided images
    """

    # useful constants
    nframes = timeseries.shape[0]
    frame_shape = timeseries.shape[1:]

    # reconcile kwargs and steps
    steps = [(a, {**kwargs, **b}) for a, b in steps]

    # put timeseries (as zarr), target, and mask, in temporary directory
    temporary_directory = temporary_directory or os.getcwd()
    temporary_directory = tempfile.TemporaryDirectory(prefix='.', dir=temporary_directory)
    zarr_blocks = (1,) + frame_shape
    timeseries_zarr_path = temporary_directory.name + '/timeseries.zarr'
    timeseries_zarr = ut.numpy_to_zarr(timeseries, zarr_blocks, timeseries_zarr_path)
    if mask is not None: np.save(temporary_directory.name + '/mask.npy', mask)

    # determine registration pairs if there is no stride relaxation
    if stride_relaxation_radius is None:
        index_lists = [list(range(0, nframes, x)) for x in strides]
        index_lists = [x + [nframes-1,] if x[-1] < nframes-1 else x for x in index_lists]
        index_lists = [np.array(x) for x in index_lists]

    # determine registration pairs if there is stride relaxation
    if stride_relaxation_radius is not None and np.any(np.array(strides, dtype=int) > 1):

        # find all possible registration pairs, largest strides first
        edge_matrices = []
        for stride in strides[::-1]:
            upstream_nodes = [0,]
            edge_matrix = np.zeros((nframes, nframes), dtype=bool)
            while upstream_nodes:
                upstream_node = upstream_nodes[0]
                if upstream_node == nframes-1: break
                mn = upstream_node + round(stride*(1 - stride_relaxation_radius))
                mx = upstream_node + round(stride*(1 + stride_relaxation_radius))
                mn = min(nframes-1, mn)
                mx = min(nframes-1, mx)
                edge_matrix[upstream_node, mn:mx+1] = True
                mn = max(mn, upstream_nodes[-1]+1)
                if mx >= mn: upstream_nodes += range(mn, mx+1)
                upstream_nodes.pop(0)
            edge_matrices.append(edge_matrix)

        # define how and then get metric matrix
        cirm_kwargs = inspect.getfullargspec(configure_irm)[0]
        cirm_kwargs = {x:y for s in steps for x, y in s[1].items() if x in cirm_kwargs}
        def metric_matrix_row(row_index, columns_mask):
            values = np.zeros(nframes, dtype=np.float64)
            irm = configure_irm(**cirm_kwargs)
            if os.path.isfile(temporary_directory.name + '/mask.npy'):
                mask = np.load(temporary_directory.name + '/mask.npy')
                mask = ut.numpy_to_sitk(mask, spacing)
                irm.SetMetricFixedMask(mask)
                irm.SetMetricMovingMask(mask)
            fix = ut.numpy_to_sitk(timeseries_zarr[row_index].astype(np.float32), spacing)
            for iii in np.nonzero(columns_mask)[0]:
                mov = ut.numpy_to_sitk(timeseries_zarr[iii].astype(np.float32), spacing)
                values[iii] = irm.MetricEvaluate(fix, mov)
            return values
        possible_alignments = np.any(np.array(edge_matrices), axis=0)
        futures = cluster.client.map(metric_matrix_row, range(nframes), possible_alignments)
        metric_matrix = np.array(cluster.client.gather(futures))
        metric_matrix_max = np.max(metric_matrix)

        # find optimal registration pairs
        index_lists = []
        for iii, stride in enumerate(strides[::-1]):
            if stride == 1:
                index_lists.append(np.arange(nframes))
            else:
                edge_matrix = edge_matrices[iii]
                metric_matrix_copy = np.copy(metric_matrix)
                for key_node in list(set([x for xx in index_lists for x in xx])):
                    edge_matrix[:key_node, key_node+1:] = False
                    if not np.any(edge_matrix[:key_node, key_node]) and key_node > 0:
                        edge_matrix[key_node-stride:key_node, key_node] = 1
                        metric_matrix_copy[key_node-stride:key_node, key_node] = metric_matrix_max
                graph = DiGraph(metric_matrix_copy * edge_matrix)
                index_list = shortest_path(graph, source=0, target=nframes-1, weight='weight')
                index_lists.append(np.array(index_list))
        # restore to smallest strides first
        index_lists = index_lists[::-1]

    # format output
    transform_shape = (4, 4)
    if 'deform' in [x[0] for x in steps]:
        transform_shape = frame_shape + (len(frame_shape),)
    if write_path:
        outputs = []
        for stride, index_list in zip(strides, index_lists):
            output_shape = (len(index_list),)
            zarr_blocks = output_shape
            dataset_path = f'/index_list_stride_{stride}'
            index_list_store = ut.create_zarr(
                write_path, output_shape, zarr_blocks, index_list.dtype, array_path=dataset_path,
            )
            index_list_store[:] = index_list
            output_shape = (len(index_list)-1,) + transform_shape
            zarr_blocks = (1,) + transform_shape
            dataset_path = f'/transforms_stride_{stride}'
            outputs.append(ut.create_zarr(
                write_path, output_shape, zarr_blocks, np.float32, array_path=dataset_path,
            ))
    else:
        outputs = []
        for index_list in index_lists:
            output_shape = (len(index_list),) + transform_shape
            outputs.append(np.empty(output_shape, dtype=np.float32))

    # define alignments
    def align_frame_pair(fix_index, mov_index, stride_index, write_index):
        print(f'FIX INDEX: {fix_index}    MOV INDEX: {mov_index}', flush=True)
        fix = timeseries_zarr[fix_index]
        mov = timeseries_zarr[mov_index]
        if os.path.isfile(temporary_directory.name + '/mask.npy'):
            mask = np.load(temporary_directory.name + '/mask.npy')
        transform = alignment_pipeline(
            fix, mov, spacing, spacing, steps,
            fix_mask=mask,
            mov_mask=mask,
        )
        if write_path:
            outputs[stride_index][write_index] = transform
            return True
        return transform

    # submit all alignments
    futures = []
    for iii, index_list in enumerate(index_lists):
        fix_indices = index_list[:-1]
        mov_indices = index_list[1:]
        stride_indices = (iii,) * len(fix_indices,)
        write_indices = range(len(fix_indices))
        futures += cluster.client.map(
            align_frame_pair,
            fix_indices, mov_indices, stride_indices, write_indices,
        )

    # execute futures and handle outputs
    if not write_path:
        # TODO: in memory mode is not tested
        future_keys = [f.key for f in futures]
        for batch in as_completed(futures, with_results=True).batches():
            for future, result in batch:
                stride_index = 0
                write_index = future_keys.index(future.key)
                while write_index >= 0:
                    stride_index += 1
                    write_index -= len(index_lists[stride_index])
                stride_index -= 1
                write_index += len(index_lists[stride_index])
                outputs[stride_index][write_index] = result
        return outputs, index_lists
    else:
        all_written = np.all(cluster.client.gather(futures))
        return zarr.open(write_path, 'r'), index_lists


def compose_delta_transforms(
    transforms,
    spacing=None,
    sigma=None,
    sigma_threshold=None,
    write_path=None,
    **kwargs,
):
    """
    Compose all transforms returned by delta_motion_correct into a single
    displacement flow

    Parameters
    ----------
    transforms : nd array
        The output of delta_motion_correct. Sould be a time series of transforms
        with time as the first axis.

    spacing : nd array (default: None)
        The voxel spacing of the transforms. Only set this if transforms are vector
        fields.

    sigma : float or iterable of floats (default: None)
        If not None, the cumulative transform is periodically smoothed

    sigma_period : int (default: 1)
        If sigma is not None, then after every sigma_period compositions, the
        cumulative transform is smoothed.

    write_path : string (default: None)
        If not None, the final array of transforms will be written to disk as a zarr
        array at this path. If None then this array is returned in memory. Only use
        this if transforms are deformation and if the resultant set of transforms
        is too large to fit in memory. 

    kwargs : passed to compose_transforms

    Returns
    -------
    transforms : nd array
        An nd array the same shape as the input transforms, but with each transform
        based in the first time point coordinate system.
    """

    # extract all index lists and transform arrays, sort by shortest to longest
    index_lists, transform_arrays = [], []
    for key in transforms.array_keys():
        if 'index_list' in key: index_lists.append(transforms[key][1:])  # first index is always 0
        if 'transforms' in key: transform_arrays.append(transforms[key])
    sort_indices = np.argsort([x.shape[0] for x in index_lists])
    index_lists = [index_lists[x] for x in sort_indices]
    transform_arrays = [transform_arrays[x] for x in sort_indices]

    # format output
    nframes = max([x.shape[0] for x in transform_arrays])
    frame_shape = transform_arrays[0].shape[1:]
    output_shape = (nframes,) + frame_shape
    if write_path:
        zarr_blocks = (1,) + frame_shape
        new_transforms = ut.create_zarr(
            write_path,
            output_shape,
            zarr_blocks,
            transform_arrays[0].dtype,
        )
    else:
        new_transforms = np.empty(output_shape, dtype=transform_arrays[0].dtype)

    # position field needed for boundary checking, array for magnitudes
    grid = tuple(slice(None, x) for x in frame_shape[:-1])
    position_field = np.mgrid[grid].astype(transform_arrays[0].dtype)
    position_field = np.moveaxis(position_field, 0, -1) * spacing

    # flags to keep track of which transforms have been written
    transform_written = np.zeros(nframes, dtype=bool)

    # compose large stride transforms first
    for index_list, transform_array in zip(index_lists, transform_arrays):

        # initialize
        new_transforms[index_list[0]-1] = transform_array[0]
        cumulative_transform = new_transforms[index_list[0]-1]
        cumulative_magnitude = np.zeros(frame_shape[:-1], dtype=transform_array.dtype)
        for iii, write_index in enumerate(index_list[1:]):

            # re-initialize if transform was already written at higher stride level
            if transform_written[write_index-1]:
                cumulative_transform = new_transforms[write_index-1]
                cumulative_magnitude[...] = 0

            # compose
            else:
                transform = transform_array[iii+1]
                new_cumulative_transform = compose_transforms(
                    transform, cumulative_transform, spacing, spacing,
                    **kwargs,
                )
    
                # prevent vectors from leaving domain
                new_positions = position_field + new_cumulative_transform
                breaches = (new_positions < 0) + (new_positions > position_field[-1, -1, -1])
                breaches = np.any(breaches, axis=-1)
                new_cumulative_transform[breaches] = cumulative_transform[breaches]
                cumulative_transform = new_cumulative_transform

                # smooth, prevents build up of interpolation high frequencies
                if sigma is not None and sigma_threshold is not None:
                    cumulative_magnitude += np.linalg.norm(transform, axis=-1)
                    median_magnitude = np.median(cumulative_magnitude)
                    if median_magnitude > sigma_threshold:
                        print(f'ITERATION {iii}: smoothing')
                        cumulative_transform = gaussian_filter(
                            cumulative_transform, sigma/spacing,
                            axes=tuple(range(len(spacing))),
                            mode='constant',
                        )
                        cumulative_magnitude[...] = 0
                new_transforms[write_index-1] = cumulative_transform
                transform_written[write_index-1] = True
    return new_transforms


def save_transforms(path, transforms):
    """
    """

    n = transforms.shape[0]
    d = {i:transforms[i].tolist() for i in range(n)}
    with open(path, 'w') as f:
        json.dump(d, f, indent=4)


def read_transforms(path):
    """
    """

    with open(path, 'r') as f:
        d = json.load(f)
    return np.array([d[str(i)] for i in range(len(d))])


@cluster
def resample_frames(
    target,
    timeseries,
    target_spacing,
    timeseries_spacing,
    transform_list,
    timeseries_start_index=0,
    mask=None,
    interpolator='1',
    static_transform_list_before=[],
    static_transform_list_after=[],
    cluster=None,
    cluster_kwargs={},
    temporary_directory=None,
    write_path=None,
    **kwargs,
):
    """
    Resample a 4D dataset using a set of motion correction transforms

    Parameters
    ----------
    target : numpy.ndarray
        The reference image. Frames will be resampled onto this voxel grid.

    timeseries : numpy.ndarray or zarr.Array
        The moving image frames. This should be a 3D or 4D. The time axis should be
        the first axis.

    target_spacing : 1d array
        the voxel spacing of the target image in micrometers

    timeseries_spacing : 1d array
        The voxel spacing of the timeseries frames in micrometers

    transform_list : nd array
        A list of transform objects returned by motion correction functions

    mask : ndarray (default: None)
        A binary mask to apply to resampled frames before writing them.
        A masked 4D dataset like this will take up considerably less space on disk
        due to more efficient compression of zeroed background voxels. We're talking
        60% fewer gigabytes.

    timeseries_start_index : int (default: 0)
        The first frame you wish to resample is at this index
        Allows passing full zarr arrays without slicing

    interpolator : string (default: '1')
        The interpolator to use for resampling. See bigstream.configure_irm.configure_irm
        documentation for options

    static_transform_list_before : list of ndarrays (default: [])
        Transforms to apply to the moving frames before the motion correction transform

    static_transform_list_after : list of ndarrays (default: [])
        Transforms to apply to the moving frames after the motion correction transform

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
        Temporary files may be created during alignment. The temporary files will be
        in their own folder within the `temporary_directory`. The default is the
        current directory. Temporary files are removed if the function completes
        successfully.

    write_path : string (default: None)
        If not None, the final array of resampled frames will be written to disk as a zarr
        array at this path. If None then this array is returned in memory.

    kwargs : any additional keyword arguments
        Passed to bigstream.transform.apply_transform

    Returns
    -------
    resampled data : zarr Array
        A reference to the zarr array on disk containing the resampled data
    """

    # put timeseries (as zarr), target, and mask, in temporary directory
    temporary_directory = temporary_directory or os.getcwd()
    temporary_directory = tempfile.TemporaryDirectory(prefix='.', dir=temporary_directory)
    zarr_blocks = (1,) + timeseries.shape[1:]
    timeseries_zarr_path = temporary_directory.name + '/timeseries.zarr'
    timeseries_zarr = ut.numpy_to_zarr(timeseries, zarr_blocks, timeseries_zarr_path)
    np.save(temporary_directory.name + '/target.npy', target)
    if mask is not None:
        mask_sh, timeseries_sh = mask.shape, timeseries.shape[1:]
        if mask_sh != timeseries_sh:
            mask = zoom(mask, np.array(timeseries_sh) / mask_sh, order=0)
        np.save(temporary_directory.name + '/mask.npy', mask)

    # save motion correction transforms as zarr array
    transform_list_zarr = []
    for iii, transforms in enumerate(transform_list):
        zarr_blocks = (1,) + transforms.shape[1:]
        transforms_zarr_path = temporary_directory.name + f'/transforms{iii}.zarr'
        transform_list_zarr.append(ut.numpy_to_zarr(transforms, zarr_blocks, transforms_zarr_path))

    # save before/after static deforms in temporary directory
    new_list = []
    for iii, transform in enumerate(static_transform_list_before):
        if len(transform.shape) > 2:
            path = temporary_directory.name + f'/deform{iii}.npy'
            np.save(path, transform)
            transform = path
        new_list.append(transform)
    static_transform_list_before = new_list
    new_list = []
    for iii, transform in enumerate(static_transform_list_after):
        if len(transform.shape) > 2:
            path = temporary_directory.name + f'/deform{iii}.npy'
            np.save(path, transform)
            transform = path
        new_list.append(transform)
    static_transform_list_after = new_list

    # format output
    output_shape = (transform_list_zarr[0].shape[0],) + target.shape
    if write_path:
        zarr_blocks = (1,) + output_shape[1:]
        output = ut.create_zarr(write_path, output_shape, zarr_blocks, target.dtype)

    # define resample function for frames
    def apply_transform_to_frame(
        index,
        static_transform_list_before,
        static_transform_list_after,
    ):
        fix = np.load(temporary_directory.name + '/target.npy')
        mov = timeseries_zarr[timeseries_start_index + index]
        transform_list = [x[index] for x in transform_list_zarr]
        a = [np.load(x) if isinstance(x, str) else x for x in static_transform_list_before]
        b = [np.load(x) if isinstance(x, str) else x for x in static_transform_list_after]
        transform_list = a + transform_list + b
        aligned = apply_transform(
            fix, mov, target_spacing, timeseries_spacing,
            transform_list=transform_list,
            interpolator=interpolator,
            **kwargs,
        )
        if os.path.isfile(temporary_directory.name + '/mask.npy'):
            mask = np.load(temporary_directory.name + '/mask.npy')
            aligned = aligned * mask
        if write_path:
            output[index] = aligned
            return True
        return aligned

    # distribute and wait for completion
    futures = cluster.client.map(
        apply_transform_to_frame, range(transform_list_zarr[0].shape[0]),
        static_transform_list_before=static_transform_list_before,
        static_transform_list_after=static_transform_list_after,
    )
    if not write_path:
        output = np.empty(output_shape, dtype=target.dtype)
        future_keys = [f.key for f in futures]
        for batch in as_completed(futures, with_results=True).batches():
            for future, result in batch:
                output[future_keys.index(future.key)] = result
    else:
        all_written = np.all(cluster.client.gather(futures))
    return output

