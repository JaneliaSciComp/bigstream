import numpy as np
import os
from ClusterWrap.decorator import cluster
import dask.array as da
from dask.distributed import as_completed
import bigstream.utility as ut
from bigstream.align import affine_align
from bigstream.align import deformable_align
from bigstream.align import alignment_pipeline
from bigstream.transform import apply_transform
from bigstream.transform import compose_transforms
from scipy.ndimage import median_filter
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import map_coordinates
from scipy.ndimage import zoom
import zarr
from glob import glob
import json
import tempfile
from skimage.exposure import match_histograms


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
    fix, mov_zarr,
    fix_spacing, mov_spacing,
    time_stride=1,
    sigma=None,
    fix_mask=None,
    cluster=None,
    cluster_kwargs={},
    **kwargs,
):
    """
    Affine align all frames in a 4D dataset to a single 3D reference frame.
    Efficiently distributed with Dask.

    Parameters
    ----------
    fix : ndarray
        The fixed image reference. All frames will be aligned to this image.
        This should be an in-memory numpy array.

    mov_zarr : zarr.Array
        The moving image frames. This should be a 4D zarr Array object that is
        only lazy loaded, e.g. not in memory. The time axis should be the first
        axis.

    fix_spacing : 1d array
        The voxel spacing of the fixed reference image in micrometers

    mov_spacing : 1d array
        The voxel spacing of the moving images in micrometers

    time_stride : int
        The stride along the time axis along which to align. E.g. if time_stride is 10,
        then only every 10th frame in the provided dataset is aligned. Intermediate
        frame transforms are interpolated from the computed ones.

    sigma : float or None
        Standard deviation of a Gaussian kernel applied to the found transforms along
        the time axis. This stabilizes the alignment at the expense of allowing a bit
        more long term motion. This is in units of frames *after* time stride has been
        applied. So time_stride=4 and sigma=2 gives the same overall smoothing value as
        time_stride=1 and sigma=8.

    fix_mask : binary ndarray
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

    kwargs : any additional arguments
        Passed to configure_irm. Control the nature of alignments through these arguments

    Returns
    -------
    transforms : 3d array, Nx4x4
        All transforms aligning the dataset to the reference. N == number of provided images
        and for each one there is a 4x4 affine (or rigid) transform matrix.
    """

    # put temp copies of fixed and mask on disk for workers to retrieve
    temporary_directory = tempfile.TemporaryDirectory(
        prefix='.', dir=os.getcwd(),
    )
    np.save(temporary_directory.name + '/fix.npy', fix)
    if fix_mask is not None:
        np.save(temporary_directory.name + '/fix_mask.npy', fix_mask)

    # determine which frames will be aligned
    total_frames = mov_zarr.shape[0]
    mov_indices = range(0, total_frames, time_stride)
    compute_frames = len(mov_indices)

    # set alignment defaults
    alignment_defaults = {
        'rigid':True,
        'alignment_spacing':2.0,
        'shrink_factors':(2,),
        'smooth_sigmas':(2.,),
        'metric':'MS',
        'optimizer_args':{
            'learningRate':0.25,
            'minStep':0.,
            'numberOfIterations':400,
        },
    }
    kwargs = {**alignment_defaults, **kwargs}

    # wrap align function
    def wrapped_affine_align(index):
        fix = np.load(temporary_directory.name + '/fix.npy')
        fix_mask = None
        if os.path.isfile(temporary_directory.name + '/fix_mask.npy'):
            fix_mask = np.load(temporary_directory.name + '/fix_mask.npy')
        mov = mov_zarr[index]
        t = affine_align(
            fix, mov, fix_spacing, mov_spacing,
            fix_mask=fix_mask,
            **kwargs,
        )
        # TODO: 2 lines below assume its a rigid transform
        e = ut.matrix_to_euler_transform(t)
        return ut.euler_transform_to_parameters(e)

    # distribute
    futures = cluster.client.map(wrapped_affine_align, mov_indices)
    params = np.array(cluster.client.gather(futures))

    # smooth and interpolate
    # TODO: this kind of smoothing will not work with affine transforms
    if sigma:
        params = gaussian_filter1d(params, sigma / time_stride, axis=0)
    if time_stride > 1:
        x = np.arange(total_frames) / time_stride
        coords = np.meshgrid(x, np.mgrid[:params.shape[1]], indexing='ij')
        params = map_coordinates(params, coords, order=1, mode='nearest')

    # convert to matrices
    # TODO: again this assumes rigid transforms
    transforms = np.empty((total_frames, fix.ndim+1, fix.ndim+1))
    for i in range(params.shape[0]):
        e = ut.parameters_to_euler_transform(params[i])
        t = ut.affine_transform_to_matrix(e)
        transforms[i] = t

    # return all transforms
    return transforms


@cluster
def delta_motion_correct(
    timeseries,
    spacing,
    steps,
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
        Arguments specified here override any global arguments give through kwargs
        for their specific step only.

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
        If not None, the final array of transforms will be written to disk as a zarr
        array at this path. If None then this array is returned in memory. Only use
        this if transforms are deformation and if the resultant set of transforms
        is too large to fit in memory. 

    kwargs : any additional arguments
        Passed to alignment_pipeline. Control the nature of alignments through these arguments

    Returns
    -------
    transforms : nd-array
        All transforms aligning the dataset to the reference. N == number of provided images
    """

    # convenient names
    nframes = timeseries.shape[0]
    frame_shape = timeseries.shape[1:]

    # ensure input is a zarr array
    temporary_directory = tempfile.TemporaryDirectory(
        prefix='.', dir=temporary_directory or os.getcwd(),
    )
    zarr_blocks = (1,) + frame_shape
    timeseries_zarr_path = temporary_directory.name + '/timeseries.zarr'
    timeseries_zarr = ut.numpy_to_zarr(timeseries, zarr_blocks, timeseries_zarr_path)

    # ensure mask is readable by all workers
    if mask is not None:
        np.save(temporary_directory.name + '/mask.npy', mask)

    # zarr file for output
    if write_path:
        output_transform = ut.create_zarr(
            write_path,
            (nframes-1,) + frame_shape + (3,),
            zarr_blocks + (3,),
            np.float32,
        )

    # establish all keyword arguments
    steps = [(a, {**kwargs, **b}) for a, b in steps]

    # closure for alignment pipeline
    def align_frame_pair(index):

        # load frames and (optionally) mask
        fix = timeseries_zarr[index]
        mov = timeseries_zarr[index+1]
        mask = None
        if os.path.isfile(temporary_directory.name + '/mask.npy'):
            mask = np.load(temporary_directory.name + '/mask.npy')

        # align and return or write result
        transform = alignment_pipeline(
            fix, mov, spacing, spacing, steps,
            fix_mask=mask,
            mov_mask=mask,
        )
        if not write_path:
            return transform
        else:
            output_transform[index] = transform
            return True

    # submit all pairs
    futures = cluster.client.map(align_frame_pair, range(nframes-1))
    if not write_path:
        transform = np.empty((nframes-1,) + frame_shape + (3,), dtype=np.float32)
        future_keys = [f.key for f in futures]
        for batch in as_completed(futures, with_results=True).batches():
            for future, result in batch:
                iii = future_keys.index(future.key)
                transform[iii] = result
        return transform
    else:
        all_written = np.all(cluster.client.gather(future))
        return output_transform


def compose_delta_transforms(
    transforms,
    spacing=None,
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

    Returns
    -------
    transforms : nd array
        An nd array the same shape as the input transforms, but with each transform
        based in the first time point coordinate system.
    """

    new_transforms = np.empty_like(transforms)
    new_transforms[0] = np.copy(transforms[0])
    for iii, transform in enumerate(transforms[1:]):
        new_transforms[iii+1] = compose_transforms(
            transform, new_transforms[iii], spacing, spacing,
        )
    return new_transforms
        


# TODO: VERY OLD - UPDATE TO TAKE ZARR INPUTS AND BE
#    CONSISTENT WITH motion_correct FUNCTION
def deformable_motion_correct(
    fix, frames,
    fix_spacing, frames_spacing,
    time_stride=1,
    sigma=7,
    fix_mask=None,
    affine_kwargs={},
    bspline_kwargs={},
    cluster_kwargs={},
):
    """
    """

    with ClusterWrap.cluster(**cluster_kwargs) as cluster:

        # wrap fixed data as delayed object
        fix_d = delayed(fix)

        # wrap fixx mask if given
        fix_mask_d = delayed(np.ones(fix.shape, dtype=np.uint8))
        if fix_mask is not None:
            fix_mask_d = delayed(fix_mask)

        # get total number of frames
        total_frames = len(csio.globPaths(
            frames['folder'], frames['prefix'], frames['suffix'],
        ))

        # create dask array of all frames
        frames_data = csio.daskArrayBackedByHDF5(
            frames['folder'], frames['prefix'],
            frames['suffix'], frames['dataset_path'],
            stride=time_stride,
        )
        compute_frames = frames_data.shape[0]

        # affine defaults
        affine_defaults = {
            'alignment_spacing':2.0,
            'metric':'MS',
            'sampling':'random',
            'sampling_percentage':0.1,
            'optimizer':'RGD',
            'iterations':50,
            'max_step':2.0,
        }
        for k, v in affine_defaults.items():
            if k not in affine_kwargs: affine_kwargs[k] = v

        # bspline defaults
        bspline_defaults = {
            'alignment_spacing':1.0,
            'metric':'MI',
            'sampling':'random',
            'sampling_percentage':0.01,
            'iterations':250,
            'shrink_factors':[2,],
            'smooth_sigmas':[2,],
            'max_step':1.0,
            'control_point_spacing':100.,
            'control_point_levels':[1,],
        }
        for k, v in bspline_defaults.items():
            if k not in bspline_kwargs: bspline_kwargs[k] = v
 
        # wrap align function
        def wrapped_bspline_align(mov, fix_d, fix_mask_d):
            mov = mov.squeeze()
            a = affine_align(
                fix_d, mov, fix_spacing, frames_spacing,
                **affine_kwargs,
            )
            b = bspline_deformable_align(
                fix_d, mov, fix_spacing, frames_spacing,
                fix_mask=fix_mask_d,
                initial_transform=a,
                return_parameters=True,
                **bspline_kwargs,
            )
            return np.hstack((a.flatten(), b))[None, :]

        # total number of params
        # 16 for affine, 18 for bspline fixed params
        # need to compute number of bspline control point params
        xxx = fix.shape * fix_spacing
        y = bspline_kwargs['control_point_spacing']
        cp_grid = [max(1, int(x/y)) + 3 for x in xxx]
        n_params = 16 + 18 + np.prod(cp_grid)*3

        # execute
        params = da.map_blocks(
            wrapped_bspline_align, frames_data,
            fix_d=fix_d,
            fix_mask_d=fix_mask_d,
            dtype=np.float64,
            drop_axis=[2, 3,],
            chunks=[1, n_params],
        ).compute()

    # (weak) outlier removal and smoothing
    params = median_filter(params, footprint=np.ones((3,1)))
    params = gaussian_filter1d(params, sigma, axis=0)

    # interpolate
    if time_stride > 1:
        x = np.linspace(0, compute_frames-1, total_frames)
        coords = np.meshgrid(x, np.mgrid[:n_params], indexing='ij')
        params = map_coordinates(params, coords, order=1)

    # return all parameters
    return params


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
    fix,
    mov,
    fix_spacing,
    mov_spacing,
    transforms,
    write_path,
    mask=None,
    time_stride=1,
    interpolator='1',
    compression_level=4,
    static_transform_list_before=[],
    static_transform_list_after=[],
    cluster=None,
    cluster_kwargs={},
):
    """
    Resample a 4D dataset using a set of motion correction transforms

    Parameters
    ----------
    fix : numpy.ndarray
        The reference image. Frames will be resampled onto this voxel grid.

    mov : numpy.ndarray or zarr.Array
        The moving image frames. This should be a 3D or 4D. The time axis should be
        the first axis.

    fix_spacing : 1d array
        the voxel spacing of the fixed image in micrometers

    mov_spacing : 1d array
        The voxel spacing of the moving images in micrometers

    transforms : nd array
        The transforms returned by a motion correction function

    write_path : string
        The location on disk to write the zarr folder containing the resampled data

    mask : ndarray (default: None)
        A binary mask to apply to resampled frames before writing them.
        A masked 4D dataset like this will take up considerably less space on disk
        due to more efficient compression of zeroed background voxels. We're talking
        60% fewer gigabytes.

    time_stride : int (default: 1)
        The stride along time axis at which to resmaple the images

    interpolator : string (default: '1')
        The interpolator to use for resampling. See bigstream.configure_irm.configure_irm
        documentation for options

    compression_level : int between 1 and 9; (default: 4)
        How much to compress the resampled data. Lower numbers will write to disk
        much faster but also take up more space. High numbers take considerably longer
        to write to disk.

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

    Returns
    -------
    resampled data : zarr Array
        A reference to the zarr array on disk containing the resampled data
    """

    # format and depost mask in location accessible to all workers
    temporary_directory = tempfile.TemporaryDirectory(
        prefix='.', dir=os.getcwd(),
    )
    np.save(temporary_directory.name + '/fix.npy', fix)
    if mask is not None:
        mask_sh, mov_sh = mask.shape, mov.shape[1:]
        if mask_sh != mov_sh:
            mask = zoom(mask, np.array(mov_sh) / mask_sh, order=0)
        np.save(temporary_directory.name + '/mask.npy', mask)

    # save moving image frames as zarr array
    zarr_blocks = (1,) + mov.shape[1:]
    mov_zarr_path = temporary_directory.name + '/mov.zarr'
    mov_zarr = ut.numpy_to_zarr(mov, zarr_blocks, mov_zarr_path)

    # save transforms as zarr array
    zarr_blocks = (1,) + transforms.shape[1:]
    transforms_zarr_path = temporary_directory.name + '/transforms.zarr'
    transforms_zarr = ut.numpy_to_zarr(transforms, zarr_blocks, transforms_zarr_path)

    # save initial deforms to location accessible to all workers
    new_list = []
    for iii, transform in enumerate(static_transform_list_before):
        if len(transform.shape) > 2:
            path = temporary_directory.name + f'/deform{iii}.npy'
            np.save(path, transform)
            transform = path
        new_list.append(transform)
    static_transform_list_before = new_list

    # save subsequent deforms to location accessible to all workers
    new_list = []
    for iii, transform in enumerate(static_transform_list_after):
        if len(transform.shape) > 2:
            path = temporary_directory.name + f'/deform{iii}.npy'
            np.save(path, transform)
            transform = path
        new_list.append(transform)
    static_transform_list_after = new_list

    # determine which frames will be resampled
    total_frames = mov_zarr.shape[0]
    mov_indices = range(0, total_frames, time_stride)
    compute_frames = len(mov_indices)

    # create an output zarr file
    output_zarr = ut.create_zarr(
        write_path,
        (compute_frames,) + mov_zarr.shape[1:],
        (1,) + mov_zarr.shape[1:],
        mov_zarr.dtype,
    )

    # wrap transform function
    def wrapped_apply_transform(
        read_index, write_index,
        static_transform_list_before,
        static_transform_list_after,
    ):

        # read frame and mask data
        fix = np.load(temporary_directory.name + '/fix.npy')
        mov = mov_zarr[read_index]
        transform = transforms_zarr[read_index]
        mask = None
        if os.path.isfile(temporary_directory.name + '/mask.npy'):
            mask = np.load(temporary_directory.name + '/mask.npy')

        # format transform_list
        a = [np.load(x) if isinstance(x, str) else x for x in static_transform_list_before]
        b = [np.load(x) if isinstance(x, str) else x for x in static_transform_list_after]
        transform_list = [transform,]
        if len(transform.shape) == 1:  # affine + bspline case
            # deformable_motion_correct not generalize to 2d yet, so this is fine
            transform_list = [transform[:16].reshape((4,4)), transform[16:]]
        transform_list = a + transform_list + b

        # apply transform_list
        aligned = apply_transform(
            fix, mov, fix_spacing, mov_spacing,
            transform_list=transform_list,
            interpolator=interpolator,
        )

        # mask and write result
        if mask is not None: aligned = aligned * mask
        output_zarr[write_index] = aligned
        return True

    # distribute and wait for completion
    futures = cluster.client.map(
        wrapped_apply_transform,
        mov_indices, range(compute_frames),
        static_transform_list_before=static_transform_list_before,
        static_transform_list_after=static_transform_list_after,
    )
    all_written = np.all( cluster.client.gather(futures) )
    if not all_written: print("SOMETHING FAILED, CHECK LOGS")
    return output_zarr


