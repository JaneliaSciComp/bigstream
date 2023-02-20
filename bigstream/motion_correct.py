import numpy as np
import os
from ClusterWrap.decorator import cluster
import dask.array as da
import bigstream.utility as ut
from bigstream.align import affine_align
from bigstream.align import deformable_align
from bigstream.transform import apply_transform
from scipy.ndimage import median_filter
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import map_coordinates
from scipy.ndimage import zoom
import zarr
from glob import glob
import json
import tempfile


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
def motion_correct(
    fix, mov_zarr,
    fix_spacing, mov_spacing,
    time_stride=1,
    sigma=7,
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

    sigma : float
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
    params = gaussian_filter1d(params, sigma, axis=0)
    if time_stride > 1:
        x = np.arange(total_frames) / time_stride
        coords = np.meshgrid(x, np.mgrid[:6], indexing='ij')
        params = map_coordinates(params, coords, mode='nearest')

    # convert to matrices
    # TODO: again this assumes rigid transforms
    transforms = np.empty((total_frames, 4, 4))
    for i in range(params.shape[0]):
        e = ut.parameters_to_euler_transform(params[i])
        t = ut.affine_transform_to_matrix(e)
        transforms[i] = t

    # return all transforms
    return transforms


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
    mov_zarr,
    fix_spacing,
    mov_spacing,
    transforms,
    write_path,
    mask=None,
    time_stride=1,
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

    mov_zarr : zarr.Array
        The moving image frames. This should be a 4D zarr Array object that is
        only lazy loaded, e.g. not in memory. The time axis should be the first
        axis.

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
        mask_sh, mov_sh = mask.shape, mov_zarr.shape[1:]
        if mask_sh != mov_sh:
            mask = zoom(mask, np.array(mov_sh) / mask_sh, order=0)
        np.save(temporary_directory.name + '/mask.npy', mask)

    # save initial deforms to location accessible to all workers
    new_list = []
    for iii, transform in enumerate(static_transform_list_before):
        if transform.shape != (4, 4) and len(transform.shape) != 1:
            path = temporary_directory.name + f'/deform{iii}.npy'
            np.save(path, transform)
            transform = path
        new_list.append(transform)
    static_transform_list_before = new_list

    # save subsequent deforms to location accessible to all workers
    new_list = []
    for iii, transform in enumerate(static_transform_list_after):
        if transform.shape != (4, 4) and len(transform.shape) != 1:
            path = temporary_directory.name + f'/deform{iii}.npy'
            np.save(path, transform)
            transform = path
        new_list.append(transform)
    static_transform_list_after = new_list

    # determine which frames will be resampled, ensure transforms are list
    total_frames = mov_zarr.shape[0]
    mov_indices = range(0, total_frames, time_stride)
    compute_frames = len(mov_indices)
    transforms = list(transforms)

    # create an output zarr file
    output_zarr = ut.create_zarr(
        write_path,
        (compute_frames,) + mov_zarr.shape[1:],
        (1,) + mov_zarr.shape[1:],
        np.uint16,
    )

    # wrap transform function
    def wrapped_apply_transform(
        read_index, write_index, transform,
        static_transform_list_before,
        static_transform_list_after,
    ):

        # read frame and mask data
        fix = np.load(temporary_directory.name + '/fix.npy')
        mov = mov_zarr[read_index]
        if os.path.isfile(temporary_directory.name + '/mask.npy'):
            mask = np.load(temporary_directory.name + '/mask.npy')

        # format transform_list
        a = [np.load(x) if isinstance(x, str) else x for x in static_transform_list_before]
        b = [np.load(x) if isinstance(x, str) else x for x in static_transform_list_after]
        transform_list = [transform,]
        if len(transform.shape) == 1:  # affine + bspline case
            transform_list = [transform[:16].reshape((4,4)), transform[16:]]
        transform_list = a + transform_list + b

        # apply transform_list
        aligned = apply_transform(
            fix, mov, fix_spacing, mov_spacing,
            transform_list=transform_list,
        )

        # mask and write result
        if mask is not None: aligned = aligned * mask
        output_zarr[write_index] = aligned
        return True

    # distribute and wait for completion
    futures = cluster.client.map(
        wrapped_apply_transform,
        mov_indices, range(compute_frames), transforms,
        static_transform_list_before=static_transform_list_before,
        static_transform_list_after=static_transform_list_after,
    )
    all_written = np.all( cluster.client.gather(futures) )
    if not all_written: print("SOMETHING FAILED, CHECK LOGS")
    return output_zarr

