import numpy as np
import os
from ClusterWrap.decorator import cluster
import CircuitSeeker.fileio as csio
import CircuitSeeker.utility as ut
from CircuitSeeker.align import affine_align
from CircuitSeeker.align import deformable_align
from CircuitSeeker.transform import apply_transform
import dask.array as da
import dask.delayed as delayed
from scipy.ndimage import median_filter
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import map_coordinates
from scipy.ndimage import zoom
import zarr
from glob import glob
import json


@cluster
def distributed_image_mean(
    frames,
    cluster=None,
    cluster_kwargs={},
):
    """
    Voxelwise mean over all images specified by frames.
    Distributed with Dask for expediency.

    Parameters
    ----------
    frames : dict
        Specifies the image set on disk to be averaged. At least three
        keys must be defined:
            'folder' : directory containing the image set
            'prefix' : common prefix to all images being averaged
            'suffix' : common suffix - typically the file extension
        Common values for suffix are '.h5', '.stack', or '.tiff'

        If suffix is '.h5' then an additional key must be defined:
            'dataset_path' : path to image dataset within hdf5 container

        If suffix is '.stack' then two additional keys must be defined:
            'dtype' : a numpy datatype object for the datatype in the raw images
            'shape' : the array shape of the data as a tuple

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
        The voxelwise mean of the images specified in frames
    """

    # hdf5 files use dask.array
    if csio.testPathExtensionForHDF5(frames['suffix']):
        frames = csio.daskArrayBackedByHDF5(
            frames['folder'], frames['prefix'],
            frames['suffix'], frames['dataset_path'],
        )
        frames_mean = frames.mean(axis=0, dtype=np.float32).compute()
        frames_mean = np.round(frames_mean).astype(frames[0].dtype)
    # stack files use dask.array
    elif csio.testPathExtensionForSTACK(frames['suffix']):
        frames = csio.daskArrayBackedBySTACK(
            frames['folder'], frames['prefix'], frames['suffix'],
            frames['dtype'], frames['shape'],
        )
        frames_mean = frames.mean(axis=0, dtype=np.float32).compute()
        frames_mean = np.round(frames_mean).astype(frames[0].dtype)
    # other types use dask.bag
    else:
        frames = csio.daskBagOfFilePaths(
            frames['folder'], frames['prefix'], frames['suffix'],
        )
        nframes = frames.npartitions
        frames_mean = frames.map(csio.readImage).reduction(sum, sum).compute()
        dtype = frames_mean.dtype
        frames_mean = np.round(frames_mean/np.float(nframes)).astype(dtype)

    # return reference to mean image
    return frames_mean


@cluster
def motion_correct(
    fix, frames,
    fix_spacing, frames_spacing,
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

    frames : dict
        Specifies the image set on disk to be motion corrected. At least three
        keys must be defined:
            'folder' : directory containing the image set
            'prefix' : common prefix to all images being averaged
            'suffix' : common suffix - typically the file extension
        Common values for suffix are '.h5', '.stack', or '.tiff'

        If suffix is '.h5' then an additional key must be defined:
            'dataset_path' : path to image dataset within hdf5 container

        If suffix is '.stack' then two additional keys must be defined:
            'dtype' : a numpy datatype object for the datatype in the raw images
            'shape' : the array shape of the data as a tuple

    fix_spacing : 1d array
        The voxel spacing of the fixed reference image in micrometers

    frames_spacing : 1d array
        The voxel spacing of the moving images in micrometers

    time_stride : int
        The stride along the time axis along which to align. E.g. if time_stride is 10,
        then only every 10th frame in the provided dataset is aligned. Intermediate
        frame transforms are interpolated from the computed ones.

    sigma : float
        Standard deviation of a Gaussian kernel applied to the found transforms along
        the time axis. This stabilizes the alignment at the expense of allowing a bit
        more long term motion.

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
    transforms : 2d array, NxP
        All transforms aligning the dataset to the reference. N == number of provided images
        P == the number of parameters in each transform. E.g. a dataset with 100 images and
        full affine alignment to the reference would yield a return array with shape 100x12.
    """


    # scatter fixed image to cluster
    fix_d = cluster.client.scatter(fix, broadcast=True)
    # scatter fixed mask to cluster
    fix_mask_d = None
    if fix_mask is not None:
        fix_mask_d = cluster.client.scatter(fix_mask, broadcast=True)

    # get total number of frames
    total_frames = len(csio.globPaths(
        frames['folder'], frames['prefix'], frames['suffix'],
    ))

    # create dask array of all frames
    if csio.testPathExtensionForHDF5(frames['suffix']):
        frames_data = csio.daskArrayBackedByHDF5(
            frames['folder'], frames['prefix'],
            frames['suffix'], frames['dataset_path'],
            stride=time_stride,
        )
    elif csio.testPathExtensionForSTACK(frames['suffix']):
        frames_data = csio.daskArrayBackedBySTACK(
            frames['folder'], frames['prefix'], frames['suffix'],
            frames['dtype'], frames['shape'],
            stride=time_stride,
        )
    compute_frames = frames_data.shape[0]

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
    def wrapped_affine_align(mov, fix_d, fix_mask_d):
        mov = mov.squeeze()
#        t = affine_align(
#            fix_d, mov, fix_spacing, frames_spacing,
#            fix_mask=fix_mask_d,
#            **kwargs,
#        )
        t = affine_align(
            mov, fix_d, frames_spacing, fix_spacing,
            mov_mask=fix_mask_d,
            **kwargs,
        )
        t = np.linalg.inv(t)
        # TODO: 2 lines below assume its a rigid transform
        e = ut.matrix_to_euler_transform(t)
        p = ut.euler_transform_to_parameters(e)
        return p[None, :]

    params = da.map_blocks(
        wrapped_affine_align, frames_data,
        fix_d=fix_d,
        fix_mask_d=fix_mask_d,
        dtype=np.float64,
        drop_axis=[2, 3,],
        chunks=[1, 6],  # TODO: chunk size here assumes rigid transform
    ).compute()

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


# TODO: not yet refactored
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
    frames,
    frames_spacing,
    transforms,
    write_path,
    mask=None,
    time_stride=1,
    compression_level=4,
    cluster=None,
    cluster_kwargs={},
):
    """
    Resample a 4D dataset using a set of motion correction transforms
    Frames are resampled on the same voxel grid they are defined on. This works
    fine when transforms are relatively small, but will cause foreground to go
    out of the field of view for larger transforms. If you've done motion correction
    correctly, this is unlikely, but still possible for some experimental designs.

    Parameters
    ----------
    frames : dict
        Specifies the image set on disk to be resampled. At least three
        keys must be defined:
            'folder' : directory containing the image set
            'prefix' : common prefix to all images being averaged
            'suffix' : common suffix - typically the file extension
        Common values for suffix are '.h5', '.stack', or '.tiff'

        If suffix is '.h5' then an additional key must be defined:
            'dataset_path' : path to image dataset within hdf5 container

        If suffix is '.stack' then two additional keys must be defined:
            'dtype' : a numpy datatype object for the datatype in the raw images
            'shape' : the array shape of the data as a tuple

    frames_spacing : 1d array
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


    # create dask array of all frames
    if csio.testPathExtensionForHDF5(frames['suffix']):
        frames_data = csio.daskArrayBackedByHDF5(
            frames['folder'], frames['prefix'],
            frames['suffix'], frames['dataset_path'],
            stride=time_stride,
        )
    elif csio.testPathExtensionForSTACK(frames['suffix']):
        frames_data = csio.daskArrayBackedBySTACK(
            frames['folder'], frames['prefix'], frames['suffix'],
            frames['dtype'], frames['shape'],
            stride=time_stride,
        )
    compute_frames = frames_data.shape[0]

    # wrap transforms as dask array
    # extra dimension to match frames_data ndims
    if len(transforms.shape) == 3:
        transforms = transforms[::time_stride, None, :, :]
    elif len(transforms.shape) == 2:
        transforms = transforms[::time_stride, None, None, :]
    transforms_d = da.from_array(transforms, chunks=(1,)+transforms[0].shape)

    # wrap mask
    mask_d = None
    if mask is not None:
        mask_sh, frame_sh = mask.shape, frames_data.shape[1:]
        if mask_sh != frame_sh:
            mask = zoom(mask, np.array(frame_sh) / mask_sh, order=0)
        mask_d = cluster.client.scatter(mask, broadcast=True)

    # wrap transform function
    def wrapped_apply_transform(mov, t, mask_d=None):
        mov = mov.squeeze()
        t = t.squeeze()

        # just an affine matrix
        transform_list = [t,]

        # affine plus bspline
        if len(t.shape) == 1:
            transform_list = [t[:16].reshape((4,4)), t[16:]]

        # apply transform(s)
        aligned = apply_transform(
            mov, mov, frames_spacing, frames_spacing,
            transform_list=transform_list,
        )
        if mask_d is not None:
            aligned = aligned * mask_d
        return aligned[None, ...]

    # apply transform to all frames
    frames_aligned = da.map_blocks(
        wrapped_apply_transform, frames_data, transforms_d,
        mask_d=mask_d,
        dtype=np.uint16,
        chunks=[1,] + list(frames_data.shape[1:]),
    )

    # write in parallel as 4D array to zarr file
    da.to_zarr(frames_aligned, write_path)
    return zarr.open(write_path, mode='r+')


