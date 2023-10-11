import numpy as np
import os, tempfile
import SimpleITK as sitk
from bigstream.configure_irm import configure_irm
import bigstream.utility as ut
from itertools import product
from ClusterWrap.decorator import cluster


def patch_mutual_information(
    fix,
    mov,
    spacing,
    radius,
    stride,
    percentile_cutoff=0,
    fix_mask=None,
    mov_mask=None,
    return_metric_image=False,
    **kwargs,
):
    """
    Local mutual information metric between two images
    MI is computed patch-wise across both images and the mean over all
    patches is returned

    Parameters
    ----------
    fix : nd-array
        fixed image
    mov : nd-array
        moving image
    spacing : 1d-array
        The voxel spacing of the two images (must be the same)
    radius : scalar float
        Neighborhood half-width in physical units
    stride : scalar int
        Spacing between neighborhood centers
    percentile_cutoff : scalar float (default: 0)
        local MI scores below this value are ignored in final mean computation
    fix_mask : binary nd-array (default: None)
        mask over fixed data (only data in foreground is considered)
    mov_mask : binary nd-array (default: None)
        mask over moving data (only data in foreground is considered)
    return_metric_image : bool (default: False)
        Return an image with local MIs
    **kwargs : any additional arguments
        Passed to bigstream.configure_irm.configure_irm
        Use these arguments to parameterize the metric

    Returns
    -------
    score : scalar float
        The local MI averaged over all patches
    metric_image : nd-array
        Optional output only returned if return_metric_image == True
        The local MIs rendered in an image
    """

    # create sitk versions of data
    fix_sitk = ut.numpy_to_sitk(fix.transpose(2, 1, 0), spacing[::-1])
    fix_sitk = sitk.Cast(fix_sitk, sitk.sitkFloat32)
    mov_sitk = ut.numpy_to_sitk(mov.transpose(2, 1, 0), spacing[::-1])
    mov_sitk = sitk.Cast(mov_sitk, sitk.sitkFloat32)

    # determine patch sample centers
    samples = np.zeros_like(fix)
    radius = np.round(radius / spacing).astype(np.uint16)
    stride = np.round(stride / spacing).astype(np.uint16)
    samples[tuple(slice(r, -r, s) for r, s in zip(radius, stride))] = 1
    if fix_mask is not None: samples = samples * fix_mask
    if mov_mask is not None: samples = samples * mov_mask
    samples = np.column_stack(np.nonzero(samples))

    # create irm and containers for results
    irm = configure_irm(**kwargs)
    if return_metric_image:
        metric_image = np.zeros(fix.shape, dtype=np.float32)
    scores = []

    # score all blocks
    for sample in samples:
        # get patches
        patch = tuple(slice(s-r, s+r+1) for s, r in zip(sample, radius))
        f = fix_sitk[patch]
        m = mov_sitk[patch]
        # evaluate metric
        try:
            scores.append( irm.MetricEvaluate(f, m) )
        except Exception as e:
            scores.append( 0 )
        # update metric image
        if return_metric_image:
            metric_image[patch] = scores[-1]

    # threshold scores
    scores = np.array(scores)
    if percentile_cutoff > 0:
        cutoff = np.percentile(-scores, percentile_cutoff)
        scores = scores[-scores > cutoff]

    # return results
    if return_metric_image:
        return np.mean(scores), metric_image
    else:
        return np.mean(scores)


def local_correlation_coefficient(
    fix,
    mov,
    spacing,
    radius,
    return_image=False,
    tolerance=1e-6,
):
    """
    Compute correlation coefficient for neighborhoods around every voxel
    Return the average of this value across the whole image and optionally
    the LCC image itself.

    Parameters
    ----------
    fix : ndarray
        One of the images
        This algorithm is symmetric, it does not matter which image is fix or mov
        Terms are borrowed from registration functions for consistency

    mov : ndarray
        The other image
        This algorithm is symmetric, it does not matter which image is fix or mov
        Terms are borrowed from registration functions for consistency

    spacing : 1d array
        The voxel spacing of the input images in physical units
        fix and mov must be sampled on the exact same grid for this function to work

    radius : float
        The half width of the neighborhood around each voxel to compute the local
        correlations. This is a scalar value in physical units.

    return_image : bool (default: False)
        If True this function will also return the image of the local correlation
        coefficients

    tolerance : float (default: 1e-6)
        The lower bound on variance for CC to adequately be computed

    Returns
    -------
    LCC : float
        A single scalar value - the average of the LCCs across the whole image domain

    LCC_image : ndarray
        Only returned if return_image is True
    """

    # convert radius to integer voxel units
    radius = np.round(radius / spacing).astype(int)

    # get local means and variances, use high precision and zero center images for stability
    fix_means = fix.astype(np.longdouble) - np.mean(fix)
    mov_means = mov.astype(np.longdouble) - np.mean(mov)
    fix_square = fix_means**2
    mov_square = mov_means**2
    fix_mov_product = fix_means * mov_means
    fix_means = _local_means(fix_means, radius)
    mov_means = _local_means(mov_means, radius)
    fix_var = _local_means(fix_square, radius) - fix_means**2
    mov_var = _local_means(mov_square, radius) - mov_means**2
    fix_mov_cov = _local_means(fix_mov_product, radius) - fix_means*mov_means

    # compute LCCs
    with np.errstate(divide='ignore', invalid='ignore'):
        lcc = fix_mov_cov**2 / (fix_var * mov_var)

    # replace NaNs (occur when there is no data, or data values are constant)
    mn, mx = np.percentile(fix, [0.1, 99.9])
    fix_mask = fix_var / (mx - mn) < tolerance
    mn, mx = np.percentile(mov, [0.1, 99.9])
    mov_mask = mov_var / (mx - mn) < tolerance
    lcc[fix_mask + mov_mask] = 0.
    
    # return
    if return_image:
        return lcc.mean(), lcc.astype(np.float32)
    else:
        return lcc.mean()


def _local_means(image, radius):

    # create a high precision summed area table (normalized by neighborhood volume)
    image = image.astype(np.longdouble) / np.prod(2 * np.array(radius) + 1)
    sat = np.pad(image, tuple((r+1, r) for r in radius), mode='reflect')
    for iii in range(image.ndim):
        sat.cumsum(axis=iii, out=sat)

    # take appropriate differences to get local sums (actually means because already normalized)
    binary_strings = ["".join(x) for x in product("01", repeat=image.ndim)]
    means = np.copy(sat[_get_crop(binary_strings.pop(-1), radius)])
    for binary_string in binary_strings:
        sign = (-1)**(image.ndim - np.sum([int(x) for x in binary_string]))
        crop = _get_crop(binary_string, radius)
        means += sign * sat[crop]
    return means


def _get_crop(binary_string, radius):

    crop = []
    for bit, r in zip(binary_string, radius):
        if bit == "0": crop.append( slice(None, -2*r - 1, None) )
        else: crop.append( slice(2*r + 1, None, None) )
    return tuple(crop)


@cluster
def roi_correlations(
    fix,
    mov,
    rois,
    radius=None,
    cluster=None,
    cluster_kwargs={},
    temporary_directory=None,
):
    """
    Compute the correlation between fixed and moving in all the
    given ROIs. Distributed.

    Parameters
    ----------
    fix : ndarray
        The fixed image data

    mov : ndarray
        The moving image data

    rois : list of tuples of slices
        The ROIs. A single ROI should be a tuple of slices, one
        per axis of fix/mov.

    radius : int or tuple of int (default: None)
        How much to extend the ROI along each axis.

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

    Returns
    -------
    correlations : 1d numpy array of floats
        The correlation between fix and mov in all rois
    """

    # ensure zarr images
    temporary_directory = tempfile.TemporaryDirectory(
        prefix='.', dir=temporary_directory or os.getcwd(),
    )
    zarr_blocks = [128,]* fix.ndim
    fix_zarr_path = temporary_directory.name + '/fix.zarr'
    mov_zarr_path = temporary_directory.name + '/mov.zarr'
    fix_zarr = ut.numpy_to_zarr(fix, zarr_blocks, fix_zarr_path)
    mov_zarr = ut.numpy_to_zarr(mov, zarr_blocks, mov_zarr_path)

    # ensure radius is a tuple
    if radius is not None and not isinstance(radius, tuple):
        radius = (radius,) * fix.ndim

    # record shape
    full_shape = fix.shape

    def roi_correlation(roi):

        # adjust for radius
        if radius is not None:
            new_roi = []
            for s, r, sh in zip(roi, radius, full_shape):
                new_roi.append(slice(max(s.start - r, 0), min(s.stop + r, sh)))
            roi = tuple(new_roi)

        # crop and flatten the data
        fix_crop = fix_zarr[roi].flatten()
        mov_crop = mov_zarr[roi].flatten()

        # return correlation
        return np.corrcoef(fix_crop, mov_crop)[0, 1]

    # run everything in parallel
    futures = cluster.client.map(roi_correlation, rois)
    return np.array(cluster.client.gather(futures))

