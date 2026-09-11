import logging
import numpy as np
import morphsnakes

from scipy.ndimage import zoom
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_erosion, binary_dilation, binary_fill_holes
from scipy.ndimage.measurements import label, labeled_comprehension

from skimage import filters


logger = logging.getLogger(__name__)


def estimate_background(image, ignore_zeros=True, zero_frac_thresh=0.01):
    """
    Estimate the background intensity level of an image with the triangle
    threshold.

    Exact zeros are dropped before histogramming when they account for more
    than zero_frac_thresh of the voxels. Stitched, rotated or ROI extracted
    volumes carry large out of FOV regions of exact zeros, and
    threshold_triangle anchors its construction line on the histogram peak -
    when the zero bin is the peak the lever arm collapses and the estimate
    drops towards zero, which floods the mask.

    Parameters
    ----------
    image : nd array
        The image

    ignore_zeros : bool (default: True)
        Whether to drop exact zeros before estimating

    zero_frac_thresh : scalar float (default: 0.01)
        Only drop exact zeros when they account for more than this fraction of
        the voxels - an image can legitimately hold a few zero valued voxels

    Returns
    -------
    background_estimate : scalar float
        The estimated background intensity level
    """

    samples = np.asarray(image).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if ignore_zeros and samples.size > 0:
        zero_frac = np.count_nonzero(samples == 0) / samples.size
        if zero_frac > zero_frac_thresh:
            logger.info((
                f'Exclude {zero_frac * 100:.1f}% exact zero voxels '
                '(out of FOV padding) from the background estimate'
            ))
            samples = samples[samples != 0]
    if samples.size == 0:
        logger.warning('No valid voxels to estimate the background from - use 0')
        return 0.
    if samples.min() == samples.max():
        logger.warning(f'Constant image ({samples.min()}) - use it as the background')
        return float(samples.min())
    return float(filters.threshold_triangle(samples))


def segment(
    image,
    lambda1,
    lambda2,
    iterations,
    smoothing=1,
    threshold=None,
    init=None
):
    """
    A wrapper for morphsnakes.morphological_chan_vese
    Allows an initial intensity threshold and creates a simple initialization if one
    is not given

    Parameters
    ----------
    image : nd-array
        The image whose foreground you want to segment

    lambda1 : scalar float
        Controls how it penalizes the inside

    lambda2 : scalar float
        Controls variance of foreground region. A larger number means a larger segment.

    iterations : scalar int
        The maximum number of iterations to run the morphological_chan_vese algorithm

    smoothing : scalar int (default: 1)
        The number of times to apply morphological smoothing to the foreground mask
        each iteration. Larger numbers mean smoother mask boundaries, but also take
        a lot more time. Reasonable values are [0, 4]

    threshold : scalar float (default: None)
        An intensity threshold to apply to the data before segmenting

    init : binary nd-array (default: None)
        Optional initialization for the level set. Must be the same shape as image.

    Returns
    -------
    foreground_mask : binary nd-array
        Foreground segmentation, same shape as image, uint8
    """

    if threshold is not None:
        # never write into the caller's array
        image = np.where(image < threshold, 0, image)
    if init is None:
        if threshold is not None:
            # seed from the image's own intensity so
            # every region already above threshold is included
            init = (image > 0).astype(np.uint8)
            logger.info(f'Seed initial mask from threshold {threshold}: {init.sum()} voxels')
        else:
            init = np.zeros_like(image, dtype=np.uint8)
            bounds = np.ceil(np.array(image.shape) * 0.1).astype(int)
            init[tuple(slice(b, -b) for b in bounds)] = 1
    else:
        logger.info(f'Use initial mask: {init.shape}')

    logger.info((
        f'Morphological segmentation: iterations: {iterations} '
        f'smoothing: {smoothing}, lambda2: {lambda2}, lambda1: {lambda1} '
    ))

    return morphsnakes.morphological_chan_vese(
        image,
        iterations,
        init_level_set=init,
        smoothing=smoothing,
        lambda1=lambda1,
        lambda2=lambda2,
    ).astype(np.uint8)


def largest_connected_component(mask):
    """
    Return only the largest connected component from a foreground segmentation

    Parameters
    ----------
    mask : binary nd-array
        A foreground segmentation

    Returns
    -------
    new_mask : binary nd-array
        The same mask as the input, but with only the largest connected component
        present.
    """

    lbls, nlbls = label(mask)
    vols = labeled_comprehension(mask, lbls, range(1, nlbls+1), np.sum, float, 0)
    mask[lbls != np.argmax(vols)+1] = 0
    return mask


def foreground_segmentation(
    image,
    voxel_spacing,
    iterations=(40,8,2),
    shrink_factors=(4,2,1),
    smooth_sigmas=(8.,4.,2.),
    lambda1=1.,
    lambda2=1.,
    background=None,
    return_largest_cc_only=True,
    mask=None,
    mask_smoothing=1,
):
    """
    Multiscale foreground detection - runs segment at multiple resolutions. The lengths
    of the iterations, shrink_factors, and smooth_sigmas parameters must all be the same
    and determines the number of scales used.

    Parameters
    ----------
    image : nd-array
        The image whose foreground you want to segment

    voxel_spacing : 1d-array
        The physical sampling rate of the image

    iterations : tuple of int (default: (40, 8, 2))
        The number of iterations to run at each scale.

    shrink_factors : tuple of int (default: (4, 2, 1))
        The downsampling factors to use at each level

    smooth_sigmas : tuple of float (default: (8., 4., 2.))
        The gaussian_smoothing kernel width to use at each scale in physical units

    lambda1 : scalar float (default: 1.)

    lambda2 : scalar float (default: 20.)
        Controls variance of foreground region. A larger number means a larger segment.

    background : scalar float (default: None)
        The background intensity level. Voxels below it are zeroed before
        segmenting and, when no mask is given, seed the level set. If None it
        is estimated with level_set.estimate_background - once on the input
        image, which is the value returned to the caller, and again at every
        scale on the smoothed and decimated image actually being segmented.

    return_largest_cc_only : bool (default: True)
        If true, final mask is eroded, then connected components are found,
        only the largest cc is dilated and retained. If false, the entire
        final level set is returned, even if the topology changed (e.g. multiple
        discontinuous regions were found)

    mask : binary nd-array (default: None)
        Optional initialization for the level set. Must be the same shape as image.

    mask_smoothing : scalar int (default: 1)
        The number of times to apply morphological smoothing to the foreground mask
        each iteration. Larger numbers mean smoother mask boundaries, but also take
        a lot more time. Reasonable values are [0, 4]

    Returns
    -------
    foreground_mask : binary nd-array
        Foreground segmentation, same shape as image, uint8

    background : scalar float
        The background level resolved for the input image - either the value
        given by the caller or the one estimated from image
    """

    # resolve the background once, on the image as given, before any smoothing
    # or decimation, so the value returned describes the data the caller passed
    explicit_background = background is not None
    if explicit_background:
        logger.info(f'Use the given background level: {background}')
    else:
        background = estimate_background(image)
        logger.info(f'Estimated background for {image.ndim}-D image: {background}')

    # segment
    seg_iter_params = list(zip(iterations, shrink_factors, smooth_sigmas))
    logger.debug(f'Segmentation iteration params (iter/shrink_factor/sigma): {seg_iter_params}')
    for its, sf, ss in seg_iter_params:
        logger.debug(f'Apply gaussian: {ss}/{voxel_spacing} -> {ss/voxel_spacing}, shrink factor: {sf}')
        smoothed_image = gaussian_filter(image, ss/voxel_spacing) if ss > 0 else image
        # mode='nearest': the default 'constant'/cval=0 zero-fills the outer
        # boundary planes for some shape/factor combinations (e.g. 48 -> 24),
        # which fakes up a block of background voxels the image never had
        image_small = zoom(smoothed_image, 1./sf, order=3, mode='nearest')
        if mask is not None:
            zoom_factors = [x/y for x, y in zip(image_small.shape, mask.shape)]
            logger.debug(f'Zoom factors: {zoom_factors}')
            mask = zoom(mask, zoom_factors, order=0, mode='nearest')

        if explicit_background:
            scale_background = background
        else:
            # re-estimate on the image that is actually segmented: smoothing
            # and decimation narrow the background peak, so a level taken at
            # the coarsest scale sits inside the noise at the finest one
            scale_background = estimate_background(image_small)
            logger.debug((
                f'Background for scale (sigma: {ss}, shrink: {sf}) '
                f'{image_small.shape}: {scale_background}'
            ))

        mask = segment(
            image_small,
            lambda1,
            lambda2,
            its,
            smoothing=mask_smoothing,
            threshold=scale_background,
            init=mask,
        )

    # basic topological correction
    if return_largest_cc_only:
        mask = binary_erosion(mask, iterations=2)
        mask = largest_connected_component(mask)
        mask = binary_dilation(mask, iterations=2)

    mask = binary_fill_holes(mask).astype(np.uint8)

    # ensure output is on correct grid
    if mask.shape != image.shape:
        to_reshape = mask.shape
        zoom_factors = [x/y for x, y in zip(image.shape, to_reshape)]
        mask = zoom(mask, zoom_factors, order=0, mode='nearest')
        logger.info(f'Final mask reshape {to_reshape} to {image.shape} -> {mask.shape}')
    return mask, background


