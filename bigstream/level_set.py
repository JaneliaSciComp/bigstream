import numpy as np
from scipy.ndimage import zoom
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_erosion, binary_dilation, binary_fill_holes
from scipy.ndimage.measurements import label, labeled_comprehension
import morphsnakes


def estimate_background(image, rad=5):
    """
    Estimate typical background intensity value from the 8 corners of the image

    Parameters
    ----------
    image : 3d array
        The image

    rad : int (default: 5)
        The length of cubes samples from corners

    Returns
    -------
    background_estimate : scalar of image.dtype
        The estimated background value
    """

    a, b = slice(0, rad), slice(-rad, None)
    corners = ((a,a,a), (a,a,b), (a,b,a), (a,b,b),
               (b,a,a), (b,a,b), (b,b,a), (b,b,b))
    return np.median([np.mean(image[c]) for c in corners])


def segment(
    image,
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
        image[image < threshold] = 0
    if init is None:
        init = np.zeros_like(image, dtype=np.uint8)
        bounds = np.ceil(np.array(image.shape) * 0.1).astype(int)
        init[tuple(slice(b, -b) for b in bounds)] = 1
    return morphsnakes.morphological_chan_vese(
        image,
        iterations,
        init_level_set=init,
        smoothing=smoothing,
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
    lambda2=20.,
    background=None,
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
        The physical sampling rate of the image (if unknown, use [1, 1, 1, ...])
    iterations : tuple of int (default: (40, 8, 2))
        The number of iterations to run at each scale.
    shrink_factors : tuple of int (default: (4, 2, 1))
        The downsampling factors to use at each level
    smooth_sigmas : tuple of float (default: (8., 4., 2.))
        The gaussian_smoothing kernel width to use at each scale in physical unit
        This is relative to the given voxel_spacing
    lambda2 : scalar float (default: 20.)
        Controls variance of foreground region. A larger number means a larger segment.
    background : scalar float (default: None)
        An estimate of the average background intensity value. If None, it will
        automatically be estimated from the image.
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
    """

    # segment
    if background is None:
        background = estimate_background(image)
    for its, sf, ss in zip(iterations, shrink_factors, smooth_sigmas):
        image_small = zoom(gaussian_filter(image, ss/voxel_spacing), 1./sf, order=1)
        if mask is not None:
            zoom_factors = [x/y for x, y in zip(image_small.shape, mask.shape)]
            mask = zoom(mask, zoom_factors, order=0)
        mask = segment(
            image_small,
            lambda2=lambda2,
            iterations=its,
            smoothing=mask_smoothing,
            threshold=background,
            init=mask,
        )

    # basic topological correction
    mask = binary_erosion(mask, iterations=2)
    mask = largest_connected_component(mask)
    mask = binary_dilation(mask, iterations=2)
    mask = binary_fill_holes(mask).astype(np.uint8)

    # ensure output is on correct grid
    if mask.shape != image.shape:
        zoom_factors = [x/y for x, y in zip(image.shape, mask.shape)]
        mask = zoom(mask, zoom_factors, order=0)
    return mask


