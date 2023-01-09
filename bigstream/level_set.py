import numpy as np
from scipy.ndimage import zoom
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_erosion, binary_dilation, binary_fill_holes
from scipy.ndimage.measurements import label, labeled_comprehension
import morphsnakes


def estimate_background(image, rad=5):
    """
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
    lbls, nlbls = label(mask)
    vols = labeled_comprehension(mask, lbls, range(1, nlbls+1), np.sum, float, 0)
    mask[lbls != np.argmax(vols)+1] = 0
    return mask


def brain_detection(
    image,
    voxel_spacing,
    iterations=[40,8,2],
    shrink_factors=[4,2,1],
    smooth_sigmas=[8,4,2],
    lambda2=20,
    background=None,
    mask=None,
    mask_smoothing=1,
    ):
    """
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


