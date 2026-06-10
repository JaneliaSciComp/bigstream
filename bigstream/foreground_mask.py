import logging
import numpy as np

from scipy.ndimage import zoom, binary_closing, binary_dilation
from scipy.ndimage.filters import gaussian_filter

from . import level_set


logger = logging.getLogger(__name__)


def generate_foreground_mask(image,
                             image_spacing,
                             image_subsampling=4,
                             mask_smoothing=2,
                             iterations=[40,20,10],
                             smooth_sigmas=[32,24,16],
                             shrink_factors=(4,2,1),
                             lambda1=1,
                             lambda2=10,
                             background=None,
                             percentile_thresh=None):
    subsampled_image = image[::image_subsampling, ::image_subsampling, ::image_subsampling]
    subsampled_image_spacing = image_spacing * image_subsampling
    logger.debug((
        f'Sample {image.shape} image with resolution {image_spacing} '
        f'at {image_subsampling} -> {subsampled_image.shape}, new resolution {subsampled_image_spacing} '
        f'using smooth_sigmas: {smooth_sigmas} and iterations: {iterations}'
    ))
    if percentile_thresh is None:
        logger.info('Generate foreground mask using level_set.foreground_segmentation')
        mask = level_set.foreground_segmentation(
            subsampled_image, subsampled_image_spacing,
            mask_smoothing=mask_smoothing,
            iterations=iterations,
            shrink_factors=shrink_factors,
            smooth_sigmas=smooth_sigmas,
            lambda1=lambda1,
            lambda2=lambda2,
            background=background,
            return_largest_cc_only=False,
        )
    else:
        # only get a threshold and apply a mask for that threshold
        thresh = np.percentile(subsampled_image, percentile_thresh)
        logger.info(f'Use a threshold of {thresh} for {percentile_thresh}th percentile to determine the mask')
        mask = gaussian_filter(subsampled_image, smooth_sigmas[-1]) > thresh

    # enlarge and smooth mask
    mask = binary_closing(mask, np.ones((10,10,10))).astype(np.uint8)
    mask = binary_dilation(mask, np.ones((5,5,5))).astype(np.uint8)
    mask = zoom(mask, np.array(image.shape) / subsampled_image.shape, order=0)
    return mask, subsampled_image_spacing / image_subsampling
