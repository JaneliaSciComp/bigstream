import logging
import numpy as np

from scipy.ndimage import binary_closing, binary_dilation, label, zoom
from scipy.ndimage.filters import gaussian_filter

from . import level_set


logger = logging.getLogger(__name__)


def generate_foreground_mask(image,
                             image_spacing,
                             image_subsampling=(4,4,4),
                             mask_smoothing=2,
                             iterations=[40,20,10],
                             smooth_sigmas=[32,24,16],
                             shrink_factors=(4,2,1),
                             lambda1=1,
                             lambda2=10,
                             background=None,
                             percentile_thresh=None,
                             final_closing=(5,5,5),
                             final_dilation=(10,10,10)):
    subsampled_image = image[::image_subsampling[0], ::image_subsampling[1], ::image_subsampling[2]]
    subsampled_image_spacing = image_spacing * image_subsampling
    logger.debug((
        f'Sample {image.shape} image with resolution {image_spacing} '
        f'at {image_subsampling} -> {subsampled_image.shape}, new resolution {subsampled_image_spacing} '
        f'using smooth_sigmas: {smooth_sigmas} and iterations: {iterations}'
    ))
    if percentile_thresh is None:
        logger.info((
            'Generate foreground mask using level_set.foreground_segmentation '
            f'mask_smoothing: {mask_smoothing}, '
            f'iterations: {iterations}, '
            f'shrink factors: {shrink_factors}, '
            f'lambda1: {lambda1}, '
            f'lambda2: {lambda2}, '
            f'background: {background}, '
        ))
        mask, background = level_set.foreground_segmentation(
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
    mask = binary_closing(mask, np.ones(final_closing)).astype(np.uint8)
    mask = binary_dilation(mask, np.ones(final_dilation)).astype(np.uint8)
    mask = zoom(mask, np.array(image.shape) / subsampled_image.shape, order=0)
    mask_spacing = subsampled_image_spacing / image_subsampling
    if mask.any():
        logger.info((
            f'Complete foreground mask with shape {mask.shape} for {image.shape} image '
            f'mask spacing: {mask_spacing}, image spacing: {image_spacing} '
        ))
        _mask_report(image, mask, background=(background if background is not None else 0))
    else:
        logger.warning(f'No foreground mask found for {image.shape} image')
    return mask, mask_spacing


def _mask_report(image, mask, background=0):
    signal = image > background
    inside = np.logical_and(signal, mask > 0).sum()
    # label on a 2x-decimated copy - this is good enough for component count
    _, n_components = label(mask[::2, ::2, ::2] > 0)
    logger.info((
        f'Coverage {mask.mean() * 100:.1f}%, '
        f'signal captured {inside / max(signal.sum(), 1) * 100:.1f}%, '
        f'{n_components} components '
    ))
