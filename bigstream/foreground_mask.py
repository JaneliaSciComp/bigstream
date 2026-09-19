import logging
import numpy as np

from scipy.ndimage import (binary_closing, binary_dilation, gaussian_filter,
                           label, zoom)

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
                             normalize=False,
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

        if normalize:
            subsampled_image = _normalize_image(subsampled_image, shrink_factor=1, num_fitting_levels=2)
            logger.info(f'Normalized {subsampled_image.shape} image')

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
        # report against the threshold that actually produced the mask,
        # otherwise _mask_report compares the image against 0
        background = thresh

    # enlarge and smooth mask
    mask = binary_closing(mask, np.ones(final_closing)).astype(np.uint8)
    mask = binary_dilation(mask, np.ones(final_dilation)).astype(np.uint8)
    # mode='nearest' so the default cval=0 does not shave the mask's outer
    # boundary planes off on the way back up to full resolution
    mask = zoom(mask, np.array(image.shape) / subsampled_image.shape, order=0,
                mode='nearest')
    mask_spacing = subsampled_image_spacing / image_subsampling
    if mask.any():
        logger.info((
            f'Complete foreground mask with shape {mask.shape} for {image.shape} image '
            f'mask spacing: {mask_spacing}, image spacing: {image_spacing} '
        ))
        _mask_report(image, mask, background=(background if background is not None else 0))
    else:
        logger.warning(f'No foreground mask found for {image.shape} image')
    return mask, mask_spacing, background


def _normalize_image(volume, shrink_factor=4, num_fitting_levels=4, mask=None):
    import SimpleITK as sitk

    vol = volume.astype(np.float32)
    image = sitk.GetImageFromArray(vol)

    if mask is None:
        mask_image = sitk.OtsuThreshold(image, 0, 1, 200)
    else:
        mask_image = sitk.GetImageFromArray(mask.astype(np.uint8))

    # Fit the bias field on a downsampled copy for speed...
    shrunk_image = sitk.Shrink(image, [shrink_factor] * image.GetDimension())
    shrunk_mask = sitk.Shrink(mask_image, [shrink_factor] * image.GetDimension())

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50] * num_fitting_levels)
    _ = corrector.Execute(shrunk_image, shrunk_mask)
 
    # ...then apply the fitted field at full resolution
    log_bias_field = corrector.GetLogBiasFieldAsImage(image)
    normalized_image = image / sitk.Exp(log_bias_field)
 
    normalized = sitk.GetArrayFromImage(normalized_image)

    return normalized


def _mask_report(image, mask, background=0):
    signal = image > background
    inside = np.logical_and(signal, mask > 0).sum()
    # label on a 2x-decimated copy - this is good enough for component count
    _, n_components = label(mask[::2, ::2, ::2] > 0)
    logger.info((
        f'Background {background:.2f}, '
        f'coverage {mask.mean() * 100:.1f}%, '
        f'signal captured {inside / max(signal.sum(), 1) * 100:.1f}%, '
        f'{n_components} components '
    ))
