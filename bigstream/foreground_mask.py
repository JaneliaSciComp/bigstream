import numpy as np

from scipy.ndimage import zoom, binary_closing, binary_dilation
from . import level_set


def generate_foreground_mask(image,
                             image_spacing,
                             image_subsampling=4,
                             mask_smoothing=2,
                             iterations=[80,40,10],
                             smooth_sigmas=[32,24,16],
                             lambda2=1.5,
                             return_largest_cc_only=False):

    print('!!!!! IMAGE SHAPE ', image.shape)
    print('!!!!!! IMAGE SPACING ', image_spacing)
    subsampled_image = image[::image_subsampling, ::image_subsampling, ::image_subsampling]
    subsampled_image_spacing = image_spacing * image_subsampling
    mask = level_set.foreground_segmentation(
        subsampled_image, subsampled_image_spacing,
        mask_smoothing=mask_smoothing,
        iterations=iterations,
        smooth_sigmas=smooth_sigmas,
        lambda2=lambda2,
        return_largest_cc_only=return_largest_cc_only,
    )
    # enlarge and smooth mask
    mask = binary_closing(mask, np.ones((5,5,5))).astype(np.uint8)
    mask = binary_dilation(mask, np.ones((10,10,10))).astype(np.uint8)
    mask = zoom(mask, np.array(mask.shape) / subsampled_image.shape, order=0)

    return mask
