from scipy import spatial
from scipy.ndimage import convolve
from scipy.ndimage.filters import maximum_filter
from scipy.stats import multivariate_normal
import numpy as np
from skimage.feature.blob import _blob_overlap


def difference_of_gaussians_3d(image, big_sigma, small_sigma):
    """
    """

    x = int(round(2*big_sigma))
    x = np.arange(-x, x+.5, 1)
    x = np.moveaxis(np.array(np.meshgrid(x, x, x)), 0, -1)
    g1 = multivariate_normal.pdf(x, mean=[0,]*3, cov=np.diag([small_sigma,]*3))
    g2 = multivariate_normal.pdf(x, mean=[0,]*3, cov=np.diag([big_sigma,]*3))
    return convolve(image, g1 - g2)


def local_max(image, min_distance=3, threshold=None):
    """
    """

    image_max = maximum_filter(image, min_distance)
    if threshold is not None:
        mask = np.logical_and(image == image_max, image >= threshold)
    else:
        mask = image == image_max
    return column_stack(np.nonzero(mask))


def dog_filter_3d(
    image,
    big_sigma=1.6,
    small_sigma=1,
    threshold=2.0,
    min_distance=5,
    ):
    """
    """

    # get dog locations with intensity greater than threshold
    dog = difference_of_gaussians_3d(image, big_sigma, small_sigma)
    coord = local_max(dog, min_distance=min_distance)
    intensities = image[coord[:,0], coord[:,1], coord[:,2]]
    filtered = intensities > threshold

    # concat all results and return
    coord = coord[filtered]
    intensities = intensities[filtered][..., np.newaxis]
    small_sigma_array = np.full((sum(filtered), 1), small_sigma)
    big_sigma_array = np.full((sum(filtered), 1), big_sigma_array)
    return np.hstack((coord, intensities, small_sigma_array, big_sigma_array))


def get_context(image, position, radius):
    """
    """

    p, r = position, radius  # shorthand
    w = image[p[0]-r:p[0]+r+1, p[1]-r:p[1]+r+1, p[2]-r:p[2]+r+1]
    if np.product(w.shape) != (2*r+1)**3:  # just ignore near the edge
        return None
    else:
        return w


def get_all_context(image, spots, radius):
    """
    """

    output = []
    for spot in spots:
        context = get_context(image, spot, radius)
        if context is not None:
            output.append([spot, context])
    return output


def prune_overlaps():
