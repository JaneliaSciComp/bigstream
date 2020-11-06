import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree
from skimage.feature.blob import _blob_overlap


def difference_of_gaussians_3d(image, spacing, big_sigma, small_sigma):
    """
    """

    spacing = np.array(spacing)
    g1 = gaussian_filter(image, small_sigma/spacing)
    g2 = gaussian_filter(image, big_sigma/spacing)
    return g1 - g2


def local_max_points(image, min_distance=3, threshold=None):
    """
    """

    image_max = maximum_filter(image, min_distance)
    if threshold is not None:
        mask = np.logical_and(image == image_max, image >= threshold)
    else:
        mask = image == image_max
    return np.column_stack(np.nonzero(mask))


def dog_filter_3d(
    image,
    image_spacing,
    big_sigma=6.0,
    small_sigma=3.5,
    threshold=2.0,
    min_distance=5,
    ):
    """
    """

    # get dog locations with intensity greater than threshold
    dog = difference_of_gaussians_3d(image, image_spacing, big_sigma, small_sigma)
    coord = local_max_points(dog, min_distance=min_distance)
    intensities = image[coord[:,0], coord[:,1], coord[:,2]]
    filtered = intensities > threshold

    # concat all results and return
    coord = coord[filtered] * image_spacing  # points are in physical units
    intensities = intensities[filtered][..., np.newaxis]
    small_sigma_array = np.full((sum(filtered), 1), small_sigma)
    big_sigma_array = np.full((sum(filtered), 1), big_sigma)
    return np.hstack((coord, intensities, small_sigma_array, big_sigma_array))


def prune_neighbors(spots, overlap=0.01, distance=6):
    """
    """

    # get points within distance of each other
    tree = cKDTree(spots[:, :-2])
    pairs = np.array(list(tree.query_pairs(distance)))

    # tag gaussian blobs that overlap too much
    c = [0, 1, 2, 4]  # fancy index for coordinate + small_sigma
    for (i, j) in pairs:
        if _blob_overlap(spots[i, c], spots[j, c]) > overlap:
            if spots[i, 3] > spots[j, 3]:
                spots[j, 3] = 0
            else:
                spots[i, 3] = 0

    # remove tagged spots and return as array
    return np.array([s for s in spots if s[3] > 0])


def get_context(image, position, radius):
    """
    """

    p, r = np.round(position).astype(int), radius  # shorthand
    w = image[p[0]-r:p[0]+r+1, p[1]-r:p[1]+r+1, p[2]-r:p[2]+r+1]
    if np.product(w.shape) != (2*r+1)**3:  # just ignore near the edge
        w = None
    return w


def get_all_context(image, spots, vox, radius):
    """
    """

    output = []
    for spot in spots:
        spot[:3] = spot[:3] / vox  # points were in physical units, need voxel units
        context = get_context(image, spot, radius)
        if context is not None:
            output.append([spot, context])
    return output    

