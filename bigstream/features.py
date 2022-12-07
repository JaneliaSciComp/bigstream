import numpy as np
from fishspot.filter import white_tophat
from fishspot.detect import detect_spots_log


def blob_detection(
    image,
    min_blob_radius,
    max_blob_radius,
    **kwargs,
):
    """
    """

    wth = white_tophat(image, max_blob_radius)
    spots = detect_spots_log(
        wth,
        min_blob_radius,
        max_blob_radius,
        **kwargs,
    ).astype(int)
    intensities = image[spots[:, 0], spots[:, 1], spots[:, 2]]
    return np.hstack((spots[:, :3], intensities[..., None]))


def get_contexts(image, coords, radius):
    """
    """

    contexts = []
    for coord in coords:
        crop = tuple(slice(x-radius, x+radius+1) for x in coord)
        contexts.append(image[crop])
    return contexts    


def _stats(arr):
    """
    """

    # compute mean and standard deviation along columns
    arr = arr.astype(np.float64)
    means = np.mean(arr, axis=1)
    sqr_means = np.mean(np.square(arr), axis=1)
    stddevs = np.sqrt( sqr_means - np.square(means) )
    return means, stddevs


def pairwise_correlation(A, B):
    """
    """

    # flatten contexts into array
    a_con = np.array( [a.flatten() for a in A] )
    b_con = np.array( [b.flatten() for b in B] )

    # get means and std for all contexts, center contexts
    a_mean, a_std = _stats(a_con)
    b_mean, b_std = _stats(b_con)
    a_con = a_con - a_mean[..., None]
    b_con = b_con - b_mean[..., None]

    # compute pairwise correlations
    corr = np.matmul(a_con, b_con.T)
    corr = corr / a_std[..., None]
    corr = corr / b_std[None, ...]
    corr = corr / a_con.shape[1]

    # contexts with no variability are nan, set to 0
    corr[np.isnan(corr)] = 0
    return corr


def match_points(a_pos, b_pos, scores, threshold):
    """
    """

    # get highest scores above threshold
    best_indcs = np.argmax(scores, axis=1)
    a_indcs = range(len(a_pos))
    keeps = scores[(a_indcs, best_indcs)] > threshold

    # return positions of corresponding points
    return a_pos[keeps, :3], b_pos[best_indcs[keeps], :3]

