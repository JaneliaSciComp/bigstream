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


def get_spot_context(image, spots, vox, radius):
    """
    """

    output = []
    for spot in spots:
        s = (spot/vox).astype(int)
        w = image[s[0]-radius:s[0]+radius+1,
                  s[1]-radius:s[1]+radius+1,
                  s[2]-radius:s[2]+radius+1]
        output.append( [spot, w] )
    return output    


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

    # grab and flatten context
    a_con = np.array( [a[1].flatten() for a in A] )
    b_con = np.array( [b[1].flatten() for b in B] )

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


def match_points(A, B, scores, threshold):
    """
    """

    # split positions from context
    a_pos = np.array( [a[0] for a in A] )
    b_pos = np.array( [b[0] for b in B] )

    # get highest scores above threshold
    best_indcs = np.argmax(scores, axis=1)
    a_indcs = range(len(a_pos))
    keeps = scores[(a_indcs, best_indcs)] > threshold

    # return positions of corresponding points
    return a_pos[keeps, :3], b_pos[best_indcs[keeps], :3]

