import numpy as np
import cv2


def stats(arr):
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
    a_mean, a_std = stats(a_con)
    b_mean, b_std = stats(b_con)
    a_con = a_con - a_mean[..., None]
    b_con = b_con - b_mean[..., None]

    # compute pairwise correlations
    corr = np.matmul(a_con, b_con.T) / a_std[..., None] / b_std[None, ...] / a_con.shape[1]

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


def ransac_align_points(
    pA, pB, threshold, diagonal_constraint=0.75, default=np.eye(4)[:3],
):
    """
    """

    # sensible requirement of 51 or more spots to compute ransac affine
    if len(pA) <= 50 or len(pB) <= 50:
        if default is not None:
            print("Insufficient spot matches for ransac, returning default identity")
            return default
        else:
            raise ValueError("Insufficient spot matches for ransac, need more than 50")

    # compute the affine
    r, Aff, inline = cv2.estimateAffine3D(pA, pB, ransacThreshold=threshold, confidence=0.999)

    # rarely ransac just doesn't work (depends on data and parameters)
    # sensible choices for hard constraints on the affine matrix
    if np.any( np.diag(Aff) < diagonal_constraint ):
        if default is not None:
            print("Degenerate affine produced, returning default identity")
            return default
        else:
            raise ValueError("Degenerate affine produced, ransac failed")

    return Aff

    
