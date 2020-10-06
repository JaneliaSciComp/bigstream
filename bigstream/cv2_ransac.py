import numpy as np
import cv2


def stats(arr):
    """
    """

    arr = arr.astype(np.float64)
    means = np.mean(arr, axis=1)
    sqr_means = np.mean(np.square(arr), axis=1)
    stddevs = np.sqrt( sqr_means - np.square(means) )
    return means, stddevs


def pairwise_correlation(A, B):
    """
    """

    # split positions from context, flatten context
    a_pos = np.array( [a[0] for a in A] )
    a_con = np.array( [a[1].flatten() for a in A] )
    b_pos = np.array( [b[0] for b in B] )
    b_con = np.array( [b[1].flatten() for b in B] )

    # get means and std for all contexts, center contexts
    a_mean, a_std = stats(a_con)
    b_mean, b_std = stats(b_con)
    a_cent = a_con - a_mean[..., None]
    b_cent = b_con - b_mean[..., None]

    # compute pairwise correlations
    correlations = np.matmul(a_cent, b_cent.T) / a_std[..., None] / b_std[None, ...] / a_cent.shape[1]
    correlations[np.isnan(correlations)] = 0
    return correlations


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


def ransac_align_points(pA, pB, threshold):
    """
    """

    r, Aff, inline = cv2.estimateAffine3D(pA, pB, ransacThreshold=threshold, confidence=0.999)
    return Aff

    
