import numpy as np
import cv2


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

    
