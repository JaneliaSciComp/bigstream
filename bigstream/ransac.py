import numpy as np
import cv2


def ransac_align_points(
    pA, pB, 
    threshold,
    diagonal_constraint=0.75,
    default=np.eye(4),
):
    """
    """

    # sensible requirement of 50 or more spots to compute ransac affine
    if len(pA) < 50 or len(pB) < 50:
        if default is not None:
            print("Insufficient spot matches for ransac")
            print("Returning default")
            return default
        else:
            message = "Insufficient spot matches for ransac"
            message += ", need 50 or more"
            raise ValueError(message)

    # compute the affine
    r, Aff, inline = cv2.estimateAffine3D(
        pA, pB,
        ransacThreshold=threshold,
        confidence=0.999,
    )

    # rarely ransac just doesn't work (depends on data and parameters)
    # sensible choices for hard constraints on the affine matrix
    if np.any( np.diag(Aff) < diagonal_constraint ):
        if default is not None:
            print("Degenerate affine produced")
            print("Returning default")
            return default
        else:
            message = "Degenerate affine produced"
            message += ", ransac failed"
            raise ValueError(message)

    # augment affine to 4x4 matrix
    affine = np.eye(4)
    affine[:3, :] = Aff

    return affine

    
