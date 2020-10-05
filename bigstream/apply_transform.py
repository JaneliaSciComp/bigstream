import numpy as np
from scipy.ndimage import map_coordinates


def position_grid(shape, dtype=np.uint16):
    """
    """

    coords = np.array(
        np.meshgrid(*[range(x) for x in shape], indexing='ij'),
        dtype=dtype,
    )
    return np.ascontiguousarray(np.moveaxis(coords, 0, -1))


def affine_to_grid(matrix, grid):
    """
    """

    mm = matrix[:3, :-1]
    tt = matrix[:3, -1]
    return np.einsum('...ij,...j->...i', mm, grid) + tt


def interpolate_image(image, X, order=1):
    """
    """

    X = np.moveaxis(X, -1, 0)
    return map_coordinates(image, X, order=order, mode='constant')



