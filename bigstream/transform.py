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


def apply_global_affine(fix, mov, fix_spacing, mov_spacing, affine, order=1):
    """
    """

    grid = position_grid(fix.shape) * fix_spacing
    coords = affine_to_grid(affine, grid)
    return interpolate_image(mov, coords, order=order) 
