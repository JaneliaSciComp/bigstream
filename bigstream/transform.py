import numpy as np
from scipy.ndimage import map_coordinates
import zarr
from numcodecs import Blosc
from bigstream import distributed
import dask.array as da


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
    coords = affine_to_grid(affine, grid) / mov_spacing
    return interpolate_image(mov, coords, order=order)


# DASK versions for big data
def position_grid_dask(shape, dtype=np.uint16):
    """
    """

    coords = da.from_array(
        da.meshgrid(*[range(x) for x in shape], indexing='ij'),
    )
    return da.moveaxis(coords, 0, -1)


def affine_to_grid_dask(matrix, grid):
    """
    """

    mm = matrix[:3, :-1]
    tt = matrix[:3, -1]
    return da.einsum('...ij,...j->...i', mm, grid, dtype=np.float32) + tt



def global_affine_to_position_field(shape, spacing, affine, output):
    """
    """

    with distributed.distributedState() as ds:

        # set up the cluster
        ds.initializeLSFCluster(job_extra="-P multifish")
        ds.initializeClient()
        ds.scaleCluster(njobs=int(np.prod(np.array(shape)/[256,]*3)))

        # compute affine transform as position coordinates, lazy dask arrays
        grid = position_grid_dask(shape) * spacing
        coords = affine_to_grid_dask(affine, grid)
    
        # write in parallel as 4D array to zarr file
        compressor = Blosc(cname='zstd', clevel=9, shuffle=Blosc.BITSHUFFLE)
        coords_disk = zarr.open(write_path, 'w',
            shape=coords.shape, chunks=(256, 256, 256, 3),
            dtype=coords.dtype, compressor=compressor,
        )
    
        # executes computation on cluster and writes result to zarr on disk
        return da.to_zarr(coords, coords_disk)

def piecewise_affine_to_position_field(shape, spacing, ):
    """
    """

    



def apply_position_field():
    """
    """
