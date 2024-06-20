import numpy as np
import os
import psutil
import SimpleITK as sitk
import zarr

from distributed import Lock
from scipy.spatial.transform import Rotation
from zarr import blosc
from zarr.indexing import BasicIndexer


def skip_sample(image, spacing, ss_spacing):
    """
    Return a subset of the voxels in an image to approximate a different
    sampling rate

    Parameters
    ----------
    image : nd-array
        The image to sub-sample

    spacing : 1d-array
        The physical spacing of the voxels in image

    ss_spacing : 1d-array
        The desired physical spacing to approximate by skipping voxels

    Returns
    -------
    smaller_image : nd-array
        a subset of voxels from image with a new spacing as close as
        possible to ss_spacing

    new_spacing : 1d-array
        the physical spacing of smaller_image
    """

    spacing = np.array(spacing)
    ss = np.maximum(np.round(ss_spacing / spacing), 1).astype(int)
    slc = tuple(slice(None, None, x) for x in ss)
    return image[slc], spacing * ss


def numpy_to_sitk(image, spacing=None, origin=None, vector=False):
    """
    Convert a numpy array to a sitk image object

    Parameters
    ----------
    image : nd-array
        The image data

    spacing : 1d-array (default: None)
        The physical spacing of image

    origin : 1d-array (default: None)
        The physical origin of image

    vector : bool (default:False)
        If the last axis of image is a vector dimension

    Returns
    -------
    sitk_image : sitk.image object
        The given image data with correct spacing and origin set
    """

    # check endianness of data - some sitk operations seem to
    # only work with little endian
    if str(image.dtype)[0] == '>':
        error = "Array cannot be big endian. Convert arrays with ndarray.astype\n"
        error += "Given array dtype is " + str(image.dtype)
        raise TypeError(error)

    image = sitk.GetImageFromArray(image, isVector=vector)
    if spacing is None: spacing = np.ones(image.GetDimension())
    image.SetSpacing(spacing[::-1])
    if origin is None: origin = np.zeros(image.GetDimension())
    image.SetOrigin(origin[::-1])
    return image

# TODO: function that takes a numpy array and return transform type


def relative_spacing(query_shape, reference_shape, reference_spacing):
    """
    Determine a voxel spacing from two images and one voxel spacing

    Parameters
    ----------
    query_shape : shape of a voxel grid
        for which we want to find the spacing

    reference_shape : shape of a voxel grid
        in the same domain as the query grid with known spacing

    reference_spacing : tuple
        The known voxel spacing

    Returns
    -------
    query_spacing : 1d-array
        The spacing of the query voxel grid
    """

    ndim = len(reference_spacing)
    ratio = np.array(reference_shape[:ndim]) / query_shape[:ndim]
    return reference_spacing * ratio


def create_zarr(
    path,
    shape,
    chunks,
    dtype,
    multithreaded=False,
    chunk_locked=False,
    client=None,
):
    """
    Create a new zarr array on disk

    Parameters
    ----------
    path : string
        The location of the new zarr array

    shape : tuple
        The shape of the new zarr array

    chunks : tuple
        The shape of individual chunks in the zarr array

    dtype : a numpy.dtype object
        The data type of the new zarr array data

    Returns
    -------
    zarr_array : zarr array
        Reference to the newly created zarr array on disk
    """

    synchronizer = None
    if multithreaded:
        blosc.use_threads = True
        synchronizer = zarr.ThreadSynchronizer()
    zarr_disk = zarr.open(
        path, 'w',
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        synchronizer=synchronizer,
    )

    # this code is currently never used within bigstream
    # keeping it aroung in case a use case comes up
    if chunk_locked:
        indexer = BasicIndexer(slice(None), zarr_disk)
        keys = (zarr_disk._chunk_key(idx.chunk_coords) for idx in indexer)
        lock = {key: Lock(key, client=client) for key in keys}
        lock['.zarray'] = Lock('.zarray', client=client)
        zarr_disk = zarr.open(
            store=zarr_disk.store, path=zarr_disk.path,
            synchronizer=lock, mode='r+',
        )

    return zarr_disk


def numpy_to_zarr(array, chunks, path):
    """
    Convert a numpy array to a zarr array on disk

    Parameters
    ----------
    array : nd-array
        The numpy array to convert to zarr format

    chunks : tuple
        The shape of individual chunks in the zarr array

    path : string
        On disk location to create the zarr array

    Returns
    -------
    zarr_array : zarr array
        Reference to the zarr array copy of the given numpy array
    """

    if not isinstance(array, zarr.Array):
        zarr_disk = create_zarr(path, array.shape, chunks, array.dtype)
        zarr_disk[...] = array
        return zarr_disk
    else:
        return array


def get_number_of_cores():
    """
    Get number of physical cores available to the python process.
    Currently only considers LSF environment variable. If not an LSF
    cluster job the uses psutil.cpu_count
    """

    if "LSB_DJOB_NUMPROC" in os.environ:
        ncores = int(os.environ["LSB_DJOB_NUMPROC"])
    else:
        ncores = psutil.cpu_count(logical=False)
    return ncores
