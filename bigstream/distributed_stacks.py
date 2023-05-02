import glob
import h5py
import numpy as np

import bigstream.utility as ut

from ClusterWrap.decorator import cluster


@cluster
def distributed_directory_of_hdf5_to_zarr(
    directory,
    write_path,
    dataset_path=None,
    chunks=None,
    suffix='.h5',
    cluster=None,
    cluster_kwargs={},
):
    """
    Copy a directory of hdf5 files to a zarr array on disk, data is duplicated.
    Any file with an hdf5 extension is included. All hdf5 files must contain
    an array and all such arrays must have the same shape. The lexicographical order
    determined by glob.glob will determine the order the arrays are indexed across
    the first axis in the result. A dask cluster is used to distribute this work
    over parallel resources.

    Parameters
    ----------
    directory : string
        The path to the directory containing the hdf5 files

    write_path : string
        The path to the zarr array you will create

    dataset_path : string (default: None)
        A path to the array dataset within the hdf5 files

    chunks : tuple (default: None)
        The shape of individual chunks in the zarr array.
        If None, each array dataset will be one chunk.
        WARNING: this function is not protected against parallel writes
        If you use the chunks keyword argument, you must know that your
        chunk size and array size will not result in parallel writes

    suffix : string (default: '.h5')
        The file extension for all the hdf5 files

    cluster : ClusterWrap.cluster object (default: None)
        Only set if you have constructed your own static cluster. The default behavior
        is to construct a cluster for the duration of this function, then close it
        when the function is finished.

    cluster_kwargs : dict (default: {})
        Arguments passed to ClusterWrap.cluster
        If working with an LSF cluster, this will be ClusterWrap.janelia_lsf_cluster.
        If on a workstation this will be ClusterWrap.local_cluster.
        This is how distribution parameters are specified.

    Returns
    -------
    dataset_as_zarr : zarr.Array
        A reference to the zarr array on disk
    """

    # get all paths, look at first file to get shape and datatype
    paths = glob.glob(directory + '/*' + suffix)
    with h5py.File(paths[0], 'r') as ex:
        example_array = ex[dataset_path] if dataset_path else ex
        shape = example_array.shape
        dtype = example_array.dtype

    # create zarr array
    if chunks is None: chunks = (1,) + shape
    shape = (len(paths),) + shape
    zarr_array = ut.create_zarr(write_path, shape, chunks, dtype)

    # define write function
    def write_frame(path, index, zarr_array):
        with h5py.File(path, 'r') as a:
            array = a[dataset_path] if dataset_path else a
            data = array[...]
            zarr_array[index] = data
        return True

    # distribute and wait for completion
    futures = cluster.client.map(
        write_frame, paths, range(0, len(paths)),
        zarr_array=zarr_array,
    )
    all_written = np.all( cluster.client.gather(futures) )
    if not all_written: print('SOMETHING FAILED, CHECK LOGS')
    return zarr_array


@cluster
def distributed_directory_of_stack_to_zarr(
    directory,
    write_path,
    shape,
    dtype,
    chunks=None,
    suffix='.stack',
    cluster=None,
    cluster_kwargs={},
):
    """
    Copy a directory of stack files to a zarr array on disk, data is duplicated.
    Any file with a stack extension is included. All stack files must contain
    an array and all such arrays must have the same shape. The lexicographical order
    determined by glob.glob will determine the order the arrays are indexed across
    the first axis in the result. A dask cluster is used to distribute this work
    over parallel resources.

    Parameters
    ----------
    directory : string
        The path to the directory containing the stack files

    write_path : string
        The path to the zarr array you will create

    shape : tuple
        The array dimensions of the dataset in each stack file

    dtype : a numpy datatype (e.g. np.uint16)
        The datatype of the dataset in each stack file

    chunks : tuple (default: None)
        The shape of individual chunks in the zarr array.
        If None, each array dataset will be one chunk.
        WARNING: this function is not protected against parallel writes
        If you use the chunks keyword argument, you must know that your
        chunk size and array size will not result in parallel writes

    suffix : string (default: '.stack')
        The file extension for all the stack files

    cluster : ClusterWrap.cluster object (default: None)
        Only set if you have constructed your own static cluster. The default behavior
        is to construct a cluster for the duration of this function, then close it
        when the function is finished.

    cluster_kwargs : dict (default: {})
        Arguments passed to ClusterWrap.cluster
        If working with an LSF cluster, this will be ClusterWrap.janelia_lsf_cluster.
        If on a workstation this will be ClusterWrap.local_cluster.
        This is how distribution parameters are specified.

    Returns
    -------
    dataset_as_zarr : zarr.Array
        A reference to the zarr array on disk
    """

    # get all paths, look at first file to get shape and datatype
    paths = glob.glob(directory + '/*' + suffix)

    # create zarr array
    if chunks is None: chunks = (1,) + shape
    zarr_array = ut.create_zarr(
        write_path,
        (len(paths),) + shape,
        chunks,
        dtype,
    )

    # define write function
    def write_frame(path, index, zarr_array):
        data = np.fromfile(path, dtype=dtype).reshape(shape)
        zarr_array[index] = data
        return True

    # distribute and wait for completion
    futures = cluster.client.map(
        write_frame, paths, range(0, len(paths)),
        zarr_array=zarr_array,
    )
    all_written = np.all( cluster.client.gather(futures) )
    if not all_written: print('SOMETHING FAILED, CHECK LOGS')
    return zarr_array
