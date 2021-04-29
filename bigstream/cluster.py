import ClusterWrap
import zarr
from numcodecs import Blosc
from dask.array import to_zarr


def execute(array, write_path=None, **kwargs):
    """
    """

    with ClusterWrap.cluster(**kwargs) as cluster:

        # if user wants to write to disk
        if write_path is not None:
            compressor = Blosc(
                cname='zstd',
                clevel=4,
                shuffle=Blosc.BITSHUFFLE,
            )
            zarr_disk = zarr.open(
                write_path, 'w',
                shape=array.shape,
                chunks=array.chunksize,
                dtype=array.dtype,
                compressor=compressor,
            )
            to_zarr(array, zarr_disk)
            return zarr_disk

        # otherwise user wants result returned to local process
        return array.compute()

