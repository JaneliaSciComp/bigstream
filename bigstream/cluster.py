from ClusterWrap.clusters import janelia_lsf_cluster
import numpy as np
import zarr
from numcodecs import Blosc
import sys


def execute(
    array,
    write_path=None,
    cluster_kwargs={},
    worker_buffer=4,
):
    """
    """

    # start the cluster
    with janelia_lsf_cluster(**cluster_kwargs) as cluster:

        # print dashboard url
        print("cluster dashboard link: ", cluster.get_dashboard())
        sys.stdout.flush()

        # scale cluster based on array chunks and buffer
        nchunks = np.prod(array.numblocks)
        cluster.scale_cluster(nchunks + worker_buffer)

        # if the user wants to write result to disk
        if write_path:
            compressor = Blosc(cname='zstd', clevel=9, shuffle=Blosc.BITSHUFFLE)
            array_disk = zarr.open(write_path, 'w',
                shape=array.shape, chunks=array.chunksize,
                dtype=array.dtype, compressor=compressor,
            )
            da.to_zarr(array, array_disk)
            return array_disk

        # if the user wants the result back in memory
        else:
            return array.compute()

