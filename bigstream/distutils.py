import logging
import numpy as np
from distributed import Semaphore

from .io_utility import read_block as io_utility_read_block
from .ome_utils import get_spatial_values


logger = logging.getLogger(__name__)


class ThrottledArraySliceReader:

    def __init__(self, max_leases=0, name=None):
        self.max_leases = max_leases
        if max_leases > 0:
            self.semaphore = Semaphore(max_leases=max_leases, name=name)
        else:
            self.semaphore = None

    def read_slice(self, slice, image=None, image_path=None,
                   image_subpath=None, image_timeindex=None,
                   image_channel=None):
        if self.semaphore is None:
            return io_utility_read_block(
                slice,
                image=image,
                image_path=image_path,
                image_subpath=image_subpath,
                image_timeindex=image_timeindex,
                image_channel=image_channel,
            )
        # a semaphore is set
        self.semaphore.acquire()
        try:
            return io_utility_read_block(
                slice,
                image=image,
                image_path=image_path,
                image_subpath=image_subpath,
                image_timeindex=image_timeindex,
                image_channel=image_channel,
            )
        finally:
            self.semaphore.release()

    def get_slice(self, arr, slice):
        if self.semaphore is None:
            return arr[slice]
        # a semaphore is set
        self.semaphore.acquire()
        try:
            return arr[slice]
        finally:
            self.semaphore.release()


def validate_processing_block_size(output_array, processing_size, reverse_output_axes=False):
    """
    Verify that the output zarr chunk size, and shard size when sharded,
    does not exceed the processing block size. If chunks/shards are larger
    than blocks, concurrent workers may write to the same zarr chunk/shard,
    causing race conditions.

    Parameters
    ----------
    output_array : zarr array or None
        The output array to validate. If None or not chunked, no check is performed.

    processing_size : 1d array
        The spatial block partition size used for distributing work.

    Raises
    ------
    ValueError
        If any spatial chunk or shard dimension exceeds the corresponding block dimension.
    """
    if output_array is None or not hasattr(output_array, 'chunks'):
        return

    _check_storage_unit_size(output_array.chunks, 'chunk', processing_size, reverse_output_axes)

    output_shards = getattr(output_array, 'shards', None)
    if output_shards is not None:
        _check_storage_unit_size(output_shards, 'shard', processing_size, reverse_output_axes)


def _check_storage_unit_size(unit_shape, unit_label, processing_size, reverse_output_axes):
    if unit_shape is None:
        # if everything is processed in memory for a numpy array for example there's no restriction
        return

    unit = np.array(get_spatial_values(unit_shape, reverse_axes=reverse_output_axes))
    if np.any(unit > processing_size):
        logger.error((
            f'Output zarr {unit_label} size {unit} exceeds '
            f'block partition size {processing_size}. '
            f'This may cause race conditions during write.'
        ))
        raise ValueError(
            f'Processing block size {processing_size} is too small '
            f'compared to {unit_label} size: {unit_shape}'
        )
