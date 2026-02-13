import logging
import numpy as np

from .ome_utils import get_spatial_values


logger = logging.getLogger(__name__)


def validate_processing_block_size(output_array, block_partition_size):
    """
    Verify that the output zarr chunk size does not exceed the processing
    block size. If chunks are larger than blocks, concurrent workers may
    write to the same zarr chunk, causing race conditions.

    Parameters
    ----------
    output_array : zarr array or None
        The output array to validate. If None or not chunked, no check is performed.

    block_partition_size : 1d array
        The spatial block partition size used for distributing work.

    Raises
    ------
    ValueError
        If any spatial chunk dimension exceeds the corresponding block dimension.
    """
    if output_array is None or not hasattr(output_array, 'chunks'):
        return

    output_chunks = np.array(get_spatial_values(output_array.chunks))
    if np.any(output_chunks > block_partition_size):
        logger.error((
            f'Output zarr chunk size {output_chunks} exceeds '
            f'block partition size {block_partition_size}. '
            f'This may cause race conditions during write.'
        ))
        raise ValueError(
            f'Processing block size {block_partition_size} is too small '
            f'compared to chunk size: {output_array.chunks}'
        )
