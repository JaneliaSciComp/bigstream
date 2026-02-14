import logging
import numpy as np

from .ome_utils import get_spatial_values


logger = logging.getLogger(__name__)


def validate_processing_block_size(output_array, processing_block_size, reverse_output_axes=False):
    """
    Verify that the output zarr chunk size does not exceed the processing
    block size. If chunks are larger than blocks, concurrent workers may
    write to the same zarr chunk, causing race conditions.

    Parameters
    ----------
    output_array : zarr array or None
        The output array to validate. If None or not chunked, no check is performed.

    processing_block_size : 1d array
        The spatial block partition size used for distributing work.

    Raises
    ------
    ValueError
        If any spatial chunk dimension exceeds the corresponding block dimension.
    """
    if output_array is None or not hasattr(output_array, 'chunks'):
        return

    output_chunks = np.array(get_spatial_values(output_array.chunks, reverse_axes=reverse_output_axes))
    if np.any(output_chunks > processing_block_size):
        logger.error((
            f'Output zarr chunk size {output_chunks} exceeds '
            f'block partition size {processing_block_size}. '
            f'This may cause race conditions during write.'
        ))
        raise ValueError(
            f'Processing block size {processing_block_size} is too small '
            f'compared to chunk size: {output_array.chunks}'
        )
