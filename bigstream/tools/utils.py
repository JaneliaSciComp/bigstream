import logging

from math import ceil


logger = logging.getLogger(__name__)


def derive_shard_shape(sharding_factor_xyz, output_blocksize_zyx, zarr_format):
    """Compute the absolute zarr v3 shard shape from a sharding factor.

    Parameters
    ----------
    sharding_factor_xyz : tuple[int, ...] or None
        Sharding factor as given on the CLI (xyz order). When None, empty,
        or zarr_format != 3, returns None.

    output_blocksize_zyx : tuple[int, ...]
        Output chunk shape in zyx order (after the launcher has reversed the
        xyz CLI value and padded for non-spatial axes).

    zarr_format : int
        Target zarr format. Sharding is only emitted for format 3.

    Returns
    -------
    tuple[int, ...] or None
        Absolute shard shape in the same axis order as output_blocksize_zyx,
        equal to blocksize * factor elementwise. None when sharding is off.
    """
    if not sharding_factor_xyz or zarr_format != 3:
        return None
    factor = tuple(sharding_factor_xyz[::-1])  # xyz -> zyx
    if len(factor) < len(output_blocksize_zyx):
        factor = (1,) * (len(output_blocksize_zyx) - len(factor)) + factor
    elif len(factor) > len(output_blocksize_zyx):
        raise ValueError(
            f'output_sharding_factor {sharding_factor_xyz} has higher rank '
            f'than output_blocksize {output_blocksize_zyx}'
        )
    if any(f <= 0 for f in factor):
        raise ValueError(
            f'output_sharding_factor components must be positive: '
            f'{sharding_factor_xyz}'
        )
    return tuple(b * f for b, f in zip(output_blocksize_zyx, factor))


def get_processing_size(input_processing_size, shard_shape=None, blocksize=None):
    """Round up processing_size_zyx to a multiple of the storage unit.

    For zarr3 pass shard_shape_zyx; for zarr2 pass blocksize_zyx.
    Returns final_processing_size.
    """
    unit = shard_shape if shard_shape is not None else blocksize
    if unit is None:
        return tuple(input_processing_size)
    processing_size = tuple(int(ceil(p / s)) * s for p, s in zip(input_processing_size, unit))
    logger.info(
        f'Final processing size: {processing_size} '
        f'based on processing unit size: {unit} and provided processing size: {input_processing_size} '
    )

    return processing_size
