import argparse
import logging
import bigstream.io_utility as io_utility
import numpy as np

from scipy.ndimage import uniform_filter, zoom
from bigstream.configure_bigstream import configure_logging
from bigstream.image_data import ImageData
from bigstream.ome_utils import get_spatial_values

from bigstream.io_utility import read_block as read_image

from .cli import (dictfromjson, floattuple, inttuple)

from .utils import derive_shard_shape


logger:logging.Logger


def _define_args():
    args_parser = argparse.ArgumentParser(description='Prepare mask')

    args_parser.add_argument('--fix', '--fixed',
                             dest='fixed',
                             help='Path to the fixed image')
    args_parser.add_argument('--fix-subpath', '--fixed-subpath',
                             dest='fixed_subpath',
                             help='Fixed image subpath')
    args_parser.add_argument('--fix-timeindex', '--fixed-timeindex',
                             dest='fixed_timeindex',
                             type=int,
                             default=None,
                             help='Fixed image time index')
    args_parser.add_argument('--fix-channel', '--fixed-channel',
                             dest='fixed_channel',
                             type=int,
                             default=None,
                             help='Fixed image channel')
    args_parser.add_argument('--fix-spacing', '--fixed-spacing',
                             dest='fixed_spacing',
                             type=floattuple,
                             help='Fixed image voxel spacing')

    args_parser.add_argument('--mov', '--moving',
                             dest='moving',
                             help='Path to the moving image')
    args_parser.add_argument('--mov-subpath', '--moving-subpath',
                             dest='moving_subpath',
                             help='Moving image subpath')
    args_parser.add_argument('--mov-timeindex', '--moving-timeindex',
                             dest='moving_timeindex',
                             type=int,
                             default=None,
                             help='Moving image time index')
    args_parser.add_argument('--mov-channel', '--moving-channel',
                             dest='moving_channel',
                             type=int,
                             default=None,
                             help='Moving image channel')
    args_parser.add_argument('--mov-spacing', '--moving-spacing',
                             dest='moving_spacing',
                             type=floattuple,
                             help='Moving image voxel spacing')

    args_parser.add_argument('--output',
                             dest='output',
                             help='Output directory')
    args_parser.add_argument('--output-subpath',
                             dest='output_subpath',
                             help='Subpath for the warped output')
    args_parser.add_argument('--output-chunk-size',
                             dest='output_chunk_size',
                             default=128,
                             type=int,
                             help='Output chunk size')
    args_parser.add_argument('--output-blocksize',
                             dest='output_blocksize',
                             type=inttuple,
                             help='Output chunk size as a tuple.')
    args_parser.add_argument('--compression', '--compressor',
                             dest='compressor',
                             default='zstd',
                             type=str,
                             help='Codec used for zarr arrays. ' +
                             'Valid values are: raw,lz4,gzip,bz2,blosc,zstd')
    args_parser.add_argument('--compression-opts', '--compressor-opts',
                             dest='compressor_opts',
                             default={},
                             type=dictfromjson,
                             help='Zarr array compression options')
    args_parser.add_argument('--output-zarr-format', '--output_zarr_format',
                             dest='output_zarr_format',
                             default=2,
                             type=int,
                             help='Zarr output format')
    args_parser.add_argument('--output-sharding-factor', '--output_sharding_factor',
                             dest='output_sharding_factor',
                             default=None,
                             type=inttuple,
                             help='Zarr v3 sharding factor in xyz order, '
                                  'e.g. 8,8,4. The shard shape is computed '
                                  'elementwise as output_blocksize * '
                                  'sharding_factor (each factor must be a '
                                  'positive integer). Ignored when '
                                  '--output-zarr-format is not 3.')

    args_parser.add_argument('--radius',
                             dest='radius',
                             type=int,
                             help='Neighborhood radius in micrometers for evaluating the alignment quality')

    args_parser.add_argument('--logging-config',
                             dest='logging_config',
                             type=str,
                             help='Logging configuration')
    args_parser.add_argument('--verbose',
                             dest='verbose',
                             action='store_true',
                             help='Set logging level to verbose')

    return args_parser


def _compute_registration_metric(args):
    fix_subpath = args.fixed_subpath
    mov_subpath = args.moving_subpath if args.moving_subpath else fix_subpath

    fix_data = ImageData(
        args.fixed,
        fix_subpath,
        image_timeindex=args.fixed_timeindex,
        image_channel=args.fixed_channel,
        open_image=True,
    )
    mov_data = ImageData(
        args.moving,
        mov_subpath,
        image_timeindex=args.moving_timeindex,
        image_channel=args.moving_channel,
        open_image=True
    )

    if args.fixed_spacing:
        fix_data.voxel_spacing = args.fixed_spacing[::-1] # xyz -> zyx

    if args.moving_spacing:
        mov_data.voxel_spacing = args.moving_spacing[::-1] # xyz -> zyx
    elif args.fixed_spacing:
        mov_data.voxel_spacing = fix_data.voxel_spacing

    logger.info(f'Fixed volume: {fix_data}')
    logger.info(f'Moving volume: {mov_data}')

    mov_image = read_image(
        tuple([slice(None), slice(None), slice(None)]),
        image=mov_data.image_array,
        image_timeindex=mov_data.image_timeindex,
        image_channel=mov_data.image_channel,
    )
    fix_image = read_image(
        tuple([slice(None), slice(None), slice(None)]),
        image=fix_data.image_array,
        image_timeindex=fix_data.image_timeindex,
        image_channel=fix_data.image_channel,
    )

    logger.info(f'Compute correlation between {fix_data} and {mov_data}')

    if fix_image.shape != mov_image.shape:
        logger.info(f'Resample fix and moving image of shapes: {fix_image.shape} and {mov_image.shape}')
        fix_image, mov_image = _resample_images(
            fix_image,
            mov_image,
            get_spatial_values(fix_data.voxel_spacing),
            get_spatial_values(mov_data.voxel_spacing),
        )

    mean_value, array_values = _local_correlation(
        fix_image,
        mov_image,
        args.radius,
    )

    logger.info(f'Mean alignment quality: {mean_value}')

    if args.output and array_values is not None:
        output_subpath = args.output_subpath if args.output_subpath else mov_subpath
        logger.info(f'Save {array_values.shape} quality values to {args.output}:{output_subpath}')

        if (args.output_blocksize is not None and
            len(args.output_blocksize) > 0):
            output_chunk_size = args.output_blocksize[::-1] # make it zyx
        else:
            # default to output_chunk_size
            output_chunk_size = (args.output_chunk_size,) * mov_data.spatial_ndim

        axes = [a for a in (mov_data.get_attr('axes') or []) if a.get('type') == 'space']
        coordinate_transformations = [
            {'type': ct['type'], ct['type']: get_spatial_values(ct[ct['type']])}
            for ct in (mov_data.get_attr('coordinateTransformations') or [])
            if ct.get('type') in ('scale', 'translation')
        ]

        output_attrs = io_utility.prepare_parent_group_attrs(
            args.output,
            output_subpath,
            axes=axes,
            dataset_transformations=coordinate_transformations,
            zarr_format=args.output_zarr_format,
        )
        output_shard_size = derive_shard_shape(
            args.output_sharding_factor, output_chunk_size, args.output_zarr_format
        )
        output_array = io_utility.create_dataset_array(
            args.output,
            output_subpath,
            array_values.shape,
            output_chunk_size,
            array_values.dtype,
            compressor=args.compressor,
            compression_opts=args.compressor_opts,
            parent_attrs=output_attrs,
            zarr_format=args.output_zarr_format,
            shard_shape=output_shard_size,
        )
        output_array[...] = array_values


def _resample_images(fix, mov, fix_spacing, mov_spacing):
    zoom_factor = fix_spacing / mov_spacing
    resampled_fix = zoom(fix, tuple(zoom_factor), order=3)
    logger.info(f'Resampled {fix.shape} with {zoom_factor} to {resampled_fix.shape} in order to match {mov.shape}')

    crop = tuple(slice(s) for s in np.minimum(resampled_fix.shape, mov.shape))
    return resampled_fix[crop].astype(np.float64), mov[crop].astype(np.float64)


def _local_correlation(fix, mov, radius, tolerance=1e-6):
    size = 2*radius + 1

    mean_A = uniform_filter(fix, size)
    mean_B  = uniform_filter(mov, size)
    mean_AB = uniform_filter(fix*mov, size)
    mean_AA = uniform_filter(fix*fix, size)
    mean_BB = uniform_filter(mov*mov, size)

    cov = mean_AB - mean_A*mean_B
    var_A = mean_AA - mean_A**2
    var_B = mean_BB - mean_B**2

    # avoid tiny negative variance
    var_A = np.maximum(var_A, 0)
    var_B = np.maximum(var_B, 0)

    corr = cov / np.sqrt(var_A * var_B + tolerance)

    return corr.mean(), corr


def main():
    args_parser = _define_args()
    args = args_parser.parse_args()
    # prepare logging
    global logger
    logger = configure_logging(args.logging_config, args.verbose)

    logger.info(f'Invoked registration metric: {args}')
    _compute_registration_metric(args)


if __name__ == '__main__':
    main()