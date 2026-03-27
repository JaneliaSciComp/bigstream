import argparse
import logging
import bigstream.io_utility as io_utility

from bigstream.configure_bigstream import configure_logging
from bigstream.image_data import ImageData
from bigstream.ome_utils import get_spatial_values

from bigstream.foreground_mask import generate_foreground_mask
from bigstream.io_utility import read_block as read_image

from .cli import (dictfromjson, floattuple, inttuple)


logger:logging.Logger


def _define_args():
    args_parser = argparse.ArgumentParser(description='Prepare mask')

    args_parser.add_argument('--image',
                             dest='image',
                             required=True,
                             help='Path to the image container')
    args_parser.add_argument('--image-subpath', '--image_subpath',
                             dest='image_subpath',
                             help='Image subpath')
    args_parser.add_argument('--timeindex',
                             dest='timeindex',
                             type=int,
                             default=None,
                             help='Image time index')
    args_parser.add_argument('--channel',
                             dest='channel',
                             type=int,
                             default=None,
                             help='Image channel')
    args_parser.add_argument('--spacing',
                             dest='spacing',
                             type=floattuple,
                             help='Image voxel spacing')
    args_parser.add_argument('--expansion-factor',
                             dest='expansion_factor',
                             type=float,
                             help='Image expansion factor')

    args_parser.add_argument('--mask-subsampling',
                            dest='mask_subsampling',
                            type=int,
                            default=2,
                            help='Mask subsampling')
    args_parser.add_argument('--mask-smoothing',
                             dest='mask_smoothing',
                             type=int,
                             default=1,
                             help='Number of iterations to apply mask smoothing - smaller value results in a rougher mask')  
    args_parser.add_argument('--mask-sigma', '--mask-smooth-sigmas',
                             dest='mask_smooth_sigmas',
                             type=floattuple,
                             default=(),
                             help='Gaussian smoothing sigma at the finest shrink factor; coarser shrink factors use proportionally larger sigmas')
    args_parser.add_argument('--mask-lambda',
                             dest='mask_lambda2',
                             type=float,
                             default=2,
                             metavar='lambda2',
                             help='Controls the variance of the foreground region. A larger number means larger segment(s).')
    args_parser.add_argument('--mask-threshold-percentile',
                             dest='mask_thresh_percentile',
                             type=int,
                             help='Intensity percentile used for finding the foreground threshold; when set, uses a simple threshold cutoff instead of the MorphACWE method')

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
                             default='gzip',
                             type=str,
                             help='Codec used for zarr arrays. ' +
                             'Valid values are: raw,lz4,gzip,bz2,blosc,zstd')
    args_parser.add_argument('--compression-opts', '--compressor-opts',
                             dest='compressor_opts',
                             default={},
                             type=dictfromjson,
                             help='Zarr array compression options')

    args_parser.add_argument('--logging-config', dest='logging_config',
                             type=str,
                             help='Logging configuration')
    args_parser.add_argument('--verbose',
                             dest='verbose',
                             action='store_true',
                             help='Set logging level to verbose')

    return args_parser


def _generate_foreground_mask(args):
    image_data = ImageData(
        args.image,
        args.image_subpath,
        image_timeindex=args.timeindex,
        image_channel=args.channel,
        open_image=True,
    )

    logger.info(f'Create mask for {image_data}')

    if (args.output_blocksize is not None and
        len(args.output_blocksize) > 0):
        output_chunk_size = args.output_blocksize[::-1] # make it zyx
    else:
        # default to output_chunk_size
        output_chunk_size = (args.output_chunk_size,) * image_data.spatial_ndim

    image_array = read_image(
        tuple([slice(None), slice(None), slice(None)]),
        image=image_data.image_array,
        image_timeindex=image_data.image_timeindex,
        image_channel=image_data.image_channel,
    )
    logger.debug(f'Read image of shape: {image_array.shape}')

    mask_iterations = [10, 20, 40]
    if not args.mask_smooth_sigmas:
        smooth_sigmas = [8, 16, 24]
    elif len(args.mask_smooth_sigmas) < len(mask_iterations):
        smooth_sigmas = list(args.mask_smooth_sigmas)
        n_missing = len(mask_iterations) - len(smooth_sigmas)
        if n_missing > 0:
            last_sigma = smooth_sigmas[-1]
            smooth_sigmas.extend(last_sigma + 8 * i for i in range(1, n_missing + 1))
    else:
        smooth_sigmas = args.mask_smooth_sigmas[0:len(mask_iterations)]

    mask, mask_spacing = generate_foreground_mask(
        image_array,
        get_spatial_values(image_data.voxel_spacing),
        image_subsampling=args.mask_subsampling,
        mask_smoothing=args.mask_smoothing,
        lambda2=args.mask_lambda2,
        iterations=mask_iterations[::-1],
        smooth_sigmas=smooth_sigmas[::-1],
        percentile_thresh=args.mask_thresh_percentile,
    )

    logger.info(f'Write {mask.shape} mask to {args.output}:{args.output_subpath} with spacing: {mask_spacing}')

    axes = [a for a in (image_data.get_attr('axes') or []) if a.get('type') == 'space']
    coordinate_transformations = [
        {'type': ct['type'], ct['type']: (
            mask_spacing if ct['type'] == 'scale'
            else get_spatial_values(ct.get('translation'))
        )}
        for ct in (image_data.get_attr('coordinateTransformations') or [])
        if ct.get('type') in ('scale', 'translation')
    ]

    output_attrs = io_utility.prepare_parent_group_attrs(
        args.output,
        args.output_subpath,
        axes=axes,
        dataset_transformations=coordinate_transformations,
    )

    output_array = io_utility.create_dataset_array(
        args.output,
        args.output_subpath,
        mask.shape,
        output_chunk_size,
        mask.dtype,
        compressor=args.compressor,
        compression_opts=args.compressor_opts,
        parent_attrs=output_attrs,
        zarr_format=2,
    )
    output_array[...] = mask


def main():
    args_parser = _define_args()
    args = args_parser.parse_args()
    # prepare logging
    global logger
    logger = configure_logging(args.logging_config, args.verbose)

    logger.info(f'Invoked foreground mask generation: {args}')
    _generate_foreground_mask(args)


if __name__ == '__main__':
    main()
