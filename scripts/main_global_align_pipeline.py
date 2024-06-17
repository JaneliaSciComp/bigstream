import argparse
import numpy as np
import bigstream.io_utility as io_utility

from os.path import exists
from bigstream.align import alignment_pipeline
from bigstream.cli import (CliArgsHelper, define_registration_input_args,
                           extract_align_pipeline,
                           extract_registration_input_args,
                           get_input_images)
from bigstream.configure_logging import configure_logging
from bigstream.image_data import ImageData
from bigstream.transform import apply_transform


logger = None # initialized in main as a result of calling configure_logging


def _define_args(args_descriptor):
    args_parser = argparse.ArgumentParser(description='Registration pipeline')

    define_registration_input_args(
        args_parser.add_argument_group(
            description='Global registration input volumes'),
        args_descriptor,
    )

    args_parser.add_argument('--align-config',
                             dest='align_config',
                             help='Align config file')
    args_parser.add_argument('--reuse-existing-transform',
                             dest='reuse_existing_transform',
                             action='store_true',
                             help='If set and the transform exists do not recompute it')

    args_parser.add_argument('--logging-config', dest='logging_config',
                             type=str,
                             help='Logging configuration')
    args_parser.add_argument('--verbose',
                             dest='verbose',
                             action='store_true',
                             help='Set logging level to verbose')

    return args_parser


def _run_global_align(args, align_config):
    global_steps, _ = extract_align_pipeline(align_config,
                                             'global_align',
                                             args.registration_steps)
    if len(global_steps) == 0:
        logger.info('Skip global alignment because no global steps were specified.')
        return None

    (fix, fix_mask, mov, mov_mask) = get_input_images(args)
    if fix.has_data() and mov.has_data():
        # calculate and apply the global transform
        affine, aligned = _align_global_data(fix, fix_mask, mov, mov_mask, global_steps)
        # save the global transform
        _save_global_transform(args, affine)
        # save global aligned volume
        return _save_aligned_volume(
            args, aligned,
            mov.get_attr('pixelResolution'),
            mov.get_attr('downsamplingFactors'),
        )
    else:
        logger.info('Skip global alignment - both fix and moving image are needed')
        return None




def _align_global_data(fix_image, fix_mask,
                       mov_image, mov_mask,
                       steps):
    logger.info('Read image data for global alignment')
    fix_image.read_image()
    mov_image.read_image()
    if isinstance(fix_mask, ImageData):
        fix_mask.read_image()
        fix_mask = fix_mask.image_array
    else:
        fix_mask = fix_mask
    if isinstance(mov_mask, ImageData):
        mov_mask.read_image()
        mov_mask = mov_mask.image_array
    else:
        mov_mask = mov_mask

    logger.info(f'Calculate global transform using: {steps}')
    affine = alignment_pipeline(fix_image.image_array,
                                mov_image.image_array,
                                fix_image.voxel_spacing,
                                mov_image.voxel_spacing,
                                steps,
                                fix_mask=fix_mask,
                                mov_mask=mov_mask)

    logger.info('Apply affine transform')
    # apply transform
    aligned = apply_transform(fix_image.image_array,
                              mov_image.image_array,
                              fix_image.voxel_spacing,
                              mov_image.voxel_spacing,
                              transform_list=[affine,])

    return affine, aligned


def _apply_global_transform(args, affine):
    (fix_image, _, mov_image, _) = get_input_images(args)
    if fix_image.has_data() and mov_image.has_data():
        fix_image.read_image()
        mov_image.read_image()
        # apply transform
        aligned = apply_transform(fix_image.image_array,
                                  mov_image.image_array,
                                  fix_image.voxel_spacing,
                                  mov_image.voxel_spacing,
                                  transform_list=[affine,])
        _save_aligned_volume(
            args, aligned,
            mov_image.get_attr('pixelResolution'), mov_image.get_attr('downsamplingFactors'),
        )
    else:
        # both fix and mov volume must be valid
        return None


def _save_global_transform(args, transform):
    transform_file = args.transform_path()
    if transform_file:
        logger.info(f'Save global transformation to {transform_file}')
        np.savetxt(transform_file, transform)
    else:
        logger.info('Skip saving global transformation')

    inv_transform_file = args.inv_transform_path()
    if inv_transform_file:
        try:
            inv_transform = np.linalg.inv(transform)
            logger.info(f'Save global inverse transformation to {inv_transform_file}')
            np.savetxt(inv_transform_file, inv_transform)
        except Exception:
            logger.error(f'Global affine {transform} is not invertible')

    else:
        logger.info('Skip saving global inverse transformation')


def _save_aligned_volume(args, aligned_array, pixel_resolution, downsampling):
    align_path = args.align_path()
    if align_path:
        if args.align_blocksize:
            align_blocksize = args.align_blocksize[::-1]
        else:
            align_blocksize = (128,) * aligned_array.ndim

        logger.info(f'Save global aligned volume to {align_path} ' +
                    f'with blocksize {align_blocksize}')

        return io_utility.create_dataset(
            align_path,
            args.align_dataset(),
            aligned_array.shape,
            align_blocksize,
            aligned_array.dtype,
            data=aligned_array,
            pixelResolution=pixel_resolution,
            downsamplingFactors=downsampling,
        )
    else:
        logger.info('Skip saving global aligned volume')
        return  None


def main():
    global_descriptor = CliArgsHelper('global')
    args_parser = _define_args(global_descriptor)
    args = args_parser.parse_args()
    # prepare logging
    global logger
    logger = configure_logging(args.logging_config, args.verbose)

    logger.info(f'Global registration: {args}')

    reg_inputs = extract_registration_input_args(args, global_descriptor)
    global_transform = None
    global_transform_file = reg_inputs.transform_path()

    if args.reuse_existing_transform:
        # try to read the global transform
        logger.info(f'Global transform file: {global_transform_file}')
        if global_transform_file and exists(global_transform_file):
            logger.info(f'Read global transform from {global_transform_file}')
            global_transform = np.loadtxt(global_transform_file)

    if global_transform is None:
        # no global transform found -> calculate it and then apply it
        _run_global_align(reg_inputs, args.align_config)
    else:
        # global transform found -> just apply it
        _apply_global_transform(reg_inputs, global_transform)


if __name__ == '__main__':
    main()