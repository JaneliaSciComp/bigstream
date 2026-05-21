import argparse
import logging
import numpy as np
import os
import bigstream.io_utility as io_utility

from bigstream.align import alignment_pipeline
from bigstream.configure_bigstream import (configure_logging,
                                           set_cpu_resources)
from bigstream.io_utility import read_block
from bigstream.image_data import (ImageData,
                                  calc_full_voxel_resolution_attr, calc_downsampling_attr)
from bigstream.ome_utils import (get_spatial_values, compose_origin_transform)
from bigstream.transform import apply_transform

from .cli import (CliArgsHelper, RegistrationInputs,
                  define_registration_input_args, get_algorithm_parameters,
                  extract_registration_input_args, get_input_images,
                  dictfromjson, inttuple)


logger:logging.Logger


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
                             help='Do not recompute global transform if found')

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
    args_parser.add_argument('--output-shard-shape', '--output_shard_shape',
                             dest='output_shard_shape',
                             default=None,
                             type=inttuple,
                             help='Zarr v3 shard shape in xyz order, '
                                  'e.g. 512,512,512. Each component must be '
                                  'a positive multiple of the corresponding '
                                  'chunk dimension. Ignored when '
                                  '--output-zarr-format is not 3.')

    args_parser.add_argument('--cpus', dest='cpus',
                             type=int, default=0,
                             help='Number of cpus allocated')

    args_parser.add_argument('--logging-config', dest='logging_config',
                             type=str,
                             help='Logging configuration')
    args_parser.add_argument('--verbose',
                             dest='verbose',
                             action='store_true',
                             help='Set logging level to verbose')

    return args_parser


def _run_global_align(reg_args:RegistrationInputs,
                      align_config,
                      compressor,
                      compressor_opts,
                      zarr_format,
                      shard_shape=None):
    global_steps, _ = get_algorithm_parameters(align_config,
                                               'global_align',
                                               reg_args.registration_steps)
    if len(global_steps) == 0:
        logger.info('Skip global alignment because no global steps were specified.')
        return None

    fix, fix_mask, mov, mov_mask = get_input_images(reg_args)
    if fix.has_data() and mov.has_data():
        # compose mov origin transform from user affine + OME translations
        mov_origin_transform = compose_origin_transform(
            reg_args.get_mov_origin_transform(),
            mov.get_attr('globalCoordinateTransformations'),
        )
        # calculate and apply the global transform
        affine, aligned = _align_global_data(fix, fix_mask, mov, mov_mask,
                                             global_steps,
                                             mov_origin_transform,
                                             reg_args.get_static_transforms())
        logger.debug(f'Global affine: {affine}')
        # save the global transform
        _save_global_transform(reg_args, affine)
        # save global aligned volume
        return _save_aligned_volume(
            reg_args,
            fix,
            aligned,
            mov.get_attr('axes'),
            mov.get_attr('coordinateTransformations'),
            mov.voxel_spacing,
            mov.voxel_downsampling,
            compressor,
            compressor_opts,
            zarr_format,
            shard_shape,
        )
    else:
        logger.info('Skip global alignment - both fix and moving image are needed')
        return None


def _align_global_data(fix_image, fix_mask,
                       mov_image, mov_mask,
                       steps,
                       mov_origin_transform,
                       static_transforms):
    logger.info('Read image data for global alignment')
    if isinstance(fix_mask, ImageData):
        fix_mask = fix_mask.image_array
    else:
        fix_mask = fix_mask
    if isinstance(mov_mask, ImageData):
        mov_mask = mov_mask.image_array
    else:
        mov_mask = mov_mask

    logger.info(f'Calculate global transform using: {steps}')
    fix_spacing = get_spatial_values(fix_image.voxel_spacing)
    logger.info(f'Fix image voxel spacing: {fix_spacing}')
    mov_spacing = get_spatial_values(mov_image.voxel_spacing)
    logger.info(f'Moving image voxel spacing: {mov_spacing}')

    if mov_origin_transform is not None:
        mov_origin = mov_origin_transform[:3, 3]
    else:
        mov_origin = None
    full_image_coords = tuple(slice(None) for _ in range(fix_image.spatial_ndim))
    fix_image_array = read_block(
        full_image_coords,
        image=fix_image.image_array,
        image_path=fix_image.image_path,
        image_subpath=fix_image.image_subpath,
        image_timeindex=fix_image.image_timeindex,
        image_channel=fix_image.image_channel,
    )
    mov_image_array = read_block(
        full_image_coords,
        image=mov_image.image_array,
        image_path=mov_image.image_path,
        image_subpath=mov_image.image_subpath,
        image_timeindex=mov_image.image_timeindex,
        image_channel=mov_image.image_channel,
    )
    affine = alignment_pipeline(fix_image_array,
                                mov_image_array,
                                fix_spacing / fix_image.expansion_factor,
                                mov_spacing / fix_image.expansion_factor,
                                steps,
                                fix_mask=fix_mask,
                                mov_mask=mov_mask,
                                fix_origin=None,
                                mov_origin=mov_origin,
                                static_transform_list=static_transforms)
    logger.info(f'Apply affine transform: {affine}')
    # apply transform
    aligned = apply_transform(fix_image_array,
                              mov_image_array,
                              fix_spacing / fix_image.expansion_factor,
                              mov_spacing / mov_image.expansion_factor,
                              transform_list=static_transforms + [affine,])
    return affine, aligned


def _apply_global_transform(reg_args:RegistrationInputs,
                            affine,
                            compressor,
                            compressor_opts,
                            zarr_format,
                            shard_shape=None):
    (fix_image, _, mov_image, _) = get_input_images(reg_args)
    if fix_image.has_data() and mov_image.has_data():
        full_image_coords = tuple(slice(None) for _ in range(fix_image.spatial_ndim))
        fix_image_array = read_block(
            full_image_coords,
            image=fix_image.image_array,
            image_path=fix_image.image_path,
            image_subpath=fix_image.image_subpath,
            image_timeindex=fix_image.image_timeindex,
            image_channel=fix_image.image_channel,
        )
        mov_image_array = read_block(
            full_image_coords,
            image=mov_image.image_array,
            image_path=mov_image.image_path,
            image_subpath=mov_image.image_subpath,
            image_timeindex=mov_image.image_timeindex,
            image_channel=mov_image.image_channel,
        )
        # apply transform
        transform_list = reg_args.get_static_transforms() + [affine,]
        aligned = apply_transform(fix_image_array,
                                  mov_image_array,
                                  fix_image.voxel_spacing / fix_image.expansion_factor,
                                  mov_image.voxel_spacing / mov_image.expansion_factor,
                                  transform_list=transform_list)
        _save_aligned_volume(
            reg_args,
            fix_image,
            aligned,
            mov_image.get_attr('axes'),
            mov_image.get_attr('coordinateTransformations'),
            mov_image.voxel_spacing,
            mov_image.voxel_downsampling,
            compressor,
            compressor_opts,
            zarr_format,
            shard_shape,
        )
    else:
        # both fix and mov volume must be valid
        return None


def _save_global_transform(args, transform):
    transform_file = args.transform_path()
    if transform_file:
        transform_location = os.path.dirname(transform_file)
        os.makedirs(transform_location, exist_ok=True)
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


def _save_aligned_volume(reg_args:RegistrationInputs,
                         fix_image,
                         aligned_array,
                         axes,
                         dataset_transformations,
                         voxel_resolution,
                         downsampling,
                         compressor,
                         compressor_opts,
                         zarr_format,
                         shard_shape=None):
    align_path = reg_args.align_path()
    # prepare global coordinate transform from reg_args.mov_origin_transform
    global_transformations = []
    mov_origin_transform = reg_args.get_mov_origin_transform()
    if mov_origin_transform is not None and reg_args.persist_mov_origin_transform:
        spatial_translation = mov_origin_transform[:3, 3].tolist()
        # prepend 0 for each non-spatial axis (time, channel)
        non_spatial_count = sum(
            1
            for a in (axes or []) if a.get('type') != 'space'
        )
        translation = [0,] * non_spatial_count + spatial_translation
        global_transformations.append({
            'type': 'translation',
            'translation': translation,
        })

    if align_path:
        align_attrs = io_utility.prepare_parent_group_attrs(
            align_path,
            reg_args.align_dataset(),
            axes=axes,
            dataset_transformations=dataset_transformations,
            global_transformations=global_transformations,
            zarr_format=zarr_format,
        )
        fix_shape = fix_image.shape

        if reg_args.align_blocksize:
            align_blocksize = reg_args.align_blocksize[::-1]
        else:
            align_blocksize = (128,) * fix_image.spatial_ndim

        if len(aligned_array.shape) < len(fix_shape):
            aligned_dataset_shape = (1,) * (len(fix_shape) - len(aligned_array.shape)) + aligned_array.shape
            logger.info(f'Reshape align dataset to: {aligned_dataset_shape}')
        else:
            aligned_dataset_shape = aligned_array.shape

        if len(align_blocksize) < len(fix_shape):
            # align_blocksize is not set, so use default block size
            aligned_dataset_chunksize = (1,) * (len(fix_shape)-len(align_blocksize)) + align_blocksize
        else:
            aligned_dataset_chunksize = align_blocksize

        if shard_shape:
            aligned_dataset_shardsize = shard_shape[::-1]  # xyz -> zyx
            if len(aligned_dataset_shardsize) < len(aligned_dataset_chunksize):
                aligned_dataset_shardsize = (
                    (1,) * (len(aligned_dataset_chunksize) - len(aligned_dataset_shardsize))
                    + tuple(aligned_dataset_shardsize)
                )
        else:
            aligned_dataset_shardsize = None

        logger.info((
            f'Save global aligned volume to {align_path} '
            f'shape: {aligned_dataset_shape} '
            f'blocksize {aligned_dataset_chunksize} '
            f'shardsize {aligned_dataset_shardsize} '
            f'attrs: {align_attrs} '
        ))
        dataset_array = io_utility.create_dataset_array(
            align_path,
            reg_args.align_dataset(),
            aligned_dataset_shape,
            aligned_dataset_chunksize,
            aligned_array.dtype,
            overwrite=False,
            compressor=compressor,
            compression_opts=compressor_opts,
            for_timeindex=reg_args.get_align_timeindex(),
            for_channel=reg_args.get_align_channel(),
            parent_attrs=align_attrs,
            pixelResolution=calc_full_voxel_resolution_attr(voxel_resolution,
                                                            downsampling),
            downsamplingFactors=calc_downsampling_attr(downsampling),
            zarr_format=zarr_format,
            shard_shape=aligned_dataset_shardsize,
        )
        write_coords = []
        if reg_args.get_align_timeindex() is not None and len(dataset_array.shape) > 3:
            write_coords.append(reg_args.get_align_timeindex())
        if reg_args.get_align_channel() is not None and len(dataset_array.shape) > 3:
            write_coords.append(reg_args.get_align_channel())
        if write_coords:
            logger.info(f'Write {aligned_array.shape} array to {dataset_array.shape} dataset at {write_coords}')
            dataset_array[tuple(write_coords)] = aligned_array
        else:
            dataset_array[...] = aligned_array
        return dataset_array
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
        if global_transform_file and os.path.exists(global_transform_file):
            logger.info(f'Read global transform from {global_transform_file}')
            global_transform = np.loadtxt(global_transform_file)

    set_cpu_resources(args.cpus)

    if global_transform is None:
        # no global transform found -> calculate it and then apply it
        _run_global_align(reg_inputs,
                          args.align_config,
                          args.compressor,
                          args.compressor_opts,
                          args.output_zarr_format,
                          shard_shape=args.output_shard_shape)
    else:
        # global transform found -> just apply it
        _apply_global_transform(reg_inputs,
                                global_transform,
                                args.compressor, args.compressor_opts,
                                args.output_zarr_format,
                                shard_shape=args.output_shard_shape)


if __name__ == '__main__':
    main()