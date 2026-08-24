import argparse
import logging
import numpy as np
import os
import tempfile
import bigstream.io_utility as io_utility
import bigstream.transform as bst
import zarr

from dask.distributed import (Client, LocalCluster)

from bigstream.align import alignment_pipeline
from bigstream.configure_bigstream import (configure_logging,
                                           set_cpu_resources)
from bigstream.distributed_align import distributed_alignment_pipeline
from bigstream.io_utility import read_block
from bigstream.image_data import (ImageData,
                                  calc_full_voxel_resolution_attr, calc_downsampling_attr,
                                  clip_arr_to_roi)
from bigstream.ome_utils import (get_spatial_values, compose_origin_transform)
from bigstream.transform import (apply_transform,
                                 invert_displacement_vector_field)

from .cli import (CliArgsHelper, RegistrationInputs,
                  define_registration_input_args, get_algorithm_parameters,
                  extract_registration_input_args, get_input_images,
                  dictfromjson, get_transform, inttuple, floattuple)

from .utils import derive_shard_shape, get_zarr_format


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

    args_parser.add_argument('--prealign-downsample',
                             dest='prealign_downsample',
                             type=int,
                             default=1,
                             help='Pre-align downsampling')

    args_parser.add_argument('--reuse-existing-transform',
                             dest='reuse_existing_transform',
                             action='store_true',
                             help='Do not recompute global transform if found')
    args_parser.add_argument('--save-composed-transform',
                             dest='save_composed_transform',
                             action='store_true',
                             help='Persisted global transformation is composed with the provided static transformations')

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

    args_parser.add_argument('--cpus', dest='cpus',
                             type=int, default=0,
                             help='Number of cpus allocated')
    args_parser.add_argument('--local-dask-workers', '--local_dask_workers',
                             dest='local_dask_workers',
                             type=int,
                             default=1,
                             help='Number of workers when using a local cluster')

    args_parser.add_argument('--logging-config', dest='logging_config',
                             type=str,
                             help='Logging configuration')
    args_parser.add_argument('--verbose',
                             dest='verbose',
                             action='store_true',
                             help='Set logging level to verbose')

    # inverse deformation field generation parameters (used when the global
    # transform is a deformation field and an inverse output is requested)
    args_parser.add_argument('--inv-step',
                             dest='inv_step',
                             type=float,
                             default=1.0,
                             help="Inverse transformation step")
    args_parser.add_argument('--inv-iterations',
                             dest='inv_iterations',
                             type=inttuple,
                             default=(10,),
                             help="Number of iterations for the inverse transformation")
    args_parser.add_argument('--inv-shrink-spacings',
                             dest='inv_shrink_spacings',
                             type=floattuple,
                             default=None,
                             help="Inverse shrink spacings")
    args_parser.add_argument('--inv-smooth-sigmas',
                             dest='inv_smooth_sigmas',
                             type=floattuple,
                             default=(0.,),
                             help="Inverse smooth sigmas")
    args_parser.add_argument('--inv-step-cut-factor',
                             dest='inv_step_cut_factor',
                             type=float,
                             default=0.5,
                             help="Inverse step cut factor")
    args_parser.add_argument('--inv-pad',
                             dest='inv_pad',
                             type=float,
                             default=0.1,
                             help="Inverse pad value")
    args_parser.add_argument('--inv-use-root',
                             dest='inv_use_root',
                             action='store_true',
                             default=False,
                             help="Use root for inverse displacement")

    return args_parser


def _run_global_align(reg_args:RegistrationInputs,
                      align_config,
                      compressor,
                      compressor_opts,
                      zarr_format,
                      prealign_downsample=1,
                      sharding_factor=None,
                      save_composed_transform=False,
                      inv_transform_args=None,
                      local_workers=1):
    global_steps, _ = get_algorithm_parameters(align_config,
                                               'global_align',
                                               reg_args.registration_steps)
    if len(global_steps) == 0:
        logger.info('Skip global alignment because no global steps were specified.')
        return None

    fix, fix_mask, mov, mov_mask, roi, fix_mask_roi, mov_mask_roi = get_input_images(reg_args)
    if fix.has_data() and mov.has_data():
        # compose mov origin transform from user affine + OME translations
        mov_origin_transform = compose_origin_transform(
            reg_args.get_mov_origin_transform(),
            mov.get_attr('globalCoordinateTransformations'),
        )
        prealign_steps, _ = get_algorithm_parameters(align_config,
                                                     'global_prealign',
                                                     reg_args.prealign_steps)
        # calculate and apply the global transform
        static_transforms, static_transforms_spacings = reg_args.get_static_transforms()
        transform, aligned = _align_global_data(fix, fix_mask,
                                                mov, mov_mask,
                                                roi,
                                                fix_mask_roi, mov_mask_roi,
                                                prealign_steps,
                                                global_steps,
                                                reg_args.processing_size,
                                                reg_args.processing_overlap_factor,
                                                reg_args.foreground_percentage,
                                                mov_origin_transform,
                                                static_transforms,
                                                static_transforms_spacings,
                                                prealign_downsample=prealign_downsample,
                                                local_workers=local_workers)
        if len(transform.shape) == 2:
            logger.info(f'Global affine transform: {transform}')
        else:
            logger.info(f'Global deform transform: {transform.shape}')
        # save global aligned volume
        _save_aligned_volume(
            reg_args,
            fix,
            aligned,
            fix.get_attr('axes'),
            fix.get_attr('coordinateTransformations'),
            fix.voxel_spacing,
            fix.voxel_downsampling,
            compressor,
            compressor_opts,
            zarr_format,
            sharding_factor,
        )
        # save the transform
        transform_subpath = reg_args.transform_subpath or reg_args.mov_subpath
        if reg_args.transform_blocksize:
            # block chunks are defined as x,y,z so reverse them to z,y,x
            transform_blocksize = reg_args.transform_blocksize[::-1]
        else:
            transform_blocksize = reg_args.output_blocksize[::-1]

        if save_composed_transform:
            if len(static_transforms) > 0:
                logger.info(f'Composing global transformation result with {len(static_transforms)} static transforms')
                final_transforms = static_transforms + [transform]
                final_transforms_spacings = static_transforms_spacings + (fix.voxel_spacing,)
                transform_to_save = bst.compose_transform_list(final_transforms, final_transforms_spacings)
            else:
                transform_to_save = transform
        else:
            transform_to_save = transform
        _save_transform(transform_to_save,
                        reg_args.transform_path(),
                        transform_subpath,
                        fix_image=fix,
                        compressor=compressor,
                        compressor_opts=compressor_opts,
                        zarr_format=zarr_format,
                        transform_blocksize=transform_blocksize,
                        sharding_factor=sharding_factor)
        # generate and save the global inverse transform
        inv_transform_path = reg_args.inv_transform_path()
        if inv_transform_path:
            inverse_transform = _generate_inverse_transform(transform, fix,
                                                            **(inv_transform_args or {}))
            if inverse_transform is not None:
                inv_transform_subpath = reg_args.inv_transform_subpath or transform_subpath
                _save_transform(inverse_transform,
                                inv_transform_path,
                                inv_transform_subpath,
                                fix_image=fix,
                                compressor=compressor,
                                compressor_opts=compressor_opts,
                                zarr_format=zarr_format,
                                transform_blocksize=transform_blocksize,
                                sharding_factor=sharding_factor)
        else:
            logger.info('Skip saving global inverse transformation')

    else:
        logger.info('Skip global alignment - both fix and moving image are needed')


def _align_global_data(
        fix_image, fix_mask_arg,
        mov_image, mov_mask_arg,
        roi,
        fix_mask_roi, mov_mask_roi,
        prealign_steps,
        steps,
        processing_size,
        processing_overlap_factor,
        foreground_percentage,
        mov_origin_transform,
        static_transforms,
        static_transforms_spacings=(),
        prealign_downsample=1,
        local_workers=1,
):
    logger.info('Read image data for global alignment')
    if isinstance(fix_mask_arg, ImageData):
        logger.info(f'Alignment fix mask: {fix_mask_arg}')
        if fix_mask_roi:
            logger.info(f'Clip fix mask to roi: {fix_mask_roi}')
            fix_mask = clip_arr_to_roi(fix_mask_arg.image_array[...], fix_mask_roi)
        else:
            fix_mask = fix_mask_arg.image_array[...]
    else:
        fix_mask = fix_mask_arg
    if isinstance(mov_mask_arg, ImageData):
        logger.info(f'Alignment mov mask: {mov_mask_arg}')
        if mov_mask_roi:
            logger.info(f'Clip mov mask to roi: {mov_mask_roi}')
            mov_mask = clip_arr_to_roi(mov_mask_arg.image_array[...], mov_mask_roi)
        else:
            mov_mask = mov_mask_arg.image_array[...]
    else:
        mov_mask = mov_mask_arg

    logger.info(f'Calculate global transform using: {steps}')
    fix_spacing = get_spatial_values(fix_image.voxel_spacing)
    logger.info(f'Fix image voxel spacing: {fix_spacing}')
    mov_spacing = get_spatial_values(mov_image.voxel_spacing)
    logger.info(f'Moving image voxel spacing: {mov_spacing}')

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
    # moving-image origin (translation) used by both the non-blockwise alignment
    # and the blockwise prealign; compute it once so both branches can use it
    if mov_origin_transform is not None:
        mov_origin = mov_origin_transform[:3, 3]
    else:
        mov_origin = None

    if processing_size is None:
        transform = alignment_pipeline(fix_image_array,
                                       mov_image_array,
                                       fix_spacing / fix_image.expansion_factor,
                                       mov_spacing / fix_image.expansion_factor,
                                       steps,
                                       fix_mask=fix_mask,
                                       mov_mask=mov_mask,
                                       roi=roi,
                                       fix_origin=None,
                                       mov_origin=mov_origin,
                                       static_transform_list=static_transforms)
    else:
        # if the processing_size is specified run a blockwise alignment using a
        block_zyx = tuple(int(s) for s in processing_size[::-1])
        displacement_vector_ndim = int(fix_image.spatial_ndim)
        transform_tmp_dir = tempfile.TemporaryDirectory(prefix='.global_deform_',
                                                        dir=os.getcwd())
        if len(prealign_steps) > 0:
            # Run a rough registration to improve the odds for the blockwise registration
            prealign_transform = _prealign(
                fix_image_array, mov_image_array,
                fix_spacing / fix_image.expansion_factor,
                mov_spacing / fix_image.expansion_factor,
                fix_mask,
                mov_mask,
                prealign_steps,
                static_transforms=static_transforms,
                downsample=prealign_downsample
            )
            logger.info(f'Pre-align transform: {prealign_transform}')
            transforms_list = static_transforms + [ prealign_transform ]
        else:
            logger.info('Skip pre-align')
            prealign_transform = None
            transforms_list = static_transforms

        transform = zarr.open(
            os.path.join(transform_tmp_dir.name, 'global-deform-field.zarr'),
            mode='w',
            shape=tuple(int(d) for d in fix_image.spatial_dims) + (displacement_vector_ndim,),
            chunks=block_zyx + (displacement_vector_ndim,),
            dtype=np.float32,
        )
        logger.info((
            f'Run a blockwise alignment with {processing_size} processing size and {processing_overlap_factor} overlap '
            f'to align a {mov_image.spatial_dims} moving image to a {fix_image.spatial_dims} fixed image '
            f'and generate a {transform.shape} deformation field '
        ))

        # create a local dask scheduler. memory_limit=0 disables the per-worker
        # memory monitor so it does not spill/pause/terminate or emit the noisy
        # "unmanaged memory is high" warnings when a block's buffers exceed the
        # (auto-detected, often tiny) per-worker limit
        cluster_client = Client(LocalCluster(n_workers=local_workers,
                                             threads_per_worker=1,
                                             memory_limit=0,
                                             processes=False))
        deform_ok = distributed_alignment_pipeline(
            fix_image,
            fix_spacing / fix_image.expansion_factor,
            mov_image,
            mov_spacing / fix_image.expansion_factor,
            steps,
            block_zyx,
            cluster_client,
            overlap_factor=processing_overlap_factor,
            fix_mask=fix_mask_arg,
            mov_mask=mov_mask_arg,
            roi=roi,
            foreground_percentage=foreground_percentage,
            mov_origin_transform=mov_origin_transform,
            static_transform_list=transforms_list,
            output_transform=transform,
        )
        logger.info((
            f'Finished computing the {transform.shape} deformation field '
            f'for the alignment of {mov_image} to {fix_image} -> {deform_ok} '
        ))
        # materialize the assembled field so downstream apply/save see a numpy
        # array (same as the non-blockwise path), then drop the temp store
        transform = transform[...]
        transform_tmp_dir.cleanup()
        if not transform.any():
            logger.info('Got an empty deformation field (all zeros)')
        # fold the prealign into the result
        if prealign_transform is not None:
            transform = bst.compose_transform_list(
                [prealign_transform, transform],
                [None, fix_spacing / fix_image.expansion_factor],
            )

    transform_spacing = fix_spacing / fix_image.expansion_factor
    if len(transform.shape) == 2:
        logger.info(f'Apply affine transform: {transform}, spacing: {transform_spacing}')
    else:
        logger.info(f'Apply deform transform: {transform.shape}, spacing: {transform_spacing}')
    # apply transform. the static transforms carry their own spacings (a global
    # deform may have been generated at a different scale); the freshly-computed
    # transform is at the current fix scale (a matrix ignores spacing).
    transforms_list = static_transforms + [transform]
    transforms_spacings = tuple(static_transforms_spacings) + (transform_spacing,)
    aligned = apply_transform(fix_image_array,
                              mov_image_array,
                              fix_spacing / fix_image.expansion_factor,
                              mov_spacing / mov_image.expansion_factor,
                              transform_list=transforms_list,
                              transform_spacing=transforms_spacings)
    return transform, aligned


def _prealign(fix, mov,
              fix_spacing, mov_spacing,
              fix_mask, mov_mask,
              prealign_steps,
              static_transforms=[],
              downsample=1):
    """
    Really rough fix and mov image registration to give better chance of 
    success to the next step.
    """
    logger.info((
        f'Prealign {fix.shape} image to {mov.shape} image '
        f'fix spacing {fix_spacing}, mov spacing {mov_spacing} '
        f'downsample: {downsample}, steps: {prealign_steps}'
    ))
    if downsample > 1:
        f = fix[::downsample, ::downsample, ::downsample]
        m = mov[::downsample, ::downsample, ::downsample]
        fspacing = np.asarray(fix_spacing, float) * downsample
        mspacing = np.asarray(mov_spacing, float) * downsample
        if fix_mask is not None:
            fm = fix_mask[::downsample, ::downsample, ::downsample]
        else:
            fm = None
        if mov_mask is not None:
            mm = mov_mask[::downsample, ::downsample, ::downsample]
        else:
            mm = None
    else:
        f = fix
        m = mov
        fspacing = np.asarray(fix_spacing, float)
        mspacing = np.asarray(mov_spacing, float)
        if fix_mask is not None:
            fm = fix_mask
        else:
            fm = None
        if mov_mask is not None:
            mm = mov_mask
        else:
            mm = None

    return alignment_pipeline(
        f, m,
        fspacing, mspacing,                                    
        prealign_steps,
        fix_mask=fm,
        mov_mask=mov_mask,
        static_transform_list=static_transforms,
    )


def _apply_global_transform(reg_args:RegistrationInputs,
                            transform,
                            transform_spacing,
                            compressor,
                            compressor_opts,
                            zarr_format,
                            sharding_factor=None):
    (fix_image, _, mov_image, _, _, _, _) = get_input_images(reg_args)
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
        static_transforms, static_transforms_spacings = reg_args.get_static_transforms()
        transforms_list = static_transforms + [transform,]
        transforms_spacings = static_transforms_spacings + (transform_spacing,)
        fix_spatial_spacing = get_spatial_values(fix_image.voxel_spacing) / fix_image.expansion_factor
        mov_spatial_spacing = get_spatial_values(mov_image.voxel_spacing) / mov_image.expansion_factor
        aligned = apply_transform(fix_image_array,
                                  mov_image_array,
                                  fix_spatial_spacing,
                                  mov_spatial_spacing,
                                  transform_list=transforms_list,
                                  transform_spacing=transforms_spacings)
        _save_aligned_volume(
            reg_args,
            fix_image,
            aligned,
            fix_image.get_attr('axes'),
            fix_image.get_attr('coordinateTransformations'),
            fix_image.voxel_spacing,
            fix_image.voxel_downsampling,
            compressor,
            compressor_opts,
            zarr_format,
            sharding_factor,
        )
    else:
        # both fix and mov volume must be valid
        return None


def _save_transform(transform, transform_path, transform_subpath,
                    fix_image=None,
                    compressor=None, compressor_opts=None,
                    zarr_format=None,
                    transform_blocksize=None, sharding_factor=None):
    if not transform_path:
        logger.info('Skip saving global transformation')
        return

    if len(transform.shape) == 2:
        transform_location = os.path.dirname(transform_path)
        os.makedirs(transform_location, exist_ok=True)
        logger.info(f'Save global affine transformation to {transform_path}')
        np.savetxt(transform_path, transform)
    else:
        # global transform is a deformation field - save it as an OME-ZARR
        logger.info((
            f'Save global deform transformation to '
            f'{transform_path}:{transform_subpath}'
        ))
        _save_global_deform_field(
            transform, transform_path, transform_subpath,
            transform_blocksize, fix_image,
            compressor, compressor_opts, zarr_format, sharding_factor,
        )


def _generate_inverse_transform(transform, fix_image,
                                inv_step=1,
                                inv_iterations=(100,),
                                inv_shrink_spacings=(None,),
                                inv_smooth_sigmas=(0.,),
                                inv_step_cut_factor=0.5,
                                inv_pad=0.1,
                                inv_use_root=True,
                                ):
    """
    Generate (but do not save) the inverse of a global transform.

    Returns the inverse affine matrix or the inverted deformation field, or None
    if the transform cannot be inverted (singular affine, or a deformation field
    without the fixed image needed to determine its physical spacing).
    """
    if len(transform.shape) == 2:
        try:
            return np.linalg.inv(transform)
        except np.linalg.LinAlgError:
            logger.error(f'Global affine {transform} is not invertible')
            return None

    if fix_image is None:
        logger.warning(
            'Cannot compute global inverse deform field without the fixed image; skipping'
        )
        return None

    # the field displacements are in the same physical frame the alignment
    # used, i.e. fix spacing scaled by the expansion factor
    field_spacing = (np.asarray(get_spatial_values(fix_image.voxel_spacing),
                                dtype=np.float64) / fix_image.expansion_factor)
    logger.info((
        f'Compute global inverse deform field from {transform.shape} '
        f'using spacing {field_spacing} '
    ))
    return invert_displacement_vector_field(
        transform,
        field_spacing,
        step=inv_step,
        iterations=inv_iterations,
        shrink_spacings=inv_shrink_spacings,
        smooth_sigmas=inv_smooth_sigmas,
        step_cut_factor=inv_step_cut_factor,
        pad=inv_pad,
        use_root=inv_use_root,
        verbose=True,
    )


def _save_global_deform_field(deformfield_array,
                              deformfield_path, deformfield_subpath,
                              deformfield_blocksize, fix_image,
                              compressor, compressor_opts,
                              zarr_format, sharding_factor=None):
    """
    Persist a global deformation (displacement) field as an OME-ZARR dataset.

    The layout mirrors the local pipeline: spatial axes plus a trailing
    'displacement' axis, so global and local deform fields are read the same way.
    """
    if fix_image is None:
        logger.warning('Cannot save global deform field without the fixed image; skipping')
        return None

    # spatial voxel spacing / downsampling, with a trailing entry for the vector axis
    transform_downsampling = tuple(get_spatial_values(fix_image.voxel_downsampling)) + (1,)
    transform_voxel_spacing = tuple(get_spatial_values(fix_image.voxel_spacing)) + (1,)
    deformfield_shape = deformfield_array.shape
    vector_ndim = deformfield_shape[-1]
    logger.info(f'Global deform field shape: {deformfield_shape}')

    # build OME axes: spatial axes + a displacement axis for the vector components.
    # copy the spatial axes into a fresh list - get_spatial_values may return a
    # reference into the ImageData's cached axes, and appending in place would
    # corrupt it across calls (e.g. forward then inverse field).
    spatial_axes = get_spatial_values(fix_image.get_attr('axes'))
    if spatial_axes is not None:
        deformfield_axes = list(spatial_axes) + [{
            'name': 'd',
            'type': 'displacement',
            'discrete': True,
        }]
    else:
        deformfield_axes = None
    # extend the coordinate transformations with a value for the displacement axis
    deformfield_coord_transforms = fix_image.get_attr('coordinateTransformations')
    if deformfield_coord_transforms is not None:
        new_transforms = []
        for ct in deformfield_coord_transforms:
            cttype = ct['type']
            tx = ct[cttype]
            chtx = tx[1]
            new_transforms.append({
                'type': cttype,
                cttype: get_spatial_values(tx) + [chtx],
            })
        deformfield_coord_transforms = new_transforms

    deformfield_attrs = io_utility.prepare_parent_group_attrs(
        deformfield_path,
        deformfield_subpath,
        axes=deformfield_axes,
        dataset_transformations=deformfield_coord_transforms,
        zarr_format=zarr_format,
    )

    deformfield_spatial_chunksize = tuple(get_spatial_values(deformfield_blocksize))
    deformfield_output_chunksize = deformfield_spatial_chunksize + (vector_ndim,)

    # factor applies to spatial axes only; the vector axis is never sharded
    deformfield_spatial_shard = derive_shard_shape(
        sharding_factor,
        deformfield_spatial_chunksize,
        zarr_format,
    )
    if deformfield_spatial_shard is not None:
        deformfield_output_shardsize = tuple(deformfield_spatial_shard) + (vector_ndim,)
    else:
        deformfield_output_shardsize = None

    logger.info((
        f'Create global deform field dataset {deformfield_path}:{deformfield_subpath} '
        f'shape {deformfield_shape} '
        f'blocksize {deformfield_output_chunksize} '
        f'shardsize {deformfield_output_shardsize} '
    ))
    deformfield = io_utility.create_dataset_array(
        deformfield_path,
        deformfield_subpath,
        deformfield_shape,
        deformfield_output_chunksize,
        np.float32,
        overwrite=True,
        compressor=compressor,
        compression_opts=compressor_opts,
        parent_attrs=deformfield_attrs,
        pixelResolution=calc_full_voxel_resolution_attr(transform_voxel_spacing,
                                                        transform_downsampling),
        downsamplingFactors=calc_downsampling_attr(transform_downsampling),
        zarr_format=zarr_format,
        shard_shape=deformfield_output_shardsize,
    )
    deformfield[...] = deformfield_array.astype(np.float32)
    logger.info(f'Saved global deform field to {deformfield_path}:{deformfield_subpath}')
    return deformfield


def _save_aligned_volume(reg_args:RegistrationInputs,
                         fix_image,
                         aligned_array,
                         axes,
                         dataset_transformations,
                         aligned_spacing,
                         aligned_downsampling,
                         compressor,
                         compressor_opts,
                         zarr_format,
                         sharding_factor=None):
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
        logger.info(f'Prepare to save global alignment to {align_path}')
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
            align_blocksize = reg_args.output_blocksize[::-1]

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

        aligned_dataset_shardsize = derive_shard_shape(
            sharding_factor, aligned_dataset_chunksize, zarr_format
        )

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
            pixelResolution=calc_full_voxel_resolution_attr(aligned_spacing,
                                                            aligned_downsampling),
            downsamplingFactors=calc_downsampling_attr(aligned_downsampling),
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
    global_transform_path = reg_inputs.transform_path()
    global_transform = None
    global_transform_spacing = None
    if args.reuse_existing_transform:
        # try to read the global transform
        transform_subpath = reg_inputs.transform_subpath or reg_inputs.mov_subpath
        logger.info(f'Global transform path: {global_transform_path}:{transform_subpath}')
        global_transform, global_transform_spacing = get_transform(global_transform_path, transform_subpath)

    set_cpu_resources(args.cpus)

    output_zarr_format = get_zarr_format(reg_inputs.align_path(), args.output_zarr_format)

    # inverse deform field generation parameters. shrink spacings default to one
    # None per iteration level when not provided
    inv_shrink_spacings = (args.inv_shrink_spacings
                           if (args.inv_shrink_spacings is not None and
                               len(args.inv_shrink_spacings) > 0)
                           else (None,) * len(args.inv_iterations))
    inv_transform_args = dict(
        inv_step=args.inv_step,
        inv_iterations=args.inv_iterations,
        inv_shrink_spacings=inv_shrink_spacings,
        inv_smooth_sigmas=args.inv_smooth_sigmas,
        inv_step_cut_factor=args.inv_step_cut_factor,
        inv_pad=args.inv_pad,
        inv_use_root=args.inv_use_root,
    )

    if global_transform is None:
        # no global transform found -> calculate it and then apply it
        _run_global_align(reg_inputs,
                          args.align_config,
                          args.compressor,
                          args.compressor_opts,
                          output_zarr_format,
                          prealign_downsample=args.prealign_downsample,
                          sharding_factor=args.output_sharding_factor,
                          save_composed_transform=args.save_composed_transform,
                          inv_transform_args=inv_transform_args,
                          local_workers=args.local_dask_workers)
    else:
        # global transform found -> just apply it
        _apply_global_transform(reg_inputs,
                                global_transform,
                                global_transform_spacing,
                                args.compressor, args.compressor_opts,
                                output_zarr_format,
                                sharding_factor=args.output_sharding_factor)


if __name__ == '__main__':
    main()
