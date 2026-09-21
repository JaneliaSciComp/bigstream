"""
Compute a local deformation field with the multi-pass blockwise pipeline.

This is the CLI for `bigstream.distributed_align` and it only computes the 
transformation for the fine alignment.

The registration is described by `local_align.alignment_passes` in the align
config: several passes over the volume, each on its own block lattice, each
fitting only the residual its predecessors left. A config that has no
`alignment_passes` still works - its single `steps:` list is run as one pass.

Note that the processing size does not have to be a multiple of the output
chunk or shard. Writes are locked on the write unit rather than on the
block, so any block geometry is safe - see `max_write_locks`.
"""

import argparse
import logging

from copy import deepcopy

import numpy as np
import pydantic.v1.utils as pu
import yaml

from dask.distributed import (Client, LocalCluster)

import bigstream.io_utility as io_utility

from bigstream.configure_bigstream import (configure_logging,
                                           default_bigstream_config_str)
from bigstream.configure_dask import (ConfigureWorkerPlugin, load_dask_config)
from bigstream.distributed_align import (
    MAX_WRITE_LOCKS,
    AlignmentPass,
    DisplacementDiagnostics,
    alignment_passes_from_config,
    alignment_steps_from_config,
    blockwise_alignment_pipeline,
    default_blend_ramp_from_config,
    default_pass_geometry_from_config,
)
from bigstream.distributed_transform import distributed_apply_transform
from bigstream.image_data import (ImageData,
                                  calc_full_voxel_resolution_attr,
                                  calc_downsampling_attr)
from bigstream.ome_utils import (get_spatial_values, compose_origin_transform)

from .cli import (CliArgsHelper, RegistrationInputs,
                  define_registration_input_args,
                  extract_registration_input_args, get_algorithm_parameters,
                  get_input_images, get_transform, dictfromjson, inttuple)

from .utils import derive_shard_shape, get_zarr_format


# rebound to the configured root logger by main(); a real logger up
# front so the module's functions can also be called directly
logger = logging.getLogger(__name__)

# Block overlap used when warping the moving image, as a fraction of the
# block size. This is resampling overlap - enough halo that interpolation
# near a block face has neighbours to read - and has nothing to do with the
# alignment halo, which exists to blend disagreeing per block fits.
DEFAULT_TRANSFORM_OVERLAP = 0.125


# Arguments `define_registration_input_args` contributes that this tool does
# not implement. They stay in the parser so the shared definition is not
# forked, but they are hidden from --help and refused if actually passed -
# silently ignoring an output the caller asked for would be worse.
#
# Warping the moving image is back in scope (the `--local-align-*` group);
# only the inverse field is not, because inverting is a separate, expensive
# fixed point solve that has its own tool.
_INVERSE = ('this tool does not compute the inverse field; '
            'use main_compute_local_inverse')
_UNSUPPORTED_ARGS = {
    'inv_transform_name': _INVERSE,
    'inv_transform_subpath': _INVERSE,
    'inv_transform_blocksize': _INVERSE,
}


def _define_args(local_descriptor):
    args_parser = argparse.ArgumentParser(
        description='Compute a local deformation field (multi-pass '
                    'blockwise) and optionally warp the moving image '
                    'through it')

    define_registration_input_args(
        args_parser.add_argument_group(
            description='Local registration input volumes'),
        local_descriptor,
    )
    _hide_unsupported_args(args_parser, local_descriptor)

    args_parser.add_argument('--align-config',
                             dest='align_config',
                             help='Align config file holding the '
                                  'local_align.alignment_passes section')
    args_parser.add_argument('--initial-transform', '--global-transform',
                             dest='initial_transform',
                             help='Initial transform path')
    args_parser.add_argument('--initial-transform-subpath',
                             '--global-transform-subpath',
                             dest='initial_transform_subpath',
                             help='Initial transform subpath')

    args_parser.add_argument('--local-transform-overlap-factor',
                             '--transform-overlap-factor',
                             dest='transform_overlap_factor',
                             type=float, default=None,
                             help='Block overlap, as a fraction of the block '
                                  'size, used when applying the transform to '
                                  'warp the moving image. Unrelated to the '
                                  'alignment halo. Defaults to '
                                  'local_align.transform_overlap in the '
                                  f'config, else {DEFAULT_TRANSFORM_OVERLAP}.')
    args_parser.add_argument('--pass-fields-path',
                             dest='pass_fields_path',
                             help='Container for the intermediate per-pass '
                                  'fields (defaults to the transform '
                                  'container). Only used when the config has '
                                  'more than one pass. These are kept after '
                                  'the run - they are what the per-pass '
                                  'diagnostics are read from - and are never '
                                  'deleted automatically.')
    args_parser.add_argument('--resume-from-pass', '--resume_from_pass',
                             dest='resume_from_pass',
                             type=int, default=None,
                             help='1-indexed pass to resume from, e.g. after '
                                  'a later pass failed. Passes before it are '
                                  'not recomputed; their fields are read '
                                  'back from --pass-fields-path (or the '
                                  'transform container) instead, at the '
                                  'same location a normal run would have '
                                  'written them. A value greater than the '
                                  'number of configured passes is ignored '
                                  'and the run starts from pass 1.')
    args_parser.add_argument('--max-write-locks', '--max_write_locks',
                             dest='max_write_locks',
                             type=int, default=MAX_WRITE_LOCKS,
                             help='Largest number of lock names one block '
                                  'write may hold. Block writes are locked '
                                  'per output chunk (or shard), so the '
                                  'processing size need not be a multiple of '
                                  'either; a block much larger than the chunk '
                                  'coarsens its lock grid to stay under this.')

    args_parser.add_argument('--dask-scheduler', dest='dask_scheduler',
                             type=str, default=None,
                             help='Run with distributed scheduler')
    args_parser.add_argument('--dask-config', dest='dask_config',
                             type=str, default=None,
                             help='YAML file containing dask configuration')
    args_parser.add_argument('--local-dask-workers', '--local_dask_workers',
                             dest='local_dask_workers',
                             type=int,
                             help='Number of workers when using a local cluster')
    args_parser.add_argument('--worker-cpus', dest='worker_cpus',
                             type=int, default=1,
                             help='Number of cpus allocated to a dask worker')
    args_parser.add_argument('--max-worker-threads-per-cpu',
                             dest='max_worker_threads_per_cpu',
                             type=int, default=1,
                             help='Maximum number of threads to run on a worker')
    args_parser.add_argument('--max-cluster-jobs', '--max_cluster_jobs',
                             dest='max_cluster_jobs',
                             type=int, default=0,
                             help='Maximum number of cluster jobs executed in parallel')
    args_parser.add_argument('--max-concurrent-zarr-reads',
                             dest='max_concurrent_zarr_reads',
                             type=int, default=0,
                             help='Maximum number of concurrent reads from a zarr array')

    args_parser.add_argument('--displacement-diagnostics',
                             '--displacement_diagnostics',
                             dest='displacement_diagnostics',
                             choices=[m.value for m in DisplacementDiagnostics],
                             default=None,
                             help='When to run displacement field '
                                  'diagnostics (jacobian folding statistics) '
                                  'over a whole assembled field. PER_STEP: '
                                  'after every alignment pass, and on the '
                                  'composed result. FINAL_STEP: on the final '
                                  'field only. Omit for none. Per-block '
                                  'diagnostics are separate and always go to '
                                  'the debug log.')
    args_parser.add_argument('--error-if-displacement-check-fails',
                             '--error_if_displacement_check_fails',
                             dest='error_if_displacement_check_fails',
                             action='store_true',
                             help='Fail instead of warning when a pass '
                                  'configures displacement bounds that are '
                                  'too loose for its block lattice')

    args_parser.add_argument('--compression', '--compressor',
                             dest='compressor',
                             default='zstd', type=str,
                             help='Codec used for zarr arrays. '
                                  'Valid values are: raw,lz4,gzip,bz2,blosc,zstd')
    args_parser.add_argument('--compression-opts', '--compressor-opts',
                             dest='compressor_opts',
                             default={}, type=dictfromjson,
                             help='Zarr array compression options')
    args_parser.add_argument('--output-zarr-format', '--output_zarr_format',
                             dest='output_zarr_format',
                             type=int,
                             help='Zarr output format')
    args_parser.add_argument('--output-sharding-factor',
                             '--output_sharding_factor',
                             dest='output_sharding_factor',
                             default=None, type=inttuple,
                             help='Zarr v3 sharding factor in xyz order, '
                                  'e.g. 8,8,4, applied to the deformfield '
                                  'chunk shape. Ignored when '
                                  '--output-zarr-format is not 3.')

    args_parser.add_argument('--logging-config', dest='logging_config',
                             type=str, help='Logging configuration')
    args_parser.add_argument('--verbose', dest='verbose',
                             action='store_true',
                             help='Set logging level to verbose')

    return args_parser


def _hide_unsupported_args(args_parser, args_descriptor: CliArgsHelper):
    """
    Keep the shared input args but hide the ones this tool cannot honour.

    Reaches into `_actions` because argparse has no public way to drop an
    argument someone else added; the alternative is forking two hundred
    lines of shared input definitions, which would drift.
    """
    unsupported = {args_descriptor.argdest(name)
                   for name in _UNSUPPORTED_ARGS}
    for action in args_parser._actions:
        if action.dest in unsupported:
            action.help = argparse.SUPPRESS


def _reject_unsupported_args(args_parser, args,
                             args_descriptor: CliArgsHelper):
    """
    Fail on any `_UNSUPPORTED_ARGS` the caller actually passed.

    "Actually passed" means "differs from the parser default", not "is
    truthy": channel 0 and time index 0 are real values and both are falsy,
    so a truthiness test lets exactly the arguments most likely to be typed
    slip through and be silently ignored.
    """
    given = []
    for name, reason in _UNSUPPORTED_ARGS.items():
        dest = args_descriptor.argdest(name)
        if not hasattr(args, dest):
            continue
        if getattr(args, dest) != args_parser.get_default(dest):
            given.append(f'{args_descriptor.argflag(name.replace("_", "-"))} '
                         f'({reason})')
    if given:
        raise SystemExit(
            'These options are not supported by this tool:\n  '
            + '\n  '.join(given)
        )


def _load_align_config(config_filename):
    """The bigstream defaults with the user's config layered on top."""
    config = yaml.safe_load(default_bigstream_config_str)
    if config_filename:
        with open(config_filename) as f:
            external_config = yaml.safe_load(f)
        logger.info(f'Read external config from {config_filename}')
        config = pu.deep_update(config, external_config)
    return config


def _get_alignment_passes(config, registration_steps, context='local_align'):
    """
    The passes to run, from `local_align.alignment_passes`.

    A config written for the older single-pass pipeline has no
    `alignment_passes`, only a flat `steps:` list. Rather than refuse it, run
    it as one pass - which is exactly what the multi-pass pipeline reduces to
    at one pass and a zero lattice offset, so such a config keeps its old
    meaning.
    """
    passes = alignment_passes_from_config(config, context)
    if passes:
        return passes

    context_config = config.get(context) or {}
    steps = registration_steps or context_config.get('steps') or []
    if not steps:
        return []
    logger.info(f'No {context}.alignment_passes in the config; running the '
                f'single steps list {steps} as one pass')
    # mirror get_algorithm_parameters' precedence: global per-step defaults,
    # then the local_align overrides for that step. deep_update writes into
    # its first argument, so copy rather than edit the config in place
    step_defaults = {name: pu.deep_update(deepcopy(config.get(name) or {}),
                                          deepcopy(context_config.get(name) or {}))
                     for name in steps}
    return [AlignmentPass(
        alignment_steps=alignment_steps_from_config(list(steps),
                                                     step_defaults=step_defaults),
        name='pass1',
    )]


def _run_local_alignment(reg_args: RegistrationInputs,
                         align_config,
                         initial_transform,
                         initial_transform_spacing=None,
                         transform_overlap_factor=None,
                         pass_fields_path=None,
                         resume_from_pass=None,
                         max_write_locks=MAX_WRITE_LOCKS,
                         dask_scheduler_address=None,
                         dask_config_file=None,
                         dask_workers=None,
                         worker_cpus=1,
                         worker_threads_per_cpu=1,
                         logging_config=None,
                         compressor=None,
                         compressor_opts={},
                         zarr_format=3,
                         sharding_factor=None,
                         verbose=False,
                         max_concurrent_zarr_reads=0,
                         max_cluster_jobs=0,
                         displacement_diagnostics=None,
                         error_if_displacement_check_fails=False):
    config = _load_align_config(align_config)
    alignment_passes = _get_alignment_passes(config, reg_args.registration_steps)
    if not alignment_passes:
        logger.info('Skip local alignment: no alignment passes and no steps.')
        return True

    default_size, default_halo_factor = default_pass_geometry_from_config(config)
    default_blend_ramp = default_blend_ramp_from_config(config)

    if reg_args.processing_size:
        # the CLI takes xyz, everything below here is zyx
        default_size = tuple(reg_args.processing_size)[::-1]
        logger.info(f'Default processing size {default_size} (zyx) from '
                    f'--local-processing-size {reg_args.processing_size} (xyz)')
    if reg_args.processing_overlap_factor:
        default_halo_factor = reg_args.processing_overlap_factor
        if isinstance(default_halo_factor, (tuple, list)):
            default_halo_factor = tuple(default_halo_factor)[::-1]
    # deliberately not rounded up to the chunk or shard: writes are locked on
    # the write unit, so the block geometry is free

    (fix_image, fix_mask, mov_image, mov_mask, roi) = get_input_images(reg_args)
    if mov_image.ndim != fix_image.ndim:
        raise ValueError(f'{mov_image} is expected to have the same ndim as '
                         f'{fix_image}')
    if not (fix_image.has_data() and mov_image.has_data()):
        raise ValueError('Either the fixed or the moving image has no data')

    deformfield_path = reg_args.transform_path()
    if not deformfield_path:
        raise SystemExit('No transform output was given; the deformation '
                         'field is this tool\'s primary output, so there is '
                         'nothing to do. Set --local-transform-name.')
    deformfield_subpath = reg_args.transform_subpath or reg_args.mov_subpath
    if reg_args.transform_blocksize:
        deformfield_chunksize = tuple(reg_args.transform_blocksize)[::-1]
    else:
        deformfield_chunksize = tuple(reg_args.output_blocksize)[::-1]

    # warping the moving image is optional and only happens when an output
    # for it was named
    align_path = reg_args.align_path()
    align_subpath = reg_args.align_dataset()
    if reg_args.align_blocksize:
        align_chunksize = tuple(reg_args.align_blocksize)[::-1]
    else:
        align_chunksize = deformfield_chunksize

    transform_overlap = transform_overlap_factor
    if transform_overlap is None:
        transform_overlap = (config.get('local_align') or {}).get(
            'transform_overlap', DEFAULT_TRANSFORM_OVERLAP)
    if not 0 <= transform_overlap < 1:
        raise SystemExit('--local-transform-overlap-factor must be in '
                         f'[0, 1), got {transform_overlap}')

    # how the warp resamples - order, mode - is configured separately from
    # the alignment steps, under `apply_deform`
    transform_coords_args = {}
    if align_path:
        apply_deform_steps, _ = get_algorithm_parameters(
            align_config, 'apply_deform', ['map_coordinates'])
        for step, step_args in apply_deform_steps:
            if step == 'map_coordinates':
                transform_coords_args.update(step_args)

    load_dask_config(dask_config_file)
    if dask_scheduler_address:
        logger.info(f'Use dask scheduler at: {dask_scheduler_address}')
        cluster_client = Client(address=dask_scheduler_address)
    else:
        logger.info(f'Use a local dask with {dask_workers} local workers')
        cluster_client = Client(LocalCluster(n_workers=dask_workers,
                                             threads_per_worker=worker_cpus))
    worker_config = ConfigureWorkerPlugin(logging_config, verbose,
                                          worker_cpus=worker_cpus,
                                          worker_threads_per_cpu=worker_threads_per_cpu)
    cluster_client.register_plugin(worker_config, name='WorkerConfig')
    try:
        # the spacings are only needed to *apply* a field - the block
        # machinery derives a static field's spacing from its shape relative
        # to the fixed image - but the warp stage below needs them, so they
        # are carried through in step with the transforms themselves
        static_transforms, static_transforms_spacings = \
            reg_args.get_static_transforms()
        if initial_transform is not None:
            static_transforms = static_transforms + [initial_transform,]
            static_transforms_spacings = (tuple(static_transforms_spacings)
                                          + (initial_transform_spacing,))
        mov_origin_transform = compose_origin_transform(
            reg_args.get_mov_origin_transform(),
            mov_image.get_attr('globalCoordinateTransformations'),
        )
        deform_ok, deformfield = _compute_deform_field(
            fix_image, fix_mask, mov_image, mov_mask, roi,
            alignment_passes,
            default_size,
            default_halo_factor,
            default_blend_ramp,
            mov_origin_transform,
            static_transforms,
            deformfield_path,
            deformfield_subpath,
            deformfield_chunksize,
            pass_fields_path or deformfield_path,
            resume_from_pass,
            cluster_client,
            compressor,
            compressor_opts,
            zarr_format,
            sharding_factor,
            reg_args.foreground_percentage,
            max_concurrent_zarr_reads,
            max_cluster_jobs,
            max_write_locks,
            displacement_diagnostics,
            error_if_displacement_check_fails,
            not reg_args.norebalance_missing_neighbors,
        )
        if not align_path:
            logger.info('No aligned output was given, so the moving image '
                        'is not warped; set --local-align-name to warp it')
            return deform_ok
        # a run with static transforms still has something to apply even if
        # every block of this run's own field failed
        if not (deform_ok or static_transforms):
            logger.error('Skip warping the moving image: the deformation '
                         'field is incomplete and there are no static '
                         'transforms to fall back on')
            return False
        _apply_deform_field(
            fix_image, mov_image,
            deformfield if deform_ok else None,
            static_transforms,
            static_transforms_spacings,
            mov_origin_transform,
            reg_args.persist_mov_origin_transform,
            align_path,
            align_subpath,
            reg_args.align_timeindex,
            reg_args.align_channel,
            align_chunksize,
            transform_overlap,
            transform_coords_args,
            cluster_client,
            compressor,
            compressor_opts,
            zarr_format,
            sharding_factor,
        )
        return deform_ok
    finally:
        cluster_client.close()


def _compute_deform_field(fix_image: ImageData,
                          fix_mask,
                          mov_image: ImageData,
                          mov_mask,
                          roi,
                          alignment_passes,
                          default_processing_size,
                          default_halo_factor,
                          default_blend_ramp,
                          mov_origin_transform,
                          static_transforms,
                          deformfield_path,
                          deformfield_subpath,
                          deformfield_chunksize,
                          pass_fields_path,
                          resume_from_pass,
                          cluster_client,
                          compressor,
                          compressor_opts,
                          zarr_format,
                          sharding_factor,
                          foreground_percentage,
                          max_concurrent_zarr_reads,
                          max_cluster_jobs,
                          max_write_locks,
                          displacement_diagnostics,
                          error_if_displacement_check_fails,
                          rebalance_for_missing_neighbors):
    """
    Run the passes and write the composed field.

    Returns `(ok, deformfield)`. The array itself comes back so the warp
    stage can hand it straight to `distributed_apply_transform` instead of
    reopening what was just written.
    """
    logger.info(f'Compute the deformation field aligning {mov_image} to '
                f'{fix_image} over {len(alignment_passes)} pass(es)')

    deformfield_shape = (tuple(fix_image.spatial_dims)
                         + (len(fix_image.spatial_dims),))
    create_field = _deformfield_factory(
        fix_image, roi, alignment_passes, deformfield_chunksize,
        compressor, compressor_opts, zarr_format, sharding_factor,
        foreground_percentage, rebalance_for_missing_neighbors,
        default_blend_ramp,
        default_processing_size=default_processing_size,
        default_halo_factor=default_halo_factor,
    )

    deformfield = create_field(deformfield_path, deformfield_subpath,
                               deformfield_shape)

    def pass_output_factory(pass_index, shape):
        # only called when there is more than one pass; each pass's own
        # (residual) field is kept so the pass over pass diagnostics the
        # multi-pass design depends on can be read back afterwards
        subpath = f'{deformfield_subpath}_passes/pass{pass_index + 1}'
        logger.info(f'Create pass {pass_index + 1} field at '
                    f'{pass_fields_path}:{subpath}')
        return create_field(pass_fields_path, subpath, shape)

    npasses = len(alignment_passes)
    resumed_pass_fields = []
    # resume_from_pass beyond npasses is out of range; leave it to
    # blockwise_alignment_pipeline to ignore it and log why - reading fields
    # here would just be for a start_pass it is not going to honour
    if resume_from_pass is not None and 1 < resume_from_pass <= npasses:
        for n in range(1, resume_from_pass):
            subpath = f'{deformfield_subpath}_passes/pass{n}'
            logger.info(f'Resume from pass {resume_from_pass}: reading pass '
                        f'{n} field from {pass_fields_path}:{subpath}')
            resumed_pass_fields.append(
                ImageData(pass_fields_path, subpath, open_image=True))

    deform_ok = blockwise_alignment_pipeline(
        fix_image,
        np.array(get_spatial_values(fix_image.voxel_spacing)) / fix_image.expansion_factor,
        mov_image,
        np.array(get_spatial_values(mov_image.voxel_spacing)) / mov_image.expansion_factor,
        alignment_passes,
        cluster_client,
        processing_size=default_processing_size,
        processing_halo_factor=default_halo_factor,
        blend_ramp=default_blend_ramp,
        fix_mask=fix_mask,
        mov_mask=mov_mask,
        roi=roi,
        foreground_percentage=foreground_percentage,
        mov_origin_transform=mov_origin_transform,
        static_transform_list=static_transforms,
        deformfield_final_result=deformfield,
        deformfield_output_factory=pass_output_factory,
        max_concurrent_reads=max_concurrent_zarr_reads,
        max_cluster_jobs=max_cluster_jobs,
        max_write_locks=max_write_locks,
        rebalance_for_missing_neighbors=rebalance_for_missing_neighbors,
        displacement_diagnostics=displacement_diagnostics,
        error_if_displacement_check_fails=error_if_displacement_check_fails,
        start_pass=resume_from_pass or 1,
        resumed_pass_fields=resumed_pass_fields,
    )
    if deform_ok:
        logger.info(f'Wrote the deformation field to '
                    f'{deformfield_path}:{deformfield_subpath}')
    else:
        logger.error('Some blocks failed; the deformation field at '
                     f'{deformfield_path}:{deformfield_subpath} is incomplete')
    return bool(deform_ok), deformfield


def _apply_deform_field(fix_image: ImageData,
                        mov_image: ImageData,
                        deformfield,
                        static_transforms,
                        static_transforms_spacings,
                        mov_origin_transform,
                        persist_mov_origin_transform,
                        align_path,
                        align_subpath,
                        align_timeindex,
                        align_channel,
                        align_chunksize,
                        transform_overlap_factor,
                        transform_coords_args,
                        cluster_client,
                        compressor,
                        compressor_opts,
                        zarr_format,
                        sharding_factor):
    """
    Warp the moving image onto the fixed image grid and write it out.

    `deformfield` is this run's composed field, or None when every block of
    it failed and only the static transforms are left to apply. The static
    transforms come first in `transform_list` and the local deform last,
    which is the order `compose_transform_list` reads them in.
    """
    axes = mov_image.get_attr('axes')

    # An OME translation on the moving image is normally folded into the
    # alignment rather than written out. Persisting it instead records it as
    # a coordinate transformation on the output group, so a viewer places
    # the warped volume where the original sat.
    global_transformations = []
    if mov_origin_transform is not None and persist_mov_origin_transform:
        spatial_translation = mov_origin_transform[:3, 3].tolist()
        # one leading 0 per non-spatial axis (time, channel), so the
        # translation lines up with the axis list it is attached to
        non_spatial_count = sum(1 for a in (axes or [])
                                if a.get('type') != 'space')
        global_transformations.append({
            'type': 'translation',
            'translation': [0] * non_spatial_count + spatial_translation,
        })

    align_attrs = io_utility.prepare_parent_group_attrs(
        align_path, align_subpath,
        axes=axes,
        dataset_transformations=fix_image.get_attr('coordinateTransformations'),
        global_transformations=global_transformations,
        zarr_format=zarr_format,
    )
    align_shape = fix_image.shape
    if len(align_chunksize) < len(align_shape):
        # a spatial-only chunk shape against a t/c/z/y/x output: one chunk
        # per non-spatial position
        align_chunk_size = ((1,) * (len(align_shape) - len(align_chunksize))
                            + tuple(get_spatial_values(align_chunksize)))
    else:
        align_chunk_size = tuple(get_spatial_values(align_chunksize))
    align_shard_size = derive_shard_shape(sharding_factor, align_chunk_size,
                                          zarr_format)
    align = io_utility.create_dataset_array(
        align_path, align_subpath, align_shape, align_chunk_size,
        fix_image.dtype,
        overwrite=False,
        compressor=compressor,
        compression_opts=compressor_opts,
        for_timeindex=align_timeindex,
        for_channel=align_channel,
        parent_attrs=align_attrs,
        pixelResolution=calc_full_voxel_resolution_attr(
            mov_image.voxel_spacing, mov_image.voxel_downsampling),
        downsamplingFactors=calc_downsampling_attr(mov_image.voxel_downsampling),
        zarr_format=zarr_format,
        shard_shape=align_shard_size,
    )
    # unlike the alignment writes, which are locked per write unit, the warp
    # writes each block exactly once and they do not overlap - so a whole
    # shard per worker is what keeps two workers out of one shard object
    align_processing_size = getattr(align, 'shards', None) or align_chunk_size

    deform_transforms = [deformfield] if deformfield is not None else []
    transform_list = list(static_transforms) + deform_transforms
    # the static transforms carry their own spacings (a global deform may
    # have been produced at a different scale); this run's field is on the
    # current fixed grid. An affine has None.
    fix_deform_spacing = (get_spatial_values(fix_image.voxel_spacing)
                          / fix_image.expansion_factor)
    transforms_spacings = (tuple(static_transforms_spacings)
                           + tuple(fix_deform_spacing
                                   for _ in deform_transforms))

    logger.info(f'Apply {len(static_transforms)} static transform(s) and '
                f'{len(deform_transforms)} local deform to warp {mov_image} '
                f'-> {align_path}:{align_subpath} in '
                f'{align_processing_size} blocks, spacings '
                f'{transforms_spacings}, map_coordinates args '
                f'{transform_coords_args}')
    distributed_apply_transform(
        fix_image,
        np.array(get_spatial_values(fix_image.voxel_spacing)) / fix_image.expansion_factor,
        mov_image,
        np.array(get_spatial_values(mov_image.voxel_spacing)) / mov_image.expansion_factor,
        align_processing_size,
        transform_list,
        cluster_client,
        overlap_factor=transform_overlap_factor,
        aligned_data=align,
        aligned_data_timeindex=align_timeindex,
        aligned_data_channel=align_channel,
        transform_spacing=transforms_spacings,
        **transform_coords_args,
    )
    logger.info(f'Wrote the aligned volume to {align_path}:{align_subpath}')
    return align


def _deformfield_factory(fix_image, roi, alignment_passes, chunksize,
                         compressor, compressor_opts, zarr_format,
                         sharding_factor, foreground_percentage,
                         rebalance_for_missing_neighbors,
                         default_blend_ramp=None,
                         default_processing_size=None,
                         default_halo_factor=None):
    """
    Build the maker for a displacement field array on the fixed image grid.

    The per-pass fields and the composed output all have the same shape,
    chunking and metadata, so they are all created through this.
    """
    downsampling = tuple(get_spatial_values(fix_image.voxel_downsampling)) + (1,)
    voxel_spacing = tuple(get_spatial_values(fix_image.voxel_spacing)) + (1,)

    axes = get_spatial_values(fix_image.get_attr('axes'))
    if axes is not None:
        axes = list(axes) + [{'name': 'd', 'type': 'displacement',
                              'discrete': True}]
    coord_transforms = fix_image.get_attr('coordinateTransformations')
    if coord_transforms is not None:
        coord_transforms = [
            {'type': ct['type'],
             ct['type']: get_spatial_values(ct[ct['type']]) + [ct[ct['type']][1]]}
            for ct in coord_transforms
        ]

    spatial_chunks = tuple(get_spatial_values(chunksize))
    output_chunks = spatial_chunks + (len(spatial_chunks),)
    # the factor applies to the spatial axes only; the vector axis is never sharded
    spatial_shard = derive_shard_shape(sharding_factor, spatial_chunks, zarr_format)
    output_shards = (tuple(spatial_shard) + (output_chunks[-1],)
                     if spatial_shard is not None else None)

    # Enough to reproduce this field: every pass's geometry and every step's
    # full argument dict. Recording only the step *names* would say which
    # algorithms ran but not how they were configured, which is the thing
    # worth keeping - the arguments are what a later run has to match.
    #
    # The geometry recorded is the **resolved** value each pass actually ran
    # with, not what the config literally said: a pass that inherits a top
    # level default would otherwise record a null, which reads as "unset"
    # rather than as the number that was used.
    resolved_passes = [
        p.resolved(fix_image.spatial_ndim,
                   default_processing_size=default_processing_size,
                   default_halo_factor=default_halo_factor,
                   default_blend_ramp=default_blend_ramp)
        for p in alignment_passes
    ]
    passes_description = [
        {'name': p.name,
         'processing_size': p.processing_size,
         'processing_offset': p.processing_offset,
         # the factor is kept alongside the voxel count for provenance; it is
         # None when the halo was configured directly
         'processing_halo_factor': p.processing_halo_factor,
         'processing_halo': p.processing_halo,
         'blend_ramp': p.blend_ramp,
         # single-key mappings, the same shape the config's `alignment_steps`
         # uses, so this reads back as a config fragment
         'steps': [{name: args} for name, args in p.alignment_steps]}
        for p in resolved_passes
    ]

    def create(container_path, subpath, shape):
        attrs = io_utility.prepare_parent_group_attrs(
            container_path, subpath,
            axes=axes,
            dataset_transformations=coord_transforms,
            zarr_format=zarr_format,
            alignment_passes=passes_description,
            roi=roi,
            voxel_scaling=list(fix_image.voxel_spacing),
            volume_expansion=fix_image.expansion_factor,
            foreground_percentage=foreground_percentage,
            rebalance_for_missing_neighbors=rebalance_for_missing_neighbors,
        )
        return io_utility.create_dataset_array(
            container_path, subpath, shape, output_chunks, np.float32,
            overwrite=True,
            compressor=compressor,
            compression_opts=compressor_opts,
            parent_attrs=attrs,
            pixelResolution=calc_full_voxel_resolution_attr(voxel_spacing,
                                                            downsampling),
            downsamplingFactors=calc_downsampling_attr(downsampling),
            zarr_format=zarr_format,
            shard_shape=output_shards,
        )

    return create


def main():
    local_descriptor = CliArgsHelper('local')
    args_parser = _define_args(local_descriptor)
    args = args_parser.parse_args()

    global logger
    logger = configure_logging(args.logging_config, args.verbose)

    _reject_unsupported_args(args_parser, args, local_descriptor)
    logger.info(f'Local alignment: {args}')

    reg_inputs = extract_registration_input_args(args, local_descriptor)

    # the spacing matters only to the warp stage - the block machinery
    # derives a static field's spacing from its shape relative to the fixed
    # image - but it has to be carried in step with the transform itself
    initial_transform, initial_transform_spacing = get_transform(
        args.initial_transform, args.initial_transform_subpath,
        expansion_factor=reg_inputs.fix_expansion_factor,
    )
    output_zarr_format = get_zarr_format(reg_inputs.transform_path(),
                                         args.output_zarr_format)
    ok = _run_local_alignment(
        reg_inputs,
        args.align_config,
        initial_transform,
        initial_transform_spacing=initial_transform_spacing,
        transform_overlap_factor=args.transform_overlap_factor,
        pass_fields_path=args.pass_fields_path,
        resume_from_pass=args.resume_from_pass,
        max_write_locks=args.max_write_locks,
        dask_scheduler_address=args.dask_scheduler,
        dask_config_file=args.dask_config,
        dask_workers=args.local_dask_workers,
        worker_cpus=args.worker_cpus,
        worker_threads_per_cpu=args.max_worker_threads_per_cpu,
        logging_config=args.logging_config,
        compressor=args.compressor,
        compressor_opts=args.compressor_opts,
        zarr_format=output_zarr_format,
        sharding_factor=args.output_sharding_factor,
        verbose=args.verbose,
        max_concurrent_zarr_reads=args.max_concurrent_zarr_reads,
        max_cluster_jobs=args.max_cluster_jobs,
        displacement_diagnostics=args.displacement_diagnostics,
        error_if_displacement_check_fails=args.error_if_displacement_check_fails,
    )
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
