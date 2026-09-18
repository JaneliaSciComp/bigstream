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
    default_pass_geometry_from_config,
)
from bigstream.image_data import (ImageData,
                                  calc_full_voxel_resolution_attr,
                                  calc_downsampling_attr)
from bigstream.ome_utils import (get_spatial_values, compose_origin_transform)

from .cli import (CliArgsHelper, RegistrationInputs,
                  define_registration_input_args,
                  extract_registration_input_args, get_input_images,
                  get_transform, dictfromjson, inttuple)

from .utils import derive_shard_shape, get_zarr_format


# rebound to the configured root logger by main(); a real logger up
# front so the module's functions can also be called directly
logger = logging.getLogger(__name__)


# Arguments `define_registration_input_args` contributes that this tool does
# not implement. They stay in the parser so the shared definition is not
# forked, but they are hidden from --help and refused if actually passed -
# silently ignoring an output the caller asked for would be worse.
_INVERSE = ('this tool does not compute the inverse field; '
            'use main_compute_local_inverse')
_WARP = ('this tool does not warp the moving image; '
         'use main_apply_local_transform')
_UNSUPPORTED_ARGS = {
    'inv_transform_name': _INVERSE,
    'inv_transform_subpath': _INVERSE,
    'inv_transform_blocksize': _INVERSE,
    'align_dir': _WARP,
    'align_name': _WARP,
    'align_subpath': _WARP,
    'align_timeindex': _WARP,
    'align_channel': _WARP,
    'align_blocksize': _WARP,
    'persist_mov_origin_transform': _WARP,
}


def _define_args(local_descriptor):
    args_parser = argparse.ArgumentParser(
        description='Compute a local deformation field (multi-pass blockwise)')

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

    args_parser.add_argument('--pass-fields-path',
                             dest='pass_fields_path',
                             help='Container for the intermediate per-pass '
                                  'fields (defaults to the transform '
                                  'container). Only used when the config has '
                                  'more than one pass. These are kept after '
                                  'the run - they are what the per-pass '
                                  'diagnostics are read from - and are never '
                                  'deleted automatically.')
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
            'These options are not supported by this tool, which only '
            'computes the deformation field:\n  ' + '\n  '.join(given)
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
                         pass_fields_path=None,
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

    default_size, default_halo_factor, default_consistency = \
        default_pass_geometry_from_config(config)

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

    (fix_image, fix_mask, mov_image, mov_mask, roi, _, _) = get_input_images(reg_args)
    if mov_image.ndim != fix_image.ndim:
        raise ValueError(f'{mov_image} is expected to have the same ndim as '
                         f'{fix_image}')
    if not (fix_image.has_data() and mov_image.has_data()):
        raise ValueError('Either the fixed or the moving image has no data')

    deformfield_path = reg_args.transform_path()
    if not deformfield_path:
        raise SystemExit('No transform output was given; this tool only '
                         'computes the deformation field, so there is '
                         'nothing to do. Set --local-transform-name.')
    deformfield_subpath = reg_args.transform_subpath or reg_args.mov_subpath
    if reg_args.transform_blocksize:
        deformfield_chunksize = tuple(reg_args.transform_blocksize)[::-1]
    else:
        deformfield_chunksize = tuple(reg_args.output_blocksize)[::-1]

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
        static_transforms, _ = reg_args.get_static_transforms()
        if initial_transform is not None:
            static_transforms = static_transforms + [initial_transform,]
        mov_origin_transform = compose_origin_transform(
            reg_args.get_mov_origin_transform(),
            mov_image.get_attr('globalCoordinateTransformations'),
        )
        return _compute_deform_field(
            fix_image, fix_mask, mov_image, mov_mask, roi,
            alignment_passes,
            default_size,
            default_halo_factor,
            default_consistency,
            mov_origin_transform,
            static_transforms,
            deformfield_path,
            deformfield_subpath,
            deformfield_chunksize,
            pass_fields_path or deformfield_path,
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
                          default_neighbor_consistency,
                          mov_origin_transform,
                          static_transforms,
                          deformfield_path,
                          deformfield_subpath,
                          deformfield_chunksize,
                          pass_fields_path,
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
    logger.info(f'Compute the deformation field aligning {mov_image} to '
                f'{fix_image} over {len(alignment_passes)} pass(es)')

    deformfield_shape = (tuple(fix_image.spatial_dims)
                         + (len(fix_image.spatial_dims),))
    create_field = _deformfield_factory(
        fix_image, roi, alignment_passes, deformfield_chunksize,
        compressor, compressor_opts, zarr_format, sharding_factor,
        foreground_percentage, rebalance_for_missing_neighbors,
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

    deform_ok = blockwise_alignment_pipeline(
        fix_image,
        np.array(get_spatial_values(fix_image.voxel_spacing)) / fix_image.expansion_factor,
        mov_image,
        np.array(get_spatial_values(mov_image.voxel_spacing)) / mov_image.expansion_factor,
        alignment_passes,
        cluster_client,
        processing_size=default_processing_size,
        processing_halo_factor=default_halo_factor,
        neighbor_consistency=default_neighbor_consistency,
        fix_mask=fix_mask,
        mov_mask=mov_mask,
        roi=roi,
        foreground_percentage=foreground_percentage,
        mov_origin_transform=mov_origin_transform,
        static_transform_list=static_transforms,
        output_transform=deformfield,
        pass_output_factory=pass_output_factory,
        max_concurrent_reads=max_concurrent_zarr_reads,
        max_cluster_jobs=max_cluster_jobs,
        max_write_locks=max_write_locks,
        rebalance_for_missing_neighbors=rebalance_for_missing_neighbors,
        displacement_diagnostics=displacement_diagnostics,
        error_if_displacement_check_fails=error_if_displacement_check_fails,
    )
    if deform_ok:
        logger.info(f'Wrote the deformation field to '
                    f'{deformfield_path}:{deformfield_subpath}')
    else:
        logger.error('Some blocks failed; the deformation field at '
                     f'{deformfield_path}:{deformfield_subpath} is incomplete')
    return bool(deform_ok)


def _deformfield_factory(fix_image, roi, alignment_passes, chunksize,
                         compressor, compressor_opts, zarr_format,
                         sharding_factor, foreground_percentage,
                         rebalance_for_missing_neighbors):
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

    passes_description = [
        {'name': p.name,
         'processing_size': p.processing_size,
         'processing_offset': p.processing_offset,
         'processing_halo_factor': p.processing_halo_factor,
         'processing_halo': p.processing_halo,
         'steps': [name for name, _ in p.alignment_steps]}
        for p in alignment_passes
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
    logger.info(f'Local deformation field: {args}')

    reg_inputs = extract_registration_input_args(args, local_descriptor)

    # the spacing `get_transform` reports is only needed to *apply* a field;
    # the block machinery derives a static field's spacing from its shape
    # relative to the fixed image, so it is dropped here
    initial_transform, _ = get_transform(
        args.initial_transform, args.initial_transform_subpath,
        expansion_factor=reg_inputs.fix_expansion_factor,
    )
    output_zarr_format = get_zarr_format(reg_inputs.transform_path(),
                                         args.output_zarr_format)
    ok = _run_local_alignment(
        reg_inputs,
        args.align_config,
        initial_transform,
        pass_fields_path=args.pass_fields_path,
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
