import argparse
import logging
import numpy as np
import os
import bigstream.io_utility as io_utility
import bigstream.utility as ut

from dask.distributed import (Client, LocalCluster)

from bigstream.configure_bigstream import (configure_logging)
from bigstream.configure_dask import (ConfigureWorkerPlugin,
                                      load_dask_config)
from bigstream.distributed_align import distributed_alignment_pipeline
from bigstream.distributed_transform import (distributed_apply_transform,
        distributed_invert_displacement_vector_field)
from bigstream.image_data import (ImageData,
                                  calc_full_voxel_resolution_attr,
                                  calc_downsampling_attr)
from bigstream.ome_utils import (get_spatial_values, compose_origin_transform)

from .cli import (CliArgsHelper, RegistrationInputs,
                  define_registration_input_args, get_algorithm_parameters,
                  extract_registration_input_args, get_input_images,
                  dictfromjson, inttuple, floattuple)

from .utils import get_processing_size, derive_shard_shape


logger:logging.Logger


def _define_args(local_descriptor):
    args_parser = argparse.ArgumentParser(description='Registration pipeline')

    define_registration_input_args(
        args_parser.add_argument_group(
            description='Local registration input volumes'),
        local_descriptor,
    )

    args_parser.add_argument('--align-config',
                             dest='align_config',
                             help='Align config file that contains registration algorithm parameters for fine tune up')
    args_parser.add_argument('--global-affine-transform',
                             dest='global_affine',
                             help='Global affine transform path')
    args_parser.add_argument('--local-processing-size',
                             dest='local_processing_size',
                             type=inttuple,
                             help='partition size for splitting the work')
    args_parser.add_argument('--local-processing-overlap-factor',
                             dest='local_processing_overlap_factor',
                             type=float,
                             help='partition overlap when splitting the work - a fractional number between 0 - 1')
    args_parser.add_argument('--local-transform-overlap-factor',
                             dest='local_transform_overlap_factor',
                             type=float,
                             help='partition overlap when splitting the work for applying a transformation - a fractional number between 0 - 1')
    args_parser.add_argument('--foreground-percentage', '--foreground_percentage',
                             dest='foreground_percentage',
                             default=0.0,
                             type=float,
                             help='Signal percentage per block in order for the block to be considered')

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
                             type=int, default=0,
                             help='Number of cpus allocated to a dask worker')
    args_parser.add_argument('--max-cluster-jobs', '--max_cluster_jobs',
                             dest='max_cluster_jobs',
                             type=int, default=0,
                             help='Maximum number of cluster jobs executed in parallel',
                            )

    args_parser.add_argument('--max-concurrent-zarr-reads',
                             dest='max_concurrent_zarr_reads',
                             type=int, default=0,
                             help='Maximum number of concurrent reads from a zarr array')
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
                                  'e.g. 8,8,4. Applied to each output chunk '
                                  'shape (deformfield, inv-deformfield, '
                                  'aligned volume) to derive its absolute '
                                  'shard shape. When sharding is enabled and '
                                  '--local-processing-size is not set, the '
                                  'processing block size defaults to the '
                                  'shard spatial shape. Ignored when '
                                  '--output-zarr-format is not 3.')

    args_parser.add_argument('--logging-config', dest='logging_config',
                             type=str,
                             help='Logging configuration')
    args_parser.add_argument('--verbose',
                             dest='verbose',
                             action='store_true',
                             help='Set logging level to verbose')

    return args_parser


def _run_local_alignment(reg_args: RegistrationInputs,
                         align_config,
                         global_affine,
                         processing_size=None,
                         processing_overlap=None,
                         transform_overlap=0.1,
                         default_overlap=0.5,
                         inv_step=1.0,
                         inv_iterations=(10,),
                         inv_shrink_spacings=(None,),
                         inv_smooth_sigmas=(0.,),
                         inv_step_cut_factor=0.5,
                         inv_pad:float=0.1,
                         inv_use_root:bool=True,
                         dask_scheduler_address:str|None=None,
                         dask_config_file:str|None=None,
                         dask_workers:int|None=None,
                         worker_cpus=0,
                         logging_config:str|None=None,
                         compressor:str|None=None,
                         compressor_opts:dict={},
                         zarr_format:int=2,
                         sharding_factor=None,
                         verbose=False,
                         foreground_percentage=0,
                         max_concurrent_zarr_reads=0,
                         max_cluster_jobs=0,
                         ):
    local_steps, local_config = get_algorithm_parameters(align_config,
                                                         'local_align',
                                                         reg_args.registration_steps)
    if len(local_steps) == 0:
        logger.info('Skip local alignment because no local steps were specified.')
        return None

    logger.info(f'Run local registration with: {reg_args}, {local_steps}')

    (fix_image, fix_mask, mov_image, mov_mask) = get_input_images(reg_args)
    if mov_image.ndim != fix_image.ndim:
        # only check for ndim and not shape because as it happens 
        # the test data has different shape for fix.highres and mov.highres
        raise Exception(f'{mov_image} expected to have ',
                        f'the same ndim as {fix_image}')

    # Compute storage unit once; used both for default and alignment check below
    output_blocksize_zyx = reg_args.output_blocksize[::-1]
    shard_shape_zyx = derive_shard_shape(sharding_factor, output_blocksize_zyx, zarr_format)

    if processing_size:
        # block are defined as x,y,z so I am reversing it to z,y,x
        local_processing_size = processing_size[::-1]
        logger.info(f'Set processing size to {local_processing_size} (from process size arg: {processing_size})')
    else:
        # when sharding is on, default the processing size to the shard shape
        # so each worker processes one shard
        default_processing_size = shard_shape_zyx if shard_shape_zyx is not None else output_blocksize_zyx
        local_processing_size = local_config.get('block_size', default_processing_size)
        logger.info((
            f'Set processing size to {local_processing_size} '
            f'from blocksize: {output_blocksize_zyx} and sharding factor: {sharding_factor} '
        ))

    # Round up to storage boundary (shard for zarr3, blocksize for zarr2)
    local_processing_size = get_processing_size(
        local_processing_size,
        shard_shape=shard_shape_zyx,
        blocksize=output_blocksize_zyx if shard_shape_zyx is None else None,
    )

    if processing_overlap:
        local_processing_overlap_factor = processing_overlap
    else:
        local_processing_overlap_factor = local_config.get('block_overlap', default_overlap)
    if (local_processing_overlap_factor <= 0 and
        local_processing_overlap_factor >= 1):
        raise ValueError((
            'Invalid block overlap value '
            f'{local_processing_overlap_factor} '
            'must be greater than 0 and less than 1 '
        ))

    if transform_overlap:
        local_transform_overlap_factor = transform_overlap
    else:
        local_transform_overlap_factor = local_config.get('transform_overlap', 0.125)

    apply_deform_steps, _ = get_algorithm_parameters(align_config,
                                                     'apply_deform',
                                                     ['map_coordinates'])
    transform_coords_args = {}
    for step, step_args in apply_deform_steps:
        if step == 'map_coordinates':
            transform_coords_args.update(step_args)

    if reg_args.transform_subpath:
        deformfield_subpath = reg_args.transform_subpath
    else:
        deformfield_subpath = reg_args.mov_subpath

    if reg_args.transform_blocksize:
        # block chunks are define as x,y,z so I am reversing it to z,y,x
        deformfield_chunksize = reg_args.transform_blocksize[::-1]
    else:
        # default to processing
        deformfield_chunksize = reg_args.output_blocksize[::-1]

    if reg_args.inv_transform_subpath:
        inv_deformfield_subpath = reg_args.inv_transform_subpath
    else:
        inv_deformfield_subpath = deformfield_subpath

    if reg_args.inv_transform_blocksize:
        # block chunks are define as x,y,z so I am reversing it to z,y,x
        inv_deformfield_chunksize = reg_args.inv_transform_blocksize[::-1]
    else:
        # default to output_chunk_size
        inv_deformfield_chunksize = deformfield_chunksize

    align_subpath = reg_args.align_dataset()

    if reg_args.align_blocksize:
        # block chunks are define as x,y,z so I am reversing it to z,y,x
        align_chunksize = reg_args.align_blocksize[::-1]
    else:
        # default to output_chunk_size
        align_chunksize = deformfield_chunksize

    # start a dask client
    load_dask_config(dask_config_file)

    if dask_scheduler_address:
        logger.info(f'Use dask scheduler at: {dask_scheduler_address}')
        cluster_client = Client(address=dask_scheduler_address)
    else:
        logger.info(f'Use a local dask with {dask_workers} local workers')
        cluster_client = Client(LocalCluster(n_workers=dask_workers,
                                             threads_per_worker=worker_cpus))
    # create worker plugin
    worker_config = ConfigureWorkerPlugin(logging_config,
                                          verbose,
                                          worker_cpus=worker_cpus)
    cluster_client.register_plugin(worker_config, name='WorkerConfig')
    try:
        if global_affine is not None:
            static_transforms = reg_args.get_static_transforms() + [ global_affine, ]
        else:
            static_transforms = reg_args.get_static_transforms()
        # compose mov origin transform from user affine + OME translations
        mov_origin_transform = compose_origin_transform(
            reg_args.get_mov_origin_transform(),
            mov_image.get_attr('globalCoordinateTransformations'),
        )
        _align_local_data(
            fix_image,
            fix_mask,
            mov_image,
            mov_mask,
            local_steps,
            local_processing_size,
            local_processing_overlap_factor,
            mov_origin_transform,
            static_transforms,
            reg_args.persist_mov_origin_transform,
            reg_args.transform_path(),
            deformfield_subpath,
            deformfield_chunksize,
            reg_args.inv_transform_path(),
            inv_deformfield_subpath,
            inv_deformfield_chunksize,
            reg_args.align_path(),
            align_subpath,
            reg_args.align_timeindex,
            reg_args.align_channel,
            align_chunksize,
            local_transform_overlap_factor,
            transform_coords_args,
            inv_step,
            inv_iterations,
            inv_shrink_spacings,
            inv_smooth_sigmas,
            inv_step_cut_factor,
            inv_pad,
            inv_use_root,
            cluster_client,
            compressor,
            compressor_opts,
            zarr_format,
            sharding_factor,
            foreground_percentage,
            max_concurrent_zarr_reads,
            max_cluster_jobs,
        )
    finally:
        cluster_client.close()


def _align_local_data(fix_image: ImageData,
                      fix_mask: ImageData|None,
                      mov_image: ImageData,
                      mov_mask: ImageData|None,
                      steps,
                      processing_size,
                      processing_overlap_factor,
                      mov_origin_transform,
                      static_transforms,
                      persist_mov_origin_transform,
                      deformfield_path,
                      deformfield_subpath,
                      deformfield_chunksize,
                      inv_deformfield_path,
                      inv_deformfield_subpath,
                      inv_deformfield_chunksize,
                      align_path,
                      align_subpath,
                      align_timeindex,
                      align_channel,
                      align_chunksize,
                      transform_overlap_factor,
                      transform_coords_args,
                      inv_step,
                      inv_iterations,
                      inv_shrink_spacings,
                      inv_smooth_sigmas,
                      inv_step_cut_factor,
                      inv_pad,
                      inv_use_root,
                      cluster_client,
                      compressor,
                      compressor_opts,
                      zarr_format,
                      sharding_factor,
                      foreground_percentage,
                      max_concurrent_zarr_reads,
                      max_cluster_jobs):
    logger.info(f'Align moving data {mov_image} to reference {fix_image} ' +
                f'using {ut.get_number_of_cores()} cpus')

    transform_downsampling = tuple(get_spatial_values(fix_image.voxel_downsampling)) + (1,)
    logger.info(f'Transform downsampling: {transform_downsampling}')
    transform_voxel_spacing = tuple(get_spatial_values(fix_image.voxel_spacing)) + (1,)
    logger.info(f'Transform voxel spacing: {transform_voxel_spacing}')
    deformfield_shape = tuple(fix_image.spatial_dims) + (3,)
    logger.info(f'Transform shape: {fix_image.spatial_dims} => {deformfield_shape}')

    if deformfield_path:
        # transform shape
        deformfield_axes = get_spatial_values(fix_image.get_attr('axes'))
        if deformfield_axes is not None:
            deformfield_axes.append({
                'name': 'd',
                'type': 'displacement',
                'discrete': True,
            })
        coordinate_transformations = fix_image.get_attr('coordinateTransformations')
        if coordinate_transformations is not None:
            new_transforms = []
            for ct in coordinate_transformations:
                cttype = ct['type']
                tx = ct[cttype]
                chtx = tx[1]
                new_transforms.append({
                    'type': cttype,
                    ct['type']: get_spatial_values(tx) + [chtx],
                })
            coordinate_transformations = new_transforms
        deformfield_attrs = io_utility.prepare_parent_group_attrs(
            deformfield_path,
            deformfield_subpath,
            axes=deformfield_axes,
            dataset_transformations=coordinate_transformations,
            zarr_format=zarr_format,
        )
        deformfield_output_chunksize = tuple(get_spatial_values(deformfield_chunksize)) + (1,)
        # factor applies to spatial axes only; vector axis is never sharded
        deformfield_spatial_shard = derive_shard_shape(
            sharding_factor,
            tuple(get_spatial_values(deformfield_chunksize)),
            zarr_format,
        )
        if deformfield_spatial_shard is not None:
            deformfield_output_shardsize = tuple(deformfield_spatial_shard) + (1,)
        else:
            deformfield_output_shardsize = None
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
    else:
        deformfield = None
        deformfield_axes = None

    logger.info((
        f'Calculate transformation {deformfield_path} '
        f'for the local alignment of {mov_image} '
        f'to {fix_image} '
    ))
    if fix_image.has_data() and mov_image.has_data():
        deform_ok = distributed_alignment_pipeline(
            fix_image,
            np.array(get_spatial_values(fix_image.voxel_spacing)) / fix_image.expansion_factor,
            mov_image,
            np.array(get_spatial_values(mov_image.voxel_spacing)) / mov_image.expansion_factor,
            steps,
            processing_size, # parallelize on processing size
            cluster_client,
            overlap_factor=processing_overlap_factor,
            fix_mask=fix_mask,
            mov_mask=mov_mask,
            mov_origin_transform=mov_origin_transform,
            static_transform_list=static_transforms,
            output_transform=deformfield,
            foreground_percentage=foreground_percentage,
            max_concurrent_reads=max_concurrent_zarr_reads,
            max_cluster_jobs=max_cluster_jobs,
        )
        logger.info((
            'Finished computing the deformation field '
            f'{deformfield_path} for the local alignment of '
            f'{mov_image} to {fix_image} '
        ))
    else:
        deform_ok = False
        logger.warning('Fix image or moving image has no data or ' +
                       'the distributed alignment failed')

    if deform_ok and deformfield and inv_deformfield_path:
        logger.info(f'Create inverse deform field container {inv_deformfield_path}')
        if len(inv_iterations) == 0:
            raise ValueError(f'Invalid inverse iterations: {inv_iterations}')
        
        if (len(inv_iterations) != len(inv_shrink_spacings) and
            len(inv_iterations) != len(inv_smooth_sigmas)):
            raise ValueError((
                'Inverse iterations, inverse shrink spacings '
                'and inverse smooth sigmas must all have the same length '
                f'{inv_iterations} vs {inv_shrink_spacings} vs {inv_smooth_sigmas} '
            ))

        inv_deformfield_attrs = io_utility.prepare_parent_group_attrs(
            inv_deformfield_path,
            inv_deformfield_subpath,
            axes=deformfield_axes,
            dataset_transformations=coordinate_transformations,
            zarr_format=zarr_format,
        )
        inv_deformfield_output_chunksize = tuple(get_spatial_values(deformfield_chunksize)) + (1,)
        inv_deformfield_spatial_shard = derive_shard_shape(
            sharding_factor,
            tuple(get_spatial_values(deformfield_chunksize)),
            zarr_format,
        )
        if inv_deformfield_spatial_shard is not None:
            inv_deformfield_output_shardsize = tuple(inv_deformfield_spatial_shard) + (1,)
        else:
            inv_deformfield_output_shardsize = None
        inv_deformfield = io_utility.create_dataset_array(
            inv_deformfield_path,
            inv_deformfield_subpath,
            deformfield_shape,
            inv_deformfield_output_chunksize,
            np.float32,
            overwrite=True,
            compressor=compressor,
            compression_opts=compressor_opts,
            parent_attrs=inv_deformfield_attrs,
            pixelResolution=calc_full_voxel_resolution_attr(transform_voxel_spacing,
                                                            transform_downsampling),
            downsamplingFactors=calc_downsampling_attr(transform_downsampling),
            zarr_format=zarr_format,
            shard_shape=inv_deformfield_output_shardsize,
        )

        deform_field_spacing = get_spatial_values(fix_image.voxel_spacing)

        logger.info((
            'Calculate inverse transformation '
            f'{inv_deformfield_path}:{inv_deformfield_subpath} '
            f'from {deformfield_path}:{deformfield_subpath} '
            f'for local alignment of {mov_image} '
            f'to reference {fix_image} '
            f'deform field spacing is {deform_field_spacing}, expansion factor {fix_image.expansion_factor} '
        ))
        distributed_invert_displacement_vector_field(
            deformfield,
            deform_field_spacing / fix_image.expansion_factor,
            inv_deformfield_chunksize, # use blocksize for partitioning the work
            inv_deformfield,
            cluster_client,
            overlap_factor=transform_overlap_factor,
            step=inv_step,
            iterations=inv_iterations,
            shrink_spacings=inv_shrink_spacings,
            smooth_sigmas=inv_smooth_sigmas,
            step_cut_factor=inv_step_cut_factor,
            pad=inv_pad,
            use_root=inv_use_root,
        )
        del inv_deformfield
    else:
        if not inv_deformfield_path:
            logger.info('Skip the inverse because it is not set')

    if (deform_ok or len(static_transforms) > 0) and align_path:
        axes = mov_image.get_attr('axes')
        # prepare global coordinate transform from mov_origin_transform
        global_transformations = []
        if mov_origin_transform is not None and persist_mov_origin_transform:
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

        # Apply local transformation only if 
        # highres aligned output name is set
        align_attrs = io_utility.prepare_parent_group_attrs(
            align_path,
            align_subpath,
            axes=axes,
            dataset_transformations=mov_image.get_attr('coordinateTransformations'),
            global_transformations=global_transformations,
            zarr_format=zarr_format,
        )
        align_shape = fix_image.shape
        if len(align_chunksize) < len(align_shape):
            # align_blocksize is not set, so use default block size
            align_chunk_size = (1,) * (len(align_shape)-len(align_chunksize)) + tuple(get_spatial_values(align_chunksize))
        else:
            align_chunk_size = tuple(get_spatial_values(align_chunksize))
        align_shard_size = derive_shard_shape(
            sharding_factor, align_chunk_size, zarr_format
        )
        align = io_utility.create_dataset_array(
            align_path,
            align_subpath,
            align_shape,
            align_chunk_size,
            fix_image.dtype,
            overwrite=False,
            compressor=compressor,
            compression_opts=compressor_opts,
            for_timeindex=align_timeindex,
            for_channel=align_channel,
            parent_attrs=align_attrs,
            pixelResolution=calc_full_voxel_resolution_attr(mov_image.voxel_spacing,
                                                            mov_image.voxel_downsampling),
            downsamplingFactors=calc_downsampling_attr(mov_image.voxel_downsampling),
            zarr_format=zarr_format,
            shard_shape=align_shard_size,
        )
        logger.info(f'Apply static transforms {static_transforms}' +
                    f'and local transform {deformfield_path}:{deformfield_subpath}' +
                    f'to warp {mov_image} -> {align_path}:{align_subpath}')
        if deform_ok:
            deform_transforms = [deformfield]
        else:
            deform_transforms = []
        affine_spacings = [None for i in range(len(static_transforms))]
        transforms_spacings = tuple(affine_spacings + [get_spatial_values(fix_image.voxel_spacing) / fix_image.expansion_factor])
        logger.info(f'Transforms spacings: {transforms_spacings}, transform map coordinates args: {transform_coords_args}')

        distributed_apply_transform(
            fix_image,
            np.array(get_spatial_values(fix_image.voxel_spacing)) / fix_image.expansion_factor,
            mov_image,
            np.array(get_spatial_values(mov_image.voxel_spacing)) / mov_image.expansion_factor,
            align_chunksize, # use block chunk size for distributing work
            static_transforms + deform_transforms, # transform_list
            cluster_client,
            overlap_factor=transform_overlap_factor,
            aligned_data=align,
            aligned_data_timeindex=align_timeindex,
            aligned_data_channel=align_channel,
            transform_spacing=transforms_spacings,
            **transform_coords_args,
        )
    else:
        align = None
        if not align_path:
            logger.info('Align arg is not set, so no deformation is applied')

    return deformfield, align


def main():
    local_descriptor = CliArgsHelper('local')
    args_parser = _define_args(local_descriptor)
    args = args_parser.parse_args()
    # prepare logging
    global logger
    logger = configure_logging(args.logging_config, args.verbose)

    logger.info(f'Local registration: {args}')

    global_affine = None
    if args.global_affine and os.path.exists(args.global_affine):
        logger.info(f'Read global affine from {args.global_affine}')
        global_affine = np.loadtxt(args.global_affine)

    reg_inputs = extract_registration_input_args(args, local_descriptor)

    inv_shrink_spacings = (args.inv_shrink_spacings 
                            if (args.inv_shrink_spacings is not None and
                                len(args.inv_shrink_spacings) > 0)
                            else (None,) * len(args.inv_iterations))
    _run_local_alignment(
        reg_inputs,
        args.align_config,
        global_affine,
        processing_size=args.local_processing_size,
        processing_overlap=args.local_processing_overlap_factor,
        transform_overlap=args.local_transform_overlap_factor,
        inv_step=args.inv_step,
        inv_iterations=args.inv_iterations,
        inv_shrink_spacings=inv_shrink_spacings,
        inv_smooth_sigmas=args.inv_smooth_sigmas,
        inv_step_cut_factor=args.inv_step_cut_factor,
        inv_pad=args.inv_pad,
        inv_use_root=args.inv_use_root,
        dask_scheduler_address=args.dask_scheduler,
        dask_config_file=args.dask_config,
        dask_workers=args.local_dask_workers,
        worker_cpus=args.worker_cpus,
        logging_config=args.logging_config,
        compressor=args.compressor,
        compressor_opts=args.compressor_opts,
        zarr_format=args.output_zarr_format,
        sharding_factor=args.output_sharding_factor,
        verbose=args.verbose,
        foreground_percentage=args.foreground_percentage,
        max_concurrent_zarr_reads=args.max_concurrent_zarr_reads,
        max_cluster_jobs=args.max_cluster_jobs,
    )


if __name__ == '__main__':
    main()