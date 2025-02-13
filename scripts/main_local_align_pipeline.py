import argparse
import numpy as np
import bigstream.io_utility as io_utility
import bigstream.utility as ut

from dask.distributed import (Client, LocalCluster)
from os.path import exists

from bigstream.cli import (CliArgsHelper, RegistrationInputs,
                           define_registration_input_args,
                           extract_align_pipeline,
                           extract_registration_input_args,
                           inttuple, floattuple, get_input_images)
from bigstream.configure_bigstream import (configure_logging)
from bigstream.configure_dask import (ConfigureWorkerPlugin,
                                      load_dask_config)
from bigstream.distributed_align import distributed_alignment_pipeline
from bigstream.distributed_transform import (distributed_apply_transform,
        distributed_invert_displacement_vector_field)
from bigstream.image_data import ImageData


logger = None # initialized in main as a result of calling configure_logging


def _define_args(local_descriptor):
    args_parser = argparse.ArgumentParser(description='Registration pipeline')

    define_registration_input_args(
        args_parser.add_argument_group(
            description='Local registration input volumes'),
        local_descriptor,
    )

    args_parser.add_argument('--align-config',
                             dest='align_config',
                             help='Align config file')
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
    args_parser.add_argument('--worker-cpus', dest='worker_cpus',
                             type=int, default=0,
                             help='Number of cpus allocated to a dask worker')

    args_parser.add_argument('--compression', dest='compression',
                             default='gzip',
                             type=str,
                             help='Codec used for zarr arrays. ' +
                             'Valid values are: raw,lz4,gzip,bz2,blosc,zstd')

    args_parser.add_argument('--logging-config', dest='logging_config',
                             type=str,
                             help='Logging configuration')
    args_parser.add_argument('--verbose',
                             dest='verbose',
                             action='store_true',
                             help='Set logging level to verbose')

    return args_parser


def _run_local_alignment(reg_args: RegistrationInputs,
                         align_config, global_affine,
                         processing_size=None,
                         processing_overlap=None,
                         default_blocksize=128,
                         default_overlap=0.5,
                         inv_step=1.0,
                         inv_iterations=(10,),
                         inv_shrink_spacings=(None,),
                         inv_smooth_sigmas=(0.,),
                         inv_step_cut_factor=0.5,
                         inv_pad=0.1,
                         inv_use_root=True,
                         dask_scheduler_address=None,
                         dask_config_file=None,
                         worker_cpus=0,
                         logging_config=None,
                         compressor=None,
                         verbose=False,
                         ):
    local_steps, local_config = extract_align_pipeline(align_config,
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

    if processing_size:
        # block are defined as x,y,z so I am reversing it to z,y,x
        local_processing_size = processing_size[::-1]
    else:
        default_processing_size = (default_blocksize,) * fix_image.ndim
        local_processing_size = local_config.get('block_size',
                                                 default_processing_size)

    if processing_overlap:
        local_processing_overlap_factor = processing_overlap
    else:
        local_processing_overlap_factor = local_config.get('block_overlap',
                                                           default_overlap)
    if (local_processing_overlap_factor <= 0 and
        local_processing_overlap_factor >= 1):
        raise Exception('Invalid block overlap value'
                        f'{local_processing_overlap_factor}',
                        'must be greater than 0 and less than 1')

    if reg_args.transform_subpath:
        transform_subpath = reg_args.transform_subpath
    else:
        transform_subpath = reg_args.mov_subpath

    if reg_args.transform_blocksize:
        # block chunks are define as x,y,z so I am reversing it to z,y,x
        transform_blocksize = reg_args.transform_blocksize[::-1]
    else:
        # default to processing
        transform_blocksize = local_processing_size

    if reg_args.inv_transform_subpath:
        inv_transform_subpath = reg_args.inv_transform_subpath
    else:
        inv_transform_subpath = transform_subpath

    if reg_args.inv_transform_blocksize:
        # block chunks are define as x,y,z so I am reversing it to z,y,x
        inv_transform_blocksize = reg_args.inv_transform_blocksize[::-1]
    else:
        # default to output_chunk_size
        inv_transform_blocksize = transform_blocksize

    align_subpath = reg_args.align_dataset()

    if reg_args.align_blocksize:
        # block chunks are define as x,y,z so I am reversing it to z,y,x
        align_blocksize = reg_args.align_blocksize[::-1]
    else:
        # default to output_chunk_size
        align_blocksize = transform_blocksize

    # start a dask client
    load_dask_config(dask_config_file)

    worker_config = ConfigureWorkerPlugin(logging_config, verbose,
                                          worker_cpus=worker_cpus)
    if dask_scheduler_address:
        cluster_client = Client(address=dask_scheduler_address)
    else:
        cluster_client = Client(LocalCluster())
    cluster_client.register_plugin(worker_config, name='WorkerConfig')
    try:
        _align_local_data(
            fix_image,
            fix_mask,
            mov_image,
            mov_mask,
            local_steps,
            local_processing_size,
            local_processing_overlap_factor,
            [global_affine] if global_affine is not None else [],
            reg_args.transform_path(),
            transform_subpath,
            transform_blocksize,
            reg_args.inv_transform_path(),
            inv_transform_subpath,
            inv_transform_blocksize,
            reg_args.align_path(),
            align_subpath,
            align_blocksize,
            inv_step,
            inv_iterations,
            inv_shrink_spacings,
            inv_smooth_sigmas,
            inv_step_cut_factor,
            inv_pad,
            inv_use_root,
            cluster_client,
            compressor,
        )
    finally:
        cluster_client.close()


def _align_local_data(fix_image: ImageData,
                      fix_mask: ImageData,
                      mov_image: ImageData,
                      mov_mask: ImageData,
                      steps,
                      processing_size,
                      processing_overlap_factor,
                      global_affine_transforms,
                      transform_path,
                      transform_subpath,
                      transform_blocksize,
                      inv_transform_path,
                      inv_transform_subpath,
                      inv_transform_blocksize,
                      align_path,
                      align_subpath,
                      align_blocksize,
                      inv_step,
                      inv_iterations,
                      inv_shrink_spacings,
                      inv_smooth_sigmas,
                      inv_step_cut_factor,
                      inv_pad,
                      inv_use_root,
                      cluster_client,
                      compressor):

    fix_shape = fix_image.shape
    fix_ndim = fix_image.ndim

    logger.info(f'Align moving data {mov_image} to reference {fix_image} ' +
                f'using {ut.get_number_of_cores()} cpus')

    transform_downsampling = (list(fix_image.downsampling) + [1])[::-1]
    transform_spacing = (list(fix_image.get_downsampled_voxel_resolution(False)) + [1])[::-1]
    if transform_path:
        transform = io_utility.create_dataset(
            transform_path,
            transform_subpath,
            fix_shape + (fix_ndim,),
            tuple(transform_blocksize) + (fix_ndim,),
            np.float32,
            overwrite=True,
            compressor=compressor,
            pixelResolution=transform_spacing,
            downsamplingFactors=transform_downsampling, 
        )
    else:
        transform = None
    logger.info(f'Calculate transformation {transform_path}' +
                f'for the local alignment of {mov_image}' +
                f'to {fix_image}')
    if fix_image.has_data() and mov_image.has_data():
        deform_ok = distributed_alignment_pipeline(
            fix_image, mov_image,
            fix_image.voxel_spacing, mov_image.voxel_spacing,
            steps,
            processing_size, # parallelize on processing size
            cluster_client,
            overlap_factor=processing_overlap_factor,
            fix_mask=fix_mask,
            mov_mask=mov_mask,
            static_transform_list=global_affine_transforms,
            output_transform=transform,
        )
        logger.info('Finished computing the deformation field ' +
                    f'{transform_path} for the local alignment of ' +
                    f'{mov_image} to {fix_image}')
    else:
        deform_ok = False
        logger.warning('Fix image or moving image has no data or ' +
                       'the distributed alignment failed')

    if deform_ok and transform and inv_transform_path:
        if len(inv_iterations) == 0:
            raise ValueError(f'Invalid inverse iterations: {inv_iterations}')
        
        if (len(inv_iterations) != len(inv_shrink_spacings) and
            len(inv_iterations) != len(inv_smooth_sigmas)):
            raise ValueError((
                'Inverse iterations, inverse shrink spacings '
                'and inverse smooth sigmas must all have the same length '
                f'{inv_iterations} vs {inv_shrink_spacings} vs {inv_smooth_sigmas} '
            ))

        inv_transform = io_utility.create_dataset(
            inv_transform_path,
            inv_transform_subpath,
            fix_shape + (fix_ndim,),
            tuple(inv_transform_blocksize) + (fix_ndim,),
            np.float32,
            overwrite=True,
            compressor=compressor,
            pixelResolution=transform_spacing,
            downsamplingFactors=transform_downsampling,            
        )
        logger.info('Calculate inverse transformation' +
                    f'{inv_transform_path}:{inv_transform_subpath}' +
                    f'from {transform_path}:{transform_subpath}' +
                    f'for local alignment of {mov_image}' +
                    f'to reference {fix_image}')
        distributed_invert_displacement_vector_field(
            transform,
            fix_image.voxel_spacing,
            inv_transform_blocksize, # use blocksize for partitioning the work
            inv_transform,
            cluster_client,
            overlap_factor=processing_overlap_factor,
            step=inv_step,
            iterations=inv_iterations,
            shrink_spacings=inv_shrink_spacings,
            smooth_sigmas=inv_smooth_sigmas,
            step_cut_factor=inv_step_cut_factor,
            pad=inv_pad,
            use_root=inv_use_root,
        )
    else:
        if not inv_transform_path:
            logger.info('Skip the inverse because it is not set')

    if (deform_ok or len(global_affine_transforms) > 0) and align_path:
        # Apply local transformation only if 
        # highres aligned output name is set
        align = io_utility.create_dataset(
            align_path,
            align_subpath,
            fix_shape,
            align_blocksize,
            fix_image.dtype,
            overwrite=True,
            compressor=compressor,
            pixelResolution=mov_image.get_attr('pixelResolution'),
            downsamplingFactors=mov_image.get_attr('downsamplingFactors'),
        )
        logger.info(f'Apply affine transform {global_affine_transforms}' +
                    f'and local transform {transform_path}:{transform_subpath}' +
                    f'to warp {mov_image} -> {align_path}:{align_subpath}')
        if deform_ok:
            deform_transforms = [transform]
        else:
            deform_transforms = []
        affine_spacings = [(1.,) * mov_image.ndim for i in range(len(global_affine_transforms))]
        transform_spacing = tuple(affine_spacings + [fix_image.voxel_spacing])
        distributed_apply_transform(
            fix_image, mov_image,
            fix_image.voxel_spacing, mov_image.voxel_spacing,
            align_blocksize, # use block chunk size for distributing work
            global_affine_transforms + deform_transforms, # transform_list
            cluster_client,
            overlap_factor=processing_overlap_factor,
            aligned_data=align,
            transform_spacing=transform_spacing,
        )
    else:
        align = None
        if not align_path:
            logger.info('Align arg is not set, so no deformation is applied')

    return transform, align


def main():
    local_descriptor = CliArgsHelper('local')
    args_parser = _define_args(local_descriptor)
    args = args_parser.parse_args()
    # prepare logging
    global logger
    logger = configure_logging(args.logging_config, args.verbose)

    logger.info(f'Local registration: {args}')

    global_affine = None
    if args.global_affine and exists(args.global_affine):
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
        inv_step=args.inv_step,
        inv_iterations=args.inv_iterations,
        inv_shrink_spacings=inv_shrink_spacings,
        inv_smooth_sigmas=args.inv_smooth_sigmas,
        inv_step_cut_factor=args.inv_step_cut_factor,
        inv_pad=args.inv_pad,
        inv_use_root=args.inv_use_root,
        dask_scheduler_address=args.dask_scheduler,
        dask_config_file=args.dask_config,
        worker_cpus=args.worker_cpus,
        logging_config=args.logging_config,
        compressor=args.compression,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    main()