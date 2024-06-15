import argparse
import numpy as np
import bigstream.io_utility as io_utility
import yaml

from os.path import exists
from dask.distributed import (Client, LocalCluster)
from flatten_json import flatten
from bigstream.cli import (CliArgsHelper, define_registration_input_args,
                           extract_align_pipeline,
                           extract_registration_input_args,
                           inttuple, get_input_images)
from bigstream.distributed_align import distributed_alignment_pipeline
from bigstream.distributed_transform import (distributed_apply_transform,
        distributed_invert_displacement_vector_field)
from bigstream.image_data import ImageData
from bigstream.transform import apply_transform


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
                             help='partition overlap when splitting the work - a fractional number between 0 - 1')
    args_parser.add_argument('--local-processing-overlap-factor',
                             dest='local_processing_overlap_factor',
                             type=float,
                             help='partition overlap when splitting the work - a fractional number between 0 - 1')
    args_parser.add_argument('--inv-iterations',
                             dest='inv_iterations',
                             default=10,
                             type=int,
                             help="Number of iterations for getting the inverse transformation")
    args_parser.add_argument('--inv-order',
                             dest='inv_order',
                             default=2,
                             type=int,
                             help="Order value for the inverse transformation")
    args_parser.add_argument('--inv-sqrt-iterations',
                             dest='inv_sqrt_iterations',
                             default=10,
                             type=int,
                             help="Number of square root iterations for getting the inverse transformation")

    args_parser.add_argument('--dask-scheduler', dest='dask_scheduler',
                             type=str, default=None,
                             help='Run with distributed scheduler')

    args_parser.add_argument('--dask-config', dest='dask_config',
                             type=str, default=None,
                             help='YAML file containing dask configuration')
    args_parser.add_argument('--cluster-max-tasks', dest='cluster_max_tasks',
                             type=int, default=0,
                             help='Maximum number of parallel cluster tasks if >= 0')

    return args_parser


def _run_local_alignment(args, align_config, global_affine,
                         processing_size=None,
                         processing_overlap=None,
                         default_blocksize=128,
                         default_overlap=0.5,
                         inv_iterations=10,
                         inv_sqrt_iterations=10,
                         inv_order=2,
                         dask_scheduler_address=None,
                         dask_config_file=None,
                         max_tasks=0,
                         ):
    local_steps, local_config = extract_align_pipeline(align_config,
                                                       'local_align',
                                                       args.registration_steps)
    if len(local_steps) == 0:
        print('Skip local alignment because no local steps were specified.')
        return None

    print('Run local registration with:', args, local_steps, flush=True)

    (fix_image, fix_mask, mov_image, mov_mask) = get_input_images(args)
    if mov_image.ndim != fix_image.ndim:
        # only check for ndim and not shape because as it happens 
        # the test data has different shape for fix.highres and mov.highres
        raise Exception(f'{mov_image} expected to have ',
                        f'the same ndim as {fix_image}')

    if dask_config_file:
        import dask.config
        with open(dask_config_file) as f:
            dask_config = flatten(yaml.safe_load(f))
            dask.config.set(dask_config)

    if dask_scheduler_address:
        cluster_client = Client(address=dask_scheduler_address)
    else:
        cluster_client = Client(LocalCluster())

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

    if args.transform_subpath:
        transform_subpath = args.transform_subpath
    else:
        transform_subpath = args.mov_subpath

    if args.transform_blocksize:
        # block chunks are define as x,y,z so I am reversing it to z,y,x
        transform_blocksize = args.transform_blocksize[::-1]
    else:
        # default to processing
        transform_blocksize = local_processing_size

    if args.inv_transform_subpath:
        inv_transform_subpath = args.inv_transform_subpath
    else:
        inv_transform_subpath = transform_subpath

    if args.inv_transform_blocksize:
        # block chunks are define as x,y,z so I am reversing it to z,y,x
        inv_transform_blocksize = args.inv_transform_blocksize[::-1]
    else:
        # default to output_chunk_size
        inv_transform_blocksize = transform_blocksize

    align_subpath = args.align_dataset()

    if args.align_blocksize:
        # block chunks are define as x,y,z so I am reversing it to z,y,x
        align_blocksize = args.align_blocksize[::-1]
    else:
        # default to output_chunk_size
        align_blocksize = transform_blocksize

    _align_local_data(
        fix_image,
        fix_mask,
        mov_image,
        mov_mask,
        local_steps,
        local_processing_size,
        local_processing_overlap_factor,
        [global_affine] if global_affine is not None else [],
        args.transform_path(),
        transform_subpath,
        transform_blocksize,
        args.inv_transform_path(),
        inv_transform_subpath,
        inv_transform_blocksize,
        args.align_path(),
        align_subpath,
        align_blocksize,
        inv_iterations,
        inv_sqrt_iterations,
        inv_order,
        cluster_client,
        max_tasks,
    )


def _align_local_data(fix_image,
                      fix_mask,
                      mov_image,
                      mov_mask,
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
                      inv_iterations,
                      inv_sqrt_iterations,
                      inv_order,
                      cluster_client,
                      cluster_max_tasks):

    fix_shape = fix_image.shape
    fix_ndim = fix_image.ndim

    print('Align moving data', mov_image, 'to reference', fix_image,
          flush=True)

    transform_downsampling = (list(fix_image.downsampling) + [1])[::-1]
    transform_spacing = (list(fix_image.get_downsampled_voxel_resolution(False)) + [1])[::-1]
    if transform_path:
        transform = io_utility.create_dataset(
            transform_path,
            transform_subpath,
            fix_shape + (fix_ndim,),
            tuple(transform_blocksize) + (fix_ndim,),
            np.float32,
            pixelResolution=transform_spacing,
            downsamplingFactors=transform_downsampling, 
        )
    else:
        transform = None
    print('Calculate transformation', transform_path, 'for local alignment of',
          mov_image, 'to reference', fix_image,
          flush=True)
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
            max_tasks=cluster_max_tasks,
        )
    else:
        deform_ok = False

    if deform_ok and transform and inv_transform_path:
        inv_transform = io_utility.create_dataset(
            inv_transform_path,
            inv_transform_subpath,
            fix_shape + (fix_ndim,),
            tuple(inv_transform_blocksize) + (fix_ndim,),
            np.float32,
            pixelResolution=transform_spacing,
            downsamplingFactors=transform_downsampling,            
        )
        print('Calculate inverse transformation',
              f'{inv_transform_path}:{inv_transform_subpath}',
              'from', 
              f'{transform_path}:{transform_subpath}',
              'for local alignment of',
              f'{mov_image}',
              'to reference',
              f'{fix_image}',
              flush=True)
        distributed_invert_displacement_vector_field(
            transform,
            fix_image.voxel_spacing,
            inv_transform_blocksize, # use blocksize for partitioning the work
            inv_transform,
            cluster_client,
            overlap_factor=processing_overlap_factor,
            iterations=inv_iterations,
            sqrt_order=inv_order,
            sqrt_iterations=inv_sqrt_iterations,
            max_tasks=cluster_max_tasks,
        )

    if (deform_ok or global_affine_transforms) and align_path:
        # Apply local transformation only if 
        # highres aligned output name is set
        align = io_utility.create_dataset(
            align_path,
            align_subpath,
            fix_shape,
            align_blocksize,
            fix_image.dtype,
            pixelResolution=mov_image.get_attr('pixelResolution'),
            downsamplingFactors=mov_image.get_attr('downsamplingFactors'),
        )
        print('Apply', 
              f'{transform_path}:{transform_subpath}',              
              'to warp',
              f'{mov_image}',
              '->',
              f'{align_path}:{align_subpath}',
              flush=True)
        if deform_ok:
            deform_transforms = [transform]
        else:
            deform_transforms = []
        affine_spacing = (1.,) * mov_image.ndim
        transform_spacing = affine_spacing + fix_image.voxel_spacing
        distributed_apply_transform(
            fix_image, mov_image,
            fix_image.voxel_spacing, mov_image.voxel_spacing,
            align_blocksize, # use block chunk size for distributing work
            global_affine_transforms + deform_transforms, # transform_list
            cluster_client,
            overlap_factor=processing_overlap_factor,
            aligned_data=align,
            transform_spacing=transform_spacing,
            max_tasks=cluster_max_tasks,
        )
    else:
        align = None

    return transform, align


def main():
    local_descriptor = CliArgsHelper('local')
    args_parser = _define_args(local_descriptor)
    args = args_parser.parse_args()
    print('Local registration:', args, flush=True)

    global_affine = None
    if args.global_affine and exists(args.global_affine):
        print('Read global affine from', args.global_affine, flush=True)
        global_affine = np.loadtxt(args.global_affine)

    reg_inputs = extract_registration_input_args(args, local_descriptor)

    _run_local_alignment(
        reg_inputs,
        args.align_config,
        global_affine,
        processing_size=args.local_processing_size,
        processing_overlap=args.local_processing_overlap_factor,
        inv_iterations=args.inv_iterations,
        inv_sqrt_iterations=args.inv_sqrt_iterations,
        inv_order=args.inv_order,
        dask_scheduler_address=args.dask_scheduler,
        dask_config_file=args.dask_config,
        max_tasks=args.cluster_max_tasks,
    )


if __name__ == '__main__':
    main()