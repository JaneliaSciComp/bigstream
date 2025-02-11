import argparse
import numpy as np
import bigstream.io_utility as io_utility

from dask.distributed import (Client, LocalCluster)

from bigstream.cli import (inttuple, floattuple, stringlist)
from bigstream.configure_bigstream import (configure_logging)
from bigstream.configure_dask import (ConfigureWorkerPlugin,
                                      load_dask_config)
from bigstream.distributed_transform import distributed_apply_transform
from bigstream.image_data import ImageData


def _define_args():
    args_parser = argparse.ArgumentParser(description='Apply transformation')
    args_parser.add_argument('--fix', '--fixed',
                             dest='fixed',
                             help='Path to the fixed image')
    args_parser.add_argument('--fix-subpath', '--fixed-subpath',
                             dest='fixed_subpath',
                             help='Fixed image subpath')
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
    args_parser.add_argument('--mov-spacing', '--moving-spacing',
                             dest='moving_spacing',
                             type=floattuple,
                             help='Moving image voxel spacing')

    args_parser.add_argument('--affine-transformations',
                             dest='affine_transformations',
                             type=stringlist,
                             help='Affine transformations')

    args_parser.add_argument('--local-transform', dest='local_transform',
                             help='Local (vector field) transformation')
    args_parser.add_argument('--local-transform-subpath',
                             dest='local_transform_subpath',
                             help='Local transformation dataset to be applied')
    args_parser.add_argument('--transform-spacing',
                             dest='transform_spacing',
                             type=floattuple,
                             help='Transform spacing')

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
    args_parser.add_argument('--blocks-overlap-factor',
                             dest='blocks_overlap_factor',
                             default=0.5,
                             type=float,
                             help='partition overlap when splitting the work - a fractional number between 0 - 1')

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


def _run_apply_transform(args):

    # Read the datasets - if the moving dataset is not defined it defaults to the fixed dataset
    fix_subpath = args.fixed_subpath
    mov_subpath = args.moving_subpath if args.moving_subpath else fix_subpath
    output_subpath = args.output_subpath if args.output_subpath else mov_subpath

    fix_data = ImageData(args.fixed, fix_subpath)
    mov_data = ImageData(args.moving, mov_subpath)
    if args.fixed_spacing:
        fix_data.voxel_spacing = np.array(args.fixed_spacing)[::-1] # xyz -> zyx
    if args.moving_spacing:
        mov_data.voxel_spacing = np.array(args.moving_spacing)[::-1] # xyz -> zyx
    elif args.fixed_spacing:
        mov_data.voxel_spacing = fix_data.voxel_spacing

    logger.info(f'Fixed volume: {fix_data}')
    logger.info(f'Moving volume: {mov_data}')

    local_deform_data = ImageData(args.local_transform, args.local_transform_subpath)
    if args.transform_spacing:
        local_deform_data.voxel_spacing = np.array((1,) + args.transform_spacing)[::-1]

    if local_deform_data.voxel_spacing is None:
        local_deform_data.voxel_spacing = np.array((1,) + tuple(fix_data.voxel_spacing[::-1]))[::-1]
    # deform spacing axes are: v,x,y,z in reverse so z,y,x,v

    logger.info(f'Local transform spacing for {local_deform_data} -> ' +
                f'{local_deform_data.voxel_spacing}')

    transform_spacing = ()
    if (args.output_blocksize is not None and
        len(args.output_blocksize) > 0):
        output_blocks = args.output_blocksize[::-1] # make it zyx
    else:
        # default to output_chunk_size
        output_blocks = (args.output_chunk_size,) * fix_data.ndim

    if args.output:
        output_dataarray = io_utility.create_dataset(
            args.output,
            output_subpath,
            fix_data.shape,
            output_blocks,
            fix_data.dtype,
            compressor=args.compression,
            pixelResolution=mov_data.get_attr('pixelResolution'),
            downsamplingFactors=mov_data.get_attr('downsamplingFactors'),
        )

        if args.affine_transformations:
            logger.info(f'Affine transformations arg: {args.affine_transformations}')
            applied_affines = [args.affine_transformations]
            affine_transforms_list = [np.loadtxt(tfile)
                                      for tfile in args.affine_transformations]
            affine_spacing = (1.,) * mov_data.ndim
            transform_spacing = transform_spacing + (affine_spacing,)
        else:
            applied_affines = []
            affine_transforms_list = []

        logger.info(f'Check if {local_deform_data} has data')
        if local_deform_data.has_data():
            logger.info(f'Read image for {local_deform_data}')
            local_deform_data.read_image()
            all_transforms = affine_transforms_list + [local_deform_data.image_array]
            applied_transforms = applied_affines + [f'{local_deform_data}']
            transform_spacing = transform_spacing + (local_deform_data.voxel_spacing[:local_deform_data.ndim-1],)
        else:
            all_transforms = affine_transforms_list
            applied_transforms = applied_affines

        logger.info(f'Apply {applied_transforms} to ' +
                    f'{mov_data} -> {args.output}:{output_subpath} ' +
                    f'transform spacing: {transform_spacing}')

        # open a dask client
        load_dask_config(args.dask_config)
        worker_config = ConfigureWorkerPlugin(args.logging_config,
                                              args.verbose,
                                              worker_cpus=args.worker_cpus)
        if args.dask_scheduler:
            cluster_client = Client(address=args.dask_scheduler)
        else:
            cluster_client = Client(LocalCluster())
        cluster_client.register_plugin(worker_config, name='WorkerConfig')
        try:
            distributed_apply_transform(
                fix_data, mov_data,
                fix_data.voxel_spacing, mov_data.voxel_spacing,
                output_blocks, # use block chunk size for distributing the work
                all_transforms, # transform_list
                cluster_client,
                overlap_factor=args.blocks_overlap_factor,
                aligned_data=output_dataarray,
                transform_spacing=transform_spacing,
            )
        finally:
            cluster_client.close()
        return output_dataarray
    else:
        return None


def main():
    args_parser = _define_args()
    args = args_parser.parse_args()
    # prepare logging
    global logger
    logger = configure_logging(args.logging_config, args.verbose)

    logger.info(f'Invoked transformation: {args}')

    _run_apply_transform(args)


if __name__ == '__main__':
    main()