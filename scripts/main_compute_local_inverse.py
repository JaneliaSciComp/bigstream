import argparse
import numpy as np
import bigstream.io_utility as io_utility

from dask.distributed import (Client, LocalCluster)
from bigstream.cli import (inttuple, floattuple)
from bigstream.configure_logging import (configure_logging)
from bigstream.distributed_transform import (distributed_invert_displacement_vector_field)
from bigstream.image_data import ImageData
from bigstream.workers_config import (ConfigureWorkerLoggingPlugin,
                                      load_dask_config)


logger = None # initialized in main as a result of calling configure_logging


def _define_args():
    args_parser = argparse.ArgumentParser(description='Compute inverse deformation vector')
    args_parser.add_argument('--transform-dir', dest='transform_dir',
                             required=True,
                             help='Transform directory')
    args_parser.add_argument('--transform-name', dest='transform_name',
                             help='Transform container name')
    args_parser.add_argument('--transform-subpath', dest='transform_subpath',
                             help='Transform dataset subpath')
    args_parser.add_argument('--transform-spacing',
                             dest='transform_spacing',
                             type=floattuple,
                             help='Transform spacing')

    args_parser.add_argument('--inv-transform-dir', dest='inv_transform_dir',
                             help='Inverse transform directory')
    args_parser.add_argument('--inv-transform-name', dest='inv_transform_name',
                             help='Inverse transform container name')
    args_parser.add_argument('--inv-transform-subpath',
                             dest='inv_transform_subpath',
                             help='Inverse transform dataset subpath')
    args_parser.add_argument('--inv-transform-blocksize',
                             dest='inv_transform_blocksize',
                             type=inttuple,
                             help='Inverse transform blocksize')

    args_parser.add_argument('--processing-overlap-factor',
                             dest='processing_overlap_factor',
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


def _run_compute_inverse(args):

    transform_path = (f'{args.transform_dir}/{args.transform_name}'
                      if args.transform_name
                      else args.transform_dir)
    deform_field = ImageData(transform_path, args.transform_subpath)
    if args.transform_spacing:
        deform_field.voxel_spacing = np.array((1,) + args.transform_spacing)[::-1]

    inv_transform_dir = (args.inv_transform_dir
                         if args.inv_transform_dir
                         else args.transform_dir)
    inv_transform_path = (f'{inv_transform_dir}/{args.inv_transform_name}'
                          if args.inv_transform_name
                          else inv_transform_dir)

    inv_transform_subpath = (args.inv_transform_subpath
                             if args.inv_transform_subpath
                             else args.transform_subpath)
    
    if (inv_transform_path == transform_path and
        inv_transform_subpath == args.transform_subpath):
        raise ValueError(f'Inverse transform overrides the direct transform')

    deform_blocksize = deform_field.get_attr('blockSize')

    print('!!!!!! DEFORM ATTRS ', deform_field.attrs)
    print('!!!!!! DEFORM BLOCKSIZE ', deform_blocksize)
    print('!!!!!! DEFORM VOXEL SPACING ', deform_field.voxel_spacing)
    print('!!!!!! DEFORM SHAPE ', deform_field.shape)

    inv_transform_blocksize=(args.inv_transform_blocksize[::-1]
                             if args.inv_transform_blocksize
                             else deform_blocksize[0:-1])
    print('!!!!!! INV DEFORM BLOCK ', inv_transform_blocksize, len(inv_transform_blocksize))

    transform_downsampling = (list(deform_field.downsampling))[::-1]
    print('!!!!!! transform downsampling ', transform_downsampling)

    transform_spacing = (list(deform_field.get_downsampled_voxel_resolution(False)))[::-1]
    print('!!!!!! transform spacing ', transform_spacing)


    inv_deform_field = io_utility.create_dataset(
        inv_transform_path,
        inv_transform_subpath,
        deform_field.shape,
        tuple(inv_transform_blocksize) + (len(inv_transform_blocksize),),
        np.float32,
        overwrite=True,
        compressor=args.compression,
        pixelResolution=transform_spacing,
        downsamplingFactors=transform_downsampling,            
    )

    # open a dask client
    load_dask_config(args.dask_config)

    if args.dask_scheduler:
        cluster_client = Client(address=args.dask_scheduler)
    else:
        cluster_client = Client(LocalCluster())

    cluster_client.register_plugin(ConfigureWorkerLoggingPlugin(args.logging_config,
                                                                args.verbose))

    try:
        logger.info('Calculate inverse transformation' +
                    f'{inv_transform_path}:{inv_transform_subpath}' +
                    f'from {transform_path}:{args.transform_subpath}')

        # distributed_invert_displacement_vector_field(
        #     deform_field,
        #     fix_image.voxel_spacing,
        #     inv_transform_blocksize, # use blocksize for partitioning the work
        #     inv_deform_field,
        #     cluster_client,
        #     overlap_factor=args.processing_overlap_factor,
        #     iterations=args.inv_iterations,
        #     sqrt_order=args.inv_order,
        #     sqrt_iterations=args.inv_sqrt_iterations,
        #     max_tasks=args.cluster_max_tasks,
        # )
    finally:
        cluster_client.close()


def main():
    args_parser = _define_args()
    args = args_parser.parse_args()
    # prepare logging
    global logger
    logger = configure_logging(args.logging_config, args.verbose)

    logger.info(f'Invoke inverse: {args}')

    _run_compute_inverse(args)


if __name__ == '__main__':
    main()