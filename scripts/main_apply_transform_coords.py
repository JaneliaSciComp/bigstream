import argparse
import numpy as np
import bigstream.io_utility as io_utility

from dask.distributed import (Client, LocalCluster)
from bigstream.cli import (inttuple, floattuple, stringlist)
from bigstream.configure_bigstream import (configure_logging)
from bigstream.distributed_transform import (
    distributed_apply_transform_to_coordinates)
from bigstream.configure_dask import (ConfigureWorkerPlugin,
                                      load_dask_config)


def _define_args():
    args_parser = argparse.ArgumentParser(description='Apply transformation')
    args_parser.add_argument('--input-coords', dest='input_coords',
                             help='Path to input coordinates file')
    args_parser.add_argument('--pixel-resolution',
                             dest='pixel_resolution',
                             metavar='xres,yres,zres',
                             type=floattuple,
                             help='Pixel resolution')
    args_parser.add_argument('--downsampling',
                             dest='downsampling',
                             metavar='xfactor,yfactor,zfactor',
                             type=inttuple,
                             help='Downsampling factors')
    args_parser.add_argument('--input-volume', dest='input_volume',
                             help='Path to input volume')
    args_parser.add_argument('--input-dataset', dest='input_dataset',
                             help='Input volume dataset')

    args_parser.add_argument('--output-coords', dest='output_coords',
                             help='Path to warped coordinates file')

    args_parser.add_argument('--affine-transformations',
                             dest='affine_transformations',
                             type=stringlist,
                             help='Affine transformations')

    args_parser.add_argument('--local-transform',
                             '--vector-field-transform',
                             dest='local_transform',
                             help='Local (vector field) transformation')
    args_parser.add_argument('--local-transform-subpath',
                             '--vector-field-transform-subpath',
                             dest='local_transform_subpath',
                             help='Local transformation dataset to be applied')
    args_parser.add_argument('--processing-blocksize',
                             dest='processing_blocksize',
                             type=inttuple,
                             help='Processing block size')
    args_parser.add_argument('--partition-size',
                             dest='partition_size',
                             default=128,
                             type=int,
                             help='Partition size for splitting the work')

    args_parser.add_argument('--dask-scheduler', dest='dask_scheduler',
                             type=str, default=None,
                             help='Run with distributed scheduler')

    args_parser.add_argument('--dask-config', dest='dask_config',
                             type=str, default=None,
                             help='YAML file containing dask configuration')

    args_parser.add_argument('--worker-cpus', dest='worker_cpus',
                             type=int, default=0,
                             help='Number of cpus allocated to a dask worker')

    args_parser.add_argument('--logging-config', dest='logging_config',
                             type=str,
                             help='Logging configuration')
    args_parser.add_argument('--verbose',
                             dest='verbose',
                             action='store_true',
                             help='Set logging level to verbose')

    return args_parser


def _get_coords_spacing(pixel_resolution, downsampling_factors,
                        input_volume_path, input_dataset):
    if (pixel_resolution is not None and
        downsampling_factors is not None):
        voxel_spacing = (np.array(pixel_resolution) * 
                         np.array(downsampling_factors))
        return voxel_spacing[::-1] # zyx order
    elif (pixel_resolution is not None):
        voxel_spacing = np.array(pixel_resolution)
        return voxel_spacing[::-1] # zyx order

    if input_volume_path is not None:
        volume_attrs = io_utility.read_attributes(
            input_volume_path, input_dataset)
        voxel_spacing = io_utility.get_voxel_spacing(volume_attrs)
        return voxel_spacing[::-1] if voxel_spacing is not None else None
    else:
        logger.info('Not enough information to get voxel spacing')
        return None


def _run_apply_transform(args):

    if not args.input_coords:
        # Nothing to do
        return

    # Read the input coordinates (as x,y,z)
    input_coords = np.float32(np.loadtxt(args.input_coords, delimiter=','))
    # flip them to z,y,x
    zyx_coords = np.empty_like(input_coords)
    zyx_coords[:, 0:3] = input_coords[:, [2, 1, 0]]
    zyx_coords[:, 3:] = input_coords[:, 3:]

    load_dask_config(args.dask_config)
    worker_config = ConfigureWorkerPlugin(args.logging_config,
                                          args.verbose,
                                          worker_cpus=args.worker_cpus)
    if args.dask_scheduler:
        cluster_client = Client(address=args.dask_scheduler)
    else:
        cluster_client = Client(LocalCluster())
    cluster_client.register_plugin(worker_config, name='WorkerConfig')
    # read local deform, but ignore attributes as they are not needed
    local_deform, _ = io_utility.open(args.local_transform,
                                      args.local_transform_subpath)

    if (args.processing_blocksize is not None and
        len(args.processing_blocksize) > 0):
        processing_blocksize = args.processing_blocksize
    else:
        # default to output blocksize
        processing_blocksize = (args.partition_size,) * (local_deform.ndim-1)

    if args.output_coords:
        if args.affine_transformations:
            affine_transforms_list = [np.loadtxt(tfile)
                                      for tfile in args.affine_transformations]
        else:
            affine_transforms_list = []

        voxel_spacing = _get_coords_spacing(args.pixel_resolution,
                                            args.downsampling,
                                            args.input_volume,
                                            args.input_dataset)
        logger.info(f'Voxel spacing for transform coords: {voxel_spacing}')
        warped_zyx_coords = distributed_apply_transform_to_coordinates(
            zyx_coords,
            affine_transforms_list + [local_deform], # transform_list
            processing_blocksize,
            cluster_client,
            coords_spacing=voxel_spacing,
        )
        output_coords = np.empty_like(warped_zyx_coords)
        # flip z,y,x back to x,y,z before writing them to file
        output_coords[:, 0:3] = warped_zyx_coords[:, [2, 1, 0]]
        output_coords[:, 3:] = warped_zyx_coords[:, 3:]
        logger.info(f'Save warped coords to {args.output_coords}')
        np.savetxt(args.output_coords, output_coords, delimiter=',')
        return args.output_coords
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
