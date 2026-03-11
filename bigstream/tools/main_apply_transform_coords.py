import argparse
import logging
import numpy as np
import bigstream.io_utility as io_utility

from dask.distributed import (Client, LocalCluster)
from bigstream.configure_bigstream import (configure_logging)
from bigstream.distributed_transform import (distributed_apply_transform_to_coordinates)
from bigstream.configure_dask import (ConfigureWorkerPlugin,
                                      load_dask_config)
from bigstream.image_data import ImageData
from bigstream.ome_utils import get_spatial_values

from .cli import (inttuple, floattuple, stringlist)


logger:logging.Logger


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
    args_parser.add_argument('--input-timeindex', '--input_timeindex',
                             dest='input_timeindex',
                             type=int,
                             default=None,
                             help='Input volume time index')
    args_parser.add_argument('--input-channel', '--input_channel',
                             dest='input_channel',
                             type=int,
                             default=None,
                             help='Input volume channel')

    args_parser.add_argument('--output-coords', dest='output_coords',
                             help='Path to warped coordinates file')

    args_parser.add_argument('--static-transforms',
                             dest='static_transforms',
                             type=stringlist,
                             help='Static transformations')

    args_parser.add_argument('--affine-transform', '--affine-transformations',
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
    args_parser.add_argument('--local-transform-spacing', '--transform-spacing',
                             dest='local_transform_spacing',
                             type=floattuple,
                             help='Local transform spacing')

    args_parser.add_argument('--inverse-transforms',
                             dest='inverse_transforms',
                             action='store_true',
                             help='Flag should be true if the transforms are actually inverse and should be applied in reverse order')

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

    args_parser.add_argument('--local-dask-workers', '--local_dask_workers',
                             dest='local_dask_workers',
                             type=int,
                             help='Number of workers when using a local cluster')
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


def _get_coords_spacing(input_volume_path, input_dataset,
                        pixel_resolution, downsampling_factors,
                        ):
    if input_volume_path is not None:
        volume_attrs = io_utility.read_image_container_attributes(
            input_volume_path, input_dataset)
        voxel_spacing = io_utility.get_voxel_spacing(volume_attrs)
        voxel_resolution = get_spatial_values(voxel_spacing)
        if voxel_resolution is not None:
            return voxel_resolution

    if (pixel_resolution is not None and
        downsampling_factors is not None):
        voxel_spacing = (np.array(pixel_resolution) * 
                         np.array(downsampling_factors))
        return voxel_spacing[::-1][:3]  # zyx order
    elif (pixel_resolution is not None):
        voxel_spacing = np.array(pixel_resolution)
        return voxel_spacing[::-1][:3]  # zyx order

    logger.info('Not enough information to get voxel spacing from attributes.')
    return np.array([1.0, 1.0, 1.0])  # default voxel spacing


def _run_apply_transform(args):

    if not args.input_coords:
        # Nothing to do
        return

    # Read the input coordinates (as x,y,z)
    # Handle optional header row (e.g. "x,y,z,...")
    header = None
    with open(args.input_coords, 'r') as f:
        first_line = f.readline().strip()
        try:
            [float(v) for v in first_line.split(',')]
        except ValueError:
            header = first_line
    input_coords = np.float32(np.loadtxt(args.input_coords, delimiter=',',
                                         skiprows=1 if header else 0))
    # flip them to z,y,x
    zyx_coords = np.empty_like(input_coords)
    zyx_coords[:, 0:3] = input_coords[:, [2, 1, 0]]
    zyx_coords[:, 3:] = input_coords[:, 3:]

    load_dask_config(args.dask_config)
    if args.dask_scheduler:
        cluster_client = Client(address=args.dask_scheduler)
    else:
        cluster_client = Client(LocalCluster(n_workers=args.local_dask_workers,
                                             threads_per_worker=args.worker_cpus))
    worker_config = ConfigureWorkerPlugin(args.logging_config,
                                          args.verbose,
                                          worker_cpus=args.worker_cpus)
    cluster_client.register_plugin(worker_config, name='WorkerConfig')

    # get input image spacing
    voxel_spacing = _get_coords_spacing(args.input_volume,
                                        args.input_dataset,
                                        args.pixel_resolution,
                                        args.downsampling)
    spatial_ndim = len(voxel_spacing)

    local_deform_field = ImageData(args.local_transform, args.local_transform_subpath)
    if args.local_transform_spacing:
        # in case the transform spacing arg has the channel dimension - truncate it
        local_deform_spacing = args.local_transform_spacing[::-1][:spatial_ndim]  # xyz -> zyx
    else:
        local_deform_spacing = local_deform_field.voxel_spacing[:spatial_ndim]

    applied_affines = []
    affine_transforms_list = []

    if args.output_coords:
        # read static transformations
        if args.static_transforms:
            logger.info(f'Static transformations arg: {args.static_transforms}')
            applied_affines.append(args.static_transforms)
            affine_transforms_list.extend([np.loadtxt(tfile) for tfile in args.static_transforms])

        # read affine transformations
        if args.affine_transformations:
            logger.info(f'Affine transformations arg: {args.affine_transformations}')
            applied_affines = [args.affine_transformations]
            affine_transforms_list.extend([np.loadtxt(tfile) for tfile in args.affine_transformations])

        if len(affine_transforms_list) > 0:
            # affines should already be in physical space of the fixed image
            affine_spacing = None
            transforms_spacings = (affine_spacing,) * len(affine_transforms_list)
        else:
            transforms_spacings = ()

        logger.info(f'Check if {local_deform_field} has data')
        if local_deform_field.has_data():
            logger.info(f'Read image for {local_deform_field}')
            local_deform_field.read_image(convert_to_little_endian=False)
            all_transforms = affine_transforms_list + [local_deform_field.image_array]
            applied_transforms = applied_affines + [f'{local_deform_field}']
            transforms_spacings = transforms_spacings + (local_deform_spacing,)
        else:
            all_transforms = affine_transforms_list
            applied_transforms = applied_affines

        all_transforms = all_transforms[::-1] if args.inverse_transforms else all_transforms
        applied_transforms = applied_transforms[::-1] if args.inverse_transforms else applied_transforms
        transforms_spacings = transforms_spacings[::-1] if args.inverse_transforms else transforms_spacings

        logger.info((
            f'Apply {applied_transforms} '
            + ('(inversed) ' if args.inverse_transforms else '')
            + f'to {args.input_coords} -> {args.output_coords}, '
            f'transform spacing: {transforms_spacings} '
        ))

        if (args.processing_blocksize is not None and
            len(args.processing_blocksize) > 0):
            processing_blocksize = args.processing_blocksize
        else:
            # default to output blocksize
            processing_blocksize = (args.partition_size,) * 3

        warped_zyx_coords = distributed_apply_transform_to_coordinates(
            zyx_coords,
            all_transforms, # transform_list
            processing_blocksize,
            cluster_client,
            transform_spacing=transforms_spacings,
            coords_spacing=voxel_spacing,
        )
        output_coords = np.empty_like(warped_zyx_coords)
        # flip z,y,x back to x,y,z before writing them to file
        output_coords[:, 0:3] = warped_zyx_coords[:, [2, 1, 0]]
        output_coords[:, 3:] = warped_zyx_coords[:, 3:]

        logger.info(f'Save warped coords to {args.output_coords}')
        if header:
            with open(args.output_coords, 'w') as f:
                f.write(header + '\n')
            with open(args.output_coords, 'ab') as f:
                np.savetxt(f, output_coords, delimiter=',', fmt='%4.4f')
        else:
            np.savetxt(args.output_coords, output_coords, delimiter=',', fmt='%4.4f')

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
