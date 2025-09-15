import argparse
import numpy as np
import bigstream.io_utility as io_utility

from dask.distributed import (Client, LocalCluster)

from bigstream.configure_bigstream import (configure_logging)
from bigstream.configure_dask import (ConfigureWorkerPlugin,
                                      load_dask_config)
from bigstream.distributed_transform import distributed_apply_transform
from bigstream.image_data import (ImageData, get_spatial_values,
                                  calc_full_voxel_resolution_attr,
                                  calc_downsampling_attr)

from .cli import (inttuple, floattuple, stringlist)


logger = None


def _define_args():
    args_parser = argparse.ArgumentParser(description='Apply transformation')
    args_parser.add_argument('--fix', '--fixed',
                             dest='fixed',
                             help='Path to the fixed image')
    args_parser.add_argument('--fix-subpath', '--fixed-subpath',
                             dest='fixed_subpath',
                             help='Fixed image subpath')
    args_parser.add_argument('--fix-timeindex', '--fixed-timeindex',
                             dest='fixed_timeindex',
                             type=int,
                             default=None,
                             help='Fixed image time index')
    args_parser.add_argument('--fix-channel', '--fixed-channel',
                             dest='fixed_channel',
                             type=int,
                             default=None,
                             help='Fixed image channel')
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
    args_parser.add_argument('--mov-timeindex', '--moving-timeindex',
                             dest='moving_timeindex',
                             type=int,
                             default=None,
                             help='Moving image time index')
    args_parser.add_argument('--mov-channel', '--moving-channel',
                             dest='moving_channel',
                             type=int,
                             default=None,
                             help='Moving image channel')
    args_parser.add_argument('--mov-spacing', '--moving-spacing',
                             dest='moving_spacing',
                             type=floattuple,
                             help='Moving image voxel spacing')

    args_parser.add_argument('--affine-transform', '--affine-transformations',
                             dest='affine_transformations',
                             type=stringlist,
                             help='Affine transformations')

    args_parser.add_argument('--local-transform', dest='local_transform',
                             help='Local (vector field) transformation')
    args_parser.add_argument('--local-transform-subpath',
                             dest='local_transform_subpath',
                             help='Local transformation dataset to be applied')
    args_parser.add_argument('--local-transform-spacing', '--transform-spacing',
                             dest='local_transform_spacing',
                             type=floattuple,
                             help='Local transform spacing')

    args_parser.add_argument('--output',
                             dest='output',
                             help='Output directory')
    args_parser.add_argument('--output-subpath',
                             dest='output_subpath',
                             help='Subpath for the warped output')
    args_parser.add_argument('--output-timeindex',
                             dest='output_timeindex',
                             type=int,
                             default=None,
                             help='Output image time index')
    args_parser.add_argument('--output-channel',
                             dest='output_channel',
                             type=int,
                             default=None,
                             help='Output image channel')
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
                             default=0.1,
                             type=float,
                             help='partition overlap when splitting the work - a fractional number between 0 - 1')

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

    fix_data = ImageData(
        args.fixed,
        fix_subpath,
        image_timeindex=args.fixed_timeindex,
        image_channel=args.fixed_channel,
    )
    mov_data = ImageData(
        args.moving,
        mov_subpath,
        image_timeindex=args.moving_timeindex,
        image_channel=args.moving_channel,
    )
    if args.fixed_spacing:
        fix_data.voxel_spacing = args.fixed_spacing[::-1] # xyz -> zyx

    if args.moving_spacing:
        mov_data.voxel_spacing = args.moving_spacing[::-1] # xyz -> zyx
    elif args.fixed_spacing:
        mov_data.voxel_spacing = fix_data.voxel_spacing

    logger.info(f'Fixed volume: {fix_data}')
    logger.info(f'Moving volume: {mov_data}')

    local_deform_field = ImageData(args.local_transform, args.local_transform_subpath)
    if args.local_transform_spacing:
        # in case the transform spacing arg has the channel dimension - truncate it
        local_deform_spacing = args.local_transform_spacing[::-1][:fix_data.spatial_ndim]  # xyz -> zyx
    else:
        local_deform_spacing = local_deform_field.voxel_spacing[:fix_data.spatial_ndim]

    if (args.output_blocksize is not None and
        len(args.output_blocksize) > 0):
        output_blocks = args.output_blocksize[::-1] # make it zyx
    else:
        # default to output_chunk_size
        output_blocks = (args.output_chunk_size,) * fix_data.spatial_ndim

    if args.output:
        output_attrs = io_utility.prepare_parent_group_attrs(
            args.output,
            output_subpath,
            axes=mov_data.get_attr('axes'),
            coordinateTransformations=mov_data.get_attr('coordinateTransformations'),
        )
        output_shape = fix_data.shape
        if len(output_blocks) < len(output_shape):
            # align_blocksize is not set, so use default block size
            output_chunk_size = (1,) * (len(output_shape)-len(output_blocks)) + tuple(get_spatial_values(output_blocks))
        else:
            output_chunk_size = tuple(get_spatial_values(output_blocks))

        output_dataarray = io_utility.create_dataset(
            args.output,
            output_subpath,
            output_shape,
            output_chunk_size,
            fix_data.dtype,
            compressor=args.compression,
            for_timeindex=args.output_timeindex,
            for_channel=args.output_channel,
            parent_attrs=output_attrs,
            pixelResolution=calc_full_voxel_resolution_attr(mov_data.voxel_spacing,
                                                            mov_data.voxel_downsampling),
            downsamplingFactors=calc_downsampling_attr(mov_data.voxel_downsampling),
        )

        # read affine transformations
        if args.affine_transformations:
            logger.info(f'Affine transformations arg: {args.affine_transformations}')
            applied_affines = [args.affine_transformations]
            affine_transforms_list = [np.loadtxt(tfile)
                                      for tfile in args.affine_transformations]
            affine_spacing = (1.,) * mov_data.spatial_ndim
            transforms_spacings = (affine_spacing,) * len(applied_affines)
        else:
            applied_affines = []
            affine_transforms_list = []
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

        logger.info((
            f'Apply {applied_transforms} to '
            f'{mov_data} -> {args.output}:{output_subpath} '
            f'transform spacing: {transforms_spacings} '
        ))

        # open a dask client
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

        try:
            distributed_apply_transform(
                fix_data, mov_data,
                output_blocks, # use block chunk size for distributing the work
                all_transforms, # transform_list
                cluster_client,
                overlap_factor=args.blocks_overlap_factor,
                aligned_data=output_dataarray,
                aligned_data_timeindex=args.output_timeindex,
                aligned_data_channel=args.output_channel,
                transform_spacing=transforms_spacings,
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