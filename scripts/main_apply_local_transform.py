import argparse
import numpy as np
import bigstream.io_utility as io_utility
import yaml

from flatten_json import flatten
from os.path import exists
from dask.distributed import (Client, LocalCluster)
from bigstream.distributed_transform import distributed_apply_transform
from bigstream.image_data import ImageData


def _inttuple(arg):
    if arg is not None and arg.strip():
        return tuple([int(d) for d in arg.split(',')])
    else:
        return ()


def _stringlist(arg):
    if arg is not None and arg.strip():
        return list(filter(lambda x: x, [s.strip() for s in arg.split(',')]))
    else:
        return []


def _define_args():
    args_parser = argparse.ArgumentParser(description='Apply transformation')
    args_parser.add_argument('--fixed', dest='fixed',
                             help='Path to the fixed image')
    args_parser.add_argument('--fixed-subpath',
                             dest='fixed_subpath',
                             help='Fixed image subpath')

    args_parser.add_argument('--moving', dest='moving',
                             help='Path to the moving image')
    args_parser.add_argument('--moving-subpath',
                             dest='moving_subpath',
                             help='Moving image subpath')

    args_parser.add_argument('--affine-transformations',
                             dest='affine_transformations',
                             type=_stringlist,
                             help='Affine transformations')

    args_parser.add_argument('--local-transform', dest='local_transform',
                             help='Local (vector field) transformation')
    args_parser.add_argument('--local-transform-subpath',
                             dest='local_transform_subpath',
                             help='Local transformation dataset to be applied')

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
                             type=_inttuple,
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

    return args_parser


def _run_apply_transform(args):

    # Read the datasets - if the moving dataset is not defined it defaults to the fixed dataset
    fix_subpath = args.fixed_subpath
    mov_subpath = args.moving_subpath if args.moving_subpath else fix_subpath
    output_subpath = args.output_subpath if args.output_subpath else mov_subpath

    fix_data = ImageData(args.fixed, fix_subpath)
    mov_data = ImageData(args.moving, mov_subpath)


    print(f'Fixed volume: {fix_data}', flush=True)
    print(f'Moving volume: {mov_data}', flush=True)

    if (args.dask_config):
        import dask.config
        with open(args.dask_config) as f:
            dask_config = flatten(yaml.safe_load(f))
            dask.config.set(dask_config)

    if args.dask_scheduler:
        cluster_client = Client(address=args.dask_scheduler)
    else:
        cluster_client = Client(LocalCluster())

    # read local deform, but ignore attributes as they are not needed
    local_deform, _ = io_utility.open(args.local_transform,
                                      args.local_transform_subpath)

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
            pixelResolution=mov_data.get_attr('pixelResolution'),
            downsamplingFactors=mov_data.get_attr('downsamplingFactors'),
        )

        if args.affine_transformations:
            print('Affine transformations arg:', args.affine_transformations,
                  flush=True)
            affine_transforms_list = [np.loadtxt(tfile)
                                      for tfile in args.affine_transformations]
        else:
            affine_transforms_list = []

        all_transforms = (args.affine_transformations +
                          [(args.local_transform, args.local_transform_subpath)])
        print('Apply', all_transforms, ' to ',
              args.moving, mov_subpath, '->', args.output, output_subpath)

        distributed_apply_transform(
            fix_data, mov_data,
            fix_data.voxel_spacing, mov_data.voxel_spacing,
            output_blocks, # use block chunk size for distributing the work
            affine_transforms_list + [local_deform], # transform_list
            cluster_client,
            overlap_factor=args.blocks_overlap_factor,
            aligned_data=output_dataarray,
        )
        return output_dataarray
    else:
        return None


if __name__ == '__main__':
    args_parser = _define_args()
    args = args_parser.parse_args()
    print('Invoked transformation:', args, flush=True)

    _run_apply_transform(args)
