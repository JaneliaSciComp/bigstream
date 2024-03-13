import argparse
import numpy as np
import bigstream.io_utility as io_utility
import yaml

from flatten_json import flatten
from os.path import exists
from ClusterWrap.clusters import (local_cluster, remote_cluster)
from bigstream.distributed_transform import distributed_apply_transform


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

    fix_attrs = io_utility.read_attributes(args.fixed, fix_subpath)
    mov_attrs = io_utility.read_attributes(args.moving, mov_subpath)

    fix_shape, fix_ndim = io_utility.get_dimensions(fix_attrs)
    mov_shape, mov_ndim = io_utility.get_dimensions(mov_attrs)

    fix_voxel_spacing = io_utility.get_voxel_spacing(fix_attrs)
    mov_voxel_spacing = io_utility.get_voxel_spacing(mov_attrs)

    print('Fixed volume attributes:',
          fix_shape, fix_voxel_spacing, flush=True)
    print('Moving volume attributes:',
          mov_shape, mov_voxel_spacing, flush=True)

    if (args.dask_config):
        with open(args.dask_config) as f:
            dask_config = flatten(yaml.safe_load(f))
    else:
        dask_config = {}

    if args.dask_scheduler:
        cluster = remote_cluster(args.dask_scheduler, config=dask_config)
    else:
        cluster = local_cluster(config=dask_config)

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
            fix_shape,
            output_blocks,
            io_utility.get_dtype(fix_attrs),
            pixelResolution=mov_attrs.get('pixelResolution'),
            downsamplingFactors=mov_attrs.get('downsamplingFactors'),
        )

        if args.affine_transformations:
            affine_transforms_list = [np.loadtxt(tfile)
                                      for tfile in args.affine_transformations]
        else:
            affine_transforms_list = []

        all_transforms = (args.affine_transformations +
                          [(args.local_transform, args.local_transform_subpath)])
        print('Apply', all_transforms,
              args.moving, mov_subpath, '->', args.output, output_subpath)
        distributed_apply_transform(
            args.fixed, fix_subpath,
            args.moving, mov_subpath,
            fix_shape, mov_shape,
            fix_voxel_spacing, mov_voxel_spacing,
            output_blocks, # use block chunk size for distributing the work
            affine_transforms_list + [local_deform], # transform_list
            cluster.client,
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
