import argparse
import numpy as np
import bigstream.n5_utils as n5_utils
import yaml

from flatten_json import flatten
from os.path import exists
from ClusterWrap.clusters import (local_cluster, remote_cluster)
from bigstream.align import alignment_pipeline
from bigstream.transform import apply_transform
from bigstream.distributed_align import distributed_alignment_pipeline
from bigstream.distributed_transform import (distributed_apply_transform,
    distributed_invert_displacement_vector_field)


def _inttuple(arg):
    if arg is not None and arg.strip():
        return tuple([int(d) for d in arg.split(',')])
    else:
        return ()


def _floattuple(arg):
    if arg is not None and arg.strip():
        return tuple([float(d) for d in arg.split(',')])
    else:
        return ()


def _stringlist(arg):
    if arg is not None and arg.strip():
        return list(filter(lambda x: x, [s.strip() for s in arg.split(',')]))
    else:
        return []


def _intlist(arg):
    if arg is not None and arg.strip():
        return [int(d) for d in arg.split(',')]
    else:
        return []


class _ArgsHelper:

    def __init__(self, prefix):
        self._prefix = prefix

    def _argflag(self, argname):
        return '--{}-{}'.format(self._prefix, argname)

    def _argdest(self, argname):
        return '{}_{}'.format(self._prefix, argname)


def _define_args(global_descriptor, local_descriptor):
    args_parser = argparse.ArgumentParser(description='Registration pipeline')
    args_parser.add_argument('--fixed-global',
                             '--fixed-lowres',
                             dest='fixed_global',
                             help='Fixed global (low resolution) volume path')
    args_parser.add_argument('--fixed-global-subpath',
                             '--fixed-lowres-subpath',
                             dest='fixed_global_subpath',
                             help='Fixed global (low resolution) subpath')

    args_parser.add_argument('--moving-global',
                             '--moving-lowres',
                             dest='moving_global',
                             help='Moving global (low resolution) volume path')
    args_parser.add_argument('--moving-global-subpath',
                             '--moving-lowres-subpath',
                             dest='moving_global_subpath',
                             help='Moving global (low resolution) subpath')

    args_parser.add_argument('--fixed-local',
                             '--fixed-highres',
                             dest='fixed_local',
                             help='Path to the fixed local (high resolution) volume')
    args_parser.add_argument('--fixed-local-subpath',
                             '--fixed-highres-subpath',
                             dest='fixed_local_subpath',
                             help='Fixed local (high resolution) subpath')

    args_parser.add_argument('--moving-local',
                             '--moving-highres',
                             dest='moving_local',
                             help='Path to the moving local (high resolution) volume')
    args_parser.add_argument('--moving-local-subpath',
                             '--moving-highres-subpath',
                             dest='moving_local_subpath',
                             help='Moving local (high resolution) subpath')

    args_parser.add_argument('--use-existing-global-transform',
                             dest='use_existing_global_transform',
                             action='store_true',
                             help='If set use an existing global transform')

    args_parser.add_argument('--output-chunk-size', 
                             dest='output_chunk_size',
                             default=128,
                             type=int,
                             help='Output chunk size')

    args_parser.add_argument(global_descriptor._argflag('output-dir'),
                             dest=global_descriptor._argdest('output_dir'),
                             help='Global alignment output directory')
    args_parser.add_argument(local_descriptor._argflag('output-dir'),
                             dest=local_descriptor._argdest('output_dir'),
                             help='Local alignment output directory')
    args_parser.add_argument(local_descriptor._argflag('working-dir'),
                             dest=local_descriptor._argdest('working_dir'),
                             help='Local alignment working directory')
    args_parser.add_argument('--global-registration-steps',
                             dest='global_registration_steps',
                             type=_stringlist,
                             help='Global (lowres) registration steps, e.g. ransac,affine')
    args_parser.add_argument('--global-transform-name',
                             dest='global_transform_name',
                             default='affine-transform.mat',
                             type=str,
                             help='Global transform name')
    args_parser.add_argument('--global-inv-transform-name',
                             dest='global_inv_transform_name',
                             default='inv-affine-transform.mat',
                             type=str,
                             help='Inverse global transform name')
    args_parser.add_argument('--global-aligned-name',
                             dest='global_aligned_name',
                             type=str,
                             help='Global aligned name')

    _define_ransac_args(args_parser.add_argument_group(
        description='Global ransac arguments'),
        global_descriptor)
    _define_affine_args(args_parser.add_argument_group(
        description='Global affine arguments'),
        global_descriptor)

    args_parser.add_argument('--local-registration-steps',
                             dest='local_registration_steps',
                             type=_stringlist,
                             help='Local (highres) registration steps, .e.g. ransac,deform')
    args_parser.add_argument('--blocks-partitionsize',
                             dest='blocks_partitionsize',
                             default=128,
                             type=int,
                             help='blocksize for splitting the work')
    args_parser.add_argument('--overlap-factor',
                             dest='overlap_factor',
                             default=0.5,
                             type=float,
                             help='partition overlap when splitting the work - a fractional number between 0 - 1')
    args_parser.add_argument('--local-transform-name',
                             dest='local_transform_name',
                             default='deform-transform',
                             type=str,
                             help='Local transform name')
    args_parser.add_argument('--local-inv-transform-name',
                             dest='local_inv_transform_name',
                             default='inv-deform-transform',
                             type=str,
                             help='Local inverse transform name')
    args_parser.add_argument('--local-aligned-name',
                             dest='local_aligned_name',
                             type=str,
                             help='Local aligned name')
    args_parser.add_argument('--local-write-group-interval',
                             dest='local_write_group_interval',
                             default=30,
                             type=int,
                             help="Write group interval for distributed processed blocks")
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

    _define_ransac_args(args_parser.add_argument_group(
        description='Local ransac arguments'),
        local_descriptor)
    _define_deform_args(args_parser.add_argument_group(
        description='Local deform arguments'),
        local_descriptor)

    args_parser.add_argument('--dask-scheduler', dest='dask_scheduler',
                             type=str, default=None,
                             help='Run with distributed scheduler')

    args_parser.add_argument('--dask-config', dest='dask_config',
                             type=str, default=None,
                             help='YAML file containing dask configuration')

    args_parser.add_argument('--debug', dest='debug',
                             action='store_true',
                             help='Save some intermediate images for debugging')

    return args_parser


def _define_ransac_args(ransac_args, args):
    ransac_args.add_argument(args._argflag('ransac-num-sigma-max'),
                             dest=args._argdest('num_sigma_max'),
                             type=int,
                             default=15,
                             help='Ransac sigma max')
    ransac_args.add_argument(args._argflag('ransac-cc-radius'),
                             dest=args._argdest('cc_radius'),
                             type=int,
                             default=12,
                             help='Ransac radius')
    ransac_args.add_argument(args._argflag('ransac-nspots'),
                             dest=args._argdest('nspots'),
                             type=int,
                             default=5000,
                             help='Ransac nspots')
    ransac_args.add_argument(args._argflag('ransac-diagonal-constraint'),
                             dest=args._argdest('diagonal_constraint'),
                             type=float,
                             default=0.75,
                             help='Ransac diagonal constraint')
    ransac_args.add_argument(args._argflag('ransac-match-threshold'),
                             dest=args._argdest('match_threshold'),
                             type=float,
                             default=0.7,
                             help='Ransac match threshold')
    ransac_args.add_argument(args._argflag('ransac-align-threshold'),
                             dest=args._argdest('align_threshold'),
                             type=float,
                             default=2.0,
                             help='Ransac match threshold')
    ransac_args.add_argument(args._argflag('ransac-blob-sizes'),
                             dest=args._argdest('blob_sizes'),
                             metavar='s1,s2,...,sn',
                             type=_intlist,
                             default=[6, 20],
                             help='Ransac blob sizes')


def _define_affine_args(affine_args, args):
    affine_args.add_argument(args._argflag('shrink-factors'),
                             dest=args._argdest('shrink_factors'),
                             metavar='sf1,...,sfn',
                             type=_inttuple, default=None,
                             help='Shrink factors')
    affine_args.add_argument(args._argflag('smooth-sigmas'),
                             dest=args._argdest('smooth_sigmas'),
                             metavar='s1,...,sn',
                             type=_floattuple,
                             help='Smoothing sigmas')
    affine_args.add_argument(args._argflag('learning-rate'),
                             dest=args._argdest('learning_rate'),
                             type=float, default=0.25,
                             help='Learning rate')
    affine_args.add_argument(args._argflag('min-step'),
                             dest=args._argdest('min_step'),
                             type=float, default=0.,
                             help='Minimum step')
    affine_args.add_argument(args._argflag('iterations'),
                             dest=args._argdest('iterations'),
                             type=int, default=100,
                             help='Number of iterations')


def _define_deform_args(deform_args, args):
    deform_args.add_argument(args._argflag('control-point-spacing'),
                             dest=args._argdest('control_point_spacing'),
                             type=float, default=50.,
                             help='Control point spacing')
    deform_args.add_argument(args._argflag('control-point-levels'),
                             dest=args._argdest('control_point_levels'),
                             metavar='s1,...,sn',
                             type=_inttuple, default=(1,),
                             help='Control point levels')
    # deform args are actually a superset of affine args
    _define_affine_args(deform_args, args)


def _check_attr(args, argdescriptor, argname):
    attr_value = getattr(args, argdescriptor._argdest(argname), None)
    return attr_value is not None


def _extract_ransac_args(args, argdescriptor):
    ransac_args = {}
    if _check_attr(args, argdescriptor, 'num_sigma_max'):
        ransac_args['num_sigma_max'] = getattr(
            args, argdescriptor._argdest('num_sigma_max'))
    if _check_attr(args, argdescriptor, 'cc_radius'):
        ransac_args['cc_radius'] = getattr(
            args, argdescriptor._argdest('cc_radius'))
    if _check_attr(args, argdescriptor, 'nspots'):
        ransac_args['nspots'] = getattr(
            args, argdescriptor._argdest('nspots'))
    if _check_attr(args, argdescriptor, 'diagonal_constraint'):
        ransac_args['diagonal_constraint'] = getattr(
            args, argdescriptor._argdest('diagonal_constraint'))
    if _check_attr(args, argdescriptor, 'match_threshold'):
        ransac_args['match_threshold'] = getattr(
            args, argdescriptor._argdest('match_threshold'))
    if _check_attr(args, argdescriptor, 'align_threshold'):
        ransac_args['align_threshold'] = getattr(
            args, argdescriptor._argdest('align_threshold'))
    if _check_attr(args, argdescriptor, 'blob_sizes'):
        ransac_args['blob_sizes'] = getattr(
            args, argdescriptor._argdest('blob_sizes'))
    return ransac_args


def _extract_affine_args(args, argdescriptor):
    affine_args = {'optimizer_args': {}}
    _extract_affine_args_to(args, argdescriptor, affine_args)
    return affine_args


def _extract_affine_args_to(args, argdescriptor, affine_args):
    if _check_attr(args, argdescriptor, 'shrink_factors'):
        affine_args['shrink_factors'] = getattr(
            args, argdescriptor._argdest('shrink_factors'))
    if _check_attr(args, argdescriptor, 'smooth_sigmas'):
        affine_args['smooth_sigmas'] = getattr(
            args, argdescriptor._argdest('smooth_sigmas'))
    if _check_attr(args, argdescriptor, 'learning_rate'):
        affine_args['optimizer_args']['learningRate'] = getattr(
            args, argdescriptor._argdest('learning_rate'))
    if _check_attr(args, argdescriptor, 'min_step'):
        affine_args['optimizer_args']['minStep'] = getattr(
            args, argdescriptor._argdest('min_step'))
    if _check_attr(args, argdescriptor, 'iterations'):
        affine_args['optimizer_args']['numberOfIterations'] = getattr(
            args, argdescriptor._argdest('iterations'))


def _extract_deform_args(args, argdescriptor):
    deform_args = {'optimizer_args': {}}
    _extract_affine_args_to(args, argdescriptor, deform_args)
    if _check_attr(args, argdescriptor, 'control_point_spacing'):
        deform_args['control_point_spacing'] = getattr(
            args, argdescriptor._argdest('control_point_spacing'))
    if _check_attr(args, argdescriptor, 'control_point_levels'):
        deform_args['control_point_levels'] = getattr(
            args, argdescriptor._argdest('control_point_levels'))
    return deform_args


def _extract_output_dir(args, argdescriptor):
    return getattr(args, argdescriptor._argdest('output_dir'))


def _extract_working_dir(args, argdescriptor):
    return getattr(args, argdescriptor._argdest('working_dir'))


def _run_global_alignment(args, steps, global_output_dir):
    if global_output_dir: 
        if args.global_transform_name:
            global_transform_file = (global_output_dir + '/' + 
                                    args.global_transform_name)
        else:
            global_transform_file = None
        if args.global_inv_transform_name:
            global_inv_transform_file = (global_output_dir + '/' + 
                                         args.global_inv_transform_name)
        else:
            global_inv_transform_file = None
    else:
        global_transform_file = None
        global_inv_transform_file = None

    if (args.use_existing_global_transform and
        global_transform_file and
            exists(global_transform_file)):
        print('Read global transform from', global_transform_file, flush=True)
        global_transform = np.loadtxt(global_transform_file)
    elif steps:
        print('Run global registration with:', args, steps, flush=True)
        # Read the global inputs
        fix_arraydata, fix_attrs = n5_utils.open(
            args.fixed_global, args.fixed_global_subpath)
        mov_arraydata, mov_attrs = n5_utils.open(
            args.moving_global, args.moving_global_subpath)
        fix_voxel_spacing = n5_utils.get_voxel_spacing(fix_attrs)
        mov_voxel_spacing = n5_utils.get_voxel_spacing(mov_attrs)

        print('Fixed lowres volume attributes:',
              fix_arraydata.shape, fix_voxel_spacing, flush=True)
        print('Moving lowres volume attributes:',
              mov_arraydata.shape, mov_voxel_spacing, flush=True)

        global_transform, global_alignment = _align_global_data(
            fix_arraydata[...],  # read image in memory
            mov_arraydata[...],
            fix_voxel_spacing,
            mov_voxel_spacing,
            steps)

        if global_transform_file:
            print('Save global transformation to', global_transform_file)
            np.savetxt(global_transform_file, global_transform)
        else:
            print('Skip saving global transformation')

        if global_inv_transform_file:
            try:
                global_inv_transform = np.linalg.inv(global_transform)
                print('Save global inverse transformation to', global_inv_transform_file)
                np.savetxt(global_inv_transform_file, global_inv_transform)
            except Exception:
                print('Global transformation', global_transform, 'is not invertible')

        if global_output_dir and args.global_aligned_name:
            global_aligned_file = (global_output_dir + '/' + 
                                   args.global_aligned_name)
            print('Save global aligned volume to', global_aligned_file)
            n5_utils.create_dataset(
                global_aligned_file,
                args.moving_global_subpath, # same dataset as the moving image
                global_alignment.shape,
                (args.output_chunk_size,)*global_alignment.ndim,
                global_alignment.dtype,
                data=global_alignment,
                pixelResolution=mov_attrs.get('pixelResolution'),
                downsamplingFactors=mov_attrs.get('downsamplingFactors'),
            )
        else:
            print('Skip savign lowres aligned volume')
    else:
        print('Skip global alignment because no global steps were specified.')
        global_transform = None

    return global_transform


def _align_global_data(fix_data,
                       mov_data,
                       fix_spacing,
                       mov_spacing,
                       steps):
    print('Run low res alignment:', steps, flush=True)
    affine = alignment_pipeline(fix_data,
                                mov_data,
                                fix_spacing,
                                mov_spacing,
                                steps)
    print('Apply affine transform', flush=True)
    # apply transform
    aligned = apply_transform(fix_data,
                              mov_data,
                              fix_spacing,
                              mov_spacing,
                              transform_list=[affine,])

    return affine, aligned


def _run_local_alignment(args, steps, global_transform, output_dir, working_dir):
    if steps:
        print('Run local registration with:', steps, flush=True)

        # Read the highres inputs - if highres is not defined default it to lowres
        fix_highres_path = args.fixed_local if args.fixed_local else args.fixed_global
        mov_highres_path = args.fixed_local if args.fixed_local else args.fixed_global

        fix_highres_ldata, fix_highres_attrs = n5_utils.open(
            fix_highres_path, args.fixed_local_subpath)
        mov_highres_ldata, mov_highres_attrs = n5_utils.open(
            mov_highres_path, args.moving_local_subpath)

        if (args.dask_config):
            with open(args.dask_config) as f:
                dask_config = flatten(yaml.safe_load(f))
        else:
            dask_config = {}

        if args.dask_scheduler:
            cluster = remote_cluster(args.dask_scheduler, config=dask_config)
        else:
            cluster = local_cluster(config=dask_config)

        overlap_factor = (0.5 if (args.overlap_factor <= 0 or
                                    args.overlap_factor >= 1)
                                 else
                                    args.overlap_factor)
        _align_local_data(
            (fix_highres_path, args.fixed_local_subpath, fix_highres_ldata),
            (mov_highres_path, args.moving_local_subpath, mov_highres_ldata),
            fix_highres_attrs,
            mov_highres_attrs,
            steps,
            args.blocks_partitionsize,
            overlap_factor,
            [global_transform] if global_transform is not None else [],
            output_dir,
            args.local_transform_name,
            args.local_inv_transform_name,
            args.local_aligned_name,
            args.output_chunk_size,
            args.local_write_group_interval,
            args.inv_iterations,
            args.inv_order,
            args.inv_sqrt_iterations,
            cluster,
        )
    else:
        print('Skip local alignment because no local steps were specified.')


def _align_local_data(fix_input,
                      mov_input,
                      fix_attrs,
                      mov_attrs,
                      steps,
                      partitionsize,
                      overlap_factor,
                      global_transforms_list,
                      output_dir,
                      local_transform_name,
                      local_inv_transform_name,
                      local_aligned_name,
                      output_chunk_size,
                      write_group_interval,
                      inv_iterations,
                      inv_order,
                      inv_sqrt_iterations,
                      cluster):
    fix_path, fix_dataset, fix_dataarray = fix_input
    mov_path, mov_dataset, mov_dataarray = mov_input

    print('Run local alignment:', steps, partitionsize, flush=True)
    output_blocks_chunksize = (output_chunk_size,) * fix_dataarray.ndim
    blocks_partitionsize = (partitionsize,) * fix_dataarray.ndim

    fix_spacing = n5_utils.get_voxel_spacing(fix_attrs)
    mov_spacing = n5_utils.get_voxel_spacing(mov_attrs)

    print('Align moving data',
          mov_path, mov_dataset, mov_dataarray.shape, mov_spacing,
          'to reference',
          fix_path, fix_dataset, fix_dataarray.shape, fix_spacing,
          flush=True)

    if output_dir and local_transform_name:
        deform_path = output_dir + '/' + local_transform_name
        local_deform = n5_utils.create_dataset(
            deform_path,
            None, # no dataset subpath
            fix_dataarray.shape + (fix_dataarray.ndim,),
            output_blocks_chunksize + (fix_dataarray.ndim,),
            np.float32,
            # the transformation does not have to have spacing attributes
        )
    else:
        deform_path = None
        local_deform = None
    print('Calculate transformation', deform_path, 'for local alignment of',
          mov_path, mov_dataset,
          'to reference',
          fix_path, fix_dataset,
          flush=True)
    distributed_alignment_pipeline(
        fix_dataarray, mov_dataarray,
        fix_spacing, mov_spacing,
        steps,
        blocks_partitionsize,
        overlap_factor=overlap_factor,
        static_transform_list=global_transforms_list,
        output_transform=local_deform,
        write_group_interval=write_group_interval,
        cluster=cluster,
    )
    if deform_path and local_inv_transform_name:
        inv_deform_path = output_dir + '/' + local_inv_transform_name
        local_inv_deform = n5_utils.create_dataset(
            inv_deform_path,
            None, # no dataset subpath
            fix_dataarray.shape + (fix_dataarray.ndim,),
            output_blocks_chunksize + (fix_dataarray.ndim,),
            np.float32,
            # the transformation does not have to have spacing attributes
        )
        print('Calculate inverse transformation',
              inv_deform_path, 'from', deform_path,
              'for local alignment of',
              mov_path, mov_dataset,
              'to reference',
              fix_path, fix_dataset,
              flush=True)
        distributed_invert_displacement_vector_field(
            local_deform,
            fix_spacing,
            blocks_partitionsize,
            local_inv_deform,
            overlap_factor=overlap_factor,
            iterations=inv_iterations,
            order=inv_order,
            sqrt_iterations=inv_sqrt_iterations,
            cluster=cluster,
        )

    if output_dir and local_aligned_name:
        # Apply local transformation only if 
        # highres aligned output name is set
        aligned_path = output_dir + '/' + local_aligned_name
        local_aligned = n5_utils.create_dataset(
            aligned_path,
            mov_dataset,
            fix_dataarray.shape,
            output_blocks_chunksize,
            fix_dataarray.dtype,
            pixelResolution=mov_attrs.get('pixelResolution'),
            downsamplingFactors=mov_attrs.get('downsamplingFactors'),
        )
        print('Apply', deform_path, 'to',
              mov_path, mov_dataset, '->', aligned_path, mov_dataset,
              flush=True)
        distributed_apply_transform(
            fix_dataarray, mov_dataarray,
            fix_spacing, mov_spacing,
            blocks_partitionsize,
            overlap_factor=overlap_factor,
            transform_list=global_transforms_list + [local_deform],
            aligned_data=local_aligned,
            cluster=cluster,
        )
    else:
        local_aligned = None

    return local_deform, local_aligned


if __name__ == '__main__':
    global_descriptor = _ArgsHelper('global')
    local_descriptor = _ArgsHelper('local')
    args_parser = _define_args(global_descriptor, local_descriptor)
    args = args_parser.parse_args()
    print('Invoked registration:', args, flush=True)

    if args.global_registration_steps:
        args_for_global_steps = {
            'ransac': _extract_ransac_args(args, global_descriptor),
            'affine': _extract_affine_args(args, global_descriptor),
        }
        global_steps = [(s, args_for_global_steps.get(s, {}))
                        for s in args.global_registration_steps]
    else:
        global_steps = []

    global_transform = _run_global_alignment(
        args,
        global_steps,
        _extract_output_dir(args, global_descriptor)
    )

    if args.local_registration_steps:
        args_for_local_steps = {
            'ransac': _extract_ransac_args(args, local_descriptor),
            'deform': _extract_deform_args(args, local_descriptor),
        }
        local_steps = [(s, args_for_local_steps.get(s, {}))
                       for s in args.local_registration_steps]
    else:
        local_steps = []

    _run_local_alignment(
        args,
        local_steps,
        global_transform,
        _extract_output_dir(args, local_descriptor),
        _extract_working_dir(args, local_descriptor),
    )
