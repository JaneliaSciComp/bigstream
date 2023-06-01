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
                             dest='fixed_global',
                             help='Fixed global (low resolution) volume path')
    args_parser.add_argument('--fixed-global-subpath',
                             '--fixed-lowres-subpath',
                             dest='fixed_global_subpath',
                             help='Fixed global (low resolution) subpath')

    args_parser.add_argument('--fixed-global-mask',
                             dest='fixed_global_mask',
                             help='Fixed global (low resolution) mask path')
    args_parser.add_argument('--fixed-global-mask-subpath',
                             dest='fixed_global_mask_subpath',
                             help='Fixed global (low resolution) mask subpath')

    args_parser.add_argument('--moving-global',
                             dest='moving_global',
                             help='Moving global (low resolution) volume path')
    args_parser.add_argument('--moving-global-subpath',
                             '--moving-lowres-subpath',
                             dest='moving_global_subpath',
                             help='Moving global (low resolution) subpath')

    args_parser.add_argument('--moving-global-mask',
                             dest='moving_global_mask',
                             help='Moving global (low resolution) mask path')
    args_parser.add_argument('--moving-global-mask-subpath',
                             dest='moving_global_mask_subpath',
                             help='Moving global (low resolution) mask subpath')

    args_parser.add_argument('--fixed-local',
                             dest='fixed_local',
                             help='Path to the fixed local (high resolution) volume')
    args_parser.add_argument('--fixed-local-subpath',
                             dest='fixed_local_subpath',
                             help='Fixed local (high resolution) subpath')

    args_parser.add_argument('--fixed-local-mask',
                             dest='fixed_local_mask',
                             help='Fixed local (high resolution) mask path')
    args_parser.add_argument('--fixed-local-mask-subpath',
                             dest='fixed_local_mask_subpath',
                             help='Fixed local (high resolution) mask subpath')

    args_parser.add_argument('--moving-local',
                             dest='moving_local',
                             help='Path to the moving local (high resolution) volume')
    args_parser.add_argument('--moving-local-subpath',
                             dest='moving_local_subpath',
                             help='Moving local (high resolution) subpath')

    args_parser.add_argument('--moving-local-mask',
                             dest='moving_local_mask',
                             help='Moving local (high resolution) mask path')
    args_parser.add_argument('--moving-local-mask-subpath',
                             dest='moving_local_mask_subpath',
                             help='Moving local (high resolution) mask subpath')

    args_parser.add_argument('--use-existing-global-transform',
                             dest='use_existing_global_transform',
                             action='store_true',
                             help='If set use an existing global transform')

    args_parser.add_argument('--output-chunk-size',
                             dest='output_chunk_size',
                             default=128,
                             type=int,
                             help='Output chunk size')
    args_parser.add_argument('--output-blocksize',
                             dest='output_blocksize',
                             type=_inttuple,
                             help='Output chunk size as a tuple.')

    args_parser.add_argument(global_descriptor._argflag('output-dir'),
                             dest=global_descriptor._argdest('output_dir'),
                             help='Global alignment output directory')
    args_parser.add_argument(local_descriptor._argflag('output-dir'),
                             dest=local_descriptor._argdest('output_dir'),
                             help='Local alignment output directory')
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
    args_parser.add_argument('--blocks-overlap-factor',
                             dest='blocks_overlap_factor',
                             default=0.5,
                             type=float,
                             help='partition overlap when splitting the work - a fractional number between 0 - 1')
    args_parser.add_argument('--local-transform-name',
                             dest='local_transform_name',
                             default='deform-transform',
                             type=str,
                             help='Local transform name')
    args_parser.add_argument('--local-transform-subpath',
                             dest='local_transform_subpath',
                             default='',
                             type=str,
                             help='Local transform subpath')
    args_parser.add_argument('--local-transform-blocksize',
                             dest='local_transform_blocksize',
                             type=_inttuple,
                             help='Local transform chunk size')
    args_parser.add_argument('--local-inv-transform-name',
                             dest='local_inv_transform_name',
                             default='inv-deform-transform',
                             type=str,
                             help='Local inverse transform name')
    args_parser.add_argument('--local-inv-transform-subpath',
                             dest='local_inv_transform_subpath',
                             default='',
                             type=str,
                             help='Local inverse transform subpath')
    args_parser.add_argument('--local-inv-transform-blocksize',
                             dest='local_inv_transform_blocksize',
                             type=_inttuple,
                             help='Local inverse transform chunk size')
    args_parser.add_argument('--local-aligned-name',
                             dest='local_aligned_name',
                             type=str,
                             help='Local aligned name')
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
    affine_args.add_argument(args._argflag('metric'),
                             dest=args._argdest('metric'),
                             type=str,
                             default='MMI',
                             help='Metric')
    affine_args.add_argument(args._argflag('optimizer'),
                             dest=args._argdest('optimizer'),
                             type=str,
                             default='RSGD',
                             help='Optimizer')
    affine_args.add_argument(args._argflag('sampling'),
                             dest=args._argdest('sampling'),
                             type=str,
                             default='NONE',
                             help='Sampling')
    affine_args.add_argument(args._argflag('interpolator'),
                             dest=args._argdest('interpolator'),
                             type=str,
                             default='1',
                             help='Interpolator')
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
    affine_args.add_argument(args._argflag('sampling-percentage'),
                             dest=args._argdest('sampling_percentage'),
                             type=float,
                             help='Sampling percentage')
    affine_args.add_argument(args._argflag('alignment-spacing'),
                             dest=args._argdest('alignment_spacing'),
                             type=float,
                             help='Alignment spacing')


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
    if _check_attr(args, argdescriptor, 'metric'):
        affine_args['metric'] = getattr(
            args, argdescriptor._argdest('metric'))
    if _check_attr(args, argdescriptor, 'optimizer'):
        affine_args['optimizer'] = getattr(
            args, argdescriptor._argdest('optimizer'))
    if _check_attr(args, argdescriptor, 'sampling'):
        affine_args['sampling'] = getattr(
            args, argdescriptor._argdest('sampling'))
    if _check_attr(args, argdescriptor, 'interpolator'):
        affine_args['interpolator'] = getattr(
            args, argdescriptor._argdest('interpolator'))
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
    if _check_attr(args, argdescriptor, 'sampling_percentage'):
        affine_args['sampling_percentage'] = getattr(
            args, argdescriptor._argdest('sampling_percentage'))
    if _check_attr(args, argdescriptor, 'alignment_spacing'):
        affine_args['alignment_spacing'] = getattr(
            args, argdescriptor._argdest('alignment_spacing'))


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
            if (args.output_blocksize is not None and
                len(args.output_blocksize) > 0):
                output_blocksize = args.output_blocksize[::-1]
            else:
                # default to output_chunk_size
                output_blocksize = (args.output_chunk_size,) * global_alignment.ndim
            print('Save global aligned volume to', global_aligned_file,
                  'with blocksize', output_blocksize)
            n5_utils.create_dataset(
                global_aligned_file,
                args.moving_global_subpath, # same dataset as the moving image
                global_alignment.shape,
                output_blocksize,
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


def _run_local_alignment(args, steps, global_transform, output_dir):
    if steps:
        print('Run local registration with:', steps, flush=True)

        # Read the highres inputs - if highres is not defined default it to lowres
        fix_local_path = args.fixed_local if args.fixed_local else args.fixed_global
        mov_local_path = args.moving_local if args.moving_local else args.moving_global

        fix_highres_ldata, fix_local_attrs = n5_utils.open(
            fix_local_path, args.fixed_local_subpath)
        mov_highres_ldata, mov_local_attrs = n5_utils.open(
            mov_local_path, args.moving_local_subpath)

        if (args.dask_config):
            with open(args.dask_config) as f:
                dask_config = flatten(yaml.safe_load(f))
        else:
            dask_config = {}

        if args.dask_scheduler:
            cluster = remote_cluster(args.dask_scheduler, config=dask_config)
        else:
            cluster = local_cluster(config=dask_config)

        blocks_overlap_factor = (0.5 if (args.blocks_overlap_factor <= 0 or
                                    args.blocks_overlap_factor >= 1)
                                 else
                                    args.blocks_overlap_factor)

        if (args.output_blocksize is not None and
            len(args.output_blocksize) > 0):
            # block chunks are define as x,y,z so I am reversing it to z,y,x
            output_blocksize = args.output_blocksize[::-1]
        else:
            # default to output_chunk_size
            output_blocksize = (args.output_chunk_size,) * fix_highres_ldata.ndim

        if (args.local_transform_blocksize is not None and
            len(args.local_transform_blocksize) > 0):
            local_transform_blocksize = args.local_transform_blocksize[::-1]
        else:
            # default to output blocksize
            local_transform_blocksize = output_blocksize

        if (args.local_inv_transform_blocksize is not None and
            len(args.local_inv_transform_blocksize) > 0):
            local_inv_transform_blocksize = args.local_inv_transform_blocksize[::-1]
        else:
            # default to local transform blocksize
            local_inv_transform_blocksize = local_transform_blocksize

        _align_local_data(
            (fix_local_path, args.fixed_local_subpath, fix_highres_ldata),
            (mov_local_path, args.moving_local_subpath, mov_highres_ldata),
            fix_local_attrs,
            mov_local_attrs,
            steps,
            blocks_overlap_factor,
            [global_transform] if global_transform is not None else [],
            output_dir,
            args.local_transform_name,
            args.local_transform_subpath,
            local_transform_blocksize,
            args.local_inv_transform_name,
            args.local_inv_transform_subpath,
            local_inv_transform_blocksize,
            args.local_aligned_name,
            output_blocksize,
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
                      blocks_overlap_factor,
                      global_transforms_list,
                      output_dir,
                      local_transform_name,
                      local_transform_subpath,
                      local_transform_blocksize,
                      local_inv_transform_name,
                      local_inv_transform_subpath,
                      local_inv_transform_blocksize,
                      local_aligned_name,
                      output_blocksize,
                      inv_iterations,
                      inv_order,
                      inv_sqrt_iterations,
                      cluster):
    fix_path, fix_dataset, fix_dataarray = fix_input
    mov_path, mov_dataset, mov_dataarray = mov_input

    print('Run local alignment:', steps, output_blocksize, flush=True)

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
            local_transform_subpath,
            fix_dataarray.shape + (fix_dataarray.ndim,),
            local_transform_blocksize + (fix_dataarray.ndim,),
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
        output_blocksize, # parallelize on the block chunk size
        overlap_factor=blocks_overlap_factor,
        static_transform_list=global_transforms_list,
        output_transform=local_deform,
        cluster=cluster,
    )
    if deform_path and local_inv_transform_name:
        inv_deform_path = output_dir + '/' + local_inv_transform_name
        local_inv_deform = n5_utils.create_dataset(
            inv_deform_path,
            local_inv_transform_subpath,
            fix_dataarray.shape + (fix_dataarray.ndim,),
            local_inv_transform_blocksize + (fix_dataarray.ndim,),
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
            local_inv_transform_blocksize, # use blocksize for partitioning
            local_inv_deform,
            overlap_factor=blocks_overlap_factor,
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
            output_blocksize,
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
            output_blocksize, # use block chunk size for distributing work
            overlap_factor=blocks_overlap_factor,
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
            'random': _extract_affine_args(args, global_descriptor),
            'affine': _extract_affine_args(args, global_descriptor),
            'rigid': _extract_affine_args(args, global_descriptor),
            'deform': _extract_deform_args(args, global_descriptor),
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
            'random': _extract_affine_args(args, local_descriptor),
            'affine': _extract_affine_args(args, local_descriptor),
            'rigid': _extract_affine_args(args, local_descriptor),
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
    )
