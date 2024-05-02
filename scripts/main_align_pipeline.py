import argparse
import numpy as np
import bigstream.io_utility as io_utility
import pydantic.v1.utils as pu
import yaml

from flatten_json import flatten
from os.path import exists
from dask.distributed import (Client, LocalCluster)
from flatten_json import flatten
from bigstream.align import alignment_pipeline
from bigstream.config import default_bigstream_config_str
from bigstream.distributed_align import distributed_alignment_pipeline
from bigstream.distributed_transform import (distributed_apply_transform,
    distributed_invert_displacement_vector_field)
from bigstream.transform import apply_transform


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


class _RegistrationInputs:

    def transform_path(self):
        if self.output_dir and self.transform_name:
            return f'{self.output_dir}/{self.transform_name}'
        else:
            return None

    def inv_transform_path(self):
        if self.output_dir and self.inv_transform_name:
            return f'{self.output_dir}/{self.inv_transform_name}'
        else:
            return None

    def align_path(self):
        if self.output_dir and self.align_name:
            return f'{self.output_dir}/{self.align_name}'
        else:
            return None

    def align_dataset(self):
        if self.align_subpath:
            return self.align_subpath
        else:
            return self.mov_subpath


def _define_args(global_descriptor, local_descriptor):
    args_parser = argparse.ArgumentParser(description='Registration pipeline')

    _define_registration_input_args(
        args_parser.add_argument_group(
            description='Global registration input volumes'),
        global_descriptor,
    )
    _define_registration_input_args(
        args_parser.add_argument_group(
            description='Local registration input volumes'),
        local_descriptor,
    )

    args_parser.add_argument('--align-config',
                             dest='align_config',
                             help='Align config file')
    args_parser.add_argument('--global-use-existing-transform',
                             dest='global_use_existing_transform',
                             action='store_true',
                             help='If set use an existing global transform')
    args_parser.add_argument('--local-processing-size',
                             dest='local_processing_size',
                             type=_inttuple,
                             help='partition overlap when splitting the work - a fractional number between 0 - 1')
    args_parser.add_argument('--local-processing-overlap-factor',
                             dest='local_processing_overlap_factor',
                             default=0.5,
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
    args_parser.add_argument('--debug', dest='debug',
                             action='store_true',
                             help='Save some intermediate images for debugging')

    return args_parser


def _define_registration_input_args(args, args_descriptor):
    args.add_argument(args_descriptor._argflag('fix'),
                      dest=args_descriptor._argdest('fix'),
                      help='Fixed volume')
    args.add_argument(args_descriptor._argflag('fix-subpath'),
                      dest=args_descriptor._argdest('fix_subpath'),
                      help='Fixed volume subpath')
    args.add_argument(args_descriptor._argflag('fix-spacing'),
                      dest=args_descriptor._argdest('fix_spacing'),
                      type=_floattuple,
                      help='Fixed volume voxel spacing')
    args.add_argument(args_descriptor._argflag('fix-mask'),
                      dest=args_descriptor._argdest('fix_mask'),
                      help='Fixed volume mask')
    args.add_argument(args_descriptor._argflag('fix-mask-subpath'),
                      dest=args_descriptor._argdest('fix_mask_subpath'),
                      help='Fixed volume mask subpath')

    args.add_argument(args_descriptor._argflag('mov'),
                      dest=args_descriptor._argdest('mov'),
                      help='Moving volume')
    args.add_argument(args_descriptor._argflag('mov-subpath'),
                      dest=args_descriptor._argdest('mov_subpath'),
                      help='Moving volume subpath')
    args.add_argument(args_descriptor._argflag('mov-spacing'),
                      dest=args_descriptor._argdest('mov_spacing'),
                      type=_floattuple,
                      help='Moving volume voxel spacing')
    args.add_argument(args_descriptor._argflag('mov-mask'),
                      dest=args_descriptor._argdest('mov_mask'),
                      help='Moving volume mask')
    args.add_argument(args_descriptor._argflag('mov-mask-subpath'),
                      dest=args_descriptor._argdest('mov_mask_subpath'),
                      help='Moving volume mask subpath')

    args.add_argument(args_descriptor._argflag('output-dir'),
                      dest=args_descriptor._argdest('output_dir'),
                      help='Alignment output directory')
    args.add_argument(args_descriptor._argflag('transform-name'),
                      dest=args_descriptor._argdest('transform_name'),
                      help='Transform name')
    args.add_argument(args_descriptor._argflag('transform-subpath'),
                      dest=args_descriptor._argdest('transform_subpath'),
                      help='Transform subpath')
    args.add_argument(args_descriptor._argflag('inv-transform-name'),
                      dest=args_descriptor._argdest('inv_transform_name'),
                      help='Inverse transform name')
    args.add_argument(args_descriptor._argflag('inv-transform-subpath'),
                      dest=args_descriptor._argdest('inv_transform_subpath'),
                      help='Transform subpath')
    args.add_argument(args_descriptor._argflag('align-name'),
                      dest=args_descriptor._argdest('align_name'),
                      help='Alignment name')
    args.add_argument(args_descriptor._argflag('align-subpath'),
                      dest=args_descriptor._argdest('align_subpath'),
                      help='Alignment subpath')

    args.add_argument(args_descriptor._argflag('transform-blocksize'),
                      dest=args_descriptor._argdest('transform_blocksize'),
                      type=_inttuple,
                      help='Transform blocksize')
    args.add_argument(args_descriptor._argflag('inv-transform-blocksize'),
                      dest=args_descriptor._argdest('inv_transform_blocksize'),
                      type=_inttuple,
                      help='Inverse transform blocksize')
    args.add_argument(args_descriptor._argflag('align-blocksize'),
                      dest=args_descriptor._argdest('align_blocksize'),
                      type=_inttuple,
                      help='Alignment blocksize')

    args.add_argument(args_descriptor._argflag('registration-steps'),
                      dest=args_descriptor._argdest('registration_steps'),
                      type=_stringlist,
                      help='Registration steps')


def _extract_arg(args, args_descriptor, argname, args_dict):
    args_dict[argname] = getattr(args, args_descriptor._argdest(argname))


def _extract_registration_input_args(args, args_descriptor):
    registration_args = {}
    _extract_arg(args, args_descriptor, 'fix', registration_args)
    _extract_arg(args, args_descriptor, 'fix_subpath', registration_args)
    _extract_arg(args, args_descriptor, 'fix_spacing', registration_args)
    _extract_arg(args, args_descriptor, 'fix_mask', registration_args)
    _extract_arg(args, args_descriptor, 'fix_mask_subpath', registration_args)
    _extract_arg(args, args_descriptor, 'mov', registration_args)
    _extract_arg(args, args_descriptor, 'mov_subpath', registration_args)
    _extract_arg(args, args_descriptor, 'mov_spacing', registration_args)
    _extract_arg(args, args_descriptor, 'mov_mask', registration_args)
    _extract_arg(args, args_descriptor, 'mov_mask_subpath', registration_args)
    _extract_arg(args, args_descriptor, 'output_dir', registration_args)
    _extract_arg(args, args_descriptor, 'transform_name', registration_args)
    _extract_arg(args, args_descriptor, 'transform_subpath', registration_args)
    _extract_arg(args, args_descriptor, 'transform_blocksize', registration_args)
    _extract_arg(args, args_descriptor, 'inv_transform_name', registration_args)
    _extract_arg(args, args_descriptor, 'inv_transform_subpath', registration_args)
    _extract_arg(args, args_descriptor, 'inv_transform_blocksize', registration_args)
    _extract_arg(args, args_descriptor, 'align_name', registration_args)
    _extract_arg(args, args_descriptor, 'align_subpath', registration_args)
    _extract_arg(args, args_descriptor, 'align_blocksize', registration_args)
    _extract_arg(args, args_descriptor, 'registration_steps', registration_args)
    registration_inputs = _RegistrationInputs()
    registration_inputs.__dict__ = registration_args
    return registration_inputs


def _extract_align_pipeline(config_filename, context, steps):
    """
    config_filename:
    context: 'global_align' or 'local_align'
    """

    default_config = yaml.safe_load(default_bigstream_config_str)
    if config_filename:
        with open(config_filename) as f:
            external_config = yaml.safe_load(f)
            print(f'Read external config from {config_filename}: ',
                  external_config, flush= True)
            config = pu.deep_update(default_config, external_config)
            print(f'Final config {config}')
    else:
        config = default_config
    context_config = config[context]
    align_pipeline = []
    for step in steps:
        alg_args = config.get(step, {})
        context_alg_args = context_config.get(step, {})
        print(f'Default {step} args: {alg_args}')
        print(f'Context {step} args: {context_alg_args}')
        step_args = pu.deep_update(alg_args, context_alg_args)
        print(f'Final {step} args: {step_args}')
        align_pipeline.append((step, step_args))

    return align_pipeline


def _run_global_alignment(args, steps):
    print('Run global registration with:', args, steps, flush=True)
    # Read the global inputs
    print(f'Open fix vol {args.fix} {args.fix_subpath}',
            'for global registration',
            flush=True)
    fix_arraydata, fix_attrs = io_utility.open(args.fix, args.fix_subpath)
    print(f'Open moving vol {args.mov} {args.mov_subpath}',
            'for global registration',
            flush=True)
    mov_arraydata, mov_attrs = io_utility.open(args.mov, args.mov_subpath)
    # get voxel spacing for fix and moving volume
    if args.fix_spacing:
        fix_voxel_spacing = np.array(args.fix_spacing)[::-1] # xyz -> zyx
    else:
        fix_voxel_spacing = io_utility.get_voxel_spacing(fix_attrs)
    if args.mov_spacing:
        mov_voxel_spacing = np.array(args.mov_spacing)[::-1] # xyz -> zyx
    elif args.fix_spacing: # fix voxel spacing were specified - use the same for moving vol
        mov_voxel_spacing = fix_voxel_spacing
    else:
        mov_voxel_spacing = io_utility.get_voxel_spacing(mov_attrs)

    print('Global alignment - fixed volume attributes:',
          fix_arraydata.shape, fix_attrs, fix_voxel_spacing, flush=True)
    print('Global alignment - moving volume attributes:',
          mov_arraydata.shape, mov_attrs, mov_voxel_spacing, flush=True)

    if args.fix_mask:
        fix_maskarray, _ = io_utility.open(args.fix_mask,
                                           args.fix_mask_subpath)
    else:
        fix_maskarray = None

    if args.mov_mask:
        mov_maskarray, _ = io_utility.open(args.mov_mask,
                                           args.mov_mask_subpath)
    else:
        mov_maskarray = None

    transform, alignment = _align_global_data(
        fix_arraydata,
        mov_arraydata[...],
        fix_voxel_spacing,
        mov_voxel_spacing,
        steps,
        fix_maskarray,
        mov_maskarray)

    transform_file = args.transform_path()
    if transform_file:
        print('Save global transformation to', transform_file)
        np.savetxt(transform_file, transform)
    else:
        print('Skip saving global transformation')

    inv_transform_file = args.inv_transform_path()
    if inv_transform_file:
        try:
            inv_transform = np.linalg.inv(transform)
            print('Save global inverse transformation to', inv_transform_file)
            np.savetxt(inv_transform_file, inv_transform)
        except Exception:
            print('Global transformation', transform, 'is not invertible')

    else:
        print('Skip saving global inverse transformation')

    align_path = args.align_path()
    if align_path:
        if args.align_blocksize:
            align_blocksize = args.align_blocksize[::-1]
        else:
            align_blocksize = (128,) * alignment.ndim

        print('Save global aligned volume to', align_path,
              'with blocksize', align_blocksize)

        io_utility.create_dataset(
            align_path,
            args.align_dataset(),
            alignment.shape,
            align_blocksize,
            alignment.dtype,
            data=alignment,
            pixelResolution=list(mov_voxel_spacing),
            downsamplingFactors=mov_attrs.get('downsamplingFactors'),
        )
    else:
        print('Skip saving lowres aligned volume')

    return transform


def _align_global_data(fix_data,
                       mov_data,
                       fix_spacing,
                       mov_spacing,
                       steps,
                       fix_mask,
                       mov_mask):
    print('Run low res alignment:', steps, flush=True)
    affine = alignment_pipeline(fix_data,
                                mov_data,
                                fix_spacing,
                                mov_spacing,
                                steps,
                                fix_mask=fix_mask,
                                mov_mask=mov_mask)
    print('Apply affine transform', flush=True)
    # apply transform
    aligned = apply_transform(fix_data,
                              mov_data,
                              fix_spacing,
                              mov_spacing,
                              transform_list=[affine,])

    return affine, aligned


def _run_local_alignment(args, steps, global_transform, 
                         default_args=_RegistrationInputs(),
                         default_blocksize=128,
                         processing_size=None,
                         processing_overlap=0.5,
                         inv_iterations=10,
                         inv_sqrt_iterations=10,
                         inv_order=2,
                         dask_scheduler_address=None,
                         dask_config_file=None,
                         max_tasks=0,
                         ):
    print('Run local registration with:', args, steps, flush=True)

    # Read the highres inputs - if highres is not defined default it to lowres
    fix_path = args.fix if args.fix else default_args.fix
    mov_path = args.mov if args.mov else default_args.mov

    print(f'Open fix vol {fix_path}:{args.fix_subpath}',
            'for local registration',
            flush=True)
    fix_arraydata, fix_attrs = io_utility.open(fix_path, args.fix_subpath)
    print(f'Open mov vol {mov_path}:{args.mov_subpath}',
            'for local registration',
            flush=True)
    mov_arraydata, mov_attrs = io_utility.open(mov_path, args.mov_subpath)
    if mov_arraydata.ndim != fix_arraydata.ndim:
        # only check for ndim and not shape because as it happens 
        # the test data has different shape for fix.highres and mov.highres
        raise Exception(f'{mov_path}:{args.mov_subpath} expected to have ',
                        f'the same ndim as {fix_path}:{args.fix_subpath}')

    # get voxel spacing for fix and moving volume
    if args.fix_spacing:
        fix_voxel_spacing = np.array(args.fix_spacing)[::-1] # xyz -> zyx
    else:
        fix_voxel_spacing = io_utility.get_voxel_spacing(fix_attrs)
    if args.mov_spacing:
        mov_voxel_spacing = np.array(args.mov_spacing)[::-1] # xyz -> zyx
    elif args.fix_spacing: # fix voxel spacing were specified - use the same for moving vol
        mov_voxel_spacing = fix_voxel_spacing
    else:
        mov_voxel_spacing = io_utility.get_voxel_spacing(mov_attrs)

    print('Local alignment - fixed volume attributes:',
          fix_arraydata.shape, fix_attrs, fix_voxel_spacing, flush=True)
    print('Local alignment - moving volume attributes:',
          mov_arraydata.shape, mov_attrs, mov_voxel_spacing, flush=True)

    if args.fix_mask:
        fix_maskarray, _ = io_utility.open(args.fix_mask,
                                           args.fix_mask_subpath)
    else:
        fix_maskarray = None

    if args.mov_mask:
        mov_maskarray, _ = io_utility.open(args.mov_mask,
                                           args.mov_mask_subpath)
    else:
        mov_maskarray = None

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
    elif args.transform_blocksize:
        local_processing_size = args.transform_blocksize[::-1]
    else:
        # default
        local_processing_size = (default_blocksize,) * fix_arraydata.ndim

    local_processing_overlap_factor = (0.5 if (processing_overlap <= 0 or
                                               processing_overlap >= 1)
                                           else processing_overlap)

    if args.transform_subpath:
        transform_subpath = args.transform_subpath
    else:
        transform_subpath = args.mov_subpath

    if args.transform_blocksize:
        # block chunks are define as x,y,z so I am reversing it to z,y,x
        transform_blocksize = args.transform_blocksize[::-1]
    else:
        # default to output_chunk_size
        transform_blocksize = (default_blocksize,) * fix_arraydata.ndim

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
        fix_path, args.fix_subpath,
        fix_arraydata, fix_attrs, fix_voxel_spacing,
        fix_maskarray,
        mov_path, args.mov_subpath,
        mov_arraydata, mov_attrs, mov_voxel_spacing,
        mov_maskarray,
        steps,
        local_processing_size,
        local_processing_overlap_factor,
        [global_transform] if global_transform is not None else [],
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


def _align_local_data(fix_path, fix_subpath,
                      fix_arraydata, fix_attrs, fix_voxel_spacing,
                      fix_mask,
                      mov_path, mov_subpath,
                      mov_arraydata, mov_attrs, mov_voxel_spacing,
                      mov_mask,
                      steps,
                      processing_size,
                      processing_overlap_factor,
                      global_transforms_list,
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

    fix_shape = fix_arraydata.shape
    fix_ndim = fix_arraydata.ndim
    mov_shape = mov_arraydata.shape

    print('Align moving data',
          mov_path, mov_subpath, mov_shape, mov_voxel_spacing,
          'to reference',
          fix_path, fix_subpath, fix_shape, fix_voxel_spacing,
          flush=True)

    if transform_path:
        transform = io_utility.create_dataset(
            transform_path,
            transform_subpath,
            fix_shape + (fix_ndim,),
            transform_blocksize + (fix_ndim,),
            np.float32,
            # use the voxel spacing from the fix image
            pixelResolution=list(fix_voxel_spacing),
            downsamplingFactors=fix_attrs.get('downsamplingFactors'),
        )
    else:
        transform = None
    print('Calculate transformation', transform_path, 'for local alignment of',
          mov_path, mov_subpath,
          'to reference',
          fix_path, fix_subpath,
          flush=True)
    deform_ok = distributed_alignment_pipeline(
        fix_arraydata, mov_arraydata,
        fix_voxel_spacing, mov_voxel_spacing,
        steps,
        processing_size, # parallelize on processing size
        cluster_client,
        overlap_factor=processing_overlap_factor,
        fix_mask=fix_mask,
        mov_mask=mov_mask,
        static_transform_list=global_transforms_list,
        output_transform=transform,
        max_tasks=cluster_max_tasks,
    )
    if deform_ok and transform and inv_transform_path:
        inv_transform = io_utility.create_dataset(
            inv_transform_path,
            inv_transform_subpath,
            fix_shape + (fix_ndim,),
            inv_transform_blocksize + (fix_ndim,),
            np.float32,
            # use the voxel spacing from the fix image
            pixelResolution=list(fix_voxel_spacing),
            downsamplingFactors=fix_attrs.get('downsamplingFactors'),
        )
        print('Calculate inverse transformation',
              f'{inv_transform_path}:{inv_transform_subpath}',
              'from', 
              f'{transform_path}:{transform_subpath}',
              'for local alignment of',
              f'{mov_path}:{mov_subpath}',
              'to reference',
              f'{fix_path}:{fix_subpath}',
              flush=True)
        distributed_invert_displacement_vector_field(
            transform,
            fix_voxel_spacing,
            inv_transform_blocksize, # use blocksize for partitioning the work
            inv_transform,
            cluster_client,
            overlap_factor=processing_overlap_factor,
            iterations=inv_iterations,
            sqrt_order=inv_order,
            sqrt_iterations=inv_sqrt_iterations,
            max_tasks=cluster_max_tasks,
        )

    if deform_ok and align_path:
        # Apply local transformation only if 
        # highres aligned output name is set
        align = io_utility.create_dataset(
            align_path,
            align_subpath,
            fix_shape,
            align_blocksize,
            fix_arraydata.dtype,
            pixelResolution=list(mov_voxel_spacing),
            downsamplingFactors=mov_attrs.get('downsamplingFactors'),
        )
        print('Apply', 
              f'{transform_path}:{transform_subpath}',              
              'to warp',
              f'{mov_path}:{mov_subpath}',
              '->',
              f'{align_path}:{align_subpath}',
              flush=True)
        distributed_apply_transform(
            fix_arraydata, mov_arraydata,
            fix_voxel_spacing, mov_voxel_spacing,
            align_blocksize, # use block chunk size for distributing work
            global_transforms_list + [transform], # transform_list
            cluster_client,
            overlap_factor=processing_overlap_factor,
            aligned_data=align,
            max_tasks=cluster_max_tasks,
        )
    else:
        align = None

    return transform, align


def main():
    global_descriptor = _ArgsHelper('global')
    local_descriptor = _ArgsHelper('local')
    args_parser = _define_args(global_descriptor, local_descriptor)
    args = args_parser.parse_args()
    print('Invoked registration:', args, flush=True)

    global_inputs = _extract_registration_input_args(args, global_descriptor)
    global_transform = None

    if args.global_use_existing_transform:
        global_transform_file = global_inputs.transform_path()
        if global_transform_file and exists(global_transform_file):
            print('Read global transform from', global_transform_file, flush=True)
            global_transform = np.loadtxt(global_transform_file)

    if not global_transform:
        if global_inputs.registration_steps:
            global_steps = _extract_align_pipeline(args.align_config,
                                                   'global_align',
                                                   global_inputs.registration_steps)
            global_transform = _run_global_alignment(
                global_inputs,
                global_steps,
            )
        else:
            print('Skip global alignment because no global steps were specified.')

    local_inputs = _extract_registration_input_args(args, local_descriptor)

    if args.local_registration_steps:
        local_steps = _extract_align_pipeline(args.align_config,
                                              'local_align',
                                              args.local_registration_steps)
        _run_local_alignment(
            local_inputs,
            local_steps,
            global_transform,
            default_args=global_inputs,
            processing_size=args.local_processing_size,
            processing_overlap=args.local_processing_overlap_factor,
            inv_iterations=args.inv_iterations,
            inv_sqrt_iterations=args.inv_sqrt_iterations,
            inv_order=args.inv_order,
            dask_scheduler_address=args.dask_scheduler,
            dask_config_file=args.dask_config,
            max_tasks=args.cluster_max_tasks,
        )
    else:
        print('Skip local alignment because no local steps were specified.')


if __name__ == '__main__':
    main()