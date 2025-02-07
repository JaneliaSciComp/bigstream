import logging
import pydantic.v1.utils as pu
import yaml

from .configure_bigstream import default_bigstream_config_str
from .image_data import ImageData


logger = logging.getLogger(__name__)


def inttuple(arg):
    if arg is not None and arg.strip():
        return tuple([int(d) for d in arg.split(',')])
    else:
        return ()


def intlist(arg):
    if arg is not None and arg.strip():
        return [int(d) for d in arg.split(',')]
    else:
        return []


def floattuple(arg):
    if arg is not None and arg.strip():
        return tuple([float(d) for d in arg.split(',')])
    else:
        return ()


def stringlist(arg):
    if arg is not None and arg.strip():
        return list(filter(lambda x: x, [s.strip() for s in arg.split(',')]))
    else:
        return []


class CliArgsHelper:

    def __init__(self, prefix):
        self._prefix = prefix

    def argflag(self, argname):
        return '--{}-{}'.format(self._prefix, argname)

    def argdest(self, argname):
        return '{}_{}'.format(self._prefix, argname)


class RegistrationInputs:

    def transform_path(self):
        output_dir = (self.transform_dir if self.transform_dir
                                         else self.default_output_dir)
        if output_dir and self.transform_name:
            return f'{output_dir}/{self.transform_name}'
        else:
            return None

    def inv_transform_path(self):
        output_dir = (self.transform_dir if self.transform_dir
                                         else self.default_output_dir)
        if output_dir and self.inv_transform_name:
            return f'{output_dir}/{self.inv_transform_name}'
        else:
            return None

    def align_path(self):
        output_dir = (self.align_dir if self.align_dir
                                     else self.default_output_dir)
        if output_dir and self.align_name:
            return f'{output_dir}/{self.align_name}'
        else:
            return None

    def align_dataset(self):
        if self.align_subpath:
            return self.align_subpath
        else:
            return self.mov_subpath


def define_registration_input_args(args, args_descriptor: CliArgsHelper):
    args.add_argument(args_descriptor.argflag('fix'),
                      dest=args_descriptor.argdest('fix'),
                      help='Fixed volume')
    args.add_argument(args_descriptor.argflag('fix-subpath'),
                      dest=args_descriptor.argdest('fix_subpath'),
                      help='Fixed volume subpath')
    args.add_argument(args_descriptor.argflag('fix-spacing'),
                      dest=args_descriptor.argdest('fix_spacing'),
                      type=floattuple,
                      help='Fixed volume voxel spacing')
    args.add_argument(args_descriptor.argflag('fix-mask'),
                      dest=args_descriptor.argdest('fix_mask'),
                      help='Fixed volume mask')
    args.add_argument(args_descriptor.argflag('fix-mask-subpath'),
                      dest=args_descriptor.argdest('fix_mask_subpath'),
                      help='Fixed volume mask subpath')
    args.add_argument(args_descriptor.argflag('fix-mask-descriptor'),
                      dest=args_descriptor.argdest('fix_mask_descriptor'),
                      type=inttuple,
                      help='Fixed volume mask descriptor')

    args.add_argument(args_descriptor.argflag('mov'),
                      dest=args_descriptor.argdest('mov'),
                      help='Moving volume')
    args.add_argument(args_descriptor.argflag('mov-subpath'),
                      dest=args_descriptor.argdest('mov_subpath'),
                      help='Moving volume subpath')
    args.add_argument(args_descriptor.argflag('mov-spacing'),
                      dest=args_descriptor.argdest('mov_spacing'),
                      type=floattuple,
                      help='Moving volume voxel spacing')
    args.add_argument(args_descriptor.argflag('mov-mask'),
                      dest=args_descriptor.argdest('mov_mask'),
                      help='Moving volume mask')
    args.add_argument(args_descriptor.argflag('mov-mask-subpath'),
                      dest=args_descriptor.argdest('mov_mask_subpath'),
                      help='Moving volume mask subpath')
    args.add_argument(args_descriptor.argflag('mov-mask-descriptor'),
                      dest=args_descriptor.argdest('mov_mask_descriptor'),
                      type=inttuple,
                      help='Moving volume mask descriptor')

    args.add_argument(args_descriptor.argflag('output-dir'),
                      dest=args_descriptor.argdest('default_output_dir'),
                      help='Default output directory')
    args.add_argument(args_descriptor.argflag('transform-dir'),
                      dest=args_descriptor.argdest('transform_dir'),
                      help='Transform output directory')
    args.add_argument(args_descriptor.argflag('transform-name'),
                      dest=args_descriptor.argdest('transform_name'),
                      help='Transform name')
    args.add_argument(args_descriptor.argflag('transform-subpath'),
                      dest=args_descriptor.argdest('transform_subpath'),
                      help='Transform subpath')
    args.add_argument(args_descriptor.argflag('inv-transform-name'),
                      dest=args_descriptor.argdest('inv_transform_name'),
                      help='Inverse transform name')
    args.add_argument(args_descriptor.argflag('inv-transform-subpath'),
                      dest=args_descriptor.argdest('inv_transform_subpath'),
                      help='Transform subpath')
    args.add_argument(args_descriptor.argflag('align-dir'),
                      dest=args_descriptor.argdest('align_dir'),
                      help='Alignment output directory')
    args.add_argument(args_descriptor.argflag('align-name'),
                      dest=args_descriptor.argdest('align_name'),
                      help='Alignment name')
    args.add_argument(args_descriptor.argflag('align-subpath'),
                      dest=args_descriptor.argdest('align_subpath'),
                      help='Alignment subpath')

    args.add_argument(args_descriptor.argflag('transform-blocksize'),
                      dest=args_descriptor.argdest('transform_blocksize'),
                      type=inttuple,
                      help='Transform blocksize')
    args.add_argument(args_descriptor.argflag('inv-transform-blocksize'),
                      dest=args_descriptor.argdest('inv_transform_blocksize'),
                      type=inttuple,
                      help='Inverse transform blocksize')
    args.add_argument(args_descriptor.argflag('align-blocksize'),
                      dest=args_descriptor.argdest('align_blocksize'),
                      type=inttuple,
                      help='Alignment blocksize')

    args.add_argument(args_descriptor.argflag('registration-steps'),
                      dest=args_descriptor.argdest('registration_steps'),
                      type=stringlist,
                      help='Registration steps')


def extract_align_pipeline(config_filename, context, steps):
    """
    config_filename:
    context: 'global_align' or 'local_align'
    """

    default_config = yaml.safe_load(default_bigstream_config_str)
    if config_filename:
        with open(config_filename) as f:
            external_config = yaml.safe_load(f)
            logger.info('Read external config from ' +
                        f'{config_filename}: {external_config}')
            config = pu.deep_update(default_config, external_config)
            logger.info(f'Final config {config}')
    else:
        config = default_config
    context_config = config[context]
    align_pipeline = []
    if steps and len(steps) > 0:
        # the steps are defined
        pipeline_steps = steps
    else:
        pipeline_steps = context_config.get('steps', [])
    for step in pipeline_steps:
        alg_args = config.get(step, {})
        context_alg_args = context_config.get(step, {})
        logger.info(f'Default {step} args: {alg_args}')
        logger.info(f'Context {step} overriden args: {context_alg_args}')
        step_args = pu.deep_update(alg_args, context_alg_args)
        logger.info(f'Final {step} args: {step_args}')
        align_pipeline.append((step, step_args))

    return align_pipeline, context_config


def extract_registration_input_args(args, args_descriptor: CliArgsHelper) -> RegistrationInputs:
    registration_args = {}
    _extract_arg(args, args_descriptor, 'fix', registration_args)
    _extract_arg(args, args_descriptor, 'fix_subpath', registration_args)
    _extract_arg(args, args_descriptor, 'fix_spacing', registration_args)
    _extract_arg(args, args_descriptor, 'fix_mask', registration_args)
    _extract_arg(args, args_descriptor, 'fix_mask_subpath', registration_args)
    _extract_arg(args, args_descriptor, 'fix_mask_descriptor', registration_args)
    _extract_arg(args, args_descriptor, 'mov', registration_args)
    _extract_arg(args, args_descriptor, 'mov_subpath', registration_args)
    _extract_arg(args, args_descriptor, 'mov_spacing', registration_args)
    _extract_arg(args, args_descriptor, 'mov_mask', registration_args)
    _extract_arg(args, args_descriptor, 'mov_mask_subpath', registration_args)
    _extract_arg(args, args_descriptor, 'mov_mask_descriptor', registration_args)
    _extract_arg(args, args_descriptor, 'default_output_dir', registration_args)
    _extract_arg(args, args_descriptor, 'transform_dir', registration_args)
    _extract_arg(args, args_descriptor, 'transform_name', registration_args)
    _extract_arg(args, args_descriptor, 'transform_subpath', registration_args)
    _extract_arg(args, args_descriptor, 'transform_blocksize', registration_args)
    _extract_arg(args, args_descriptor, 'inv_transform_name', registration_args)
    _extract_arg(args, args_descriptor, 'inv_transform_subpath', registration_args)
    _extract_arg(args, args_descriptor, 'inv_transform_blocksize', registration_args)
    _extract_arg(args, args_descriptor, 'align_dir', registration_args)
    _extract_arg(args, args_descriptor, 'align_name', registration_args)
    _extract_arg(args, args_descriptor, 'align_subpath', registration_args)
    _extract_arg(args, args_descriptor, 'align_blocksize', registration_args)
    _extract_arg(args, args_descriptor, 'registration_steps', registration_args)
    registration_inputs = RegistrationInputs()
    registration_inputs.__dict__ = registration_args
    return registration_inputs


def _extract_arg(args: RegistrationInputs, args_descriptor: CliArgsHelper,
                 argname: str, args_dict: dict[str, any]):
    args_dict[argname] = getattr(args, args_descriptor.argdest(argname))


def get_input_images(args: RegistrationInputs) -> tuple[ImageData]:
    # Read the global inputs
    fix = ImageData(args.fix, args.fix_subpath)
    logger.info(f'Open fix vol {fix} for registration')
    mov = ImageData(args.mov, args.mov_subpath)
    logger.info(f'Open moving vol {mov} for registration')
    # get voxel spacing for fix and moving volume
    if args.fix_spacing:
        fix.voxel_spacing = args.fix_spacing[::-1] # xyz -> zyx
    logger.info(f'Fix volume attributes: {fix.shape} {fix.attrs} {fix.voxel_spacing}')

    if args.mov_spacing:
        mov.voxel_spacing = args.mov_spacing[::-1] # xyz -> zyx
    elif args.fix_spacing: # fix voxel spacing were specified - use the same for moving vol
        mov.voxel_spacing = fix.voxel_spacing
    logger.info(f'Mov volume attributes: {mov.shape} {mov.attrs} {mov.voxel_spacing}')

    if args.fix_mask:
        fix_mask = ImageData(args.fix_mask, args.fix_mask_subpath)
    elif args.fix_mask_descriptor:
        fix_mask = args.fix_mask_descriptor
    else:
        fix_mask = None
    if args.mov_mask:
        mov_mask = ImageData(args.mov_mask, args.mov_mask_subpath)
    elif args.mov_mask_descriptor:
        mov_mask = args.mov_mask_descriptor
    else:
        mov_mask = None

    return (fix, fix_mask, mov, mov_mask)
