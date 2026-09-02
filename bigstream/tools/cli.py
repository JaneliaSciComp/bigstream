import json
import logging
import numpy as np
import os
import pydantic.v1.utils as pu
import re
import yaml

from pathlib import Path
from typing import Optional, Tuple

from bigstream.configure_bigstream import default_bigstream_config_str
from bigstream.image_data import ImageData
from bigstream.io_utility import get_voxel_spacing
from bigstream.ome_utils import get_spatial_values


logger = logging.getLogger(__name__)


def dictfromjson(arg:str):
    if arg:
        return json.loads(arg)
    else:
        return {}


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

    def get_mov_origin_transform(self):
        if self.mov_origin_transform and os.path.exists(self.mov_origin_transform):
            logger.info(f'Read moving image origin transform from {self.mov_origin_transform}')
            return np.loadtxt(self.mov_origin_transform)
        elif self.mov_origin_transform:
            logger.warning(f'Initial transform file not found: {transform_path}')
        return None

    def get_static_transforms(self):
        return get_transforms(self.static_transforms, expansion_factor=self.fix_expansion_factor)

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

    def get_align_timeindex(self):
        if self.align_timeindex:
            return self.align_timeindex
        else:
            return self.mov_timeindex

    def get_align_channel(self):
        if self.align_channel:
            return self.align_channel
        else:
            return self.mov_channel


def define_registration_input_args(args, args_descriptor: CliArgsHelper):
    args.add_argument(args_descriptor.argflag('fix'),
                      dest=args_descriptor.argdest('fix'),
                      help='Fixed volume')
    args.add_argument(args_descriptor.argflag('fix-subpath'),
                      dest=args_descriptor.argdest('fix_subpath'),
                      help='Fixed volume subpath')
    args.add_argument(args_descriptor.argflag('fix-timeindex'),
                      dest=args_descriptor.argdest('fix_timeindex'),
                      type=int,
                      default=None,
                      help='Fixed volume time index')
    args.add_argument(args_descriptor.argflag('fix-channel'),
                      dest=args_descriptor.argdest('fix_channel'),
                      type=int,
                      default=None,
                      help='Fixed volume channel')
    args.add_argument(args_descriptor.argflag('fix-spacing'),
                      dest=args_descriptor.argdest('fix_spacing'),
                      type=floattuple,
                      help='Fixed volume voxel spacing')
    args.add_argument(args_descriptor.argflag('fix-expansion'),
                      dest=args_descriptor.argdest('fix_expansion_factor'),
                      type=float,
                      default=1.0,
                      help='Fixed volume expansion factor')

    args.add_argument(args_descriptor.argflag('fix-mask'),
                      dest=args_descriptor.argdest('fix_mask'),
                      help='Fixed volume mask')
    args.add_argument(args_descriptor.argflag('fix-mask-subpath'),
                      dest=args_descriptor.argdest('fix_mask_subpath'),
                      help='Fixed volume mask subpath')
    args.add_argument(args_descriptor.argflag('roi'),
                      dest=args_descriptor.argdest('roi'),
                      type=floattuple,
                      metavar="xmin,ymin,zmin[,xmax,ymax,zmax]",
                      help='Registration ROI on the fixed image, a tuple of 6 values '
                           'representing min and max physical coordinates (xyz). '
                           'Restricts the region that is actually registered.')
    args.add_argument(args_descriptor.argflag('fix-mask-roi'),
                      dest=args_descriptor.argdest('fix_mask_roi'),
                      type=floattuple,
                      metavar="xmin,ymin,zmin[,xmax,ymax,zmax]",
                      help='Fix mask ROI as a tuple of 6 values '
                           'representing min and max voxel coordinates (xyz). '
                           'Restricts the fix image mask for the initial transformation.')
    args.add_argument(args_descriptor.argflag('mov-mask-roi'),
                      dest=args_descriptor.argdest('mov_mask_roi'),
                      type=floattuple,
                      metavar="xmin,ymin,zmin[,xmax,ymax,zmax]",
                      help='Moving mask ROI as a tuple of 6 values '
                           'representing min and max voxel coordinates (xyz). '
                           'Restricts the moving image mask for the initial transformation.')
    args.add_argument(args_descriptor.argflag('foreground-percentage'),
                      dest=args_descriptor.argdest('foreground_percentage'),
                      type=float,
                      default=0.0,
                      help='Signal percentage per block in order for the block to be considered')
    args.add_argument(args_descriptor.argflag('mov'),
                      dest=args_descriptor.argdest('mov'),
                      help='Moving volume')
    args.add_argument(args_descriptor.argflag('mov-subpath'),
                      dest=args_descriptor.argdest('mov_subpath'),
                      help='Moving volume subpath')
    args.add_argument(args_descriptor.argflag('mov-timeindex'),
                      dest=args_descriptor.argdest('mov_timeindex'),
                      type=int,
                      default=0,
                      help='Moving volume time index')
    args.add_argument(args_descriptor.argflag('mov-channel'),
                      dest=args_descriptor.argdest('mov_channel'),
                      type=int,
                      default=-1, # last channel
                      help='Fixed volume channel')
    args.add_argument(args_descriptor.argflag('mov-spacing'),
                      dest=args_descriptor.argdest('mov_spacing'),
                      type=floattuple,
                      help='Moving volume voxel spacing')
    args.add_argument(args_descriptor.argflag('mov-expansion'),
                      dest=args_descriptor.argdest('mov_expansion_factor'),
                      type=float,
                      default=1.0,
                      help='Moving volume expansion factor')

    args.add_argument(args_descriptor.argflag('mov-mask'),
                      dest=args_descriptor.argdest('mov_mask'),
                      help='Moving volume mask')
    args.add_argument(args_descriptor.argflag('mov-mask-subpath'),
                      dest=args_descriptor.argdest('mov_mask_subpath'),
                      help='Moving volume mask subpath')

    args.add_argument(args_descriptor.argflag('mov-origin-transform'),
                      dest=args_descriptor.argdest('mov_origin_transform'),
                      type=str,
                      help='Moving image origin transform applied before computing the alignment - this will be incorporated in the transform result')
    args.add_argument(args_descriptor.argflag('static-transforms'),
                      dest=args_descriptor.argdest('static_transforms'),
                      type=stringlist,
                      help='Static transforms applied before computing the alignment that are not incorporated in the final transform result')
    args.add_argument(args_descriptor.argflag('persist-mov-origin-transform'),
                      dest=args_descriptor.argdest('persist_mov_origin_transform'),
                      action='store_true',
                      help='Persist initial transform with aligned result metadata')

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
    args.add_argument(args_descriptor.argflag('align-timeindex'),
                      dest=args_descriptor.argdest('align_timeindex'),
                      type=int,
                      default=None,
                      help='Aligned volume time index')
    args.add_argument(args_descriptor.argflag('align-channel'),
                      dest=args_descriptor.argdest('align_channel'),
                      type=int,
                      default=None,
                      help='Aligned volume channel')

    args.add_argument(args_descriptor.argflag('processing-size'),
                      dest=args_descriptor.argdest('processing_size'),
                      type=inttuple,
                      help='Output blocksize')
    args.add_argument(args_descriptor.argflag('processing-overlap-factor'),
                      dest=args_descriptor.argdest('processing_overlap_factor'),
                      type=float,
                      default=0.,
                      help='Processing overlap factor - a fractional number between 0 and 1 that specifies the percentage overlap')

    args.add_argument(args_descriptor.argflag('output-blocksize'),
                      dest=args_descriptor.argdest('output_blocksize'),
                      type=inttuple,
                      default=(128,128,128),
                      help='Output blocksize')
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

    args.add_argument(args_descriptor.argflag('prealign-steps'),
                      dest=args_descriptor.argdest('prealign_steps'),
                      type=stringlist,
                      help='Pre-registration steps')
    args.add_argument(args_descriptor.argflag('registration-steps'),
                      dest=args_descriptor.argdest('registration_steps'),
                      type=stringlist,
                      help='Registration steps')


def get_algorithm_parameters(config_filename, context, steps):
    """
    config_filename:
    context: 'global_align' or 'local_align'
    """
    logger.info(f'Extract {context} pipeline configuration from {config_filename} for steps: {steps}')
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
    # a missing section (e.g. an optional 'global_prealign') is treated as empty
    # rather than raising KeyError, so callers can request it unconditionally
    context_config = config.get(context, {})
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
    _extract_arg(args, args_descriptor, 'fix_timeindex', registration_args)
    _extract_arg(args, args_descriptor, 'fix_channel', registration_args)
    _extract_arg(args, args_descriptor, 'fix_spacing', registration_args)
    _extract_arg(args, args_descriptor, 'fix_expansion_factor', registration_args)
    _extract_arg(args, args_descriptor, 'fix_mask', registration_args)
    _extract_arg(args, args_descriptor, 'fix_mask_subpath', registration_args)
    _extract_arg(args, args_descriptor, 'roi', registration_args)
    _extract_arg(args, args_descriptor, 'fix_mask_roi', registration_args)
    _extract_arg(args, args_descriptor, 'mov_mask_roi', registration_args)
    _extract_arg(args, args_descriptor, 'foreground_percentage', registration_args)
    _extract_arg(args, args_descriptor, 'mov', registration_args)
    _extract_arg(args, args_descriptor, 'mov_subpath', registration_args)
    _extract_arg(args, args_descriptor, 'mov_timeindex', registration_args)
    _extract_arg(args, args_descriptor, 'mov_channel', registration_args)
    _extract_arg(args, args_descriptor, 'mov_spacing', registration_args)
    _extract_arg(args, args_descriptor, 'mov_expansion_factor', registration_args)
    _extract_arg(args, args_descriptor, 'mov_mask', registration_args)
    _extract_arg(args, args_descriptor, 'mov_mask_subpath', registration_args)
    _extract_arg(args, args_descriptor, 'default_output_dir', registration_args)
    _extract_arg(args, args_descriptor, 'processing_size', registration_args)
    _extract_arg(args, args_descriptor, 'processing_overlap_factor', registration_args)
    _extract_arg(args, args_descriptor, 'output_blocksize', registration_args)
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
    _extract_arg(args, args_descriptor, 'align_timeindex', registration_args)
    _extract_arg(args, args_descriptor, 'align_channel', registration_args)
    _extract_arg(args, args_descriptor, 'align_blocksize', registration_args)
    _extract_arg(args, args_descriptor, 'prealign_steps', registration_args)
    _extract_arg(args, args_descriptor, 'registration_steps', registration_args)
    _extract_arg(args, args_descriptor, 'mov_origin_transform', registration_args)
    _extract_arg(args, args_descriptor, 'static_transforms', registration_args)
    _extract_arg(args, args_descriptor, 'persist_mov_origin_transform', registration_args)
    registration_inputs = RegistrationInputs()
    registration_inputs.__dict__.update(registration_args)
    return registration_inputs


def _extract_arg(args: RegistrationInputs, args_descriptor: CliArgsHelper,
                 argname: str, args_dict: dict[str, any]):
    args_dict[argname] = getattr(args, args_descriptor.argdest(argname))


def _roi_to_zyx(roi):
    """
    Reverse a user-provided ROI box from xyz to zyx order.

    The min triple and the max triple are reversed independently - this is not
    a plain roi[::-1], which would also swap min and max.

    roi : (xmin, ymin, zmin[, xmax, ymax, zmax]) or None/empty
    Returns a 3- or 6-tuple in zyx order, or None if roi is falsy.
    """
    mn = tuple(reversed(roi[:3]))                    # (zmin, ymin, xmin)
    mx = tuple(reversed(roi[3:6])) if len(roi) >= 6 else None
    return mn + mx if mx is not None else mn


def get_input_images(reg_args: RegistrationInputs) -> Tuple[ImageData, Optional[ImageData], # fix, fix_mask
                                                            ImageData, Optional[ImageData], # mov, mov_mask
                                                            Optional[Tuple[float]], # roi
                                                            Optional[Tuple[float]], # fix_mask_roi
                                                            Optional[Tuple[float]], # mov_mask_roi
                                                            ]:
    # Read the global inputs
    fix = ImageData(
        reg_args.fix, reg_args.fix_subpath,
        image_timeindex=reg_args.fix_timeindex,
        image_channel=reg_args.fix_channel,
        expansion_factor=reg_args.fix_expansion_factor,
        open_image=True,
    )
    logger.info(f'Open fix vol {fix} for registration')
    mov = ImageData(
        reg_args.mov, reg_args.mov_subpath,
        image_timeindex=reg_args.mov_timeindex,
        image_channel=reg_args.mov_channel,
        expansion_factor=reg_args.mov_expansion_factor,
        open_image=True,
    )
    logger.info(f'Open moving vol {mov} for registration')
    # get voxel spacing for fix and moving volume
    if reg_args.fix_spacing:
        fix.voxel_spacing = reg_args.fix_spacing[::-1] # xyz -> zyx
    logger.info(f'Fix volume attributes: {fix.shape} {fix.attrs} {fix.voxel_spacing}')

    if reg_args.mov_spacing:
        mov.voxel_spacing = reg_args.mov_spacing[::-1] # xyz -> zyx
    elif reg_args.fix_spacing: # fix voxel spacing were specified - use the same for moving vol
        mov.voxel_spacing = fix.voxel_spacing
    logger.info(f'Mov volume attributes: {mov.shape} {mov.attrs} {mov.voxel_spacing}')

    # masks are not mandatory so check if the file exists first
    # but if the container exists then the subpath must be valid
    if reg_args.fix_mask and Path(reg_args.fix_mask).exists():
        fix_mask = ImageData(reg_args.fix_mask, reg_args.fix_mask_subpath, open_image=True)
        logger.info(f'Using fix mask {fix_mask}')
    else:
        fix_mask = None
        logger.info('No fix mask was provided')

    if reg_args.mov_mask and Path(reg_args.mov_mask).exists():
        mov_mask = ImageData(reg_args.mov_mask, reg_args.mov_mask_subpath, open_image=True)
        logger.info(f'Using mov mask {mov_mask}')
    else:
        mov_mask = None
        logger.info('No mov mask was provided')

    if reg_args.roi is not None:
        # reverse the user ROI (xyz) to zyx so downstream code consumes it directly
        roi = _roi_to_zyx(reg_args.roi)
        logger.info(f'Using registration ROI {reg_args.roi} (xyz) -> {roi} (zyx)')
    else:
        roi = None

    if reg_args.fix_mask_roi is not None:
        fix_mask_roi = _roi_to_zyx(reg_args.fix_mask_roi)
        logger.info(f'Using fix mask ROI {reg_args.fix_mask_roi} (xyz) -> {fix_mask_roi} (zyx)')
    else:
        fix_mask_roi = None

    if reg_args.mov_mask_roi is not None:
        mov_mask_roi = _roi_to_zyx(reg_args.mov_mask_roi)
        logger.info(f'Using fix mask ROI {reg_args.mov_mask_roi} (xyz) -> {mov_mask_roi} (zyx)')
    else:
        mov_mask_roi = None

    return fix, fix_mask, mov, mov_mask, roi, fix_mask_roi, mov_mask_roi


def get_transforms(transforms_locations, expansion_factor=1):
    """
    Parse transforms_locations which is a comma separated list of path, subpath pairs.
    The path and subpath are separated by a '~' and the subpath is optional if the path is non empty.
    To read the transformation check the path if it exists first.
    If it exists and is a file read it as an affine transform
    otherwise if it's a directory assume it's zarr and read it as a deformation field

    For each transform a matching entry is appended to transforms_spacings:
    - an affine transform gets None (spacing is irrelevant for an affine matrix)
    - a deformation field gets the spatial voxel spacing read from the OME-ZARR
      metadata (scaled by the expansion factor). If the container is a plain zarr
      without OME spacing metadata, None is appended.
    """
    transforms = []
    transforms_spacings = []
    if not transforms_locations:
        return transforms, tuple(transforms_spacings)

    if isinstance(transforms_locations, str):
        transforms_locations = transforms_locations.split(',')

    for location in transforms_locations:
        location = location.strip()
        if not location:
            continue
        # split the optional subpath
        location_parts = re.split(r'(~|\^)', location)
        if len(location_parts) > 1:
            path = location_parts[0].strip()
            subpath = location_parts[2].strip()
        else:
            path = location_parts[0].strip()
            subpath = None

        transform, transform_spacing = get_transform(path, subpath, expansion_factor=expansion_factor)

        if transform is None:
            continue

        transforms.append(transform)
        transforms_spacings.append(transform_spacing)

    return transforms, tuple(transforms_spacings)


def get_transform(transform_path, transform_subpath, expansion_factor=1):
    if not transform_path or not os.path.exists(transform_path):
        logger.warning(f'Transform location {transform_path} does not exist - skipping')
        return None, None

    if os.path.isfile(transform_path):
        # a plain file holds an affine transform matrix - transform_subpath is ignored
        logger.info(f'Read affine transform from {transform_path}')
        return np.loadtxt(transform_path), None
    else:
        # a directory is typically a zarr container holding a deformation field
        logger.info(f'Read deformation field from {transform_path}:{transform_subpath}')
        deformfield = ImageData(transform_path, transform_subpath,
                                open_image=True,
                                expansion_factor=expansion_factor)
        # spacing is only meaningful if the container carries OME/N5 metadata;
        # a plain zarr without spacing metadata yields None
        raw_spacing = get_voxel_spacing(deformfield.attrs) if deformfield.attrs else None
        if raw_spacing is not None:
            # a deformation field stores its spatial axes first (z, y, x) followed
            # by the displacement/vector axis, so take the leading spatial values
            deform_spacing = get_spatial_values(raw_spacing, reverse_axes=True) / expansion_factor
        else:
            deform_spacing = None
        logger.info(f'Deformation field {transform_path} spacing: {deform_spacing}')
        return deformfield, deform_spacing
