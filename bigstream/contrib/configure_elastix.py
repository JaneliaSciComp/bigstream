import logging
import os

import SimpleITK as sitk

import bigstream.utility as ut


logger = logging.getLogger(__name__)


def configure_elastix_threads(context=''):
    """
    Determine the thread count elastix should use.

    Mirrors ``configure_irm``'s thread logic and reads the same ``ITK_THREADS``
    environment variable (set per dask worker by ``set_cpu_resources`` from
    ``worker_cpus``). Callers apply the result via
    ``ElastixImageFilter.SetNumberOfThreads``, which -- per SimpleITK's own
    documentation -- also updates ITK's global thread default as a side
    effect, so it bounds the later ``TransformixImageFilter`` densify step too
    (that filter has no per-instance thread setter of its own).

    Returns
    -------
    nthreads : int
        The thread count callers should apply.
    """
    ncores = ut.get_number_of_cores() or 1
    if 'ITK_THREADS' in os.environ and os.environ['ITK_THREADS']:
        nthreads = int(os.environ['ITK_THREADS'])
    elif 'NO_HYPERTHREADING' in os.environ:
        nthreads = ncores
    else:
        nthreads = 2 * ncores

    logger.debug(f'{context} elastix threads set to {nthreads}')
    return nthreads


def _as_str_list(value):
    """Coerce a python value into the list-of-strings elastix expects."""
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def build_elastix_parameter_object(
    align_method='bspline',
    control_grid_spacing=None,
    context='',
    **align_args,
):
    # NumberOfResolutions must be given to GetDefaultParameterMap up front:
    # it determines the length of the auto-generated GridSpacingSchedule, so
    # setting it afterward as a verbatim override (below) leaves a schedule
    # sized for the default resolution count and elastix rejects the mismatch
    # ("Invalid GridSpacingSchedule!").
    number_of_resolutions = int(align_args.pop('NumberOfResolutions', 4))
    pm = sitk.GetDefaultParameterMap(align_method, number_of_resolutions)
    pm['NumberOfResolutions'] = _as_str_list(number_of_resolutions)
    # never write result images to disk from within a worker
    pm['WriteResultImage'] = ['false']

    if control_grid_spacing is not None:
        pm['FinalGridSpacingInPhysicalUnits'] = _as_str_list(
            [float(v) for v in control_grid_spacing]
        )

    # Use non-smoothing (shrinking) image pyramids instead of the elastix
    # default (FixedSmoothingImagePyramid/MovingSmoothingImagePyramid). The
    # smoothing pyramid runs RecursiveGaussianImageFilter at every level along
    # every axis, which hard-requires >=4 voxels along that axis regardless of
    # NumberOfResolutions -- a thin/edge block (common at domain boundaries)
    # then throws an unhelpful generic "Internal elastix error" and the whole
    # block silently falls back to the identity/zero field. Shrinking pyramids
    # just subsample (no anti-aliasing) and have no such minimum, at the cost
    # of slightly noisier coarse levels -- a better tradeoff than a hard crash.
    pm['FixedImagePyramid'] = ['FixedShrinkingImagePyramid']
    pm['MovingImagePyramid'] = ['MovingShrinkingImagePyramid']

    # verbatim overrides last
    for k, v in align_args.items():
        pm[k] = _as_str_list(v)

    logger.debug((
        f'{context} elastix parameter map: preset={align_method}, '
        f'align_args={align_args} '
    ))
    return pm
