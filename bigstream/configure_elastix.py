import itk
import logging
import os

import bigstream.utility as ut


logger = logging.getLogger(__name__)


def configure_elastix_threads(context=''):
    """
    Bound the number of threads the ``itk`` package uses for elastix.

    Mirrors ``configure_irm``'s thread logic and reads the same ``ITK_THREADS``
    environment variable (set per dask worker by ``set_cpu_resources`` from
    ``worker_cpus``). Note: SimpleITK and the ``itk`` package have independent
    global thread state, so ``configure_irm``'s SimpleITK setting does NOT bound
    elastix -- this sets the ``itk`` side explicitly.

    Returns
    -------
    nthreads : int
        The thread count applied, so callers can also set it per-object (e.g.
        ``ElastixRegistrationMethod.SetNumberOfWorkUnits``).
    """
    ncores = ut.get_number_of_cores() or 1
    if 'ITK_THREADS' in os.environ and os.environ['ITK_THREADS']:
        nthreads = int(os.environ['ITK_THREADS'])
    elif 'NO_HYPERTHREADING' in os.environ:
        nthreads = ncores
    else:
        nthreads = 2 * ncores

    itk.MultiThreaderBase.SetGlobalDefaultNumberOfThreads(nthreads)
    logger.debug(f'{context} elastix itk threads set to {nthreads}')
    return nthreads


# elastix parameter maps are dict[str, list[str]] -- every value is a list of
# strings. These presets are the transform names understood by
# itk.ParameterObject.GetDefaultParameterMap.
ELASTIX_PRESETS = ('translation', 'rigid', 'affine', 'bspline', 'spline', 'groupwise')

ELASTIX_METRIX = {
    'MSD': 'AdvancedMeanSquares',
    'NCC': 'AdvancedNormalizedCorrelation',
    'MI': 'AdvancedMattesMutualInformation',
    'NMI': 'NormalizedMutualInformation',
}

def _as_str_list(value):
    """Coerce a python value into the list-of-strings elastix expects."""
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def build_elastix_parameter_object(
    preset='bspline',
    number_of_resolutions=4,
    final_grid_spacing_xyz=None,
    metric='MI',
    maximum_iterations=256,
    number_of_spatial_samples=4096,
    number_of_histogram_bins=32,
    bending_energy_weight=1.0,
    context='',
    **extra_parameters,
):
    """
    Build an ``itk.ParameterObject`` for elastix deformable registration.

    Starts from the elastix default parameter map for ``preset`` and overrides
    the entries that bigstream surfaces as named arguments, then merges any
    ``extra_parameters`` verbatim (final say). All values are stored as lists of
    strings, as elastix requires; callers pass native python types.

    Parameters
    ----------
    preset : str or itk.ParameterObject (default: 'bspline')
        If a string, one of ``ELASTIX_PRESETS`` -- the elastix default map to
        start from, with the named-argument overrides below applied. If already
        an ``itk.ParameterObject`` it is returned as-is (full custom control;
        the named arguments are ignored).

    ndim : int (default: 3)
        Image dimensionality (kept for symmetry/validation; the default maps are
        dimension agnostic).

    number_of_resolutions : int (default: 4)
        Number of multi-resolution pyramid levels.

    final_grid_spacing_xyz : 1d array or None (default: None)
        B-spline control point grid spacing at the finest level, in physical
        units, in xyz order (already reversed from the bigstream zyx
        convention by the caller). If None, the preset default is used.

    metric : str (default: 'AdvancedMattesMutualInformation')
        elastix metric name.

    maximum_iterations : int (default: 256)
        Optimizer iterations per resolution.

    number_of_spatial_samples : int (default: 4096)
        Random image sampler count per resolution.

    number_of_histogram_bins : int (default: 32)
        Histogram bins for mutual-information metrics.

    bending_energy_weight : float (default: 1.0)
        Weight of the transform bending-energy penalty (only applied for the
        'bspline' preset, where it is the second metric term).

    context : str (default: '')
        Prefix for log messages.

    **extra_parameters
        Verbatim elastix parameter-map overrides merged last (final say), passed
        as keyword arguments, e.g. ``RandomSeed=42, ImageSampler='RandomCoordinate'``.
        Values may be native python types or lists; they are stringified.

    Returns
    -------
    parameter_object : itk.ParameterObject
    """
    # pass-through if the caller already built the object
    if isinstance(preset, itk.ParameterObject):
        return preset

    if preset not in ELASTIX_PRESETS:
        raise ValueError(
            f'Unknown elastix preset {preset!r}; expected one of {ELASTIX_PRESETS} '
            'or a prebuilt itk.ParameterObject'
        )

    parameter_object = itk.ParameterObject.New()

    # default map for the preset; pass final grid spacing (isotropic mean) to
    # the constructor then override with the exact per-axis vector below
    if final_grid_spacing_xyz is not None:
        grid_mean = float(sum(final_grid_spacing_xyz) / len(final_grid_spacing_xyz))
        pm = parameter_object.GetDefaultParameterMap(preset, number_of_resolutions, grid_mean)
        pm['FinalGridSpacingInPhysicalUnits'] = _as_str_list(
            [float(v) for v in final_grid_spacing_xyz]
        )
    else:
        pm = parameter_object.GetDefaultParameterMap(preset, number_of_resolutions)

    # common overrides
    resolved_metric = ELASTIX_METRIX.get(metric) or metric
    # bspline's default Metric is a two-term list (similarity + bending-energy
    # penalty); only replace the similarity (first) term so the regularizer
    # survives -- Metric0Weight/Metric1Weight below refer to these two terms.
    if preset == 'bspline':
        pm['Metric'] = [resolved_metric, 'TransformBendingEnergyPenalty']
        pm['Metric0Weight'] = ['1.0']
        pm['Metric1Weight'] = _as_str_list(float(bending_energy_weight))
    else:
        pm['Metric'] = _as_str_list(resolved_metric)
    pm['MaximumNumberOfIterations'] = _as_str_list(int(maximum_iterations))
    pm['NumberOfSpatialSamples'] = _as_str_list(int(number_of_spatial_samples))
    pm['NumberOfHistogramBins'] = _as_str_list(int(number_of_histogram_bins))
    # never write result images to disk from within a worker
    pm['WriteResultImage'] = ['false']

    # Use non-smoothing (shrinking) image pyramids instead of the elastix
    # default (FixedSmoothingImagePyramid/MovingSmoothingImagePyramid). The
    # smoothing pyramid runs RecursiveGaussianImageFilter at every level along
    # every axis, which hard-requires >=4 voxels along that axis regardless of
    # number_of_resolutions -- a thin/edge block (common at domain boundaries)
    # then throws an unhelpful generic "Internal elastix error" and the whole
    # block silently falls back to the identity/zero field. Shrinking pyramids
    # just subsample (no anti-aliasing) and have no such minimum, at the cost
    # of slightly noisier coarse levels -- a better tradeoff than a hard crash.
    pm['FixedImagePyramid'] = ['FixedShrinkingImagePyramid']
    pm['MovingImagePyramid'] = ['MovingShrinkingImagePyramid']

    # verbatim overrides last
    for k, v in extra_parameters.items():
        pm[k] = _as_str_list(v)

    parameter_object.AddParameterMap(pm)
    logger.debug((
        f'{context} elastix parameter object: preset={preset}, '
        f'resolutions={number_of_resolutions}, metric={metric}, '
        f'iterations={maximum_iterations}, samples={number_of_spatial_samples}, '
        f'final_grid_xyz={final_grid_spacing_xyz}'
    ))
    return parameter_object
