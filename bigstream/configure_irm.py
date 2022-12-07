import os, psutil
import SimpleITK as sitk


def configure_irm(
    metric='MMI',
    optimizer='RSGD',
    sampling='NONE',
    interpolator='1',
    shrink_factors=(1,),
    smooth_sigmas=(0,),
    metric_args={},
    optimizer_args={},
    sampling_percentage=None,
    exhaustive_step_sizes=None,
    callback=None,
):
    """
    Wrapper exposing the itk::simple::ImageRegistrationMethod API
    Rarely called by the user. Typically used in custom registration functions.
    See SimpleITK documentation for itk::simple::ImageRegistrationMethod for details.

    Parameters
    ----------
    metric : string (default: 'MMI')
        The image matching function optimized during alignment
        Some options have required metric_args. See SimpleITK documentation for
        itk::simple::ImageRegistrationMethod and look for SetMetricAs* functions
        Options:
            'ANC':   AntsNiehborhoodCorrelation
            'C':     Correlation
            'D':     Demons
            'JHMI':  JointHistogramMutualInformation
            'MMI':   MattesMutualInformation
            'MS':    MeanSquares

    optimizer : string (default: 'RSGD')
        Optimization algorithm used to improve metric and update transform
        Some options have required optimizer_args. See SimpleITK documentation for
        itk::simple::ImageRegistrationMethod and look for SetOptimizerAs* functions
        Options:
            'A':        Amoeba
            'CGLS':     ConjugateGradientLineSearch
            'E':        Exhaustive
            'GD':       GradientDescent
            'GDLS':     GradientDescentLineSearch
            'LBFGS2':   LimitedMemoryBroydenFletcherGoldfarbShannon w/o bounds
            'LBFGSB':   LimitedMemoryBroydenFletcherGoldfardShannon w/ simple bounds
            'OPOE':     OnePlueOneEvolutionary
            'P':        Powell
            'RSGD':     RegularStepGradientDescent

    sampling : string (default: 'NONE')
        How image intensities are sampled in space during metric calculation
        'REGULAR' and 'RANDOM' options influenced by 'sampling_percentage'
        Options:
            'NONE':     All voxels are used, values from voxel centers
            'REGULAR':  Regular spacing between samples, small random perturbation from voxel centers
            'RANDOM':   Sample positions are totally random

    interpolator : string (default: '1')
        Interpolation function used to compute image values at non-voxel center locations
        See SimpleITK documentation for itk:simple Namespace Reference and search for
        InterpolatorEnum
        Options:
            '0':    NearestNeighbor,
            '1':    Linear,
            'CBS':   CubicBSpline,
            'G':    Gaussian,
            'LG':   LabelGaussian,
            'HWS':  HammingWindowedSinc,
            'CWS':  CosineWindowedSinc,
            'WWS':  WelchWindowedSinc,
            'LWS':  LanczosWindowedSinc,
            'BWS':  BlackmanWindowedSinc,

    shrink_factors : tuple of int (default: (1,))
        Downsampling scale levels at which to optimize

    smooth_sigmas : tuple of float (default: (0,))
        Sigma of Gaussian used to smooth each scale level image
        Must be same length as `shrink_factors`
        Should be specified in physical units, e.g. mm or um

    metric_args : dict (default: {})
        Based on choice of metric, some additional arguments may be required.
        Pass arguments to the metric function through this dictionary.
        See itk::simple::ImageRegistrationMethod and look for SetMetricAs*
        functions for valid arguments.

    optimizer_args : dict (default: {})
        Based on choice of optimizer, some additional arguments may be required.
        Pass arguments to the optimizer function through this dictionary.
        See itk::simple::ImageRegistrationMethod and look for SetOptimizerAs*
        functions for valid arguments.

    sampling_percentage : float in range [0., 1.] (default: None)
        Required if sampling is 'REGULAR' or 'RANDOM'
        Percentage of voxels used during metric sampling

    exhaustive_step_sizes : tuple of float (default: None)
        Required of optimizer is 'EXHAUSTIVE'
        Grid search step sizes for each parameter in the transform

    callback : callable object, e.g. function (default: None)
        A function run at every iteration of optimization
        Should take only the ImageRegistrationMethod object as input: `irm`
        If None then the Level, Iteration, and Metric values are
        printed at each iteration

    Returns
    -------
    irm : itk::simple::ImageRegistrationMethod object
        The configured ImageRegistrationMethod object. Simply needs
        images and a transform type to be ready for optimization.
    """

    # identify number of cores available, assume hyperthreading
    if "LSB_DJOB_NUMPROC" in os.environ:
        ncores = int(os.environ["LSB_DJOB_NUMPROC"])
    else:
        ncores = psutil.cpu_count(logical=False)

    # initialize IRM object, be completely sure nthreads is set
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2*ncores)
    irm = sitk.ImageRegistrationMethod()
    irm.SetNumberOfThreads(2*ncores)

    # interpolator switch
    interpolator_switch = {
        '0':sitk.sitkNearestNeighbor,
        '1':sitk.sitkLinear,
        'CBS':sitk.sitkBSpline,
        'G':sitk.sitkGaussian,
        'LG':sitk.sitkLabelGaussian,
        'HWS':sitk.sitkHammingWindowedSinc,
        'CWS':sitk.sitkCosineWindowedSinc,
        'WWS':sitk.sitkWelchWindowedSinc,
        'LWS':sitk.sitkLanczosWindowedSinc,
        'BWS':sitk.sitkBlackmanWindowedSinc,
    }
    irm.SetInterpolator(interpolator_switch[interpolator])

    # metric switch
    metric_switch = {
        'ANC':irm.SetMetricAsANTSNeighborhoodCorrelation,
        'C':irm.SetMetricAsCorrelation,
        'D':irm.SetMetricAsDemons,
        'JHMI':irm.SetMetricAsJointHistogramMutualInformation,
        'MMI':irm.SetMetricAsMattesMutualInformation,
        'MS':irm.SetMetricAsMeanSquares,
    }
    metric_switch[metric](**metric_args)

    # sampling switch
    sampling_switch = {
        'NONE':irm.NONE,
        'REGULAR':irm.REGULAR,
        'RANDOM':irm.RANDOM,
    }
    irm.SetMetricSamplingStrategy(sampling_switch[sampling])
    if sampling in ('REGULAR', 'RANDOM'):
        irm.SetMetricSamplingPercentage(sampling_percentage)

    # optimizer switch
    optimizer_switch = {
        'A':irm.SetOptimizerAsAmoeba,
        'CGLS':irm.SetOptimizerAsConjugateGradientLineSearch,
        'E':irm.SetOptimizerAsExhaustive,
        'GD':irm.SetOptimizerAsGradientDescent,
        'GDLS':irm.SetOptimizerAsGradientDescentLineSearch,
        'LBFGS2':irm.SetOptimizerAsLBFGS2,
        'LBFGSB':irm.SetOptimizerAsLBFGSB,
        'OPOE':irm.SetOptimizerAsOnePlusOneEvolutionary,
        'P':irm.SetOptimizerAsPowell,
        'RSGD':irm.SetOptimizerAsRegularStepGradientDescent,
    }
    optimizer_switch[optimizer](**optimizer_args)
    irm.SetOptimizerScalesFromPhysicalShift()
    if optimizer == 'E':
        irm.SetOptimizerScales(exhaustive_step_sizes)

    # set pyramid
    irm.SetShrinkFactorsPerLevel(shrink_factors)
    irm.SetSmoothingSigmasPerLevel(smooth_sigmas)
    irm.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # set callback function
    if callback is None:
        def callback(irm):
            level = irm.GetCurrentLevel()
            iteration = irm.GetOptimizerIteration()
            metric = irm.GetMetricValue()
            print("LEVEL: ", level, " ITERATION: ", iteration, " METRIC: ", metric, flush=True)
    irm.AddCommand(sitk.sitkIterationEvent, lambda: callback(irm))

    # return configured irm
    return irm

