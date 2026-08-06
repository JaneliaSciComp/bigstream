import logging

import numpy as np
import SimpleITK as sitk

import bigstream.transform as bst
import bigstream.utility as ut
from bigstream.align import (
    realize_mask,
    apply_alignment_spacing,
    images_to_sitk,
    format_static_transform_data,
)


logger = logging.getLogger(__name__)


_DEMONS_VARIANTS = {
    'demons':         sitk.DemonsRegistrationFilter,
    'symmetric':      sitk.SymmetricForcesDemonsRegistrationFilter,
    'fast_symmetric': sitk.FastSymmetricForcesDemonsRegistrationFilter,
    'diffeomorphic':  sitk.DiffeomorphicDemonsRegistrationFilter,
}


def demons_align(
    fix,
    mov,
    fix_spacing,
    mov_spacing,
    iterations,
    smooth_sigmas,
    shrink_factors,
    alignment_spacing=None,
    fix_mask=None,
    mov_mask=None,
    fix_roi=None,
    fix_origin=None,
    mov_origin=None,
    static_transform_list=[],
    default=None,
    final_metric_check=True,
    context='',
    variant='diffeomorphic',
    field_smoothing_sigma=1.0,
    update_smoothing_sigma=0.0,
    smooth_displacement_field=True,
    smooth_update_field=False,
    max_rms_error=0.00,
    histogram_match=True,
    histogram_match_levels=1024,
    histogram_match_points=7,
    histogram_match_threshold_at_mean=True,
):
    """
    Register moving to fixed image with intensity-based Demons deformable registration.

    Uses a manual multi-resolution pyramid with the SimpleITK Demons filter family.
    Unlike deformable_align (BSpline via IRM), Demons operates directly on image
    pairs and returns a dense displacement field; there is no metric class, no
    optimizer, and no built-in mask interface.

    Parameters
    ----------
    fix : ndarray
        The fixed image.

    mov : ndarray
        The moving image; fix.ndim must equal mov.ndim.

    fix_spacing : 1d array
        Physical spacing between voxels of the fixed image.

    mov_spacing : 1d array
        Physical spacing between voxels of the moving image.

    iterations : list of int
        Number of iterations per pyramid level, ordered coarse -> fine.

    smooth_sigmas : list of float
        Gaussian smoothing sigma (physical units) per pyramid level.

    shrink_factors : list of int
        Downsampling factor per pyramid level (1 = no downsampling).
        Passing [1], [0], [N] is supported and bypasses the pyramid.

    alignment_spacing : float (default: None)
        Skip-sample fix and mov to approximately this voxel spacing before
        running the pyramid.

    fix_mask : ndarray, tuple, or callable (default: None)
        Foreground mask for the fixed image. Because Demons has no metric-mask
        interface, masking is approximated by zeroing fix/mov outside the mask.
        This biases Demons forces toward zero in masked-out regions and is NOT
        equivalent to IRM's SetMetricFixedMask.

    mov_mask : ndarray, tuple, or callable (default: None)
        Foreground mask for the moving image (same caveat as fix_mask).

    fix_origin : 1d array (default: None)
        Physical origin of the fixed image.

    mov_origin : 1d array (default: None)
        Physical origin of the moving image.

    static_transform_list : list of ndarray (default: [])
        Transforms applied to the moving image before Demons, analogous to IRM's
        SetMovingInitialTransform. The moving image is physically resampled by
        the composite of these transforms. The returned field is the Demons
        correction ONLY (residual on top of the static transforms), not the
        total displacement, so alignment_pipeline can compose steps correctly.

    default : any (default: None)
        Returned on failure. If None, a zero displacement field matching
        fix.shape + (ndim,) is used.

    final_metric_check : bool (default: True)
        Return default if the final SSD metric (lower is better) is worse than
        the metric measured before the final pyramid level.

    context : str (default: '')
        Prefix for log messages.

    variant : str (default: 'diffeomorphic')
        Demons filter variant: 'demons' | 'symmetric' | 'fast_symmetric' |
        'diffeomorphic'. Use 'diffeomorphic' to guarantee invertible warps,
        which matters for downstream spot warping that depends on the inverse
        field. For non-matched modalities use deformable_align with MMI instead
        -- Demons assumes intensity correspondence.

    field_smoothing_sigma : float (default: 1.0)
        Physical-unit standard deviation for displacement field regularization
        applied between iterations.

    update_smoothing_sigma : float (default: 0.0)
        Physical-unit standard deviation for update field smoothing; 0 disables.

    smooth_displacement_field : bool (default: True)
        Enable displacement field smoothing.

    smooth_update_field : bool (default: False)
        Enable update field smoothing.

    max_rms_error : float (default: 0.01)
        Convergence threshold (RMS displacement change between iterations).

    histogram_match : bool (default: True)
        Histogram-match mov to fix before running. Improves convergence for
        same-modality multi-round acquisitions but adds ~50 ms per block.

    histogram_match_levels : int (default: 1024)
        Number of histogram bins for matching.

    histogram_match_points : int (default: 7)
        Number of quantile-matched control points.

    histogram_match_threshold_at_mean : bool (default: True)
        Threshold histogram matching at mean intensity.

    Returns
    -------
    params : 1d array
        Flattened displacement field (field.ravel(), float32). Demons has no
        compact parameterization; params is exposed so callers that store
        params for caching still get a valid array.

    field : ndarray, shape fix.shape + (ndim,), float32
        Dense displacement field on the original (pre-alignment-spacing) fixed
        image grid, in physical units.
    """
    if len(iterations) != len(smooth_sigmas) or len(iterations) != len(shrink_factors):
        raise ValueError(
            f'iterations, smooth_sigmas, and shrink_factors must have equal length; '
            f'got {len(iterations)}, {len(smooth_sigmas)}, {len(shrink_factors)}'
        )

    # store original fixed image grid; the returned field lives on this grid
    initial_fix_shape = fix.shape
    initial_fix_spacing = fix_spacing
    initial_fix_origin = fix_origin

    # format static transform data
    a, b = format_static_transform_data(
        static_transform_list, fix, fix_spacing, fix_origin,
    )
    static_transform_spacing = a
    static_transform_origin = b

    # realize masks
    fix_mask = realize_mask(fix, fix_mask, roi=fix_roi)
    mov_mask = realize_mask(mov, mov_mask)

    # skip-sample and convert to SITK images (images_to_sitk casts to float32)
    X = apply_alignment_spacing(
        fix, mov,
        fix_mask, mov_mask,
        fix_spacing, mov_spacing,
        alignment_spacing,
        context=context,
    )
    fix, mov, fix_mask, mov_mask = images_to_sitk(*X, fix_origin, mov_origin)

    ndim = fix.GetDimension()

    # set default identity field
    if default is None:
        zero_field = np.zeros(initial_fix_shape + (ndim,), dtype=np.float32)
        default = (zero_field.ravel(), zero_field)

    # pre-warp mov by static transforms before running demons, analogous to
    # IRM's SetMovingInitialTransform. This keeps the returned field as the
    # demons correction ONLY (residual), not the total displacement, so that
    # alignment_pipeline can compose steps correctly without double-counting.
    initial_tx = sitk.Transform(ndim, sitk.sitkIdentity)
    if static_transform_list:
        T = bst.transform_list_to_composite_transform(
            static_transform_list,
            static_transform_spacing,
            static_transform_origin,
        )
        mov = sitk.Resample(mov, fix, T, sitk.sitkLinear, 0.0)
        if mov_mask is not None:
            mov_mask = sitk.Resample(mov_mask, fix, T, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)

    # optional histogram matching: match mov intensity distribution to fix
    if histogram_match:
        hm = sitk.HistogramMatchingImageFilter()
        hm.SetNumberOfHistogramLevels(histogram_match_levels)
        hm.SetNumberOfMatchPoints(histogram_match_points)
        hm.SetThresholdAtMeanIntensity(histogram_match_threshold_at_mean)
        mov = hm.Execute(mov, fix)

    def _resample_mask_to_image(mask, image):
        if mask is None:
            return None
        return sitk.Resample(
            mask, image, initial_tx,
            sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8,
        )

    def _zero_outside_mask(image, mask):
        if mask is None:
            return image
        image_arr = sitk.GetArrayFromImage(image)
        image_arr[sitk.GetArrayViewFromImage(mask) == 0] = 0.0
        masked_image = sitk.GetImageFromArray(image_arr)
        masked_image.CopyInformation(image)
        return masked_image

    # approximate masking: zero fix/mov outside masks before the pyramid.
    # Demons has no metric-mask interface; zeroing biases forces toward zero in
    # masked-out regions (not identical to IRM masking). Masks may have a
    # different sampling from the images, so first resample them to image grids.
    fix_mask = _resample_mask_to_image(fix_mask, fix)
    mov_mask = _resample_mask_to_image(mov_mask, mov)
    fix = _zero_outside_mask(fix, fix_mask)
    mov = _zero_outside_mask(mov, mov_mask)

    # zero initial displacement field; demons builds up from here
    disp = sitk.TransformToDisplacementField(
        initial_tx, sitk.sitkVectorFloat64,
        fix.GetSize(), fix.GetOrigin(), fix.GetSpacing(), fix.GetDirection(),
    )

    # baseline metric (SSD) before any pyramid level, for final_metric_check
    fix_np = sitk.GetArrayViewFromImage(fix).astype(np.float64)
    mov_np = sitk.GetArrayViewFromImage(mov).astype(np.float64)
    initial_metric_value = float(np.mean((fix_np - mov_np) ** 2))

    # multi-resolution pyramid loop, coarse -> fine
    demons_filter = None
    last_level_initial_metric = initial_metric_value
    for level, (shrink, sigma, n_iter) in enumerate(
        zip(shrink_factors, smooth_sigmas, iterations)
    ):
        # smooth fix and mov (sigma in physical units)
        if sigma > 0:
            smoother = sitk.SmoothingRecursiveGaussianImageFilter()
            smoother.SetSigma(float(sigma))
            f_l = smoother.Execute(fix)
            m_l = smoother.Execute(mov)
        else:
            f_l = fix
            m_l = mov

        # downsample to level resolution
        if shrink > 1:
            f_l = sitk.Shrink(f_l, [int(shrink)] * ndim)
            m_l = sitk.Shrink(m_l, [int(shrink)] * ndim)

        # resample current displacement field to level grid
        disp_l = sitk.Resample(
            disp, f_l, initial_tx, sitk.sitkLinear, 0.0, sitk.sitkVectorFloat64,
        )

        # record the metric just before the final level for metric check
        is_final_level = (level == len(shrink_factors) - 1)
        if final_metric_check and is_final_level:
            disp_tx = sitk.DisplacementFieldTransform(
                sitk.Cast(disp_l, sitk.sitkVectorFloat64)
            )
            warped = sitk.Resample(m_l, f_l, disp_tx, sitk.sitkLinear, 0.0)
            f_arr = sitk.GetArrayViewFromImage(f_l).astype(np.float64)
            w_arr = sitk.GetArrayViewFromImage(warped).astype(np.float64)
            last_level_initial_metric = float(np.mean((f_arr - w_arr) ** 2))

        # configure demons filter for this level
        demons = _DEMONS_VARIANTS[variant]()
        demons.SetNumberOfIterations(int(n_iter))
        demons.SetStandardDeviations(float(field_smoothing_sigma))
        demons.SetSmoothDisplacementField(smooth_displacement_field)
        demons.SetSmoothUpdateField(smooth_update_field)
        if update_smoothing_sigma > 0:
            demons.SetUpdateFieldStandardDeviations(float(update_smoothing_sigma))
        demons.SetMaximumRMSError(float(max_rms_error))

        def _make_iter_callback(df, lv, ctx):
            def _iter_callback():
                iteration = df.GetElapsedIterations()
                metric = df.GetMetric()
                logger.debug((
                    f'{ctx} LEVEL: {lv} '
                    f'ITERATION: {iteration} '
                    f'METRIC: {metric}'
                ))
            return _iter_callback

        demons.AddCommand(sitk.sitkIterationEvent, _make_iter_callback(demons, level, context))

        try:
            disp = demons.Execute(f_l, m_l, disp_l)
        except Exception as e:
            logger.error(f'{context} Demons registration failed at level {level}: {e}')
            logger.info(f'{context} Returning default')
            return default

        demons_filter = demons
        logger.info(
            f'{context} Demons level {level}: '
            f'shrink={shrink}, sigma={sigma}, iters={n_iter}, '
            f'metric={demons.GetMetric():.6f}'
        )

    # final metric check: SSD is lower-is-better; compare last level final vs initial
    if final_metric_check and demons_filter is not None:
        final_metric_value = demons_filter.GetMetric()
        if final_metric_value > last_level_initial_metric:
            logger.warning((
                f'{context} Demons align optimization failed to improve metric: '
                f'initial: {last_level_initial_metric}, '
                f'final: {final_metric_value} '
            ))
            logger.info(f'{context} Demons align returning default')
            return default

    # resample displacement field back to the original (pre-alignment-spacing) fix grid.
    # the alignment_spacing round-trip ensures the returned field always has shape
    # fix.shape + (ndim,), consistent with deformable_align.
    ref_origin = (
        np.asarray(initial_fix_origin, dtype=np.float64)
        if initial_fix_origin is not None
        else np.zeros(ndim)
    )
    ref = ut.numpy_to_sitk(
        np.zeros(initial_fix_shape, dtype=np.float32),
        np.asarray(initial_fix_spacing, dtype=np.float64),
        ref_origin,
    )
    disp_full = sitk.Resample(
        disp, ref, initial_tx, sitk.sitkLinear, 0.0, sitk.sitkVectorFloat64,
    )

    # convert from SITK XYZ vector components to bigstream ZYX convention
    field = sitk.GetArrayFromImage(disp_full).astype(np.float32)[..., ::-1]
    params = field.ravel().astype(np.float32)

    final_metric = demons_filter.GetMetric() if demons_filter is not None else float('nan')
    logger.info((
        f'{context} Demons align succeeded: '
        f'(initial_metric={initial_metric_value:.6f}, '
        f'final_metric={final_metric:.6f}) '
    ))
    return params, field
