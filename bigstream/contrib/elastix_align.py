import logging
import shutil
import tempfile

import numpy as np
import SimpleITK as sitk

import bigstream.transform as bst
import bigstream.utility as ut
from bigstream.align import (
    realize_mask,
    apply_alignment_spacing,
    images_to_sitk,
    format_static_transform_data,
    deform_field_diagnostics,
)
from .configure_elastix import (
    build_elastix_parameter_object, configure_elastix_threads,
)


logger = logging.getLogger(__name__)


def _robust_normalize(arr, stats=None, p_low=1.0, p_high=99.0):
    """
    Clip to the [p_low, p_high] percentile range and rescale to [0, 1].

    Used to make an SSD-based comparison scale-invariant: a plain shared
    rescale (e.g. dividing both images by the same constant) does NOT change
    which of two SSD values is smaller, since SSD(fix/s, mov/s) = SSD(fix,
    mov) / s**2 for any positive s -- the ordering is preserved. What DOES
    change the comparison is normalizing fix and mov INDEPENDENTLY (each using
    its own statistics) and clipping outliers, since that removes a difference
    in absolute intensity scale/exposure between fix and mov and stops a
    handful of very bright voxels from dominating the squared-error sum --
    both are common with raw, wide-dynamic-range (e.g. uint16) fluorescence
    data and can otherwise make a genuinely-improved MI-based registration
    look worse under raw-intensity SSD.

    Parameters
    ----------
    arr : ndarray
        The array to normalize.

    stats : tuple of (lo, hi) or None (default: None)
        Percentile values to use instead of computing them from `arr`. Pass
        the same `mov` image's stats when normalizing `warped` (a resample of
        `mov`), so the "before" and "after" comparison uses one consistent
        scale for that image rather than two independently (re)computed ones.

    p_low, p_high : float (default: 1.0, 99.0)
        Percentiles defining the clip range when `stats` is None.

    Returns
    -------
    normalized : ndarray
        `arr` clipped and rescaled to [0, 1].
    """
    if stats is None:
        lo, hi = np.percentile(arr, [p_low, p_high])
    else:
        lo, hi = stats
    denom = max(float(hi - lo), 1e-6)
    return np.clip((arr - lo) / denom, 0.0, 1.0)


def elastix_affine_align(
    fix,
    mov,
    fix_spacing,
    mov_spacing,
    align_method='affine',
    alignment_spacing=None,
    fix_mask=None,
    mov_mask=None,
    fix_roi=None,
    fix_mask_percentile=None,
    mov_mask_percentile=None,
    fix_origin=None,
    mov_origin=None,
    static_transform_list=[],
    default=None,
    final_metric_check=True,
    context='',
    keep_logs=False,
    **align_args,
):
    # bound the elastix thread count; see configure_elastix
    nthreads = configure_elastix_threads(context=context)

    # format static transform data
    a, b = format_static_transform_data(
        static_transform_list, fix, fix_spacing, fix_origin,
    )
    static_transform_spacing = a
    static_transform_origin = b

    # realize masks
    fix_mask = realize_mask(fix, fix_mask, mask_percentile=fix_mask_percentile, roi=fix_roi)
    logger.debug(f'{context} Realized fix mask shape {fix_mask.shape if fix_mask is not None else None}')
    mov_mask = realize_mask(mov, mov_mask, mask_percentile=mov_mask_percentile)
    logger.debug(f'{context} Realized mov mask shape {mov_mask.shape if mov_mask is not None else None}')

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

    # default identity affine (4x4 for 3D); returned as a matrix, not a field
    if default is None:
        default = np.eye(ndim + 1)

    identity_tx = sitk.Transform(ndim, sitk.sitkIdentity)

    # resample the fix mask onto the fix image grid (masks may be sampled
    # differently than the image they belong to)
    if fix_mask is not None:
        fix_mask = sitk.Resample(
            fix_mask, fix, identity_tx, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8,
        )

    # pre-warp mov (and its mask) by the static transforms so the returned affine
    # is the elastix correction ONLY (residual), consistent with affine_align.
    if static_transform_list:
        T = bst.transform_list_to_composite_transform(
            static_transform_list,
            static_transform_spacing,
            static_transform_origin,
        )
        is_field = any(len(t.shape) not in (1, 2) for t in static_transform_list)
        mov_interp = sitk.sitkBSpline if is_field else sitk.sitkLinear
        mov = sitk.Resample(mov, fix, T, mov_interp, 0.0)
        if mov_mask is not None:
            mov_mask = sitk.Resample(
                mov_mask, fix, T, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8,
            )
    elif mov_mask is not None:
        # resample the mov mask onto the moving image grid
        mov_mask = sitk.Resample(
            mov_mask, mov, identity_tx, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8,
        )

    parameter_map_obj = build_elastix_parameter_object(
        align_method,
        context=context,
        **align_args,
    )
    logger.debug((
        f'{context} '
        f'Elastix registration parameters {parameter_map_obj}'
    ))

    # elastix writes side-effect files (TransformParameters.0.txt, result.0.nii)
    # into its output directory regardless of WriteResultImage -- under dask many
    # workers would otherwise collide on the cwd, so point it at a private,
    # per-call temp directory and remove it afterward.
    log_dir = tempfile.mkdtemp(prefix=f'bigstream_{context}')
    logger.debug(f'{context} Log elastix affine registration to {log_dir}')
    try:
        elastix = sitk.ElastixImageFilter()
        elastix.SetFixedImage(fix)
        elastix.SetMovingImage(mov)
        elastix.SetParameterMap(parameter_map_obj)
        elastix.SetNumberOfThreads(nthreads)
        if fix_mask is not None:
            elastix.SetFixedMask(fix_mask)
        if mov_mask is not None:
            elastix.SetMovingMask(mov_mask)
        elastix.SetOutputDirectory(log_dir)
        elastix.LogToConsoleOn()
        elastix.Execute()
        # a linear transform is fully described by its parameter map; no
        # transformix densification into a displacement field is needed.
        result_transform_maps = elastix.GetTransformParameterMaps()
    except Exception as e:
        logger.error(f'{context} Registration failed due to elastix exception: {e}')
        logger.info(f'{context} Returning default')
        return default
    finally:
        if not keep_logs:
            shutil.rmtree(log_dir, ignore_errors=True)

    # reconstruct the elastix result as a sitk linear transform. Unlike the
    # deformable path, an affine lives in physical coordinates and is invariant
    # to the (skip-sampled) grid, so no resampling back to the original fix grid
    # is required.
    result_tx_map = result_transform_maps[-1]
    transform_type = result_tx_map['Transform'][0]
    logger.debug(f'{context} result transformation type: {result_tx_map}')
    tx_params = [float(v) for v in result_tx_map['TransformParameters']]
    center = (
        [float(v) for v in result_tx_map['CenterOfRotationPoint']]
        if 'CenterOfRotationPoint' in result_tx_map else [0.0] * ndim
    )
    try:
        if transform_type == 'TranslationTransform':
            # promote a pure translation to an affine (identity matrix + shift)
            sitk_transform = sitk.AffineTransform(ndim)
            sitk_transform.SetTranslation(tx_params)
        else:
            if transform_type == 'AffineTransform':
                sitk_transform = sitk.AffineTransform(ndim)
            elif transform_type == 'EulerTransform':
                sitk_transform = (
                    sitk.Euler3DTransform() if ndim == 3 else sitk.Euler2DTransform()
                )
            elif transform_type == 'SimilarityTransform':
                sitk_transform = (
                    sitk.Similarity3DTransform() if ndim == 3 else sitk.Similarity2DTransform()
                )
            else:
                logger.error((
                    f'{context} Unsupported elastix transform type '
                    f'{transform_type} for affine align; returning default'
                ))
                return default
            sitk_transform.SetParameters(tx_params)
            sitk_transform.SetFixedParameters(center)
    except Exception as e:
        logger.error((
            f'{context} Could not build affine from elastix result '
            f'({transform_type}): {e}'
        ))
        logger.info(f'{context} Returning default')
        return default

    # final metric check on the skip-sampled grid (SSD, lower is better); images
    # are independently, robustly normalized first (see _robust_normalize).
    if final_metric_check:
        fix_arr = sitk.GetArrayViewFromImage(fix).astype(np.float64)
        mov_arr = sitk.GetArrayViewFromImage(mov).astype(np.float64)
        # mov may live on its own grid (different shape/spacing than fix) unless
        # it was pre-warped onto the fix grid by the static transforms above --
        # e.g. an affine step with no static transforms. Resample it onto the
        # fix grid with the identity transform so the "before" SSD is computed on
        # the same grid as fix (and as the warped "after" image below); a direct
        # fix_arr - mov_arr would otherwise raise a shape-broadcast error.
        mov_on_fix = sitk.Resample(mov, fix, identity_tx, sitk.sitkLinear, 0.0)
        mov_on_fix_arr = sitk.GetArrayViewFromImage(mov_on_fix).astype(np.float64)
        fix_norm = _robust_normalize(fix_arr)
        # normalize with mov's own full-extent dynamic range (raw mov, no resample
        # padding), applied consistently to both the before and after images
        mov_stats = tuple(np.percentile(mov_arr, [1.0, 99.0]))
        mov_norm = _robust_normalize(mov_on_fix_arr, stats=mov_stats)
        initial_metric_value = float(np.mean((fix_norm - mov_norm) ** 2))
        warped = sitk.Resample(mov, fix, sitk_transform, sitk.sitkLinear, 0.0)
        warped_arr = sitk.GetArrayViewFromImage(warped).astype(np.float64)
        warped_norm = _robust_normalize(warped_arr, stats=mov_stats)
        final_metric_value = float(np.mean((fix_norm - warped_norm) ** 2))
        if final_metric_value > initial_metric_value:
            logger.warning((
                f'{context} Elastix affine optimization failed to improve metric '
                f'(normalized SSD): initial: {initial_metric_value}, '
                f'final: {final_metric_value}'
            ))
            logger.info(f'{context} Elastix affine align returning default')
            return default
        else:
            logger.info((
                f'{context} Elastix affine align succeeded: '
                f'(normalized SSD): initial: {initial_metric_value}, '
                f'final: {final_metric_value}'
            ))

    # convert the sitk transform to a bigstream 4x4 matrix (physical, zyx order)
    affine_matrix = bst.affine_transform_to_matrix(sitk_transform)
    logger.info((
        f'{context} Elastix affine align succeeded, '
        f'affine matrix: {affine_matrix} '
    ))
    return affine_matrix


def elastix_deformable_align(
    fix,
    mov,
    fix_spacing,
    mov_spacing,
    align_method='bspline',
    alignment_spacing=None,
    control_point_spacing=None,
    fix_mask=None,
    mov_mask=None,
    fix_roi=None,
    fix_mask_percentile=None,
    mov_mask_percentile=None,
    fix_origin=None,
    mov_origin=None,
    static_transform_list=[],
    default=None,
    final_metric_check=True,
    context='',
    keep_logs=False,
    **align_args,
):
    # bound the elastix thread count; see configure_elastix
    nthreads = configure_elastix_threads(context=context)

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
    fix_mask = realize_mask(fix, fix_mask, mask_percentile=fix_mask_percentile, roi=fix_roi)
    mov_mask = realize_mask(mov, mov_mask, mask_percentile=mov_mask_percentile)

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

    # default identity field
    if default is None:
        zero_field = np.zeros(initial_fix_shape + (ndim,), dtype=np.float32)
        default = (zero_field.ravel(), zero_field)

    identity_tx = sitk.Transform(ndim, sitk.sitkIdentity)
    # resample the fix mask onto the image grid
    if fix_mask is not None:
        fix_mask = sitk.Resample(
            fix_mask, fix, identity_tx, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8,
        )

    # pre-warp mov (and its mask) by the static transforms so the returned field
    # is the elastix correction ONLY (residual).
    if static_transform_list:
        T = bst.transform_list_to_composite_transform(
            static_transform_list,
            static_transform_spacing,
            static_transform_origin,
        )
        is_field = any(len(t.shape) not in (1, 2) for t in static_transform_list)
        mov_interp = sitk.sitkBSpline if is_field else sitk.sitkLinear
        mov = sitk.Resample(mov, fix, T, mov_interp, 0.0)
        if mov_mask is not None:
            mov_mask = sitk.Resample(
                mov_mask, fix, T, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8,
            )
    elif mov_mask is not None:
        # resample the mov mask onto the moving image grid
        mov_mask = sitk.Resample(
            mov_mask, mov, identity_tx, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8,
        )


    # final grid spacing: zyx -> xyz, supporting scalar / short / long arrays
    control_grid_xyz = None
    if control_point_spacing is not None:
        g = np.atleast_1d(np.asarray(control_point_spacing, dtype=float))
        if g.size > ndim:
            g = g[:ndim]
        elif g.size < ndim:
            g = np.concatenate([g, np.full(ndim - g.size, g[-1])])
        control_grid_xyz = list(g[::-1])

    parameter_map_obj = build_elastix_parameter_object(
        align_method,
        control_grid_spacing=control_grid_xyz,
        context=context,
        **align_args,
    )
    logger.debug((
        f'{context} '
        f'Elastix registration parameters {parameter_map_obj}'
    ))

    # Both filters write side-effect files (TransformParameters.0.txt,
    # result.0.nii, deformationField.nii) into their output directory
    # regardless of WriteResultImage -- under dask many workers would
    # otherwise collide on the cwd, so point both at a private, per-call temp
    # directory and remove it afterward.
    log_dir = tempfile.mkdtemp(prefix=f'bigstream_{context}')
    logger.debug(f'{context} Log elastix deform registration to {log_dir}')
    try:
        elastix = sitk.ElastixImageFilter()
        elastix.SetFixedImage(fix)
        elastix.SetMovingImage(mov)
        elastix.SetParameterMap(parameter_map_obj)
        elastix.SetNumberOfThreads(nthreads)
        if fix_mask is not None:
            elastix.SetFixedMask(fix_mask)
        if mov_mask is not None:
            elastix.SetMovingMask(mov_mask)
        elastix.SetOutputDirectory(log_dir)
        elastix.LogToConsoleOn()
        elastix.Execute()

        # densify the result transform to a displacement field on the
        # (skip-sampled) fixed grid; the output domain follows the transform
        # parameter map's own captured Size/Spacing/Origin/Direction (the fix
        # grid used above), regardless of the moving image passed in here.
        transformix = sitk.TransformixImageFilter()
        transformix.SetTransformParameterMaps(elastix.GetTransformParameterMaps())
        transformix.SetMovingImage(fix)
        transformix.ComputeDeformationFieldOn()
        transformix.SetOutputDirectory(log_dir)
        transformix.LogToConsoleOn()
        transformix.Execute()
        disp = transformix.GetDeformationField()
    except Exception as e:
        logger.error(f'{context} Registration failed due to elastix exception: {e}')
        logger.info(f'{context} Returning default')
        return default
    finally:
        if not keep_logs:
            shutil.rmtree(log_dir, ignore_errors=True)

    disp.SetSpacing(fix.GetSpacing())
    disp.SetOrigin(fix.GetOrigin())
    disp.SetDirection(fix.GetDirection())

    # final metric check on the skip-sampled grid (SSD, lower is better).
    # Images are independently, robustly normalized first: a plain shared
    # rescale would NOT change the comparison (SSD/s**2 preserves ordering for
    # any s), but raw wide-dynamic-range data (e.g. uint16 fluorescence) can
    # make SSD dominated by a few bright/outlier voxels or by an absolute
    # intensity-scale difference between fix and mov -- independent per-image
    # percentile normalization removes both effects. mov and warped (a
    # resample of mov) share mov's own stats so "before" and "after" use one
    # consistent scale.
    if final_metric_check:
        fix_arr = sitk.GetArrayViewFromImage(fix).astype(np.float64)
        mov_arr = sitk.GetArrayViewFromImage(mov).astype(np.float64)
        # mov may live on its own grid (different shape/spacing than fix) unless
        # it was pre-warped onto the fix grid by the static transforms above --
        # e.g. an affine step with no static transforms. Resample it onto the
        # fix grid with the identity transform so the "before" SSD is computed on
        # the same grid as fix (and as the warped "after" image below); a direct
        # fix_arr - mov_arr would otherwise raise a shape-broadcast error.
        mov_on_fix = sitk.Resample(mov, fix, identity_tx, sitk.sitkLinear, 0.0)
        mov_on_fix_arr = sitk.GetArrayViewFromImage(mov_on_fix).astype(np.float64)
        fix_norm = _robust_normalize(fix_arr)
        # normalize with mov's own full-extent dynamic range (raw mov, no resample
        # padding), applied consistently to both the before and after images
        mov_stats = tuple(np.percentile(mov_arr, [1.0, 99.0]))
        mov_norm = _robust_normalize(mov_on_fix_arr, stats=mov_stats)
        initial_metric_value = float(np.mean((fix_norm - mov_norm) ** 2))
        disp_tx = sitk.DisplacementFieldTransform(
            sitk.Cast(sitk.Image(disp), sitk.sitkVectorFloat64)
        )
        warped = sitk.Resample(mov, fix, disp_tx, sitk.sitkBSpline, 0.0)
        warped_arr = sitk.GetArrayViewFromImage(warped).astype(np.float64)
        warped_norm = _robust_normalize(warped_arr, stats=mov_stats)
        final_metric_value = float(np.mean((fix_norm - warped_norm) ** 2))
        if final_metric_value > initial_metric_value:
            logger.warning((
                f'{context} Elastix deform optimization failed to improve metric '
                f'(normalized SSD): initial: {initial_metric_value}, '
                f'final: {final_metric_value}'
            ))
            logger.info(f'{context} Elastix deform align returning default')
            return default
        else:
            logger.info((
                f'{context} Elastix deform align succeeded: '
                f'(normalized SSD): initial: {initial_metric_value}, '
                f'final: {final_metric_value}'
            ))

    # resample field back to the original (pre-alignment-spacing) fix grid so
    # the returned field always has shape initial_fix_shape + (ndim,)
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
        disp, ref, identity_tx, sitk.sitkLinear, 0.0, sitk.sitkVectorFloat64,
    )

    # convert from ITK/SITK xyz vector components to bigstream zyx convention
    field = sitk.GetArrayFromImage(disp_full).astype(np.float32)[..., ::-1]
    params = field.ravel().astype(np.float32)

    # diagnostics: check the deformation field for folding and discontinuities
    deform_field_diagnostics(field, initial_fix_spacing, context=context)

    logger.info(f'{context} Elastix deform align succeeded')
    return params, field
