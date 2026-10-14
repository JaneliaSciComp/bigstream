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
)
from .configure_elastix import (
    build_elastix_parameter_object, configure_elastix_threads,
)


logger = logging.getLogger(__name__)


def _robust_normalize(arr, stats=None, p_low=1.0, p_high=99.0):
    """
    Clip to the [p_low, p_high] percentile range and rescale to [0, 1].

    Used to make the final before/after metric check scale-invariant: a plain shared
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
    # float32 throughout: this feeds a mean-of-squares comparison between two
    # [0, 1] images, where float32 is far more precision than the decision
    # needs, and a float64 temporary of a whole block is a large part of this
    # function's peak memory.
    out = np.subtract(arr, lo, dtype=np.float32)
    out /= denom
    return np.clip(out, 0.0, 1.0, out=out)


FINAL_METRICS = ('mi', 'abs_ncc', 'ssd')

# fewer valid voxels than this cannot support a meaningful before/after
# comparison of two images
MIN_VALID_VOXELS_FOR_CHECK = 1000


def _alignment_score(fix_vals, mov_vals, method, bins=32):
    """
    Similarity of two intensity vectors; higher is better.

    Both vectors hold the same voxels, already normalized to [0, 1].

    method == 'mi'      mutual information (nats) of the joint `bins` x `bins`
                        histogram. It is the same family of metric elastix
                        optimizes, and it does not assume that fix and mov
                        have the same (or even a monotone) contrast
                        relationship.
    method == 'abs_ncc' |Pearson correlation|: sign agnostic, so inverted
                        contrast scores as well as matching contrast, but it
                        assumes a roughly linear relation.
    method == 'ssd'     negative mean squared difference. Only meaningful when
                        fix and mov have the same contrast: for inverted
                        contrast a better alignment has a LARGER SSD.
    """
    if method == 'mi':
        fi = np.minimum((fix_vals * bins).astype(np.intp), bins - 1)
        mi_ = np.minimum((mov_vals * bins).astype(np.intp), bins - 1)
        joint = np.bincount(
            fi * bins + mi_, minlength=bins * bins,
        ).reshape(bins, bins).astype(np.float64)
        pxy = joint / joint.sum()
        px = pxy.sum(axis=1, keepdims=True)
        py = pxy.sum(axis=0, keepdims=True)
        nz = pxy > 0
        return float(np.sum(pxy[nz] * np.log(pxy[nz] / (px @ py)[nz])))
    if method == 'abs_ncc':
        fc = fix_vals.astype(np.float64)
        mc = mov_vals.astype(np.float64)
        fc -= fc.mean()
        mc -= mc.mean()
        denom = np.sqrt(np.dot(fc, fc) * np.dot(mc, mc))
        return float(abs(np.dot(fc, mc)) / denom) if denom > 0 else 0.0
    if method == 'ssd':
        return -float(np.mean((fix_vals - mov_vals) ** 2))
    raise ValueError(
        f'unknown final_metric {method!r}; expected one of {FINAL_METRICS}'
    )


def _final_metric_check(
    fix,
    mov,
    after_transform,
    after_interpolator,
    fix_mask,
    method,
    tolerance,
    context,
    label,
):
    """
    Decide whether a registration result may be kept: it must not score worse
    than the starting point by more than `tolerance`.

    Parameters
    ----------
    fix, mov : sitk.Image
        The images elastix registered. `mov` may be on its own grid; it is
        resampled onto `fix` for the "before" score.

    after_transform : sitk.Transform
        The result, mapping fix points to mov points.

    after_interpolator : int
        sitk interpolator used to warp `mov` with `after_transform`.

    fix_mask : sitk.Image or None
        Fix mask on the fix grid. The comparison only uses voxels inside it.

    method : str
        One of `FINAL_METRICS`.

    tolerance : float
        The result is rejected when `after < before - tolerance`.

    label : str
        'affine' or 'deform', for the log messages.

    Returns
    -------
    keep : bool
        True when the result is acceptable.

    Notes
    -----
    Only voxels where the fix mask is set and where `mov` actually covers the
    fix voxel, both before and after, are scored, so the zero padding written
    outside the moving image never contributes, and the before and after
    scores always cover exactly the same voxels. The images are normalized
    independently (percentile clipping, see `_robust_normalize`) using
    statistics of those voxels, with mov's statistics shared between before
    and after.
    """
    ndim = fix.GetDimension()
    identity_tx = sitk.Transform(ndim, sitk.sitkIdentity)

    mov_on_fix = sitk.Resample(mov, fix, identity_tx, sitk.sitkLinear, 0.0)
    warped = sitk.Resample(mov, fix, after_transform, after_interpolator, 0.0)

    ones = sitk.Image(mov.GetSize(), sitk.sitkFloat32)
    ones.CopyInformation(mov)
    ones = ones + 1.0
    covered = (
        sitk.GetArrayViewFromImage(
            sitk.Resample(ones, fix, identity_tx, sitk.sitkLinear, 0.0)
        ) > 0.5
    ) & (
        sitk.GetArrayViewFromImage(
            sitk.Resample(ones, fix, after_transform, sitk.sitkLinear, 0.0)
        ) > 0.5
    )
    del ones
    if fix_mask is not None:
        valid = covered & (sitk.GetArrayViewFromImage(fix_mask) > 0)
    else:
        valid = covered
    nvalid = int(valid.sum())
    valid_fraction = nvalid / valid.size
    if nvalid < MIN_VALID_VOXELS_FOR_CHECK:
        logger.warning((
            f'{context} Elastix {label} result cannot be evaluated: only '
            f'{nvalid} valid voxels ({valid_fraction:.2%} of the block) '
            f'in the fix mask and covered by mov'
        ))
        return False

    fix_raw = sitk.GetArrayViewFromImage(fix)[valid]
    before_raw = sitk.GetArrayViewFromImage(mov_on_fix)[valid]
    after_raw = sitk.GetArrayViewFromImage(warped)[valid]
    del valid, covered, mov_on_fix, warped

    fix_vals = _robust_normalize(fix_raw)
    mov_stats = tuple(np.percentile(before_raw, [1.0, 99.0]))
    before_vals = _robust_normalize(before_raw, stats=mov_stats)
    after_vals = _robust_normalize(after_raw, stats=mov_stats)

    before = _alignment_score(fix_vals, before_vals, method)
    after = _alignment_score(fix_vals, after_vals, method)
    # a sign agnostic reference alongside whichever method decides
    ncc = (
        ''
        if method == 'abs_ncc' else
        f', |NCC| {_alignment_score(fix_vals, before_vals, "abs_ncc"):.4f} '
        f'-> {_alignment_score(fix_vals, after_vals, "abs_ncc"):.4f}'
    )
    detail = (
        f'({method}, higher is better): initial: {before}, final: {after}'
        f'{ncc}; valid voxels {nvalid} ({valid_fraction:.2%})'
    )
    if after < before - tolerance:
        logger.warning(
            f'{context} Elastix {label} optimization failed to improve '
            f'metric {detail}'
        )
        return False
    logger.info(f'{context} Elastix {label} align succeeded: {detail}')
    return True


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
    fix_background=None,
    mov_background=None,
    static_transform_list=[],
    default=None,
    final_metric_check=True,
    final_metric='mi',
    final_metric_tolerance=0.0,
    context='',
    keep_logs=False,
    **align_args,
):
    if final_metric_check and final_metric not in FINAL_METRICS:
        raise ValueError(
            f'unknown final_metric {final_metric!r}; '
            f'expected one of {FINAL_METRICS}'
        )

    # bound the elastix thread count; see configure_elastix
    nthreads = configure_elastix_threads(context=context)

    # format static transform data
    a, b = format_static_transform_data(
        static_transform_list, fix, fix_spacing, fix_origin,
    )
    static_transform_spacing = a
    static_transform_origin = b

    # realize masks
    fix_mask = realize_mask(fix, fix_mask, mask_percentile=fix_mask_percentile, background=fix_background, roi=fix_roi)
    logger.debug(f'{context} Realized fix mask shape {fix_mask.shape if fix_mask is not None else None}')
    mov_mask = realize_mask(mov, mov_mask, mask_percentile=mov_mask_percentile, background=mov_background)
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
        if is_field:
            mov_interp = sitk.sitkBSpline
            logger.info(f'{context} composed static transforms: {T}')
        else:
            logger.info(f'{context} composed static transforms (deform field): {T}')
            mov_interp = sitk.sitkLinear
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
        f'Elastix registration parameters {dict(parameter_map_obj)}'
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
        del elastix
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

    # final metric check on the skip-sampled grid; see _final_metric_check
    if final_metric_check and not _final_metric_check(
            fix, mov, sitk_transform, sitk.sitkLinear, fix_mask,
            final_metric, final_metric_tolerance, context, 'affine',
        ):
        logger.info(f'{context} Elastix affine align returning default')
        return default

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
    fix_background=None,
    mov_background=None,
    static_transform_list=[],
    default=None,
    final_metric_check=True,
    final_metric='mi',
    final_metric_tolerance=0.0,
    context='',
    keep_logs=False,
    **align_args,
):
    if final_metric_check and final_metric not in FINAL_METRICS:
        raise ValueError(
            f'unknown final_metric {final_metric!r}; '
            f'expected one of {FINAL_METRICS}'
        )

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
    fix_mask = realize_mask(fix, fix_mask, mask_percentile=fix_mask_percentile, background=fix_background, roi=fix_roi)
    mov_mask = realize_mask(mov, mov_mask, mask_percentile=mov_mask_percentile, background=mov_background)

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
        f'Elastix registration parameters {dict(parameter_map_obj)}'
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
        resultTransformParameters = elastix.GetTransformParameterMaps()
        del elastix

        # densify the result transform to a displacement field on the
        # (skip-sampled) fixed grid; the output domain follows the transform
        # parameter map's own captured Size/Spacing/Origin/Direction (the fix
        # grid used above), regardless of the moving image passed in here.
        transformix = sitk.TransformixImageFilter()
        transformix.SetTransformParameterMaps(resultTransformParameters)
        transformix.SetMovingImage(fix)
        transformix.ComputeDeformationFieldOn()
        transformix.SetOutputDirectory(log_dir)
        transformix.LogToConsoleOn()
        transformix.Execute()
        disp = transformix.GetDeformationField()
        del transformix
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

    # final metric check on the skip-sampled grid; see _final_metric_check
    if final_metric_check:
        disp_tx = sitk.DisplacementFieldTransform(
            sitk.Cast(sitk.Image(disp), sitk.sitkVectorFloat64)
        )
        keep = _final_metric_check(
            fix, mov, disp_tx, sitk.sitkBSpline, fix_mask,
            final_metric, final_metric_tolerance, context, 'deform',
        )
        del disp_tx
        if not keep:
            logger.info(f'{context} Elastix deform align returning default')
            return default

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
    # resample straight to float32: the field is returned as float32 anyway, so
    # a float64 intermediate here cost 24 bytes/voxel for nothing
    disp_full = sitk.Resample(
        disp, ref, identity_tx, sitk.sitkLinear, 0.0, sitk.sitkVectorFloat32,
    )
    del disp, ref

    # convert from ITK/SITK xyz vector components to bigstream zyx convention
    field = np.ascontiguousarray(
        sitk.GetArrayViewFromImage(disp_full)[..., ::-1], dtype=np.float32,
    )
    del disp_full
    # `field` is contiguous, so this reshape is a view: unlike the bspline
    # coefficient vector `deformable_align` returns, a densified field has no
    # compact parameterization, and materializing a second flat copy of the
    # whole field doubled what this function returns for no caller (the
    # alignment_pipeline uses element [1]).
    params = field.reshape(-1)

    logger.info(f'{context} Elastix deform align succeeded')
    return params, field
