import bigstream.transform as bst
import cv2
import itertools
import numpy as np
import os
import SimpleITK as sitk
import bigstream.utility as ut
import logging

from bigstream.configure_irm import configure_irm
from bigstream.metrics import patch_mutual_information
from bigstream.metrics import local_correlation_coefficient
from bigstream import features

from fishspot.filter import apply_foreground_mask


logger = logging.getLogger(__name__)


def realize_mask(image, mask):
    """
    Ensure that mask is an ndarray

    Parameters
    ----------
    image : nd-array
        The image from which mask is derived

    mask : None, nd-array, tuple of floats, or function
        The mask data. If None, return None.
        If an nd-array, threshold at zero.
        If a tuple of floats, mask specified values.
        If a function, apply it.

    Returns
    -------
    A mask for image, which is either None or a binary nd-array
    dtype is always uint8
    """

    if mask is None:
        return None
    if isinstance(mask, np.ndarray):
        return (mask > 0).astype(np.uint8)
    if isinstance(mask, (tuple, list)):
        return np.isin(image, mask, invert=True).astype(np.uint8)
    if callable(mask):
        return mask(image).astype(np.uint8)


def apply_alignment_spacing(
    fix,
    mov,
    fix_mask,
    mov_mask,
    fix_spacing,
    mov_spacing,
    alignment_spacing,
    context=''
):
    """
    Skip sample all images to as close to alignment_spacing as possible
    Determine new voxel spacings

    Parameters
    ----------
    fix : nd-array
        The fixed image

    mov : nd-array
        The moving image

    fix_mask : nd-array
        The fixed image mask (can be None)
        Can have a different shape than fix, but assumed to have the same
        domain or field of view

    mov_mask : nd-array
        The moving image mask (can be None)
        Can have a different shape than mov, but assumed to have the same
        domain or field of view

    fix_spacing : 1d-array
        The fixed image voxel spacing

    mov_spacing : 1d-array
        The moving image voxel spacing

    Returns
    -------
    Returns 8 values in a tuple

    1. skip sampled fixed image
    2. skip sampled moving image
    3. skip sampled fix_mask (or None)
    4. skip sampled mov_mask (or None)
    5. spacing of skip sampled fixed image
    6. spacing of skip sampled moving image
    7. spacing of skip sampled fixed mask (or None)
    8. spacing of skip sampled moving mask (or None)
    """

    # ensure spacings are floating point
    fix_spacing = fix_spacing.astype(np.float64)
    mov_spacing = mov_spacing.astype(np.float64)

    # get mask spacings
    fix_mask_spacing = None
    if fix_mask is not None:
        fix_mask_spacing = ut.relative_spacing(fix_mask.shape,
                                               fix.shape,
                                               fix_spacing)
        logger.debug((
            f'{context} '
            f'Fix shape {fix.shape}, '
            f'Fix spacing {fix_spacing}, '
            f'Fix mask shape {fix_mask.shape} => '
            f'Fix mask spacing {fix_mask_spacing} '
        ))

    mov_mask_spacing = None
    if mov_mask is not None:
        mov_mask_spacing = ut.relative_spacing(mov_mask.shape,
                                               mov.shape,
                                               mov_spacing)
        logger.debug((
            f'{context} '
            f'Mov shape {mov.shape}, '
            f'Mov spacing {mov_spacing}, '
            f'Mov mask shape {mov_mask.shape} => '
            f'Mov mask spacing {mov_mask_spacing} '
        ))

    # skip sample
    if alignment_spacing:
        resampled_fix, resampled_fix_spacing = ut.skip_sample(fix, fix_spacing, alignment_spacing)
        logger.debug((
            f'{context} '
            f'Resampled fix {fix.shape} to {resampled_fix.shape} '
            f'spacing from {alignment_spacing}/{fix_spacing} to {resampled_fix_spacing} '
        ))

        resampled_mov, resampled_mov_spacing = ut.skip_sample(mov, mov_spacing, alignment_spacing)
        logger.debug((
            f'{context} '
            f'Resampled mov {mov.shape} to {resampled_mov.shape} '
            f'spacing from {alignment_spacing}/{mov_spacing} to {resampled_mov_spacing} '
        ))
        if fix_mask is not None:
            fix_mask, fix_mask_spacing = ut.skip_sample(
                fix_mask, fix_mask_spacing, alignment_spacing,
            )
            logger.debug((
                f'{context} '
                f'Resampled fix mask to {fix_mask.shape}, '
                f'new fix mask spacing is {fix_mask_spacing} '
            ))
        if mov_mask is not None:
            mov_mask, mov_mask_spacing = ut.skip_sample(
                mov_mask, mov_mask_spacing, alignment_spacing,
            )
            logger.debug((
                f'{context} '
                f'Resampled mov mask to {mov_mask.shape}, '
                f'new mov mask spacing is {mov_mask_spacing} '
            ))
    else:
        resampled_fix, resampled_fix_spacing = (fix, fix_spacing)
        resampled_mov, resampled_mov_spacing = (mov, mov_spacing)

    return (resampled_fix, resampled_mov, fix_mask, mov_mask,
            resampled_fix_spacing, resampled_mov_spacing, fix_mask_spacing, mov_mask_spacing,)


def images_to_sitk(
    fix,
    mov,
    fix_mask,
    mov_mask,
    fix_spacing,
    mov_spacing,
    fix_mask_spacing,
    mov_mask_spacing,
    fix_origin,
    mov_origin,
):
    """
    Convert all image inputs to SimpleITK image objects

    Parameters
    ----------
    fix : nd-array
        The fixed image

    mov : nd-array
        The moving image

    fix_mask : nd-array
        The fixed image mask (can be None)

    mov_mask : nd-array
        The moving image mask (can be None)

    fix_spacing : 1d-array
        The voxel spacing of the fixed image

    mov_spacing : 1d-array
        The voxel spacing of the moving image

    fix_mask_spacing : 1d-array
        The voxel spacing of the fixed image mask (can be None)
        fix and fix_mask are assumed to have the same domain,
        but this assumption can be slightly broken after skip_sampling

    mov_mask_spacing : 1d-array
        The voxel spacing of the moving image mask (can be None)
        mov and mov_mask are assumed to have the same domain,
        but this assumption can be slightly broken after skip_sampling

    Returns
    -------
    Returns 4 values in a tuple

    1. fix image as sitk.Image object
    2. mov image as sitk.Image object
    3. fix_mask as sitk.Image object (or None)
    4. mov_mask as sitk.Image object (or None)
    """

    fix = sitk.Cast(ut.numpy_to_sitk(
        fix, fix_spacing, origin=fix_origin), sitk.sitkFloat32)
    mov = sitk.Cast(ut.numpy_to_sitk(
        mov, mov_spacing, origin=mov_origin), sitk.sitkFloat32)
    if fix_mask is not None:
        fix_mask = ut.numpy_to_sitk(
            fix_mask, fix_mask_spacing, origin=fix_origin)
    if mov_mask is not None:
        mov_mask = ut.numpy_to_sitk(
            mov_mask, mov_mask_spacing, origin=mov_origin)
    return fix, mov, fix_mask, mov_mask


def _physical_bounding_box(image):
    """
    Physical (min, max) bounding box corners of an sitk image, in the
    image's own axis order (i.e. matching image.GetOrigin()/GetSpacing()).
    """
    origin = np.array(image.GetOrigin())
    spacing = np.array(image.GetSpacing())
    size = np.array(image.GetSize())
    far_corner = origin + (size - 1) * spacing
    return np.minimum(origin, far_corner), np.maximum(origin, far_corner)


def _check_and_correct_overlap(
    fix,
    mov,
    transform,
    min_overlap_fraction,
    context='',
):
    """
    Diagnose the physical overlap between `fix` and `mov` as seen through
    `transform` (the moving initial transform that will be handed to
    `ImageRegistrationMethod.SetMovingInitialTransform`, or None for
    identity), and log it. This overlap is what the registration metric
    actually samples; when it collapses to (near) zero, ITK raises
    "All samples map outside moving image buffer" and the caller falls
    back to its identity default.

    If the overlap fraction is at or below `min_overlap_fraction`, a
    corrective translation that re-centers fix's (transformed) bounding
    box on mov's bounding box is composed on top of `transform`, so
    registration has a chance to run instead of failing outright. This
    is a last-resort fallback, not a substitute for correct fix/mov
    origins or a correct static_transform_list -- it only kicks in when
    those would otherwise produce a hard failure.

    Parameters
    ----------
    fix : sitk.Image
        The fixed image, already at its final registration spacing/origin

    mov : sitk.Image
        The moving image, already at its final registration spacing/origin

    transform : sitk.Transform or None
        The moving initial transform that will be used, or None if none
        is being used (identity)

    min_overlap_fraction : float
        If the computed overlap fraction is at or below this value, the
        corrective translation fallback is applied

    context : string (default: '')
        Prefix for log messages

    Returns
    -------
    transform : sitk.Transform or None
        `transform` unchanged, or a new composite transform with a
        corrective translation applied on top if a correction was made
    """
    ndim = fix.GetDimension()
    fix_mins, fix_maxs = _physical_bounding_box(fix)
    mov_mins, mov_maxs = _physical_bounding_box(mov)

    corners = np.array(list(itertools.product(*zip(fix_mins, fix_maxs))))
    if transform is not None:
        mapped_corners = np.array(
            [transform.TransformPoint(tuple(c)) for c in corners]
        )
    else:
        mapped_corners = corners
    mapped_mins = mapped_corners.min(axis=0)
    mapped_maxs = mapped_corners.max(axis=0)

    inter_extent = np.clip(
        np.minimum(mapped_maxs, mov_maxs) - np.maximum(mapped_mins, mov_mins),
        0, None,
    )
    fix_extent = np.clip(mapped_maxs - mapped_mins, 1e-9, None)
    overlap_fraction = float(np.prod(inter_extent) / np.prod(fix_extent))

    logger.info((
        f'{context} overlap diagnostic: '
        f'fix bbox in moving space [{mapped_mins}, {mapped_maxs}], '
        f'mov bbox [{mov_mins}, {mov_maxs}], '
        f'overlap fraction: {overlap_fraction:.4f} '
    ))

    if min_overlap_fraction <= 0 or overlap_fraction > min_overlap_fraction:
        return transform

    delta = (mapped_mins + mapped_maxs) / 2 - (mov_mins + mov_maxs) / 2
    logger.warning((
        f'{context} insufficient overlap ({overlap_fraction:.4f} <= '
        f'{min_overlap_fraction}); applying fallback translation of '
        f'{tuple(-delta)} to re-center fix and mov before registration '
    ))
    correction = sitk.TranslationTransform(ndim, tuple(-delta))
    corrected = sitk.CompositeTransform(ndim)
    corrected.AddTransform(correction)
    if transform is not None:
        corrected.AddTransform(transform)
    return corrected


def format_static_transform_data(
    transforms,
    fix,
    fix_spacing,
    fix_origin,
):
    """
    Set transform_spacings and transform_origins explicitly

    Parameters
    ----------
    transforms : list of nd-arrays
        The list of static transforms

    fix : nd-array
        The fixed image

    fix_spacing : 1d-array
        The voxel spacing of the fixed image

    fix_origin : 1d-array
        The origin of the fixed image (can be None)

    Returns
    -------
    Returns 2 values in a tuple

    1. The tuple of transform spacings
    2. The tuple of transform origins
    """

    spacings = []
    for transform in transforms:
        spacing = fix_spacing
        if len(transform.shape) not in [1, 2]:
            spacing = ut.relative_spacing(transform.shape,
                                          fix.shape,
                                          fix_spacing)
        spacings.append(spacing)
    spacings = tuple(spacings)
    origins = (fix_origin,)*len(transforms)
    return (spacings, origins)


def _detect_and_match_feature_points(
    fix, mov,
    fix_spacing,
    mov_spacing,
    blob_sizes,
    safeguard_exceptions,
    alignment_spacing,
    num_sigma_max,
    cc_radius,
    nspots,
    match_threshold,
    max_spot_match_distance,
    point_matches_threshold,
    fix_spot_detection_kwargs,
    mov_spot_detection_kwargs,
    fix_spots,
    fix_spots_count_threshold,
    mov_spots,
    mov_spots_count_threshold,
    fix_mask,
    mov_mask,
    fix_origin,
    mov_origin,
    static_transform_list,
    context,
):
    """
    Detect (or reuse user-supplied) feature point spots in fix and mov,
    extract neighborhood contexts, and match corresponding points by
    neighborhood correlation. Shared by `feature_point_ransac_affine_align`
    and `feature_point_ransac_thinplate_align`.

    Returns
    -------
    On success: (fix_spots, mov_spots, mov_landmark_origin)
        fix_spots, mov_spots : Px3 arrays of matched point coordinates in
            physical units (zyx order), relative to (but not offset by) each
            image's own origin.
        mov_landmark_origin : the origin that applies to mov_spots' physical
            coordinate frame. This is `mov_origin` when no static transforms
            were applied (mov_spots live in mov's own grid), or `fix_origin`
            when static_transform_list is non-empty (mov was resampled onto
            the fix grid via bst.apply_transform before spot detection, so
            its coordinate frame is fix's, not mov's).

    On safeguard failure: raises ValueError if safeguard_exceptions is True,
    otherwise logs and returns None.
    """
    # realize masks
    fix_mask = realize_mask(fix, fix_mask)
    mov_mask = realize_mask(mov, mov_mask)

    # apply static transforms
    mov_landmark_origin = mov_origin
    if static_transform_list:
        mov = bst.apply_transform(
            fix, mov, fix_spacing, mov_spacing,
            transform_list=static_transform_list,
            fix_origin=fix_origin,
            mov_origin=mov_origin,
        )
        if mov_mask is not None:
            mov_mask = bst.apply_transform(
                fix.astype(mov_mask.dtype), mov_mask,
                fix_spacing, mov_spacing,
                transform_list=static_transform_list,
                fix_origin=fix_origin,
                mov_origin=mov_origin,
                interpolator='0',
            )
        mov_spacing = fix_spacing
        mov_landmark_origin = fix_origin

    # skip sample and determine mask spacings
    X = apply_alignment_spacing(
        fix, mov,
        fix_mask, mov_mask,
        fix_spacing, mov_spacing,
        alignment_spacing,
        context=context,
    )
    fix = X[0]
    mov = X[1]
    fix_mask = X[2]
    mov_mask = X[3]
    fix_spacing = X[4]
    mov_spacing = X[5]

    # format inputs
    if type(cc_radius) not in (tuple,):
        cc_radius = (cc_radius,) * fix.ndim
    A, B = blob_sizes[0], blob_sizes[1]
    if not isinstance(A, (tuple, list, np.ndarray)):
        A = (A,)*fix.ndim
    if not isinstance(B, (tuple, list, np.ndarray)):
        B = (B,)*fix.ndim
    blob_sizes = (np.array(A), np.array(B))

    # get fix spots
    num_sigma = int(min(np.max(blob_sizes[1] - blob_sizes[0]), num_sigma_max))
    assert num_sigma > 0, 'num_sigma must be greater than 0, make sure blob_sizes[1] > blob_sizes[0]'

    logger.info(f'{context} computing fixed spots')
    if fix_spots is None:
        fix_kwargs = {
            'num_sigma':num_sigma,
            'exclude_border':cc_radius,
        }
        fix_kwargs = {**fix_kwargs, **fix_spot_detection_kwargs}
        logger.debug(f'{context} fixed spots detection using {fix_kwargs}')
        fix_spots = features.blob_detection(
            fix, blob_sizes[0], blob_sizes[1],
            mask=fix_mask,
            **fix_kwargs,
        )
    elif fix_mask is not None:
        fix_spots = apply_foreground_mask(fix_spots, fix_mask)
    logger.info(f'{context} found {len(fix_spots)} fixed spots')
    if len(fix_spots) < fix_spots_count_threshold:
        logger.info(f'{context} insufficient fixed spots found ({len(fix_spots)}) expected {fix_spots_count_threshold}')
        if safeguard_exceptions:
            raise ValueError('fix spot detection safeguard failed')
        else:
            logger.info(f'{context} - feature point correspondence safeguard failed')
            return None

    # get mov spots
    logger.info(f'{context} computing moving spots')
    if mov_spots is None:
        mov_kwargs = {
            'num_sigma':num_sigma,
            'exclude_border':cc_radius,
        }
        mov_kwargs = {**mov_kwargs, **mov_spot_detection_kwargs}
        logger.debug(f'{context} moving spots detection using {mov_kwargs}')
        mov_spots = features.blob_detection(
            mov, blob_sizes[0], blob_sizes[1],
            mask=mov_mask,
            **mov_kwargs,
        )
    elif mov_mask is not None:
        mov_spots = apply_foreground_mask(mov_spots, mov_mask)
    logger.info(f'{context} found {len(mov_spots)} moving spots')
    if len(mov_spots) < mov_spots_count_threshold:
        logger.info(f'{context} insufficient moving spots found ({len(mov_spots)}) expected {mov_spots_count_threshold}')
        if safeguard_exceptions:
            raise ValueError('mov spot detection safeguard failed')
        else:
            logger.info(f'{context} - feature point correspondence safeguard failed')
            return None

    # sort
    logger.info(f'{context} sorting spots')
    sort_idx = np.argsort(fix_spots[:, -1])[::-1]
    fix_spots = fix_spots[sort_idx, :-1][:nspots]
    sort_idx = np.argsort(mov_spots[:, -1])[::-1]
    mov_spots = mov_spots[sort_idx, :-1][:nspots]

    # get contexts
    logger.info(f'{context} extracting contexts')
    fix_spot_contexts = features.get_contexts(fix, fix_spots, cc_radius)
    mov_spot_contexts = features.get_contexts(mov, mov_spots, cc_radius)

    # get pairwise correlations
    logger.info(f'{context} computing pairwise correlations')
    correlations = features.pairwise_correlation(
        fix_spot_contexts, mov_spot_contexts,
    )

    # convert to physical units
    fix_spots = fix_spots * fix_spacing
    mov_spots = mov_spots * mov_spacing

    # get matching points
    fix_spots, mov_spots = features.match_points(
        fix_spots, mov_spots,
        correlations, match_threshold,
        max_distance=max_spot_match_distance,
    )
    logger.info(f'{context} {len(fix_spots)} - {len(mov_spots)} matched spots')
    if len(fix_spots) < point_matches_threshold or len(mov_spots) < point_matches_threshold:
        logger.info(f'{context} - insufficient point matches found')
        if safeguard_exceptions:
            raise ValueError('point matches safeguard failed')
        else:
            logger.info(f'{context} - feature point correspondence safeguard failed')
            return None

    return fix_spots, mov_spots, mov_landmark_origin


def feature_point_ransac_affine_align(
    fix, mov,
    fix_spacing,
    mov_spacing,
    blob_sizes,
    safeguard_exceptions=True,
    alignment_spacing=None,
    num_sigma_max=15,
    cc_radius=12,
    nspots=5000,
    match_threshold=0.7,
    max_spot_match_distance=None,
    point_matches_threshold=50,
    align_threshold=2.0,
    diagonal_constraint=0.25,
    fix_spot_detection_kwargs={},
    mov_spot_detection_kwargs={},
    fix_spots=None,
    fix_spots_count_threshold=100,
    mov_spots=None,
    mov_spots_count_threshold=100,
    confidence=0.999,
    fix_mask=None,
    mov_mask=None,
    fix_origin=None,
    mov_origin=None,
    static_transform_list=[],
    default=None,
    context='',
    **kwargs,
):
    """
    Currently this function only works on 3D images.

    Compute an affine alignment from feature points and ransac.
    A blob detector finds feature points in fix and mov. Correspondence
    between the fix and mov point sets is estimated using neighborhood
    correlation. A ransac filter determines the affine transform that brings
    the largest number of corresponding points to the same locations.

    Several safeguards are implemented to ensure degenerate or poorly behaved
    affines won't be returned. If your alignment is returning a ValueError,
    then likely one of the safeguards is being triggered. See the
    safeguard_exceptions parameter description below for more information.
    When running this function as part of the distributed pipeline,
    safeguard_exceptions is set to False automatically.

    Parameters
    ----------
    fix : ndarray
        the fixed image

    mov : ndarray
        the moving image; `fix.ndim` must equal `mov.ndim`

    fix_spacing : 1d array
        The spacing in physical units (e.g. mm or um) between voxels
        of the fixed image.
        Length must equal `fix.ndim`

    mov_spacing : 1d array
        The spacing in physical units (e.g. mm or um) between voxels
        of the moving image.
        Length must equal `mov.ndim`

    blob_sizes : list of two floats
        The [minimum, maximum] size of feature point objects in voxel units.
        These are radii; so if your data contains features that are 10 voxels
        diameter on average, a reasonable value for this parameter would be
        [3, 7] (symmetric about a radius of 5).

    safeguard_exceptions : bool (default: True)
        When this value is True, a failed safeguard test will return a
        ValueError and a message indicating which safeguard failed.
        This behavior is desired when working with one image at a time.

        When this value if False, a failed safeguard test will print
        a warning message, but return the identity transform without
        throwing an exception. This behavior is desired when working
        with many images (or tiles/blocks) at the same time.

        Feature point detection and correspondence estimation are noisy
        algorithms. Even with ransac, it is possible that insufficient
        point detections or poor correspondence estimation will result
        in a poor affine. Several safeguards are on by default to prevent
        the return of a bad affine transform. These include:

            * too few spots found in fix or moving image
            * too few correspondences are identified between fix and moving spots
            * an affine that is too far from identity is produced

        These safeguards can all be relaxed through parameters described
        below.

    alignment_spacing : float (default: None)
        Fixed and moving images are skip sampled to a voxel spacing
        as close as possible to this value. Many alignments can be solved
        at far lower resolution than the collected data. This parameter
        can significantly speed up computation.

    num_sigma_max : scalar int (default: 15)
        The maximum number of laplacians to use in the feature point LoG detector

    cc_radius : scalar int or tuple of int (default: 12)
        The halfwidth of neighborhoods around feature points used to determine
        correlation and correspondence. If an int, the same value is used for all
        axes. If a tuple, the tuple length must equal the number of image axes.
        Best practice is to use a tuple for anisotropic data.

    nspots : scalar int (default: 5000)
        The maximum number of feature point spots to use in each image
        If more spots are found the brightest ones are used.

    match_threshold : scalar float in range [0, 1] (default: 0.7)
        The minimum correlation two feature point neighborhoods must have to
        consider them corresponding points. This number can vary significantly
        with input data quality. Consider lowering this before lowering
        point_matches_threshold.

    max_spot_match_distance : scalar float (default: None)
        The maximum distance a fix and mov spot can be before alignment
        to still be considered matching spots; in microns. This helps
        prevent false positive correspondences.

    point_matches_threshold : scalar int (default: 50)
        Minimum number of matching points to proceed with alignment
        Finding fewer matching point pairs than this threshold is a
        safeguard test failure.

    align_threshold : scalar float (default: 2.0)
        The maximum distance two points can be to be considered aligned
        by the affine transform; in microns.

    diagonal_constraint : scalar float (default: 0.25)
        Diagonal entries of the affine matrix cannot be lower than
        1 - diagonal_contraint or higher than 1 + diagonal_contraint.
        Failing this condition is a safeguard test failure. Raising this
        value will allow increasingly extreme affine transforms to be
        returned.

    fix_spot_detection_kwargs : dict (default {})
        Arguments passed to bigstream.features.blob_detection for fixed image
        See docstring for that function for valid arguments.
        You may need to modify these in order to pass the spot count threshold
        safeguards, consider doing that before lowering fix_spots_count_threshold.

    mov_spot_detection_kwargs : dict (default {})
        Arguments passed to bigstream.features.blob_detection for moving image
        See docstring for that function for valid arguments.
        You may need to modify these in order to pass the spot count threshold
        safeguards, consider doing that before lowering mov_spots_count_threshold.

    fix_spots : nd-array Nx3 (default: None)
        Skip the spot detection for the fixed image and provide your own spot coordinate

    fix_spots_count_threshold : scalar int (default: 100)
        Minimum number of fixed spots that need to exist for a valid alignment.
        Note that many times in order to have a better alignment it is better to tweak
        threshold and/or threshold_rel in fix_spot_detection_kwargs then to lower this value

    mov_spots : nd-array Nx3 (default: None)
        Skip the spot detection for the moving image and provide your own spot coordinate

    mov_spots_count_threshold : scalar int (default: 100)
        Minimum number of fixed spots that need to exist for a valid alignment.
        Note that many times in order to have a better alignment it is better to tweak
        threshold and/or threshold_rel in mov_spot_detection_kwargs then to lower this value

    fix_mask : nd-array, tuple of floats, or function (default: None)
        Spots from fixed image can only be found in the foreground of this mask.
        If an nd-array, any non-zero value is considered foreground and any
        zero value is considered background. If a tuple of floats, any voxel
        with value in the tuple is considered background. If a function, it
        must take a single nd-array argument as input and return an array
        of the same shape as the input but with dtype bool.

    mov_mask : nd-array (default: None)
        Spots from moving image can only be found in the foreground of this mask.
        If an nd-array, any non-zero value is considered foreground and any
        zero value is considered background. If a tuple of floats, any voxel
        with value in the tuple is considered background. If a function, it
        must take a single nd-array argument as input and return an array
        of the same shape as the input but with dtype bool.

    fix_origin : 1d array (default: all zeros)
        The origin of the fixed image in physical units

    mov_origin : 1d array (default: all zeros)
        The origin of the moving image in physical units

    static_transform_list : list of numpy arrays (default: [])
        Transforms applied to moving image before applying query transform
        Assumed to have the same domain as the fixed image, though sampling
        can be different. I.e. the origin and span are the same (in phyiscal
        units) but the number of voxels can be different.

    default : 2d array 4x4 (default: identity)
        A default transform to return if the method fails to find a valid one

    context : string
        Additional context information for logging purposes only
        - for local alignment it contains the block index that is being processed

    **kwargs : any additional keyword arguments
        Passed to cv2.estimateAffine3D

    Returns
    -------
    affine_matrix : 2d array 4x4
        An affine matrix matching the moving image to the fixed image
    """
    # establish default
    if default is None: default = np.eye(fix.ndim + 1)

    # find feature point correspondences
    matched = _detect_and_match_feature_points(
        fix, mov, fix_spacing, mov_spacing, blob_sizes,
        safeguard_exceptions, alignment_spacing, num_sigma_max, cc_radius,
        nspots, match_threshold, max_spot_match_distance, point_matches_threshold,
        fix_spot_detection_kwargs, mov_spot_detection_kwargs,
        fix_spots, fix_spots_count_threshold,
        mov_spots, mov_spots_count_threshold,
        fix_mask, mov_mask, fix_origin, mov_origin,
        static_transform_list, context,
    )
    if matched is None:
        logger.info(f'{context} - RANSAC returning default affine')
        return default
    fix_spots, mov_spots, _ = matched

    # align
    logger.debug(f'{context} Found enough spots to estimate the affine ' +
                 f'fix: {len(fix_spots)} ' +
                 f'moving: {len(mov_spots)}')
    _, Aff, _ = cv2.estimateAffine3D(
        fix_spots, mov_spots,
        ransacThreshold=align_threshold,
        confidence=confidence,
        **kwargs,
    )

    # ensure affine is sensible
    if np.any( np.abs(np.diag(Aff) - 1) > diagonal_constraint ):
        logger.info(f'{context} RANSAC produced degenerate affine: {Aff}')
        if safeguard_exceptions:
            raise ValueError('diagonal_constraint safeguard failed')
        else:
            logger.info(f'{context} - RANSAC returning default affine')
            return default

    # augment matrix and return
    affine = np.eye(fix.ndim + 1)
    affine[:fix.ndim, :] = Aff
    logger.debug(f'{context} - RANSAC affine: {Aff}')
    return affine


def feature_point_ransac_thinplate_align(
    fix, mov,
    fix_spacing,
    mov_spacing,
    blob_sizes,
    control_point_spacing,
    control_point_levels,
    alignment_spacing=None,
    num_sigma_max=15,
    cc_radius=12,
    nspots=5000,
    match_threshold=0.7,
    max_spot_match_distance=None,
    point_matches_threshold=50,
    align_threshold=2.0,
    diagonal_constraint=0.25,
    filter_landmarks_with_ransac=True,
    fix_spot_detection_kwargs={},
    mov_spot_detection_kwargs={},
    fix_spots=None,
    fix_spots_count_threshold=100,
    mov_spots=None,
    mov_spots_count_threshold=100,
    confidence=0.999,
    fix_mask=None,
    mov_mask=None,
    fix_origin=None,
    mov_origin=None,
    static_transform_list=[],
    default=None,
    context='',
    **kwargs,
):
    """
    Currently this function only works on 3D images.

    Compute a thin-plate-style BSpline alignment from feature points and
    ransac. A blob detector finds feature points in fix and mov.
    Correspondence between the fix and mov point sets is estimated using
    neighborhood correlation. Those correspondences (optionally RANSAC-
    filtered to their affine-inlier subset) are used as landmarks to
    initialize a BSpline transform via SimpleITK's
    `LandmarkBasedTransformInitializerFilter` (a thin-plate-spline-style
    solve constrained to the requested control point mesh). That landmark-
    initialized transform is then refined with intensity-based BSpline
    registration, using the same registration engine as `deformable_align`.

    Several safeguards are implemented to ensure degenerate or poorly behaved
    transforms won't be returned.
    Parameters
    ----------
    fix : ndarray
        the fixed image

    mov : ndarray
        the moving image; `fix.ndim` must equal `mov.ndim`

    fix_spacing : 1d array
        The spacing in physical units (e.g. mm or um) between voxels
        of the fixed image.
        Length must equal `fix.ndim`

    mov_spacing : 1d array
        The spacing in physical units (e.g. mm or um) between voxels
        of the moving image.
        Length must equal `mov.ndim`

    blob_sizes : list of two floats
        The [minimum, maximum] size of feature point objects in voxel units.
        These are radii; so if your data contains features that are 10 voxels
        diameter on average, a reasonable value for this parameter would be
        [3, 7] (symmetric about a radius of 5).

    control_point_spacing : float or 1d array
        The spacing in physical units (e.g. mm or um) between control
        points that parameterize the deformation, used both to size the
        landmark-fit BSpline mesh and to drive the intensity refinement
        stage. Same semantics as `deformable_align`'s parameter of the same
        name: a scalar is broadcast to every spatial axis; an array shorter
        than the image dimensionality is extended by repeating its last
        element.

    control_point_levels : list of type int
        The optimization scales for control point spacing, for the intensity
        refinement stage. Same semantics as `deformable_align`. Its first
        element also determines the mesh resolution used for the landmark
        fit (see control_point_spacing).

    alignment_spacing : float (default: None)
        Fixed and moving images are skip sampled to a voxel spacing
        as close as possible to this value. Many alignments can be solved
        at far lower resolution than the collected data. This parameter
        can significantly speed up computation.

    num_sigma_max : scalar int (default: 15)
        The maximum number of laplacians to use in the feature point LoG detector

    cc_radius : scalar int or tuple of int (default: 12)
        The halfwidth of neighborhoods around feature points used to determine
        correlation and correspondence. If an int, the same value is used for all
        axes. If a tuple, the tuple length must equal the number of image axes.
        Best practice is to use a tuple for anisotropic data.

    nspots : scalar int (default: 5000)
        The maximum number of feature point spots to use in each image
        If more spots are found the brightest ones are used.

    match_threshold : scalar float in range [0, 1] (default: 0.7)
        The minimum correlation two feature point neighborhoods must have to
        consider them corresponding points. This number can vary significantly
        with input data quality. Consider lowering this before lowering
        point_matches_threshold.

    max_spot_match_distance : scalar float (default: None)
        The maximum distance a fix and mov spot can be before alignment
        to still be considered matching spots; in microns. This helps
        prevent false positive correspondences.

    point_matches_threshold : scalar int (default: 50)
        Minimum number of matching points to proceed with alignment.
        Also the minimum number of RANSAC inlier landmarks required, when
        filter_landmarks_with_ransac is True.

    align_threshold : scalar float (default: 2.0)
        Only used when filter_landmarks_with_ransac is True. The maximum
        distance two points can be to be considered inliers by the RANSAC
        pre-fit affine (in microns). This affine is discarded; it is only
        used to select which correspondences become landmarks.

    diagonal_constraint : scalar float (default: 0.25)
        Only used when filter_landmarks_with_ransac is True. Diagonal
        entries of the RANSAC pre-fit affine matrix cannot be lower than
        1 - diagonal_contraint or higher than 1 + diagonal_contraint, or the
        pre-fit (and hence the landmark selection) is considered a safeguard
        failure. This gates the throwaway pre-fit affine only -- it has no
        bearing on the returned BSpline transform.

    filter_landmarks_with_ransac : bool (default: True)
        If True, fit a throwaway affine with `cv2.estimateAffine3D` over the
        matched correspondences and keep only its inlier subset as landmarks
        for the thin plate BSpline fit. The affine itself is never returned.
        If False, every matched correspondence is used as a landmark.

    fix_spot_detection_kwargs : dict (default {})
        Arguments passed to bigstream.features.blob_detection for fixed image
        See docstring for that function for valid arguments.
        You may need to modify these in order to pass the spot count threshold
        safeguards, consider doing that before lowering fix_spots_count_threshold.

    mov_spot_detection_kwargs : dict (default {})
        Arguments passed to bigstream.features.blob_detection for moving image
        See docstring for that function for valid arguments.
        You may need to modify these in order to pass the spot count threshold
        safeguards, consider doing that before lowering mov_spots_count_threshold.

    fix_spots : nd-array Nx3 (default: None)
        Skip the spot detection for the fixed image and provide your own spot coordinate

    fix_spots_count_threshold : scalar int (default: 100)
        Minimum number of fixed spots that need to exist for a valid alignment.
        Note that many times in order to have a better alignment it is better to tweak
        threshold and/or threshold_rel in fix_spot_detection_kwargs then to lower this value

    mov_spots : nd-array Nx3 (default: None)
        Skip the spot detection for the moving image and provide your own spot coordinate

    mov_spots_count_threshold : scalar int (default: 100)
        Minimum number of fixed spots that need to exist for a valid alignment.
        Note that many times in order to have a better alignment it is better to tweak
        threshold and/or threshold_rel in mov_spot_detection_kwargs then to lower this value

    fix_mask : nd-array, tuple of floats, or function (default: None)
        Spots from fixed image can only be found in the foreground of this mask.
        Also used as the intensity refinement stage's fixed metric mask.
        If an nd-array, any non-zero value is considered foreground and any
        zero value is considered background. If a tuple of floats, any voxel
        with value in the tuple is considered background. If a function, it
        must take a single nd-array argument as input and return an array
        of the same shape as the input but with dtype bool.

    mov_mask : nd-array (default: None)
        Spots from moving image can only be found in the foreground of this mask.
        Also used as the intensity refinement stage's moving metric mask.
        If an nd-array, any non-zero value is considered foreground and any
        zero value is considered background. If a tuple of floats, any voxel
        with value in the tuple is considered background. If a function, it
        must take a single nd-array argument as input and return an array
        of the same shape as the input but with dtype bool.

    fix_origin : 1d array (default: all zeros)
        The origin of the fixed image in physical units

    mov_origin : 1d array (default: all zeros)
        The origin of the moving image in physical units

    static_transform_list : list of numpy arrays (default: [])
        Transforms applied to moving image before applying query transform
        Assumed to have the same domain as the fixed image, though sampling
        can be different. I.e. the origin and span are the same (in phyiscal
        units) but the number of voxels can be different.

    default : tuple of (1d array, nd-array) (default: identity bspline)
        A default (params, field) result to return if the method fails to
        find a valid transform. If None, the identity BSpline transform at
        the requested control point resolution is used (matching
        `deformable_align`'s default convention).

    final_metric_check : bool (default: True)
        The metric function is checked before and after the intensity
        refinement stage. If this flag is True then the function will only
        return the refined transform if the final metric value is better
        than the initial (landmark-only) metric value. Otherwise it will
        return the default.

    context : string
        Additional context information for logging purposes only
        - for local alignment it contains the block index that is being processed

    **kwargs : any additional keyword arguments
        Passed to `configure_irm` for the intensity refinement stage.
        This is where you would set things like:
        metric, iterations, shrink_factors, and smooth_sigmas

    Returns
    -------
    params : 1d array
        The complete set of BSpline control point parameters concatenated
        as a 1d array.

    field : ndarray
        The displacement field parameterized by the BSpline control points
    """
    ndim = fix.ndim
    initial_fix_shape = fix.shape
    initial_fix_spacing = np.array(fix_spacing)

    fix_origin_arr = np.zeros(ndim) if fix_origin is None else np.asarray(fix_origin, dtype=float)

    # format static transform data explicitly, for the intensity refinement stage
    static_transform_spacing, static_transform_origin = format_static_transform_data(
        static_transform_list, fix, fix_spacing, fix_origin,
    )

    # realize masks (used both for spot detection, via the helper below, and
    # for the intensity refinement stage's metric masks)
    fix_mask_r = realize_mask(fix, fix_mask)
    mov_mask_r = realize_mask(mov, mov_mask)

    # skip sample and convert to sitk images; this defines the domain used
    # for the landmark reference image, the bspline mesh, and the intensity
    # refinement stage
    X = apply_alignment_spacing(
        fix, mov,
        fix_mask_r, mov_mask_r,
        fix_spacing, mov_spacing,
        alignment_spacing,
        context=context,
    )
    fix_img, mov_img, fix_mask_img, mov_mask_img = images_to_sitk(
        *X, fix_origin, mov_origin,
    )

    # bspline mesh sizing (identical convention to deformable_align)
    cp_spacing = np.atleast_1d(control_point_spacing)
    if cp_spacing.size < 1:
        raise ValueError('control_point_spacing must not be empty')
    if cp_spacing.size > ndim:
        cp_spacing = cp_spacing[:ndim]
    if cp_spacing.size < ndim:
        cp_spacing = np.concatenate([
            cp_spacing, np.full(ndim - cp_spacing.size, cp_spacing[-1]),
        ])
    cp_spacing_xyz = cp_spacing[::-1]
    cp_divisor = cp_spacing_xyz * control_point_levels[0]
    initial_cp_grid = [
        max(1, int(x*y/d))
        for x, y, d in zip(fix_img.GetSize(), fix_img.GetSpacing(), cp_divisor)
    ]
    identity_transform = sitk.BSplineTransformInitializer(
        image1=fix_img, transformDomainMeshSize=initial_cp_grid, order=3,
    )
    logger.debug((
        f'{context} '
        'thin plate BSpline control point grid: '
        f'{fix_img.GetSize()}*{fix_img.GetSpacing()}/'
        f'({cp_spacing_xyz}*{control_point_levels[0]})={initial_cp_grid} '
    ))

    # establish default: identity bspline (params, field), same convention as
    # deformable_align
    if default is None:
        params = np.concatenate((
            identity_transform.GetFixedParameters(), identity_transform.GetParameters(),
        ))
        field = bst.bspline_to_displacement_field(
            identity_transform, initial_fix_shape,
            spacing=initial_fix_spacing, origin=fix_origin,
            direction=np.eye(ndim),
        )
        default = (params, field)

    # find feature point correspondences
    matched = _detect_and_match_feature_points(
        fix, mov, fix_spacing, mov_spacing, blob_sizes,
        False, alignment_spacing, num_sigma_max, cc_radius,
        nspots, match_threshold, max_spot_match_distance, point_matches_threshold,
        fix_spot_detection_kwargs, mov_spot_detection_kwargs,
        fix_spots, fix_spots_count_threshold,
        mov_spots, mov_spots_count_threshold,
        fix_mask, mov_mask, fix_origin, mov_origin,
        static_transform_list, context,
    )
    if matched is None:
        logger.info(f'{context} - thin plate align returning default')
        return default
    try:
        landmark_fix, landmark_mov, mov_landmark_origin = matched
        mov_landmark_origin = (
            np.zeros(ndim) if mov_landmark_origin is None
            else np.asarray(mov_landmark_origin, dtype=float)
        )

        # optionally RANSAC-filter the correspondences: fit a throwaway affine
        # purely to obtain an inlier mask, then keep only inlier correspondences
        # as landmarks. The affine itself is never returned.
        if filter_landmarks_with_ransac:
            logger.debug(f'{context} RANSAC pre-fit over {len(landmark_fix)} matched '
                        'spots to select inlier landmarks')
            success, Aff, inliers = cv2.estimateAffine3D(
                landmark_fix, landmark_mov,
                ransacThreshold=align_threshold,
                confidence=confidence,
            )
            if not success or np.any(np.abs(np.diag(Aff) - 1) > diagonal_constraint):
                logger.info((
                    f'{context} RANSAC pre-fit produced a degenerate affine: {Aff} '
                    '- thin plate align returning default'
                ))
                return default

            inliers = inliers.astype(bool).reshape(-1)
            n_matched = len(landmark_fix)
            landmark_fix = landmark_fix[inliers]
            landmark_mov = landmark_mov[inliers]
            logger.info((
                f'{context} {len(landmark_fix)} RANSAC inlier landmarks '
                f'(of {n_matched} matched)'
            ))
            if len(landmark_fix) < point_matches_threshold:
                logger.info((
                    f'{context} insufficient RANSAC inlier landmarks found '
                    f'({len(landmark_fix)}) expected {point_matches_threshold} '
                    '- thin plate align returning default'
                ))
                return default

        # landmark points are physical coordinates relative to (but not offset
        # by) each image's own origin; add the appropriate origin, then reverse
        # zyx -> xyz to match sitk's landmark/point ordering convention
        landmark_fix_xyz = (landmark_fix + fix_origin_arr)[:, ::-1]
        landmark_mov_xyz = (landmark_mov + mov_landmark_origin)[:, ::-1]

        # landmark-initialize a bspline transform (thin-plate-style fit)
        lm_initializer = sitk.LandmarkBasedTransformInitializerFilter()
        lm_initializer.SetFixedLandmarks(landmark_fix_xyz.ravel().tolist())
        lm_initializer.SetMovingLandmarks(landmark_mov_xyz.ravel().tolist())
        lm_initializer.SetReferenceImage(fix_img)

        transform = lm_initializer.Execute(identity_transform)
        logger.debug(f'{context} landmark-initialized thin plate BSpline transform')

        params = np.concatenate((transform.GetFixedParameters(), transform.GetParameters()))
        field = bst.bspline_to_displacement_field(
            transform, initial_fix_shape,
            spacing=initial_fix_spacing, origin=fix_origin,
            direction=np.eye(ndim),
        )

        # diagnostics: check the deformation field for folding and discontinuities
        deform_field_diagnostics(field, initial_fix_spacing, context=context)

        return params, field
    except Exception as e:
        logger.error(f'{context} Thin plate registration failed due to ITK exception: {e}')
        logger.info(f'{context} Returning default')
        return default


def random_affine_search(
    fix,
    mov,
    fix_spacing,
    mov_spacing,
    random_iterations,
    nreturn=1,
    max_translation=None,
    max_rotation=None,
    max_scale=None,
    max_shear=None,
    alignment_spacing=None,
    fix_mask=None,
    mov_mask=None,
    fix_origin=None,
    mov_origin=None,
    static_transform_list=[],
    use_patch_mutual_information=False,
    context='',
    **kwargs,
):
    """
    Apply random affine matrices within given bounds to moving image.
    This function is intended to find good initialization for a full affine
    alignment obtained by calling `affine_align`

    Parameters
    ----------
    fix : ndarray
        the fixed image

    mov : ndarray
        the moving image; `fix.ndim` must equal `mov.ndim`

    fix_spacing : 1d array
        The spacing in physical units (e.g. mm or um) between voxels
        of the fixed image.
        Length must equal `fix.ndim`

    mov_spacing : 1d array
        The spacing in physical units (e.g. mm or um) between voxels
        of the moving image.
        Length must equal `mov.ndim`

    random_iterations : int
        The number of random affine matrices to sample

    nreturn : int (default: 1)
        The number of affine matrices to return. The best scoring results
        are returned.

    max_translation : float or tuple of float
        The maximum amplitude translation allowed in random sampling.
        Specified in physical units (e.g. um or mm)
        Can be specified per axis.

    max_rotation : float or tuple of float
        The maximum amplitude rotation allowed in random sampling.
        Specified in radians
        Can be specified per axis.

    max_scale : float or tuple of float
        The maximum amplitude scaling allowed in random sampling.
        Can be specified per axis.

    max_shear : float or tuple of float
        The maximum amplitude shearing allowed in random sampling.
        Can be specified per axis.

    alignment_spacing : float (default: None)
        Fixed and moving images are skip sampled to a voxel spacing
        as close as possible to this value. Intended for very fast
        simple alignments (e.g. low amplitude motion correction)

    fix_mask : ndarray, tuple of floats, or function (default: None)
        A mask limiting metric evaluation region of the fixed image
        If an nd-array, any non-zero value is considered foreground and any
        zero value is considered background. If a tuple of floats, any voxel
        with value in the tuple is considered background. If a function, it
        must take a single nd-array argument as input and return an array
        of the same shape as the input but with dtype bool.

        If an nd-array, it is assumed to have the same domain as the fixed
        image, though sampling can be different. I.e. the origin and span
        are the same (in phyiscal units) but the number of voxels can
        be different.

    mov_mask : ndarray, tuple of floats, or function (default: None)
        A mask limiting metric evaluation region of the moving image
        If an nd-array, any non-zero value is considered foreground and any
        zero value is considered background. If a tuple of floats, any voxel
        with value in the tuple is considered background. If a function, it
        must take a single nd-array argument as input and return an array
        of the same shape as the input but with dtype bool.

        If an nd-array, it is assumed to have the same domain as the fixed
        image, though sampling can be different. I.e. the origin and span
        are the same (in phyiscal units) but the number of voxels can
        be different.

    fix_origin : 1d array (default: None)
        Origin of the fixed image.
        Length must equal `fix.ndim`

    mov_origin : 1d array (default: None)
        Origin of the moving image.
        Length must equal `mov.ndim`

    static_transform_list : list of numpy arrays (default: [])
        Transforms applied to moving image before applying query transform
        Assumed to have the same domain as the fixed image, though sampling
        can be different. I.e. the origin and span are the same (in phyiscal
        units) but the number of voxels can be different.

    use_patch_mutual_information : bool (default: False)
        Uses a custom metric function in bigstream.metrics

    **kwargs : any additional arguments
        Passed to `configure_irm` This is how you customize the metric.
        If `use_path_mutual_information` is True this is passed to
        the `patch_mutual_information` function instead.

    Returns
    -------
    best transforms : sorted list of 4x4 numpy.ndarrays (affine matrices)
        best nreturn results, first element of list is the best result
    """

    # function to help generalize parameter limits to 3d
    def expand_param_to_3d(param, null_value):
        if isinstance(param, (int, float)):
            param = (param,) * 2
        if isinstance(param, tuple):
            param += (null_value,)
        return param

    # TODO: consider moving to native 2D
    # generalize 2d inputs to 3d
    if fix.ndim == 2:
        fix = fix.reshape(fix.shape + (1,))
        mov = mov.reshape(mov.shape + (1,))
        fix_spacing = tuple(fix_spacing) + (1.,)
        mov_spacing = tuple(mov_spacing) + (1.,)
        max_translation = expand_param_to_3d(max_translation, 0)
        max_rotation = expand_param_to_3d(max_rotation, 0)
        max_scale = expand_param_to_3d(max_scale, 1)
        max_shear = expand_param_to_3d(max_shear, 0)
        if fix_mask is not None: fix_mask = fix_mask.reshape(fix_mask.shape + (1,))
        if mov_mask is not None: mov_mask = mov_mask.reshape(mov_mask.shape + (1,))
        if fix_origin is not None: fix_origin = tuple(fix_origin) + (0.,)
        if mov_origin is not None: mov_origin = tuple(mov_origin) + (0.,)

    # generate random parameters, first row is always identity
    params = np.zeros((random_iterations+1, 12))
    params[:, 6:9] = 1  # default for scale params

    def F(mx):
        return 2 * (mx * np.random.rand(random_iterations, 3)) - mx

    if max_translation: params[1:, 0:3] = F(max_translation)
    if max_rotation: params[1:, 3:6] = F(max_rotation)
    if max_scale: params[1:, 6:9] = np.e**F(np.log(max_scale))
    if max_shear: params[1:, 9:] = F(max_shear)
    center = np.array(fix.shape) / 2 * fix_spacing  # center of rotation

    # format static transform data explicitly
    a, b = format_static_transform_data(
        static_transform_list, fix, fix_spacing, fix_origin,
    )
    static_transform_spacing = a
    static_transform_origin = b

    # realize masks as arrays
    fix_mask = realize_mask(fix, fix_mask)
    mov_mask = realize_mask(mov, mov_mask)

    # skip sample and determine mask spacings
    X = apply_alignment_spacing(
        fix, mov,
        fix_mask, mov_mask,
        fix_spacing, mov_spacing,
        alignment_spacing,
        context=context,
    )
    fix = X[0]
    mov = X[1]
    fix_mask = X[2]
    mov_mask = X[3]
    fix_spacing = X[4]
    mov_spacing = X[5]
    fix_mask_spacing = X[6]
    mov_mask_spacing = X[7]

    # a useful value later, storing prevents redundant function calls
    WORST_POSSIBLE_SCORE = np.finfo(np.float64).max

    # define metric evaluation
    if use_patch_mutual_information:
        # wrap patch_mi metric
        def score_affine(affine):
            # apply transform
            transform_list = static_transform_list + [affine,]
            aligned = bst.apply_transform(
                fix, mov, fix_spacing, mov_spacing,
                transform_list=transform_list,
                fix_origin=fix_origin,
                mov_origin=mov_origin,
                transform_spacing=static_transform_spacing,
                transform_origin=static_transform_origin,
            )
            mov_mask_aligned = None
            if mov_mask is not None:
                mov_mask_aligned = bst.apply_transform(
                    fix_mask, mov_mask, fix_mask_spacing, mov_mask_spacing,
                    transform_list=transform_list,
                    fix_origin=fix_origin,
                    mov_origin=mov_origin,
                    transform_spacing=static_transform_spacing,
                    transform_origin=static_transform_origin,
                    interpolator='0',
                )
            # evaluate metric
            # TODO: this function needs to be updated for different
            #       mask and image sizes
            return patch_mutual_information(
                fix, aligned, fix_spacing,
                fix_mask=fix_mask,
                mov_mask=mov_mask_aligned,
                return_metric_image=False,
                **kwargs,
            )

    # use an irm metric
    else:
        # construct irm, set images, masks, transforms
        kwargs['optimizer'] = 'LBFGS2'    # optimizer is not used, just a dummy value
        kwargs['optimizer_args'] = {}
        irm = configure_irm(context=context, **kwargs)
        fix, mov, fix_mask, mov_mask = images_to_sitk(
            fix, mov, fix_mask, mov_mask,
            fix_spacing, mov_spacing,
            fix_mask_spacing, mov_mask_spacing,
            fix_origin, mov_origin,
        )
        if fix_mask is not None: irm.SetMetricFixedMask(fix_mask)
        if mov_mask is not None: irm.SetMetricMovingMask(mov_mask)
        if static_transform_list:
            T = bst.transform_list_to_composite_transform(
                static_transform_list,
                static_transform_spacing,
                static_transform_origin,
            )
            irm.SetMovingInitialTransform(T)

        # wrap irm metric
        def score_affine(affine):
            irm.SetInitialTransform(bst.matrix_to_affine_transform(affine))
            try:
                return irm.MetricEvaluate(fix, mov)
            except Exception as e:
                return WORST_POSSIBLE_SCORE

    # score all random affines
    current_best_score = WORST_POSSIBLE_SCORE
    scores = np.empty(random_iterations + 1, dtype=np.float64)
    for iii, ppp in enumerate(params):
        scores[iii] = score_affine(bst.physical_parameters_to_affine_matrix_3d(ppp, center))
        if scores[iii] < current_best_score:
            current_best_score = scores[iii]
            logger.debug(f'{context} - best score found {iii} : {current_best_score}')

    # return top results
    partition_indx = np.argpartition(scores, nreturn)[:nreturn]
    params, scores = params[partition_indx], scores[partition_indx]
    return [bst.physical_parameters_to_affine_matrix_3d(p, center) for p in params[np.argsort(scores)]]


def affine_align(
    fix,
    mov,
    fix_spacing,
    mov_spacing,
    rigid=False,
    initial_condition=None,
    alignment_spacing=None,
    fix_mask=None,
    mov_mask=None,
    fix_origin=None,
    mov_origin=None,
    static_transform_list=[],
    default=None,
    final_metric_check=True,
    context='',
    **kwargs,
):
    """
    Affine or rigid alignment of a fixed/moving image pair.
    Lots of flexibility in speed/accuracy trade off.
    Highly configurable and useful in many contexts.

    Parameters
    ----------
    fix : ndarray
        the fixed image

    mov : ndarray
        the moving image; `fix.ndim` must equal `mov.ndim`

    fix_spacing : 1d array
        The spacing in physical units (e.g. mm or um) between voxels
        of the fixed image.
        Length must equal `fix.ndim`

    mov_spacing : 1d array
        The spacing in physical units (e.g. mm or um) between voxels
        of the moving image.
        Length must equal `mov.ndim`

    rigid : bool (default: False)
        Restrict the alignment to rigid motion only

    initial_condition : str or 4x4 ndarray (default: None)
        How to begin the optimization. Only one string value is allowed:
        "CENTER" in which case the alignment is initialized by a center
        of mass alignment. If a 4x4 ndarray is given the optimization
        is initialized with that transform. static_transform_list is
        ignored.

    alignment_spacing : float (default: None)
        Fixed and moving images are skip sampled to a voxel spacing
        as close as possible to this value. Intended for very fast
        simple alignments (e.g. low amplitude motion correction)

    fix_mask : ndarray, tuple of floats, or function (default: None)
        A mask limiting metric evaluation region of the fixed image
        If an nd-array, any non-zero value is considered foreground and any
        zero value is considered background. If a tuple of floats, any voxel
        with value in the tuple is considered background. If a function, it
        must take a single nd-array argument as input and return an array
        of the same shape as the input but with dtype bool.

        If an nd-array, it is assumed to have the same domain as the fixed
        image, though sampling can be different. I.e. the origin and span
        are the same (in phyiscal units) but the number of voxels can
        be different.

    mov_mask : ndarray, tuple of floats, or function (default: None)
        A mask limiting metric evaluation region of the moving image
        If an nd-array, any non-zero value is considered foreground and any
        zero value is considered background. If a tuple of floats, any voxel
        with value in the tuple is considered background. If a function, it
        must take a single nd-array argument as input and return an array
        of the same shape as the input but with dtype bool.

        If an nd-array, it is assumed to have the same domain as the fixed
        image, though sampling can be different. I.e. the origin and span
        are the same (in phyiscal units) but the number of voxels can
        be different.

    fix_origin : 1d array (default: None)
        Origin of the fixed image.
        Length must equal `fix.ndim`

    mov_origin : 1d array (default: None)
        Origin of the moving image.
        Length must equal `mov.ndim`

    static_transform_list : list of numpy arrays (default: [])
        Transforms applied to moving image before applying query transform
        Assumed to have the same domain as the fixed image, though sampling
        can be different. I.e. the origin and span are the same (in phyiscal
        units) but the number of voxels can be different.

    default : 4x4 array (default: identity matrix)
        If the optimization fails, print error message but return this value

    final_metric_check : bool (default: True)
        The metric function is checked before and after alignment. If this flag is
        True then the function will only return the optimized transform if the final metric
        value is better than the initial metric value. Otherwise it will return the default.
        If this flag is False, then the optimized transform is returned regardless.


    **kwargs : any additional arguments
        Passed to `configure_irm`
        This is where you would set things like:
        metric, iterations, shrink_factors, and smooth_sigmas

    Returns
    -------
    transform : 4x4 array
        The affine or rigid transform matrix matching moving to fixed
    """
    logger.info(f'Affine align {context} -> {kwargs}')
    # determine the correct default
    if default is None:
        default = np.eye(fix.ndim + 1)
    initial_transform_given = isinstance(initial_condition, np.ndarray)
    if initial_transform_given and np.all(default == np.eye(fix.ndim + 1)):
        default = initial_condition

    # format static transform data explicitly
    a, b = format_static_transform_data(
        static_transform_list, fix, fix_spacing, fix_origin,
    )
    static_transform_spacing = a
    static_transform_origin = b

    # realize masks
    fix_mask = realize_mask(fix, fix_mask)
    mov_mask = realize_mask(mov, mov_mask)

    # skip sample and convert inputs to sitk images
    X = apply_alignment_spacing(
        fix, mov,
        fix_mask, mov_mask,
        fix_spacing, mov_spacing,
        alignment_spacing,
        context=context,
    )
    fix, mov, fix_mask, mov_mask = images_to_sitk(
        *X, fix_origin, mov_origin,
    )
    fix_spacing = X[4]
    mov_spacing = X[5]
    fix_mask_spacing = X[6]
    mov_mask_spacing = X[7]

    # set up registration object
    logger.debug(f'Configure {context} IRM args: {kwargs}')
    irm = configure_irm(context=context, **kwargs)
    # set initial static transforms
    if static_transform_list:
        T = bst.transform_list_to_composite_transform(
            static_transform_list,
            static_transform_spacing,
            static_transform_origin,
        )
        irm.SetMovingInitialTransform(T)

    # distinguish between 2D and 3D for rigid transforms
    ndims = fix.GetDimension()
    rigid_transform_constructor = sitk.Euler2DTransform if ndims == 2 else sitk.Euler3DTransform

    # set transform to optimize
    if isinstance(initial_condition, str):
        if initial_condition == 'CENTER':
            a, b = fix, mov
            initializer = sitk.CenteredTransformInitializer(
                a, # fix
                b, # mov
                rigid_transform_constructor(), # EulerTransform
                sitk.CenteredTransformInitializerFilter.GEOMETRY,
            )
            x = rigid_transform_constructor(initializer).GetTranslation()[::-1]
            initial_condition = np.eye(ndims+1)
            initial_condition[:ndims, -1] = x
            logger.info(f'CENTER initial condition: {initial_condition}')
            initial_transform_given = True
        elif initial_condition == 'MOMENTS':
            a, b = fix, mov
            initial_transform = sitk.CenteredTransformInitializer(
                a, # fix
                b, # mov
                rigid_transform_constructor(), # EulerTransform
                sitk.CenteredTransformInitializerFilter.MOMENTS,
            )
            initial_condition = bst.affine_transform_to_matrix(initial_transform)
            logger.info(f'MOMENTS initial condition: {initial_condition}')
            initial_transform_given = True

    if rigid and not initial_transform_given:
        transform = rigid_transform_constructor()
    elif rigid and initial_transform_given:
        transform = bst.matrix_to_euler_transform(initial_condition)
    elif not rigid and not initial_transform_given:
        transform = sitk.AffineTransform(fix.GetDimension())
    elif not rigid and initial_transform_given:
        transform = bst.matrix_to_affine_transform(initial_condition)
    else:
        transform = None
    irm.SetInitialTransform(transform, inPlace=True)
    # set masks
    if fix_mask is not None:
        irm.SetMetricFixedMask(fix_mask)
    if mov_mask is not None:
        irm.SetMetricMovingMask(mov_mask)

    # execute alignment, for any exceptions return default
    try:
        initial_metric_value = irm.MetricEvaluate(fix, mov)
        irm.Execute(fix, mov)
        final_metric_value = irm.MetricEvaluate(fix, mov)
    except Exception as e:
        logger.error(f'{context} Affine align failed due to ITK exception: {e}')
        logger.info(f'{context} Affine align returning default')
        return default

    # if registration improved metric return result
    # otherwise return default
    if final_metric_check and final_metric_value > initial_metric_value:
        logger.warning((
            f'{context} Affine align optimization failed to improve metric: '
            f'initial: {initial_metric_value}, '
            f'final: {final_metric_value} '
        ))
        logger.info(f'{context} Affine align returning default')
        return default
    else:
        affine_ndarray = bst.affine_transform_to_matrix(transform)
        logger.info((
            f'{context} Affine align succeeded: '
            f'(initial_metric={initial_metric_value}, final_metric={final_metric_value}), '
            f'affine matrix: {affine_ndarray} '
        ))
        return affine_ndarray


def deformable_align(
    fix,
    mov,
    fix_spacing,
    mov_spacing,
    control_point_spacing,
    control_point_levels,
    alignment_spacing=None,
    fix_mask=None,
    mov_mask=None,
    fix_origin=None,
    mov_origin=None,
    static_transform_list=[],
    default=None,
    final_metric_check=True,
    min_overlap_fraction=0.01,
    context='',
    **kwargs,
):
    """
    Register moving to fixed image with a bspline parameterized deformation field

    Parameters
    ----------
    fix : ndarray
        the fixed image

    mov : ndarray
        the moving image; `fix.ndim` must equal `mov.ndim`

    fix_spacing : 1d array
        The spacing in physical units (e.g. mm or um) between voxels
        of the fixed image.
        Length must equal `fix.ndim`

    mov_spacing : 1d array
        The spacing in physical units (e.g. mm or um) between voxels
        of the moving image.

    control_point_spacing : float or 1d array
        The spacing in physical units (e.g. mm or um) between control
        points that parameterize the deformation. Smaller means
        more precise alignment, but also longer compute time. Larger
        means shorter compute time and smoother transform, but less
        precise.

        A scalar is broadcast to every spatial axis. An array shorter
        than the image dimensionality is extended by repeating its last
        element, e.g. [128, 64] becomes [128, 64, 64] for a 3D image.
        Values are given in the same axis order as `fix_spacing` (zyx),
        which allows different control point spacing per axis to
        account for anisotropic images.

    control_point_levels : list of type int
        The optimization scales for control point spacing. E.g. if
        `control_point_spacing` is 100.0 and `control_point_levels`
        is [4, 2, 1] then method will optimize at 400.0 units control
        points spacing, then optimize again at 200.0 units, then again
        at the requested 100.0 units control point spacing.
    
    alignment_spacing : float (default: None)
        Fixed and moving images are skip sampled to a voxel spacing
        as close as possible to this value. Intended for very fast
        simple alignments (e.g. low amplitude motion correction)

    fix_mask : ndarray, tuple of floats, or function (default: None)
        A mask limiting metric evaluation region of the fixed image
        If an nd-array, any non-zero value is considered foreground and any
        zero value is considered background. If a tuple of floats, any voxel
        with value in the tuple is considered background. If a function, it
        must take a single nd-array argument as input and return an array
        of the same shape as the input but with dtype bool.

        If an nd-array, it is assumed to have the same domain as the fixed
        image, though sampling can be different. I.e. the origin and span
        are the same (in phyiscal units) but the number of voxels can
        be different.

    mov_mask : ndarray, tuple of floats, or function (default: None)
        A mask limiting metric evaluation region of the moving image
        If an nd-array, any non-zero value is considered foreground and any
        zero value is considered background. If a tuple of floats, any voxel
        with value in the tuple is considered background. If a function, it
        must take a single nd-array argument as input and return an array
        of the same shape as the input but with dtype bool.

        If an nd-array, it is assumed to have the same domain as the fixed
        image, though sampling can be different. I.e. the origin and span
        are the same (in phyiscal units) but the number of voxels can
        be different.

    fix_origin : 1d array (default: None)
        Origin of the fixed image.
        Length must equal `fix.ndim`

    mov_origin : 1d array (default: None)
        Origin of the moving image.
        Length must equal `mov.ndim`

    static_transform_list : list of numpy arrays (default: [])
        Transforms applied to moving image before applying query transform
        Assumed to have the same domain as the fixed image, though sampling
        can be different. I.e. the origin and span are the same (in phyiscal
        units) but the number of voxels can be different.

    default : any object (default: None)
        If optimization fails to improve image matching metric,
        print an error but also return this object. If None
        the parameters and displacement field for an identity
        transform are returned.

    final_metric_check : bool (default: True)
        The metric function is checked before and after alignment. If this flag is
        True then the function will only return the optimized transform if the final metric
        value is better than the initial metric value. Otherwise it will return the default.
        If this flag is False, then the optimized transform is returned regardless.

    min_overlap_fraction : float (default: 0.01)
        Fix and mov's physical bounding box overlap (as seen through the
        moving initial transform built from `static_transform_list`) is
        logged before registration. If the overlap fraction is at or
        below this value, a corrective translation is composed on top of
        the moving initial transform to re-center the images, so the
        metric has some chance to run instead of raising "All samples
        map outside moving image buffer" and returning the default.
        Set to 0 to disable this fallback (correction never triggers).

    **kwargs : any additional arguments
        Passed to `configure_irm`
        This is where you would set things like:
        metric, iterations, shrink_factors, and smooth_sigmas

    Returns
    -------
    params : 1d array
        The complete set of control point parameters concatenated
        as a 1d array.

    field : ndarray
        The displacement field parameterized by the bspline control
        points
    """
    # store initial fixed image shape
    initial_fix_shape = fix.shape
    initial_fix_spacing = fix_spacing

    # format static transform data explicitly
    a, b = format_static_transform_data(
        static_transform_list, fix, fix_spacing, fix_origin,
    )
    static_transform_spacing = a
    static_transform_origin = b

    # realize masks
    fix_mask = realize_mask(fix, fix_mask)
    mov_mask = realize_mask(mov, mov_mask)

    # skip sample and convert inputs to sitk images
    X = apply_alignment_spacing(
        fix, mov,
        fix_mask, mov_mask,
        fix_spacing, mov_spacing,
        alignment_spacing,
        context=context,
    )
    fix, mov, fix_mask, mov_mask = images_to_sitk(
        *X, fix_origin, mov_origin,
    )
    fix_spacing = X[4]
    mov_spacing = X[5]
    fix_mask_spacing = X[6]
    mov_mask_spacing = X[7]

    # set up registration object
    irm = configure_irm(context=context, **kwargs)

    # allow control_point_spacing to be a scalar or a per-axis array; an
    # array shorter than the image dimensionality is extended by repeating
    # its last value, e.g. [64, 128] -> [64, 128, 128] for a 3D image.
    # Values are given in zyx order (matching fix_spacing), so reverse to
    # xyz to align with fix.GetSize()/fix.GetSpacing() (sitk order).
    ndim = fix.GetDimension()
    control_point_spacing = np.atleast_1d(control_point_spacing)
    if control_point_spacing.size < 1:
        raise ValueError('control_point_spacing must not be empty')
    if control_point_spacing.size > ndim:
        # truncate to the image dimensions
        control_point_spacing = control_point_spacing[:ndim]

    if control_point_spacing.size < ndim:
        control_point_spacing = np.concatenate([
            control_point_spacing,
            np.full(ndim - control_point_spacing.size, control_point_spacing[-1]),
        ])
    control_point_spacing_xyz = control_point_spacing[::-1]

    # initial control point grid
    cp_divisor = control_point_spacing_xyz * control_point_levels[-1]
    initial_cp_grid = [
        max(1, int(x*y/d))
        for x, y, d in zip(fix.GetSize(), fix.GetSpacing(), cp_divisor)
    ]
    transform = sitk.BSplineTransformInitializer(
        image1=fix, transformDomainMeshSize=initial_cp_grid, order=3,
    )
    logger.debug((
        f'{context} '
        'BSpline control point grid: '
        f'{fix.GetSize()}*{fix.GetSpacing()}/({control_point_spacing_xyz}*{control_point_levels[0]})={initial_cp_grid}, '
        f'BSpline transform {transform} '
    ))
    irm.SetInitialTransformAsBSpline(
        transform, inPlace=True, scaleFactors=control_point_levels[::-1],
    )

    # set initial static transforms
    moving_initial_transform = None
    if static_transform_list:
        moving_initial_transform = bst.transform_list_to_composite_transform(
            static_transform_list,
            static_transform_spacing,
            static_transform_origin,
        )

    # diagnose fix/mov physical overlap under the current moving initial
    # transform, and fall back to a corrective translation if it's
    # (near) zero -- otherwise the metric raises "All samples map outside
    # moving image buffer" and this block returns the identity default.
    moving_initial_transform = _check_and_correct_overlap(
        fix, mov, moving_initial_transform, min_overlap_fraction, context=context,
    )
    if moving_initial_transform is not None:
        irm.SetMovingInitialTransform(moving_initial_transform)

    # set masks
    if fix_mask is not None:
        irm.SetMetricFixedMask(fix_mask)
    if mov_mask is not None:
        irm.SetMetricMovingMask(mov_mask)

    # now we can set the default
    if not default:
        params = np.concatenate((transform.GetFixedParameters(), transform.GetParameters()))
        field = bst.bspline_to_displacement_field(
            transform, initial_fix_shape,
            spacing=initial_fix_spacing, origin=fix_origin,
            direction=np.eye(fix.GetDimension()),
        )
        default = (params, field)

    # execute alignment, for any exceptions return default
    try:
        initial_metric_value = irm.MetricEvaluate(fix, mov)
        irm.Execute(fix, mov)
        final_metric_value = irm.MetricEvaluate(fix, mov)
    except Exception as e:
        logger.error(f'{context} Registration failed due to ITK exception: {e}')
        logger.info(f'{context} Returning default')
        return default

    # if registration improved metric return result
    # otherwise return default
    if final_metric_check and final_metric_value > initial_metric_value:
        logger.warning((
            f'{context} Deform align optimization failed to improve metric: '
            f'initial: {initial_metric_value}, '
            f'final: {final_metric_value} '
        ))
        logger.info(f'{context} Deform align returning default')
        return default
    else:
        params = np.concatenate((transform.GetFixedParameters(), transform.GetParameters()))
        field = bst.bspline_to_displacement_field(
            transform, initial_fix_shape,
            spacing=initial_fix_spacing, origin=fix_origin,
            direction=np.eye(fix.GetDimension()),
        )

        # diagnostics: check the deformation field for folding and discontinuities
        deform_field_diagnostics(field, initial_fix_spacing, context=context)

        logger.info((
            f'{context} Deform align succeeded: '
            f'(initial_metric={initial_metric_value}, final_metric={final_metric_value}) '
        ))
        return params, field


def deform_field_diagnostics(field, spacing, context=''):
    """
    Log diagnostics for a displacement vector field: jacobian determinant
    (folding), field sanity (NaN/Inf), displacement magnitude statistics, and
    a crude discontinuity check in voxel index space.

    Parameters
    ----------
    field : nd-array
        The displacement vector field in zyx order, with the last axis holding
        the displacement components in zyx order.

    spacing : 1d array
        The physical voxel spacing of the field (zyx order).

    context : str (default: '')
        A prefix prepended to all log messages.
    """

    # build a sitk displacement field image (xyz vector order) from the
    # numpy field (zyx order, components in zyx)
    disp = ut.numpy_to_sitk(
        field[..., ::-1].astype(np.float64),
        spacing, vector=True,
    )
    # Jacobian determinant: det(I + grad(u)); values <= 0 indicate folding
    jac = sitk.DisplacementFieldJacobianDeterminant(disp)
    jac_arr = sitk.GetArrayFromImage(jac)
    n_folded = int(np.count_nonzero(jac_arr <= 0))
    logger.info((
        f'{context} Deform align jacobian determinant: '
        f'min={jac_arr.min()}, max={jac_arr.max()}, '
        f'mean={jac_arr.mean()}, '
        f'folded voxels (det<=0)={n_folded} '
        f'({100.0 * n_folded / jac_arr.size:.4f}%)'
    ))

    # field sanity and displacement magnitude statistics
    u = field
    mag = np.linalg.norm(u, axis=-1)
    logger.info((
        f'{context} Deform align field stats: '
        f'has NaN={bool(np.isnan(u).any())}, has Inf={bool(np.isinf(u).any())}, '
        f'disp magnitude min={mag.min()}, max={mag.max()}, '
        f'mean={mag.mean()}, p99={np.percentile(mag, 99)}'
    ))

    # crude discontinuity check in voxel index space
    gx = np.linalg.norm(np.diff(u, axis=2), axis=-1)
    gy = np.linalg.norm(np.diff(u, axis=1), axis=-1)
    gz = np.linalg.norm(np.diff(u, axis=0), axis=-1)
    for name, g in [("dx", gx), ("dy", gy), ("dz", gz)]:
        logger.info((
            f'{context} Deform align field smoothness {name}: '
            f'max jump={g.max()}, p99 jump={np.percentile(g, 99)}'
        ))


def alignment_pipeline(
    fix,
    mov,
    fix_spacing,
    mov_spacing,
    steps,
    fix_mask=None,
    mov_mask=None,
    fix_origin=None,
    mov_origin=None,
    static_transform_list=[],
    return_format='flatten',
    context='',
    **kwargs,
):
    """
    Compose random, rigid, affine, and deformable alignments with one function call

    Parameters
    ----------
    fix : ndarray
        the fixed image

    mov : ndarray
        the moving image; `fix.ndim` must equal `mov.ndim`

    fix_spacing : 1d array
        The spacing in physical units (e.g. mm or um) between voxels
        of the fixed image.
        Length must equal `fix.ndim`

    mov_spacing : 1d array
        The spacing in physical units (e.g. mm or um) between voxels
        of the moving image.

    steps : list of tuples in this form [(str, dict), (str, dict), ...]
        For each tuple, the str specifies which alignment to run. The options are:
        'ransac' : run `feature_point_ransac_affine_align`
        'random' : run `random_affine_search`
        'rigid' : run `affine_align` with `rigid=True`
        'affine' : run `affine_align`
        'deform' : run `deformable_align`
        'demons' : run `demons_align`
        'elastix' : run `elastix_align`
        'thinplate' : run `feature_point_ransac_thinplate_align`
        For each tuple, the dict specifies the arguments to that alignment function
        Arguments specified here override any global arguments given through kwargs
        for their specific step only.

    fix_mask : ndarray, tuple of floats, or function (default: None)
        A mask limiting metric evaluation region of the fixed image
        If an nd-array, any non-zero value is considered foreground and any
        zero value is considered background. If a tuple of floats, any voxel
        with value in the tuple is considered background. If a function, it
        must take a single nd-array argument as input and return an array
        of the same shape as the input but with dtype bool.

        If an nd-array, it is assumed to have the same domain as the fixed
        image, though sampling can be different. I.e. the origin and span
        are the same (in phyiscal units) but the number of voxels can
        be different.

    mov_mask : ndarray, tuple of floats, or function (default: None)
        A mask limiting metric evaluation region of the moving image
        If an nd-array, any non-zero value is considered foreground and any
        zero value is considered background. If a tuple of floats, any voxel
        with value in the tuple is considered background. If a function, it
        must take a single nd-array argument as input and return an array
        of the same shape as the input but with dtype bool.

        If an nd-array, it is assumed to have the same domain as the fixed
        image, though sampling can be different. I.e. the origin and span
        are the same (in phyiscal units) but the number of voxels can
        be different.

    fix_origin : 1d array (default: None)
        Origin of the fixed image.
        Length must equal `fix.ndim`

    mov_origin : 1d array (default: None)
        Origin of the moving image.
        Length must equal `mov.ndim`

    static_transform_list : list of numpy arrays (default: [])
        Transforms applied to moving image before applying query transform
        Assumed to have the same domain as the fixed image, though sampling
        can be different. I.e. the origin and span are the same (in phyiscal
        units) but the number of voxels can be different.

    return_format : str (default: 'flatten')
        The way in which transforms are returned to the user. Options are:
        'independent' : one transform per step is returned, no compositions
        'compressed' : adjacent affines and adjacent deforms are composed,
                       but affines are not composed with deforms. For example:
                       ['random', 'affine', 'deform', 'deform', 'affine', 'deform']
                       will return a list of 4 transforms.
        'flatten' : compose all transforms regardless of type into a single transform

    **kwargs : any additional keyword arguments
        Global arguments that apply to all alignment steps
        These are overwritten by specific arguments passed via
        the dictionaries in steps

    Returns
    -------
    transform : ndarray or tuple of ndarray
        Transform(s) aligning moving to fixed image.

        If neither 'deform' nor 'demons' is in `steps` then this is a single
        4x4 matrix -- all steps ('random', 'rigid', and/or 'affine') are composed.

        If 'deform' or 'demons' is in `steps` then this is a tuple. The first
        element is the composed 4x4 affine matrix, the second is the output of
        deformable_align or demons_align: a displacement field with shape
        equal to fix.shape + (ndim,).
    """

    # check default case
    if fix is None or mov is None:
        ndim = len(fix_spacing)
        field_steps = {'deform', 'demons', 'elastix', 'thinplate'}
        if field_steps & {x[0] for x in steps}:
            # if a field-producing step is present, create a zero displacement field
            shape = fix.shape if fix is not None else mov.shape
            return np.zeros(shape + (ndim,), dtype=np.float32)
        else:
            # otherwise return the identity
            return np.eye(ndim + 1)

    # lazy imports so bigstream.align has no hard dependency on bigstream.contrib
    from bigstream.contrib.demons_align import demons_align
    from bigstream.contrib.elastix_align import elastix_align

    # define how to run alignment functions
    a = (fix, mov, fix_spacing, mov_spacing)
    b = {'fix_mask':fix_mask, 'mov_mask':mov_mask,
         'fix_origin':fix_origin, 'mov_origin':mov_origin,}
    align = {'ransac':lambda **c: feature_point_ransac_affine_align(*a, **{**b, **c}),
             'random':lambda **c: random_affine_search(*a, **{**b, **c})[0],
             'rigid': lambda **c: affine_align(*a, **{**b, **c}, rigid=True),
             'affine':lambda **c: affine_align(*a, **{**b, **c}),
             'deform':lambda **c: deformable_align(*a, **{**b, **c})[1],
             'thinplate':lambda **c: feature_point_ransac_thinplate_align(*a, **{**b, **c})[1],
             'demons':lambda **c: demons_align(*a, **{**b, **c})[1],
             'elastix':lambda **c: elastix_align(*a, **{**b, **c})[1],
             }

    # loop over steps
    new_transforms = []
    for alignment, arguments in steps:
        logger.debug(f'{context} {alignment} args: {arguments}')
        arguments = {**kwargs, **arguments}
        # append new transforms to the static transforms
        arguments['static_transform_list'] = static_transform_list + new_transforms
        printable_args = { **arguments }
        printable_args['static_transform_list'] = [
            str(t) if t.ndim == 2 else f'{t.shape} deformfield'
            for t in arguments['static_transform_list']
        ]
        logger.debug(f'Run {context} {alignment} {printable_args}')
        alignment_function = alignment.split('-')[0]
        alignment_result = align[alignment_function](context=f'{alignment} {context}', **arguments)
        logger.debug(f'Completed {context} {alignment} {printable_args}')
        new_transforms.append(alignment_result)

    # return in the requested format
    if return_format == 'independent':
        return new_transforms
    elif return_format == 'compressed':
        return bst.compress_transform_list(new_transforms, [fix_spacing,]*len(new_transforms))[0]
    elif return_format == 'flatten':
        return bst.compose_transform_list(new_transforms, fix_spacing)


def ransac_masks_meta_align(
    fix,
    mov,
    fix_spacing,
    mov_spacing,
    steps,
    ransac_iterations,
    number_of_connected_components,
    neighborhood_radius,
    lcc_radius,
    lcc_spacing=None,
    number_of_transforms_to_return=None,
    fix_mask=None,
    mov_mask=None,
    fix_origin=None,
    mov_origin=None,
    static_transform_list=[],
    context='',
    **kwargs,
):
    """
    This will run the alignment_pipeline inside of a RANSAC algorithm.
    Random sampling is over the foreground mask for the fixed image. That is, for each
    RANSAC iteration, the alignment pipeline is run using a randomly generated foreground
    mask for the fixed image. When an alignment is complete, the local correlation coefficient (LCC)
    is computed around every voxel. A well aligned voxel is assumed to have LCC comparable
    to the median LCC of voxels in the ransac mask foreground region. The number of
    well aligned voxels is counted across the entire image. Results across ransac iterations
    are sorted according to this count.

    Random foreground masks are generated by randomly selecting points and drawing
    rectangular neighborhoods around them.

    This algorithm is useful for registering two images that have some shared content,
    but also content that is not shared (e.g., artifacts, non-overlapping field of view, etc.),
    or for pairs of images where the extent of shared content is difficult to determine or not
    known in advance. Generally it makes more sense to run this algorithm with affine alignment
    pipelines rather than deformations.

    The function inputs are the same as for the alignment_pipeline, with the addition of parameters
    related to the outer ransac loop. For example, the foreground mask number of connected components
    and their size.

    Parameters
    ----------
    fix : ndarray
        the fixed image

    mov : ndarray
        the moving image; `fix.ndim` must equal `mov.ndim`

    fix_spacing : 1d array
        The spacing in physical units (e.g. mm or um) between voxels
        of the fixed image.
        Length must equal `fix.ndim`

    mov_spacing : 1d array
        The spacing in physical units (e.g. mm or um) between voxels
        of the moving image.

    steps : list of tuples in this form [(str, dict), (str, dict), ...]
        For each tuple, the str specifies which alignment to run. The options are:
        'ransac' : run `feature_point_ransac_affine_align`
        'random' : run `random_affine_search`
        'rigid' : run `affine_align` with `rigid=True`
        'affine' : run `affine_align`
        'deform' : run `deformable_align`
        'demons' : run `demons_align`
        'elastix' : run `elastix_align`
        'thinplate' : run `feature_point_ransac_thinplate_align`
        For each tuple, the dict specifies the arguments to that alignment function
        Arguments specified here override any global arguments given through kwargs
        for their specific step only.

    ransac_iterations : int
        The number of ransac outer loop iterations to run, i.e., the number of
        foreground mask samples to try.

    number_of_connected_components : int or tuple of two ints
        The number of randomly selected points around which to draw neighborhoods
        for random foreground mask generation. If a tuple of two ints, then the number
        of connected components is randomly sampled between the two values (inclusive).

    neighborhood_radius : float
        A number in physical units (e.g., microns) that determines the size of foreground
        mask neighborhoods.

    lcc_radius : float
        A number in physical units (e.g., microns) that determines the size of neighborhoods
        for the local correlation coefficient metric calculation.

    lcc_spacing : float (default: None)
        The lcc calculation is expensive. You may want to speed it up by only evaluating
        it at a subset of voxels. lcc_spacing is a number in physical units (e.g. microns)
        that determines a spacing between lcc evaluation points. Increase this number
        to make the lcc step faster, but also less robust. If None, then no skip sampling
        is done.

    number_of_transforms_to_return : int (default: None)
        The top number_of_transforms_to_return transforms with respect to the well matched
        voxels score are returned. If None, all transforms are returned, sorted by their score.

    fix_mask : ndarray, tuple of floats, or function (default: None)
        A mask limiting metric evaluation region of the fixed image
        If an nd-array, any non-zero value is considered foreground and any
        zero value is considered background. If a tuple of floats, any voxel
        with value in the tuple is considered background. If a function, it
        must take a single nd-array argument as input and return an array
        of the same shape as the input but with dtype bool.

        If an nd-array, it is assumed to have the same domain as the fixed
        image, though sampling can be different. I.e. the origin and span
        are the same (in phyiscal units) but the number of voxels can
        be different.

        ransac sample foreground masks will be subsets of this global foreground
        mask.

    mov_mask : ndarray, tuple of floats, or function (default: None)
        A mask limiting metric evaluation region of the moving image
        If an nd-array, any non-zero value is considered foreground and any
        zero value is considered background. If a tuple of floats, any voxel
        with value in the tuple is considered background. If a function, it
        must take a single nd-array argument as input and return an array
        of the same shape as the input but with dtype bool.

        If an nd-array, it is assumed to have the same domain as the fixed
        image, though sampling can be different. I.e. the origin and span
        are the same (in phyiscal units) but the number of voxels can
        be different.

        ransac sample foreground masks will be intersected with this global
        foreground mask.

    fix_origin : 1d array (default: None)
        Origin of the fixed image.
        Length must equal `fix.ndim`

    mov_origin : 1d array (default: None)
        Origin of the moving image.
        Length must equal `mov.ndim`

    static_transform_list : list of numpy arrays (default: [])
        Transforms applied to moving image before applying query transform
        Assumed to have the same domain as the fixed image, though sampling
        can be different. I.e. the origin and span are the same (in phyiscal
        units) but the number of voxels can be different.

    **kwargs : any additional keyword arguments
        Global arguments that apply to all alignment steps
        These are overwritten by specific arguments passed via
        the dictionaries in steps

    Returns
    -------
    list_of_transforms : list of ndarray
        A list of transforms, length determined by the number_of_transforms_to_return
        parameter.

    scores : list of float
        Parallel list to list_of_transforms. The scores for each transform
    """

    # realize masks
    fix_mask = realize_mask(fix, fix_mask)
    mov_mask = realize_mask(mov, mov_mask)

    # determine skip sampling
    if lcc_spacing is None: lcc_spacing = fix_spacing
    skip_values = np.round( lcc_spacing / fix_spacing ).astype(int)
    skip_sampling = tuple(slice(None, None, x) for x in skip_values)

    # get baseline lcc image
    identity_aligned = bst.apply_transform(
        fix[skip_sampling], mov,
        fix_spacing * skip_values, mov_spacing,
        transform_list=static_transform_list + [np.eye(fix.ndim+1),],
        fix_origin=fix_origin,
        mov_origin=mov_origin,
    )
    _, identity_lcc_image = local_correlation_coefficient(
        fix[skip_sampling], identity_aligned,
        fix_spacing * skip_values,
        lcc_radius, return_image=True,
    )
    if fix_mask is not None:
        identity_lcc_image = identity_lcc_image * fix_mask[skip_sampling]

    # create containers for the ransac related components
    transforms, scores = [], []
    ransac_mask = np.empty(fix.shape, dtype=np.uint8)
    radius = np.round(neighborhood_radius / fix_spacing).astype(int)

    # start the ransac loop
    for iii in range(ransac_iterations):

        # determine the number of connected components for this iteration mask
        if isinstance(number_of_connected_components, (tuple,)):
            low, high = number_of_connected_components
            ncc = np.random.randint(low=low, high=high+1)
        else:
            ncc = number_of_connected_components

        # get the foreground points
        if fix_mask is None:
            points = np.random.randint(low=0, high=fix.shape, size=(ncc, fix.ndim))
        else:
            points = []
            while len(points) < ncc:
                point = np.random.randint(low=0, high=fix.shape, size=(1, fix.ndim))[0]
                _point = tuple(slice(x, x+1) for x in point)
                if fix_mask[_point]:
                    points.append(point)
            points = np.array(points)

        # create the foreground mask
        ransac_mask[...] = 0
        for point in points:
            neighborhood = tuple(slice(max(0, x-r), x+r+1) for x, r in zip(point, radius))
            ransac_mask[neighborhood] = 1
        if fix_mask is not None:
            ransac_mask = ransac_mask * fix_mask

        # call the alignment pipeline
        transform = alignment_pipeline(
            fix, mov, fix_spacing, mov_spacing, steps,
            fix_mask=ransac_mask,
            mov_mask=mov_mask,
            fix_origin=fix_origin,
            mov_origin=mov_origin,
            static_transform_list=static_transform_list,
            return_format='flatten',
            **kwargs,
        )

        # apply the transform    NOTE: static_transform_list spacings not considered, latent bug
        aligned = bst.apply_transform(
                fix[skip_sampling], mov,
                fix_spacing * skip_values, mov_spacing,
                transform_list=static_transform_list + [transform,],
                fix_origin=fix_origin,
                mov_origin=mov_origin,
        )

        # get the LCCs
        _, lcc_image = local_correlation_coefficient(
            fix[skip_sampling], aligned,
            fix_spacing * skip_values,
            lcc_radius, return_image=True,
        )
        if fix_mask is not None:
            lcc_image = lcc_image * fix_mask[skip_sampling]

        # count the improved voxels
        score = np.nansum( (lcc_image > identity_lcc_image).astype(np.uint32) )

        # store result
        transforms.append(transform)
        scores.append(score)

    # sort the transforms by score
    sort_indices = np.argsort(scores)[::-1]
    scores = np.array(scores)[sort_indices]
    transforms = np.array(transforms)[sort_indices]

    # return
    if number_of_transforms_to_return is not None:
        X = number_of_transforms_to_return
        return transforms[:X], scores[:X]
    else:
        return transforms, scores

