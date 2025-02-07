import bigstream.transform as bst
import cv2
import numpy as np
import SimpleITK as sitk
import bigstream.utility as ut
import logging

from bigstream.configure_irm import configure_irm
from bigstream.metrics import patch_mutual_information
from bigstream import features


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

    if mask is None: return None
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
    mov_mask_spacing = None
    if mov_mask is not None:
        mov_mask_spacing = ut.relative_spacing(mov_mask.shape,
                                               mov.shape,
                                               mov_spacing)

    # skip sample
    if alignment_spacing:
        fix, fix_spacing = ut.skip_sample(fix, fix_spacing, alignment_spacing)
        mov, mov_spacing = ut.skip_sample(mov, mov_spacing, alignment_spacing)
        if fix_mask is not None:
            fix_mask, fix_mask_spacing = ut.skip_sample(
                fix_mask, fix_mask_spacing, alignment_spacing,
            )
        if mov_mask is not None:
            mov_mask, mov_mask_spacing = ut.skip_sample(
                mov_mask, mov_mask_spacing, alignment_spacing,
            )

    return (fix, mov, fix_mask, mov_mask,
            fix_spacing, mov_spacing, fix_mask_spacing, mov_mask_spacing,)


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

    # realize masks
    fix_mask = realize_mask(fix, fix_mask)
    mov_mask = realize_mask(mov, mov_mask)

    # apply static transforms
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

    # skip sample and determine mask spacings
    X = apply_alignment_spacing(
        fix, mov,
        fix_mask, mov_mask,
        fix_spacing, mov_spacing,
        alignment_spacing,
    )
    fix = X[0]
    mov = X[1]
    fix_mask = X[2]
    mov_mask = X[3]
    fix_spacing = X[4]
    mov_spacing = X[5]
    fix_mask_spacing = X[6]
    mov_mask_spacing = X[7]

    # format inputs
    if type(cc_radius) not in (tuple,): cc_radius = (cc_radius,) * fix.ndim
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
    logger.info(f'{context} found {len(fix_spots)} fixed spots')
    if len(fix_spots) < fix_spots_count_threshold:
        logger.info(f'{context} insufficient fixed spots found')
        if safeguard_exceptions:
            raise ValueError('fix spot detection safeguard failed')
        else:
            logger.info(f'{context} returning default')
            return default

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
    logger.info(f'{context} found {len(mov_spots)} moving spots')
    if len(mov_spots) < mov_spots_count_threshold:
        logger.info('insufficient moving spots found')
        if safeguard_exceptions:
            raise ValueError('mov spot detection safeguard failed')
        else:
            logger.info(f'{context} returning default')
            return default

    # sort
    logger.info(f'{context} sorting spots')
    sort_idx = np.argsort(fix_spots[:, 3])[::-1]
    fix_spots = fix_spots[sort_idx, :3][:nspots]
    sort_idx = np.argsort(mov_spots[:, 3])[::-1]
    mov_spots = mov_spots[sort_idx, :3][:nspots]

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
    logger.info(f'{len(fix_spots)} - {len(mov_spots)} matched spots')
    if len(fix_spots) < point_matches_threshold or len(mov_spots) < point_matches_threshold:
        logger.info('insufficient point matches found')
        if safeguard_exceptions:
            raise ValueError('point matches safeguard failed')
        else:
            logger.info('returning default')
            return default

    # align
    logger.debug(f'{context} Found enough spots to estimate the affine ' +
                 f'fix: {len(fix_spots)}' +
                 f'moving: {len(mov_spots)}')
    _, Aff, _ = cv2.estimateAffine3D(
        fix_spots, mov_spots,
        ransacThreshold=align_threshold,
        confidence=0.999,
        **kwargs,
    )

    # ensure affine is sensible
    if np.any( np.abs(np.diag(Aff) - 1) > diagonal_constraint ):
        logger.info(f'{context} Degenerate affine produced')
        if safeguard_exceptions:
            raise ValueError('diagonal_constraint safeguard failed')
        else:
            logger.info(f'{context} returning default')
            return default

    # augment matrix and return
    affine = np.eye(fix.ndim + 1)
    affine[:fix.ndim, :] = Aff
    return affine


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
    F = lambda mx: 2 * (mx * np.random.rand(random_iterations, 3)) - mx
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
            logger.debug('Best score found {iii} : {current_best_score}')

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
    if default is None: default = np.eye(fix.ndim + 1)
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
    # TODO: enable initialization with second moment as well
    if isinstance(initial_condition, str) and initial_condition == "CENTER":
        a, b = fix, mov
        x = sitk.CenteredTransformInitializer(a, b, rigid_transform_constructor())
        x = rigid_transform_constructor(x).GetTranslation()[::-1]
        initial_condition = np.eye(ndims+1)
        initial_condition[:ndims, -1] = x
        initial_transform_given = True
    if rigid and not initial_transform_given:
        transform = rigid_transform_constructor()
    elif rigid and initial_transform_given:
        transform = bst.matrix_to_euler_transform(initial_condition)
    elif not rigid and not initial_transform_given:
        transform = sitk.AffineTransform(fix.GetDimension())
    elif not rigid and initial_transform_given:
        transform = bst.matrix_to_affine_transform(initial_condition)
    irm.SetInitialTransform(transform, inPlace=True)
    # set masks
    if fix_mask is not None: irm.SetMetricFixedMask(fix_mask)
    if mov_mask is not None: irm.SetMetricMovingMask(mov_mask)

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
    if final_metric_value < initial_metric_value:
        logger.info(f'{context} Registration succeeded')
        return bst.affine_transform_to_matrix(transform)
    else:
        logger.warn(f'{context} Optimization failed to improve metric')
        logger.info(f'METRIC VALUES initial: {context} ',
                    f'{initial_metric_value} final: {final_metric_value}')
        logger.info(f'{context} Returning default')
        return default


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

    control_point_spacing : float
        The spacing in physical units (e.g. mm or um) between control
        points that parameterize the deformation. Smaller means
        more precise alignment, but also longer compute time. Larger
        means shorter compute time and smoother transform, but less
        precise.

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

    # initial control point grid
    z = control_point_spacing * control_point_levels[0]
    initial_cp_grid = [max(1, int(x*y/z)) for x, y in zip(fix.GetSize(), fix.GetSpacing())]
    transform = sitk.BSplineTransformInitializer(
        image1=fix, transformDomainMeshSize=initial_cp_grid, order=3,
    )
    irm.SetInitialTransformAsBSpline(
        transform, inPlace=True, scaleFactors=control_point_levels[::-1],
    )

    # set initial static transforms
    if static_transform_list:
        T = bst.transform_list_to_composite_transform(
            static_transform_list,
            static_transform_spacing,
            static_transform_origin,
        )
        irm.SetMovingInitialTransform(T)
    # set masks
    if fix_mask is not None: irm.SetMetricFixedMask(fix_mask)
    if mov_mask is not None: irm.SetMetricMovingMask(mov_mask)

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
    if final_metric_value < initial_metric_value:
        params = np.concatenate((transform.GetFixedParameters(), transform.GetParameters()))
        field = bst.bspline_to_displacement_field(
            transform, initial_fix_shape,
            spacing=initial_fix_spacing, origin=fix_origin,
            direction=np.eye(fix.GetDimension()),
        )
        logger.info(f'{context} Registration succeeded')
        return params, field
    else:
        logger.warn(f'{context} Optimization failed to improve metric')
        logger.info(f'{context} METRIC VALUES initial: {initial_metric_value} ',
                    f'final: {final_metric_value}')
        logger.info(f'{context} Returning default')
        return default


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

        If 'deform' is not in `steps` then this is a single 4x4 matrix - all
        steps ('random', 'rigid', and/or 'affine') are composed.

        If 'deform' is in `steps` then this is a tuple. The first element
        is the composed 4x4 affine matrix, the second is the output of
        deformable align: a tuple with the bspline parameters and the
        vector field with shape equal to fix.shape + (3,)
    """
    # define how to run alignment functions
    a = (fix, mov, fix_spacing, mov_spacing)
    b = {'fix_mask':fix_mask, 'mov_mask':mov_mask,
         'fix_origin':fix_origin, 'mov_origin':mov_origin,}
    align = {'ransac':lambda **c: feature_point_ransac_affine_align(*a, **{**b, **c}),
             'random':lambda **c: random_affine_search(*a, **{**b, **c})[0],
             'rigid': lambda **c: affine_align(*a, **{**b, **c}, rigid=True),
             'affine':lambda **c: affine_align(*a, **{**b, **c}),
             'deform':lambda **c: deformable_align(*a, **{**b, **c})[1],}

    # loop over steps
    new_transforms = []
    for alignment, arguments in steps:
        logger.debug(f'Run {context} {alignment} {arguments}')
        arguments = {**kwargs, **arguments}
        logger.debug(f'All {alignment} args: {arguments}')
        arguments['static_transform_list'] = static_transform_list + new_transforms
        alignment_result = align[alignment](context=context, **arguments)
        logger.debug(f'Completed {context} {alignment} {arguments}')
        new_transforms.append(alignment_result)

    # return in the requested format
    if return_format == 'independent':
        return new_transforms
    elif return_format == 'compressed':
        return bst.compress_transform_list(new_transforms, [fix_spacing,]*len(new_transforms))[0]
    elif return_format == 'flatten':
        return bst.compose_transform_list(new_transforms, fix_spacing)
