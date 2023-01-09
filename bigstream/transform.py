import numpy as np
import SimpleITK as sitk
import bigstream.utility as ut
import os, psutil
from scipy.ndimage import map_coordinates


def apply_transform(
    fix, mov,
    fix_spacing, mov_spacing,
    transform_list,
    transform_spacing=None,
    transform_origin=None,
    fix_origin=None,
    mov_origin=None,
    interpolate_with_nn=False,
    extrapolate_with_nn=False,
):
    """
    Resample moving image onto fixed image through a list
    of transforms.

    Parameters
    ----------
    fix : ndarray
        the fixed image

    mov : ndarray
        the moving image; `fix.ndim` must equal `mov.ndim`

    fix_spacing : 1d array
        The spacing in physical units (e.g. mm or um) between voxels
        of the fixed image. Length must equal `fix.ndim`.

    mov_spacing : 1d array
        The spacing in physical units (e.g. mm or um) between voxels
        of the moving image. Length must equal `mov.ndim`.

    transform_list : list
        The list of transforms to apply. These may be 2d arrays of shape 4x4
        (affine transforms), or ndarrays of `fix.ndim` + 1 dimension (deformations).
        Zarr arrays work just fine.

    transform_spacing : None (default), 1d array, or tuple of 1d arrays
        The spacing in physical units (e.g. mm or um) between voxels
        of any deformations in transform_list. If None, all deforms
        are assumed to have fix_spacing. If a single 1d array all
        deforms have that spacing. If a tuple, then it's length must
        be the same as transform_list, thus each deformation can be
        given its own spacing. Spacings given for affine transforms
        are ignored.

    transform_origin : None (default), 1d array, or tuple of 1d arrays
        The origin in physical units (e.g. mm or um) of the given transforms.
        If None, all origins are assumed to be (0, 0, 0, ...); otherwise, follows
        the same logic as transform_spacing.

    fix_origin : None (defaut) or 1darray
        The origin in physical units (e.g. mm or um) of the fixed image. If None
        the origin is assumed to be (0, 0, 0, ...)

    mov_origin : None (default) or 1darray
        The origin in physical units (e.g. mm or um) of the moving image. If None
        the origin is assumed to be (0, 0, 0, ...)

    interpolate_with_nn : Bool (default: False)
        If true interpolations are done with Nearest Neighbors. Use if warping
        segmentation/multi-label data.

    extrapolate_with_nn : Bool (default: False)
        If true extrapolations are done with Nearest Neighbors. Use if warping
        segmentation/multi-label data. Also prevents edge effects from padding
        when warping image data.

    Returns
    -------
    warped image : ndarray
        The moving image warped through transform_list and resampled onto the
        fixed image grid.
    """

    # set global number of threads
    if "LSB_DJOB_NUMPROC" in os.environ:
        ncores = int(os.environ["LSB_DJOB_NUMPROC"])
    else:
        ncores = psutil.cpu_count(logical=False)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2*ncores)

    # convert images to sitk objects
    dtype = fix.dtype
    fix = sitk.Cast(ut.numpy_to_sitk(fix, fix_spacing, fix_origin), sitk.sitkFloat32)
    mov = sitk.Cast(ut.numpy_to_sitk(mov, mov_spacing, mov_origin), sitk.sitkFloat32)

    # construct transform
    fix_spacing = np.array(fix_spacing)
    if transform_spacing is None: transform_spacing = fix_spacing
    transform = ut.transform_list_to_composite_transform(
        transform_list, transform_spacing, transform_origin,
    )

    # set up resampler object
    resampler = sitk.ResampleImageFilter()
    resampler.SetNumberOfThreads(2*ncores)
    resampler.SetReferenceImage(fix)
    resampler.SetTransform(transform)

    # check for NN interpolation
    if interpolate_with_nn:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    # check for NN extrapolation
    if extrapolate_with_nn:
        resampler.SetUseNearestNeighborExtrapolator(True)

    # execute, return as numpy array
    resampled = resampler.Execute(mov)
    return sitk.GetArrayFromImage(resampled).astype(dtype)


def apply_transform_to_coordinates(
    coordinates,
    transform_list,
    transform_spacing=None,
    transform_origin=None,
):
    """
    Move a set of coordinates through a list of transforms

    Parameters
    ----------
    coordinates : Nxd array
        The coordinates to move. N such coordinates in d dimensions.

    transform_list : list
        The transforms to apply, in stack order. Elements must be 2d 4x4 arrays
        (affine transforms) or d + 1 dimension ndarrays (deformations).

    transform_spacing : None (default), 1d array, or tuple of 1d arrays
        The spacing in physical units (e.g. mm or um) between voxels
        of any deformations in transform_list. If any transform_list
        contains any deformations then transform_spacing cannot be
        None. If a single 1d array then all deforms have that spacing.
        If a tuple, then its length must be the same as transform_list,
        thus each deformation can be given its own spacing. Spacings
        given for affine transforms are ignored.

    transform_origin : None (default), 1d array, or tuple of 1d arrays
        The origin in physical units (e.g. mm or um) of the given transforms.
        If None, all origins are assumed to be (0, 0, 0, ...); otherwise, follows
        the same logic as transform_spacing. Origins given for affine transforms
        are ignored.

    Returns
    -------
    transform_coordinates : Nxd array
        The given coordinates transformed by the given transform_list
    """

    # transform list should be a stack, last added is first applied
    for iii, transform in enumerate(transform_list[::-1]):

        # if transform is an affine matrix
        if transform.shape == (4, 4):

            # matrix vector multiply
            mm, tt = transform[:3, :3], transform[:3, -1]
            coordinates = np.einsum('...ij,...j->...i', mm, coordinates) + tt

        # if transform is a deformation vector field
        else:

            # transform_spacing must be given
            error_message = "If transform is a displacement vector field, "
            error_message += "transform_spacing must be given."
            assert (transform_spacing is not None), error_message

            # handle multiple spacings and origins
            spacing = transform_spacing
            origin = transform_origin
            if isinstance(spacing, tuple): spacing = spacing[iii]
            if isinstance(origin, tuple): origin = origin[iii]

            # get coordinates in transform voxel units, reformat for map_coordinates
            if origin is not None: coordinates -= origin
            coordinates = ( coordinates / spacing ).transpose()
    
            # interpolate position field at coordinates, reformat, return
            interp = lambda x: map_coordinates(x, coordinates, order=1, mode='nearest')
            dX = np.array([interp(transform[..., i]) for i in range(3)]).transpose()
            coordinates = coordinates.transpose() * spacing + dX
            if origin is not None: coordinates += origin

    return coordinates


def compose_displacement_vector_fields(
    first_field,
    second_field,
    spacing,
):
    """
    Compose two displacement vector fields into a single field

    Parameters
    ----------
    first_field : nd-array
        The first field

    second_field : nd-array
        The second field

    spacing : 1d-array
        The voxel spacing for the two fields (fields must have same spacing)

    Returns
    -------
    composite_field : nd-array
        The single field composition of first_field and second_field
    """

    # container for warped first field
    first_field_warped = np.empty_like(first_field)

    # loop over components
    for iii in range(3):

        # warp first field with second
        first_field_warped[..., iii] = apply_transform(
            first_field[..., iii], first_field[..., iii],
            spacing, spacing,
            transform_list=[second_field,],
            extrapolate_with_nn=True,
        )

    # combine warped first field and second field
    return first_field_warped + second_field


def compose_transforms(transform_one, transform_two, spacing):
    """
    Compose two transforms into a single transform

    Parameters
    ----------
    transform_one : nd-array
        Can be either a 4x4 affine matrix or a displacement vector field

    transform_two : nd-array
        Can be either a 4x4 affine matrix or a displacement vector field

    spacing : 1d-array
        The voxel spacing for the two transforms (transforms must have same spacing)
        Ignored for affine matrices.

    Returns
    -------
    composite_transform : nd-array
        The single transform composition of transform_one and transform_two
        If both given transforms are affine this is a 4x4 matrix. Otherwise,
        it is a displacement vector field.
    """

    # two affines
    if transform_one.shape == (4, 4) and transform_two.shape == (4, 4):
        return np.matmul(transform_one, transform_two)

    # one affine, two field
    elif transform_one.shape == (4, 4):
        transform_one = ut.matrix_to_displacement_field(
            transform_one, transform_two.shape[:-1], spacing,
        )

    # one field, two affine
    elif transform_two.shape == (4, 4):
        transform_two = ut.matrix_to_displacement_field(
            transform_two, transform_one.shape[:-1], spacing,
        )

    # compose fields
    return compose_displacement_vector_fields(
        transform_one, transform_two, spacing,
    )


def compose_transform_list(transforms, spacing):
    """
    Compose a list of transforms into a single transform

    Parameters
    ----------
    transforms : list
        Elements of list must be either 4x4 affine matrices or displacement
        vector fields

    spacing : 1d-array
        The voxel spacing of all transforms in the list (all transforms must
        have the same spacing). Ignored for affine transforms.

    Returns
    -------
    composite_transform : nd-array
        The single transform composition of all elements in transforms.
        If all transforms are affine, this is a 4x4 matrix. Otherwise,
        it is a displacement vector field.
    """

    transform = transforms.pop()
    while transforms:
        transform = compose_transforms(transforms.pop(), transform, spacing)
    return transform


def invert_displacement_vector_field(
    field,
    spacing,
    iterations=10,
    order=2,
    sqrt_iterations=5,
):
    """
    Numerically find the inverse of a displacement vector field.

    Parameters
    ----------
    field : nd-array
        The displacement vector field to invert

    spacing : 1d-array
        The physical voxel spacing of the displacement field

    iterations : scalar int (default: 10)
        The number of stationary point iterations to find inverse. More
        iterations gives a more accurate inverse but takes more time.

    order : scalar int (default: 2)
        The number of roots to take before stationary point iterations

    sqrt_iterations : scalar int (default: 5)
        The number of iterations to find the field composition square root

    Returns
    -------
    inverse_field : nd-array
        The numerical inverse of the given displacement vector field.
        field(inverse_field) should be nearly zeros everywhere.
        inverse_field(field) should be nearly zeros everywhere.
        If precision is not high enough, look at iterations,
        order, and sqrt_iterations.
    """

    # initialize inverse as negative root
    root = _displacement_field_composition_nth_square_root(
        field, spacing, order, sqrt_iterations,
    )
    inv = - np.copy(root)

    # iterate to invert
    for i in range(iterations):
        inv -= compose_transforms(root, inv, spacing)

    # square-compose inv order times
    for i in range(order):
        inv = compose_transforms(inv, inv, spacing)

    # return result
    return inv


def _displacement_field_composition_nth_square_root(
    field,
    spacing,
    order,
    sqrt_iterations=5,
):
    """
    """

    # initialize with given field
    root = np.copy(field)

    # iterate taking square roots
    for i in range(order):
        root = _displacement_field_composition_square_root(
            root, spacing, iterations=sqrt_iterations,
        )

    # return result
    return root


def _displacement_field_composition_square_root(
    field,
    spacing,
    iterations=5,
):
    """
    """

    # container to hold root
    root = 0.5 * field

    # iterate
    for i in range(iterations):
        residual = (field - compose_transforms(root, root, spacing))
        root += 0.5 * residual

    # return result
    return root


