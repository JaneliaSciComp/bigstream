import numpy as np
import SimpleITK as sitk
import bigstream.utility as ut
from bigstream.configure_irm import interpolator_switch
import os
from scipy.ndimage import map_coordinates


def apply_transform(
    fix, mov,
    fix_spacing, mov_spacing,
    transform_list,
    transform_spacing=None,
    transform_origin=None,
    fix_origin=None,
    mov_origin=None,
    interpolator='1',
    extrapolate_with_nn=False,
    compress_transforms=False,
):
    """
    Resample moving image onto fixed image through a list
    of transforms.

    Parameters
    ----------
    fix : ndarray
        the fixed image
        Optionally, this can be a tuple specifying a shape.

    mov : ndarray
        the moving image; `fix.ndim` must equal `mov.ndim`

    fix_spacing : 1d array
        The spacing in physical units (e.g. mm or um) between voxels
        of the fixed image. Length must equal `fix.ndim`.

    mov_spacing : 1d array
        The spacing in physical units (e.g. mm or um) between voxels
        of the moving image. Length must equal `mov.ndim`.

    transform_list : list
        The list of transforms to apply. These may be 2d arrays of shape 3x3 or 4x4
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
        the same format as transform_spacing.

    fix_origin : None (defaut) or 1darray
        The origin in physical units (e.g. mm or um) of the fixed image. If None
        the origin is assumed to be (0, 0, 0, ...)

    mov_origin : None (default) or 1darray
        The origin in physical units (e.g. mm or um) of the moving image. If None
        the origin is assumed to be (0, 0, 0, ...)

    interpolator : string (default: '1')
        Which interpolation to use.
        See bigstream.configure_irm.configure_irm documentation for options

    extrapolate_with_nn : Bool (default: False)
        If true extrapolations are done with Nearest Neighbors. Use if warping
        segmentation/multi-label data. Also prevents edge effects from padding
        when warping image data.

    compress_transforms : Bool (default: False)
        If False, all transforms are independently added to the composite transform
        before it is applied to the moving image. If True, we compose all neighboring
        transforms of similar type in the transform list before applying. For example,
        with this transform list, [affine1, affine2, deform1, deform2, affine3] and
        compress_transform_list == False, the composite transform will include all five
        transforms as indpendent objects. Especially when more than one deformation is
        applied, this can sometimes result in interpolation artifacts on the edges of
        images. With compress_transform_list == True, the given list would compress to
        [composed_affine, composed_deform, affine3]. This reduces the likelihood of
        edge artifacts, but is also slower.

    Returns
    -------
    warped image : ndarray
        The moving image warped through transform_list and resampled onto the
        fixed image grid.
    """

    # set global number of threads
    ncores = ut.get_number_of_cores()
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2*ncores)

    # format transform spacing
    fix_spacing = np.array(fix_spacing)
    if transform_spacing is None: transform_spacing = fix_spacing
    if not isinstance(transform_spacing, tuple):
        transform_spacing = (transform_spacing,) * len(transform_list)

    # construct transform
    if compress_transforms:
        transform_list, transform_spacing = compress_transform_list(
            transform_list, list(transform_spacing),
        )
        transform_spacing = tuple(transform_spacing)
    transform = ut.transform_list_to_composite_transform(
        transform_list, transform_spacing, transform_origin,
    )

    # set up resampler object
    resampler = sitk.ResampleImageFilter()
    resampler.SetNumberOfThreads(2*ncores)

    # set reference data
    if isinstance(fix, tuple):
        dtype = mov.dtype
        resampler.SetSize(fix[::-1])
        resampler.SetOutputSpacing(fix_spacing[::-1])
        if fix_origin is not None:
            resampler.SetOutputOrigin(fix_origin[::-1])
    else:
        dtype = fix.dtype
        fix = sitk.Cast(ut.numpy_to_sitk(fix, fix_spacing, fix_origin), sitk.sitkFloat32)
        resampler.SetReferenceImage(fix)

    # set moving image and transform
    mov = sitk.Cast(ut.numpy_to_sitk(mov, mov_spacing, mov_origin), sitk.sitkFloat32)
    resampler.SetTransform(transform)

    # set interpolator
    resampler.SetInterpolator(interpolator_switch[interpolator])

    # check for NN extrapolation
    if extrapolate_with_nn:
        resampler.SetUseNearestNeighborExtrapolator(True)

    # execute, return as numpy array
    resampled = resampler.Execute(mov)
    return sitk.GetArrayViewFromImage(resampled).astype(dtype)


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
        The transforms to apply, in stack order. Elements must be 3x3 or 4x4
        arrays (affine transforms) or d + 1 dimension ndarrays (deformations).

    transform_spacing : None (default), 1d array, or tuple (not list) of 1d arrays
        The spacing in physical units (e.g. mm or um) between voxels
        of any deformations in transform_list. If transform_list
        contains any deformations then transform_spacing cannot be
        None. If a single 1d array then all deforms have that spacing.
        If a tuple, then its length must be the same as transform_list,
        thus each deformation can be given its own spacing. Spacings
        given for affine transforms are ignored.

    transform_origin : None (default), 1d array, or tuple (not list) of 1d arrays
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
        if len(transform.shape) == 2:

            # matrix vector multiply
            ndims = transform.shape[0] - 1
            mm, tt = transform[:ndims, :ndims], transform[:ndims, -1]
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
            ndims = transform.shape[-1]
            interp = lambda x: map_coordinates(x, coordinates, mode='nearest')
            dX = np.array([interp(transform[..., i]) for i in range(ndims)]).transpose()
            coordinates = coordinates.transpose() * spacing + dX
            if origin is not None: coordinates += origin

    return coordinates


def compose_displacement_vector_fields(
    first_field,
    second_field,
    first_spacing,
    second_spacing,
):
    """
    Compose two displacement vector fields into a single field

    Parameters
    ----------
    first_field : nd-array
        The first field

    second_field : nd-array
        The second field

    first_spacing : 1d-array
        The voxel spacing for the first field

    second_spacing : 1d-array
        The voxel spacing for the second field

    Returns
    -------
    composite_field : nd-array
        The single field composition of first_field and second_field
        The second field is the one learned second. Thus, the composition
        is going to be on the same voxel grid and spacing as the second field.
    """

    # container for warped first field
    first_field_warped = np.empty_like(second_field)

    # loop over components
    for iii in range(len(first_spacing)):

        # warp first field with second
        first_field_warped[..., iii] = apply_transform(
            second_field[..., iii], first_field[..., iii],
            second_spacing, first_spacing,
            transform_list=[second_field,],
            extrapolate_with_nn=True,
        )

    # combine warped first field and second field
    return first_field_warped + second_field


def compose_transforms(
    first_transform,
    second_transform,
    first_spacing,
    second_spacing,
):
    """
    Compose two transforms into a single transform

    Parameters
    ----------
    first_transform : nd-array
        Can be either a 3x3 or 4x4 affine matrix or a displacement vector field

    second_transform : nd-array
        Can be either a 3x3 or 4x4 affine matrix or a displacement vector field

    first_spacing : 1d-array
        The voxel spacing for the first transform
        Ignored for affine transforms (just put in a dummy value)

    second_spacing : 1d-array
        The voxel spacing for the second transform
        Ignored for affine transforms (just put in a dummy value)

    Returns
    -------
    composite_transform : nd-array
        The single transform composition of first_transform and second_transform
        If both given transforms are affine this is a 4x4 matrix. Otherwise,
        it is a displacement vector field.
    """

    # two affines
    if len(first_transform.shape) == 2 and len(second_transform.shape) == 2:
        return np.matmul(first_transform, second_transform)

    # one affine, two field
    elif len(first_transform.shape) == 2:
        first_transform = ut.matrix_to_displacement_field(
            first_transform, second_transform.shape[:-1], second_spacing,
        )
        first_spacing = second_spacing

    # one field, two affine
    elif len(second_transform.shape) == 2:
        second_transform = ut.matrix_to_displacement_field(
            second_transform, first_transform.shape[:-1], first_spacing,
        )
        second_spacing = first_spacing

    # compose fields
    return compose_displacement_vector_fields(
        first_transform, second_transform, first_spacing, second_spacing,
    )


def compose_transform_list(transforms, spacings):
    """
    Compose a list of transforms into a single transform

    Parameters
    ----------
    transforms : list
        Elements of list must be either 3x3 or 4x4 affine matrices or displacement
        vector fields

    spacings : list of 1d-arrays
        The voxel spacing of all transforms in the list
        Ignored for affine transforms (just put in a dummy value)

    Returns
    -------
    composite_transform : nd-array
        The single transform composition of all elements in transforms.
        If all transforms are affine, this is a 3x3 or 4x4 matrix. Otherwise,
        it is a displacement vector field.
    """

    # ensure spacings is a list
    if not isinstance(spacings, list):
        spacings = [spacings,] * len(transforms)

    transform = transforms.pop()
    transform_spacing = spacings.pop()
    while transforms:
        transform = compose_transforms(
            transforms.pop(), transform,
            spacings.pop(), transform_spacing,
        )
    return transform


def compress_transform_list(transforms, spacings):
    """
    Separately compose all neighboring transforms of the same type
    For example, [affine1, affine2, deform1, deform2, affine3, deform3]
    becomes [composed_affine, composed_deform, affine3, deform3]

    Parameters
    ----------
    transforms : list
        Elements of list must be either 3x3 or 4x4 affine matrices or displacement
        vector fields

    spacings : list of 1d-arrays
        The voxel spacing of all transforms in the list
        Ignored for affine transforms (just put in a dummy value)

    Returns
    -------
    compressed_transform_list : list of nd-arrays
        A list of transforms where all neighboring transforms of the same type
        are composed
    compressed_spacings_list : list of 1d-arrays
        A list of spacings for the transforms in compressed_transform_list
    """

    if len(transforms) == 2:
        dims = [len(x.shape) for x in transforms]
        if dims[0] == dims[1]:
            transforms = [compose_transform_list(transforms, spacings),]
            spacings = [spacings[1],]
    if len(transforms) > 2:
        dims = np.array([len(x.shape) for x in transforms])
        changes = np.where(dims[:-1] != dims[1:])[0] + 1
        changes = [0,] + list(changes) + [len(transforms),]
        F = lambda a, b: compose_transform_list(transforms[a:b], spacings[a:b])
        transforms = [F(a, b) for a, b in zip(changes[:-1], changes[1:])]
        spacings = [spacings[x-1] for x in changes[1:]]
    return transforms, spacings


def invert_displacement_vector_field(
    field,
    spacing,
    step=0.5,
    iterations=10,
    sqrt_order=2,
    sqrt_step=0.5,
    sqrt_iterations=5,
    verbose=True,
):
    """
    Numerically find the inverse of a displacement vector field.

    Parameters
    ----------
    field : nd-array
        The displacement vector field to invert

    spacing : 1d-array
        The physical voxel spacing of the displacement field

    step : float (default: 0.5)
        The step size used for each iteration of the stationary point algorithm

    iterations : scalar int (default: 10)
        The number of stationary point iterations to find inverse. More
        iterations gives a more accurate inverse but takes more time.

    sqrt_order : scalar int (default: 2)
        The number of roots to take before stationary point iterations

    sqrt_step : float (default: 0.5)
        The step size used for each iteration of the composition square root gradient descent

    sqrt_iterations : scalar int (default: 5)
        The number of iterations to find the field composition square root

    verbose : bool (default: True)
        Whether or not to print optimization feedback to standard output

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
        field, spacing, sqrt_order, sqrt_step, sqrt_iterations,
        verbose=verbose,
    )
    inv = - np.copy(root)

    # iterate to invert
    if verbose: print('INVERTING ROOT')
    for i in range(iterations):
        residual = compose_transforms(root, inv, spacing, spacing)
        inv -= step * residual
        if verbose:
            residual_magnitude = np.linalg.norm(residual)
            print(f'FITTING INVERSE  -  Iteration: {i} --> Residual: {residual_magnitude}')
    if verbose: print('', flush=True)

    # square-compose inv order times
    for i in range(sqrt_order):
        inv = compose_transforms(inv, inv, spacing, spacing)

    # return result
    return inv


def _displacement_field_composition_nth_square_root(
    field,
    spacing,
    order,
    step,
    iterations,
    verbose=True,
):
    """
    """

    # initialize with given field
    root = np.copy(field)

    # iterate taking square roots
    for i in range(order):
        if verbose: print(f'FINDING ROOT ORDER {i}')
        root = _displacement_field_composition_square_root(
            root, spacing, step, iterations, verbose=verbose,
        )

    # return result
    return root


def _displacement_field_composition_square_root(
    field,
    spacing,
    step,
    iterations,
    verbose=True,
):
    """
    """

    # container to hold root
    root = 0.5 * field

    # iterate
    for i in range(iterations):
        residual = (field - compose_transforms(root, root, spacing, spacing))
        root += step * residual
        if verbose:
            residual_magnitude = np.linalg.norm(residual)
            print(f'FITTING ROOT  -  Iteration: {i} --> Residual: {residual_magnitude}')

    # return result
    return root


def displacement_field_jacobian(field, spacing):
    """
    """

    # convert to position field
    grid = tuple(slice(None, x) for x in field.shape[:-1])
    position_field = np.mgrid[grid].astype(field.dtype)
    position_field = np.moveaxis(position_field, 0, -1) * spacing + field

    # get jacobian matrices
    jacobian = np.empty(field.shape[:-1] + (field.shape[-1],)*2, dtype=field.dtype)
    for iii in range(field.shape[-1]):
        grad = np.moveaxis( np.array( np.gradient(position_field[..., iii], *spacing) ), 0, -1)
        jacobian[..., iii, :] = np.ascontiguousarray(grad)
    return jacobian


def displacement_field_jacobian_determinant(field, spacing):
    """
    """

    jacobian = displacement_field_jacobian(field, spacing)
    return np.linalg.det(jacobian)
