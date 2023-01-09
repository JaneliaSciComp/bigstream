import numpy as np
from scipy.spatial.transform import Rotation
import SimpleITK as sitk
import zarr
from zarr.indexing import BasicIndexer
from numcodecs import Blosc
from distributed import Lock


def skip_sample(image, spacing, ss_spacing):
    """
    Return a subset of the voxels in an image to approximate a different
    sampling rate

    Parameters
    ----------
    image : nd-array
        The image to sub-sample

    spacing : 1d-array
        The physical spacing of the voxels in image

    ss_spacing : 1d-array
        The desired physical spacing to approximate by skipping voxels

    Returns
    -------
    smaller_image : nd-array
        a subset of voxels from image with a new spacing as close as
        possible to ss_spacing

    new_spacing : 1d-array
        the physical spacing of smaller_image
    """

    spacing = np.array(spacing)
    ss = np.maximum(np.round(ss_spacing / spacing), 1).astype(int)
    slc = tuple(slice(None, None, x) for x in ss)
    return image[slc], spacing * ss


def numpy_to_sitk(image, spacing=None, origin=None, vector=False):
    """
    Convert a numpy array to a sitk image object

    Parameters
    ----------
    image : nd-array
        The image data

    spacing : 1d-array (default: None)
        The physical spacing of image

    origin : 1d-array (default: None)
        The physical origin of image

    vector : bool (default:False)
        If the last axis of image is a vector dimension

    Returns
    -------
    sitk_image : sitk.image object
        The given image data with correct spacing and origin set
    """

    # check endianness of data - some sitk operations seem to
    # only work with little endian
    if str(image.dtype)[0] == '>':
        error = "Array cannot be big endian. Convert arrays with ndarray.astype\n"
        error += "Given array dtype is " + str(image.dtype)
        raise TypeError(error)

    image = sitk.GetImageFromArray(image, isVector=vector)
    if spacing is None: spacing = np.ones(image.GetDimension())
    image.SetSpacing(spacing[::-1])
    if origin is None: origin = np.zeros(image.GetDimension())
    image.SetOrigin(origin[::-1])
    return image


def invert_matrix_axes(matrix):
    """
    Permute matrix entries to reflect an xyz to zyx (or vice versa) axis reordering

    Parameters
    ----------
    matrix : 4x4 array
        The matrix to permute

    Returns
    -------
    permuted_matrix : 4x4 array
        The same matrix but with the axis order inverted
    """

    corrected = np.eye(4)
    corrected[:3, :3] = matrix[:3, :3][::-1, ::-1]
    corrected[:3, -1] = matrix[:3, -1][::-1]
    return corrected


def change_affine_matrix_origin(matrix, origin):
    """
    Affine matrix change of origin

    Parameters
    ----------
    matrix : 4x4 array
        The matrix to rebase

    origin : 1d-array
        The new origin

    Returns
    -------
    new_matrix : 4x4 array
        The same affine transform but encoded with respect to the given origin
    """

    tl, tr = np.eye(4), np.eye(4)
    origin = np.array(origin)
    tl[:3, -1], tr[:3, -1] = -origin, origin
    return np.matmul(tl, np.matmul(matrix, tr))


def affine_transform_to_matrix(transform):
    """
    Convert sitk affine transform object to a 4x4 numpy array

    Parameters
    ----------
    transform : sitk.AffineTransform
        The affine transform

    Returns
    -------
    matrix : 4x4 numpy array
        The same transform as a 4x4 matrix
    """

    matrix = np.eye(4)
    matrix[:3, :3] = np.array(transform.GetMatrix()).reshape((3,3))
    matrix[:3, -1] = np.array(transform.GetTranslation())
    return invert_matrix_axes(matrix)


def matrix_to_affine_transform(matrix):
    """
    Convert 4x4 numpy array to sitk.AffineTransform object

    Parameters
    ----------
    matrix : 4x4 array
        The affine transform as a numpy array

    Returns
    -------
    affine_transform : sitk.AffineTransform
        The same affine but as a sitk.AffineTransform object
    """

    matrix_sitk = invert_matrix_axes(matrix)
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(matrix_sitk[:3, :3].flatten())
    transform.SetTranslation(matrix_sitk[:3, -1].squeeze())
    return transform


def matrix_to_euler_transform(matrix):
    """
    Convert 4x4 numpy array to sitk.Euler3DTransform (rigid transform)

    Parameters
    ----------
    matrix : 4x4 array
        The rigid transform as a numpy array

    Returns
    -------
    rigid_transform : sitk.Euler3DTransform object
        The same rigid transform but as a sitk object
    """

    matrix_sitk = invert_matrix_axes(matrix)
    transform = sitk.Euler3DTransform()
    transform.SetMatrix(matrix_sitk[:3, :3].flatten())
    transform.SetTranslation(matrix_sitk[:3, -1].squeeze())
    return transform


def euler_transform_to_parameters(transform):
    """
    Convert a sitk.Euler3DTransform object to a list of rigid transform
    parameters

    Parameters
    ----------
    transform : sitk.Euler3DTransform
        The rigid transform object

    Returns
    -------
    rigid_parameters : 1d-array, length 6
        The rigid transform parameters: (rotX, rotY, rotZ, transX, transY, transZ)
    """

    return np.array((transform.GetAngleX(),
                     transform.GetAngleY(),
                     transform.GetAngleZ()) +
                     transform.GetTranslation()
    )


def parameters_to_euler_transform(params):
    """
    Convert rigid transform parameters to a sitk.Euler3DTransform object

    Parameters
    ----------
    rigid_parameters : 1d-array, length 6
        The rigid transform parameters: (rotX, rotY, rotZ, transX, transY, transZ)

    Returns
    -------
    transform : sitk.Euler3DTransform
        A sitk rigid transform object
    """

    transform = sitk.Euler3DTransform()
    transform.SetRotation(*params[:3])
    transform.SetTranslation(params[3:])
    return transform


def physical_parameters_to_affine_matrix(params, center):
    """
    Convert separate affine transform parameters to an affine matrix

    Parameters
    ----------
    params : 1d-array
        The affine transform parameters
        (transX, transY, transZ, rotX, rotY, rotZ, scX, scY, scZ, shX, shY, shZ)
        trans : translation
        rot : rotation
        sc : scale
        sh : shear

    center : 1d-array
        The center of rotation as a coordinate

    Returns
    -------
    matrix : 4x4 array
        The affine matrix representing those physical transform parameters
    """

    # translation
    aff = np.eye(4)
    aff[:3, -1] = params[:3]
    # rotation
    x = np.eye(4)
    x[:3, :3] = Rotation.from_rotvec(params[3:6]).as_matrix()
    x = change_affine_matrix_origin(x, -center)
    aff = np.matmul(x, aff)
    # scale
    x = np.diag(tuple(params[6:9]) + (1,))
    aff = np.matmul(x, aff)
    # shear
    shx, shy, shz = np.eye(4), np.eye(4), np.eye(4)
    shx[1, 0], shx[2, 0] = params[10], params[11]
    shy[0, 1], shy[2, 1] = params[9], params[11]
    shz[0, 2], shz[1, 2] = params[9], params[10]
    x = np.matmul(shz, np.matmul(shy, shx))
    return np.matmul(x, aff)


def matrix_to_displacement_field(matrix, shape, spacing=None):
    """
    Convert an affine matrix into a displacement vector field

    Parameters
    ----------
    matrix : 4x4 array
        The affine matrix

    shape : tuple
        The voxel grid shape for the displacement vector field

    spacing : tuple (default: (1, 1, 1, ...))
        The voxel sampling rate (spacing) for the displacement vector field

    Returns
    -------
    displacement_vector_field : nd-array
        Field of shape + (3,) shape and given spacing
    """

    if spacing is None: spacing = np.ones(len(shape))
    nrows, ncols, nstacks = shape
    grid = np.array(np.mgrid[:nrows, :ncols, :nstacks])
    grid = grid.transpose(1,2,3,0) * spacing
    mm, tt = matrix[:3, :3], matrix[:3, -1]
    return np.einsum('...ij,...j->...i', mm, grid) + tt - grid


def field_to_displacement_field_transform(field, spacing=None, origin=None):
    """
    Convert a displacement vector field numpy array to a sitk displacement field
    transform object

    Parameters
    ----------
    field : nd-array
        The displacement vector field

    spacing : tuple (default: (1, 1, 1, ...))
        The voxel spacing (sampling rate) of the field

    origin : tuple (default: (0, 0, 0, ...))
        The origin (in physical units) of the field

    Returns
    -------
    sitk_displacement_field : sitk.DisplacementFieldTransform
        A sitk displacement field transform object
    """

    field = field.astype(np.float64)[..., ::-1]
    transform = numpy_to_sitk(field, spacing, origin, vector=True)
    return sitk.DisplacementFieldTransform(transform)


def bspline_parameters_to_transform(parameters):
    """
    Convert 1d-array of b-spline parameters to sitk.BSplineTransform

    Parameters
    ----------
    parameters : 1d-array
        The control point and other parameters that fully specify a b-spline
        transform

    Returns
    -------
    trasnform_object : sitk.BSplineTransform
        A sitk.BSplineTransform object
    """

    t = sitk.BSplineTransform(3, 3)
    t.SetFixedParameters(parameters[:18])
    t.SetParameters(parameters[18:])
    return t


def bspline_to_displacement_field(
    bspline, shape, spacing=None, origin=None, direction=None,
):
    """
    Convert a sitk.BSplineTransform object to a displacement vector field

    Parameters
    ----------
    bspline : sitk.BSplineTransform
        A sitk.BSplineTransform object

    shape : tuple
        The shape of the resulting displacement field

    spacing : tuple (default: (1, 1, 1, ...))
        The desired spacing for the displacement field

    origin : tuple (default: (0, 0, 0, ...))
        The origin of the displacement field

    direction : 4x4 matrix (default identity)
        The directions cosine matrix for the field

    Returns
    -------
    displacement_field : nd-array
        The displacement vector field given by the b-spline transform
    """

    if spacing is None: spacing = np.ones(len(shape))
    if origin is None: origin = np.zeros(len(shape))
    if direction is None: direction = np.eye(len(shape))
    df = sitk.TransformToDisplacementField(
        bspline, sitk.sitkVectorFloat64,
        shape[::-1], origin[::-1], spacing[::-1],
        direction[::-1, ::-1].ravel(),
    )
    return sitk.GetArrayFromImage(df).astype(np.float32)[..., ::-1]


# TODO: function that takes a numpy array and return transform type


def relative_spacing(query, reference, reference_spacing):
    """
    Determine a voxel spacing from two images and one voxel spacing

    Parameters
    ----------
    query : nd-array
        The voxel grid whose spacing you'd like to know

    reference : nd-array
        A different voxel grid over the same domain whose spacing you do know

    reference_spacing : tuple
        The known voxel spacing

    Returns
    -------
    query_spacing : 1d-array
        The spacing of the query voxel grid
    """

    ndim = len(reference_spacing)
    ratio = np.array(reference.shape[:ndim]) / query.shape[:ndim]
    return reference_spacing * ratio



def transform_list_to_composite_transform(transform_list, spacing=None, origin=None):
    """
    Convert a list of transforms to a sitk.CompositeTransform object

    Parameters
    ----------
    transform_list : list
        A list of transforms, either 4x4 numpy arrays (affine transforms) or
        nd-arrays (displacement vector fields)

    spacing : 1d array or tuple of 1d arrays (default: None)
        The spacing in physical units (e.g. mm or um) between voxels of any
        deformations in transform_list. If a tuple, must be the same length
        as transform_list. Entries for affine matrices are ignored.

    origin : 1d array or tuple of 1d arrays (default: None)
        The origin in physical units (e.g. mm or um) of any deformations
        in transform_list. If a tuple, must be the same length as transform_list.
        Entries for affine matrices are ignored.

    Returns
    -------
    composite_transform : sitk.CompositeTransform object
        All transforms in the given list compressed into a sitk.CompositTransform 
    """

    transform = sitk.CompositeTransform(3)
    for iii, t in enumerate(transform_list):
        if t.shape == (4, 4):
            t = matrix_to_affine_transform(t)
        elif len(t.shape) == 1:
            t = bspline_parameters_to_transform(t)
        else:
            a = spacing[iii] if isinstance(spacing, tuple) else spacing
            b = origin[iii] if isinstance(origin, tuple) else origin
            t = field_to_displacement_field_transform(t, a, b)
        transform.AddTransform(t)
    return transform


def create_zarr(path, shape, chunks, dtype, chunk_locked=False, client=None):
    """
    Create a new zarr array on disk

    Parameters
    ----------
    path : string
        The location of the new zarr array

    shape : tuple
        The shape of the new zarr array

    chunks : tuple
        The shape of individual chunks in the zarr array

    dtype : a numpy.dtype object
        The data type of the new zarr array data

    chunk_locked : bool (default: False)
        DEPRECATED

    client : dask.client (default: None)
        DEPRECATED

    Returns
    -------
    zarr_array : zarr array
        Reference to the newly created zarr array on disk
    """

    compressor = Blosc(
        cname='zstd', clevel=4, shuffle=Blosc.BITSHUFFLE,
    )
    zarr_disk = zarr.open(
        path, 'w',
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressor=compressor,
    )

    # this code is currently never used within bigstream
    # keeping it aroung in case a use case comes up
    if chunk_locked:
        indexer = BasicIndexer(slice(None), zarr_disk)
        keys = (zarr_disk._chunk_key(idx.chunk_coords) for idx in indexer)
        lock = {key: Lock(key, client=client) for key in keys}
        lock['.zarray'] = Lock('.zarray', client=client)
        zarr_disk = zarr.open(
            store=zarr_disk.store, path=zarr_disk.path,
            synchronizer=lock, mode='r+',
        )

    return zarr_disk
    

def numpy_to_zarr(array, chunks, path):
    """
    Convert a numpy array to a zarr array on disk

    Parameters
    ----------
    array : nd-array
        The numpy array to convert to zarr format

    chunks : tuple
        The shape of individual chunks in the zarr array

    path : string
        On disk location to create the zarr array

    Returns
    -------
    zarr_array : zarr array
        Reference to the zarr array copy of the given numpy array
    """

    if not isinstance(array, zarr.Array):
        zarr_disk = create_zarr(path, array.shape, chunks, array.dtype)
        zarr_disk[...] = array
        return zarr_disk
    else:
        return array

